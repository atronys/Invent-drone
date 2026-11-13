"""Сервер оператора: трансляция с пометками, состояние, управление заданием, отчёты.

Запускается на Raspberry Pi; оператор открывает http://<адрес-платы>:8080 с ПК в той же сети.
Один фоновый поток читает кадры и ведёт автомат сканирования; веб-часть только отдаёт
последний кадр и снимок состояния, поэтому медленный браузер не тормозит обработку.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, abort, jsonify, request, send_from_directory

from . import __version__, overlay
from .config import scan_config
from .evidence import EvidenceStore
from .layout import Layout
from .scanner import Scanner
from .sensors import SensorHub
from .sources import BaseSource, SyntheticSource, make_source
from .wms import reconcile
from .wms.demo_data import build_demo
from .wms.emulator import EmulatorBackend
from .wms.http_api import make_blueprint
from .wms.onec import OneCHttpBackend

WEB_DIR = Path(__file__).parent / "web"


def load_references(folder: str | Path) -> dict:
    """Эталоны пустых ячеек: файлы <адрес>.png, снятые командой calibrate-empty."""
    refs = {}
    p = Path(folder)
    if p.is_dir():
        for f in p.glob("*.png"):
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                refs[f.stem] = img
    return refs


class Runtime:
    """Всё, что живёт между запросами: источник, автомат, WMS, датчики, последний кадр."""

    def __init__(self, cfg: dict, source: BaseSource | None = None):
        self.cfg = cfg
        self.layout = Layout.load(cfg["layout"]) if cfg.get("layout") else Layout.demo()
        scene = None
        wcfg = cfg["wms"]
        self.emulator: EmulatorBackend | None = None
        synthetic = cfg["camera"].get("source") == "synthetic" and source is None
        if wcfg.get("backend", "emulator") == "emulator":
            if synthetic:
                # сцена и учётные остатки создаются вместе — иначе не с чем сверять
                self.emulator, scene, self.expected = build_demo(self.layout, seed=cfg.get("seed", 7),
                                                                 db_path=wcfg.get("db") or ":memory:")
            else:
                self.emulator = EmulatorBackend(wcfg.get("db") or ":memory:")
                if not self.emulator.cells():
                    if wcfg.get("demo", True):
                        self.emulator, _, self.expected = build_demo(self.layout, seed=cfg.get("seed", 7),
                                                                     db_path=wcfg.get("db") or ":memory:")
                    else:
                        for a in self.layout.addresses():
                            self.emulator.add_cell(str(a), a.rack, a.level, a.pos)
            self.wms = self.emulator
        else:
            self.wms = OneCHttpBackend.from_config(wcfg)
            if synthetic:
                _, scene, self.expected = build_demo(self.layout, seed=cfg.get("seed", 7))
        self.source = source or make_source(cfg, scene)
        self.evidence = EvidenceStore(cfg.get("evidence_dir", "data/evidence"))
        self.sensors = SensorHub(cfg["sensors"]["port"], cfg["sensors"].get("baud", 115200)) \
            if cfg["sensors"].get("enabled") else None
        self.scanner = Scanner(self.layout, self.source.preview_cam, self.source.still_cam, self.wms, self.evidence,
                               scan_config(cfg), self.source.capture_still,
                               load_references(cfg.get("references_dir", "data/references")))
        self.snap: dict = self.scanner.snapshot()
        self.jpeg: bytes = b""
        self.fps = 0.0
        self.frame_no = 0
        self.error = ""
        self._lock = threading.Lock()          # автомат не потокобезопасен: команды и шаг — по очереди
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- цикл обработки -------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="rackscan-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.source.close()
        if self.sensors:
            self.sensors.close()

    def step_once(self) -> bool:
        frame = self.source.read()
        if frame is None:
            return False
        if self.sensors is not None:
            r = self.sensors.latest()
            if r is not None:
                frame.gyro_dps, frame.tof_mm, frame.tilt_deg = r.gyro_norm, r.tof_mm, (r.roll_deg, r.pitch_deg)
        with self._lock:
            snap = self.scanner.step(frame)
            ov = self.scanner.overlay
        self.source.after_step(snap)
        if self.sensors is not None:
            self.sensors.set_status(snap["status"])
        img = overlay.draw_live(frame.image, snap, ov, self.layout)
        sw = int(self.cfg["server"].get("stream_width", 1280))
        if img.shape[1] > sw:
            img = cv2.resize(img, (sw, int(img.shape[0] * sw / img.shape[1])), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg["server"].get("jpeg_quality", 80))])
        with self._cond:
            self.snap = snap
            if ok:
                self.jpeg = buf.tobytes()
            self.frame_no += 1
            self._cond.notify_all()
        return True

    def _loop(self) -> None:
        t_prev = time.monotonic()
        while not self._stop.is_set():
            try:
                if not self.step_once():
                    time.sleep(0.05)
                    continue
                self.error = ""
            except Exception as e:   # сбой одного кадра не должен ронять трансляцию
                self.error = f"{type(e).__name__}: {e}"
                self.evidence.journal("error", error=self.error)
                time.sleep(0.2)
            now = time.monotonic()
            self.fps = 0.9 * self.fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now

    def wait_frame(self, last: int, timeout: float = 2.0) -> tuple[int, bytes]:
        with self._cond:
            self._cond.wait_for(lambda: self.frame_no != last, timeout)
            return self.frame_no, self.jpeg

    # --- команды оператора ----------------------------------------------------------
    def command(self, name: str, **kw) -> dict:
        with self._lock:
            sc = self.scanner
            if name == "start":
                return sc.start_task(kw.get("addresses"), kw.get("mode", "task"))
            if name == "stop":
                sc.stop()
            elif name == "skip":
                sc.skip()
            elif name == "rescan":
                sc.rescan(kw["address"])
            else:
                raise KeyError(name)
        return {"ok": True}

    def state(self) -> dict:
        s = dict(self.snap)
        s.update({
            "fps": round(self.fps, 1), "error": self.error, "version": __version__,
            "source": type(self.source).__name__,
            "sim": isinstance(self.source, SyntheticSource),
            "autopilot": bool(getattr(self.source, "autopilot", False)),
            "sensors_ok": None if self.sensors is None else self.sensors.available,
            "evidence_session": self.evidence.session,
        })
        return s

    def layout_view(self) -> dict:
        """Геометрия стеллажей для мини-карты на экране оператора."""
        racks = []
        for rid, r in sorted(self.layout.racks.items()):
            racks.append({"id": rid, "levels": r.levels, "cells": r.cells_per_level,
                          "addresses": [[f"{rid:02d}-{lv:02d}-{p:02d}" for p in range(1, r.cells_per_level + 1)]
                                        for lv in range(r.levels, 0, -1)]})
        return {"name": self.layout.name, "racks": racks}


def create_app(rt: Runtime) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["JSON_AS_ASCII"] = False
    if rt.emulator is not None:
        app.register_blueprint(make_blueprint(rt.emulator, os.environ.get("RACKSCAN_WMS_USER"),
                                              os.environ.get("RACKSCAN_WMS_PASSWORD")),
                               url_prefix="/wms/hs/rackscan/v1")

    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/stream.mjpg")
    def stream():
        def gen():
            last = -1
            while True:
                last, jpg = rt.wait_frame(last)
                if jpg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + \
                        str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/frame.jpg")
    def frame():
        return Response(rt.jpeg, mimetype="image/jpeg") if rt.jpeg else abort(503)

    @app.get("/api/state")
    def state():
        return jsonify(rt.state())

    @app.get("/api/layout")
    def layout():
        return jsonify(rt.layout_view())

    @app.post("/api/<name>")
    def command(name):
        body = request.get_json(silent=True) or {}
        if name == "sim":
            if not isinstance(rt.source, SyntheticSource):
                return jsonify({"error": "источник не синтетический"}), 409
            if "autopilot" in body:
                rt.source.autopilot = bool(body["autopilot"])
            rt.source.move(float(body.get("dx", 0)), float(body.get("dy", 0)), float(body.get("dz", 0)),
                           float(body.get("dyaw", 0)))
            return jsonify({"ok": True, "autopilot": rt.source.autopilot})
        if name not in ("start", "stop", "skip", "rescan"):
            abort(404)
        try:
            return jsonify(rt.command(name, **body))
        except (KeyError, ValueError) as e:
            return jsonify({"error": str(e)}), 400

    def _report() -> dict | None:
        tid = request.args.get("task") or rt.snap.get("task_id")
        if not tid:
            return None
        try:
            return rt.wms.report(tid)
        except Exception:
            return None

    @app.get("/api/report")
    def report_json():
        rep = _report()
        return jsonify(rep) if rep else (jsonify({"error": "задание ещё не создано"}), 404)

    @app.get("/report")
    def report_html():
        rep = _report()
        return reconcile.to_html(rep) if rep else ("Задание ещё не создано", 404)

    @app.get("/report.csv")
    def report_csv():
        rep = _report()
        if not rep:
            return "Задание ещё не создано", 404
        return Response(reconcile.to_csv(rep["lines"]), mimetype="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="inventory_{rep["task_id"]}.csv"'})

    @app.get("/api/journal")
    def journal():
        return jsonify(rt.evidence.read_journal(int(request.args.get("last", 30))))

    @app.get("/evidence/<path:name>")
    def evidence(name):
        return send_from_directory(rt.evidence.root.resolve(), name)

    return app


def serve(cfg: dict) -> None:
    rt = Runtime(cfg)
    rt.start()
    app = create_app(rt)
    host, port = cfg["server"]["host"], int(cfg["server"]["port"])
    print(f"RackScan {__version__}: откройте http://{'localhost' if host in ('0.0.0.0', '') else host}:{port}")
    try:
        app.run(host=host, port=port, threaded=True, use_reloader=False)
    finally:
        rt.stop()

"""SIM-09: виртуальный облёт — весь конвейер без Gazebo и ROS.

План облёта -> в каждой точке сканирования рендер кадров левой и правой камер (пинхол книги, гл. 4)
с текстурами этикеток из генератора -> zxing-cpp -> привязка к ячейке по (возможно, неточной) позе ->
SortieRecorder -> метрики чтения и сверка с WMS.

Рендер честно моделирует то, от чего зависит чтение:
* разрешение на фасаде (текстура сжимается усреднением по площади — как интегрирует пиксель сенсора);
* смаз от движения вдоль прохода (скорость × выдержка, barcode.motion_blur_px);
* размытие оптики (гаусс σ, px) и шум сенсора.

    python sim/generate_world.py core/examples/warehouse_demo.json -o sim/worlds/demo    # текстуры нужны
    python sim/virtual_flight.py core/examples/warehouse_demo.json sim/worlds/demo --sortie 0 --hfov 29.4 --vfov 22.2

Эксперименты: оптика (--hfov/--vfov/--width), скорость и выдержка (--v-scan, --exposure-ms, --hold),
точность локализации (--pose-noise, --yaw-noise) -> точность привязки к ячейкам.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "sim"))
import eval_read_rate as ev  # noqa: E402
import generate_world as gw  # noqa: E402
from invent_core import barcode  # noqa: E402
from invent_core import frames as fr  # noqa: E402
from invent_core.association import face_point_in_warehouse, locate_in_warehouse, project_to_image  # noqa: E402
from invent_core.camera import Camera  # noqa: E402
from invent_core.coverage import ScanParams, plan  # noqa: E402
from invent_core.layout import Warehouse  # noqa: E402
from invent_core.reconcile import Status, SortieRecorder, reconcile  # noqa: E402

def mounts(p: ScanParams) -> dict:
    """Установка камер — как в sim/make_drone_model.py (высота из ScanParams.camera_z_offset)."""
    return {"left": (0.0, 0.08, p.camera_z_offset), "right": (0.0, -0.08, p.camera_z_offset)}


@dataclass
class Quad:
    kind: str            # pallet | label | address
    address: str
    corners: np.ndarray  # 4x3, map: TL, TR, BR, BL — как видит наблюдатель из прохода
    normal_y: int = 1    # куда смотрит лицевая сторона
    texture: Path | None = None


def quad(cx: float, y: float, cz: float, w: float, h: float, n: int) -> np.ndarray:
    """Прямоугольник на плоскости y=const. Для наблюдателя из прохода «вправо» = (-n, 0, 0)."""
    return np.array([[cx + n * w / 2, y, cz + h / 2], [cx - n * w / 2, y, cz + h / 2],
                     [cx - n * w / 2, y, cz - h / 2], [cx + n * w / 2, y, cz - h / 2]])


def build_scene(wh: Warehouse, truth: dict, labels_dir: Path) -> list[Quad]:
    """Геометрия как в sim/generate_world.py: паллета утоплена на 5 см, адресная этикетка на балке."""
    out = []
    for row in wh.rows:
        n = row.normal_y
        for loc in row.locations():
            lz = row.level_z[loc.level - 1]
            az = row.address_z(loc.level)
            out.append(Quad("address", loc.address, quad(loc.x, row.face_y + n * 0.004, az, *gw.ADDRESS_LABEL, n),
                            n, labels_dir / f"addr_{loc.address}.png"))
            if not truth.get(loc.address):
                continue
            h = min(1.2, (row.level_z[loc.level] - lz - 0.25) if loc.level < len(row.level_z) else 1.2)
            out.append(Quad("pallet", loc.address, quad(loc.x, row.face_y - n * 0.05, lz + h / 2, gw.PALLET_W, h, n), n))
            out.append(Quad("label", loc.address, quad(loc.x, row.face_y - n * 0.048, loc.z, *gw.PALLET_LABEL, n),
                            n, labels_dir / f"{loc.address}.png"))
    return out


class Renderer:
    def __init__(self, cam: Camera, blur_sigma: float = 0.6, noise: float = 2.0, rng=None):
        self.cam, self.blur_sigma, self.noise = cam, blur_sigma, noise
        self.rng = rng or np.random.default_rng(0)
        self.tex_cache: dict[Path, np.ndarray] = {}

    def texture(self, path: Path) -> np.ndarray:
        if path not in self.tex_cache:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"нет текстуры {path}: сгенерируйте мир с текстурами (generate_world.py)")
            self.tex_cache[path] = img
        return self.tex_cache[path]

    def frame(self, scene: list[Quad], r: fr.Mat3, t: fr.Vec3, blur_px: float) -> np.ndarray:
        cam = self.cam
        img = np.full((cam.height, cam.width), 55, np.uint8)
        # алгоритм художника: от дальних к ближним; при равной глубине паллета раньше своей этикетки.
        # Отсечение нелицевых граней: видна только сторона, обращённая к камере.
        near = [q for q in scene if abs(q.corners[0, 0] - t[0]) < 4 and abs(q.corners[0, 1] - t[1]) < 4
                and (t[1] - q.corners[0, 1]) * q.normal_y > 0]
        near.sort(key=lambda q: (-abs(q.corners[0, 1] - t[1]), q.kind != "pallet"))
        for q in near:
            c = q.corners
            px = [project_to_image(tuple(p), cam, r, t) for p in c]
            if any(p is None for p in px):
                continue
            pts = np.array(px, np.float32)
            x0, y0 = np.floor(pts.min(0)).astype(int)
            x1, y1 = np.ceil(pts.max(0)).astype(int)
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, cam.width), min(y1, cam.height)
            if x1 <= x0 or y1 <= y0:
                continue
            if q.texture is None:
                cv2.fillConvexPoly(img, np.round(pts).astype(np.int32), 115)
                continue
            tex = self.texture(q.texture)
            edge = max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3]))
            scale = min(1.0, 2.0 * edge / tex.shape[1])
            if scale < 1.0:  # усреднение по площади ~ интегрирование светового потока пикселем
                tex = cv2.resize(tex, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            h, w = tex.shape
            src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
            hm = cv2.getPerspectiveTransform(src, pts - np.array([x0, y0], np.float32))
            size = (x1 - x0, y1 - y0)
            patch = cv2.warpPerspective(tex, hm, size, flags=cv2.INTER_LINEAR)
            mask = cv2.warpPerspective(np.full_like(tex, 255), hm, size, flags=cv2.INTER_LINEAR) > 127
            roi = img[y0:y1, x0:x1]
            roi[mask] = patch[mask]
        img = img.astype(np.float32)
        if blur_px >= 1.0:  # смаз вдоль оси u (движение вдоль прохода)
            k = int(round(blur_px))
            img = cv2.filter2D(img, -1, np.full((1, k), 1.0 / k, np.float32))
        if self.blur_sigma > 0:
            img = cv2.GaussianBlur(img, (0, 0), self.blur_sigma)
        if self.noise > 0:
            img += self.rng.normal(0, self.noise, img.shape).astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)


@dataclass
class FlightResult:
    metrics: dict
    address_attribution: float   # доля адресных этикеток, увиденных там, где они висят по карте
    address_reads: int
    address_residual_m: float    # средняя невязка «где увидели» vs «где по карте» — шум наблюдения для LOC-11
    reconcile_summary: dict
    injected_found: int
    injected_in_scope: int
    frames: int
    seconds: float


def run_flight(wh: Warehouse, p: ScanParams, truth: dict, wms: dict, labels_dir: Path, sorties=None,
               exposure_ms: float = 1.0, pose_noise: float = 0.0, yaw_noise_deg: float = 0.0,
               blur_sigma: float = 0.6, noise: float = 2.0, seed: int = 0, save_frames: Path | None = None,
               max_frames: int | None = None) -> FlightResult:
    rng = np.random.default_rng(seed)
    pl = plan(wh, p)
    chosen = pl.sorties if sorties is None else [pl.sorties[i] for i in sorties]
    scene = build_scene(wh, truth, labels_dir)
    rend = Renderer(p.camera, blur_sigma, noise, rng)
    rec = SortieRecorder()
    addr_ok = addr_total = frames = 0
    residuals = []
    t0 = time.time()
    for s in chosen:
        for wp in s.waypoints:
            if wp.kind != "scan":
                continue
            rec.expect(wp.targets)
            speed = 0.0 if wp.hold_s > 0 else p.v_scan
            for look, mount in mounts(p).items():
                if max_frames is not None and frames >= max_frames:
                    break
                body = (wp.x, wp.y, wp.z)
                r, t = fr.camera_pose(body, wp.yaw, look, t_body_cam=mount)
                dist = abs(wp.y - min((row.face_y for row in wh.rows), key=lambda fy: abs(fy - wp.y)))
                blur = barcode.motion_blur_px(p.camera, dist, speed, exposure_ms / 1000)
                img = rend.frame(scene, r, t, blur)
                frames += 1
                if save_frames is not None and frames <= 6:
                    save_frames.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_frames / f"frame_{frames:03d}_{look}.jpg"), img)
                est = (body[0] + rng.normal(0, pose_noise), body[1] + rng.normal(0, pose_noise), body[2])
                r_e, t_e = fr.camera_pose(est, wp.yaw + math.radians(rng.normal(0, yaw_noise_deg)), look, t_body_cam=mount)
                for rd in barcode.decode(img):
                    u, v = rd.center
                    addr = locate_in_warehouse(u, v, p.camera, r_e, t_e, wh) or ""
                    if rd.sscc:
                        rec.read(addr, rd.sscc)
                    elif (ref := wh.address_label_position(rd.text)) is not None:   # адресная: текст = адрес
                        addr_total += 1
                        hit = face_point_in_warehouse(u, v, p.camera, r_e, t_e, wh)
                        if hit and hit[0].id == ref[0].id:
                            dx, dz = hit[1] - ref[1], hit[2] - ref[2]
                            residuals.append(math.hypot(dx, dz))
                            addr_ok += abs(dx) < ref[0].slot_width / 2 and abs(dz) < 0.15
    obs = rec.observations()
    targeted = {o.address for o in obs}
    metrics = ev.evaluate(truth, obs, targeted)
    rep = reconcile({a: wms[a] for a in targeted}, obs)
    injected = {a for a in targeted if truth[a] != wms[a]}
    found = {r.address for r in rep.results if r.status in (Status.MISMATCH, Status.MISSING, Status.UNEXPECTED)}
    return FlightResult(metrics, addr_ok / addr_total if addr_total else 0.0, addr_total,
                        float(np.mean(residuals)) if residuals else float("nan"), rep.summary(),
                        len(injected & found), len(injected), frames, time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layout")
    ap.add_argument("world", help="каталог от generate_world.py (truth.csv, wms_expected.csv, labels/)")
    ap.add_argument("--params", help="JSON ScanParams")
    ap.add_argument("--sortie", type=int, action="append", help="номер вылета (можно несколько); по умолчанию все")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--hfov", type=float)
    ap.add_argument("--vfov", type=float)
    ap.add_argument("--v-scan", type=float)
    ap.add_argument("--hold", type=float, help="stop-and-scan: зависание на точке, с (смаза нет)")
    ap.add_argument("--min-z", type=float)
    ap.add_argument("--address-labels", action="store_true", help="пролёты так, чтобы в кадре были и адресные этикетки")
    ap.add_argument("--exposure-ms", type=float, default=1.0)
    ap.add_argument("--pose-noise", type=float, default=0.0, help="СКО ошибки положения при привязке, м")
    ap.add_argument("--yaw-noise", type=float, default=0.0, help="СКО ошибки курса при привязке, град")
    ap.add_argument("--blur", type=float, default=0.6, help="размытие оптики, σ px")
    ap.add_argument("--noise", type=float, default=2.0, help="шум сенсора, СКО уровней яркости")
    ap.add_argument("--save-frames", help="сохранить первые кадры (jpg) в каталог")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    wh = Warehouse.load(a.layout)
    p = ScanParams.load(a.params) if a.params else ScanParams()
    cam = p.camera
    if a.width or a.height or a.hfov or a.vfov:
        cam = Camera(a.width or cam.width, a.height or cam.height, a.hfov or cam.hfov_deg, a.vfov or cam.vfov_deg)
    p = replace(p, camera=cam)
    if a.v_scan is not None:
        p = replace(p, v_scan=a.v_scan)
    if a.hold is not None:
        p = replace(p, hold_s=a.hold)
    if a.min_z is not None:
        p = replace(p, min_z=a.min_z)
    if a.address_labels:
        p = replace(p, cover_address_labels=True)
    world = Path(a.world)
    from invent_core.reconcile import load_expected_csv
    truth, wms = load_expected_csv(world / "truth.csv"), load_expected_csv(world / "wms_expected.csv")
    res = run_flight(wh, p, truth, wms, world / "labels", a.sortie, a.exposure_ms, a.pose_noise, a.yaw_noise,
                     a.blur, a.noise, a.seed, Path(a.save_frames) if a.save_frames else None, a.max_frames)
    m = res.metrics
    print(f"камера {cam.width}x{cam.height} HFOV {cam.hfov_deg}°, v_scan {p.v_scan} м/с, выдержка {a.exposure_ms} мс, "
          f"hold {p.hold_s} с, ошибка позы {a.pose_noise} м / {a.yaw_noise}°")
    print(f"кадров {res.frames} за {res.seconds:.0f} с")
    print(f"паллетные этикетки: read rate {m['read_rate']:.1%} ({m['occupied_in_scope']} занятых в плане), "
          f"атрибуция {m['attribution']:.1%}, чужих {m['misattributed']}, мусор {m['garbage']}")
    print(f"адресные этикетки: прочитано {res.address_reads}, на своём месте {res.address_attribution:.1%}, "
          f"средняя невязка {res.address_residual_m * 100:.1f} см")
    print(f"сверка: {json.dumps(res.reconcile_summary, ensure_ascii=False)}; "
          f"внесённых ошибок WMS найдено {res.injected_found} из {res.injected_in_scope}")


if __name__ == "__main__":
    main()

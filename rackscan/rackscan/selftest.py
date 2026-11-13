"""Самопроверка без железа: виртуальная камера обходит учебный склад, выполняя команды стенда.

Имитатор носителя на каждом шаге смещает камеру на долю показанной ошибки (плюс шум руки).
Проверяется всё сразу: поза, наведение, занятость, чтение, фото, журнал, сверка с WMS.
Запуск:  python -m rackscan selftest
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .evidence import EvidenceStore
from .layout import Layout
from .optics import camera_from_preset
from .scanner import APPROACH, FINISHED, NAVIGATE, SEARCH, Frame, ScanConfig, Scanner, follow_commands
from .wms.demo_data import build_demo


def run(seed: int = 7, preview_size: tuple[int, int] = (1536, 864), still_size: tuple[int, int] | None = None,
        max_steps: int = 1500, out_dir: str | Path | None = None, verbose: bool = False, noise_m: float = 0.004,
        layout: Layout | None = None, addresses: list[str] | None = None, cfg: ScanConfig | None = None,
        on_step=None) -> dict:
    """Возвращает {'ok', 'steps', 'results', 'expected', 'mismatches', 'summary', 'wall_s'}."""
    layout = layout or Layout.demo()
    wms, scene, expected = build_demo(layout, seed=seed)
    preview = camera_from_preset("imx708", preview_size)
    still = camera_from_preset("imx708", still_size) if still_size else camera_from_preset("imx708")
    state = {"p": np.array([0.25, 0.55, 1.5]), "yaw": 14.0}
    evidence = EvidenceStore(out_dir) if out_dir else None
    scanner = Scanner(layout, preview, still, wms, evidence, cfg or ScanConfig(done_hold_s=0.2),
                      capture_still=lambda: scene.render(still, state["p"], state["yaw"]))
    scanner.start_task(addresses)
    rng = np.random.default_rng(seed)
    t, n, last, t0 = 0.0, 0, None, time.time()
    while scanner.phase != FINISHED and n < max_steps:
        img = scene.render(preview, state["p"], state["yaw"])
        snap = scanner.step(Frame(img, t, gyro_dps=1.0, tof_mm=scene.tof_mm(state["p"], state["yaw"])))
        if on_step is not None:
            on_step(n, img, snap, scanner)
        key = (snap["phase"], snap["target"])
        if verbose and key != last:
            print(f"{n:4d} {snap['phase']:9s} {snap['target'] or '':9s} {snap['status']:6s} "
                  f"{snap['headline']} | {snap['hint']}")
            last = key
        if snap["phase"] in (NAVIGATE, APPROACH, SEARCH) and snap["errors"]:
            state["p"], state["yaw"] = follow_commands(state["p"], state["yaw"], snap)
            state["p"] = state["p"] + rng.normal(0, noise_m, 3)
            state["yaw"] += float(rng.normal(0, 0.3))
        t += 0.2
        n += 1
    wanted = expected if addresses is None else {a: expected[a] for a in addresses}
    mismatches = []
    for a, e in sorted(wanted.items()):
        r = scanner.results.get(a, {})
        if r.get("scan_status") != e["scan"] or r.get("verdict") != e["verdict"] or \
                not set(e["flags"]) <= set(r.get("flags", [])):
            mismatches.append({"address": a, "expected": e, "got": {
                "scan": r.get("scan_status"), "verdict": r.get("verdict"), "flags": r.get("flags"),
                "codes": r.get("codes"), "occupancy": r.get("occupancy")}})
    report = wms.report(scanner.task.id)
    return {"ok": not mismatches and scanner.phase == FINISHED, "steps": n, "results": scanner.results,
            "expected": wanted, "mismatches": mismatches, "summary": report["summary"], "report": report,
            "wall_s": round(time.time() - t0, 1), "truth": scene.truth()}


def main(verbose: bool = True, out_dir: str | None = None) -> int:
    res = run(verbose=verbose, out_dir=out_dir)
    s = res["summary"]
    print(f"\nшагов: {res['steps']}, время: {res['wall_s']} с, ячеек: {s['total']}, отсканировано: {s['scanned']}")
    print("итоги сверки:", {k: v for k, v in s["counts"].items() if v}, "IRA:", s["ira"])
    for m in res["mismatches"]:
        print("НЕ СОВПАЛО:", m)
    print("САМОПРОВЕРКА ПРОЙДЕНА" if res["ok"] else "САМОПРОВЕРКА НЕ ПРОЙДЕНА")
    return 0 if res["ok"] else 1

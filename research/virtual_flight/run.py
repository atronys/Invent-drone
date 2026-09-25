"""Серия экспериментов SIM-09 (виртуальный облёт): оптика, смаз, stop-and-scan, ошибка позы -> чтение и привязка.

    python research/virtual_flight/run.py [--workers 6]

Склад: 1 проход × 2 ряда × 10 секций × 4 яруса × 3 места (240 ячеек), 80% занятость, 3% ошибок WMS.
Этикетки SSCC A6 с модулем 0.5 мм (типично) и 0.8 мм (крупный). Результат — README.md рядом.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "sim"))

LANDSCAPE_50 = (4056, 3040, 50.0, 38.5)      # широкий объектив (по умолчанию в core)
LANDSCAPE_12MM = (4056, 3040, 29.4, 22.2)    # IMX477 + 12 мм, альбомная установка
PORTRAIT_12MM = (3040, 4056, 22.2, 29.4)     # то же, портретная установка

CONFIGS = [
    # имя, модуль мм, камера, доп. параметры ScanParams, параметры run_flight
    ("A: 12 Мп, 50°, альбом", 0.5, LANDSCAPE_50, {}, {}),
    ("B: 12 Мп, 12 мм, альбом", 0.5, LANDSCAPE_12MM, {}, {}),
    ("C: 12 Мп, 12 мм, портрет + адресные", 0.5, PORTRAIT_12MM, {"cover_address_labels": True}, {}),
    ("D: 12 Мп, 50°, модуль 0.8 мм", 0.8, LANDSCAPE_50, {}, {}),
    ("E: как C, выдержка 3 мс (смаз)", 0.5, PORTRAIT_12MM, {"cover_address_labels": True}, {"exposure_ms": 3.0}),
    ("F: как C, stop-and-scan 1.5 с", 0.5, PORTRAIT_12MM, {"cover_address_labels": True, "hold_s": 1.5}, {}),
    ("G: как C, ошибка позы 5 см", 0.5, PORTRAIT_12MM, {"cover_address_labels": True}, {"pose_noise": 0.05}),
    ("H: как C, ошибка позы 15 см", 0.5, PORTRAIT_12MM, {"cover_address_labels": True}, {"pose_noise": 0.15}),
    ("I: как C, ошибка позы 30 см + курс 1°", 0.5, PORTRAIT_12MM, {"cover_address_labels": True},
     {"pose_noise": 0.30, "yaw_noise_deg": 1.0}),
]


def world(module_mm: float, base: Path):
    import generate_world as gw
    from invent_core.layout import make_grid_warehouse
    out = base / f"m{module_mm}"
    (out / "labels").mkdir(parents=True, exist_ok=True)
    wh = make_grid_warehouse(n_aisles=1, bays=10, levels=4)
    _, truth, wms, _, ok = gw.build(wh, gw.Options(module_mm=module_mm, seed=11), out)
    assert ok, "нужны python-barcode и pillow"
    return wh, truth, wms, out / "labels"


def one(cfg, base: str):
    import virtual_flight as vf
    from invent_core.camera import Camera
    from invent_core.coverage import ScanParams
    name, module, cam, sp, rf = cfg
    wh, truth, wms, labels = world(module, Path(base))
    p = replace(ScanParams(camera=Camera(*cam), min_z=0.5), **sp)
    r = vf.run_flight(wh, p, truth, wms, labels, **rf)
    m = r.metrics
    return {"name": name, "frames": r.frames, "seconds": round(r.seconds), "read_rate": m["read_rate"],
            "attribution": m["attribution"], "misattributed": m["misattributed"], "garbage": m["garbage"],
            "occupied": m["occupied_in_scope"], "addr_reads": r.address_reads, "addr_ok": r.address_attribution,
            "addr_residual_cm": r.address_residual_m * 100, "found": r.injected_found, "injected": r.injected_in_scope}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    with tempfile.TemporaryDirectory() as base:
        for module in sorted({c[1] for c in CONFIGS}):   # текстуры один раз, до параллельного прогона
            world(module, Path(base))
        with ProcessPoolExecutor(a.workers) as ex:
            rows = list(ex.map(one, CONFIGS, [base] * len(CONFIGS)))
    lines = ["| Конфигурация | Кадров | Read rate паллет | Привязка паллет | Адресных прочитано | Адресные на своём месте | Невязка адресных, см | Ошибок WMS найдено |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['frames']} | {r['read_rate']:.1%} | {r['attribution']:.1%} | {r['addr_reads']} | "
                     f"{r['addr_ok']:.1%} | {r['addr_residual_cm']:.1f} | {r['found']} из {r['injected']} |")
    table = "\n".join(lines)
    print(table)
    (Path(__file__).with_name("results.json")).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (Path(__file__).with_name("results_table.md")).write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""CLI ядра: python -m invent_core <команда> --help

  make-layout  сгенерировать типовой склад (JSON)
  plan         план облёта: пролёты, вылеты, время, ячеек/час
  optics       проверка оптики: читается ли этикетка на дистанции и скорости
  reconcile    сверка наблюдений с выгрузкой WMS/1С -> CSV-отчёт
  label-pose   дистанция до этикетки по фото (PnP), упражнение LEARN-07
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace

from . import barcode
from .camera import Camera
from .coverage import ScanParams, plan
from .layout import Warehouse, make_grid_warehouse
from .reconcile import load_expected_csv, load_observations_csv, reconcile, write_report_csv


def _camera(a) -> Camera:
    return Camera(a.width, a.height, a.hfov, a.vfov)


def cmd_make_layout(a) -> None:
    wh = make_grid_warehouse(n_aisles=a.aisles, bays=a.bays, levels=a.levels, aisle_width=a.aisle_width, name=a.name)
    wh.save(a.out)
    print(f"{a.out}: рядов {len(wh.rows)}, проходов {len(wh.aisles)}, ячеек {len(wh.locations())}")


def _read_addresses(path: str) -> set[str]:
    """Список адресов: CSV со столбцом address/адрес или по одному адресу в строке."""
    lines = [ln.strip() for ln in open(path, encoding="utf-8-sig") if ln.strip()]
    head = lines[0].lower().replace(",", ";").split(";")
    col = next((i for i, h in enumerate(head) if h in ("address", "адрес")), None)
    if col is None:
        return set(lines)
    return {ln.replace(",", ";").split(";")[col] for ln in lines[1:]}


def cmd_plan(a) -> None:
    wh = Warehouse.load(a.layout)
    if a.params:
        p = ScanParams.load(a.params)
    else:
        p = replace(ScanParams(camera=_camera(a)), battery_s=a.battery, hold_s=a.hold, v_scan=a.v_scan)
    only = _read_addresses(a.only) if a.only else None
    if only is not None and not a.params:
        p = replace(p, hold_s=a.hold or 1.5)  # перескан — режим stop-and-scan
    pl = plan(wh, p, only)
    n_wp = sum(len(s.waypoints) for s in pl.sorties)
    scope = f"перескан {len(only)} адресов" if only is not None else f"ячеек {len(wh.locations())}"
    print(f"Склад «{wh.name}»: {scope}, покрыто {len(pl.covered)}")
    print(f"Вылетов: {len(pl.sorties)}, точек маршрута: {n_wp}")
    for i, s in enumerate(pl.sorties):
        print(f"  вылет {i}: {s.duration_s / 60:5.1f} мин, ячеек {len(s.targets)}")
    print(f"Полётное время: {pl.flight_s / 60:.1f} мин, с заменой АКБ: {pl.wall_clock_s / 60:.1f} мин")
    print(f"Производительность: {pl.locations_per_hour:.0f} ячеек/ч")
    for w in pl.warnings:
        print(f"ВНИМАНИЕ: {w}")
    if a.export:
        data = [[{k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(w).items()}
                 for w in s.waypoints] for s in pl.sorties]
        with open(a.export, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print(f"Маршрут сохранён: {a.export}")


def cmd_optics(a) -> None:
    cam = _camera(a)
    fw, fh = cam.footprint(a.distance)
    print(f"Камера {cam.width}x{cam.height}, HFOV {cam.hfov_deg}°, дистанция {a.distance} м")
    print(f"  кадр на фасаде: {fw:.2f} x {fh:.2f} м, {cam.px_per_mm(a.distance):.2f} px/мм")
    print(f"  модуль, мм | px/модуль | читается | макс. дистанция, м | макс. выдержка при {a.speed} м/с, мс")
    for m in (0.33, 0.5, 0.75, 1.0):
        ppm = barcode.px_per_module(cam, a.distance, m)
        t_max_ms = 0.5 * m / (a.speed * 1000) * 1000 if a.speed > 0 else float("inf")  # смаз <= 0.5 модуля
        print(f"  {m:10.2f} | {ppm:9.2f} | {'да ' if ppm >= barcode.MIN_PPM else 'НЕТ':8} | "
              f"{barcode.max_read_distance(cam, m):18.2f} | {t_max_ms:.2f}")
    spread = 0.56  # адресная этикетка на балке (-0.06) и паллетная (+0.5) — как в RackRow по умолчанию
    need = barcode.required_vertical_px(spread, 0.5)
    print(f"  чтобы в кадре были адресная и паллетная этикетки (разнос {spread} м) при модуле 0.5 мм, нужно "
          f">= {need:.0f} px по вертикали; у камеры {cam.height} px -> "
          f"{'хватает' if cam.height >= need else 'НЕ хватает: портретная установка или больше Мп'}")
    blur = barcode.motion_blur_px(cam, a.distance, a.speed, a.exposure_ms / 1000)
    print(f"  смаз при выдержке {a.exposure_ms} мс: {blur:.2f} px ({a.speed * a.exposure_ms:.2f} мм)")


def cmd_label_pose(a) -> None:
    """LEARN-07: дистанция до этикетки по фото (PnP) — сверить с рулеткой."""
    import math

    import cv2

    from .pnp import detect_label_quad, label_pose
    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit(f"не удалось открыть {a.image}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if a.camera:
        cam = Camera.load(a.camera)
        if (cam.width, cam.height) != (w, h):
            print(f"ВНИМАНИЕ: калибровка {cam.width}x{cam.height}, а снимок {w}x{h} — снимайте в том же режиме")
    else:
        vfov = math.degrees(2 * math.atan(math.tan(math.radians(a.hfov) / 2) * h / w))
        cam = Camera(w, h, a.hfov, vfov)
        print(f"калибровки нет: HFOV {a.hfov}° — грубая оценка; для LEARN-07 откалибруйте камеру (tools/calibrate_camera.py)")
    quad = detect_label_quad(gray)
    if quad is None:
        raise SystemExit("этикетка не найдена: нужен светлый прямоугольник на более тёмном фоне целиком в кадре")
    pose = label_pose(quad, cam, (a.size[0], a.size[1]))
    x, y, z = pose.t
    print(f"дистанция до центра этикетки: {pose.distance:.3f} м (x={x:+.3f}, y={y:+.3f}, z={z:.3f} м)")
    print(f"ошибка репроекции: {pose.reproj_rmse_px:.2f} px; углы: {quad.round(1).tolist()}")
    if a.out:
        cv2.polylines(img, [quad.astype("int32").reshape(-1, 1, 2)], True, (0, 0, 255), max(2, w // 400))
        cv2.putText(img, f"{pose.distance:.3f} m", tuple(int(v) for v in quad[0]), cv2.FONT_HERSHEY_SIMPLEX,
                    max(1, w // 800), (0, 0, 255), max(2, w // 400))
        cv2.imwrite(a.out, img)
        print(f"разметка: {a.out}")


def cmd_reconcile(a) -> None:
    report = reconcile(load_expected_csv(a.expected), load_observations_csv(a.observations))
    write_report_csv(report, a.out)
    print(json.dumps(report.summary(), ensure_ascii=False))
    print(f"Точность учёта (IRA): {report.accuracy:.1%}; отчёт: {a.out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="invent_core", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cam_args(p):
        d = Camera()
        p.add_argument("--width", type=int, default=d.width)
        p.add_argument("--height", type=int, default=d.height)
        p.add_argument("--hfov", type=float, default=d.hfov_deg)
        p.add_argument("--vfov", type=float, default=d.vfov_deg)

    p = sub.add_parser("make-layout")
    p.add_argument("out")
    p.add_argument("--aisles", type=int, default=2)
    p.add_argument("--bays", type=int, default=10)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--aisle-width", type=float, default=3.0)
    p.add_argument("--name", default="demo")
    p.set_defaults(func=cmd_make_layout)

    p = sub.add_parser("plan")
    p.add_argument("layout")
    p.add_argument("--battery", type=float, default=600.0, help="полётное время на АКБ, с")
    p.add_argument("--hold", type=float, default=0.0, help="зависание на точке, с (0 — непрерывная съёмка)")
    p.add_argument("--v-scan", type=float, default=0.4)
    p.add_argument("--export", help="сохранить маршрут (JSON) для узла миссии")
    p.add_argument("--params", help="JSON с ScanParams (FLT-02); перекрывает флаги камеры/АКБ/скорости")
    p.add_argument("--only", help="перескан: файл со списком адресов (MIS-03)")
    cam_args(p)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("optics")
    p.add_argument("--distance", type=float, default=1.5, help="до фасада, м")
    p.add_argument("--speed", type=float, default=0.4, help="м/с")
    p.add_argument("--exposure-ms", type=float, default=1.0)
    cam_args(p)
    p.set_defaults(func=cmd_optics)

    p = sub.add_parser("label-pose", help="дистанция до этикетки по фото (LEARN-07)")
    p.add_argument("image")
    p.add_argument("--camera", help="YAML калибровки (tools/calibrate_camera.py)")
    p.add_argument("--hfov", type=float, default=70.0, help="если нет калибровки: угол обзора, град")
    p.add_argument("--size", type=float, nargs=2, default=(0.148, 0.105), help="ширина и высота этикетки, м")
    p.add_argument("--out", help="сохранить снимок с разметкой")
    p.set_defaults(func=cmd_label_pose)

    p = sub.add_parser("reconcile")
    p.add_argument("expected", help="CSV из WMS/1С: адрес;код")
    p.add_argument("observations", help="CSV наблюдений: address;code;occupied;confidence")
    p.add_argument("-o", "--out", default="report.csv")
    p.set_defaults(func=cmd_reconcile)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # консоли Windows/Git Bash
    a = ap.parse_args(argv)
    a.func(a)

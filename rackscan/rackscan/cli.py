"""Командная строка стенда:  python -m rackscan <команда> [параметры]

    run               запустить сервер оператора (трансляция + наведение + сверка)
    selftest          самопроверка на синтетической сцене, без железа
    optics            расчёт рабочих дистанций для камеры и этикетки
    gen-layout        создать файл модели стенда
    print             листы A4: адресные карточки и этикетки на коробки
    wms               эмулятор склада: init-demo | export | import | report
    calibrate         калибровка камеры по шахматной доске
    calibrate-empty   снять эталоны пустых ячеек
    check             проверить камеру, датчики и связь с WMS
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, config


def _utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass


def cmd_run(a) -> int:
    over: dict = {}
    if a.source:
        over.setdefault("camera", {})["source"] = a.source
    if a.port:
        over.setdefault("server", {})["port"] = a.port
    cfg = config.load(a.config, over)
    from .server import serve
    serve(cfg)
    return 0


def cmd_selftest(a) -> int:
    from . import selftest
    return selftest.main(verbose=not a.quiet, out_dir=a.out)


def cmd_optics(a) -> int:
    from .optics import CAMERA_PRESETS, camera_from_preset, max_exposure_s, max_read_distance, plan_distances
    presets = [a.camera] if a.camera else list(CAMERA_PRESETS)
    print(f"Этикетка: модуль {a.module} мм; цель — не меньше {a.ppm} пикселей на модуль")
    print(f"Ячейка: {a.cell_w} × {a.cell_h} м\n")
    print(f"{'Камера':38s} {'d обзора':>9s} {'d чтения':>9s} {'d макс.':>8s} {'px/мод':>7s}  одним шагом")
    for name in presets:
        p = CAMERA_PRESETS[name]
        still = camera_from_preset(name)
        w = min(1536, p["full"][0])
        preview = camera_from_preset(name, (w, int(w * p["full"][1] / p["full"][0])))
        plan = plan_distances(preview, still, a.cell_w, a.cell_h, a.module, a.ppm)
        print(f"{p['title']:38s} {plan.d_cell:8.2f}м {plan.d_label:8.2f}м {plan.d_label_max:7.2f}м "
              f"{plan.ppm_still_at_label:7.1f}  {'да' if plan.feasible_single_step else 'нет — нужен подход'}")
    print("\nДальность чтения типовых кодов полным кадром (м), при", a.ppm, "px на модуль:")
    mods = [("QR 40 мм, версия 1", 40 / 21), ("QR 30 мм, версия 2", 30 / 25), ("GS1-128, X = 0,50 мм", 0.5),
            ("GS1-128, X = 1,00 мм", 1.0), ("EAN-13 100 %, X = 0,33 мм", 0.33), ("Data Matrix, X = 0,50 мм", 0.5)]
    print(f"{'Код':30s}" + "".join(f"{n:>14s}" for n in presets))
    for title, m in mods:
        print(f"{title:30s}" + "".join(f"{max_read_distance(camera_from_preset(n), m, a.ppm):14.2f}" for n in presets))
    print(f"\nВыдержка при движении {a.speed} м/с вдоль фасада (смаз ≤ 0,5 модуля): "
          f"{max_exposure_s(a.speed, a.module) * 1000:.1f} мс")
    return 0


def cmd_gen_layout(a) -> int:
    from .layout import Layout, Rack
    racks, x = [], 0.0
    shelf_y = [a.first_shelf + i * (a.cell_h + a.beam) for i in range(a.levels)]
    for i in range(1, a.racks + 1):
        r = Rack(id=i, cells_per_level=a.cells, cell_width_mm=a.cell_w, cell_height_mm=a.cell_h,
                 cell_depth_mm=a.depth, shelf_y_mm=list(shelf_y), beam_height_mm=a.beam,
                 upright_width_mm=a.upright, origin_x_mm=x)
        racks.append(r)
        x += r.width_mm + a.gap
    lay = Layout.build(racks, a.name, a.dictionary, a.marker)
    lay.save(a.out)
    print(f"Модель стенда записана: {a.out} — стеллажей {a.racks}, ячеек {len(lay.addresses())}")
    return 0


def cmd_print(a) -> int:
    from . import printing
    from .layout import Layout
    cfg = config.load(a.config)
    lay = Layout.load(cfg["layout"]) if cfg.get("layout") else Layout.demo()
    out = Path(a.out)
    p1 = printing.save_pdf(printing.address_cards(lay), out / "address_cards.pdf")
    print("Адресные карточки:", p1)
    from .wms.demo_data import label_text
    from .wms.emulator import EmulatorBackend
    db = cfg["wms"].get("db")
    items = []
    if db and Path(db).exists():
        for e in EmulatorBackend(db).stock():
            items.append({"text": label_text(e.sscc), "title": printing.sscc_hri(e.sscc),
                          "sub": f"{e.name} · ячейка {e.address}"})
    if not items:
        from .wms.demo_data import build_demo
        wms, _, _ = build_demo(lay, seed=cfg.get("seed", 7))
        items = [{"text": label_text(e.sscc), "title": printing.sscc_hri(e.sscc),
                  "sub": f"{e.name} · ячейка {e.address}"} for e in wms.stock()]
    lab = cfg["label"]
    p2 = printing.save_pdf(printing.box_labels(items, lab.get("format", "QRCode"), float(lab["code_mm"])),
                           out / "box_labels.pdf")
    print(f"Этикетки на коробки ({len(items)} шт.):", p2)
    print("Печатайте в масштабе 100 % и проверьте контрольную линейку 100 мм на листе.")
    return 0


def cmd_wms(a) -> int:
    from .layout import Layout
    from .wms import reconcile
    from .wms.demo_data import build_demo
    from .wms.emulator import EmulatorBackend
    cfg = config.load(a.config)
    db = a.db or cfg["wms"]["db"]
    if a.action == "init-demo":
        lay = Layout.load(cfg["layout"]) if cfg.get("layout") else Layout.demo()
        wms, _, expect = build_demo(lay, seed=cfg.get("seed", 7), db_path=db, discrepancies=False)
        print(f"Учебный склад создан: {db} — ячеек {len(wms.cells())}, контейнеров {len(wms.stock())}")
        return 0
    if a.action == "discover":
        import os
        from .wms.onec import discover
        url = a.url or cfg["wms"].get("odata_url")
        if not url:
            print("Укажите адрес OData: --url http://сервер/база/odata/standard.odata")
            return 1
        found = discover(url, os.environ.get(cfg["wms"].get("user_env", "")) or None,
                         os.environ.get(cfg["wms"].get("password_env", "")) or None)
        print(f"Сущностей в составе OData: {found.pop('всего сущностей')}")
        for group, items in found.items():
            print(f"\n{group}: {len(items)}")
            for it in items[:25]:
                print(f"  {it['entity']}")
                print("      " + ", ".join(it["properties"][:14]) + (" …" if len(it["properties"]) > 14 else ""))
        if a.file:
            Path(a.file).write_text(json.dumps(found, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
            print("\nПолный список записан:", a.file)
        return 0
    wms = EmulatorBackend(db)
    if a.action == "export":
        Path(a.file).write_text(wms.export_stock_csv(), encoding="utf-8", newline="\n")
        print("Остатки выгружены:", a.file)
    elif a.action == "import":
        n = wms.import_stock_csv(Path(a.file).read_text(encoding="utf-8-sig"))
        print(f"Загружено строк остатков: {n}; ячеек в справочнике: {len(wms.cells())}")
    elif a.action == "report":
        tasks = wms.tasks()
        if not tasks:
            print("Заданий нет")
            return 1
        rep = wms.report(str(a.task or tasks[0]["id"]))
        if a.file:
            text = reconcile.to_html(rep) if a.file.endswith(".html") else reconcile.to_csv(rep["lines"])
            Path(a.file).write_text(text, encoding="utf-8", newline="\n")
            print("Ведомость записана:", a.file)
        print(json.dumps(rep["summary"], ensure_ascii=False, indent=2))
    return 0


def cmd_calibrate(a) -> int:
    """Калибровка по снимкам шахматной доски (модель камеры — Gao & Zhang, ф. 4.5–4.12)."""
    import cv2
    import numpy as np
    files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    cols, rows = a.cols, a.rows
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * a.square
    obj, img, size = [], [], None
    for f in files:
        g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        size = g.shape[::-1]
        ok, corners = cv2.findChessboardCorners(g, (cols, rows))
        if ok:
            corners = cv2.cornerSubPix(g, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            obj.append(objp)
            img.append(corners)
    if len(obj) < 8:
        print(f"Доска найдена на {len(obj)} снимках из {len(files)} — нужно не меньше 8 с разных ракурсов")
        return 1
    rms, K, D, _, _ = cv2.calibrateCamera(obj, img, size, None, None)
    out = {"width": size[0], "height": size[1], "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
           "dist": D.ravel().tolist(), "rms_px": rms, "images": len(obj)}
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Калибровка записана: {a.out}; ошибка репроекции {rms:.2f} px по {len(obj)} снимкам")
    if rms > 1.0:
        print("Ошибка больше 1 px — переснимите: доска должна быть плоской, резкой и занимать разные части кадра")
    return 0


def cmd_calibrate_empty(a) -> int:
    """Снять эталоны пустых ячеек: стеллаж пуст, камера по очереди наводится на ячейки."""
    import cv2
    from . import markers
    from .layout import Layout
    from .pose import polygon_visible_fraction, rectify, solve_facade_pose
    from .sources import make_source
    cfg = config.load(a.config)
    lay = Layout.load(cfg["layout"]) if cfg.get("layout") else Layout.demo()
    sc = config.scan_config(cfg)
    scene = None
    if cfg["camera"]["source"] == "synthetic":
        from .synth import Scene
        scene = Scene(lay)          # пустой стеллаж
    src = make_source(cfg, scene)
    out = Path(cfg.get("references_dir", "data/references"))
    out.mkdir(parents=True, exist_ok=True)
    want = {str(x) for x in lay.addresses()}
    done: set[str] = set()
    print("Медленно проведите камерой вдоль пустого стеллажа с дистанции обзора. Ctrl+C — закончить.")
    try:
        for _ in range(a.frames):
            fr = src.read()
            if fr is None:
                continue
            pose = solve_facade_pose(markers.detect(fr.image, lay.marker_dictionary), lay, src.preview_cam)
            if pose is None or abs(pose.yaw_deg) > 10:
                continue
            for adr in lay.addresses():
                if str(adr) in done or lay.rack_of(adr).row != pose.row_id:
                    continue
                rect = lay.rack_of(adr).cell_rect_mm(adr)
                if polygon_visible_fraction(pose.rect_polygon(rect), src.preview_cam.width, src.preview_cam.height) > 0.99:
                    roi = rectify(fr.image, pose, rect, sc.roi_px_per_mm)
                    cv2.imwrite(str(out / f"{adr}.png"), cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
                    done.add(str(adr))
                    print(f"  эталон {adr} снят ({len(done)}/{len(want)})")
            if done == want:
                break
            if scene is not None:      # синтетика: сами ведём камеру змейкой
                i = len(done) % len(lay.addresses())
                cx, cy = lay.rack_of(lay.addresses()[i]).cell_center_mm(lay.addresses()[i])
                src.position[:] = (cx / 1000, cy / 1000, 0.8)
                src.yaw = 0.0
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
    print(f"Готово: эталонов {len(done)} из {len(want)} в {out}")
    return 0 if done == want else 1


def cmd_check(a) -> int:
    """Проверка составных частей по очереди: камера, метки, датчики, WMS."""
    import time
    cfg = config.load(a.config)
    ok = True
    print(f"RackScan {__version__}\n1. Камера ({cfg['camera']['source']})")
    try:
        from . import markers
        from .layout import Layout
        from .sources import make_source
        lay = Layout.load(cfg["layout"]) if cfg.get("layout") else Layout.demo()
        scene = None
        if cfg["camera"]["source"] == "synthetic":
            from .wms.demo_data import build_demo
            _, scene, _ = build_demo(lay)
        src = make_source(cfg, scene)
        fr = src.read()
        t0 = time.monotonic()
        dets = markers.detect(fr.image, lay.marker_dictionary)
        dt = (time.monotonic() - t0) * 1000
        print(f"   кадр {fr.image.shape[1]}×{fr.image.shape[0]}, меток стенда в кадре: "
              f"{sum(lay.address_of(d.id) is not None for d in dets)}, поиск меток {dt:.0f} мс")
        still = src.capture_still()
        print("   полный кадр:", "нет" if still is None else f"{still.shape[1]}×{still.shape[0]}")
        src.close()
    except Exception as e:
        ok = False
        print("   ОШИБКА:", e)
    print("2. Сенсорная головка")
    if cfg["sensors"].get("enabled"):
        from .sensors import SensorHub
        hub = SensorHub(cfg["sensors"]["port"], cfg["sensors"].get("baud", 115200))
        time.sleep(1.0)
        r = hub.latest()
        if r is None:
            ok = False
            print("   ОШИБКА: нет данных —", hub.error or "проверьте порт и прошивку")
        else:
            print(f"   крен {r.roll_deg:.1f}°, тангаж {r.pitch_deg:.1f}°, вращение {r.gyro_norm:.1f} °/с, "
                  f"дальномер {r.tof_mm} мм")
        hub.close()
    else:
        print("   отключена в настройках (sensors.enabled = false)")
    print(f"3. WMS ({cfg['wms']['backend']})")
    try:
        if cfg["wms"]["backend"] == "onec_http":
            from .wms.onec import OneCHttpBackend
            b = OneCHttpBackend.from_config(cfg["wms"])
            print("   ответ сервиса:", b.ping(), "ячеек:", len(b.cells()))
        else:
            from .wms.emulator import EmulatorBackend
            b = EmulatorBackend(cfg["wms"]["db"]) if Path(cfg["wms"]["db"]).exists() else None
            print("   эмулятор:", "база ещё не создана (появится при первом запуске)" if b is None
                  else f"ячеек {len(b.cells())}, контейнеров {len(b.stock())}")
    except Exception as e:
        ok = False
        print("   ОШИБКА:", e)
    print("ИТОГ:", "всё в порядке" if ok else "есть ошибки")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    _utf8()
    p = argparse.ArgumentParser(prog="rackscan", description="Стенд визуального сканирования стеллажей")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("run", help="запустить сервер оператора")
    s.add_argument("-c", "--config")
    s.add_argument("--source", choices=["synthetic", "picamera", "opencv", "files"])
    s.add_argument("--port", type=int)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("selftest", help="самопроверка на синтетической сцене")
    s.add_argument("-q", "--quiet", action="store_true")
    s.add_argument("--out", help="папка для фото и журнала")
    s.set_defaults(fn=cmd_selftest)

    s = sub.add_parser("optics", help="расчёт рабочих дистанций")
    s.add_argument("--camera", help="предустановка камеры (по умолчанию — все)")
    s.add_argument("--module", type=float, default=round(40 / 21, 3), help="ширина модуля кода, мм")
    s.add_argument("--ppm", type=float, default=3.0, help="пикселей на модуль")
    s.add_argument("--cell-w", type=float, default=0.40)
    s.add_argument("--cell-h", type=float, default=0.40)
    s.add_argument("--speed", type=float, default=0.3, help="скорость вдоль фасада, м/с")
    s.set_defaults(fn=cmd_optics)

    s = sub.add_parser("gen-layout", help="создать модель стенда")
    s.add_argument("--out", default="config/layout.json")
    s.add_argument("--name", default="stand")
    s.add_argument("--racks", type=int, default=1)
    s.add_argument("--levels", type=int, default=3)
    s.add_argument("--cells", type=int, default=3)
    s.add_argument("--cell-w", type=float, default=400, help="ширина ячейки, мм")
    s.add_argument("--cell-h", type=float, default=340, help="высота проёма, мм")
    s.add_argument("--depth", type=float, default=300, help="глубина ячейки, мм")
    s.add_argument("--beam", type=float, default=60, help="высота кромки полки, мм")
    s.add_argument("--upright", type=float, default=35, help="ширина стойки, мм")
    s.add_argument("--first-shelf", type=float, default=150, help="высота настила нижней полки, мм")
    s.add_argument("--gap", type=float, default=100, help="промежуток между стеллажами, мм")
    s.add_argument("--marker", type=float, default=50, help="сторона метки, мм")
    s.add_argument("--dictionary", default="DICT_5X5_1000")
    s.set_defaults(fn=cmd_gen_layout)

    s = sub.add_parser("print", help="листы A4 с метками и этикетками")
    s.add_argument("-c", "--config")
    s.add_argument("--out", default="out/print")
    s.set_defaults(fn=cmd_print)

    s = sub.add_parser("wms", help="эмулятор склада")
    s.add_argument("action", choices=["init-demo", "export", "import", "report", "discover"])
    s.add_argument("--url", help="адрес стандартного интерфейса OData базы 1С (для discover)")
    s.add_argument("-c", "--config")
    s.add_argument("--db")
    s.add_argument("--file")
    s.add_argument("--task")
    s.set_defaults(fn=cmd_wms)

    s = sub.add_parser("calibrate", help="калибровка камеры по шахматной доске")
    s.add_argument("images", help="папка со снимками доски")
    s.add_argument("--cols", type=int, default=9, help="внутренних углов по горизонтали")
    s.add_argument("--rows", type=int, default=6)
    s.add_argument("--square", type=float, default=25.0, help="сторона клетки, мм")
    s.add_argument("--out", default="config/calibration.json")
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("calibrate-empty", help="снять эталоны пустых ячеек")
    s.add_argument("-c", "--config")
    s.add_argument("--frames", type=int, default=2000)
    s.set_defaults(fn=cmd_calibrate_empty)

    s = sub.add_parser("check", help="проверить камеру, датчики и WMS")
    s.add_argument("-c", "--config")
    s.set_defaults(fn=cmd_check)

    a = p.parse_args(argv)
    if a.cmd == "wms" and a.action in ("export", "import") and not a.file:
        p.error("для export/import нужен --file")
    return a.fn(a)

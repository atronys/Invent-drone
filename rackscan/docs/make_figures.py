"""Иллюстрации для документации из синтетической сцены:  python docs/make_figures.py

Кадры окна оператора на разных этапах, фото-доказательства и лист адресных карточек
получаются прогоном настоящего конвейера, а не рисуются вручную.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rackscan import overlay, printing, selftest  # noqa: E402
from rackscan.layout import Layout  # noqa: E402

FIG = Path(__file__).parent / "figures"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    FIG.mkdir(exist_ok=True)
    lay = Layout.demo()
    want = {"live_search": None, "live_navigate": None, "live_assess": None, "live_approach": None,
            "live_done": None, "live_manual": None}

    def on_step(n, img, snap, sc):
        def grab(key):
            if want[key] is None:
                want[key] = overlay.draw_live(img, snap, sc.overlay, lay)
        if snap["phase"] == "NAVIGATE" and n == 0:
            grab("live_search")
        if snap["phase"] == "NAVIGATE" and snap["target"] == "01-02-03" and snap["commands"]:
            grab("live_navigate")
        if snap["phase"] == "ASSESS" and snap["target"] == "01-02-03":
            grab("live_assess")
        if snap["phase"] == "APPROACH" and snap["status"] == "RED" and snap["target"] == "01-02-03":
            grab("live_approach")
        if snap["phase"] == "DONE" and snap["status"] == "GREEN" and snap["last_result"]["address"] == "01-02-03":
            grab("live_done")
        if snap["phase"] == "DONE" and snap["status"] == "YELLOW":
            grab("live_manual")

    with tempfile.TemporaryDirectory() as d:
        res = selftest.run(addresses=["01-01-01", "01-01-02", "01-02-03", "02-03-02"], out_dir=d, on_step=on_step,
                           max_steps=400)
        session = next(Path(d).iterdir())
        for src, dst in (("01-02-03_READ", "evidence_read"), ("01-01-01_READ_ERROR", "evidence_error"),
                         ("02-03-02_EMPTY", "evidence_empty")):
            f = next(session.glob(f"{src}_*_annot.jpg"))
            shutil.copy(f, FIG / f"{dst}.jpg")
    for key, img in want.items():
        if img is None:
            print("не получен кадр", key)
            continue
        small = cv2.resize(img, (1152, 648), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(FIG / f"{key}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 86])
    page = printing.address_cards(lay, dpi=100)[0]
    page.save(FIG / "address_cards.png")
    from rackscan.wms.demo_data import label_text
    from rackscan.codes import make_sscc
    items = [{"text": label_text(s), "title": printing.sscc_hri(s), "sub": "Чай чёрный, 100 пак. · ячейка 01-01-01"}
             for s in (make_sscc(1, "4609999", i) for i in range(1, 7))]
    printing.box_labels(items, dpi=100)[0].save(FIG / "box_labels.png")
    print("рисунки:", sorted(p.name for p in FIG.iterdir()), "| самопроверка:", res["ok"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

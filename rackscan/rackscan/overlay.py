"""Пометки на кадре: то, что оператор видит в окне трансляции, и подписи на фото-доказательствах.

Кириллицу OpenCV рисовать не умеет (шрифты Hershey — только латиница), поэтому текст
готовится через Pillow небольшими плашками и накладывается на кадр; плашки кешируются.
Если Pillow или шрифта нет, текст транслитерируется и рисуется средствами OpenCV.
"""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None

COLORS = {"RED": (40, 40, 220), "YELLOW": (0, 190, 240), "GREEN": (60, 170, 40), "IDLE": (120, 120, 120)}
_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]
_TRANSLIT = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                     ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s",
                      "t", "u", "f", "h", "c", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya"]))


@lru_cache(maxsize=1)
def _font_path() -> str | None:
    return next((p for p in _FONTS if Path(p).exists()), None)


@lru_cache(maxsize=16)
def _font(size: int):
    return ImageFont.truetype(_font_path(), size)


def _translit(s: str) -> str:
    return "".join(_TRANSLIT.get(c, _TRANSLIT.get(c.lower(), c).upper() if c.lower() in _TRANSLIT else c) for c in s)


@lru_cache(maxsize=256)
def _text_patch(text: str, size: int, color: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(BGR-плашка, альфа 0..1) с текстом."""
    font = _font(size)
    l, t, r, b = font.getbbox(text)
    w, h = max(1, r - l + 4), max(1, b - t + 4)
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).text((2 - l, 2 - t), text, fill=255, font=font)
    alpha = np.asarray(im, dtype=np.float32) / 255.0
    patch = np.empty((h, w, 3), np.float32)
    patch[:] = color
    return patch, alpha[..., None]


def put_text(img: np.ndarray, text: str, org: tuple[int, int], size: int = 24,
             color: tuple[int, int, int] = (255, 255, 255)) -> int:
    """Нарисовать текст (левый верхний угол в org). Возвращает ширину текста в пикселях."""
    if not text:
        return 0
    if Image is None or _font_path() is None:
        scale = size / 30.0
        cv2.putText(img, _translit(text), (org[0], org[1] + size), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    max(1, int(scale * 2)), cv2.LINE_AA)
        return int(len(text) * size * 0.6)
    patch, alpha = _text_patch(text, size, tuple(int(c) for c in color))
    h, w = patch.shape[:2]
    x, y = int(org[0]), int(org[1])
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return w
    a = alpha[y0 - y:y1 - y, x0 - x:x1 - x]
    region = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (region * (1 - a) + patch[y0 - y:y1 - y, x0 - x:x1 - x] * a).astype(np.uint8)
    return w


def _poly(img, pts, color, thick=2):
    cv2.polylines(img, [np.round(np.asarray(pts)).astype(np.int32)], True, color, thick, cv2.LINE_AA)


def _arrow(img, axis: str, text: str, color, s: float):
    h, w = img.shape[:2]
    cx, cy, L, th = w // 2, h // 2, int(110 * s), max(3, int(8 * s))
    if axis == "x":
        right = "ВПРАВО" in text
        x = w - int(40 * s) if right else int(40 * s)
        cv2.arrowedLine(img, (x - L if right else x + L, cy), (x, cy), color, th, cv2.LINE_AA, tipLength=0.35)
    elif axis == "y":
        up = "ВВЕРХ" in text
        y = int(150 * s) if up else h - int(70 * s)
        cv2.arrowedLine(img, (cx, y + L if up else y - L), (cx, y), color, th, cv2.LINE_AA, tipLength=0.35)
    elif axis == "yaw":
        left = "ВЛЕВО" in text
        r = int(70 * s)
        c = (int(120 * s) if left else w - int(120 * s), h - int(120 * s))
        cv2.ellipse(img, c, (r, r), 0, 200, 340, color, th, cv2.LINE_AA)
        tip = (c[0] - r, c[1] - int(20 * s)) if left else (c[0] + r, c[1] - int(20 * s))
        cv2.arrowedLine(img, (tip[0], tip[1] - int(10 * s)), (tip[0], tip[1] + int(18 * s)), color, th, cv2.LINE_AA,
                        tipLength=0.9)


def draw_live(image: np.ndarray, snap: dict, ov: dict, layout=None) -> np.ndarray:
    """Кадр трансляции с пометками. Исходный кадр не изменяется."""
    img = image.copy()
    h, w = img.shape[:2]
    s = w / 1536.0
    color = COLORS.get(snap.get("status", "IDLE"), COLORS["IDLE"])

    for d in ov.get("markers", []):
        _poly(img, d.corners, (230, 200, 0), 1)
        if layout is not None and (a := layout.address_of(d.id)) is not None:
            c = d.corners.min(axis=0)
            cv2.putText(img, str(a), (int(c[0]), int(c[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (230, 200, 0),
                        max(1, int(s)), cv2.LINE_AA)
    if ov.get("cell_poly") is not None:
        _poly(img, ov["cell_poly"], color, max(2, int(3 * s)))
    for r in ov.get("codes", []):
        _poly(img, r.corners, (60, 200, 60) if r.decoded else (0, 140, 255), max(2, int(2 * s)))
    c = (w // 2, h // 2)
    cv2.drawMarker(img, c, (255, 255, 255), cv2.MARKER_CROSS, int(36 * s), max(1, int(2 * s)), cv2.LINE_AA)
    tp = ov.get("target_px")
    if tp is not None and np.all(np.isfinite(tp)):
        t = (int(np.clip(tp[0], -w, 2 * w)), int(np.clip(tp[1], -h, 2 * h)))
        cv2.line(img, c, t, color, max(1, int(2 * s)), cv2.LINE_AA)
        cv2.circle(img, t, int(12 * s), color, max(2, int(2 * s)), cv2.LINE_AA)
    for cmd in snap.get("commands", [])[:3]:
        _arrow(img, cmd["axis"], cmd["text"], color, s)

    # верхняя плашка: главное указание и подсказка
    bar = int(96 * s)
    cv2.rectangle(img, (0, 0), (w, bar), color, cv2.FILLED)
    put_text(img, snap.get("headline", ""), (int(18 * s), int(8 * s)), int(44 * s))
    put_text(img, snap.get("hint", ""), (int(20 * s), int(60 * s)), int(24 * s))
    # нижняя строка: цель и поза
    parts = []
    if snap.get("target"):
        parts.append(f"цель {snap['target']}")
    p = snap.get("pose")
    if p:
        need = snap.get("target_distance_m")
        parts.append(f"дистанция {p['distance_m']:.2f} м" + (f" (нужно {need:.2f})" if need else ""))
        parts.append(f"курс {p['yaw_deg']:+.0f}°")
        parts.append(f"меток {p['markers']}")
    pr = snap.get("progress") or {}
    if pr.get("total"):
        parts.append(f"готово {pr['done']}/{pr['total']}")
    if parts:
        cv2.rectangle(img, (0, h - int(40 * s)), (w, h), (30, 30, 30), cv2.FILLED)
        put_text(img, " · ".join(parts), (int(14 * s), h - int(34 * s)), int(24 * s), (235, 235, 235))
    return img


def annotate_evidence(image: np.ndarray, result: dict, pose, layout, reads) -> np.ndarray:
    """Копия кадра для отчёта: рамка ячейки, найденные коды, подпись с адресом, итогом и временем."""
    from .layout import Address
    from .wms.reconcile import FLAG_RU, SCAN_RU
    k = min(1.0, 1600.0 / image.shape[1])
    img = cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else image.copy()
    h, w = img.shape[:2]
    s = w / 1536.0
    ok = result["scan_status"] in ("READ", "EMPTY")
    color = COLORS["GREEN"] if ok else COLORS["YELLOW"]
    if pose is not None:
        a = Address.parse(result["address"])
        _poly(img, pose.rect_polygon(layout.rack_of(a).cell_rect_mm(a)) * k, color, max(2, int(3 * s)))
    for r in reads or []:
        _poly(img, r.corners * k, (60, 200, 60) if r.decoded else (0, 140, 255), max(2, int(2 * s)))
    # подпись — отдельной полосой над кадром, чтобы не закрывать изображение
    bar = int(74 * s)
    strip = np.full((bar, w, 3), 30, np.uint8)
    cv2.rectangle(strip, (0, 0), (int(14 * s), bar), color, cv2.FILLED)
    line1 = f"{result['address']} · {SCAN_RU.get(result['scan_status'], result['scan_status'])}"
    if result.get("codes"):
        line1 += " · " + ", ".join(result["codes"])
    put_text(strip, line1, (int(26 * s), int(6 * s)), int(30 * s))
    line2 = datetime.now().astimezone().strftime("%d.%m.%Y %H:%M:%S %z")
    p = result.get("pose")
    if p:
        line2 += f" · дистанция {p['distance_m']:.2f} м · курс {p['yaw_deg']:+.1f}°"
    if result.get("flags"):
        line2 += " · " + ", ".join(FLAG_RU.get(f, f) for f in result["flags"])
    put_text(strip, line2, (int(26 * s), int(42 * s)), int(22 * s), (210, 210, 210))
    return np.vstack([strip, img])

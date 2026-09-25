"""Этикетки: расчёт оптики, разбор GS1 (SSCC, GTIN, «Честный знак»), декодирование.

Декодер — zxing-cpp (Apache-2.0, годится для коммерческого продукта):
    pip install zxing-cpp numpy
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .camera import Camera

# Минимум пикселей на модуль (самый узкий штрих). ~2 — предел, 2.5–3 — проектная цель.
MIN_PPM = 2.5
GS = "\x1d"  # FNC1-разделитель полей переменной длины в «сыром» GS1

# --- оптика -------------------------------------------------------------------

def px_per_module(cam: Camera, distance: float, module_mm: float) -> float:
    return cam.px_per_mm(distance) * module_mm


def max_read_distance(cam: Camera, module_mm: float, min_ppm: float = MIN_PPM) -> float:
    """Максимальная дистанция (м), на которой модуль module_mm даёт min_ppm пикселей."""
    return cam.px_per_mm(1.0) * module_mm / min_ppm  # px_per_mm ~ 1/d


def required_vertical_px(spread_m: float, module_mm: float, min_ppm: float = MIN_PPM, margin: float = 0.2) -> float:
    """Сколько пикселей по вертикали нужно, чтобы в одном кадре были этикетки с разносом по высоте
    spread_m (адресная на балке + паллетная) и читался модуль module_mm.

    Не зависит от дистанции: высота кадра на фасаде = H / (px/мм), а px/мм задаётся модулем.
    H >= spread / (1 - margin) · (min_ppm / module_mm) · 1000.
    """
    return spread_m / (1 - margin) * (min_ppm / module_mm) * 1000


def motion_blur_px(cam: Camera, distance: float, speed: float, exposure_s: float) -> float:
    """Смаз в пикселях при движении вдоль фасада со скоростью speed (м/с)."""
    return speed * exposure_s * 1000 * cam.px_per_mm(distance)


# --- GS1 ----------------------------------------------------------------------

_FIXED = {"00": 18, "01": 14, "02": 14, "11": 6, "12": 6, "13": 6, "15": 6, "16": 6, "17": 6, "20": 2}
_VARIABLE = {"10": 20, "21": 20, "22": 20, "30": 8, "37": 8, "91": 90, "92": 90, "93": 90,
             "240": 30, "241": 30, "400": 30}


def gs1_check_digit(digits: str) -> int:
    """Контрольная цифра GS1 (mod 10) для строки без контрольной цифры."""
    total = sum(int(c) * (3 if i % 2 == 0 else 1) for i, c in enumerate(reversed(digits)))
    return (10 - total % 10) % 10


def is_valid_sscc(s: str) -> bool:
    return len(s) == 18 and s.isdigit() and gs1_check_digit(s[:17]) == int(s[17])


def _ai_len(data: str, i: int) -> tuple[str, int | None, int]:
    """(AI, фиксированная длина данных или None, макс. длина) для позиции i."""
    for n in (2, 3):
        ai = data[i:i + n]
        if ai in _FIXED:
            return ai, _FIXED[ai], _FIXED[ai]
        if ai in _VARIABLE:
            return ai, None, _VARIABLE[ai]
    ai = data[i:i + 4]
    if len(ai) == 4 and ai.isdigit() and "31" <= ai[:2] <= "36":  # 310n..369n — вес/размеры
        return ai, 6, 6
    raise ValueError(f"неизвестный GS1 AI в позиции {i}: {data[i:i + 4]!r}")


def parse_gs1(text: str) -> dict[str, str]:
    """Разбор GS1 в форматах «(00)1234…(10)ABC» и «сырой» (']C1'/']d2'/']Q3' + GS-разделители)."""
    if text.startswith("("):
        return {ai: val for ai, val in re.findall(r"\((\d{2,4})\)([^(]*)", text)}
    data = text[3:] if text[:3] in ("]C1", "]d2", "]Q3", "]e0") else text
    out, i = {}, 0
    while i < len(data):
        if data[i] == GS:
            i += 1
            continue
        ai, fixed, max_len = _ai_len(data, i)
        i += len(ai)
        if fixed is not None:
            val = data[i:i + fixed]
            i += fixed
        else:
            end = data.find(GS, i)
            end = len(data) if end == -1 else end
            val = data[i:min(end, i + max_len)]
            i += len(val)
        out[ai] = val
    return out


# --- декодирование ------------------------------------------------------------

@dataclass(frozen=True)
class Read:
    text: str
    symbology: str
    corners: tuple[tuple[float, float], ...]   # углы в пикселях

    @property
    def center(self) -> tuple[float, float]:
        n = len(self.corners)
        return sum(c[0] for c in self.corners) / n, sum(c[1] for c in self.corners) / n

    @property
    def sscc(self) -> str | None:
        try:
            s = parse_gs1(self.text).get("00")
        except ValueError:
            return None
        return s if s and is_valid_sscc(s) else None


def decode(image) -> list[Read]:
    """Все штрихкоды на изображении (numpy HxW или HxWx3). Требует zxing-cpp."""
    import zxingcpp

    reads = []
    for r in zxingcpp.read_barcodes(image):
        pos = r.position
        corners = tuple((float(p.x), float(p.y)) for p in (pos.top_left, pos.top_right, pos.bottom_right, pos.bottom_left))
        reads.append(Read(r.text, str(r.format).split(".")[-1], corners))
    return reads

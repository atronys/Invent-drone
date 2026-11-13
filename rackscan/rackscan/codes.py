"""Коды на коробках: поиск кандидатов, декодирование, разбор содержимого (GS1/SSCC/GTIN).

Принцип из статьи A (Lin et al., Drones 2024): обнаружение и декодирование разделены.
Кандидат без декодирования — не «пусто», а повод подойти ближе; если и вблизи не читается —
это отдельный исход «ошибка чтения» с фотографией.

Декодер — zxing-cpp (Apache-2.0): QR, Data Matrix, Code 128/GS1-128, EAN-13, ITF-14 и др.
Запасной поиск кандидатов — детекторы OpenCV и «белая наклейка с мелким рисунком».
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

try:  # декодер обязателен на стенде, но модуль должен импортироваться и без него (тесты геометрии)
    import zxingcpp
except ImportError:  # pragma: no cover
    zxingcpp = None

GS = "\x1d"


# --- содержимое -----------------------------------------------------------------

def gs1_check_digit(digits: str) -> int:
    """Контрольная цифра GS1 (mod 10) для строки без контрольной цифры."""
    total = sum(int(c) * (3 if i % 2 == 0 else 1) for i, c in enumerate(reversed(digits)))
    return (10 - total % 10) % 10


def is_valid_sscc(s: str) -> bool:
    return len(s) == 18 and s.isdigit() and gs1_check_digit(s[:17]) == int(s[17])


def is_valid_gtin(s: str) -> bool:
    return len(s) in (8, 12, 13, 14) and s.isdigit() and gs1_check_digit(s[:-1]) == int(s[-1])


def make_sscc(extension: int, company_prefix: str, serial: int) -> str:
    """SSCC из 18 цифр: цифра расширения + префикс компании + серийный номер + контрольная."""
    body = f"{extension}{company_prefix}"
    body += str(serial).zfill(17 - len(body))
    if len(body) != 17 or not body.isdigit():
        raise ValueError("SSCC: расширение + префикс + серийный номер должны дать 17 цифр")
    return body + str(gs1_check_digit(body))


_AI_FIXED = {"00": 18, "01": 14, "02": 14, "11": 6, "13": 6, "15": 6, "17": 6, "20": 2}
_AI_VAR = {"10": 20, "21": 20, "37": 8, "91": 90, "92": 90, "93": 90, "240": 30, "400": 30}


def parse_gs1(text: str) -> dict[str, str]:
    """Разбор строки элементов GS1: «(00)…(10)…» либо сплошная с разделителями GS.
    Неизвестное содержимое — пустой словарь (это не ошибка: код может быть не GS1)."""
    if text.startswith("("):
        return {ai: val for ai, val in re.findall(r"\((\d{2,4})\)([^(]*)", text)}
    data = text[3:] if text[:1] == "]" else text
    out: dict[str, str] = {}
    i = 0
    while i < len(data):
        for n in (2, 3):
            ai = data[i:i + n]
            if ai in _AI_FIXED:
                ln = _AI_FIXED[ai]
                out[ai] = data[i + n:i + n + ln]
                i += n + ln
                break
            if ai in _AI_VAR:
                end = data.find(GS, i + n)
                end = len(data) if end < 0 else end
                out[ai] = data[i + n:min(end, i + n + _AI_VAR[ai])]
                i = end + 1
                break
        else:
            return {}
        if i < len(data) and data[i] == GS:
            i += 1
    if any(not v for v in out.values()):
        return {}
    return out


@dataclass(frozen=True)
class Payload:
    """Смысл прочитанного кода. key — то, что сверяется с WMS."""

    raw: str
    kind: str           # 'sscc' | 'gtin' | 'text'
    key: str
    sscc: str | None = None
    gtin: str | None = None
    fields: dict = field(default_factory=dict)


def parse_payload(text: str) -> Payload:
    t = text.strip()
    digits = t if t.isdigit() else ""
    if len(digits) == 20 and digits.startswith("00") and is_valid_sscc(digits[2:]):
        return Payload(t, "sscc", digits[2:], sscc=digits[2:])
    if is_valid_sscc(digits):
        return Payload(t, "sscc", digits, sscc=digits)
    g = parse_gs1(t)
    if g.get("00") and is_valid_sscc(g["00"]):
        return Payload(t, "sscc", g["00"], sscc=g["00"], gtin=g.get("01"), fields=g)
    if g.get("01") and is_valid_gtin(g["01"]):
        key = g["01"] + (("/" + g["21"]) if g.get("21") else "")
        return Payload(t, "gtin", key, gtin=g["01"], fields=g)
    if is_valid_gtin(digits):
        return Payload(t, "gtin", digits, gtin=digits)
    return Payload(t, "text", t)


# --- результаты чтения ----------------------------------------------------------

@dataclass
class CodeRead:
    kind: str                     # 'decoded' | 'error' | 'candidate'
    corners: np.ndarray           # (4,2) в пикселях исходного кадра
    text: str = ""
    fmt: str = ""
    error: str = ""
    modules: int | None = None    # число модулей на сторону (для QR известно из версии)
    source: str = "zxing"

    @property
    def center(self) -> np.ndarray:
        return self.corners.mean(axis=0)

    @property
    def side_px(self) -> float:
        c = self.corners
        return float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))

    @property
    def width_px(self) -> float:
        c = self.corners
        return float(max(np.linalg.norm(c[0] - c[1]), np.linalg.norm(c[3] - c[2])))

    @property
    def decoded(self) -> bool:
        return self.kind == "decoded"

    @property
    def payload(self) -> Payload | None:
        return parse_payload(self.text) if self.decoded else None

    def px_per_module(self) -> float | None:
        return self.side_px / self.modules if self.modules else None


def _corners(pos, off) -> np.ndarray:
    pts = [(p.x, p.y) for p in (pos.top_left, pos.top_right, pos.bottom_right, pos.bottom_left)]
    return np.array(pts, dtype=np.float32) + np.array(off, dtype=np.float32)


def read_codes(image: np.ndarray, offset: tuple[float, float] = (0, 0), scale: float = 1.0,
               with_errors: bool = True) -> list[CodeRead]:
    """Декодировать все коды на изображении. offset/scale переводят координаты в исходный кадр
    (если на вход подан вырез, увеличенный в scale раз)."""
    if zxingcpp is None:
        raise RuntimeError("не установлен zxing-cpp: pip install zxing-cpp")
    out: list[CodeRead] = []
    kwargs = {"return_errors": True} if with_errors else {}
    for b in zxingcpp.read_barcodes(image, **kwargs):
        c = _corners(b.position, (0, 0)) / scale + np.array(offset, dtype=np.float32)
        fmt = str(b.format)
        modules = None
        extra = getattr(b, "extra", None)
        if isinstance(extra, dict) and str(extra.get("Version", "")).isdigit() and "QR" in fmt.upper():
            modules = 17 + 4 * int(extra["Version"])
        if b.valid:
            out.append(CodeRead("decoded", c, b.text, fmt, modules=modules))
        elif "QR" in fmt.upper():
            # «Найден, но не декодирован» учитываем только для QR: три поисковых узора случайно
            # не складываются. Для линейных кодов и Data Matrix декодер выдаёт ложные «ошибки» на
            # шуме и полосатой фактуре (скотч, рёбра коробок) — их ловит поиск наклеек.
            out.append(CodeRead("error", c, "", fmt, error=str(getattr(b, "error", "")) or "ошибка декодирования"))
    return out


def read_codes_robust(image: np.ndarray, offset: tuple[float, float] = (0, 0)) -> list[CodeRead]:
    """Чтение с несколькими вариантами предобработки. Используется на полном кадре в момент
    съёмки, когда время не критично: исходник, ×2, CLAHE, нерезкое маскирование."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants: list[tuple[np.ndarray, float]] = [(gray, 1.0)]
    if max(gray.shape) < 2600:
        variants.append((cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), 2.0))
    variants.append((cv2.createCLAHE(2.0, (8, 8)).apply(gray), 1.0))
    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    variants.append((cv2.addWeighted(gray, 1.8, blur, -0.8, 0), 1.0))
    best: dict[str, CodeRead] = {}
    errors: list[CodeRead] = []
    for img, s in variants:
        for r in read_codes(img, offset, s):
            if r.decoded:
                best.setdefault(r.text, r)
            else:
                errors.append(r)
        if best:
            break  # первый же вариант, давший чтение, достаточен
    return list(best.values()) + (errors if not best else [])


# --- кандидаты без декодирования -------------------------------------------------

def _box(x, y, w, h, off) -> np.ndarray:
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32) + np.array(off, np.float32)


def find_candidates(image: np.ndarray, offset: tuple[float, float] = (0, 0), min_side_px: int = 14,
                    expected_side_px: float | None = None, mask: np.ndarray | None = None) -> list[CodeRead]:
    """Места, похожие на этикетку с кодом, без попытки прочитать.

    1) детектор QR OpenCV (ищет три поисковых узора);
    2) «наклейка»: светлое пятно, внутри которого много резких перепадов яркости.
    expected_side_px — ожидаемый размер наклейки в пикселях (известен из позы и размера этикетки):
    позволяет отрезать от пятна узкие светлые полосы (скотч) и отсеять пятна не того размера.
    mask — где искать (например, только внутри проёма ячейки).
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    out: list[CodeRead] = []
    try:
        ok, pts = cv2.QRCodeDetector().detectMulti(gray)
        if ok and pts is not None:
            for quad in np.asarray(pts).reshape(-1, 4, 2):
                c = quad.mean(axis=0)
                if mask is None or mask[int(np.clip(c[1], 0, gray.shape[0] - 1)), int(np.clip(c[0], 0, gray.shape[1] - 1))]:
                    out.append(CodeRead("candidate", quad.astype(np.float32) + np.array(offset, np.float32),
                                        fmt="QR?", source="cv-qr"))
    except cv2.error:
        pass
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    thr, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = (g > max(thr, 172)).astype(np.uint8) * 255
    if mask is not None:
        bright[mask == 0] = 0
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # залить внутренность: символ кода внутри наклейки тёмный, но наклейка — одно целое
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(bright)
    cv2.drawContours(solid, contours, -1, 255, cv2.FILLED)
    if expected_side_px:
        k = max(3, int(0.4 * expected_side_px) | 1)
        solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        min_side_px = max(min_side_px, int(0.45 * expected_side_px))
    n, _, stats, _ = cv2.connectedComponentsWithStats(solid, 8)
    edges = cv2.Canny(gray, 60, 160)
    area_img = gray.shape[0] * gray.shape[1]
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < min_side_px or h < min_side_px or area > 0.6 * area_img:
            continue
        if expected_side_px and max(w, h) > 3.0 * expected_side_px:
            continue
        if not (0.25 < w / h < 4.0) or area < 0.6 * w * h:
            continue
        inner = edges[y + h // 6:y + h - h // 6, x + w // 6:x + w - w // 6]
        density = float(inner.mean()) / 255.0 if inner.size else 0.0
        if density > 0.05:  # у чистого белого листа перепадов нет, у кода — много
            quad = _box(x, y, w, h, offset)
            if not any(np.linalg.norm(quad.mean(axis=0) - c.center) < max(w, h) * 0.6 for c in out):
                out.append(CodeRead("candidate", quad, fmt="label?", source="sticker"))
    return out


def merge_reads(reads: list[CodeRead], candidates: list[CodeRead]) -> list[CodeRead]:
    """Объединить результаты: кандидат, совпавший по месту с прочитанным кодом, отбрасывается."""
    out = list(reads)
    for c in candidates:
        if not any(np.linalg.norm(c.center - r.center) < 0.7 * max(c.side_px, r.side_px) for r in out):
            out.append(c)
    return out


# --- генерация символов (для печати этикеток и синтетической сцены) ----------------

def encode_symbol(text: str, fmt: str = "QRCode") -> np.ndarray:
    """Матрица кода: 0 — тёмный модуль, 255 — светлый, 1 пиксель на модуль, без свободной зоны.
    Для линейных кодов возвращается одна строка высотой 1."""
    if zxingcpp is None:
        raise RuntimeError("не установлен zxing-cpp")
    f = getattr(zxingcpp.BarcodeFormat, fmt)
    if hasattr(zxingcpp, "create_barcode"):
        kw = {"ec_level": "M"} if fmt == "QRCode" else {}
        img = np.array(zxingcpp.write_barcode_to_image(zxingcpp.create_barcode(text, f, **kw), scale=1,
                                                       add_quiet_zones=False))
    else:  # zxing-cpp 2.x
        img = np.array(zxingcpp.write_barcode(f, text, 0, 0, 0))
    if fmt in ("Code128", "EAN13", "ITF", "Code39"):
        img = img[:1, :]
    return img.astype(np.uint8)

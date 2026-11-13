"""Печатные листы A4: адресные карточки с метками ArUco и этикетки на коробки.

Печатать строго в масштабе 100 % («фактический размер»), иначе размер метки не совпадёт с
моделью стенда и дистанция будет измеряться с ошибкой. На каждом листе есть контрольная
линейка 100 мм — проверьте её обычной линейкой после печати.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import codes, markers
from .layout import Layout
from .overlay import _font_path

A4_MM = (210.0, 297.0)


def _mm(v: float, dpi: int) -> int:
    return int(round(v / 25.4 * dpi))


def _font(size_mm: float, dpi: int):
    path = _font_path()
    return ImageFont.truetype(path, _mm(size_mm, dpi)) if path else ImageFont.load_default()


def _page(dpi: int, title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    page = Image.new("L", (_mm(A4_MM[0], dpi), _mm(A4_MM[1], dpi)), 255)
    d = ImageDraw.Draw(page)
    d.text((_mm(12, dpi), _mm(6, dpi)), title, fill=0, font=_font(3.2, dpi))
    # контрольная линейка 100 мм
    x0, y = _mm(12, dpi), _mm(289, dpi)
    d.line((x0, y, x0 + _mm(100, dpi), y), fill=0, width=3)
    for i in range(11):
        x = x0 + _mm(10 * i, dpi)
        d.line((x, y - _mm(2, dpi), x, y), fill=0, width=3)
    d.text((x0 + _mm(103, dpi), y - _mm(3.2, dpi)), "контроль масштаба: 100 мм", fill=0, font=_font(2.8, dpi))
    return page, d


def _paste_symbol(page: Image.Image, sym: np.ndarray, x_mm: float, y_mm: float, w_mm: float, h_mm: float, dpi: int):
    img = Image.fromarray(sym).resize((_mm(w_mm, dpi), _mm(h_mm, dpi)), Image.NEAREST)
    page.paste(img, (_mm(x_mm, dpi), _mm(y_mm, dpi)))


def address_cards(layout: Layout, dpi: int = 300) -> list[Image.Image]:
    """Адресные карточки: метка заданного размера, белое поле, адрес цифрами и словами."""
    ms = layout.marker_size_mm
    bits = markers.dictionary_bits(layout.marker_dictionary)
    pad = max(5.0, ms / (bits + 2))             # белое поле не уже одного модуля метки
    cw, ch = ms + 2 * pad, ms + 2 * pad + 13
    cols = int((A4_MM[0] - 20) // (cw + 4))
    rows = int((A4_MM[1] - 32) // (ch + 4))
    pages, addrs = [], layout.addresses()
    for start in range(0, len(addrs), cols * rows):
        page, d = _page(dpi, f"RackScan · адресные карточки · стенд «{layout.name}» · метка {ms:g} мм, "
                             f"{layout.marker_dictionary} · печатать в масштабе 100 %")
        for i, a in enumerate(addrs[start:start + cols * rows]):
            x = 10 + (i % cols) * (cw + 4)
            y = 14 + (i // cols) * (ch + 4)
            d.rectangle((_mm(x, dpi), _mm(y, dpi), _mm(x + cw, dpi), _mm(y + ch, dpi)), outline=160, width=2)
            sym = markers.render(layout.marker_dictionary, layout.marker_id(a), (bits + 2) * 20)
            _paste_symbol(page, sym, x + pad, y + pad, ms, ms, dpi)
            d.text((_mm(x + pad, dpi), _mm(y + pad + ms + 1.5, dpi)), str(a), fill=0, font=_font(5.5, dpi))
            d.text((_mm(x + pad, dpi), _mm(y + pad + ms + 8, dpi)),
                   f"стеллаж {a.rack} · полка {a.level} · место {a.pos} · ID {layout.marker_id(a)}",
                   fill=60, font=_font(2.3, dpi))
        pages.append(page)
    return pages


def box_labels(items: list[dict], fmt: str = "QRCode", code_mm: float = 40.0, dpi: int = 300) -> list[Image.Image]:
    """Этикетки на коробки. items: [{'text': содержимое кода, 'title': подпись, 'sub': вторая строка}]."""
    linear = fmt in ("Code128", "EAN13", "ITF")
    w_code, h_code = (code_mm, code_mm * 0.32) if linear else (code_mm, code_mm)
    lw, lh = w_code + 22, h_code + 30
    cols = int((A4_MM[0] - 20) // (lw + 4))
    rows = int((A4_MM[1] - 32) // (lh + 4))
    pages = []
    for start in range(0, len(items), cols * rows):
        page, d = _page(dpi, f"RackScan · этикетки на коробки · {fmt} · символ {code_mm:g} мм · масштаб 100 %")
        for i, it in enumerate(items[start:start + cols * rows]):
            x = 10 + (i % cols) * (lw + 4)
            y = 14 + (i // cols) * (lh + 4)
            d.rectangle((_mm(x, dpi), _mm(y, dpi), _mm(x + lw, dpi), _mm(y + lh, dpi)), outline=160, width=2)
            sym = codes.encode_symbol(it["text"], fmt)
            _paste_symbol(page, sym, x + 11, y + 9, w_code, h_code, dpi)
            d.text((_mm(x + 4, dpi), _mm(y + 9 + h_code + 4, dpi)), it.get("title", it["text"]), fill=0,
                   font=_font(3.4, dpi))
            if it.get("sub"):
                d.text((_mm(x + 4, dpi), _mm(y + 9 + h_code + 10, dpi)), it["sub"], fill=60, font=_font(2.6, dpi))
        pages.append(page)
    return pages


def save_pdf(pages: list[Image.Image], path: str | Path, dpi: int = 300) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = [p.convert("RGB") for p in pages]
    rgb[0].save(path, "PDF", resolution=dpi, save_all=True, append_images=rgb[1:])
    return path


def sscc_hri(sscc: str) -> str:
    """Читаемая запись SSCC под кодом: (00) 1 4609999 000000001 8."""
    return f"(00) {sscc[0]} {sscc[1:8]} {sscc[8:17]} {sscc[17]}"

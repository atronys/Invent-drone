"""Учебный склад: учётные остатки + физическая сцена с заранее известными расхождениями.

Номера SSCC и GTIN — тестовые (префикс 4609999 в GS1 не зарегистрирован за нами),
наносить их на реальный товар нельзя.
"""
from __future__ import annotations

import random

from ..codes import gs1_check_digit, make_sscc
from ..layout import Address, Layout
from ..synth import BoxSpec, LabelSpec, Scene
from .emulator import EmulatorBackend

TEST_PREFIX = "4609999"

ITEMS = [
    ("АРТ-0001", "Чай чёрный, 100 пак."), ("АРТ-0002", "Кофе молотый, 250 г"),
    ("АРТ-0003", "Сахар-песок, 1 кг"), ("АРТ-0004", "Макароны, 450 г"),
    ("АРТ-0005", "Масло подсолнечное, 1 л"), ("АРТ-0006", "Гречка, 800 г"),
    ("АРТ-0007", "Рис круглозёрный, 900 г"), ("АРТ-0008", "Печенье овсяное, 300 г"),
    ("АРТ-0009", "Вода питьевая, 0,5 л × 12"), ("АРТ-0010", "Салфетки бумажные, 100 шт."),
]


def demo_gtin(n: int) -> str:
    """GTIN-13 из диапазона 200–299 (внутреннее обращение, не пересекается с настоящими)."""
    body = f"200{n:09d}"
    return body + str(gs1_check_digit(body))


def label_text(sscc: str) -> str:
    """Содержимое кода на коробке: идентификатор применения (00) + SSCC, 20 цифр."""
    return "00" + sscc


def build_demo(layout: Layout | None = None, seed: int = 7, db_path: str = ":memory:",
               discrepancies: bool = True, label_fmt: str = "QRCode",
               code_mm: float = 40.0) -> tuple[EmulatorBackend, Scene, dict[str, dict]]:
    """Возвращает (эмулятор WMS, физическая сцена, ожидаемые исходы по ячейкам).

    Ожидаемый исход: {'scan': статус сканирования, 'verdict': вердикт сверки, 'flags': [...]}.
    """
    layout = layout or Layout.demo()
    rng = random.Random(seed)
    wms = EmulatorBackend(db_path)
    wms.clear()
    for i, (sku, name) in enumerate(ITEMS, 1):
        wms.add_item(sku, name, demo_gtin(i))
    scene = Scene(layout, seed=seed)
    expect: dict[str, dict] = {}
    addrs = layout.addresses()
    for a in addrs:
        wms.add_cell(str(a), a.rack, a.level, a.pos)

    serial = 0

    def new_sscc() -> str:
        nonlocal serial
        serial += 1
        return make_sscc(1, TEST_PREFIX, serial)

    def box(sscc: str | None, rack, x=None, w=None, damaged=False) -> BoxSpec:
        w = w or rng.choice([240.0, 260.0, 280.0, 300.0])
        x = (rack.cell_width_mm - w) / 2 + rng.uniform(-15, 15) if x is None else x
        h = min(rack.cell_height_mm - 40, rng.choice([200.0, 230.0, 260.0]))
        lab = LabelSpec(label_text(sscc), label_fmt, code_mm if label_fmt != "Code128" else 90.0,
                        damaged=damaged) if sscc else None
        return BoxSpec(x, w, h, lab)

    # --- учётное состояние и совпадающая с ним физика --------------------------------
    occupied: list[Address] = []
    for a in addrs:
        rack = layout.rack_of(a)
        if rng.random() < 0.78:
            sscc = new_sscc()
            wms.put(sscc, rng.choice(ITEMS)[0], str(a))
            scene.cells[a] = [box(sscc, rack)]
            occupied.append(a)
            expect[str(a)] = {"scan": "READ", "verdict": "MATCH", "flags": []}
        else:
            expect[str(a)] = {"scan": "EMPTY", "verdict": "MATCH", "flags": []}
    if not discrepancies:
        return wms, scene, expect

    empty = [a for a in addrs if a not in occupied]
    rng.shuffle(occupied)
    rng.shuffle(empty)
    pick = occupied.pop

    # 1. недостача: по учёту есть, физически пусто
    a = pick()
    scene.cells.pop(a)
    expect[str(a)] = {"scan": "EMPTY", "verdict": "SHORTAGE", "flags": []}
    # 2. излишек: по учёту пусто, физически стоит неучтённая коробка
    if empty:
        a = empty.pop()
        scene.cells[a] = [box(new_sscc(), layout.rack_of(a))]
        expect[str(a)] = {"scan": "READ", "verdict": "SURPLUS", "flags": []}
    # 3. пересортица: две коробки физически поменяны местами
    a, b = pick(), pick()
    scene.cells[a], scene.cells[b] = scene.cells[b], scene.cells[a]
    expect[str(a)] = {"scan": "READ", "verdict": "MISMATCH", "flags": []}
    expect[str(b)] = {"scan": "READ", "verdict": "MISMATCH", "flags": []}
    # 4. коробка без этикетки
    a = pick()
    scene.cells[a] = [box(None, layout.rack_of(a))]
    expect[str(a)] = {"scan": "NO_LABEL", "verdict": "MANUAL", "flags": []}
    # 5. повреждённая этикетка
    a = pick()
    old = wms.stock(str(a))[0].sscc
    scene.cells[a] = [box(old, layout.rack_of(a), damaged=True)]
    expect[str(a)] = {"scan": "READ_ERROR", "verdict": "MANUAL", "flags": []}
    # 6. две коробки в ячейке с большим зазором (обе учтены)
    a = pick()
    rack = layout.rack_of(a)
    first = wms.stock(str(a))[0].sscc
    second = new_sscc()
    wms.put(second, rng.choice(ITEMS)[0], str(a))
    scene.cells[a] = [box(first, rack, x=8.0, w=125.0), box(second, rack, x=rack.cell_width_mm - 133.0, w=125.0)]
    expect[str(a)] = {"scan": "READ", "verdict": "MATCH", "flags": ["GAP"]}
    return wms, scene, expect

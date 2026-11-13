"""Модель стенда: стеллажи, полки (ярусы), ячейки, адреса и карта адресных меток.

Система координат фасада стеллажа (F), метры/миллиметры:
    X — вправо, если смотреть на фасад из прохода; Y — вверх; Z — из фасада в проход.
    Начало — левый нижний угол стеллажа на уровне пола. Плоскость Z = 0 — передние
    кромки полок (на них висят адресные метки).

Адрес ячейки: «СС-ЯЯ-ММ» — стеллаж, ярус (полка), место. Тот же адрес заводится
в справочник ячеек WMS.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

_ADDR_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{2})$")


@dataclass(frozen=True, order=True)
class Address:
    rack: int
    level: int
    pos: int

    def __str__(self) -> str:
        return f"{self.rack:02d}-{self.level:02d}-{self.pos:02d}"

    @property
    def human(self) -> str:
        return f"Стеллаж {self.rack:02d} · Полка {self.level} · Место {self.pos}"

    @classmethod
    def parse(cls, text: str) -> "Address":
        m = _ADDR_RE.match(text.strip())
        if not m:
            raise ValueError(f"адрес ячейки должен иметь вид СС-ЯЯ-ММ, получено: {text!r}")
        return cls(int(m.group(1)), int(m.group(2)), int(m.group(3)))


@dataclass
class Rack:
    """Один стеллаж. Ячейки одного яруса одинаковой ширины; ярусы задаются высотой настила."""

    id: int
    cells_per_level: int = 3
    cell_width_mm: float = 400.0
    cell_height_mm: float = 340.0     # высота проёма над настилом (до кромки верхней полки)
    cell_depth_mm: float = 300.0
    shelf_y_mm: list[float] = field(default_factory=lambda: [150.0, 550.0, 950.0])  # верх настила яруса
    beam_height_mm: float = 60.0      # высота передней кромки, на которой висит адресная метка
    upright_width_mm: float = 35.0
    origin_x_mm: float = 0.0          # смещение стеллажа вдоль общего фасада ряда
    row: int = 1                      # ряд: стеллажи одного ряда стоят в одну линию, фасад общий

    @property
    def levels(self) -> int:
        return len(self.shelf_y_mm)

    @property
    def width_mm(self) -> float:
        return 2 * self.upright_width_mm + self.cells_per_level * self.cell_width_mm

    @property
    def height_mm(self) -> float:
        return self.shelf_y_mm[-1] + self.cell_height_mm + self.beam_height_mm

    def cells(self) -> Iterator[Address]:
        for level in range(1, self.levels + 1):
            for pos in range(1, self.cells_per_level + 1):
                yield Address(self.id, level, pos)

    def cell_rect_mm(self, a: Address) -> tuple[float, float, float, float]:
        """Проём ячейки (x0, y0, x1, y1) в системе фасада, мм."""
        self._check(a)
        x0 = self.origin_x_mm + self.upright_width_mm + (a.pos - 1) * self.cell_width_mm
        y0 = self.shelf_y_mm[a.level - 1]
        return x0, y0, x0 + self.cell_width_mm, y0 + self.cell_height_mm

    def cell_center_mm(self, a: Address) -> tuple[float, float]:
        x0, y0, x1, y1 = self.cell_rect_mm(a)
        return (x0 + x1) / 2, (y0 + y1) / 2

    def marker_center_mm(self, a: Address) -> tuple[float, float]:
        """Адресная метка — на кромке полки под ячейкой, по центру ячейки."""
        x0, y0, x1, _ = self.cell_rect_mm(a)
        return (x0 + x1) / 2, y0 - self.beam_height_mm / 2

    def _check(self, a: Address) -> None:
        if a.rack != self.id or not (1 <= a.level <= self.levels) or not (1 <= a.pos <= self.cells_per_level):
            raise KeyError(f"ячейки {a} нет в стеллаже {self.id}")


@dataclass
class Layout:
    name: str = "stand"
    marker_dictionary: str = "DICT_5X5_1000"
    marker_size_mm: float = 50.0
    racks: dict[int, Rack] = field(default_factory=dict)
    marker_ids: dict[int, Address] = field(default_factory=dict)   # ID метки -> адрес

    # --- построение -------------------------------------------------------------
    @classmethod
    def build(cls, racks: list[Rack], name: str = "stand", marker_dictionary: str = "DICT_5X5_1000",
              marker_size_mm: float = 50.0) -> "Layout":
        lay = cls(name, marker_dictionary, marker_size_mm, {r.id: r for r in racks})
        lay.assign_marker_ids()
        return lay

    @classmethod
    def demo(cls) -> "Layout":
        """Учебный стенд: 2 стеллажа по 3 яруса × 3 места = 18 ячеек."""
        r1 = Rack(id=1)
        r2 = Rack(id=2, origin_x_mm=r1.width_mm + 100.0)
        return cls.build([r1, r2], name="demo")

    def assign_marker_ids(self) -> None:
        """ID = (стеллаж−1)·100 + (ярус−1)·10 + (место−1), если влезает; иначе по порядку."""
        capacity = int(re.search(r"_(\d+)$", self.marker_dictionary).group(1))
        cells = [a for r in self.racks.values() for a in r.cells()]
        compact = all(a.level <= 10 and a.pos <= 10 for a in cells) and \
            max((a.rack for a in cells), default=1) * 100 <= capacity
        self.marker_ids = {}
        for i, a in enumerate(sorted(cells)):
            mid = (a.rack - 1) * 100 + (a.level - 1) * 10 + (a.pos - 1) if compact else i
            if mid >= capacity:
                raise ValueError(f"словарь {self.marker_dictionary} вмещает {capacity} меток, ячеек больше")
            self.marker_ids[mid] = a

    # --- запросы ----------------------------------------------------------------
    def addresses(self) -> list[Address]:
        return sorted(self.marker_ids.values())

    def address_of(self, marker_id: int) -> Address | None:
        return self.marker_ids.get(int(marker_id))

    def marker_id(self, a: Address) -> int:
        for mid, addr in self.marker_ids.items():
            if addr == a:
                return mid
        raise KeyError(f"для ячейки {a} нет метки")

    def rack_of(self, a: Address) -> Rack:
        return self.racks[a.rack]

    def marker_corners_m(self, a: Address) -> np.ndarray:
        """Углы метки в системе фасада, м: порядок как у детектора OpenCV
        (левый верхний, правый верхний, правый нижний, левый нижний)."""
        cx, cy = self.rack_of(a).marker_center_mm(a)
        h = self.marker_size_mm / 2
        pts = np.array([[cx - h, cy + h, 0], [cx + h, cy + h, 0], [cx + h, cy - h, 0], [cx - h, cy - h, 0]])
        return pts / 1000.0

    def bounds_mm(self) -> tuple[float, float, float, float]:
        x1 = max(r.origin_x_mm + r.width_mm for r in self.racks.values())
        y1 = max(r.height_mm for r in self.racks.values())
        x0 = min(r.origin_x_mm for r in self.racks.values())
        return x0, 0.0, x1, y1

    def serpentine(self) -> list[Address]:
        """Порядок обхода «змейкой»: стеллаж за стеллажом, ярусы снизу вверх, направление чередуется."""
        order: list[Address] = []
        for rid in sorted(self.racks):
            r = self.racks[rid]
            for level in range(1, r.levels + 1):
                row = [Address(rid, level, p) for p in range(1, r.cells_per_level + 1)]
                order.extend(row if level % 2 == 1 else row[::-1])
        return order

    # --- сериализация -----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "marker_dictionary": self.marker_dictionary,
            "marker_size_mm": self.marker_size_mm,
            "racks": [asdict(r) for r in self.racks.values()],
            "marker_ids": {str(k): str(v) for k, v in sorted(self.marker_ids.items())},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Layout":
        racks = [Rack(**r) for r in d["racks"]]
        lay = cls(d.get("name", "stand"), d.get("marker_dictionary", "DICT_5X5_1000"),
                  float(d.get("marker_size_mm", 50.0)), {r.id: r for r in racks})
        if d.get("marker_ids"):
            lay.marker_ids = {int(k): Address.parse(v) for k, v in d["marker_ids"].items()}
        else:
            lay.assign_marker_ids()
        return lay

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: str | Path) -> "Layout":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

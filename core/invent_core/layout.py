"""Модель склада: стеллажные ряды, проходы, ячейки (адреса).

Система координат склада `map` (ENU, метры):
  x — вдоль рядов (вдоль прохода), y — поперёк рядов, z — вверх, 0 — пол.
Ряд задаётся плоскостью фасада (y = face_y) и нормалью фасада normal_y = ±1,
которая смотрит в проход — туда, откуда дрон видит этикетки.
Адрес ячейки: "<ряд>-<секция:02>-<ярус>-<место>", например "B-07-3-2".
Проходы задаются явно (как в CAD-плане склада), а не выводятся из геометрии.
"""
from __future__ import annotations

import json
import string
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Location:
    address: str
    row: str
    bay: int      # секция, с 1
    level: int    # ярус, с 1 (1 — нижний, на полу)
    slot: int     # паллетоместо в секции, с 1
    x: float      # центр места вдоль ряда
    z: float      # высота, на которой ожидается паллетная этикетка


@dataclass
class RackRow:
    id: str
    origin_x: float                 # x начала ряда (край первой секции)
    face_y: float                   # y плоскости фасада
    normal_y: int                   # +1: фасад смотрит в +y; -1: в -y
    bays: int
    bay_width: float = 2.7          # типичная секция на 3 европаллеты
    slots_per_bay: int = 3
    level_z: list[float] = field(default_factory=lambda: [0.0, 1.8, 3.6, 5.4])
    label_height: float = 0.5       # высота паллетной этикетки над балкой (GS1: 0.4–0.8 м от низа паллеты)
    address_offset: float = -0.06   # адресная этикетка: на лицевой балке яруса, центр на 6 см ниже её верха
    floor_address_z: float = 0.10   # у нижнего (напольного) яруса — на отбойнике/стойке

    def __post_init__(self) -> None:
        if self.normal_y not in (1, -1):
            raise ValueError(f"ряд {self.id}: normal_y должен быть +1 или -1")
        if not self.level_z or list(self.level_z) != sorted(self.level_z):
            raise ValueError(f"ряд {self.id}: level_z должен быть непустым и возрастающим")

    @property
    def end_x(self) -> float:
        return self.origin_x + self.bays * self.bay_width

    @property
    def slot_width(self) -> float:
        return self.bay_width / self.slots_per_bay

    def address_z(self, level: int) -> float:
        """Высота центра адресной этикетки яруса level (с 1). Ориентир для локализации (LOC-08/11)."""
        lz = self.level_z[level - 1]
        return lz + self.address_offset if lz > 0 else self.floor_address_z

    def address(self, bay: int, level: int, slot: int) -> str:
        return f"{self.id}-{bay:02d}-{level}-{slot}"

    def locations(self) -> list[Location]:
        out = []
        for b in range(1, self.bays + 1):
            for lvl, lz in enumerate(self.level_z, start=1):
                for s in range(1, self.slots_per_bay + 1):
                    x = self.origin_x + (b - 1) * self.bay_width + (s - 0.5) * self.slot_width
                    out.append(Location(self.address(b, lvl, s), self.id, b, lvl, s, x, lz + self.label_height))
        return out

    def locate(self, x: float, z: float) -> str | None:
        """Адрес места, на фасаде которого лежит точка (x, z); None — вне ряда."""
        if not (self.origin_x <= x < self.end_x) or z < self.level_z[0]:
            return None
        rel = x - self.origin_x
        bay = int(rel // self.bay_width) + 1
        slot = min(int((rel - (bay - 1) * self.bay_width) // self.slot_width) + 1, self.slots_per_bay)
        level = max(i for i, lz in enumerate(self.level_z, start=1) if lz <= z)
        return self.address(bay, level, slot)


@dataclass(frozen=True)
class Aisle:
    id: str
    y: float                  # ось прохода — по ней летит дрон
    rows: tuple[str, ...]     # ряды, фасады которых выходят в проход (1 или 2)


@dataclass
class Warehouse:
    name: str
    rows: list[RackRow]
    aisles: list[Aisle]
    dock: tuple[float, float, float] = (-2.0, 0.0, 0.0)   # док/точка старта в map
    cross_aisle_x: float = -2.0                           # x магистрального проезда перед торцами рядов
    ceiling_z: float = 10.0

    def __post_init__(self) -> None:
        ids = [r.id for r in self.rows]
        if len(set(ids)) != len(ids):
            raise ValueError("id рядов должны быть уникальны")
        for a in self.aisles:
            for rid in a.rows:
                r = self.row(rid)
                if (a.y - r.face_y) * r.normal_y <= 0:
                    raise ValueError(f"ряд {rid} смотрит не в проход {a.id}")

    def row(self, row_id: str) -> RackRow:
        for r in self.rows:
            if r.id == row_id:
                return r
        raise KeyError(row_id)

    def aisle_of(self, row_id: str) -> Aisle:
        for a in self.aisles:
            if row_id in a.rows:
                return a
        raise KeyError(f"ряд {row_id} не принадлежит ни одному проходу")

    def locations(self) -> list[Location]:
        return [loc for r in self.rows for loc in r.locations()]

    def address_label_position(self, address: str) -> tuple[RackRow, float, float] | None:
        """Где на фасаде висит адресная этикетка ячейки: (ряд, x, z); None — нет такого адреса."""
        try:
            row_id, bay, level, slot = address.rsplit("-", 3)
            row, bay, level, slot = self.row(row_id), int(bay), int(level), int(slot)
        except (ValueError, KeyError):
            return None
        if not (1 <= bay <= row.bays and 1 <= level <= len(row.level_z) and 1 <= slot <= row.slots_per_bay):
            return None
        x = row.origin_x + (bay - 1) * row.bay_width + (slot - 0.5) * row.slot_width
        return row, x, row.address_z(level)

    # --- сериализация -------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Warehouse":
        return cls(
            name=d["name"],
            rows=[RackRow(**r) for r in d["rows"]],
            aisles=[Aisle(a["id"], a["y"], tuple(a["rows"])) for a in d["aisles"]],
            dock=tuple(d.get("dock", (-2.0, 0.0, 0.0))),
            cross_aisle_x=d.get("cross_aisle_x", -2.0),
            ceiling_z=d.get("ceiling_z", 10.0),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Warehouse":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            import yaml  # опционально: pip install pyyaml
            return cls.from_dict(yaml.safe_load(text))
        return cls.from_dict(json.loads(text))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def make_grid_warehouse(
    n_aisles: int = 2,
    bays: int = 10,
    levels: int = 4,
    level_step: float = 1.8,
    aisle_width: float = 3.0,
    rack_depth: float = 1.1,
    flue_gap: float = 0.2,
    bay_width: float = 2.7,
    slots_per_bay: int = 3,
    name: str = "demo",
) -> Warehouse:
    """Типовой склад: проходы, по обе стороны — ряды; между проходами ряды стоят спина к спине."""
    if 2 * n_aisles > len(string.ascii_uppercase):
        raise ValueError("не больше 13 проходов (ряды именуются A..Z)")
    level_z = [round(i * level_step, 3) for i in range(levels)]
    rows, aisles = [], []
    y = 0.0  # тыльная сторона очередного «левого» ряда
    for a in range(n_aisles):
        left_face = y + rack_depth
        right_face = left_face + aisle_width
        lid, rid = string.ascii_uppercase[2 * a], string.ascii_uppercase[2 * a + 1]
        common = dict(origin_x=0.0, bays=bays, bay_width=bay_width, slots_per_bay=slots_per_bay, level_z=level_z)
        rows.append(RackRow(id=lid, face_y=left_face, normal_y=1, **common))
        rows.append(RackRow(id=rid, face_y=right_face, normal_y=-1, **common))
        aisles.append(Aisle(id=str(a + 1), y=(left_face + right_face) / 2, rows=(lid, rid)))
        y = right_face + rack_depth + flue_gap
    return Warehouse(
        name=name,
        rows=rows,
        aisles=aisles,
        dock=(-2.0, aisles[0].y, 0.0),
        cross_aisle_x=-2.0,
        ceiling_z=level_z[-1] + level_step + 1.5,
    )

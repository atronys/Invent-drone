"""Карта ArUco-меток склада: где висят, какие ID, поза каждой метки в `map`.

Схема размещения (документ, разд. 6):
* якорные метки (anchor) — на торцах рядов у входа в проход, смотрят в магистральный проезд (−x);
  ID уникальны: одна метка сразу называет проход (повторная привязка после потери позы);
* путевые метки (path) — на лицевых стойках стеллажей, смотрят в проход; шаг step_bays секций,
  на противоположной стороне со сдвигом на step_bays/2 — «шахматка», метка то слева, то справа;
  ID повторяются от прохода к проходу (словарь 4×4_50 мал); различаются по ожидаемому месту.

Кадр метки — как у OpenCV (cv2.aruco, SOLVEPNP_IPPE_SQUARE): начало в центре, z — нормаль наружу
(к камере), x — вправо для смотрящего на метку, y — вверх. Углы: TL(−s/2, s/2), TR, BR, BL.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from invent_core.layout import Warehouse

DICTIONARY = "DICT_4X4_50"
DICT_SIZE = {"DICT_4X4_50": 50, "DICT_4X4_100": 100, "DICT_4X4_250": 250, "DICT_4X4_1000": 1000,
             "DICT_5X5_250": 250, "DICT_5X5_1000": 1000, "DICT_6X6_250": 250, "DICT_6X6_1000": 1000}
DICT_BITS = {"4X4": 4, "5X5": 5, "6X6": 6, "7X7": 7}


@dataclass(frozen=True)
class Marker:
    id: int
    kind: str                 # "anchor" | "path"
    aisle: str
    x: float
    y: float
    z: float
    normal: tuple[float, float, float]   # наружу, горизонтальная

    def pose(self) -> np.ndarray:
        """T_wm (4×4): кадр метки в map."""
        return marker_pose((self.x, self.y, self.z), self.normal)


def marker_rotation(normal) -> np.ndarray:
    """R_wm: столбцы — оси метки в map. z = нормаль, y = вверх, x = y × z (вправо для смотрящего)."""
    z = np.asarray(normal, dtype=float)
    z = z / np.linalg.norm(z)
    y = np.array([0.0, 0.0, 1.0])
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def marker_pose(p, normal) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = marker_rotation(normal)
    t[:3, 3] = p
    return t


def object_corners(size: float) -> np.ndarray:
    """Углы метки в её кадре в порядке детектора ArUco: TL, TR, BR, BL."""
    h = size / 2
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])


def module_size(size: float, dictionary: str = DICTIONARY) -> float:
    """Сторона модуля: n×n бит + рамка в один модуль с каждой стороны (документ, разд. 4.1)."""
    n = DICT_BITS[dictionary.split("_")[1]]
    return size / (n + 2)


@dataclass
class PlacementParams:
    size: float = 0.20                 # сторона метки (чёрный квадрат), м
    dictionary: str = DICTIONARY
    step_bays: int = 2                 # шаг путевых меток по одной стороне, секций
    heights: list[float] = field(default_factory=lambda: [1.5])   # высоты рядов путевых меток, м
    anchor_z: float = 1.0               # видна и с дока на земле, и с высоты транзита 1.5 м
    anchor_inset: float = 0.15         # якорь на торце ряда: от фасада вглубь ряда, м
    anchors: bool = True
    standoff: float = 0.01             # зазор метки от поверхности, м (против z-fighting)


@dataclass
class MarkerMap:
    dictionary: str
    size: float
    markers: list[Marker]

    def by_id(self, marker_id: int) -> list[Marker]:
        return [m for m in self.markers if m.id == marker_id]

    def anchors(self) -> list[Marker]:
        return [m for m in self.markers if m.kind == "anchor"]

    def to_dict(self) -> dict:
        return {"dictionary": self.dictionary, "size": self.size, "markers": [asdict(m) for m in self.markers]}

    @classmethod
    def from_dict(cls, d: dict) -> "MarkerMap":
        return cls(d["dictionary"], d["size"],
                   [Marker(m["id"], m["kind"], m["aisle"], m["x"], m["y"], m["z"], tuple(m["normal"]))
                    for m in d["markers"]])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MarkerMap":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def place_markers(wh: Warehouse, p: PlacementParams = PlacementParams()) -> MarkerMap:
    """Расставить метки по раскладке склада."""
    n_dict = DICT_SIZE[p.dictionary]
    markers: list[Marker] = []
    n_anchor = 2 * len(wh.aisles) if p.anchors else 0
    if n_anchor >= n_dict:
        raise ValueError(f"словарь {p.dictionary} мал: {n_anchor} якорей")
    next_anchor = 0
    pool = list(range(n_anchor, n_dict))     # ID путевых меток; в каждом проходе — с начала пула
    for aisle in wh.aisles:
        k_id = 0
        for row_id in aisle.rows:
            row = wh.row(row_id)
            n = row.normal_y
            if p.anchors:   # торец ряда у магистрального проезда, ближе к проходу
                markers.append(Marker(next_anchor, "anchor", aisle.id, row.origin_x - 0.04 - p.standoff,
                                      row.face_y - n * p.anchor_inset, p.anchor_z, (-1.0, 0.0, 0.0)))
                next_anchor += 1
        # путевые: стойки k = 0..bays; сторона 0 — чётные кратные шагу, сторона 1 — со сдвигом на полшага
        for z in p.heights:
            items = []
            for side, row_id in enumerate(aisle.rows):
                row = wh.row(row_id)
                offset = 0 if side == 0 else max(1, p.step_bays // 2)
                for k in range(offset, row.bays + 1, p.step_bays):
                    x = row.origin_x + k * row.bay_width
                    items.append((x, row.face_y + row.normal_y * p.standoff, (0.0, float(row.normal_y), 0.0)))
            for x, y, normal in sorted(items):
                markers.append(Marker(pool[k_id % len(pool)], "path", aisle.id, x, y, z, normal))
                k_id += 1
    return MarkerMap(p.dictionary, p.size, markers)


def min_repeat_distance(mm: MarkerMap) -> float:
    """Наименьшее расстояние между двумя метками с одинаковым ID (inf — ID не повторяются).
    Оно должно быть много больше неопределённости позы, иначе метки путаются (документ, разд. 4.4)."""
    best = math.inf
    by: dict[int, list[Marker]] = {}
    for m in mm.markers:
        by.setdefault(m.id, []).append(m)
    for ms in by.values():
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                best = min(best, math.dist((ms[i].x, ms[i].y, ms[i].z), (ms[j].x, ms[j].y, ms[j].z)))
    return best


def render_marker(marker_id: int, size_px: int = 600, dictionary: str = DICTIONARY, border_frac: float = 0.25):
    """Изображение метки с белым полем (border_frac стороны метки с каждой стороны). -> uint8 HxW."""
    import cv2
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    gen = getattr(cv2.aruco, "generateImageMarker", None) or cv2.aruco.drawMarker   # OpenCV ≥4.7 / 4.6
    img = gen(d, marker_id, size_px)
    pad = int(round(size_px * border_frac))
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

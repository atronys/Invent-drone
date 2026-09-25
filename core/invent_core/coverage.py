"""Планировщик облёта: маршрут сканирования фасадов с разбиением на вылеты по АКБ.

Модель (подробно — docs/02_architecture.md, раздел «Планирование миссии»):
* дрон летит носом вдоль прохода по его оси; две боковые камеры смотрят на оба фасада,
  фронтальная depth-камера — по ходу движения (видит препятствия);
* пролёт (pass) — горизонтальный отрезок на фиксированной высоте. Высоты выбираются
  жадным покрытием высот этикеток полосой обзора камеры (для 1D-покрытия жадность оптимальна);
* пролёты идут «змейкой», переходы между проходами — через магистральный проезд;
* вылеты набираются жадно: пролёт добавляется, если после него АКБ хватает на возврат к доку.
Выбор пролётов при ограничении АКБ — вариант orienteering problem (NP-трудна, ср. сведение
к set cover в статье «VOT ETO PROST SNOS»). Жадная эвристика — baseline (задача MIS-07).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from .camera import Camera
from .layout import Aisle, Warehouse


class PlanningError(ValueError):
    pass


@dataclass(frozen=True)
class ScanParams:
    camera: Camera = field(default_factory=Camera)
    view_margin: float = 0.2         # доля кадра по краям, которую не считаем надёжной
    min_z: float = 0.8               # ниже не летаем (люди, техника, эффект близости пола)
    transit_z: float = 2.0           # высота перелётов по магистральному проезду
    v_transit: float = 1.0           # м/с
    v_scan: float = 0.4              # м/с — ограничено смазом, см. barcode.motion_blur_px
    v_z: float = 0.5                 # м/с
    yaw_rate_deg: float = 30.0       # град/с — резкие развороты срывают VIO (статья A, сценарий 4)
    hold_s: float = 0.0              # зависание на точке сканирования (>0 — режим stop-and-scan)
    sortie_overhead_s: float = 20.0  # армирование, стабилизация, посадка
    battery_s: float = 600.0         # полётное время на одной АКБ
    reserve: float = 0.25            # резерв АКБ
    turnaround_s: float = 120.0      # замена/зарядка АКБ между вылетами

    @classmethod
    def load(cls, path: str | Path) -> "ScanParams":
        """JSON: {"v_scan": 0.4, ..., "camera": {...} | "camera_file": "left.yaml"} (FLT-02).

        Путь camera_file считается относительно файла параметров. Неизвестные ключи — ошибка
        (опечатка в имени параметра не должна молча игнорироваться).
        """
        path = Path(path)
        d = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)} | {"camera_file"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"{path}: неизвестные параметры {sorted(unknown)}")
        cam = Camera()
        if "camera_file" in d:
            cam = Camera.load(path.parent / d.pop("camera_file"))
        elif "camera" in d:
            cam = Camera(**d.pop("camera"))
        return replace(cls(camera=cam), **d)


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    z: float
    yaw: float                       # рад, ENU: 0 — вдоль +x
    kind: str = "transit"            # transit | scan
    hold_s: float = 0.0
    targets: tuple[str, ...] = ()    # адреса, которые должны быть в кадре на этой точке


@dataclass
class Pass:
    aisle: str
    z: float
    waypoints: list[Waypoint]


@dataclass
class Sortie:
    waypoints: list[Waypoint]
    duration_s: float

    @property
    def targets(self) -> set[str]:
        return {t for w in self.waypoints for t in w.targets}


@dataclass
class Plan:
    sorties: list[Sortie]
    warnings: list[str]
    turnaround_s: float = 0.0

    @property
    def flight_s(self) -> float:
        return sum(s.duration_s for s in self.sorties)

    @property
    def wall_clock_s(self) -> float:
        return self.flight_s + self.turnaround_s * max(0, len(self.sorties) - 1)

    @property
    def covered(self) -> set[str]:
        return set().union(*(s.targets for s in self.sorties)) if self.sorties else set()

    @property
    def locations_per_hour(self) -> float:
        return len(self.covered) / (self.wall_clock_s / 3600) if self.sorties else 0.0


def pass_heights(zs: list[float], band: float) -> list[tuple[float, list[float]]]:
    """Жадное покрытие точек на прямой отрезками длины band.

    Возвращает список (центр отрезка, покрытые точки). Минимальное число отрезков.
    """
    pts = sorted({round(z, 3) for z in zs})
    groups, i = [], 0
    while i < len(pts):
        j = i
        while j + 1 < len(pts) and pts[j + 1] - pts[i] <= band:
            j += 1
        groups.append(((pts[i] + pts[j]) / 2, pts[i : j + 1]))
        i = j + 1
    return groups


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def segment_time(a: Waypoint, b: Waypoint, p: ScanParams) -> float:
    """Консервативно: горизонталь, вертикаль и разворот выполняются последовательно."""
    speed = p.v_scan if a.kind == b.kind == "scan" else p.v_transit
    return (
        math.hypot(b.x - a.x, b.y - a.y) / speed
        + abs(b.z - a.z) / p.v_z
        + math.degrees(abs(_wrap(b.yaw - a.yaw))) / p.yaw_rate_deg
        + b.hold_s
    )


def route_time(wps: list[Waypoint], p: ScanParams) -> float:
    return sum(segment_time(a, b, p) for a, b in zip(wps, wps[1:]))


def _aisle_passes(wh: Warehouse, aisle: Aisle, p: ScanParams, direction: int):
    rows = [wh.row(r) for r in aisle.rows]
    dist = min(abs(aisle.y - r.face_y) for r in rows)  # ближний фасад — худший случай для полосы
    fp_w, fp_h = p.camera.footprint(dist)
    band, step = fp_h * (1 - p.view_margin), fp_w * (1 - p.view_margin)
    locs = [loc for r in rows for loc in r.locations()]
    x0, x1 = min(r.origin_x for r in rows), max(r.end_x for r in rows)
    n = max(1, math.ceil((x1 - x0) / step))
    spacing = (x1 - x0) / n                             # равномерно, spacing <= step
    xs = [x0 + (k + 0.5) * spacing for k in range(n)]

    passes, warnings = [], []
    for z_center, group in pass_heights([loc.z for loc in locs], band):
        z = max(z_center, p.min_z)
        lost = [g for g in group if abs(g - z) > band / 2 + 1e-9]
        if lost:
            warnings.append(
                f"проход {aisle.id}: этикетки на высотах {lost} вне полосы обзора при z={z:.2f} "
                f"(min_z={p.min_z}); нужен наклон камеры или меньший min_z"
            )
        yaw = 0.0 if direction > 0 else math.pi
        wps = []
        for x in xs if direction > 0 else reversed(xs):
            targets = tuple(
                loc.address for loc in locs if abs(loc.x - x) <= spacing / 2 + 1e-9 and abs(loc.z - z) <= band / 2 + 1e-9
            )
            wps.append(Waypoint(x, aisle.y, z, yaw, "scan", p.hold_s, targets))
        passes.append(Pass(aisle.id, z, wps))
        direction = -direction
    return passes, direction, warnings


def _transit(a: Waypoint, b: Waypoint, same_aisle: bool, x_cross: float, p: ScanParams) -> list[Waypoint]:
    """Точки перелёта от a до b (a не включается, b включается).

    В пределах прохода — сначала по вертикали, затем вдоль оси. Между проходами — выход
    в магистральный проезд, спуск/подъём до transit_z, перелёт, подъём/спуск, вход в проход.
    Носом по ходу движения, чтобы фронтальная depth-камера видела препятствия.
    """
    if same_aisle:
        pts = [(a.x, a.y, b.z), (b.x, b.y, b.z)]
    else:
        pts = [
            (x_cross, a.y, a.z),
            (x_cross, a.y, p.transit_z),
            (x_cross, b.y, p.transit_z),
            (x_cross, b.y, b.z),
            (b.x, b.y, b.z),
        ]
    out, prev, yaw = [], (a.x, a.y, a.z), a.yaw
    for q in pts:
        if math.dist(prev, q) < 1e-6:
            continue
        if math.hypot(q[0] - prev[0], q[1] - prev[1]) > 1e-6:
            yaw = math.atan2(q[1] - prev[1], q[0] - prev[0])
        out.append(Waypoint(q[0], q[1], q[2], yaw))
        prev = q
    if out and math.dist((out[-1].x, out[-1].y, out[-1].z), (b.x, b.y, b.z)) < 1e-6:
        out[-1] = b
    else:
        out.append(b)
    return out


def _restrict(ps: Pass, only: set[str]) -> Pass | None:
    """Оставить в пролёте участок от первой до последней точки, видящей адреса из only."""
    idx = [i for i, w in enumerate(ps.waypoints) if only.intersection(w.targets)]
    if not idx:
        return None
    wps = [replace(w, targets=tuple(t for t in w.targets if t in only)) for w in ps.waypoints[min(idx):max(idx) + 1]]
    return Pass(ps.aisle, ps.z, wps)


def plan(wh: Warehouse, p: ScanParams | None = None, only: set[str] | None = None) -> Plan:
    """План облёта всего склада или, если задан only, — перескана этих адресов (MIS-03)."""
    p = p or ScanParams()
    budget = p.battery_s * (1 - p.reserve) - p.sortie_overhead_s - 2 * p.transit_z / p.v_z
    dock = Waypoint(wh.dock[0], wh.dock[1], p.transit_z, 0.0)

    passes, warnings, direction = [], [], 1
    for aisle in sorted(wh.aisles, key=lambda a: a.y):
        ps, direction, w = _aisle_passes(wh, aisle, p, direction)
        passes += ps
        warnings += w
    if only is not None:
        unknown = only - {loc.address for loc in wh.locations()}
        if unknown:
            raise PlanningError(f"неизвестные адреса: {sorted(unknown)[:5]}")
        passes = [r for r in (_restrict(ps, only) for ps in passes) if r]

    def finish(route: list[Waypoint]) -> Sortie:
        full = route + _transit(route[-1], dock, False, wh.cross_aisle_x, p)
        return Sortie(full, p.sortie_overhead_s + 2 * p.transit_z / p.v_z + route_time(full, p))

    sorties: list[Sortie] = []
    route, t, cur_aisle = [dock], 0.0, None
    for ps in passes:
        for attempt in (0, 1):
            leg = _transit(route[-1], ps.waypoints[0], ps.aisle == cur_aisle, wh.cross_aisle_x, p) + ps.waypoints[1:]
            t_leg = route_time([route[-1]] + leg, p)
            t_home = route_time([leg[-1]] + _transit(leg[-1], dock, False, wh.cross_aisle_x, p), p)
            if t + t_leg + t_home <= budget:
                route, t, cur_aisle = route + leg, t + t_leg, ps.aisle
                break
            if attempt or len(route) == 1:
                raise PlanningError(
                    f"пролёт прохода {ps.aisle} на z={ps.z:.2f} не помещается в один вылет "
                    f"({t_leg + t_home:.0f} с > {budget:.0f} с): увеличьте АКБ или делите пролёт"
                )
            sorties.append(finish(route))
            route, t, cur_aisle = [dock], 0.0, None
    if len(route) > 1:
        sorties.append(finish(route))

    result = Plan(sorties, warnings, p.turnaround_s)
    missing = (only if only is not None else {loc.address for loc in wh.locations()}) - result.covered
    if missing:
        warnings.append(f"не покрыто ячеек: {len(missing)} (например, {sorted(missing)[:5]})")
    return result

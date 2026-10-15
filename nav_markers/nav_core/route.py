"""Навигационное задание: граф проходов склада и маршрут к целевым ячейкам (документ, разд. 8).

Три уровня:
1. склад — граф: узлы — входы в проходы на магистральном проезде (x = cross_aisle_x) и точки на осях
   проходов; рёбра — отрезки магистрали и осей; маршрут — Дейкстра;
2. проход — пролёт по оси прохода на высоте транзита z_transit со скоростью v_aisle;
3. ячейка — смещение с оси к фасаду до standoff и снижение/подъём к ярусу, медленно (v_approach),
   затем «захват»: зависание, пока по меткам нет свежей точной позы.

Курс постоянный (вдоль +x, к стеллажам): фронтальная камера смотрит на якоря на торцах рядов,
боковые — на путевые метки обеих сторон прохода. Резких разворотов нет — поток и метки их не любят.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from invent_core.layout import Warehouse


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    z: float
    speed: float          # с какой скоростью лететь К этой точке, м/с
    kind: str             # takeoff | cross | enter | aisle | approach | capture | exit | home | land
    label: str = ""
    hold_s: float = 0.0


@dataclass
class TaskParams:
    z_transit: float = 1.5
    v_cross: float = 1.0      # магистральный проезд
    v_aisle: float = 1.5      # по оси прохода (довод преподавателя: 1–1,5 м/с)
    v_approach: float = 0.3   # подлёт к ячейке
    v_z: float = 0.5
    standoff: float = 1.0     # от фасада до дрона при захвате ячейки, м
    capture_hold_s: float = 2.0
    entry_margin: float = 1.0  # точка «вход в проход» — на столько перед торцом ряда, м


def aisle_graph(wh: Warehouse):
    """-> (nodes {name: (x, y)}, edges {name: [(name, length)]}). Узлы: E<id> — вход в проход на магистрали."""
    nodes, edges = {}, {}

    def link(a, b):
        L = math.dist(nodes[a], nodes[b])
        edges.setdefault(a, []).append((b, L))
        edges.setdefault(b, []).append((a, L))

    aisles = sorted(wh.aisles, key=lambda a: a.y)
    for a in aisles:
        nodes[f"E{a.id}"] = (wh.cross_aisle_x, a.y)
    nodes["DOCK"] = (wh.dock[0], wh.dock[1])
    chain = sorted(nodes, key=lambda n: nodes[n][1])   # все узлы магистрали — по y
    for u, v in zip(chain, chain[1:]):
        link(u, v)
    return nodes, edges


def shortest_path(edges, start: str, goal: str) -> list[str]:
    dist, prev, heap = {start: 0.0}, {}, [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == goal:
            break
        if d > dist.get(u, math.inf):
            continue
        for v, L in edges.get(u, []):
            if d + L < dist.get(v, math.inf):
                dist[v], prev[v] = d + L, u
                heapq.heappush(heap, (d + L, v))
    if goal not in dist:
        raise ValueError(f"нет пути {start} -> {goal}")
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return path[::-1]


def cell_target(wh: Warehouse, address: str, p: TaskParams):
    """-> (проход, x, y, z) точки захвата ячейки: напротив центра паллетоместа, standoff от фасада,
    на высоте этикетки яруса (не ниже 0.5 м)."""
    pos = wh.address_label_position(address)
    if pos is None:
        raise ValueError(f"нет ячейки {address}")
    row, x, _ = pos
    loc = next(loc for loc in row.locations() if loc.address == address)
    aisle = wh.aisle_of(row.id)
    y = row.face_y + row.normal_y * p.standoff
    return aisle, x, y, max(0.5, loc.z)


def plan_task(wh: Warehouse, targets: list[str], p: TaskParams = TaskParams()) -> list[Waypoint]:
    """Маршрут: взлёт на доке -> для каждой ячейки: магистраль -> вход в проход -> ось -> подлёт -> захват
    -> обратно на ось -> ... -> выход из прохода -> док -> посадка."""
    nodes, edges = aisle_graph(wh)
    zt = p.z_transit
    dx, dy = wh.dock[0], wh.dock[1]
    wps = [Waypoint(dx, dy, zt, p.v_z, "takeoff", "взлёт")]
    here, in_aisle = "DOCK", None
    x_entry = {a.id: min(wh.row(r).origin_x for r in a.rows) - p.entry_margin for a in wh.aisles}

    def go_cross(goal):
        nonlocal here
        for n in shortest_path(edges, here, goal)[1:]:
            x, y = nodes[n]
            wps.append(Waypoint(x, y, zt, p.v_cross, "cross", n))
        here = goal

    for addr in targets:
        aisle, x, y, z = cell_target(wh, addr, p)
        if in_aisle is not None and in_aisle.id != aisle.id:   # выйти из прохода на магистраль
            wps.append(Waypoint(x_entry[in_aisle.id], in_aisle.y, zt, p.v_aisle, "exit", f"выход {in_aisle.id}"))
            wps.append(Waypoint(*nodes[f"E{in_aisle.id}"], zt, p.v_cross, "cross", f"E{in_aisle.id}"))
            here, in_aisle = f"E{in_aisle.id}", None
        if in_aisle is None:
            go_cross(f"E{aisle.id}")
            wps.append(Waypoint(x_entry[aisle.id], aisle.y, zt, p.v_cross, "enter", f"вход {aisle.id}"))
            in_aisle = aisle
        wps.append(Waypoint(x, aisle.y, zt, p.v_aisle, "aisle", f"ось {addr}"))
        wps.append(Waypoint(x, aisle.y, z, p.v_z, "aisle", f"ярус {addr}"))
        wps.append(Waypoint(x, y, z, p.v_approach, "approach", addr))
        wps.append(Waypoint(x, y, z, 0.0, "capture", addr, p.capture_hold_s))
        wps.append(Waypoint(x, aisle.y, z, p.v_approach, "aisle", f"отход {addr}"))
        wps.append(Waypoint(x, aisle.y, zt, p.v_z, "aisle", f"транзит {addr}"))
    if in_aisle is not None:
        wps.append(Waypoint(x_entry[in_aisle.id], in_aisle.y, zt, p.v_aisle, "exit", f"выход {in_aisle.id}"))
        wps.append(Waypoint(*nodes[f"E{in_aisle.id}"], zt, p.v_cross, "cross", f"E{in_aisle.id}"))
        here = f"E{in_aisle.id}"
    go_cross("DOCK")
    wps.append(Waypoint(dx, dy, zt, p.v_cross, "home", "над доком"))
    wps.append(Waypoint(dx, dy, 0.0, p.v_z, "land", "посадка"))
    return wps


def braking_distance(v: float, a_max: float) -> float:
    """Путь торможения v²/(2a) (документ, разд. 8.2)."""
    return v * v / (2 * a_max)

"""MIS-07: жадный планировщик вылетов против точного решения (глава 2, задача 1).

Задача: множество пролётов P (каждый можно пролететь в любую сторону) разбить на вылеты от дока и
обратно, каждый не длиннее бюджета АКБ, минимизируя (1) число вылетов (замен АКБ), (2) суммарное время.
Модель времени — та же, что в планировщике (`coverage.segment_time`, `_transit`), так что сравнение честное.

Точное решение для малых экземпляров (|P| ≤ 12):
1. Хелд–Карп по подмножествам: dp[S][(i, o)] — минимальное время маршрута от дока, покрывающего пролёты S и
   заканчивающегося пролётом i в направлении o; маршрут-вылет T(S) = min dp + возврат к доку.
2. Оптимальное разбиение по подмножествам: best[S] = min_{T ⊆ S, T ∋ min(S), T допустим} best[S∖T] ⊕ вылет(T),
   лексикографически (число вылетов, время). Сложность O(2^n n² + 3^n).

Жадный базовый метод — `coverage.plan`: пролёты в порядке «змейки», вылет заполняется, пока хватает АКБ.

    python research/sortie_planning/exact.py
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))
from invent_core.coverage import ScanParams, Waypoint, _aisle_passes, _transit, plan, route_time  # noqa: E402
from invent_core.layout import Warehouse, make_grid_warehouse  # noqa: E402


def _reverse(wps: list[Waypoint]) -> list[Waypoint]:
    return [replace(w, yaw=math.atan2(math.sin(w.yaw + math.pi), math.cos(w.yaw + math.pi))) for w in reversed(wps)]


def solve_exact(wh: Warehouse, p: ScanParams):
    """-> (число вылетов, полётное время с накладными, число пролётов) — оптимум модели."""
    passes = []
    for aisle in sorted(wh.aisles, key=lambda a: a.y):
        ps, _, _ = _aisle_passes(wh, aisle, p, 1)
        passes += ps
    n = len(passes)
    if n > 14:
        raise ValueError(f"{n} пролётов — слишком много для точного метода")
    dock = Waypoint(wh.dock[0], wh.dock[1], p.transit_z, 0.0)
    fixed = p.sortie_overhead_s + 2 * p.transit_z / p.v_z
    budget = p.battery_s * (1 - p.reserve) - fixed
    ori = [(ps.waypoints, _reverse(ps.waypoints)) for ps in passes]           # две ориентации пролёта
    t_pass = [route_time(ps.waypoints, p) for ps in passes]
    aisle = [ps.aisle for ps in passes]

    def trans(a, b, same):
        return route_time([a] + _transit(a, b, same, wh.cross_aisle_x, p), p)

    from_dock = [[trans(dock, ori[i][o][0], False) for o in (0, 1)] for i in range(n)]
    to_dock = [[trans(ori[i][o][-1], dock, False) for o in (0, 1)] for i in range(n)]
    between = [[[[trans(ori[i][o][-1], ori[j][q][0], aisle[i] == aisle[j]) for q in (0, 1)] for j in range(n)]
                for o in (0, 1)] for i in range(n)]

    INF = float("inf")
    full = 1 << n
    dp = [dict() for _ in range(full)]
    for i in range(n):
        for o in (0, 1):
            dp[1 << i][(i, o)] = from_dock[i][o] + t_pass[i]
    route = [INF] * full
    for S in range(1, full):
        cur = dp[S]
        if not cur:
            continue
        route[S] = min(v + to_dock[i][o] for (i, o), v in cur.items())
        best_end = min(cur.values())
        if best_end > budget:          # дальше только дольше — расширять бессмысленно
            continue
        for (i, o), v in cur.items():
            if v > budget:
                continue
            for j in range(n):
                if S >> j & 1:
                    continue
                T = S | (1 << j)
                for q in (0, 1):
                    w = v + between[i][o][j][q] + t_pass[j]
                    if w < dp[T].get((j, q), INF):
                        dp[T][(j, q)] = w
    best = [(INF, INF)] * full
    best[0] = (0, 0.0)
    for S in range(1, full):
        low = S & -S
        rest = S ^ low
        T = rest
        while True:
            sub = T | low
            if route[sub] <= budget and best[S ^ sub][0] < INF:
                k, tt = best[S ^ sub]
                cand = (k + 1, tt + route[sub] + fixed)
                if cand < best[S]:
                    best[S] = cand
            if T == 0:
                break
            T = (T - 1) & rest
    k, tt = best[full - 1]
    return k, tt, n


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cases = [  # проходы, секции, ярусы, АКБ (с): бюджет подобран, чтобы вылетов было несколько
        (1, 6, 4, 240), (1, 6, 4, 180), (2, 4, 3, 300), (2, 4, 3, 200), (2, 6, 3, 360),
        (2, 5, 4, 300), (3, 3, 3, 300), (3, 4, 3, 360), (3, 4, 4, 420), (2, 8, 5, 600),
    ]
    rows = ["| Склад (проходы × секции × ярусы), АКБ | Пролётов | Жадный «змейкой»: вылетов / время, с "
            "| Жадный + перебор: вылетов / время, с | Оптимум: вылетов / время, с | Разрыв змейки | Разрыв с перебором |",
            "|---|---|---|---|---|---|---|"]
    for aisles, bays, levels, bat in cases:
        wh = make_grid_warehouse(n_aisles=aisles, bays=bays, levels=levels)
        p = replace(ScanParams(), battery_s=bat, min_z=0.5)
        g = plan(wh, p, optimize=False)
        go = plan(wh, p, optimize=True)
        k, tt, n = solve_exact(wh, p)
        rows.append(f"| {aisles}×{bays}×{levels}, {bat} с | {n} | {len(g.sorties)} / {g.flight_s:.0f} | "
                    f"{len(go.sorties)} / {go.flight_s:.0f} | {k} / {tt:.0f} | {(g.flight_s - tt) / tt:+.1%} | "
                    f"{(go.flight_s - tt) / tt:+.1%} |")
        print(rows[-1], flush=True)
    table = "\n".join(rows)
    Path(__file__).with_name("results_table.md").write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Безопасность (SAFE-01, SAFE-02): пузырь безопасности по depth-кадру и геозона из раскладки склада.

Пузырь: точки depth-кадра фронтальной камеры (оптический кадр книги: z вперёд, x вправо, y вниз)
обратно проецируются (гл. 4, ф. 4.5) и отбираются в коридор движения шириной с дрон + запас.
Ближайшая точка в коридоре сравнивается с тормозным путём v²/(2a) + v·t_реакции + запас.

Геозона: коридоры проходов + магистральный проезд + колонна над доком. Маршрут планировщика
обязан целиком лежать в геозоне (проверяется тестом), а фактическая позиция вне её — повод
остановиться (SAFE-02). Модель данных — чистый Python; numpy нужен только для depth-кадра.
"""
from __future__ import annotations

from dataclasses import dataclass

from .camera import Camera
from .layout import Warehouse

GO, SLOW, STOP = "GO", "SLOW", "STOP"


def corridor_min_distance(depth, cam: Camera, half_width: float = 0.45, half_height: float = 0.35,
                          max_range: float = 8.0, stride: int = 2, cam_height: float | None = None,
                          floor_clearance: float = 0.1) -> float:
    """Ближайшая дистанция (м) до точки depth-кадра внутри коридора |x| <= half_width, |y| <= half_height.

    depth — массив HxW в метрах (numpy); 0, NaN и inf — нет данных. Нет точек в коридоре -> inf.
    cam_height — высота камеры над полом, м (камера без наклона). Тогда точки ниже floor_clearance над полом
    не препятствие: камера на доке стоит ~0.3 м над полом, и пол попадает в коридор высотой ±0.35 м уже
    с ~0.55 м — без этого STOP на доке и дрон не взлетит. Недооценка высоты безопасна: отбрасывается больше пола.
    """
    import numpy as np
    full = np.asarray(depth, dtype=np.float32)
    h0, w0 = full.shape
    d = full[::stride, ::stride]
    # пиксели прореженного кадра -> координаты кадра калибровки (depth может быть в другом разрешении)
    u = (np.arange(0, w0, stride) * (cam.width / w0))[None, :]
    v = (np.arange(0, h0, stride) * (cam.height / h0))[:, None]
    valid = np.isfinite(d) & (d > 0.05) & (d < max_range)
    d = np.where(valid, d, 0.0)          # inf·0 даёт NaN и предупреждения numpy
    x = (u - cam.cx) / cam.fx * d
    y = (v - cam.cy) / cam.fy * d        # оптический кадр: y вниз
    inside = valid & (np.abs(x) <= half_width) & (np.abs(y) <= half_height)
    if cam_height is not None:
        inside &= (cam_height - y) >= floor_clearance
    return float(d[inside].min()) if inside.any() else float("inf")


def stopping_distance(speed: float, decel: float = 1.0, reaction_s: float = 0.3, margin: float = 0.4) -> float:
    """Тормозной путь с реакцией и запасом, м."""
    v = abs(speed)
    return v * v / (2 * decel) + v * reaction_s + margin


@dataclass
class SafetyMonitor:
    """ЕХАТЬ / МЕДЛЕННО / СТОП с гистерезисом: снятие СТОП только после clear_frames чистых кадров."""
    slow_factor: float = 2.0      # МЕДЛЕННО, если препятствие ближе slow_factor × тормозного пути
    min_stop: float = 1.0         # не ближе 1 м даже на малой скорости (требование SAFE-01)
    clear_frames: int = 5
    state: str = GO
    _clear: int = 0

    def update(self, min_dist: float, speed: float) -> str:
        stop_d = max(self.min_stop, stopping_distance(speed))
        if min_dist < stop_d:
            self.state, self._clear = STOP, 0
        elif self.state == STOP:
            self._clear += 1
            if self._clear >= self.clear_frames:
                self.state = SLOW if min_dist < self.slow_factor * stop_d else GO
        else:
            self.state = SLOW if min_dist < self.slow_factor * stop_d else GO
        return self.state


@dataclass(frozen=True)
class Box:
    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float
    name: str

    def contains(self, p) -> bool:
        return self.x0 <= p[0] <= self.x1 and self.y0 <= p[1] <= self.y1 and self.z0 <= p[2] <= self.z1


class Geofence:
    """Разрешённый объём полёта: коридоры проходов, магистральный проезд, колонна над доком."""

    def __init__(self, wh: Warehouse, drone_radius: float = 0.35, clearance: float = 0.3, z_margin: float = 0.5):
        keep = drone_radius + clearance
        z_top = wh.ceiling_z - z_margin
        xc = wh.cross_aisle_x
        self.boxes: list[Box] = []
        ys = []
        for a in wh.aisles:
            rows = [wh.row(r) for r in a.rows]
            faces = sorted(r.face_y for r in rows)
            lo, hi = (faces[0], faces[-1]) if len(faces) > 1 else (a.y - 1.0, a.y + 1.0)
            x_end = max(r.end_x for r in rows)
            self.boxes.append(Box(xc - 0.5, x_end + 0.5, lo + keep, hi - keep, 0.0, z_top, f"проход {a.id}"))
            ys.append(a.y)
        ys.append(wh.dock[1])
        self.boxes.append(Box(xc - 0.6, xc + 0.6, min(ys) - 0.6, max(ys) + 0.6, 0.0, z_top, "магистральный проезд"))
        dx, dy = wh.dock[0], wh.dock[1]
        self.boxes.append(Box(dx - 0.6, dx + 0.6, dy - 0.6, dy + 0.6, 0.0, z_top, "док"))

    def contains(self, p) -> bool:
        return any(b.contains(p) for b in self.boxes)

    def check_route(self, waypoints) -> list[str]:
        """Точки маршрута (и середины отрезков) вне геозоны — список описаний; пусто = маршрут безопасен."""
        bad = []
        for a, b in zip(waypoints, waypoints[1:]):
            for k in (0.0, 0.5, 1.0):
                q = (a.x + (b.x - a.x) * k, a.y + (b.y - a.y) * k, a.z + (b.z - a.z) * k)
                if not self.contains(q):
                    bad.append(f"({q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f})")
        return bad

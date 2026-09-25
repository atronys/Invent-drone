"""Привязка прочитанной этикетки к адресу ячейки: пиксель -> точка фасада -> адрес.

Обратная проекция пинхол-камеры на плоскость фасада (гл. 4 книги Gao & Zhang).
Упрощение MVP: оптическая ось перпендикулярна фасаду (крен/тангаж ~0, камера смотрит
строго на ряд). Полная 6-DoF версия через позу из TF (R, t) — задача PER-06.
"""
from __future__ import annotations

from .camera import Camera
from .layout import RackRow


def pixel_to_face(u: float, v: float, cam: Camera, cam_pos: tuple[float, float, float], row: RackRow,
                  depth: float | None = None) -> tuple[float, float]:
    """Точка (x, z) на фасаде ряда, в которую проецируется пиксель (u, v).

    cam_pos — положение камеры в map; depth — измеренное расстояние до фасада
    (depth-камера/дальномер), иначе берётся геометрическое |y_cam - face_y|.
    """
    x_c, y_c, z_c = cam_pos
    if (y_c - row.face_y) * row.normal_y <= 0:
        raise ValueError(f"камера не со стороны прохода ряда {row.id}")
    d = depth if depth is not None else abs(y_c - row.face_y)
    s = -row.normal_y  # камера смотрит против нормали фасада; вектор «вправо» в кадре = (s, 0, 0)
    x = x_c + s * (u - cam.cx) / cam.fx * d
    z = z_c - (v - cam.cy) / cam.fy * d  # ось v изображения направлена вниз
    return x, z


def locate_label(u: float, v: float, cam: Camera, cam_pos: tuple[float, float, float], row: RackRow,
                 depth: float | None = None) -> str | None:
    """Адрес ячейки для этикетки с центром в пикселе (u, v); None — вне ряда."""
    return row.locate(*pixel_to_face(u, v, cam, cam_pos, row, depth))

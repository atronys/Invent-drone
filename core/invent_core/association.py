"""Привязка прочитанной этикетки к адресу ячейки: пиксель -> точка фасада -> адрес.

Обратная проекция пинхол-камеры (Gao & Zhang, гл. 4.1): пиксель -> (снятие дисторсии) ->
точка нормированной плоскости -> луч в кадре камеры -> луч в `map` через позу (R, t) ->
пересечение с плоскостью фасада y = face_y.

* `pixel_to_face_pose` — общий случай: поза камеры из TF в момент кадра (задача PER-06);
* `pixel_to_face` — упрощение MVP: камера смотрит строго на фасад (крен/тангаж ~0).
"""
from __future__ import annotations

from .camera import Camera
from .frames import Mat3, Vec3, apply, transpose
from .layout import RackRow, Warehouse


def pixel_to_face_pose(u: float, v: float, cam: Camera, r_map_cam: Mat3, t_map_cam: Vec3,
                       row: RackRow) -> tuple[float, float] | None:
    """Точка (x, z) фасада ряда, куда смотрит пиксель (u, v); None — луч не попадает на фасад."""
    if (t_map_cam[1] - row.face_y) * row.normal_y <= 0:
        return None  # камера не со стороны прохода этого ряда
    xn, yn = cam.pixel_to_normalized(u, v)
    ray = apply(r_map_cam, (xn, yn, 1.0))
    if abs(ray[1]) < 1e-12:
        return None
    s = (row.face_y - t_map_cam[1]) / ray[1]
    if s <= 0:
        return None  # фасад позади камеры
    return t_map_cam[0] + s * ray[0], t_map_cam[2] + s * ray[2]


def locate_label_pose(u: float, v: float, cam: Camera, r_map_cam: Mat3, t_map_cam: Vec3,
                      row: RackRow) -> str | None:
    """Адрес ячейки для этикетки с центром в пикселе (u, v) при известной позе камеры."""
    p = pixel_to_face_pose(u, v, cam, r_map_cam, t_map_cam, row)
    return row.locate(*p) if p else None


def locate_in_warehouse(u: float, v: float, cam: Camera, r_map_cam: Mat3, t_map_cam: Vec3,
                        wh: Warehouse, max_range: float = 4.0) -> str | None:
    """Адрес для пикселя без знания ряда: ближайший фасад, в который попадает луч."""
    best, best_d = None, max_range
    for row in wh.rows:
        d = abs(t_map_cam[1] - row.face_y)
        if d >= best_d:
            continue
        p = pixel_to_face_pose(u, v, cam, r_map_cam, t_map_cam, row)
        addr = row.locate(*p) if p else None
        if addr:
            best, best_d = addr, d
    return best


def project_to_image(p_map: Vec3, cam: Camera, r_map_cam: Mat3, t_map_cam: Vec3) -> tuple[float, float] | None:
    """Прямая проекция точки map в пиксель: P_c = R^T (P_w - t), затем K (ф. 4.8)."""
    rt = transpose(r_map_cam)
    return cam.project(apply(rt, (p_map[0] - t_map_cam[0], p_map[1] - t_map_cam[1], p_map[2] - t_map_cam[2])))


# --- упрощение MVP -----------------------------------------------------------------

def pixel_to_face(u: float, v: float, cam: Camera, cam_pos: Vec3, row: RackRow,
                  depth: float | None = None) -> tuple[float, float]:
    """Точка (x, z) на фасаде для камеры, смотрящей строго на фасад.

    depth — измеренное расстояние до фасада (depth-камера/дальномер), иначе |y_cam - face_y|.
    """
    x_c, y_c, z_c = cam_pos
    if (y_c - row.face_y) * row.normal_y <= 0:
        raise ValueError(f"камера не со стороны прохода ряда {row.id}")
    d = depth if depth is not None else abs(y_c - row.face_y)
    xn, yn = cam.pixel_to_normalized(u, v)
    s = -row.normal_y  # камера смотрит против нормали фасада; вектор «вправо» в кадре = (s, 0, 0)
    return x_c + s * xn * d, z_c - yn * d  # ось v изображения направлена вниз


def locate_label(u: float, v: float, cam: Camera, cam_pos: Vec3, row: RackRow,
                 depth: float | None = None) -> str | None:
    """Адрес ячейки для этикетки с центром в пикселе (u, v); None — вне ряда."""
    return row.locate(*pixel_to_face(u, v, cam, cam_pos, row, depth))

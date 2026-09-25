"""Системы координат и повороты (Gao & Zhang, гл. 2: матрицы поворота, углы Эйлера ZYX).

Кадры проекта (см. docs/02_architecture.md §4):
* `map`  — ENU склада: x вдоль рядов, y поперёк, z вверх;
* `body` — FLU дрона (ROS REP-103): x вперёд, y влево, z вверх;
* `cam`  — оптический кадр камеры как в книге: z по оси, x вправо, y вниз;
* PX4 local — NED с началом в точке инициализации EKF (док), тело PX4 — FRD.

Чистый Python (матрицы 3x3 — кортежи строк): ядро остаётся без зависимостей.
"""
from __future__ import annotations

import math

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]


def wrap(a: float) -> float:
    """Угол в (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


# --- ENU <-> NED (PX4) ----------------------------------------------------------

def enu_to_ned(v: Vec3) -> Vec3:
    """(E, N, U) -> (N, E, D). Преобразование — инволюция: оно же NED -> ENU."""
    return v[1], v[0], -v[2]


ned_to_enu = enu_to_ned


def yaw_enu_to_ned(yaw: float) -> float:
    """Курс ENU (от оси E против часовой) -> курс NED (от N по часовой): psi = pi/2 - yaw."""
    return wrap(math.pi / 2 - yaw)


yaw_ned_to_enu = yaw_enu_to_ned


def map_to_px4(p_map: Vec3, dock: Vec3) -> Vec3:
    """Точка map -> локальный NED PX4 (начало — док, где инициализирован EKF)."""
    return enu_to_ned((p_map[0] - dock[0], p_map[1] - dock[1], p_map[2] - dock[2]))


def px4_to_map(p_ned: Vec3, dock: Vec3) -> Vec3:
    e, n, u = ned_to_enu(p_ned)
    return e + dock[0], n + dock[1], u + dock[2]


# --- позы PX4 <-> map ---------------------------------------------------------------
# VehicleOdometry PX4: q — поворот тела FRD относительно NED (w, x, y, z). В map (ENU) тело — FLU (REP-103).
R_ENU_NED = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
R_FRD_FLU = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))   # обе матрицы симметричны и сами себе обратны


def px4_pose_to_map(pos_ned: Vec3, q_ned_frd: tuple[float, float, float, float], dock: Vec3):
    """Поза из /fmu/out/vehicle_odometry -> (p_map, q_map_flu (w, x, y, z)) для TF map->base_link."""
    r = matmul(R_ENU_NED, matmul(matrix_from_quat(*q_ned_frd), R_FRD_FLU))
    return px4_to_map(pos_ned, dock), quat_from_matrix(r)


def map_pose_to_px4(p_map: Vec3, q_map_flu: tuple[float, float, float, float], dock: Vec3):
    """Обратное: поза в map (например, от VIO) -> (pos_ned, q_ned_frd) для /fmu/in/vehicle_visual_odometry (LOC-04)."""
    r = matmul(R_ENU_NED, matmul(matrix_from_quat(*q_map_flu), R_FRD_FLU))
    return map_to_px4(p_map, dock), quat_from_matrix(r)


# --- матрицы поворота -------------------------------------------------------------

def matmul(a: Mat3, b: Mat3) -> Mat3:
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))  # type: ignore


def transpose(a: Mat3) -> Mat3:
    return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))  # type: ignore


def apply(r: Mat3, v: Vec3) -> Vec3:
    return tuple(sum(r[i][k] * v[k] for k in range(3)) for i in range(3))  # type: ignore


def rot_x(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return (1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)


def rot_y(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return (c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)


def rot_z(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return (c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)


def euler_zyx(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> Mat3:
    """R = Rz(yaw)·Ry(pitch)·Rx(roll) — порядок «рыскание-тангаж-крен» (гл. 2.3)."""
    return matmul(rot_z(yaw), matmul(rot_y(pitch), rot_x(roll)))


def quat_from_matrix(r: Mat3) -> tuple[float, float, float, float]:
    """Кватернион (w, x, y, z) из матрицы поворота (гл. 2.4). Нужен для VehicleOdometry PX4."""
    tr = r[0][0] + r[1][1] + r[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return 0.25 * s, (r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s
    i = max(range(3), key=lambda k: r[k][k])
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + r[i][i] - r[j][j] - r[k][k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (r[k][j] - r[j][k]) / s
    q[i + 1] = 0.25 * s
    q[j + 1] = (r[j][i] + r[i][j]) / s
    q[k + 1] = (r[k][i] + r[i][k]) / s
    return q[0], q[1], q[2], q[3]


def matrix_from_quat(w: float, x: float, y: float, z: float) -> Mat3:
    """Матрица поворота из единичного кватерниона (гл. 2.4.3, ф. 2.39). TF отдаёт порядок (x, y, z, w)."""
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


# Кадр звена камеры в Gazebo/ROS (x — по оси камеры, y влево, z вверх) -> оптический кадр книги
# (z по оси, x вправо, y вниз). Столбцы — оптические оси в кадре звена.
R_LINK_OPTICAL = ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))


# --- установка камер на дроне -----------------------------------------------------

def camera_mount(look: str, tilt_down_deg: float = 0.0) -> Mat3:
    """R_body_cam: столбцы — оси оптического кадра камеры в кадре тела FLU.

    look: 'left' (смотрит в +y тела), 'right' (-y), 'front' (+x). tilt_down_deg — наклон вниз
    (например, для камеры нижнего яруса, задача MIS-04).
    """
    fwd = {"left": (0.0, 1.0, 0.0), "right": (0.0, -1.0, 0.0), "front": (1.0, 0.0, 0.0)}[look]
    t = math.radians(tilt_down_deg)
    z = (fwd[0] * math.cos(t), fwd[1] * math.cos(t), -math.sin(t))   # оптическая ось
    y = (fwd[0] * -math.sin(t), fwd[1] * -math.sin(t), -math.cos(t))  # «вниз» кадра
    x = (y[1] * z[2] - y[2] * z[1], y[2] * z[0] - y[0] * z[2], y[0] * z[1] - y[1] * z[0])  # x = y × z
    return (x[0], y[0], z[0]), (x[1], y[1], z[1]), (x[2], y[2], z[2])


def camera_pose(p_body: Vec3, yaw: float, look: str, pitch: float = 0.0, roll: float = 0.0,
                t_body_cam: Vec3 = (0.0, 0.0, 0.0), tilt_down_deg: float = 0.0) -> tuple[Mat3, Vec3]:
    """Поза камеры в map: (R_map_cam, t_map_cam) по позе дрона и установке камеры."""
    r_map_body = euler_zyx(yaw, pitch, roll)
    r = matmul(r_map_body, camera_mount(look, tilt_down_deg))
    off = apply(r_map_body, t_body_cam)
    return r, (p_body[0] + off[0], p_body[1] + off[1], p_body[2] + off[2])

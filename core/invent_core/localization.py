"""LOC-11: EKF-локализация по адресным этикеткам — рабочий модуль (прототип и эксперимент: research/landmark_ekf).

Состояние x = (x, y, z, ψ) в map; крен и тангаж известны от PX4/IMU и входят как параметры.
* Прогноз — ф. (8.32) Gao & Zhang: приращения VIO в кадре тела.
* Наблюдение — **пиксель центра адресной этикетки** с известным положением L в map (ф. 4.8):
  P' = R_bc^T (R_tilt^T Rz(ψ)^T (L - p) - t_bc),  u = fx X'/Z' + cx,  v = fy Y'/Z' + cy.
  PnP по углам не нужна: центр штрихкода zxing даёт всегда, а ориентация малой цели по PnP неточна
  (core/tests/test_pnp.py).
* Якобиан — аналитический (∂(u,v)/∂P' — ф. 6.43 гл. 6.7.3) и проверяется численным в тестах.
* Обновление — ф. (8.34)–(8.35) с χ²-гейтингом (замечание 8.1.4, п. 4).

numpy обязателен для этого модуля (остальное ядро без зависимостей).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .camera import Camera
from .frames import R_LINK_OPTICAL, camera_mount, euler_zyx
from .layout import Warehouse

CHI2_2DOF_99 = 9.21


def _rz(psi: float) -> np.ndarray:
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _tilt(roll: float, pitch: float) -> np.ndarray:
    return np.array(euler_zyx(0.0, pitch, roll))


@dataclass
class MountedCamera:
    cam: Camera
    r_bc: np.ndarray          # оптический кадр камеры в кадре тела FLU
    t_bc: np.ndarray          # положение камеры в кадре тела, м

    @classmethod
    def side(cls, cam: Camera, look: str, t_bc=(0.0, 0.0, 0.12), tilt_down_deg: float = 0.0) -> "MountedCamera":
        return cls(cam, np.array(camera_mount(look, tilt_down_deg)), np.array(t_bc, dtype=float))

    @classmethod
    def from_mount(cls, cam: Camera, xyz, rpy) -> "MountedCamera":
        """Из mounts.json (sim/make_drone_model.py): звено камеры Gazebo/ROS -> оптический кадр."""
        roll, pitch, yaw = rpy
        r = np.array(euler_zyx(yaw, pitch, roll)) @ np.array(R_LINK_OPTICAL)
        return cls(cam, r, np.array(xyz, dtype=float))


def observe(state: np.ndarray, landmark: np.ndarray, mc: MountedCamera, roll: float = 0.0, pitch: float = 0.0):
    """-> (uv (2,), J (2x4)) или (None, None), если ориентир позади камеры. Без дисторсии: пиксели на входе
    должны быть исправлены (Camera.pixel_to_normalized -> normalized_to_pixel без дисторсии)."""
    p, psi = state[:3], state[3]
    rz, rt = _rz(psi), _tilt(roll, pitch)
    d = landmark - p
    q = rt.T @ rz.T @ d
    pc = mc.r_bc.T @ (q - mc.t_bc)
    x, y, z = pc
    if z <= 0.05:
        return None, None
    fx, fy = mc.cam.fx, mc.cam.fy
    uv = np.array([fx * x / z + mc.cam.cx, fy * y / z + mc.cam.cy])
    d_uv_dpc = np.array([[fx / z, 0.0, -fx * x / z ** 2], [0.0, fy / z, -fy * y / z ** 2]])   # ф. 6.43
    c, s = math.cos(psi), math.sin(psi)
    drzT = np.array([[-s, c, 0.0], [-c, -s, 0.0], [0.0, 0.0, 0.0]])                          # d(Rz^T)/dψ
    d_pc_dp = -mc.r_bc.T @ rt.T @ rz.T
    d_pc_dpsi = mc.r_bc.T @ rt.T @ drzT @ d
    return uv, d_uv_dpc @ np.hstack([d_pc_dp, d_pc_dpsi[:, None]])


class LandmarkLocalizer:
    """EKF: прогноз по VIO, обновление по пикселям адресных этикеток с известными координатами."""

    def __init__(self, mu0, sigma0=(0.02, 0.02, 0.02, math.radians(1.0)), gate: float | None = CHI2_2DOF_99,
                 inflate_after: int | None = None, inflate_factor: float = 4.0):
        """inflate_after (LOC-12): после стольких отказов подряд раздуть P в inflate_factor раз — фильтр
        признаёт, что «потерялся» (редкие ориентиры + плохой VIO). По умолчанию выключено: на паллетных
        этикетках это пропускает постоянные выбросы (research/landmark_ekf/README.md, LOC-12)."""
        self.mu = np.array(mu0, dtype=float)
        self.P = np.diag(np.square(sigma0))
        self.gate = gate
        self.inflate_after, self.inflate_factor = inflate_after, inflate_factor
        self.accepted = self.rejected = self.streak = self.inflations = 0

    def predict(self, d_body_xy, dz: float, dpsi: float, sigma_xy: float, sigma_z: float, sigma_psi: float) -> None:
        """Ф. (8.32): приращение из VIO в кадре тела (x вперёд, y влево) + вертикаль + курс."""
        x, y, z, psi = self.mu
        c, s = math.cos(psi), math.sin(psi)
        dx, dy = d_body_xy
        self.mu = np.array([x + c * dx - s * dy, y + s * dx + c * dy, z + dz, math.atan2(math.sin(psi + dpsi),
                                                                                         math.cos(psi + dpsi))])
        F = np.eye(4)
        F[0, 3], F[1, 3] = -s * dx - c * dy, c * dx - s * dy
        G = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        R = np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_z ** 2, sigma_psi ** 2])
        self.P = F @ self.P @ F.T + G @ R @ G.T

    def update_pixel(self, uv, landmark, mc: MountedCamera, sigma_px: float = 2.0,
                     roll: float = 0.0, pitch: float = 0.0) -> bool:
        """Ф. (8.34)–(8.35) с χ²-гейтингом. False — отброшено (или ориентир позади камеры)."""
        h, H = observe(self.mu, np.asarray(landmark, dtype=float), mc, roll, pitch)
        if h is None:
            return False
        nu = np.asarray(uv, dtype=float) - h
        S = H @ self.P @ H.T + np.eye(2) * sigma_px ** 2
        if self.gate is not None and float(nu @ np.linalg.solve(S, nu)) > self.gate:
            self.rejected += 1
            self.streak += 1
            if self.inflate_after and self.streak >= self.inflate_after:
                self.P = self.P * self.inflate_factor
                self.streak = 0
                self.inflations += 1
            return False
        self.streak = 0
        K = self.P @ H.T @ np.linalg.inv(S)
        self.mu = self.mu + K @ nu
        self.mu[3] = math.atan2(math.sin(self.mu[3]), math.cos(self.mu[3]))
        self.P = (np.eye(4) - K @ H) @ self.P
        self.accepted += 1
        return True


def address_landmarks(wh: Warehouse) -> dict[str, np.ndarray]:
    """Карта ориентиров: центр адресной этикетки каждой ячейки (как в sim/generate_world.py)."""
    out = {}
    for loc in wh.locations():
        row, x, z = wh.address_label_position(loc.address)
        out[loc.address] = np.array([x, row.face_y + row.normal_y * 0.004, z])
    return out

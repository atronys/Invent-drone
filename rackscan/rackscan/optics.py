"""Модель камеры и расчёты оптики: с какого расстояния читается код и когда ячейка влезает в кадр.

Пинхол-модель (Gao & Zhang, «Introduction to Visual SLAM», ф. 4.5):  u = fx·X/Z + cx.
Отсюда масштаб на расстоянии d: px/мм = fx / (1000·d). Те же формулы, что в черновике гл. 2
старого проекта («Утверждение 1»), но для неподвижной камеры стенда.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Минимум пикселей на модуль (самый узкий элемент кода). ~2 — предел декодера,
# 3 — проектная цель с запасом на расфокус и шум.
MIN_PPM = 3.0


@dataclass(frozen=True)
class CameraModel:
    """Внутренние параметры камеры для конкретного разрешения."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple[float, ...] = field(default_factory=tuple)  # k1 k2 p1 p2 k3 (модель OpenCV)

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float, vfov_deg: float | None = None) -> "CameraModel":
        fx = width / (2 * math.tan(math.radians(hfov_deg) / 2))
        fy = height / (2 * math.tan(math.radians(vfov_deg) / 2)) if vfov_deg else fx
        return cls(width, height, fx, fy, width / 2, height / 2)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    @property
    def D(self) -> np.ndarray:
        return np.array(self.dist if self.dist else (0.0,) * 5, dtype=np.float64)

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.width / (2 * self.fx)))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.height / (2 * self.fy)))

    def scaled(self, width: int, height: int) -> "CameraModel":
        """Та же камера в другом разрешении при том же поле зрения (режим без кропа)."""
        sx, sy = width / self.width, height / self.height
        return CameraModel(width, height, self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, self.dist)

    def px_per_mm(self, distance_m: float) -> float:
        return self.fx / (distance_m * 1000.0)

    def footprint(self, distance_m: float) -> tuple[float, float]:
        """Размер кадра на плоскости фасада, м (ширина, высота)."""
        return self.width / self.fx * distance_m, self.height / self.fy * distance_m


# Паспортные поля зрения. IMX708 — Raspberry Pi Camera Module 3; IMX477 — HQ Camera
# (сенсор 6.287 × 4.712 мм, поле зрения считается от фокусного расстояния объектива).
def _imx477(focal_mm: float) -> tuple[float, float]:
    return (math.degrees(2 * math.atan(6.287 / (2 * focal_mm))),
            math.degrees(2 * math.atan(4.712 / (2 * focal_mm))))


CAMERA_PRESETS: dict[str, dict] = {
    "imx708": {"title": "Camera Module 3 (IMX708, 66°)", "full": (4608, 2592), "hfov": 66.0, "vfov": 41.0},
    "imx708_wide": {"title": "Camera Module 3 Wide (IMX708, 102°)", "full": (4608, 2592), "hfov": 102.0, "vfov": 67.0},
    "imx477_6mm": {"title": "HQ Camera (IMX477) + 6 мм", "full": (4056, 3040),
                   "hfov": _imx477(6)[0], "vfov": _imx477(6)[1]},
    "imx477_16mm": {"title": "HQ Camera (IMX477) + 16 мм", "full": (4056, 3040),
                    "hfov": _imx477(16)[0], "vfov": _imx477(16)[1]},
    "webcam_720p": {"title": "USB-камера 1280×720 (≈60°)", "full": (1280, 720), "hfov": 60.0, "vfov": 36.0},
}


def camera_from_preset(name: str, size: tuple[int, int] | None = None) -> CameraModel:
    p = CAMERA_PRESETS[name]
    w, h = size or p["full"]
    return CameraModel.from_fov(w, h, p["hfov"], p["vfov"])


# --- читаемость кода ------------------------------------------------------------

def px_per_module(cam: CameraModel, distance_m: float, module_mm: float) -> float:
    return cam.px_per_mm(distance_m) * module_mm


def max_read_distance(cam: CameraModel, module_mm: float, min_ppm: float = MIN_PPM) -> float:
    """Наибольшая дистанция (м), на которой модуль module_mm занимает min_ppm пикселей:
    fx·X / (1000·d) >= p_min  =>  d_max = fx·X / (1000·p_min)."""
    return cam.fx * module_mm / (1000.0 * min_ppm)


def min_fit_distance(cam: CameraModel, width_m: float, height_m: float, margin: float = 0.15) -> float:
    """Наименьшая дистанция (м), с которой объект width×height целиком в кадре с полями margin
    (доля кадра с каждой стороны)."""
    k = 1.0 / (1.0 - 2 * margin)
    return max(width_m * k * cam.fx / cam.width, height_m * k * cam.fy / cam.height)


def motion_blur_px(cam: CameraModel, distance_m: float, speed_mps: float, exposure_s: float) -> float:
    """Смаз (px) при движении вдоль фасада: b = fx·v·τ/d."""
    return cam.fx * speed_mps * exposure_s / distance_m


def max_exposure_s(speed_mps: float, module_mm: float, max_blur_modules: float = 0.5) -> float:
    """Наибольшая выдержка, при которой смаз не больше доли модуля: β = v·τ/X <= β_max.
    Не зависит ни от камеры, ни от расстояния."""
    if speed_mps <= 0:
        return float("inf")
    return max_blur_modules * module_mm / 1000.0 / speed_mps


def marker_max_distance(cam: CameraModel, marker_mm: float, bits: int, min_ppm: float = 4.0) -> float:
    """Дальность обнаружения ArUco: модуль = S/(n+2) (n бит + рамка по модулю с каждой стороны)."""
    return max_read_distance(cam, marker_mm / (bits + 2), min_ppm)


@dataclass(frozen=True)
class DistancePlan:
    """Рабочие дистанции стенда. Все значения — метры от плоскости фасада."""

    d_cell: float           # обзор ячейки: ячейка + адресная метка в кадре предпросмотра
    d_cell_min: float       # ближе нельзя — ячейка не влезает
    d_label: float          # чтение этикетки на полном кадре
    d_label_max: float      # дальше нельзя — модуль меньше min_ppm
    ppm_preview_at_cell: float
    ppm_still_at_label: float
    feasible_single_step: bool   # читается ли код уже с дистанции обзора ячейки (на полном кадре)


def plan_distances(preview: CameraModel, still: CameraModel, cell_w_m: float, cell_h_m: float,
                   module_mm: float, min_ppm: float = MIN_PPM, margin: float = 0.15,
                   d_near: float = 0.25, d_far: float = 3.0, safety: float = 0.8) -> DistancePlan:
    """Выбор двух дистанций: «обзор ячейки» и «чтение этикетки».

    Обзор: ячейка с полями должна поместиться в кадр — это нижняя граница d_cell.
    Чтение: модуль кода должен занимать >= min_ppm пикселей полного кадра — это верхняя
    граница; берём её с запасом safety (0.8 → на 20% ближе предела).
    """
    d_cell_min = min_fit_distance(preview, cell_w_m, cell_h_m, margin)
    d_cell = min(max(d_cell_min, d_near), d_far)
    d_label_max = max_read_distance(still, module_mm, min_ppm)
    d_label = min(max(d_label_max * safety, d_near), d_cell)
    return DistancePlan(
        d_cell=round(d_cell, 3), d_cell_min=round(d_cell_min, 3),
        d_label=round(d_label, 3), d_label_max=round(d_label_max, 3),
        ppm_preview_at_cell=px_per_module(preview, d_cell, module_mm),
        ppm_still_at_label=px_per_module(still, d_label, module_mm),
        feasible_single_step=d_label_max >= d_cell,
    )

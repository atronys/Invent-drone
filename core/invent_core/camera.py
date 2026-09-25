"""Модель камеры-пинхол с дисторсией rad-tan (Gao & Zhang, гл. 4.1, ф. 4.5–4.12).

Кадр камеры как в книге (гл. 4.1.1): z — вперёд по оптической оси, x — вправо, y — вниз.
Цепочка: точка в кадре камеры -> нормированная плоскость z=1 (ф. 4.9) -> дисторсия (ф. 4.12)
-> пиксели через K (ф. 4.6).

Две формы задания:
* по углам обзора (hfov/vfov) — для прикидок оптики до покупки камеры;
* по калибровке (fx, fy, cx, cy + k1, k2, p1, p2, k3) — `Camera.load("camera.yaml")` читает
  формат ROS camera_info (его пишут `tools/calibrate_camera.py` и ROS camera_calibration).
Порядок коэффициентов дисторсии — как в OpenCV/ROS: k1, k2, p1, p2, k3 (в книге перечислены иначе).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Camera:
    width: int = 4056          # px, по умолчанию 12 Мп (IMX477-класс)
    height: int = 3040
    hfov_deg: float = 50.0     # горизонтальный угол обзора
    vfov_deg: float = 39.0     # вертикальный угол обзора
    k: tuple[float, float, float, float] | None = None       # fx, fy, cx, cy (после калибровки)
    dist: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)  # k1, k2, p1, p2, k3

    # --- внутренние параметры (ф. 4.5) -------------------------------------
    @property
    def fx(self) -> float:
        return self.k[0] if self.k else (self.width / 2) / math.tan(math.radians(self.hfov_deg) / 2)

    @property
    def fy(self) -> float:
        return self.k[1] if self.k else (self.height / 2) / math.tan(math.radians(self.vfov_deg) / 2)

    @property
    def cx(self) -> float:
        return self.k[2] if self.k else self.width / 2

    @property
    def cy(self) -> float:
        return self.k[3] if self.k else self.height / 2

    @classmethod
    def from_intrinsics(cls, width: int, height: int, fx: float, fy: float, cx: float, cy: float,
                        dist: tuple[float, ...] = (0.0,) * 5) -> "Camera":
        d = tuple(float(v) for v in (list(dist) + [0.0] * 5)[:5])
        return cls(width, height,
                   hfov_deg=math.degrees(2 * math.atan(width / (2 * fx))),
                   vfov_deg=math.degrees(2 * math.atan(height / (2 * fy))),
                   k=(fx, fy, cx, cy), dist=d)

    @classmethod
    def load(cls, path: str | Path) -> "Camera":
        """Калибровка в формате ROS camera_info (YAML) или JSON с теми же ключами."""
        text = Path(path).read_text(encoding="utf-8")
        if str(path).endswith(".json"):
            d = json.loads(text)
            km, dist = d["camera_matrix"]["data"], d["distortion_coefficients"]["data"]
            w, h = d["image_width"], d["image_height"]
        else:  # минимальный разбор фиксированного формата camera_info — без зависимости от pyyaml
            def grab(key: str) -> list[float]:
                m = re.search(key + r":.*?data:\s*\[([^\]]*)\]", text, re.S)
                if not m:
                    raise ValueError(f"{path}: нет {key}.data")
                return [float(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]
            km, dist = grab("camera_matrix"), grab("distortion_coefficients")
            w = int(re.search(r"image_width:\s*(\d+)", text).group(1))
            h = int(re.search(r"image_height:\s*(\d+)", text).group(1))
        return cls.from_intrinsics(w, h, km[0], km[4], km[2], km[5], tuple(dist))

    # --- дисторсия (ф. 4.10–4.12) ------------------------------------------
    def distort(self, x: float, y: float) -> tuple[float, float]:
        """Нормированные неискажённые координаты -> искажённые."""
        k1, k2, p1, p2, k3 = self.dist
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        return (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)

    def undistort(self, xd: float, yd: float, iters: int = 30) -> tuple[float, float]:
        """Обратная задача к distort: итерация неподвижной точки (так делает и OpenCV)."""
        if not any(self.dist):
            return xd, yd
        k1, k2, p1, p2, k3 = self.dist
        x, y = xd, yd
        for _ in range(iters):
            r2 = x * x + y * y
            radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            x, y = (xd - dx) / radial, (yd - dy) / radial
        return x, y

    # --- проекция -----------------------------------------------------------
    def pixel_to_normalized(self, u: float, v: float) -> tuple[float, float]:
        """Пиксель -> неискажённая точка нормированной плоскости z=1."""
        return self.undistort((u - self.cx) / self.fx, (v - self.cy) / self.fy)

    def normalized_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        xd, yd = self.distort(x, y)
        return self.fx * xd + self.cx, self.fy * yd + self.cy

    def project(self, p_cam: tuple[float, float, float]) -> tuple[float, float] | None:
        """Точка в кадре камеры -> пиксель; None, если точка позади камеры (сноска 4 гл. 4)."""
        x, y, z = p_cam
        if z <= 1e-9:
            return None
        return self.normalized_to_pixel(x / z, y / z)

    def in_image(self, uv: tuple[float, float] | None) -> bool:
        return uv is not None and 0 <= uv[0] < self.width and 0 <= uv[1] < self.height

    # --- прикидки для оптики этикеток --------------------------------------
    def footprint(self, distance: float) -> tuple[float, float]:
        """Ширина и высота (м) области плоскости, видимой на расстоянии distance (м)."""
        return self.width / self.fx * distance, self.height / self.fy * distance

    def px_per_mm(self, distance: float) -> float:
        """Разрешение на плоскости фасада, пикселей на миллиметр."""
        return self.fx / (distance * 1000)

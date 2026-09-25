"""Модель камеры-пинхол (гл. 4 книги Gao & Zhang, «Introduction to Visual SLAM»).

Пока без дисторсии и с фокусными расстояниями из углов обзора. После калибровки
(задача PER-02) заменить fx, fy, cx, cy реальными значениями.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Camera:
    width: int = 4056          # px, по умолчанию 12 Мп (IMX477-класс)
    height: int = 3040
    hfov_deg: float = 50.0     # горизонтальный угол обзора
    vfov_deg: float = 39.0     # вертикальный угол обзора

    @property
    def fx(self) -> float:
        return (self.width / 2) / math.tan(math.radians(self.hfov_deg) / 2)

    @property
    def fy(self) -> float:
        return (self.height / 2) / math.tan(math.radians(self.vfov_deg) / 2)

    @property
    def cx(self) -> float:
        return self.width / 2

    @property
    def cy(self) -> float:
        return self.height / 2

    def footprint(self, distance: float) -> tuple[float, float]:
        """Ширина и высота (м) области плоскости, видимой на расстоянии distance (м)."""
        return (
            2 * distance * math.tan(math.radians(self.hfov_deg) / 2),
            2 * distance * math.tan(math.radians(self.vfov_deg) / 2),
        )

    def px_per_mm(self, distance: float) -> float:
        """Разрешение на плоскости фасада, пикселей на миллиметр."""
        return self.width / (self.footprint(distance)[0] * 1000)

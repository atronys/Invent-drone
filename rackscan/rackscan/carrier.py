"""Носитель камеры: шлюз безопасности между автоматом сканирования и автопилотом дрона.

Это скелет будущего бортового модуля. Автомат сканирования выдаёт желаемую скорость в системе
носителя (guidance.velocity_command); шлюз решает, можно ли её исполнять, и урезает опасные
составляющие. Правило одно: при любом сомнении в положении — не приближаться к стеллажу.

Шлюз — чистая функция от снимка состояния, поэтому все его запреты проверяются тестами на ПК
(tests/test_carrier.py) задолго до первого полёта. Передача уставки автопилоту (PX4, режим
Offboard, скорости в системе корпуса) сюда намеренно не включена: она появится вместе с дроном.
"""
from __future__ import annotations

from dataclasses import dataclass

MOVING_PHASES = ("NAVIGATE", "APPROACH")


@dataclass(frozen=True)
class CarrierLimits:
    v_max: float = 0.30              # м/с в плоскости
    vz_max: float = 0.20             # м/с по высоте
    yaw_rate_max: float = 20.0       # °/с; быстрый разворот смазывает кадр и срывает оценку позы
    min_distance_m: float = 0.45     # ближе к фасаду не подходить (радиус винтов + запас)
    max_distance_m: float = 2.50     # дальше метки не видны — вперёд можно, назад нельзя
    max_pose_age_s: float = 0.30     # поза старше — удержание
    max_yaw_deg: float = 35.0        # при большем курсе сначала доворот, потом движение
    min_markers_forward: int = 2     # по одной метке курс ненадёжен: вперёд не летим
    tof_margin_m: float = 0.25       # дальномер видит препятствие ближе фасада на эту величину


@dataclass(frozen=True)
class Setpoint:
    vx_forward: float = 0.0
    vy_right: float = 0.0
    vz_up: float = 0.0
    yaw_rate_dps: float = 0.0
    hold: bool = True
    reason: str = ""

    def to_dict(self) -> dict:
        return {"vx_forward": round(self.vx_forward, 3), "vy_right": round(self.vy_right, 3),
                "vz_up": round(self.vz_up, 3), "yaw_rate_dps": round(self.yaw_rate_dps, 2),
                "hold": self.hold, "reason": self.reason}


def _sat(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


def gate(snap: dict, pose_age_s: float, limits: CarrierLimits | None = None) -> Setpoint:
    """Уставка, которую разрешено передать автопилоту, по снимку состояния автомата.

    snap — результат Scanner.snapshot(); pose_age_s — сколько секунд назад получена поза.
    """
    lim = limits or CarrierLimits()
    phase = snap.get("phase")
    if phase not in MOVING_PHASES:
        return Setpoint(reason=f"этап {phase}: удержание")
    pose = snap.get("pose")
    vel = snap.get("velocity") or {}
    if pose is None or pose_age_s > lim.max_pose_age_s:
        return Setpoint(reason="нет свежей позы: удержание")
    if not vel:
        return Setpoint(reason="нет уставки: удержание")

    vx = _sat(vel.get("vx_forward", 0.0), lim.v_max)
    vy = _sat(vel.get("vy_right", 0.0), lim.v_max)
    vz = _sat(vel.get("vz_up", 0.0), lim.vz_max)
    wz = _sat(vel.get("yaw_rate_dps", 0.0), lim.yaw_rate_max)
    notes = []

    if abs(pose["yaw_deg"]) > lim.max_yaw_deg:
        vx = vy = vz = 0.0
        notes.append("курс велик: только доворот")
    if pose["distance_m"] <= lim.min_distance_m and vx > 0:
        vx = 0.0
        notes.append("минимальная дистанция: вперёд запрещено")
    if pose["distance_m"] >= lim.max_distance_m and vx < 0:
        vx = 0.0
        notes.append("предел дальности: назад запрещено")
    if (pose.get("ambiguous") or pose.get("markers", 0) < lim.min_markers_forward) and vx > 0:
        vx = 0.0
        vy *= 0.5
        notes.append("поза по одной метке: вперёд запрещено")
    tof = (snap.get("sensors") or {}).get("tof_mm")
    if tof is not None and tof / 1000.0 < pose["distance_m"] - lim.tof_margin_m and vx > 0:
        vx = 0.0
        notes.append("дальномер видит препятствие перед фасадом: вперёд запрещено")

    moving = any(abs(v) > 1e-3 for v in (vx, vy, vz)) or abs(wz) > 0.1
    return Setpoint(vx, vy, vz, wz, hold=not moving, reason="; ".join(notes))

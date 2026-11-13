"""Наведение: из ошибки позы — команды оператору сегодня и уставка дрону завтра.

Целевая поза: камера напротив точки (X*, Y*) фасада, на дистанции d*, оптическая ось по
нормали к фасаду. Ошибки считаются в системе фасада:
    ex = X* − x   (> 0 — сместиться вправо)        ey = Y* − y   (> 0 — вверх)
    ez = z − d*   (> 0 — подойти ближе)            eψ = ψ        (> 0 — повернуть влево)

Для дрона та же ошибка переводится в систему корпуса (вперёд — к фасаду) и подаётся в
пропорциональный регулятор с насыщением (позиционное визуальное сервоуправление, разд. 8.3
документа «Изменение вектора развития»):  v = sat(Kp·e, v_max).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .pose import FacadePose

RED, YELLOW, GREEN, IDLE = "RED", "YELLOW", "GREEN", "IDLE"


@dataclass(frozen=True)
class Tolerances:
    xy_m: float = 0.04
    z_m: float = 0.10
    yaw_deg: float = 6.0
    pitch_deg: float = 8.0
    roll_deg: float = 6.0

    def scaled(self, k: float) -> "Tolerances":
        return Tolerances(self.xy_m * k, self.z_m * k, self.yaw_deg * k, self.pitch_deg * k, self.roll_deg * k)


@dataclass
class Command:
    axis: str            # x | y | z | yaw | pitch | roll
    text: str            # «ВПРАВО», «БЛИЖЕ», …
    value: float         # величина в единицах unit
    unit: str            # «см» | «°»
    ratio: float         # ошибка / допуск: > 1 — вне допуска

    def phrase(self) -> str:
        return f"{self.text} на {self.value:.0f} {self.unit}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["value"] = round(self.value, 1)
        d["ratio"] = round(self.ratio, 2)
        d["phrase"] = self.phrase()
        return d


@dataclass
class Guidance:
    status: str = RED
    headline: str = ""
    hint: str = ""
    commands: list[Command] = field(default_factory=list)
    aligned: bool = False
    errors: dict = field(default_factory=dict)        # ex, ey, ez (м), yaw, pitch, roll (°)
    body: dict = field(default_factory=dict)          # смещение в системе носителя: forward, right, up (м), yaw (°)

    def to_dict(self) -> dict:
        return {
            "status": self.status, "headline": self.headline, "hint": self.hint, "aligned": self.aligned,
            "commands": [c.to_dict() for c in self.commands],
            "errors": {k: round(v, 3) for k, v in self.errors.items()},
            "body": {k: round(v, 3) for k, v in self.body.items()},
        }


def body_setpoint(ex: float, ey: float, ez: float, yaw_deg: float) -> dict:
    """Требуемое смещение в системе носителя (вперёд — вдоль оптической оси, к фасаду)."""
    psi = math.radians(yaw_deg)
    return {
        "forward_m": ex * math.sin(psi) + ez * math.cos(psi),
        "right_m": ex * math.cos(psi) - ez * math.sin(psi),
        "up_m": ey,
        "yaw_deg": -yaw_deg,     # > 0 — поворот вправо (по часовой, вид сверху)
    }


def velocity_command(body: dict, kp: float = 0.8, v_max: float = 0.3, kp_yaw: float = 1.0,
                     yaw_rate_max: float = 20.0) -> dict:
    """Уставка скорости для дрона: v = sat(Kp·e, v_max). На стенде не исполняется, только
    показывается и пишется в журнал — чтобы закон управления был проверен до первого полёта."""
    def sat(v, lim):
        return max(-lim, min(lim, v))
    return {
        "vx_forward": sat(kp * body["forward_m"], v_max),
        "vy_right": sat(kp * body["right_m"], v_max),
        "vz_up": sat(kp * body["up_m"], v_max),
        "yaw_rate_dps": sat(kp_yaw * body["yaw_deg"], yaw_rate_max),
    }


def compute(pose: FacadePose, target_xy_m: tuple[float, float], target_dist_m: float,
            tol: Tolerances, check_attitude: bool = True) -> Guidance:
    """Команды для выхода в целевую позу."""
    x, y, z = (float(v) for v in pose.position)
    ex, ey, ez = target_xy_m[0] - x, target_xy_m[1] - y, z - target_dist_m
    yaw, pitch, roll = pose.yaw_deg, pose.pitch_deg, pose.roll_deg
    g = Guidance(errors={"ex": ex, "ey": ey, "ez": ez, "yaw": yaw, "pitch": pitch, "roll": roll},
                 body=body_setpoint(ex, ey, ez, yaw))
    cmds: list[Command] = []

    def add(axis, err, tolv, pos_text, neg_text, unit, k=1.0):
        if abs(err) > tolv:
            cmds.append(Command(axis, pos_text if err > 0 else neg_text, abs(err) * k, unit, abs(err) / tolv))

    add("yaw", yaw, tol.yaw_deg, "ПОВЕРНИТЕ ВЛЕВО", "ПОВЕРНИТЕ ВПРАВО", "°")
    add("x", ex, tol.xy_m, "ВПРАВО", "ВЛЕВО", "см", 100)
    add("y", ey, tol.xy_m, "ВВЕРХ", "ВНИЗ", "см", 100)
    add("z", ez, tol.z_m, "БЛИЖЕ", "ДАЛЬШЕ", "см", 100)
    if check_attitude:
        add("pitch", pitch, tol.pitch_deg, "НАКЛОНИТЕ ВНИЗ", "НАКЛОНИТЕ ВВЕРХ", "°")
        add("roll", roll, tol.roll_deg, "КРЕН: ПО ЧАСОВОЙ", "КРЕН: ПРОТИВ ЧАСОВОЙ", "°")
    cmds.sort(key=lambda c: -c.ratio)
    g.commands = cmds
    g.aligned = not cmds
    if g.aligned:
        g.status, g.headline = YELLOW, "ДЕРЖИТЕ РОВНО"
    else:
        g.status = RED
        g.headline = cmds[0].phrase()
        g.hint = "; ".join(c.phrase() for c in cmds[1:3])
    return g


def from_label_only(u: float, v: float, side_px: float, cam, code_mm: float, target_dist_m: float,
                    tol: Tolerances, fallback_dist_m: float | None = None) -> Guidance:
    """Запасное наведение, когда адресных меток в кадре нет (камера слишком близко), но
    этикетка видна. Дистанция — по видимому размеру символа: d ≈ fx·S / s."""
    d = cam.fx * (code_mm / 1000.0) / side_px if side_px > 1 else (fallback_dist_m or target_dist_m)
    ex = (u - cam.cx) / cam.fx * d
    ey = -(v - cam.cy) / cam.fy * d
    ez = d - target_dist_m
    g = Guidance(errors={"ex": ex, "ey": ey, "ez": ez, "yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                 body=body_setpoint(ex, ey, ez, 0.0))
    cmds = []
    for axis, err, tolv, pos, neg in (("x", ex, tol.xy_m, "ВПРАВО", "ВЛЕВО"), ("y", ey, tol.xy_m, "ВВЕРХ", "ВНИЗ"),
                                      ("z", ez, tol.z_m, "БЛИЖЕ", "ДАЛЬШЕ")):
        if abs(err) > tolv:
            cmds.append(Command(axis, pos if err > 0 else neg, abs(err) * 100, "см", abs(err) / tolv))
    cmds.sort(key=lambda c: -c.ratio)
    g.commands, g.aligned = cmds, not cmds
    g.status = YELLOW if g.aligned else RED
    g.headline = "ДЕРЖИТЕ РОВНО" if g.aligned else cmds[0].phrase()
    g.hint = "курс не контролируется: адресные метки вне кадра"
    return g


def search(message: str, hint: str = "") -> Guidance:
    return Guidance(status=RED, headline=message, hint=hint)

"""Заглушки rclpy / px4_msgs / invent_msgs для проверки логики узлов без ROS 2 (на Windows и в CI).

Не заменяют прогон в SITL: проверяют автоматы узлов, имена полей сообщений и обработку ошибок.
Узлы общаются через общую шину `BUS` (топик -> подписчики), время — `CLOCK` (шаг задаёт тест).

    with ros_stubs.installed():
        from invent_mission.mission_node import MissionNode
        node = MissionNode()          # параметры — ros_stubs.PARAMS[имя узла]
"""
from __future__ import annotations

import contextlib
import sys
import types
from collections import defaultdict


class Msg:
    """Сообщение: любые поля через kwargs, отсутствующие — по умолчанию из FIELDS."""
    FIELDS: dict = {}

    def __init__(self, **kw):
        for k, v in self.FIELDS.items():
            setattr(self, k, v() if callable(v) else v)
        for k, v in kw.items():
            if self.FIELDS and k not in self.FIELDS:
                raise AttributeError(f"{type(self).__name__} не имеет поля {k}")
            setattr(self, k, v)


def _msg(name: str, fields: dict, **consts) -> type:
    return type(name, (Msg,), {"FIELDS": fields, **consts})


class Header(Msg):
    FIELDS = {"stamp": None, "frame_id": ""}


# --- время и шина ------------------------------------------------------------------------

class _Clock:
    t_ns = 0

    def now(self):
        t = self.t_ns
        return types.SimpleNamespace(nanoseconds=t, to_msg=lambda: types.SimpleNamespace(sec=t // 10**9, nanosec=t % 10**9))


CLOCK = _Clock()
BUS: dict[str, list] = defaultdict(list)      # топик -> [callback]
SENT: dict[str, list] = defaultdict(list)     # топик -> [сообщения] (для проверок)
PARAMS: dict[str, dict] = defaultdict(dict)   # имя узла -> {параметр: значение}
NODES: list = []


def reset():
    CLOCK.t_ns = 0
    for d in (BUS, SENT, PARAMS):
        d.clear()
    NODES.clear()


def publish(topic: str, msg):
    SENT[topic].append(msg)
    for cb in list(BUS[topic]):
        cb(msg)


class _Pub:
    def __init__(self, topic):
        self.topic = topic

    def publish(self, msg):
        publish(self.topic, msg)


class _Logger:
    def __init__(self, name):
        self.name, self.lines = name, []

    def _log(self, level, text):
        self.lines.append((level, text))

    def info(self, t):
        self._log("INFO", t)

    def warn(self, t):
        self._log("WARN", t)

    warning = warn

    def error(self, t):
        self._log("ERROR", t)

    def text(self) -> str:
        return "\n".join(t for _, t in self.lines)


class Node:
    def __init__(self, name):
        self._name, self._params, self.timers, self.services = name, {}, [], {}
        self._logger = _Logger(name)
        NODES.append(self)

    def declare_parameter(self, name, default=None):
        self._params[name] = PARAMS[self._name].get(name, default)

    def get_parameter(self, name):
        return types.SimpleNamespace(value=self._params[name])

    def create_publisher(self, _type, topic, _qos):
        return _Pub(topic)

    def create_subscription(self, _type, topic, cb, _qos):
        BUS[topic].append(cb)

    def create_service(self, srv_type, name, cb):
        self.services[name] = (srv_type, cb)

    def call(self, name):
        srv_type, cb = self.services[name]
        return cb(srv_type.Request(), srv_type.Response())

    def create_timer(self, period, cb):
        self.timers.append([period, cb, 0.0])

    def get_logger(self):
        return self._logger

    def get_clock(self):
        return CLOCK

    def destroy_node(self):
        pass


def spin_for(seconds: float, dt: float = 0.05, hook=None):
    """Крутит таймеры всех узлов `seconds` с шагом dt; hook(t) — модель мира на каждом шаге."""
    t_end = CLOCK.t_ns + int(seconds * 1e9)
    while CLOCK.t_ns < t_end:
        CLOCK.t_ns += int(dt * 1e9)
        t = CLOCK.t_ns * 1e-9
        if hook:
            hook(t)
        for node in list(NODES):
            for tm in node.timers:
                if t - tm[2] >= tm[0] - 1e-9:
                    tm[2] = t
                    tm[1]()


# --- модули ------------------------------------------------------------------------------

def _modules() -> dict[str, types.ModuleType]:
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        return m

    enum = lambda *names: type("E", (), {n: n for n in names})  # noqa: E731
    qos = mod("rclpy.qos", QoSProfile=lambda **kw: kw, DurabilityPolicy=enum("TRANSIENT_LOCAL", "VOLATILE"),
              HistoryPolicy=enum("KEEP_LAST"), ReliabilityPolicy=enum("BEST_EFFORT", "RELIABLE"),
              qos_profile_sensor_data=None)
    rclpy = mod("rclpy", init=lambda *a, **k: None, spin=lambda n: None, try_shutdown=lambda: None,
                shutdown=lambda: None, ok=lambda: True)
    rclpy.node = mod("rclpy.node", Node=Node)
    rclpy.qos = qos

    px4 = mod("px4_msgs.msg",
              OffboardControlMode=_msg("OffboardControlMode", {"timestamp": 0, "position": False, "velocity": False,
                                                                "acceleration": False, "attitude": False,
                                                                "body_rate": False}),
              TrajectorySetpoint=_msg("TrajectorySetpoint", {"timestamp": 0, "position": None, "velocity": None,
                                                              "acceleration": None, "jerk": None, "yaw": 0.0,
                                                              "yawspeed": 0.0}),
              # значения — PX4 v1.17 msg/VehicleCommand.msg, msg/versioned/VehicleStatus.msg
              VehicleCommand=_msg("VehicleCommand", {k: 0 for k in (
                  "timestamp", "command", "param1", "param2", "param3", "param4", "param5", "param6", "param7",
                  "target_system", "target_component", "source_system", "source_component", "confirmation")}
                  | {"from_external": False},
                  VEHICLE_CMD_NAV_LAND=21, VEHICLE_CMD_DO_SET_MODE=176, VEHICLE_CMD_COMPONENT_ARM_DISARM=400),
              VehicleLocalPosition=_msg("VehicleLocalPosition", {"timestamp": 0, "x": 0.0, "y": 0.0, "z": 0.0,
                                                                  "vx": 0.0, "vy": 0.0, "vz": 0.0, "heading": 0.0,
                                                                  "xy_valid": False, "z_valid": False}),
              VehicleStatus=_msg("VehicleStatus", {"timestamp": 0, "arming_state": 1, "nav_state": 0},
                                 ARMING_STATE_DISARMED=1, ARMING_STATE_ARMED=2, NAVIGATION_STATE_POSCTL=2,
                                 NAVIGATION_STATE_AUTO_LAND=18, NAVIGATION_STATE_OFFBOARD=14),
              VehicleOdometry=_msg("VehicleOdometry", {}))
    inv = mod("invent_msgs.msg",
              MissionStatus=_msg("MissionStatus", {"header": Header, "state": "", "sortie": 0, "waypoint_index": 0,
                                                    "waypoint_count": 0, "targets": list}),
              LabelRead=_msg("LabelRead", {"header": Header, "text": "", "symbology": "", "corners": list,
                                            "sscc": "", "address": "", "confidence": 0.0}))
    std = mod("std_msgs.msg", String=_msg("String", {"data": ""}))
    trigger = type("Trigger", (), {"Request": _msg("TriggerRequest", {}),
                                   "Response": _msg("TriggerResponse", {"success": False, "message": ""})})
    srvs = mod("std_srvs.srv", Trigger=trigger)
    return {"rclpy": rclpy, "rclpy.node": rclpy.node, "rclpy.qos": qos,
            "px4_msgs": mod("px4_msgs", msg=px4), "px4_msgs.msg": px4,
            "invent_msgs": mod("invent_msgs", msg=inv), "invent_msgs.msg": inv,
            "std_msgs": mod("std_msgs", msg=std), "std_msgs.msg": std,
            "std_srvs": mod("std_srvs", srv=srvs), "std_srvs.srv": srvs}


@contextlib.contextmanager
def installed(*package_dirs: str):
    """Подменяет ROS-модули в sys.modules; package_dirs — каталоги пакетов ros2_ws/src/<пакет>."""
    fakes = _modules()
    saved = {k: sys.modules.get(k) for k in fakes}
    saved_path = list(sys.path)
    sys.modules.update(fakes)
    sys.path[:0] = list(package_dirs)
    reset()
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        for k in [k for k in sys.modules if k.startswith(("invent_mission", "invent_perception",
                                                          "invent_localization"))]:
            del sys.modules[k]
        sys.path[:] = saved_path

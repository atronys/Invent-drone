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

NAN = float("nan")


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


TF: dict[str, object] = {}                    # дочерний кадр -> последний TransformStamped (родитель в header)


def reset():
    CLOCK.t_ns = 0
    for d in (BUS, SENT, PARAMS, TF):
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

    def debug(self, t, **_):
        self._log("DEBUG", t)

    def info(self, t, **_):
        self._log("INFO", t)

    def warn(self, t, **_):      # throttle_duration_sec и т. п. — как в rclpy
        self._log("WARN", t)

    warning = warn

    def error(self, t, **_):
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


# --- tf2 ---------------------------------------------------------------------------------

class TransformException(Exception):
    pass


class _Broadcaster:
    def __init__(self, _node=None):
        pass

    def sendTransform(self, tfs):  # noqa: N802 — имя как в tf2_ros
        for t in tfs if isinstance(tfs, (list, tuple)) else [tfs]:
            TF[t.child_frame_id] = t


class _Buffer:
    """lookup_transform(target, source) = T_target_source: цепочка от source вверх по родителям до target."""

    def __init__(self, *_, **__):
        pass

    def lookup_transform(self, target, source, _time=None, timeout=None):
        from invent_core.frames import matmul, matrix_from_quat, quat_from_matrix
        r = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        t = (0.0, 0.0, 0.0)
        frame = source
        while frame != target:
            if frame not in TF:
                raise TransformException(f"нет цепочки {target} <- {source} (оборвалась на {frame})")
            tr = TF[frame].transform
            q, p = tr.rotation, tr.translation
            rp = matrix_from_quat(q.w, q.x, q.y, q.z)
            t = tuple(sum(rp[i][j] * t[j] for j in range(3)) + (p.x, p.y, p.z)[i] for i in range(3))
            r = matmul(rp, r)
            frame = TF[frame].header.frame_id
        out = _GEO["TransformStamped"]()
        out.header.frame_id, out.child_frame_id = target, source
        out.transform.translation.x, out.transform.translation.y, out.transform.translation.z = t
        w, x, y, z = quat_from_matrix(r)
        out.transform.rotation.w, out.transform.rotation.x = w, x
        out.transform.rotation.y, out.transform.rotation.z = y, z
        return out


_GEO: dict = {}


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
              # PX4 v1.17 msg/versioned/VehicleOdometry.msg (MESSAGE_VERSION = 0)
              VehicleOdometry=_msg("VehicleOdometry", {
                  "timestamp": 0, "timestamp_sample": 0, "pose_frame": 0, "position": lambda: [NAN] * 3,
                  "q": lambda: [NAN] * 4, "velocity_frame": 0, "velocity": lambda: [NAN] * 3,
                  "angular_velocity": lambda: [NAN] * 3, "position_variance": lambda: [NAN] * 3,
                  "orientation_variance": lambda: [NAN] * 3, "velocity_variance": lambda: [NAN] * 3,
                  "reset_counter": 0, "quality": 0},
                  POSE_FRAME_UNKNOWN=0, POSE_FRAME_NED=1, POSE_FRAME_FRD=2, VELOCITY_FRAME_UNKNOWN=0,
                  VELOCITY_FRAME_NED=1, VELOCITY_FRAME_FRD=2, VELOCITY_FRAME_BODY_FRD=3))
    vec = _msg("Vector3", {"x": 0.0, "y": 0.0, "z": 0.0})
    quat = _msg("Quaternion", {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})
    transform = _msg("Transform", {"translation": vec, "rotation": quat})
    point = _msg("Point", {"x": 0.0, "y": 0.0, "z": 0.0})
    pose = _msg("Pose", {"position": point, "orientation": quat})
    _GEO.update(TransformStamped=_msg("TransformStamped", {"header": Header, "child_frame_id": "",
                                                           "transform": transform}))
    geo = mod("geometry_msgs.msg", Vector3=vec, Quaternion=quat, Transform=transform, Point=point, Pose=pose,
              TransformStamped=_GEO["TransformStamped"],
              PoseWithCovariance=_msg("PoseWithCovariance", {"pose": pose, "covariance": lambda: [0.0] * 36}),
              Twist=_msg("Twist", {"linear": vec, "angular": vec}))
    geo.TwistWithCovariance = _msg("TwistWithCovariance", {"twist": geo.Twist, "covariance": lambda: [0.0] * 36})
    nav = mod("nav_msgs.msg", Odometry=_msg("Odometry", {"header": Header, "child_frame_id": "",
                                                         "pose": geo.PoseWithCovariance,
                                                         "twist": geo.TwistWithCovariance}))
    tf2_msgs = mod("tf2_msgs.msg", TFMessage=_msg("TFMessage", {"transforms": list}))
    # Image: пиксели лежат в .array (numpy) — заглушка cv_bridge отдаёт их как есть
    image = _msg("Image", {"header": Header, "height": 0, "width": 0, "encoding": "", "is_bigendian": 0,
                           "step": 0, "data": b"", "array": None})

    class CvBridge:
        def imgmsg_to_cv2(self, msg, desired_encoding="passthrough"):
            return msg.array

        def cv2_to_imgmsg(self, arr, encoding="passthrough"):
            return image(height=arr.shape[0], width=arr.shape[1], encoding=encoding, array=arr)

    class Duration:
        def __init__(self, seconds=0.0, nanoseconds=0):
            self.nanoseconds = int(seconds * 1e9) + nanoseconds

    class Time:
        def __init__(self, seconds=0.0, nanoseconds=0):
            self.nanoseconds = int(seconds * 1e9) + nanoseconds

        @classmethod
        def from_msg(cls, stamp):
            return cls(nanoseconds=(stamp.sec * 10**9 + stamp.nanosec) if stamp is not None else 0)

    rclpy.duration = mod("rclpy.duration", Duration=Duration)
    rclpy.time = mod("rclpy.time", Time=Time)
    tf2_ros = mod("tf2_ros", Buffer=_Buffer, TransformListener=lambda buf, node=None, **kw: None,
                  TransformBroadcaster=_Broadcaster, StaticTransformBroadcaster=_Broadcaster,
                  TransformException=TransformException)
    inv = mod("invent_msgs.msg",
              MissionStatus=_msg("MissionStatus", {"header": Header, "state": "", "sortie": 0, "waypoint_index": 0,
                                                    "waypoint_count": 0, "targets": list}),
              LabelRead=_msg("LabelRead", {"header": Header, "text": "", "symbology": "", "corners": list,
                                            "sscc": "", "address": "", "confidence": 0.0}))
    std = mod("std_msgs.msg", String=_msg("String", {"data": ""}), Float32=_msg("Float32", {"data": 0.0}),
              Header=Header)
    trigger = type("Trigger", (), {"Request": _msg("TriggerRequest", {}),
                                   "Response": _msg("TriggerResponse", {"success": False, "message": ""})})
    srvs = mod("std_srvs.srv", Trigger=trigger)
    sensor = mod("sensor_msgs.msg", Image=image)
    out = {"rclpy": rclpy, "rclpy.node": rclpy.node, "rclpy.qos": qos, "rclpy.duration": rclpy.duration,
           "rclpy.time": rclpy.time, "tf2_ros": tf2_ros, "cv_bridge": mod("cv_bridge", CvBridge=CvBridge)}
    for pkg, sub, m in (("px4_msgs", "msg", px4), ("invent_msgs", "msg", inv), ("std_msgs", "msg", std),
                        ("std_srvs", "srv", srvs), ("geometry_msgs", "msg", geo), ("nav_msgs", "msg", nav),
                        ("tf2_msgs", "msg", tf2_msgs), ("sensor_msgs", "msg", sensor)):
        out[pkg] = mod(pkg, **{sub: m})
        out[f"{pkg}.{sub}"] = m
    return out


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

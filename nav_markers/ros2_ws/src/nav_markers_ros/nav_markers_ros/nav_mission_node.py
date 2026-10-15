"""Навигационное задание: долететь до заданных ячеек склада и «захватить» каждую (документ, разд. 8).

Маршрут — nav_core.route.plan_task: магистраль -> вход в проход -> ось -> подлёт к ячейке -> захват -> ...
-> док. Уставки — «морковка» по отрезку с его скоростью (как в INVENT_DRONE mission_node), Offboard PX4.

Логика меток:
* до взлёта нужна хотя бы одна поза по меткам (якоря видны с дока) — иначе не взлетаем;
* в проходе метки должны приходить не реже fix_timeout_s; нет — SEARCH: зависнуть на месте и ждать;
  нашлась — продолжить; нет за search_timeout_s — посадка на месте (матрица аварийных режимов INVENT_DRONE);
* захват ячейки: зависание, пока поза по меткам не станет свежей (< capture_fix_age_s) и точной
  (σ_xy < capture_sigma); затем удержание hold_s и событие capture — его журнал сравнит с истиной.

События — /nav/mission_event (JSON), их пишет nav_logger и разбирает eval_nav.py.
Запуск: ros2 run nav_markers_ros nav_mission --ros-args -p layout:=... -p targets:="B-05-2-2,C-03-1-1"
"""
import json
import math
from enum import Enum

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus

from invent_core.frames import enu_to_ned, map_to_px4, px4_to_map, yaw_enu_to_ned
from invent_core.layout import Warehouse
from nav_core.route import TaskParams, plan_task


class State(Enum):
    WAIT = "WAIT"
    ARMING = "ARMING"
    FOLLOW = "FOLLOW"
    SEARCH = "SEARCH"
    CAPTURE = "CAPTURE"
    LAND = "LAND"
    DONE = "DONE"
    ABORT = "ABORT"


IN_AISLE = {"enter", "aisle", "approach", "capture", "exit"}


class NavMission(Node):
    def __init__(self):
        super().__init__("nav_mission")
        defaults = TaskParams()
        for name, default in [("layout", ""), ("targets", ""), ("rate_hz", 20.0), ("acceptance_radius", 0.15),
                              ("z_transit", defaults.z_transit), ("v_cross", defaults.v_cross),
                              ("v_aisle", defaults.v_aisle), ("v_approach", defaults.v_approach), ("v_z", defaults.v_z),
                              ("standoff", defaults.standoff), ("capture_hold_s", defaults.capture_hold_s),
                              ("fix_timeout_s", 4.0), ("search_timeout_s", 15.0), ("capture_fix_age_s", 0.5),
                              ("capture_sigma", 0.05), ("capture_timeout_s", 15.0),
                              ("vehicle_status_topic", "/fmu/out/vehicle_status_v1"),
                              ("local_position_topic", "/fmu/out/vehicle_local_position_v1")]:
            self.declare_parameter(name, default)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        wh = Warehouse.load(g("layout"))
        self.dock = wh.dock
        tp = TaskParams(z_transit=g("z_transit"), v_cross=g("v_cross"), v_aisle=g("v_aisle"),
                        v_approach=g("v_approach"), v_z=g("v_z"), standoff=g("standoff"),
                        capture_hold_s=g("capture_hold_s"))
        targets = [t.strip() for t in g("targets").split(",") if t.strip()]
        self.wps = plan_task(wh, targets, tp)
        self.rate, self.acc = g("rate_hz"), g("acceptance_radius")
        self.fix_timeout, self.search_timeout = g("fix_timeout_s"), g("search_timeout_s")
        self.cap_age, self.cap_sigma, self.cap_timeout = g("capture_fix_age_s"), g("capture_sigma"), g("capture_timeout_s")
        length = sum(math.dist((a.x, a.y, a.z), (b.x, b.y, b.z)) for a, b in zip(self.wps, self.wps[1:]))
        self.get_logger().info(f"задание: {len(targets)} ячеек, {len(self.wps)} точек, путь {length:.0f} м")

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_mode = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", qos)
        self.pub_sp = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos)
        self.pub_cmd = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos)
        self.pub_event = self.create_publisher(String, "/nav/mission_event", 50)
        self.pub_state = self.create_publisher(String, "/nav/mission_state", 10)
        self.create_subscription(VehicleLocalPosition, g("local_position_topic"), self.on_position, qos)
        self.create_subscription(VehicleStatus, g("vehicle_status_topic"), self.on_status, qos)
        self.create_subscription(PoseWithCovarianceStamped, "/nav/marker_fix", self.on_fix, 10)

        self.pos = None
        self.status = None
        self.fix_t = None            # время последней позы по меткам (время симуляции)
        self.fix_sigma = math.inf
        self.fix_pos = None
        self.state = State.WAIT
        self.ticks = 0
        self.i = 0
        self.carrot = None
        self.yaw = 0.0               # курс постоянный: вдоль +x (route.py)
        self.t_state = 0.0
        self.create_timer(1.0 / self.rate, self.tick)

    # --- входы ------------------------------------------------------------------------------------------
    def on_position(self, msg: VehicleLocalPosition):
        if msg.xy_valid and msg.z_valid:
            self.pos = px4_to_map((msg.x, msg.y, msg.z), self.dock)

    def on_status(self, msg: VehicleStatus):
        self.status = msg

    def on_fix(self, msg: PoseWithCovarianceStamped):
        self.fix_t = self.now_s()
        c = msg.pose.covariance
        self.fix_sigma = math.sqrt(max(0.0, c[0] + c[7]))
        p = msg.pose.pose.position
        self.fix_pos = (p.x, p.y, p.z)

    # --- выходы -----------------------------------------------------------------------------------------
    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def fix_age(self) -> float:
        return math.inf if self.fix_t is None else self.now_s() - self.fix_t

    def event(self, kind: str, **kw):
        wp = self.wps[self.i]
        d = {"t": round(self.now_s(), 3), "event": kind, "wp": self.i, "wp_kind": wp.kind, "label": wp.label,
             "target": [wp.x, wp.y, wp.z], "est": list(self.pos) if self.pos else None, **kw}
        self.pub_event.publish(String(data=json.dumps(d, ensure_ascii=False)))
        self.get_logger().info(f"{kind}: {wp.label} {kw if kw else ''}")

    def command(self, cmd: int, p1: float = 0.0, p2: float = 0.0):
        self.pub_cmd.publish(VehicleCommand(command=cmd, param1=p1, param2=p2, target_system=1, target_component=1,
                                            source_system=1, source_component=1, from_external=True, timestamp=0))

    def setpoint(self, xyz, vel=(0.0, 0.0, 0.0)):
        m = TrajectorySetpoint(timestamp=0)
        m.position = [float(v) for v in map_to_px4(xyz, self.dock)]
        m.velocity = [float(v) for v in enu_to_ned(vel)]
        m.acceleration = [math.nan] * 3
        m.yaw = float(yaw_enu_to_ned(self.yaw))
        self.pub_sp.publish(m)

    def set_state(self, s: State):
        self.state, self.t_state = s, self.now_s()
        self.pub_state.publish(String(data=s.value))

    # --- автомат ----------------------------------------------------------------------------------------
    def start_segment(self):
        self.seg_from, self.seg_s, self.hold_until = self.carrot, 0.0, None

    def tick(self):
        self.ticks += 1
        self.pub_mode.publish(OffboardControlMode(position=True, timestamp=0))
        st = self.state
        if st == State.WAIT:
            if self.pos is None:
                return
            self.carrot = self.pos
            self.setpoint(self.carrot)
            if self.fix_age() > 1.0:
                if self.ticks % int(5 * self.rate) == 0:
                    self.get_logger().warn("нет позы по меткам на доке — не взлетаю (якоря в кадре фронтальной камеры?)")
                return
            if self.ticks > self.rate:
                self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.set_state(State.ARMING)
        elif st == State.ARMING:
            self.setpoint(self.carrot)
            s = self.status
            if s and s.arming_state == VehicleStatus.ARMING_STATE_ARMED and s.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                self.i = 0
                self.event("start")
                self.set_state(State.FOLLOW)
                self.start_segment()
            elif self.now_s() - self.t_state > 5.0:
                self.get_logger().warn("нет Offboard/ARM за 5 с — повторяю")
                self.set_state(State.WAIT)
        elif st in (State.FOLLOW, State.SEARCH, State.CAPTURE):
            if self.status and self.status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                self.event("abort", reason=f"nav_state {self.status.nav_state}")
                self.set_state(State.ABORT)
                return
            {State.FOLLOW: self.follow, State.SEARCH: self.search, State.CAPTURE: self.capture}[st]()
        elif st == State.LAND:
            if self.status and self.status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                self.event("done")
                self.set_state(State.DONE)

    def follow(self):
        wp = self.wps[self.i]
        if wp.kind == "land":          # над доком: посадку ведёт PX4 (Land), а не уставки до земли
            self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.event("land")
            self.set_state(State.LAND)
            return
        if wp.kind in IN_AISLE and self.fix_age() > self.fix_timeout:
            self.event("search", fix_age=round(self.fix_age(), 2))
            self.carrot = self.pos or self.carrot
            self.set_state(State.SEARCH)
            return
        b = (wp.x, wp.y, wp.z)
        a = self.seg_from
        length = math.dist(a, b)
        self.seg_s = min(length, self.seg_s + max(wp.speed, 0.05) / self.rate)
        k = self.seg_s / length if length > 1e-6 else 1.0
        self.carrot = tuple(a[j] + (b[j] - a[j]) * k for j in range(3))
        vel = tuple((b[j] - a[j]) / length * wp.speed for j in range(3)) if 0 < self.seg_s < length else (0.0, 0.0, 0.0)
        self.setpoint(self.carrot, vel)
        if self.seg_s < length or self.pos is None or math.dist(self.pos, b) > self.acc:
            return
        if wp.kind == "capture":
            self.set_state(State.CAPTURE)
            return
        self.event("reached")
        self.next_wp()

    def next_wp(self):
        self.i += 1
        self.start_segment()

    def search(self):
        self.setpoint(self.carrot)
        if self.fix_age() < 0.5:
            self.event("found", searched_s=round(self.now_s() - self.t_state, 2))
            self.set_state(State.FOLLOW)
            self.start_segment()
        elif self.now_s() - self.t_state > self.search_timeout:
            self.event("lost_land", searched_s=round(self.now_s() - self.t_state, 2))
            self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.set_state(State.LAND)

    def capture(self):
        wp = self.wps[self.i]
        self.setpoint(self.carrot)
        waited = self.now_s() - self.t_state
        good = self.fix_age() < self.cap_age and self.fix_sigma < self.cap_sigma
        if self.hold_until is None:
            if good:
                self.hold_until = self.now_s() + wp.hold_s
            elif waited > self.cap_timeout:
                self.event("capture_failed", fix_age=round(self.fix_age(), 2), fix_sigma=round(self.fix_sigma, 3))
                self.set_state(State.FOLLOW)
                self.next_wp()
            return
        if self.now_s() >= self.hold_until:
            self.event("capture", fix_sigma=round(self.fix_sigma, 3), fix=self.fix_pos, waited_s=round(waited, 2))
            self.set_state(State.FOLLOW)
            self.next_wp()


def main():
    rclpy.init()
    node = NavMission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

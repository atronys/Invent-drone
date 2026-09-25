"""Узел миссии: ведёт PX4 по плану облёта (invent_core.coverage) в режиме Offboard.

Кадры. План — в `map` (ENU склада). Локальный кадр PX4 — NED с началом в точке
инициализации EKF (= док). map -> PX4: вычесть dock, затем ENU -> NED. С внешней
навигацией (VIO -> EKF2) выравнивание кадров проверяется отдельно (задача LOC-05).

Уставки — «морковка»: точка ползёт по отрезку с нужной скоростью (v_scan при
сканировании — ограничение смаза этикеток), рыскание — с ограничением yaw_rate
(резкие развороты срывают VIO, статья A, сценарий 4).

Запуск: ros2 run invent_mission mission_node --ros-args -p layout:=/path/warehouse.json
"""
import math
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from invent_msgs.msg import MissionStatus
from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint, VehicleCommand,
                          VehicleLocalPosition, VehicleStatus)

from invent_core.coverage import ScanParams, plan
from invent_core.layout import Warehouse


class State(Enum):
    WAIT = "WAIT"        # ждём позицию и шлём поток уставок (PX4 требует его до Offboard)
    ARMING = "ARMING"
    FOLLOW = "FOLLOW"
    LAND = "LAND"
    DONE = "DONE"
    ABORT = "ABORT"


def enu_to_ned(x, y, z):
    return y, x, -z   # преобразование симметрично: оно же NED -> ENU


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class MissionNode(Node):
    def __init__(self):
        super().__init__("invent_mission")
        self.declare_parameter("layout", "")
        self.declare_parameter("sortie", 0)
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("acceptance_radius", 0.15)
        # PX4 >= 1.16 версионирует сообщения: может быть /fmu/out/vehicle_status_v1 — проверь `ros2 topic list`
        self.declare_parameter("vehicle_status_topic", "/fmu/out/vehicle_status")
        self.declare_parameter("local_position_topic", "/fmu/out/vehicle_local_position")

        wh = Warehouse.load(self.get_parameter("layout").value)
        self.p = ScanParams()
        pl = plan(wh, self.p)
        for w in pl.warnings:
            self.get_logger().warn(w)
        self.sortie = self.get_parameter("sortie").value
        self.wps = pl.sorties[self.sortie].waypoints
        self.dock = wh.dock
        self.get_logger().info(
            f"вылет {self.sortie + 1}/{len(pl.sorties)}: {len(self.wps)} точек, "
            f"~{pl.sorties[self.sortie].duration_s / 60:.1f} мин")

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_mode = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", qos)
        self.pub_sp = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos)
        self.pub_cmd = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos)
        self.pub_status = self.create_publisher(MissionStatus, "mission/status", 10)
        self.create_subscription(VehicleLocalPosition, self.get_parameter("local_position_topic").value,
                                 self.on_position, qos)
        self.create_subscription(VehicleStatus, self.get_parameter("vehicle_status_topic").value,
                                 self.on_status, qos)

        self.rate = self.get_parameter("rate_hz").value
        self.acc = self.get_parameter("acceptance_radius").value
        self.pos = None          # (x, y, z) в map
        self.yaw = 0.0           # курс ENU
        self.status = None
        self.state = State.WAIT
        self.ticks = 0
        self.i = 0               # индекс целевой точки
        self.carrot = None       # текущая уставка (map)
        self.carrot_yaw = 0.0
        self.create_timer(1.0 / self.rate, self.tick)

    # --- входы ------------------------------------------------------------
    def on_position(self, msg: VehicleLocalPosition):
        if msg.xy_valid and msg.z_valid:
            e, n, u = enu_to_ned(msg.x, msg.y, msg.z)
            self.pos = (e + self.dock[0], n + self.dock[1], u + self.dock[2])
            self.yaw = wrap(math.pi / 2 - msg.heading)  # NED -> ENU

    def on_status(self, msg: VehicleStatus):
        self.status = msg

    # --- выходы -----------------------------------------------------------
    def now_us(self) -> int:
        return self.get_clock().now().nanoseconds // 1000

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def command(self, cmd: int, p1: float = 0.0, p2: float = 0.0):
        m = VehicleCommand(command=cmd, param1=p1, param2=p2, target_system=1, target_component=1,
                           source_system=1, source_component=1, from_external=True)
        m.timestamp = self.now_us()
        self.pub_cmd.publish(m)

    def setpoint(self, xyz, yaw_enu, vel=(0.0, 0.0, 0.0)):
        m = TrajectorySetpoint()
        m.timestamp = self.now_us()
        m.position = [float(v) for v in enu_to_ned(*(xyz[k] - self.dock[k] for k in range(3)))]
        m.velocity = [float(v) for v in enu_to_ned(*vel)]          # feed-forward вдоль отрезка
        m.acceleration = [math.nan] * 3
        m.yaw = float(wrap(math.pi / 2 - yaw_enu))                  # ENU -> NED
        self.pub_sp.publish(m)

    def publish_status(self):
        m = MissionStatus(state=self.state.value, sortie=self.sortie, waypoint_index=self.i,
                          waypoint_count=len(self.wps))
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "map"
        if self.state == State.FOLLOW:
            m.targets = list(self.wps[self.i].targets)
        self.pub_status.publish(m)

    # --- автомат ----------------------------------------------------------
    def start_segment(self):
        self.seg_from, self.seg_yaw0, self.seg_t0, self.hold_until = self.carrot, self.carrot_yaw, self.now_s(), None

    def segment_speed(self) -> float:
        a, b = self.seg_from, self.wps[self.i]
        prev_kind = self.wps[self.i - 1].kind if self.i > 0 else "transit"
        if math.hypot(b.x - a[0], b.y - a[1]) < 1e-3:
            return self.p.v_z
        return self.p.v_scan if prev_kind == b.kind == "scan" else self.p.v_transit

    def tick(self):
        self.ticks += 1
        self.pub_mode.publish(OffboardControlMode(position=True, timestamp=self.now_us()))
        self.publish_status()

        if self.state == State.WAIT:
            if self.pos is None:
                return
            self.carrot, self.carrot_yaw = self.pos, self.yaw
            self.setpoint(self.carrot, self.carrot_yaw)
            if self.ticks > self.rate:  # >= 1 с потока уставок
                self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)  # 6 = Offboard
                self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.state, self.arm_t0 = State.ARMING, self.now_s()

        elif self.state == State.ARMING:
            self.setpoint(self.carrot, self.carrot_yaw)
            s = self.status
            if s and s.arming_state == VehicleStatus.ARMING_STATE_ARMED \
                    and s.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                self.get_logger().info("Offboard + ARM: начинаю маршрут")
                self.state, self.i = State.FOLLOW, 0
                self.start_segment()
            elif self.now_s() - self.arm_t0 > 5.0:
                self.get_logger().warn("нет Offboard/ARM за 5 с — повторяю команды")
                self.state = State.WAIT

        elif self.state == State.FOLLOW:
            if self.status and self.status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                self.get_logger().error("вышли из Offboard (пилот/failsafe) — прекращаю управление")
                self.state = State.ABORT
                return
            self.follow()

        elif self.state == State.LAND:
            if self.status and self.status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                self.get_logger().info("посадка завершена")
                self.state = State.DONE

    def follow(self):
        target = self.wps[self.i]
        t = self.now_s() - self.seg_t0
        a, b = self.seg_from, (target.x, target.y, target.z)
        length = math.dist(a, b)
        speed = self.segment_speed()
        s = min(length, speed * t)
        k = s / length if length > 1e-6 else 1.0
        self.carrot = tuple(a[j] + (b[j] - a[j]) * k for j in range(3))
        vel = tuple((b[j] - a[j]) / length * speed for j in range(3)) if s < length else (0.0, 0.0, 0.0)

        dyaw = wrap(target.yaw - self.seg_yaw0)
        step = math.radians(self.p.yaw_rate_deg) * t
        self.carrot_yaw = target.yaw if abs(dyaw) <= step else self.seg_yaw0 + math.copysign(step, dyaw)
        self.setpoint(self.carrot, self.carrot_yaw, vel)

        arrived = s >= length and self.pos is not None and math.dist(self.pos, b) < self.acc \
            and abs(wrap(target.yaw - self.carrot_yaw)) < 1e-3
        if not arrived:
            return
        if self.hold_until is None:
            self.hold_until = self.now_s() + target.hold_s
        if self.now_s() < self.hold_until:
            return
        self.i += 1
        if self.i == len(self.wps):
            self.i -= 1
            self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.state = State.LAND
            self.get_logger().info("маршрут пройден, посадка на док")
        else:
            self.start_segment()


def main():
    rclpy.init()
    node = MissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

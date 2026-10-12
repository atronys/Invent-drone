"""Узел миссии: ведёт PX4 по плану облёта (invent_core.coverage) в режиме Offboard.

Кадры. План — в `map` (ENU склада). Локальный кадр PX4 — NED с началом в точке
инициализации EKF (= док). map -> PX4: вычесть dock, затем ENU -> NED. С внешней
навигацией (VIO -> EKF2) выравнивание кадров проверяется отдельно (задача LOC-05).

Уставки — «морковка»: точка ползёт по отрезку с нужной скоростью (v_scan при
сканировании — ограничение смаза этикеток), рыскание — с ограничением yaw_rate
(резкие развороты срывают VIO, статья A, сценарий 4).

Серия вылетов (FLT-03): -p series:=true — после посадки узел ждёт замены АКБ и команды
`ros2 service call /mission/continue std_srvs/srv/Trigger`, затем летит следующий вылет плана.
В SITL замену АКБ изображает пауза: -p continue_after_s:=10.

Запуск: ros2 run invent_mission mission_node --ros-args -p layout:=/path/warehouse.json
"""
import math
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from invent_msgs.msg import MissionStatus
from std_msgs.msg import String
from std_srvs.srv import Trigger
from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint, VehicleCommand,
                          VehicleLocalPosition, VehicleStatus)

from invent_core.coverage import ScanParams, plan
from invent_core.frames import enu_to_ned, map_to_px4, px4_to_map, wrap, yaw_enu_to_ned, yaw_ned_to_enu
from invent_core.layout import Warehouse


class State(Enum):
    WAIT = "WAIT"        # ждём позицию и шлём поток уставок (PX4 требует его до Offboard)
    ARMING = "ARMING"
    FOLLOW = "FOLLOW"
    LAND = "LAND"
    DONE = "DONE"
    ABORT = "ABORT"


class MissionNode(Node):
    def __init__(self):
        super().__init__("invent_mission")
        self.declare_parameter("layout", "")
        self.declare_parameter("scan_params", "")   # JSON ScanParams (FLT-02); пусто — значения по умолчанию
        self.declare_parameter("sortie", 0)
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("acceptance_radius", 0.15)
        self.declare_parameter("series", False)          # FLT-03: после посадки — следующий вылет плана
        self.declare_parameter("continue_after_s", -1.0)  # < 0 — ждать /mission/continue; >= 0 — сам (SITL)
        # PX4 >= 1.16: к топику добавляется суффикс _v<N>, если MESSAGE_VERSION != 0
        # (PX4 v1.17.0, src/modules/uxrce_dds_client/utilities.hpp, generate_topic_name).
        # VehicleStatus и VehicleLocalPosition — версия 1; TrajectorySetpoint, VehicleCommand — 0 (без суффикса).
        # Пример px4_ros_com в main использует старые имена — он устарел. Проверка: `ros2 topic list | grep fmu`.
        self.declare_parameter("vehicle_status_topic", "/fmu/out/vehicle_status_v1")
        self.declare_parameter("local_position_topic", "/fmu/out/vehicle_local_position_v1")

        wh = Warehouse.load(self.get_parameter("layout").value)
        params_file = self.get_parameter("scan_params").value
        self.p = ScanParams.load(params_file) if params_file else ScanParams()
        pl = plan(wh, self.p)
        for w in pl.warnings:
            self.get_logger().warn(w)
        self.sorties = pl.sorties
        self.series = self.get_parameter("series").value
        self.continue_after = self.get_parameter("continue_after_s").value
        self.dock = wh.dock
        self.load_sortie(self.get_parameter("sortie").value)

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
        # SAFE-01/02: GO / SLOW (скорость ×0.5) / STOP (замереть). Нет узла безопасности — считаем GO.
        self.safety = "GO"
        self.create_subscription(String, "safety/state", self.on_safety, 10)
        self.create_service(Trigger, "mission/continue", self.on_continue)

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
        self.done_t = None       # когда сел (для continue_after_s)
        self.create_timer(1.0 / self.rate, self.tick)

    def load_sortie(self, k: int):
        self.sortie, self.wps = k, self.sorties[k].waypoints
        self.get_logger().info(
            f"вылет {k + 1}/{len(self.sorties)}: {len(self.wps)} точек, ~{self.sorties[k].duration_s / 60:.1f} мин")

    def has_next(self) -> bool:
        return self.series and self.sortie + 1 < len(self.sorties)

    def next_sortie(self):
        self.load_sortie(self.sortie + 1)
        self.state, self.ticks, self.i, self.done_t = State.WAIT, 0, 0, None

    # --- входы ------------------------------------------------------------
    def on_continue(self, request, response):
        if self.state != State.DONE or not self.has_next():
            response.success = False
            response.message = (f"сейчас {self.state.value}, продолжать можно только после посадки" if self.state != State.DONE
                                else "это последний вылет серии" if self.series else "серия выключена (series:=false)")
        else:
            self.next_sortie()
            response.success, response.message = True, f"вылет {self.sortie + 1}/{len(self.sorties)}"
        return response

    def on_position(self, msg: VehicleLocalPosition):
        if msg.xy_valid and msg.z_valid:
            self.pos = px4_to_map((msg.x, msg.y, msg.z), self.dock)
            self.yaw = yaw_ned_to_enu(msg.heading)

    def on_status(self, msg: VehicleStatus):
        self.status = msg

    def on_safety(self, msg: String):
        if msg.data != self.safety:
            self.get_logger().warn(f"безопасность: {self.safety} -> {msg.data}")
        self.safety = msg.data

    # --- выходы -----------------------------------------------------------
    def now_us(self) -> int:
        """Метка времени для сообщений PX4: 0 = «время прихода». Узел живёт на времени симуляции
        (use_sim_time), а uXRCE-DDS пересчитывает ненулевые метки по системным часам агента
        (Tools/msg/templates/ucdr/msg.h.em, PX4 v1.17)."""
        return 0

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
        m.position = [float(v) for v in map_to_px4(xyz, self.dock)]
        m.velocity = [float(v) for v in enu_to_ned(vel)]           # feed-forward вдоль отрезка
        m.acceleration = [math.nan] * 3
        m.yaw = float(yaw_enu_to_ned(yaw_enu))
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
        # прогресс копится по тикам (а не по времени), чтобы STOP/SLOW не давали скачков уставки
        self.seg_from, self.seg_yaw0, self.hold_until = self.carrot, self.carrot_yaw, None
        self.seg_s, self.seg_yaw_s = 0.0, 0.0

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
                self.state, self.done_t = State.DONE, self.now_s()
                if self.has_next():
                    self.get_logger().info(
                        f"серия: вылет {self.sortie + 1}/{len(self.sorties)} завершён. Замените АКБ и выполните "
                        "ros2 service call /mission/continue std_srvs/srv/Trigger" if self.continue_after < 0 else
                        f"серия: следующий вылет через {self.continue_after:.0f} с (continue_after_s)")
                elif self.series:
                    self.get_logger().info(f"серия завершена: {len(self.sorties)} вылетов")

        elif self.state == State.DONE:
            # scan_recorder сохраняет CSV вылета по первому DONE; дальше — пауза на замену АКБ
            if self.has_next() and self.continue_after >= 0 and self.now_s() - self.done_t >= self.continue_after:
                self.next_sortie()

    def follow(self):
        target = self.wps[self.i]
        factor = {"STOP": 0.0, "SLOW": 0.5}.get(self.safety, 1.0)
        dt = 1.0 / self.rate
        a, b = self.seg_from, (target.x, target.y, target.z)
        length = math.dist(a, b)
        speed = self.segment_speed() * factor
        self.seg_s = min(length, self.seg_s + speed * dt)
        self.seg_yaw_s += math.radians(self.p.yaw_rate_deg) * factor * dt
        s = self.seg_s
        k = s / length if length > 1e-6 else 1.0
        self.carrot = tuple(a[j] + (b[j] - a[j]) * k for j in range(3))
        vel = tuple((b[j] - a[j]) / length * speed for j in range(3)) if 0 < s < length else (0.0, 0.0, 0.0)

        dyaw = wrap(target.yaw - self.seg_yaw0)
        step = self.seg_yaw_s
        self.carrot_yaw = target.yaw if abs(dyaw) <= step else self.seg_yaw0 + math.copysign(step, dyaw)
        self.setpoint(self.carrot, self.carrot_yaw, vel)
        if factor == 0.0:
            return  # STOP: держим точку, дальше по маршруту не идём

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

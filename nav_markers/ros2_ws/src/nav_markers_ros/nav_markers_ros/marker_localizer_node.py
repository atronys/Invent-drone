"""Локализация по ArUco-меткам -> EKF2 PX4 как внешнее зрение (документ, разд. 7.4: «быстрый EKF2 соединяет IMU,
поток и дальномер; узел по меткам передаёт исправленную позу»).

Вход:  кадры трёх камер меток (sensor_msgs/Image), /fmu/out/vehicle_odometry (априорная поза EKF2).
Выход: /fmu/in/vehicle_visual_odometry — поза по меткам каждого кадра (положение + курс с дисперсиями);
       /nav/marker_fix — то же в map (PoseWithCovarianceStamped) — для миссии и журнала;
       /nav/localizer_status — счётчики, последний отказ (JSON).

Кадры камер одного момента (Gazebo рендерит их синхронно, метки времени равны) обрабатываются вместе:
две метки по разные стороны прохода дают курс по базе между ними (nav_core.aruco_pose.MarkerLocalizer.fuse).

Дисперсия = граница Крамера–Рао (шум углов) + нижняя граница fix_sigma_floor: без неё EKF2 верил метке
в 7 раз больше, чем она того стоила (NEES 57, прогон SITL 12.10.2026).

Пробовали и отклонили: свой фильтр, который прогнозирует позу по скорости EKF2 и отдаёт её в EKF2 20 Гц
непрерывно (чтобы EKF2 не считал редкие метки «прерывистыми» и не сбрасывал позицию). Это петля обратной
связи: поза фильтра подтягивает скорость EKF2, фильтр интегрирует подтянутую скорость — оценка уходила
на метры (прогоны 6–10, README). Сбросы EKF2 на точную позу по метке безопаснее.

«Поза потеряна» — пока нет ни одной принятой позы по меткам, при большой дисперсии EKF2 и пока курс EKF2
расходится с курсом по меткам (до выравнивания курса без магнитометра): тогда принимаем только уникальные
якорные метки, а курс берём из PnP.

OpenCV ≥ 4.7 обязателен (в 4.6 Ubuntu 24.04 детектор теряет метки на рендере Gazebo): PYTHONPATH=~/nav_pylib
(nav_markers/gazebo/run_nav_sitl.sh).
"""
import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from px4_msgs.msg import VehicleOdometry

from invent_core.camera import Camera
from invent_core.frames import euler_zyx, map_pose_to_px4, matrix_from_quat, px4_pose_to_map, quat_from_matrix
from invent_core.layout import Warehouse
from invent_core.localization import MountedCamera
from nav_core.aruco_pose import MarkerLocalizer, Observation, Prior, make_detector, yaw_of
from nav_core.markers import MarkerMap

CHI2_3DOF_999 = 16.27


def image_to_gray(msg: Image):
    import cv2
    ch = {"mono8": 1, "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(msg.encoding)
    if ch is None:
        return None
    a = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch]
    if ch == 1:
        return a.reshape(msg.height, msg.width)
    a = a.reshape(msg.height, msg.width, ch)
    code = {"rgb8": cv2.COLOR_RGB2GRAY, "bgr8": cv2.COLOR_BGR2GRAY, "rgba8": cv2.COLOR_RGBA2GRAY,
            "bgra8": cv2.COLOR_BGRA2GRAY}[msg.encoding]
    return cv2.cvtColor(a, code)


class MarkerLocalizerNode(Node):
    def __init__(self):
        super().__init__("marker_localizer")
        for name, default in [("layout", ""), ("marker_map", ""), ("model_dir", ""), ("sigma_px", 1.0),
                              ("max_range", 4.0), ("corner_bias_px", 0.0), ("lost_var", 1.0), ("lost_yaw_deg", 10.0),
                              ("fix_sigma_floor", 0.03), ("fix_yaw_floor_deg", 0.5), ("forward_to_px4", True)]:
            self.declare_parameter(name, default)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.dock = Warehouse.load(g("layout")).dock
        self.mm = MarkerMap.load(g("marker_map"))
        model_dir = g("model_dir").rstrip("/")
        with open(f"{model_dir}/mounts.json", encoding="utf-8") as f:
            cams = json.load(f)["cameras"]
        mounted = {m["look"]: MountedCamera.from_mount(Camera.load(f"{model_dir}/{m['camera_info']}"), m["xyz"], m["rpy"])
                   for m in cams}
        self.loc = MarkerLocalizer(self.mm, mounted, sigma_px=g("sigma_px"), max_range=g("max_range"),
                                   corner_bias_px=g("corner_bias_px"))
        self.detect = make_detector(self.mm.dictionary)
        self.lost_var, self.lost_yaw = g("lost_var"), math.radians(g("lost_yaw_deg"))
        self.floor_p, self.floor_yaw = g("fix_sigma_floor"), math.radians(g("fix_yaw_floor_deg"))
        self.forward = g("forward_to_px4")

        self.odom = None            # (p_map, r_wb, cov_p, var_yaw)
        self.n_fix = 0
        self.last_fix_yaw = None
        self.last_reject = ""
        self.latency = []           # задержка «кадр -> поза», с времени симуляции
        self.pending: dict[int, dict] = {}     # метка времени кадра -> {камера: детекции}
        self.n_cams = len(cams)
        self.stats = {"frames": 0, "detections": 0, "accepted": 0, "fixes_rejected": 0, "rejected": {}}

        px4_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_ev = self.create_publisher(VehicleOdometry, "/fmu/in/vehicle_visual_odometry", px4_qos)
        self.pub_fix = self.create_publisher(PoseWithCovarianceStamped, "/nav/marker_fix", 10)
        self.pub_status = self.create_publisher(String, "/nav/localizer_status", 10)
        self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self.on_odom, px4_qos)
        for m in cams:
            self.create_subscription(Image, m["topic"], lambda msg, look=m["look"]: self.on_image(look, msg), 5)
        self.create_timer(1.0, self.publish_status)
        self.get_logger().info(f"меток в карте {len(self.mm.markers)}, камер {self.n_cams}")

    # --- входы -----------------------------------------------------------------------------------------
    def on_odom(self, msg: VehicleOdometry):
        if not all(math.isfinite(v) for v in msg.position) or not all(math.isfinite(v) for v in msg.q):
            return
        p, q = px4_pose_to_map(tuple(msg.position), tuple(msg.q), self.dock)
        var = [v if math.isfinite(v) else 1e3 for v in msg.position_variance]
        cov = np.diag([var[1], var[0], var[2]])                     # NED -> ENU
        var_yaw = msg.orientation_variance[2] if math.isfinite(msg.orientation_variance[2]) else math.radians(5) ** 2
        self.odom = (np.array(p), np.array(matrix_from_quat(*q)), cov, var_yaw)

    def prior(self) -> Prior:
        if self.odom is None:
            return Prior(np.array(self.dock, dtype=float), np.eye(3) * 100.0, np.eye(3), lost=True)
        p, r_wb, cov, var_yaw = self.odom
        lost = self.n_fix == 0 or cov[0, 0] + cov[1, 1] > self.lost_var
        if self.last_fix_yaw is not None:
            d = math.atan2(math.sin(yaw_of(r_wb) - self.last_fix_yaw), math.cos(yaw_of(r_wb) - self.last_fix_yaw))
            lost = lost or abs(d) > self.lost_yaw          # курс EKF2 ещё не выровнен по меткам
        # bool(): numpy.bool_ не сериализуется в JSON статуса
        return Prior(p, cov, r_wb, max(var_yaw, math.radians(1.0) ** 2), bool(lost))

    def on_image(self, look: str, msg: Image):
        stamp = msg.header.stamp.sec * 10 ** 9 + msg.header.stamp.nanosec
        gray = image_to_gray(msg)
        if gray is None:
            self.get_logger().warn(f"кодировка {msg.encoding} не поддерживается", once=True)
            return
        dets = self.detect(gray)
        self.stats["frames"] += 1
        self.stats["detections"] += len(dets)
        self.pending.setdefault(stamp, {})[look] = dets
        for s in sorted(self.pending):            # кадры, которые уже не дополнятся, — обработать
            if s < stamp or len(self.pending[s]) == self.n_cams:
                self.process(s, self.pending.pop(s), msg.header.stamp)

    # --- обработка -------------------------------------------------------------------------------------
    def process(self, stamp_ns: int, frame: dict, ros_stamp):
        prior = self.prior()
        obs = []
        for look, dets in frame.items():
            for mid, corners in dets:
                o = self.loc.observe(look, mid, corners, prior)
                if isinstance(o, Observation):
                    obs.append(o)
                else:
                    key = o.reason.split(" ")[0]
                    self.stats["rejected"][key] = self.stats["rejected"].get(key, 0) + 1
                    self.last_reject = f"{look} id{mid}: {o.reason}"
        if not obs:
            return
        fix = self.loc.fuse(obs, prior)
        if fix is None or not np.all(np.isfinite(fix.p)):
            return
        cov = fix.cov_p + np.eye(3) * self.floor_p ** 2
        var_yaw = fix.var_yaw + self.floor_yaw ** 2
        if not prior.lost:                         # совместная поза тоже проходит χ² против EKF2
            e = fix.p - prior.p
            if e @ np.linalg.solve(prior.cov_p + cov, e) > CHI2_3DOF_999:
                self.stats["fixes_rejected"] += 1
                self.last_reject = f"поза кадра: не согласуется с EKF2 ({100 * np.linalg.norm(e[:2]):.0f} см)"
                return
        self.n_fix += 1
        self.last_fix_yaw = fix.yaw
        self.stats["accepted"] += len(obs)
        self.latency.append(self.get_clock().now().nanoseconds * 1e-9 - stamp_ns * 1e-9)
        self.publish_fix(fix, cov, var_yaw, prior, ros_stamp)

    def publish_fix(self, fix, cov_p, var_yaw, prior: Prior, ros_stamp):
        tilt = np.array(euler_zyx(-yaw_of(prior.r_wb))) @ prior.r_wb     # крен и тангаж — от EKF2, курс — по меткам
        q_map = quat_from_matrix(tuple(map(tuple, np.array(euler_zyx(fix.yaw)) @ tilt)))
        m = PoseWithCovarianceStamped()
        m.header.stamp, m.header.frame_id = ros_stamp, "map"
        m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = (float(v) for v in fix.p)
        (m.pose.pose.orientation.w, m.pose.pose.orientation.x,
         m.pose.pose.orientation.y, m.pose.pose.orientation.z) = (float(v) for v in q_map)
        cov = np.zeros((6, 6))
        cov[:3, :3] = cov_p
        cov[5, 5] = var_yaw
        m.pose.covariance = [float(v) for v in cov.ravel()]
        self.pub_fix.publish(m)
        if not self.forward:
            return
        pos, q = map_pose_to_px4(tuple(float(v) for v in fix.p), q_map, self.dock)
        ev = VehicleOdometry()
        # 0 = «время прихода»: узел на времени симуляции, а uXRCE-DDS пересчитывает ненулевые метки по системным
        # часам агента (так же сделано в INVENT_DRONE, landmark_localizer_node). Задержку учитывает EKF2_EV_DELAY.
        ev.timestamp = ev.timestamp_sample = 0
        ev.pose_frame = VehicleOdometry.POSE_FRAME_NED
        ev.position = [float(v) for v in pos]
        ev.q = [float(v) for v in q]
        ev.velocity_frame = VehicleOdometry.VELOCITY_FRAME_UNKNOWN
        ev.velocity = [math.nan] * 3
        ev.angular_velocity = [math.nan] * 3
        ev.position_variance = [float(cov_p[1, 1]), float(cov_p[0, 0]), float(cov_p[2, 2])]      # ENU -> NED
        ev.orientation_variance = [float(var_yaw)] * 3
        ev.velocity_variance = [math.nan] * 3
        ev.quality = 100
        self.pub_ev.publish(ev)

    def publish_status(self):
        s = dict(self.stats, fixes=self.n_fix, lost=bool(self.prior().lost), last_reject=self.last_reject)
        if self.latency:
            s["latency_ms"] = round(1000 * float(np.median(self.latency[-50:])), 1)
        self.pub_status.publish(String(data=json.dumps(s, ensure_ascii=False)))


def main():
    rclpy.init()
    node = MarkerLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

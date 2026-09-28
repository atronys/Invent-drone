"""LOC-11: локализация по адресным этикеткам поверх VIO.

Вход:  одометрия VIO (nav_msgs/Odometry в кадре odom, например OpenVINS) и LabelRead от label_reader.
Логика: `invent_core.localization.LandmarkLocalizer` — прогноз по приращениям VIO, обновление по
пикселю центра прочитанной адресной этикетки с известными координатами (карта = раскладка склада).
Выход: nav_msgs/Odometry в map (`localization/odom`), TF map->odom (поправка к VIO) и, по желанию,
поза в PX4 как внешняя навигация (/fmu/in/vehicle_visual_odometry, задача LOC-04).

Старт: дрон стоит на доке (известная точка раскладки), курс — параметр initial_yaw_deg (0 — вдоль +x).
Запуск: ros2 run invent_localization landmark_localizer --ros-args -p layout:=... -p mounts_file:=...
        -p camera_info_left:=... -p camera_info_right:=... -r vio_odom:=/ov_msckf/odomimu
"""
import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster

from invent_core.camera import Camera
from invent_core.frames import euler_zyx, map_pose_to_px4, matrix_from_quat, quat_from_matrix
from invent_core.layout import Warehouse
from invent_core.localization import LandmarkLocalizer, MountedCamera, address_landmarks
from invent_msgs.msg import LabelRead
from px4_msgs.msg import VehicleOdometry


def yaw_roll_pitch(q) -> tuple[float, float, float]:
    r = matrix_from_quat(q.w, q.x, q.y, q.z)
    yaw = math.atan2(r[1][0], r[0][0])
    pitch = math.asin(max(-1.0, min(1.0, -r[2][0])))
    roll = math.atan2(r[2][1], r[2][2])
    return yaw, roll, pitch


class LandmarkLocalizerNode(Node):
    def __init__(self):
        super().__init__("landmark_localizer")
        for name, default in [("layout", ""), ("mounts_file", ""), ("camera_info_left", ""), ("camera_info_right", ""),
                              ("map_frame", "map"), ("odom_frame", "odom"), ("initial_yaw_deg", 0.0),
                              ("sigma_px", 2.0), ("publish_tf", True), ("forward_to_px4", False)]:
            self.declare_parameter(name, default)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.wh = Warehouse.load(g("layout"))
        self.landmarks = address_landmarks(self.wh)
        self.mounts = {}   # frame_id кадра камеры -> MountedCamera
        with open(g("mounts_file"), encoding="utf-8") as f:
            for m in json.load(f)["cameras"]:
                cam_file = g(f"camera_info_{m['look']}")
                cam = Camera.load(cam_file) if cam_file else Camera()
                self.mounts[m["frame"]] = MountedCamera.from_mount(cam, m["xyz"], m["rpy"])
        self.map_frame, self.odom_frame = g("map_frame"), g("odom_frame")
        self.sigma_px, self.publish_tf, self.forward = g("sigma_px"), g("publish_tf"), g("forward_to_px4")
        self.ekf = None
        self.prev = None          # (p_odom np(3), yaw_odom)
        self.roll = self.pitch = 0.0
        self.last_odom = None

        self.tf = TransformBroadcaster(self)
        self.pub = self.create_publisher(Odometry, "localization/odom", 10)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.px4 = self.create_publisher(VehicleOdometry, "/fmu/in/vehicle_visual_odometry", qos) if self.forward else None
        self.create_subscription(Odometry, "vio_odom", self.on_odom, 50)
        self.create_subscription(LabelRead, "labels", self.on_label, 100)

    def on_odom(self, msg: Odometry):
        p = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        yaw, self.roll, self.pitch = yaw_roll_pitch(msg.pose.pose.orientation)
        self.last_odom = (p, yaw, msg.header.stamp)
        if self.ekf is None:
            dock = self.wh.dock
            self.ekf = LandmarkLocalizer((dock[0], dock[1], dock[2] + p[2], math.radians(self.get_parameter(
                "initial_yaw_deg").value)))
            self.prev = (p, yaw)
            return
        dp = p - self.prev[0]
        c, s = math.cos(self.prev[1]), math.sin(self.prev[1])
        d_body = (c * dp[0] + s * dp[1], -s * dp[0] + c * dp[1])        # приращение в кадре тела по курсу VIO
        dpsi = math.atan2(math.sin(yaw - self.prev[1]), math.cos(yaw - self.prev[1]))
        step = math.hypot(*d_body)
        self.ekf.predict(d_body, float(dp[2]), dpsi, sigma_xy=0.03 * step + 0.002, sigma_z=0.01 * abs(dp[2]) + 0.002,
                         sigma_psi=math.radians(0.03))
        self.prev = (p, yaw)
        self.publish(msg.header.stamp)

    def on_label(self, msg: LabelRead):
        if self.ekf is None:
            return
        mc = self.mounts.get(msg.header.frame_id)
        L = self.landmarks.get(msg.text)          # адресная этикетка: текст = адрес ячейки
        if mc is None or L is None or len(msg.corners) != 4:
            return
        u = sum(c.x for c in msg.corners) / 4
        v = sum(c.y for c in msg.corners) / 4
        xn, yn = mc.cam.pixel_to_normalized(u, v)                # снять дисторсию: модель наблюдения — пинхол
        uv = (mc.cam.fx * xn + mc.cam.cx, mc.cam.fy * yn + mc.cam.cy)
        if not self.ekf.update_pixel(uv, L, mc, self.sigma_px, self.roll, self.pitch):
            self.get_logger().debug(f"отброшено наблюдение {msg.text}")

    def publish(self, stamp):
        x, y, z, psi = self.ekf.mu
        q_map = quat_from_matrix(euler_zyx(psi, self.pitch, self.roll))
        od = Odometry()
        od.header.stamp, od.header.frame_id, od.child_frame_id = stamp, self.map_frame, "base_link"
        od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = float(x), float(y), float(z)
        (od.pose.pose.orientation.w, od.pose.pose.orientation.x,
         od.pose.pose.orientation.y, od.pose.pose.orientation.z) = (float(v) for v in q_map)
        cov = np.zeros((6, 6))
        cov[:3, :3] = self.ekf.P[:3, :3]
        cov[5, 5] = self.ekf.P[3, 3]
        od.pose.covariance = [float(v) for v in cov.ravel()]
        self.pub.publish(od)
        if self.publish_tf and self.last_odom is not None:
            # map->odom = T_map_base · T_odom_base⁻¹ (плоская поправка: курс + смещение)
            p_o, yaw_o, _ = self.last_odom
            dyaw = psi - yaw_o
            c, s = math.cos(dyaw), math.sin(dyaw)
            tx = x - (c * p_o[0] - s * p_o[1])
            ty = y - (s * p_o[0] + c * p_o[1])
            t = TransformStamped()
            t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, self.map_frame, self.odom_frame
            t.transform.translation.x, t.transform.translation.y = float(tx), float(ty)
            t.transform.translation.z = float(z - p_o[2])
            t.transform.rotation.z, t.transform.rotation.w = math.sin(dyaw / 2), math.cos(dyaw / 2)
            self.tf.sendTransform(t)
        if self.px4 is not None:
            pos, q = map_pose_to_px4((x, y, z), q_map, self.wh.dock)
            m = VehicleOdometry()
            # 0 = «время прихода в PX4». Иначе uXRCE-DDS вычитает смещение timesync, посчитанное по системным
            # часам агента (Tools/msg/templates/ucdr/msg.h.em, v1.17), а узел в SITL живёт на времени симуляции.
            # Задержку наблюдения учитывает EKF2_EV_DELAY.
            m.timestamp = m.timestamp_sample = 0
            m.pose_frame = VehicleOdometry.POSE_FRAME_NED
            m.position = [float(v) for v in pos]
            m.q = [float(v) for v in q]
            m.velocity_frame = VehicleOdometry.VELOCITY_FRAME_UNKNOWN
            m.velocity = [math.nan] * 3
            m.angular_velocity = [math.nan] * 3
            m.position_variance = [float(self.ekf.P[1, 1]), float(self.ekf.P[0, 0]), float(self.ekf.P[2, 2])]
            m.orientation_variance = [math.nan, math.nan, float(self.ekf.P[3, 3])]
            m.velocity_variance = [math.nan] * 3
            self.px4.publish(m)


def main():
    rclpy.init()
    node = LandmarkLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

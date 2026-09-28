"""SIM-05: «фальшивый VIO» для SITL — истинная поза из Gazebo + дрейф как у визуально-инерциальной одометрии.

Истинная поза дрона — плагин PosePublisher модели (sim/make_drone_model.py) -> мост ros_gz -> /gz/true_pose
(tf2_msgs/TFMessage, child_frame_id = имя модели, время симуляции). Узел берёт позу дрона, считает истинное приращение в кадре
тела, портит его ошибкой масштаба, дрейфом курса и шумом (модель как в research/landmark_ekf) и
интегрирует в кадре odom. Выход — nav_msgs/Odometry (`vio_odom`) для landmark_localizer; TF не публикует.

Запуск: ros2 run invent_localization fake_vio --ros-args -p model_name:=x500_invent_0 -p scale_err:=0.02
"""
import math
import random

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

from invent_core.frames import euler_zyx, matrix_from_quat, quat_from_matrix


class FakeVio(Node):
    def __init__(self):
        super().__init__("fake_vio")
        for name, default in [("model_name", "x500_invent_0"), ("scale_err", 0.01), ("yaw_bias_deg_min", 0.3),
                              ("trans_noise", 0.02), ("yaw_noise_deg", 0.01), ("seed", 0),
                              ("odom_frame", "odom"), ("base_frame", "base_link")]:
            self.declare_parameter(name, default)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.rng = random.Random(g("seed"))
        self.model = g("model_name")
        self.scale = 1.0 + self.rng.gauss(0.0, g("scale_err"))           # на весь полёт, как у реального VIO
        self.yaw_bias = math.radians(self.rng.gauss(0.0, g("yaw_bias_deg_min"))) / 60.0   # рад/с
        self.trans_noise, self.yaw_noise = g("trans_noise"), math.radians(g("yaw_noise_deg"))
        self.frames = (g("odom_frame"), g("base_frame"))
        self.prev = None          # (t, x, y, z, yaw, roll, pitch) истина
        self.est = None           # (x, y, z, yaw) в odom
        self.pub = self.create_publisher(Odometry, "vio_odom", 50)
        self.create_subscription(TFMessage, "/gz/true_pose", self.on_poses, 50)
        self.warned = False
        self.get_logger().info(f"масштаб {self.scale:.4f}, дрейф курса {math.degrees(self.yaw_bias) * 60:.2f}°/мин")

    def on_poses(self, msg: TFMessage):
        tr = next((t for t in msg.transforms if t.child_frame_id == self.model), None)
        if tr is None:
            if not self.warned:
                self.warned = True
                self.get_logger().warn(f"в /gz/true_pose нет {self.model}: есть "
                                       f"{[t.child_frame_id for t in msg.transforms]} — задайте model_name")
            return
        q = tr.transform.rotation
        r = matrix_from_quat(q.w, q.x, q.y, q.z)
        yaw = math.atan2(r[1][0], r[0][0])
        pitch = math.asin(max(-1.0, min(1.0, -r[2][0])))
        roll = math.atan2(r[2][1], r[2][2])
        p = tr.transform.translation
        t = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9
        cur = (t, p.x, p.y, p.z, yaw, roll, pitch)
        if self.prev is None:
            self.prev, self.est = cur, [0.0, 0.0, p.z, 0.0]   # odom: начало в точке старта, курс 0
            return
        dt = max(0.0, t - self.prev[0])
        dx, dy, dz = p.x - self.prev[1], p.y - self.prev[2], p.z - self.prev[3]
        c, s = math.cos(self.prev[4]), math.sin(self.prev[4])
        bx, by = c * dx + s * dy, -s * dx + c * dy                 # истинное приращение в кадре тела
        step = math.hypot(bx, by)
        bx = self.scale * bx + self.rng.gauss(0.0, self.trans_noise * step + 1e-4)
        by = self.scale * by + self.rng.gauss(0.0, self.trans_noise * step + 1e-4)
        dyaw = math.atan2(math.sin(yaw - self.prev[4]), math.cos(yaw - self.prev[4]))
        dyaw += self.yaw_bias * dt + self.rng.gauss(0.0, self.yaw_noise)
        ce, se = math.cos(self.est[3]), math.sin(self.est[3])
        self.est = [self.est[0] + ce * bx - se * by, self.est[1] + se * bx + ce * by,
                    self.est[2] + dz * self.scale, self.est[3] + dyaw]
        self.prev = cur
        od = Odometry()
        od.header.stamp = tr.header.stamp
        od.header.frame_id, od.child_frame_id = self.frames
        od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = self.est[:3]
        w, x, y, z = quat_from_matrix(euler_zyx(self.est[3], pitch, roll))
        od.pose.pose.orientation.w, od.pose.pose.orientation.x = w, x
        od.pose.pose.orientation.y, od.pose.pose.orientation.z = y, z
        self.pub.publish(od)


def main():
    rclpy.init()
    node = FakeVio()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

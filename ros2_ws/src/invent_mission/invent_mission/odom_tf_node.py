"""TF для привязки этикеток: поза PX4 -> map->base_link, установка камер -> base_link->камера.

* /fmu/out/vehicle_odometry (PX4 v1.17: версия сообщения 0, без суффикса): позиция NED и q (FRD->NED)
  -> `invent_core.frames.px4_pose_to_map` -> TF map->base_link (ENU/FLU, REP-103).
  Начало локального NED PX4 — док (точка инициализации EKF), поэтому нужна раскладка склада.
* mounts.json (генерирует sim/make_drone_model.py; для реального дрона — по чертежу креплений)
  -> статические TF base_link->label_cam_*_link.

Запуск: ros2 run invent_mission odom_tf --ros-args -p layout:=... -p mounts_file:=.../mounts.json
"""
import json

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from invent_core.frames import euler_zyx, px4_pose_to_map, quat_from_matrix
from invent_core.layout import Warehouse
from px4_msgs.msg import VehicleOdometry


def make_tf(stamp, parent: str, child: str, xyz, q_wxyz) -> TransformStamped:
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (float(v) for v in xyz)
    w, x, y, z = q_wxyz
    t.transform.rotation.w, t.transform.rotation.x = float(w), float(x)
    t.transform.rotation.y, t.transform.rotation.z = float(y), float(z)
    return t


class OdomTf(Node):
    def __init__(self):
        super().__init__("odom_tf")
        self.declare_parameter("layout", "")
        self.declare_parameter("mounts_file", "")
        self.declare_parameter("odometry_topic", "/fmu/out/vehicle_odometry")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        layout = self.get_parameter("layout").value
        self.dock = Warehouse.load(layout).dock if layout else (0.0, 0.0, 0.0)
        if not layout:
            self.get_logger().warn("layout не задан: считаю, что док в начале map (0, 0, 0)")

        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        mounts_file = self.get_parameter("mounts_file").value
        if mounts_file:
            self.publish_mounts(mounts_file)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(VehicleOdometry, self.get_parameter("odometry_topic").value, self.on_odom, qos)

    def publish_mounts(self, path: str):
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        now = self.get_clock().now().to_msg()
        tfs = []
        for cam in m["cameras"]:
            roll, pitch, yaw = cam["rpy"]
            q = quat_from_matrix(euler_zyx(yaw, pitch, roll))  # rpy SDF: R = Rz·Ry·Rx
            tfs.append(make_tf(now, m.get("base_frame", self.base_frame), cam["frame"], cam["xyz"], q))
        self.static_tf.sendTransform(tfs)
        self.get_logger().info(f"статические TF камер: {[t.child_frame_id for t in tfs]}")

    def on_odom(self, msg: VehicleOdometry):
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED or msg.q[0] != msg.q[0]:  # NaN -> поза невалидна
            return
        p, q = px4_pose_to_map(tuple(msg.position), tuple(msg.q), self.dock)
        self.tf.sendTransform(make_tf(self.get_clock().now().to_msg(), self.map_frame, self.base_frame, p, q))


def main():
    rclpy.init()
    node = OdomTf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

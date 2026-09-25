"""SAFE-01/02: пузырь безопасности по фронтальной depth-камере + геозона -> /safety/state (GO/SLOW/STOP).

Логика — `invent_core.safety` (протестирована в core/tests/test_safety.py). Узел миссии на STOP
замирает на месте, на SLOW — вдвое снижает скорость.

Запуск: ros2 run invent_mission safety_monitor --ros-args -p layout:=... -r depth:=/depth_camera
"""
import math

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

from invent_core.camera import Camera
from invent_core.frames import px4_to_map
from invent_core.layout import Warehouse
from invent_core.safety import STOP, Geofence, SafetyMonitor, corridor_min_distance
from px4_msgs.msg import VehicleLocalPosition


class SafetyNode(Node):
    def __init__(self):
        super().__init__("safety_monitor")
        self.declare_parameter("layout", "")
        self.declare_parameter("depth_hfov_deg", 73.0)     # OakD-Lite в модели PX4: 1.274 рад
        self.declare_parameter("corridor_half_width", 0.45)
        self.declare_parameter("corridor_half_height", 0.35)
        self.declare_parameter("local_position_topic", "/fmu/out/vehicle_local_position_v1")
        wh = Warehouse.load(self.get_parameter("layout").value)
        self.dock = wh.dock
        self.geofence = Geofence(wh)
        self.monitor = SafetyMonitor()
        self.bridge = CvBridge()
        self.cam = None
        self.speed = 0.0
        self.outside = False
        self.pub_state = self.create_publisher(String, "safety/state", 10)
        self.pub_dist = self.create_publisher(Float32, "safety/min_distance", 10)
        self.create_subscription(Image, "depth", self.on_depth, qos_profile_sensor_data)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(VehicleLocalPosition, self.get_parameter("local_position_topic").value,
                                 self.on_position, qos)

    def on_position(self, msg: VehicleLocalPosition):
        self.speed = math.hypot(msg.vx, msg.vy)
        if msg.xy_valid and msg.z_valid:
            p = px4_to_map((msg.x, msg.y, msg.z), self.dock)
            outside = not self.geofence.contains(p)
            if outside and not self.outside:
                self.get_logger().error(f"выход из геозоны: {tuple(round(v, 2) for v in p)} — STOP")
            self.outside = outside

    def on_depth(self, msg: Image):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if msg.encoding == "16UC1":     # реальные камеры часто отдают миллиметры
            depth = depth.astype("float32") / 1000.0
        if self.cam is None:
            hfov = self.get_parameter("depth_hfov_deg").value
            fx = msg.width / (2 * math.tan(math.radians(hfov) / 2))
            self.cam = Camera.from_intrinsics(msg.width, msg.height, fx, fx, msg.width / 2, msg.height / 2)
        d = corridor_min_distance(depth, self.cam, self.get_parameter("corridor_half_width").value,
                                  self.get_parameter("corridor_half_height").value)
        state = STOP if self.outside else self.monitor.update(d, self.speed)
        self.pub_dist.publish(Float32(data=float(min(d, 1e3))))
        self.pub_state.publish(String(data=state))


def main():
    rclpy.init()
    node = SafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

"""Узел чтения этикеток: кадр -> zxing-cpp -> invent_msgs/LabelRead (+ SSCC).

Привязка к адресу ячейки (invent_core.association.locate_label по позе камеры из TF
в момент кадра) — задача PER-06. Пока поле address пустое, а в лог пишутся адреса,
которые план ожидает в кадре (из mission/status) — этого хватает для первых проверок.

Запуск: ros2 run invent_perception label_reader --ros-args -r image:=/camera/left/image_raw
"""
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from invent_core import barcode
from invent_msgs.msg import LabelRead, MissionStatus


class LabelReader(Node):
    def __init__(self):
        super().__init__("label_reader")
        self.declare_parameter("every_nth", 1)  # декодировать каждый N-й кадр (экономия CPU)
        self.bridge = CvBridge()
        self.pub = self.create_publisher(LabelRead, "labels", 50)
        self.create_subscription(Image, "image", self.on_image, qos_profile_sensor_data)
        self.create_subscription(MissionStatus, "mission/status", self.on_status, 10)
        self.targets: list[str] = []
        self.frames = 0
        self.unique: set[str] = set()

    def on_status(self, msg: MissionStatus):
        self.targets = list(msg.targets)

    def on_image(self, msg: Image):
        self.frames += 1
        if self.frames % self.get_parameter("every_nth").value:
            return
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        for r in barcode.decode(img):
            out = LabelRead(text=r.text, symbology=r.symbology, sscc=r.sscc or "", address="", confidence=1.0)
            out.header = msg.header
            out.corners = [Point(x=u, y=v, z=0.0) for u, v in r.corners]
            self.pub.publish(out)
            if r.text not in self.unique:
                self.unique.add(r.text)
                self.get_logger().info(f"{r.symbology}: {r.text} (ожидались в кадре: {self.targets[:6]})")


def main():
    rclpy.init()
    node = LabelReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

"""Узел чтения этикеток: кадр -> zxing-cpp -> invent_msgs/LabelRead (+ SSCC, + адрес ячейки).

Адрес (PER-06): поза камеры из TF (map <- кадр камеры) в момент кадра -> обратная проекция
центра этикетки на фасад -> `invent_core.association.locate_in_warehouse`.
Кадр камеры должен быть оптическим (z вперёд, x вправо, y вниз — как в книге Gao & Zhang, гл. 4).
Камеры Gazebo часто публикуют кадр звена (x вперёд) — тогда frame_is_optical:=false.

Запуск: ros2 run invent_perception label_reader --ros-args -r image:=/camera/left/image_raw \
            -p layout:=/path/warehouse.json -p camera_info:=/path/left.yaml
"""
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformException, TransformListener

from invent_core import barcode
from invent_core.pnp import unread_label_addresses
from invent_core.association import locate_in_warehouse
from invent_core.camera import Camera
from invent_core.frames import R_LINK_OPTICAL, matmul, matrix_from_quat
from invent_core.layout import Warehouse
from invent_msgs.msg import LabelRead


class LabelReader(Node):
    def __init__(self):
        super().__init__("label_reader")
        self.declare_parameter("every_nth", 1)          # декодировать каждый N-й кадр (экономия CPU)
        self.declare_parameter("layout", "")            # без раскладки адрес не заполняется
        self.declare_parameter("camera_info", "")       # YAML калибровки; пусто — Camera() по умолчанию
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("frame_is_optical", True)
        self.declare_parameter("tf_timeout", 0.05)
        self.declare_parameter("save_frames_dir", "")   # отладка чтения: сохранять кадры (пусто — нет)
        self.declare_parameter("save_every", 1)

        layout = self.get_parameter("layout").value
        self.wh = Warehouse.load(layout) if layout else None
        cam_file = self.get_parameter("camera_info").value
        self.cam = Camera.load(cam_file) if cam_file else Camera()
        self.map_frame = self.get_parameter("map_frame").value
        self.optical = self.get_parameter("frame_is_optical").value
        self.tf_timeout = Duration(seconds=self.get_parameter("tf_timeout").value)

        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.bridge = CvBridge()
        self.pub = self.create_publisher(LabelRead, "labels", 50)
        self.create_subscription(Image, "image", self.on_image, qos_profile_sensor_data)
        self.frames = 0
        self.unique: set[str] = set()
        if self.wh is None:
            self.get_logger().warn("layout не задан: адреса ячеек заполняться не будут")

    def camera_pose(self, msg: Image):
        """(R_map_cam, t_map_cam) в момент кадра или None."""
        try:
            tr = self.tf.lookup_transform(self.map_frame, msg.header.frame_id, Time.from_msg(msg.header.stamp),
                                          timeout=self.tf_timeout).transform
        except TransformException as e:
            self.get_logger().warn(f"нет TF {self.map_frame}<-{msg.header.frame_id}: {e}", throttle_duration_sec=5.0)
            return None
        q = tr.rotation
        r = matrix_from_quat(q.w, q.x, q.y, q.z)
        if not self.optical:
            r = matmul(r, R_LINK_OPTICAL)
        return r, (tr.translation.x, tr.translation.y, tr.translation.z)

    def save_frame(self, msg: Image, img, reads):
        out = self.get_parameter("save_frames_dir").value
        if not out or (self.frames // self.get_parameter("every_nth").value) % self.get_parameter("save_every").value:
            return
        import pathlib

        import cv2
        d = pathlib.Path(out)
        d.mkdir(parents=True, exist_ok=True)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        cv2.imwrite(str(d / f"{self.get_name()}_{t:09.2f}_reads{len(reads)}.png"), img)

    def on_image(self, msg: Image):
        self.frames += 1
        if self.frames % self.get_parameter("every_nth").value:
            return
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        reads = barcode.decode(img)
        self.save_frame(msg, img, reads)
        pose = self.camera_pose(msg) if self.wh else None
        if pose:  # PER-09: этикетка видна, но не прочитана -> пустой text с адресом (scan_recorder: «занято»)
            for addr in unread_label_addresses(img, reads, self.cam, pose[0], pose[1], self.wh):
                out = LabelRead(text="", symbology="", sscc="", address=addr, confidence=0.0)
                out.header = msg.header
                self.pub.publish(out)
        if not reads:
            return
        for r in reads:
            address = ""
            if pose:
                u, v = r.center
                address = locate_in_warehouse(u, v, self.cam, pose[0], pose[1], self.wh) or ""
            out = LabelRead(text=r.text, symbology=r.symbology, sscc=r.sscc or "", address=address, confidence=1.0)
            out.header = msg.header
            out.corners = [Point(x=u, y=v, z=0.0) for u, v in r.corners]
            self.pub.publish(out)
            if r.text not in self.unique:
                self.unique.add(r.text)
                self.get_logger().info(f"{r.symbology}: {r.text} -> {address or 'адрес не определён'}")


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

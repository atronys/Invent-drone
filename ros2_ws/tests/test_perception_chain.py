"""PER-06: цепочка кадров без ROS и Gazebo — одометрия PX4 -> odom_tf -> TF -> label_reader -> адрес ячейки.

Кадр рендерится виртуальным полётом (sim/virtual_flight.py) из известной позы дрона; та же поза уходит
в odom_tf как VehicleOdometry (NED/FRD), установка камер — из mounts.json генератора модели
(sim/make_drone_model.py), как в SITL. Если соглашения кадров где-то расходятся (ENU/NED, FLU/FRD,
кадр звена камеры/оптический), SSCC привяжутся не к тем ячейкам.
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ros_stubs as rs  # noqa: E402

try:
    import barcode  # noqa: F401  (python-barcode — текстуры этикеток)
    import cv2  # noqa: F401
    import numpy as np
    import zxingcpp  # noqa: F401
    from PIL import Image  # noqa: F401
    DEPS = True
except ImportError:
    DEPS = False

SRC = ROOT / "ros2_ws" / "src"
PKGS = [str(SRC / "invent_mission"), str(SRC / "invent_perception")]


@unittest.skipUnless(DEPS, "нужны numpy, opencv, zxing-cpp, python-barcode, pillow (requirements-dev.txt)")
class PerceptionChainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import generate_world as gw
        import make_drone_model as mdm
        import virtual_flight as vf
        from invent_core.camera import Camera
        from invent_core.coverage import ScanParams, plan
        from invent_core.layout import make_grid_warehouse

        cls.tmp = tempfile.TemporaryDirectory()
        out = Path(cls.tmp.name)
        (out / "labels").mkdir()
        cls.wh = make_grid_warehouse(n_aisles=1, bays=2, levels=2)
        _, cls.truth, _, _, ok = gw.build(cls.wh, gw.Options(module_mm=1.0, wms_errors=0.0, seed=3), out)
        assert ok
        cls.layout = str(out / "layout.json")
        cls.wh.save(cls.layout)
        vfov = math.degrees(2 * math.atan(math.tan(math.radians(10)) * 1200 / 1600))
        model = out / "model"
        model.mkdir()
        for name, text in mdm.build(Camera(1600, 1200, 20.0, vfov)).items():
            (model / name).write_text(text, encoding="utf-8")
        cls.model = model
        cls.cam = Camera.load(model / "camera_left.yaml")        # как в Gazebo: квадратные пиксели
        cls.p = ScanParams(camera=cls.cam, min_z=0.5)
        cls.frames_to_render = []
        rend = vf.Renderer(cls.cam, rng=np.random.default_rng(0))
        scene = vf.build_scene(cls.wh, cls.truth, out / "labels")
        wps = [w for s in plan(cls.wh, cls.p).sorties for w in s.waypoints if w.kind == "scan" and w.targets]
        from invent_core import frames as fr
        for wp in wps[:6]:
            for look, mount in vf.mounts(cls.p).items():
                r, t = fr.camera_pose((wp.x, wp.y, wp.z), wp.yaw, look, t_body_cam=mount)
                cls.frames_to_render.append((wp, look, rend.frame(scene, r, t, 0.0)))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_chain(self, frame_is_optical=False):
        """-> [(адрес по узлу, адрес по истине)] для всех прочитанных SSCC."""
        from invent_core.frames import euler_zyx, map_pose_to_px4, quat_from_matrix
        sscc_home = {code: addr for addr, code in self.truth.items() if code}
        with rs.installed(*PKGS):
            rs.PARAMS["odom_tf"].update(layout=self.layout, mounts_file=str(self.model / "mounts.json"))
            rs.PARAMS["label_reader"].update(layout=self.layout, camera_info=str(self.model / "camera_left.yaml"),
                                             frame_is_optical=frame_is_optical)
            from invent_mission.odom_tf_node import OdomTf
            from invent_perception.label_reader_node import LabelReader
            from px4_msgs.msg import VehicleOdometry
            from sensor_msgs.msg import Image as ImageMsg
            OdomTf()
            reader = LabelReader()
            for wp, look, img in self.frames_to_render:
                pos, q = map_pose_to_px4((wp.x, wp.y, wp.z), quat_from_matrix(euler_zyx(wp.yaw, 0.0, 0.0)),
                                         self.wh.dock)
                rs.publish("/fmu/out/vehicle_odometry",
                           VehicleOdometry(pose_frame=VehicleOdometry.POSE_FRAME_NED, position=list(pos), q=list(q)))
                msg = ImageMsg(height=img.shape[0], width=img.shape[1], encoding="mono8", array=img)
                msg.header.frame_id = f"label_cam_{look}_link"
                msg.header.stamp = rs.CLOCK.now().to_msg()
                rs.publish("image", msg)
            self.assertNotIn("нет TF", reader.get_logger().text())
            return [(m.address, sscc_home.get(m.sscc)) for m in rs.SENT["labels"] if m.sscc]

    def test_sscc_attributed_to_true_cell(self):
        pairs = self.run_chain()
        self.assertGreaterEqual(len(pairs), 4, "мало прочитанных SSCC — проверьте рендер/оптику теста")
        wrong = [(got, want) for got, want in pairs if got != want]
        self.assertEqual(wrong, [], "привязка SSCC к ячейке через TF расходится с истиной")

    def test_wrong_camera_frame_convention_is_detected(self):
        """Контроль чувствительности: кадр звена, принятый за оптический, ломает привязку."""
        pairs = self.run_chain(frame_is_optical=True)
        wrong = sum(got != want for got, want in pairs)
        self.assertGreater(wrong, len(pairs) // 3, pairs)   # на складе 1×2×2 часть ячеек угадывается случайно


if __name__ == "__main__":
    unittest.main()

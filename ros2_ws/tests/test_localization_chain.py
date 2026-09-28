"""SIM-05 / LOC-11: цепочка локализации без ROS и Gazebo.

Истинная поза (/gz/true_pose, как её отдаёт PosePublisher модели через ros_gz_bridge) -> fake_vio (дрейф)
-> landmark_localizer (+ LabelRead адресных этикеток, спроецированных из истинной позы камер) ->
localization/odom и /fmu/in/vehicle_visual_odometry (NED) для PX4.
"""
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ros_stubs as rs  # noqa: E402

try:
    import numpy as np
    DEPS = True
except ImportError:
    DEPS = False

PKG = str(ROOT / "ros2_ws" / "src" / "invent_localization")
MODEL = "x500_invent_0"


def stamp(t: float):
    return types.SimpleNamespace(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))


@unittest.skipUnless(DEPS, "нужен numpy")
class LocalizationChainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import make_drone_model as mdm
        from invent_core.camera import Camera
        from invent_core.layout import make_grid_warehouse
        cls.tmp = tempfile.TemporaryDirectory()
        out = Path(cls.tmp.name)
        cls.wh = make_grid_warehouse(n_aisles=1, bays=6, levels=3)
        cls.layout = str(out / "layout.json")
        cls.wh.save(cls.layout)
        cls.model = out / "model"
        cls.model.mkdir()
        for name, text in mdm.build(Camera()).items():
            (cls.model / name).write_text(text, encoding="utf-8")
        cls.cam = Camera.load(cls.model / "camera_left.yaml")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def trajectory(self, dt=0.1, v=0.5, z=1.2):
        """Док -> подъём -> вдоль прохода (курс 0: левая камера на ряд +y, правая на −y)."""
        x0, y0, _ = self.wh.dock
        x_end = max(loc.x for loc in self.wh.locations()) + 1.0
        t, pts = 0.0, []
        for k in range(int(z / (v * dt)) + 1):
            pts.append((t, x0, y0, min(z, k * v * dt), 0.0))
            t += dt
        x = x0
        while x < x_end:
            x += v * dt
            pts.append((t, x, y0, z, 0.0))
            t += dt
        return pts

    def fly(self, observe_every=5, pixel_sigma=1.0, forward=True, labels=True, model=MODEL, **vio):
        from invent_core import frames as fr
        from invent_core.association import project_to_image
        from invent_core.localization import address_landmarks
        import virtual_flight as vf
        from invent_core.coverage import ScanParams
        rng = np.random.default_rng(1)
        landmarks = address_landmarks(self.wh)
        mounts = vf.mounts(ScanParams(camera=self.cam))
        with rs.installed(PKG):
            rs.PARAMS["fake_vio"].update(trans_noise=0.0, yaw_noise_deg=0.0, scale_err=0.0, yaw_bias_deg_min=0.0)
            rs.PARAMS["landmark_localizer"].update(
                layout=self.layout, mounts_file=str(self.model / "mounts.json"), forward_to_px4=forward,
                camera_info_left=str(self.model / "camera_left.yaml"),
                camera_info_right=str(self.model / "camera_right.yaml"))
            from geometry_msgs.msg import Point, TransformStamped
            from invent_localization.fake_vio_node import FakeVio
            from invent_localization.landmark_localizer_node import LandmarkLocalizerNode
            from invent_msgs.msg import LabelRead
            from tf2_msgs.msg import TFMessage
            vio_node = FakeVio()
            for k, val in vio.items():            # ошибки VIO задаём явно, а не случайным масштабом
                setattr(vio_node, k, val)
            loc = LandmarkLocalizerNode()
            n_obs = 0
            for i, (t, x, y, z, yaw) in enumerate(self.trajectory()):
                rs.CLOCK.t_ns = int(t * 1e9)
                tr = TransformStamped()
                tr.header.stamp, tr.header.frame_id, tr.child_frame_id = stamp(t), "warehouse", model
                tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z = x, y, z
                tr.transform.rotation.z, tr.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
                rs.publish("/gz/true_pose", TFMessage(transforms=[tr]))
                if not labels or i % observe_every:
                    continue
                for look, mount in mounts.items():
                    r, tc = fr.camera_pose((x, y, z), yaw, look, t_body_cam=mount)
                    for addr, p in landmarks.items():
                        uv = project_to_image(tuple(p), self.cam, r, tc)
                        if uv is None or not (0 <= uv[0] < self.cam.width and 0 <= uv[1] < self.cam.height):
                            continue
                        u, v = uv + rng.normal(0.0, pixel_sigma, 2)
                        m = LabelRead(text=addr, address=addr, confidence=1.0,
                                      corners=[Point(x=float(u), y=float(v)) for _ in range(4)])
                        m.header.frame_id, m.header.stamp = f"label_cam_{look}_link", stamp(t)
                        rs.publish("labels", m)
                        n_obs += 1
            truth = np.array(self.trajectory()[-1][1:4])
            return types.SimpleNamespace(
                vio=rs.SENT["vio_odom"], loc=rs.SENT["localization/odom"], px4=rs.SENT["/fmu/in/vehicle_visual_odometry"],
                tf=dict(rs.TF), n_obs=n_obs, truth=truth, ekf=loc.ekf, vio_log=vio_node.get_logger().text())

    @staticmethod
    def pos(od):
        p = od.pose.pose.position
        return np.array([p.x, p.y, p.z])

    def test_perfect_vio_matches_truth(self):
        r = self.fly(labels=False)
        self.assertGreater(len(r.vio), 100)
        x0, y0, _ = self.wh.dock
        # odom начинается в точке старта с курсом 0; док в map — (x0, y0), курс дока 0
        # fake_vio добавляет шум 0.1 мм на шаг даже при нулевых ошибках — за 350 шагов ~мм
        np.testing.assert_allclose(self.pos(r.vio[-1]) + [x0, y0, 0.0], r.truth, atol=0.01)
        np.testing.assert_allclose(self.pos(r.loc[-1]), r.truth, atol=0.01)

    def test_wrong_model_name_warns(self):
        r = self.fly(labels=False, model="x500_0")
        self.assertEqual(len(r.vio), 0)
        self.assertIn("model_name", r.vio_log)

    def test_labels_correct_drifting_vio(self):
        bias = math.radians(3.0) / 60.0        # 3°/мин — плохой VIO
        r = self.fly(scale=1.05, yaw_bias=bias)
        self.assertGreater(r.n_obs, 50, "этикетки не попадают в кадр — проверьте геометрию теста")
        x0, y0, _ = self.wh.dock
        vio_err = np.linalg.norm(self.pos(r.vio[-1])[:2] + [x0, y0] - r.truth[:2])
        loc_err = np.linalg.norm(self.pos(r.loc[-1])[:2] - r.truth[:2])
        self.assertGreater(vio_err, 0.5)
        self.assertLess(loc_err, 0.1, f"VIO {vio_err:.2f} м, после этикеток {loc_err:.2f} м")
        self.assertGreater(r.ekf.accepted, r.n_obs * 0.9)

        # поза в PX4: NED от дока; map->odom в TF — поправка к VIO
        from invent_core.frames import px4_to_map
        last = r.px4[-1]
        np.testing.assert_allclose(px4_to_map(tuple(last.position), self.wh.dock), self.pos(r.loc[-1]), atol=1e-6)
        self.assertTrue(all(v == v and v >= 0 for v in last.position_variance))
        self.assertEqual((last.timestamp, last.timestamp_sample), (0, 0))   # PX4 подставит время прихода
        self.assertIn("odom", r.tf)
        self.assertEqual(r.tf["odom"].header.frame_id, "map")


if __name__ == "__main__":
    unittest.main()

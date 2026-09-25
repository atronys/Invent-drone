"""Тесты геометрии: кадры (гл. 2), камера и дисторсия (гл. 4), 6-DoF привязка (PER-06)."""
import math
import random
import tempfile
import unittest
from pathlib import Path

from invent_core import frames as fr
from invent_core.association import locate_label, locate_label_pose, pixel_to_face_pose, project_to_image
from invent_core.camera import Camera
from invent_core.layout import make_grid_warehouse

CAMERA_INFO = """image_width: 1920
image_height: 1080
camera_name: left
camera_matrix:
  rows: 3
  cols: 3
  data: [1400.5, 0, 961.2, 0, 1398.7, 540.9, 0, 0, 1]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [-0.28, 0.07, 0.0001, -0.0002, 0]
"""


def det(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def is_rotation(m, tol=1e-9):
    mt = fr.matmul(fr.transpose(m), m)
    return all(abs(mt[i][j] - (i == j)) < tol for i in range(3) for j in range(3)) and abs(det(m) - 1) < tol


class TestFrames(unittest.TestCase):
    def test_enu_ned_involution(self):
        v = (1.0, 2.0, 3.0)
        self.assertEqual(fr.enu_to_ned(v), (2.0, 1.0, -3.0))
        self.assertEqual(fr.ned_to_enu(fr.enu_to_ned(v)), v)

    def test_yaw(self):
        self.assertAlmostEqual(fr.yaw_enu_to_ned(0.0), math.pi / 2)          # восток
        self.assertAlmostEqual(fr.yaw_enu_to_ned(math.pi / 2), 0.0)          # север
        for a in (-3.0, -1.0, 0.3, 2.9):
            self.assertAlmostEqual(fr.wrap(fr.yaw_ned_to_enu(fr.yaw_enu_to_ned(a)) - a), 0.0)

    def test_map_px4_roundtrip(self):
        dock = (-2.0, 2.6, 0.0)
        p = (5.0, 4.0, 3.0)
        self.assertEqual(fr.map_to_px4(dock, dock), (0.0, 0.0, -0.0))
        back = fr.px4_to_map(fr.map_to_px4(p, dock), dock)
        self.assertTrue(all(abs(a - b) < 1e-12 for a, b in zip(back, p)))

    def test_rotations(self):
        self.assertTrue(is_rotation(fr.euler_zyx(0.3, -0.2, 0.1)))
        w, x, y, z = fr.quat_from_matrix(fr.rot_z(0.8))
        self.assertAlmostEqual(w, math.cos(0.4))
        self.assertAlmostEqual(z, math.sin(0.4))
        self.assertAlmostEqual(x, 0.0)
        for look in ("left", "right", "front"):
            for tilt in (0.0, 20.0):
                self.assertTrue(is_rotation(fr.camera_mount(look, tilt)))
        m = fr.camera_mount("left")
        self.assertEqual(tuple(m[i][2] for i in range(3)), (0.0, 1.0, 0.0))    # ось смотрит влево (+y)
        self.assertEqual(tuple(m[i][0] for i in range(3)), (1.0, 0.0, 0.0))    # «вправо» в кадре = вперёд


class TestPx4Pose(unittest.TestCase):
    def test_heading_north_is_enu_yaw_90(self):
        p, q = fr.px4_pose_to_map((0.0, 0.0, -1.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        self.assertEqual(p, (0.0, 0.0, 1.0))
        r = fr.matrix_from_quat(*q)
        self.assertAlmostEqual(math.atan2(r[1][0], r[0][0]), math.pi / 2)   # нос на север = +y ENU
        self.assertAlmostEqual(r[2][2], 1.0)                                 # z тела вверх

    def test_roundtrip_random(self):
        rng = random.Random(5)
        dock = (-2.0, 2.6, 0.0)
        for _ in range(50):
            q = fr.quat_from_matrix(fr.euler_zyx(rng.uniform(-3, 3), rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)))
            pos = (rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-3, 0))
            pm, qm = fr.px4_pose_to_map(pos, q, dock)
            pn, qn = fr.map_pose_to_px4(pm, qm, dock)
            self.assertTrue(all(abs(a - b) < 1e-9 for a, b in zip(pn, pos)))
            same = all(abs(a - b) < 1e-9 for a, b in zip(qn, q)) or all(abs(a + b) < 1e-9 for a, b in zip(qn, q))
            self.assertTrue(same)

    def test_yaw_consistent_with_heading_conversion(self):
        for psi_ned in (0.0, 0.7, -2.0):
            _, q = fr.px4_pose_to_map((0, 0, 0), fr.quat_from_matrix(fr.rot_z(psi_ned)), (0, 0, 0))
            r = fr.matrix_from_quat(*q)
            self.assertAlmostEqual(fr.wrap(math.atan2(r[1][0], r[0][0]) - fr.yaw_ned_to_enu(psi_ned)), 0.0)


class TestCamera(unittest.TestCase):
    def test_distortion_roundtrip(self):
        cam = Camera.from_intrinsics(1920, 1080, 1400, 1400, 960, 540, (-0.28, 0.07, 1e-4, -2e-4, 0.0))
        for u, v in [(10, 10), (960, 540), (1900, 1070), (300, 900)]:
            x, y = cam.pixel_to_normalized(u, v)
            uu, vv = cam.normalized_to_pixel(x, y)
            self.assertAlmostEqual(uu, u, places=4)
            self.assertAlmostEqual(vv, v, places=4)

    def test_load_camera_info(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "left.yaml"
            p.write_text(CAMERA_INFO, encoding="utf-8")
            cam = Camera.load(p)
        self.assertEqual((cam.width, cam.height), (1920, 1080))
        self.assertEqual((cam.fx, cam.fy, cam.cx, cam.cy), (1400.5, 1398.7, 961.2, 540.9))
        self.assertEqual(cam.dist, (-0.28, 0.07, 0.0001, -0.0002, 0.0))
        self.assertAlmostEqual(cam.hfov_deg, math.degrees(2 * math.atan(1920 / 2801.0)))

    def test_fov_and_intrinsics_agree(self):
        a = Camera()
        b = Camera.from_intrinsics(a.width, a.height, a.fx, a.fy, a.cx, a.cy)
        self.assertAlmostEqual(a.footprint(1.5)[0], b.footprint(1.5)[0])
        self.assertAlmostEqual(a.px_per_mm(1.5), b.px_per_mm(1.5))

    def test_behind_camera(self):
        self.assertIsNone(Camera().project((0.0, 0.0, -1.0)))


class TestPoseAssociation(unittest.TestCase):
    """Свойство: проецируем точку фасада при случайной позе -> обратная проекция возвращает её же."""

    def setUp(self):
        self.wh = make_grid_warehouse()
        self.aisle = self.wh.aisles[0]            # ряды A (face_y=1.1, +y) и B (face_y=4.1, -y)
        self.cam = Camera.from_intrinsics(1920, 1080, 1700, 1700, 955, 545, (-0.2, 0.05, 1e-4, 1e-4, 0.0))

    def test_roundtrip_random_poses(self):
        rng = random.Random(1)
        n_ok = 0
        for _ in range(300):
            row = self.wh.row(rng.choice(self.aisle.rows))
            look = "right" if row.normal_y > 0 else "left"   # летим вдоль +x: ряд A справа, B слева
            yaw, pitch, roll = (math.radians(rng.uniform(-8, 8)), math.radians(rng.uniform(-5, 5)),
                                math.radians(rng.uniform(-5, 5)))
            body = (rng.uniform(2, 25), self.aisle.y + rng.uniform(-0.3, 0.3), rng.uniform(1, 5))
            r, t = fr.camera_pose(body, yaw, look, pitch, roll, t_body_cam=(0.05, 0.0, -0.03))
            target = (body[0] + rng.uniform(-0.3, 0.3), row.face_y, body[2] + rng.uniform(-0.2, 0.2))
            uv = project_to_image(target, self.cam, r, t)
            if not self.cam.in_image(uv):
                continue
            x, z = pixel_to_face_pose(uv[0], uv[1], self.cam, r, t, row)
            self.assertAlmostEqual(x, target[0], places=5)
            self.assertAlmostEqual(z, target[2], places=5)
            self.assertEqual(locate_label_pose(uv[0], uv[1], self.cam, r, t, row), row.locate(target[0], target[2]))
            n_ok += 1
        self.assertGreater(n_ok, 200)

    def test_nominal_pose_matches_simplified(self):
        row = self.wh.row("A")
        cam = Camera()
        body = (4.05, self.aisle.y, 2.3)
        r, t = fr.camera_pose(body, 0.0, "right")
        for u, v in [(cam.cx, cam.cy), (cam.cx + 700, cam.cy - 300), (100, 2900)]:
            self.assertEqual(locate_label_pose(u, v, cam, r, t, row), locate_label(u, v, cam, body, row))

    def test_wrong_side_or_behind(self):
        row = self.wh.row("A")
        r, t = fr.camera_pose((4.0, 0.5, 2.0), 0.0, "right")   # камера за рядом A
        self.assertIsNone(pixel_to_face_pose(100, 100, self.cam, r, t, row))
        r, t = fr.camera_pose((4.0, self.aisle.y, 2.0), 0.0, "left")  # смотрит от ряда A
        self.assertIsNone(pixel_to_face_pose(960, 540, self.cam, r, t, row))


if __name__ == "__main__":
    unittest.main()

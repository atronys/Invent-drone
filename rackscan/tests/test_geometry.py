"""Оптика, модель стенда, поза по меткам, наведение."""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rackscan import guidance as gd
from rackscan import markers, optics
from rackscan.layout import Address, Layout, Rack
from rackscan.pose import polygon_visible_fraction, rectify, solve_facade_pose
from rackscan.synth import BoxSpec, Scene

PREVIEW = optics.camera_from_preset("imx708", (1536, 864))


class OpticsTest(unittest.TestCase):
    def test_focal_from_fov(self):
        cam = optics.CameraModel.from_fov(2000, 1000, 90.0)
        self.assertAlmostEqual(cam.fx, 1000.0, places=6)
        self.assertAlmostEqual(cam.hfov_deg, 90.0, places=6)

    def test_scaled_keeps_fov(self):
        full = optics.camera_from_preset("imx708")
        self.assertAlmostEqual(full.scaled(1536, 864).hfov_deg, full.hfov_deg, places=6)

    def test_read_distance_is_inverse_of_ppm(self):
        cam = optics.camera_from_preset("imx708")
        d = optics.max_read_distance(cam, 0.5, 3.0)
        self.assertAlmostEqual(optics.px_per_module(cam, d, 0.5), 3.0, places=6)
        # вдвое крупнее модуль — вдвое дальше
        self.assertAlmostEqual(optics.max_read_distance(cam, 1.0, 3.0), 2 * d, places=6)

    def test_fit_distance(self):
        d = optics.min_fit_distance(PREVIEW, 0.4, 0.4, margin=0.15)
        w, h = PREVIEW.footprint(d)
        self.assertGreaterEqual(w * 0.7 + 1e-9, 0.4)
        self.assertGreaterEqual(h * 0.7 + 1e-9, 0.4)

    def test_exposure_independent_of_camera(self):
        self.assertAlmostEqual(optics.max_exposure_s(0.4, 0.5), 0.5 * 0.5e-3 / 0.4)

    def test_plan_two_steps_for_dense_code(self):
        still = optics.camera_from_preset("imx708")
        plan = optics.plan_distances(PREVIEW, still, 0.4, 0.4, module_mm=0.33)
        self.assertFalse(plan.feasible_single_step)
        self.assertLess(plan.d_label, plan.d_cell)
        self.assertGreaterEqual(plan.ppm_still_at_label, 3.0)


class LayoutTest(unittest.TestCase):
    def test_address_roundtrip(self):
        a = Address.parse("02-03-01")
        self.assertEqual((a.rack, a.level, a.pos), (2, 3, 1))
        self.assertEqual(str(a), "02-03-01")
        with self.assertRaises(ValueError):
            Address.parse("2-3-1")

    def test_marker_ids_unique_and_decodable(self):
        lay = Layout.demo()
        self.assertEqual(len(lay.marker_ids), 18)
        for a in lay.addresses():
            self.assertEqual(lay.address_of(lay.marker_id(a)), a)

    def test_marker_sits_under_cell(self):
        lay = Layout.demo()
        a = Address(1, 2, 2)
        rack = lay.rack_of(a)
        x0, y0, x1, _ = rack.cell_rect_mm(a)
        mx, my = rack.marker_center_mm(a)
        self.assertAlmostEqual(mx, (x0 + x1) / 2)
        self.assertLess(my, y0)
        self.assertGreater(my, y0 - rack.beam_height_mm)

    def test_save_load(self):
        lay = Layout.demo()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "lay.json"
            lay.save(p)
            back = Layout.load(p)
        self.assertEqual(back.marker_ids, lay.marker_ids)
        self.assertEqual(back.racks[2].origin_x_mm, lay.racks[2].origin_x_mm)

    def test_serpentine_covers_all_once(self):
        lay = Layout.demo()
        order = lay.serpentine()
        self.assertEqual(sorted(order), lay.addresses())
        self.assertEqual([str(a) for a in order[:6]],
                         ["01-01-01", "01-01-02", "01-01-03", "01-02-03", "01-02-02", "01-02-01"])

    def test_too_many_cells_for_dictionary(self):
        with self.assertRaises(ValueError):
            Layout.build([Rack(id=1, cells_per_level=10, shelf_y_mm=[100.0 * i for i in range(1, 7)])],
                         marker_dictionary="DICT_4X4_50")


class PoseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lay = Layout.demo()
        cls.scene = Scene(cls.lay, {Address(1, 2, 2): [BoxSpec(60, 280, 250)]})

    def _pose(self, p, yaw=0.0, pitch=0.0, roll=0.0):
        img = self.scene.render(PREVIEW, p, yaw, pitch, roll)
        return solve_facade_pose(markers.detect(img, self.lay.marker_dictionary), self.lay, PREVIEW), img

    def test_synthetic_camera_angles_roundtrip(self):
        """Истинная поза, собранная из углов, возвращает те же углы и положение."""
        tp = Scene.camera_pose(PREVIEW, (0.5, 0.7, 1.2), 17.0, -6.0, 4.0)
        np.testing.assert_allclose(tp.position, (0.5, 0.7, 1.2), atol=1e-9)
        self.assertAlmostEqual(tp.yaw_deg, 17.0, places=6)
        self.assertAlmostEqual(tp.pitch_deg, -6.0, places=6)
        self.assertAlmostEqual(tp.roll_deg, 4.0, delta=0.05)

    def test_pose_accuracy_many_markers(self):
        rng = np.random.default_rng(3)
        for _ in range(8):
            p = np.array([rng.uniform(0.3, 0.9), rng.uniform(0.4, 0.9), rng.uniform(0.6, 1.5)])
            yaw, pitch = rng.uniform(-18, 18), rng.uniform(-6, 6)
            pose, _ = self._pose(p, yaw, pitch)
            self.assertIsNotNone(pose)
            self.assertGreaterEqual(pose.n_markers, 2)
            self.assertLess(np.linalg.norm(pose.position - p), 0.02)
            self.assertLess(abs(pose.yaw_deg - yaw), 1.0)
            self.assertLess(abs(pose.pitch_deg - pitch), 1.0)

    def test_no_markers_no_pose(self):
        blank = np.full((PREVIEW.height, PREVIEW.width, 3), 128, np.uint8)
        self.assertIsNone(solve_facade_pose(markers.detect(blank, self.lay.marker_dictionary), self.lay, PREVIEW))

    def test_foreign_markers_ignored(self):
        other = Layout.build([Rack(id=7)])           # те же картинки меток, но ID вне нашей модели
        scene = Scene(other)
        img = scene.render(PREVIEW, (0.6, 0.6, 1.0))
        self.assertIsNone(solve_facade_pose(markers.detect(img, self.lay.marker_dictionary), self.lay, PREVIEW))

    def test_pixel_to_facade_inverts_projection(self):
        pose, _ = self._pose((0.6, 0.7, 1.0), 8.0)
        pt = np.array([[0.75, 0.40, 0.0]])
        u, v = pose.project(pt)[0]
        x, y = pose.pixel_to_facade(u, v)
        self.assertAlmostEqual(x, 0.75, places=4)
        self.assertAlmostEqual(y, 0.40, places=4)

    def test_rectify_shows_box_in_right_place(self):
        a = Address(1, 2, 2)
        pose, img = self._pose((0.65, 0.75, 0.9), 10.0)
        rect = self.lay.rack_of(a).cell_rect_mm(a)
        roi = rectify(img, pose, rect, 0.5)
        self.assertEqual(roi.shape[:2], (170, 200))
        # коробка: x 60..340 мм, высота 250 мм от настила -> в выпрямленном виде центр картонный, угол тёмный
        self.assertGreater(roi[120, 100].mean(), 110)
        self.assertLess(roi[20, 8].mean(), 70)
        self.assertGreater(polygon_visible_fraction(pose.rect_polygon(rect), PREVIEW.width, PREVIEW.height), 0.99)


class GuidanceTest(unittest.TestCase):
    TOL = gd.Tolerances(0.03, 0.08, 5.0)

    def _g(self, p, yaw, target=(1.0, 1.0), d=0.8):
        return gd.compute(Scene.camera_pose(PREVIEW, p, yaw), target, d, self.TOL)

    def test_aligned(self):
        g = self._g((1.0, 1.0, 0.8), 0.0)
        self.assertTrue(g.aligned)
        self.assertEqual(g.status, gd.YELLOW)
        self.assertEqual(g.commands, [])

    def test_directions(self):
        cases = [((0.7, 1.0, 0.8), 0, "x", "ВПРАВО"), ((1.3, 1.0, 0.8), 0, "x", "ВЛЕВО"),
                 ((1.0, 0.8, 0.8), 0, "y", "ВВЕРХ"), ((1.0, 1.2, 0.8), 0, "y", "ВНИЗ"),
                 ((1.0, 1.0, 1.3), 0, "z", "БЛИЖЕ"), ((1.0, 1.0, 0.5), 0, "z", "ДАЛЬШЕ"),
                 ((1.0, 1.0, 0.8), 15, "yaw", "ПОВЕРНИТЕ ВЛЕВО"), ((1.0, 1.0, 0.8), -15, "yaw", "ПОВЕРНИТЕ ВПРАВО")]
        for p, yaw, axis, text in cases:
            g = self._g(p, yaw)
            self.assertFalse(g.aligned)
            self.assertEqual(g.status, gd.RED)
            self.assertEqual((g.commands[0].axis, g.commands[0].text), (axis, text), (p, yaw))

    def test_values_in_centimetres(self):
        g = self._g((0.7, 1.0, 0.8), 0.0)
        self.assertAlmostEqual(g.commands[0].value, 30.0, delta=0.1)
        self.assertEqual(g.commands[0].phrase(), "ВПРАВО на 30 см")

    def test_body_setpoint_rotates_with_yaw(self):
        # камера повёрнута на 90° вправо: «вправо вдоль фасада» для неё — это «вперёд»
        b = gd.body_setpoint(0.5, 0.0, 0.0, 90.0)
        self.assertAlmostEqual(b["forward_m"], 0.5, places=6)
        self.assertAlmostEqual(b["right_m"], 0.0, places=6)
        b = gd.body_setpoint(0.2, 0.1, 0.3, 0.0)
        self.assertEqual((round(b["forward_m"], 6), round(b["right_m"], 6), b["up_m"]), (0.3, 0.2, 0.1))

    def test_velocity_saturates(self):
        v = gd.velocity_command({"forward_m": 5.0, "right_m": -5.0, "up_m": 0.1, "yaw_deg": 90.0})
        self.assertEqual((v["vx_forward"], v["vy_right"], v["yaw_rate_dps"]), (0.3, -0.3, 20.0))
        self.assertAlmostEqual(v["vz_up"], 0.08)

    def test_follow_commands_converges(self):
        from rackscan.scanner import follow_commands
        p, yaw = np.array([0.2, 0.4, 1.6]), 20.0
        for _ in range(25):
            g = self._g(p, yaw)
            if g.aligned:
                break
            p, yaw = follow_commands(p, yaw, {"errors": g.errors})
        self.assertTrue(g.aligned)
        self.assertLess(math.dist(p, (1.0, 1.0, 0.8)), 0.09)

    def test_label_only_guidance(self):
        # символ 40 мм виден размером fx*0.04/0.5 px на дистанции 0,5 м, смещён вправо
        side = PREVIEW.fx * 0.04 / 0.5
        g = gd.from_label_only(PREVIEW.cx + 200, PREVIEW.cy, side, PREVIEW, 40.0, 0.5, self.TOL)
        self.assertEqual(g.commands[0].text, "ВПРАВО")
        self.assertAlmostEqual(g.errors["ez"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()

"""Ядро навигации по меткам: карта меток, поза по метке, выбор метки при повторяющихся ID, маршрут.

    .venv/Scripts/python -m unittest discover -s nav_markers/tests
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nav_core  # noqa: E402,F401  (добавляет INVENT_DRONE/core в sys.path)
from invent_core.camera import Camera  # noqa: E402
from invent_core.frames import euler_zyx  # noqa: E402
from invent_core.layout import Warehouse, make_grid_warehouse  # noqa: E402
from invent_core.localization import MountedCamera  # noqa: E402
from nav_core.aruco_pose import MarkerLocalizer, Observation, Prior, Rejection, expand_corners, marker_sigmas  # noqa: E402
from nav_core.markers import (MarkerMap, PlacementParams, marker_rotation, min_repeat_distance,  # noqa: E402
                              module_size, object_corners, place_markers)
from nav_core.route import TaskParams, aisle_graph, braking_distance, plan_task, shortest_path  # noqa: E402

DEMO = Path(__file__).resolve().parents[2] / "core" / "examples" / "warehouse_demo.json"


def cam640():
    return Camera.from_intrinsics(640, 480, 320.0, 320.0, 320.0, 240.0)


def mounts():
    return {
        "left": MountedCamera.from_mount(cam640(), [0.0, 0.08, 0.05], [0.0, 0.0, math.pi / 2]),
        "right": MountedCamera.from_mount(cam640(), [0.0, -0.08, 0.05], [0.0, 0.0, -math.pi / 2]),
        "front": MountedCamera.from_mount(cam640(), [0.12, 0.0, 0.05], [0.0, 0.0, 0.0]),
    }


def project(marker, size, mc: MountedCamera, p, yaw, roll=0.0, pitch=0.0, noise=0.0, rng=None):
    """Пиксели углов метки (TL, TR, BR, BL) в камере дрона с позой p, yaw; None — метка позади."""
    r_wb = np.array(euler_zyx(yaw, pitch, roll))
    r_wc = r_wb @ mc.r_bc
    t_wc = np.asarray(p, dtype=float) + r_wb @ mc.t_bc
    t_wm = marker.pose()
    out = []
    for c in object_corners(size):
        pc = r_wc.T @ (t_wm[:3, :3] @ c + t_wm[:3, 3] - t_wc)
        if pc[2] <= 0:
            return None
        u = mc.cam.fx * pc[0] / pc[2] + mc.cam.cx
        v = mc.cam.fy * pc[1] / pc[2] + mc.cam.cy
        if noise:
            u, v = u + rng.normal(0, noise), v + rng.normal(0, noise)
        out.append((u, v))
    return np.array(out)


def visible(uv, w=640, h=480):
    return uv is not None and all(0 <= u < w and 0 <= v < h for u, v in uv)


class TestMarkers(unittest.TestCase):
    def setUp(self):
        self.wh = Warehouse.load(DEMO)
        self.mm = place_markers(self.wh, PlacementParams(heights=[1.5]))

    def test_marker_frame_right_handed_and_faces_normal(self):
        for n in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]:
            r = marker_rotation(n)
            self.assertTrue(np.allclose(r.T @ r, np.eye(3)))
            self.assertAlmostEqual(np.linalg.det(r), 1.0)
            self.assertTrue(np.allclose(r[:, 2], n) and np.allclose(r[:, 1], [0, 0, 1]))

    def test_placement_demo(self):
        mm, wh = self.mm, self.wh
        anchors = mm.anchors()
        self.assertEqual(len(anchors), 2 * len(wh.aisles))
        self.assertEqual(len({m.id for m in anchors}), len(anchors))          # якоря уникальны
        path_ids = {m.id for m in mm.markers if m.kind == "path"}
        self.assertFalse(path_ids & {m.id for m in anchors})                  # пулы не пересекаются
        for m in mm.markers:                                                    # путевые смотрят в проход
            if m.kind == "path":
                a = next(a for a in wh.aisles if a.id == m.aisle)
                self.assertGreater((a.y - m.y) * m.normal[1], 0)
        self.assertGreater(min_repeat_distance(mm), 5.0)                       # повтор ID — в другом проходе
        a1 = sorted((m for m in mm.markers if m.kind == "path" and m.aisle == "1"), key=lambda m: m.x)
        self.assertTrue(all(a1[i].normal[1] != a1[i + 1].normal[1] for i in range(len(a1) - 1)))  # шахматка
        self.assertEqual(MarkerMap.from_dict(mm.to_dict()).markers, mm.markers)

    def test_module_size(self):
        self.assertAlmostEqual(module_size(0.20, "DICT_4X4_50"), 0.20 / 6)    # 33 мм — документ, разд. 4.2
        self.assertAlmostEqual(module_size(0.20, "DICT_7X7_250"), 0.20 / 9)

    def test_expand_corners(self):
        sq = np.array([[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]])   # TL, TR, BR, BL в пикселях
        e = expand_corners(sq, 0.5)
        self.assertTrue(np.allclose(e, [[9.5, 9.5], [50.5, 9.5], [50.5, 50.5], [9.5, 50.5]]))
        self.assertTrue(np.allclose(expand_corners(sq, 0.0), sq))
        # трапеция (метка в ракурсе): каждая сторона отодвигается ровно на δ
        tr = np.array([[10.0, 12.0], [50.0, 8.0], [52.0, 55.0], [8.0, 48.0]])
        e = expand_corners(tr, 0.7)
        for i in range(4):
            a, b = tr[i], tr[(i + 1) % 4]
            n = np.array([b[1] - a[1], a[0] - b[0]]) / np.linalg.norm(b - a)
            d = [(e[k] - a) @ n for k in (i, (i + 1) % 4)]
            self.assertTrue(np.allclose(np.abs(d), 0.7, atol=1e-9), d)

    def test_sigmas_match_document_table(self):
        lat, dep, _ = marker_sigmas(2.0, 1000.0, 0.20, 0.5)                    # разд. 5.3: d = 2 м -> 1,0 / 0,1 см
        self.assertAlmostEqual(dep, 0.010)
        self.assertAlmostEqual(lat, 0.001)


class TestPose(unittest.TestCase):
    def setUp(self):
        self.wh = Warehouse.load(DEMO)
        self.mm = place_markers(self.wh, PlacementParams(heights=[1.5]))
        self.loc = MarkerLocalizer(self.mm, mounts())

    def test_pose_from_marker_exact(self):
        for cam, p, yaw in [("left", (8.0, 2.6, 1.5), 0.0), ("right", (6.3, 2.6, 1.5), 0.02),
                            ("front", (-2.0, 2.6, 1.5), 0.0)]:
            with self.subTest(cam=cam):
                mc = self.loc.cameras[cam]
                r_wb = np.array(euler_zyx(yaw))
                found = 0
                for m in self.mm.markers:
                    uv = project(m, self.mm.size, mc, p, yaw)
                    if not visible(uv):
                        continue
                    o = self.loc.observe(cam, m.id, uv, Prior(np.array(p) + 0.1, np.eye(3) * 0.3 ** 2, r_wb))
                    if isinstance(o, Rejection):
                        continue
                    self.assertEqual(o.marker, m)
                    self.assertTrue(np.allclose(o.p_wb, p, atol=1e-6))
                    self.assertLess(abs(o.yaw - yaw), 1e-6)
                    found += 1
                self.assertGreaterEqual(found, 1)

    def test_repeated_id_resolved_by_prior_and_gated(self):
        mm, wh, loc = self.mm, self.wh, self.loc
        m = next(m for m in mm.markers if m.kind == "path" and m.aisle == "2" and len(mm.by_id(m.id)) > 1)
        look = "right" if m.normal[1] > 0 else "left"   # фасад смотрит в +y -> он справа от дрона с курсом 0
        p = np.array([m.x + 0.3, wh.aisles[1].y, 1.5])
        uv = project(m, mm.size, loc.cameras[look], p, 0.0)
        self.assertTrue(visible(uv))
        good = loc.observe(look, m.id, uv, Prior(p + [0.2, -0.1, 0], np.eye(3) * 0.3 ** 2, np.eye(3)))
        self.assertIsInstance(good, Observation)
        self.assertEqual(good.marker, m)
        # априорная поза в проходе 1 в 3 м от двойника: ни одна метка с этим ID не согласуется -> отказ
        bad = loc.observe(look, m.id, uv, Prior(np.array([m.x + 3.0, wh.aisles[0].y, 1.5]), np.eye(3) * 0.3 ** 2,
                                                np.eye(3)))
        self.assertIsInstance(bad, Rejection)
        # поза потеряна: путевую метку не берём, якорную — берём
        lost = loc.observe(look, m.id, uv, Prior(p, np.eye(3) * 100.0, np.eye(3), lost=True))
        self.assertIsInstance(lost, Rejection)
        a = mm.anchors()[0]
        pa = np.array([-2.0, a.y - 1.0, 1.5])
        uva = project(a, mm.size, loc.cameras["front"], pa, 0.0)
        self.assertTrue(visible(uva))
        oa = loc.observe("front", a.id, uva, Prior(np.zeros(3), np.eye(3) * 100.0, np.eye(3), lost=True))
        self.assertIsInstance(oa, Observation)
        self.assertTrue(np.allclose(oa.p_wb, pa, atol=1e-6))

    def fused_errors(self, scenes, p, yaw, n=300, prior_yaw_err=math.radians(3.0)):
        """scenes: [(камера, метка)] одного кадра. -> (средний NEES по (x,y,z,ψ), ошибки положения, ошибки курса)."""
        loc = MarkerLocalizer(self.mm, mounts(), sigma_px=0.5)
        rng = np.random.default_rng(1)
        nees, errs, yerrs = [], [], []
        r_prior = np.array(euler_zyx(yaw + prior_yaw_err))     # курс EKF2 ошибается на 3°
        for _ in range(n):
            obs = []
            for cam, m in scenes:
                uv = project(m, self.mm.size, loc.cameras[cam], p, yaw, noise=0.5, rng=rng)
                o = loc.observe(cam, m.id, uv, Prior(p, np.eye(3) * 0.2 ** 2, r_prior))
                self.assertIsInstance(o, Observation)
                obs.append(o)
            f = loc.fuse(obs, Prior(p, np.eye(3) * 0.2 ** 2, r_prior))
            e = np.array([*(f.p - p), math.atan2(math.sin(f.yaw - yaw), math.cos(f.yaw - yaw))])
            cov = np.zeros((4, 4))
            cov[:3, :3], cov[3, 3] = f.cov_p, f.var_yaw
            nees.append(e @ np.linalg.solve(cov + np.eye(4) * 1e-12, e))
            errs.append(np.linalg.norm(e[:3]))
            yerrs.append(abs(e[3]))
        return float(np.mean(nees)), np.array(errs), np.array(yerrs)

    def test_two_anchors_from_dock(self):
        """С дока фронтальная камера видит оба якоря прохода: курс по базе между ними, точность — сантиметры."""
        p = np.array([-2.0, self.wh.aisles[0].y, 1.5])
        anchors = [a for a in self.mm.anchors() if a.aisle == "1"]
        nees, errs, yerrs = self.fused_errors([("front", a) for a in anchors], p, 0.0)
        self.assertTrue(2.0 < nees < 6.5, nees)                   # ковариация честная: NEES ≈ 4 ст. св.
        self.assertLess(errs.mean(), 0.03)
        self.assertLess(np.degrees(yerrs.mean()), 1.0)           # курс EKF2 с ошибкой 3° исправлен

    def test_start_without_heading(self):
        """Старт без магнитометра: курс EKF2 ошибочен на 90° (поза «потеряна»). Якоря дают и положение, и курс."""
        p = np.array([-2.0, self.wh.aisles[0].y, 0.1])
        r_bad = np.array(euler_zyx(math.pi / 2))
        prior = Prior(np.zeros(3), np.eye(3) * 100.0, r_bad, lost=True)
        loc = MarkerLocalizer(self.mm, mounts(), sigma_px=0.5)
        rng = np.random.default_rng(3)
        obs = []
        for a in (a for a in self.mm.anchors() if a.aisle == "1"):
            uv = project(a, self.mm.size, loc.cameras["front"], p, 0.0, noise=0.5, rng=rng)
            if visible(uv):
                o = loc.observe("front", a.id, uv, prior)
                self.assertIsInstance(o, Observation)
                obs.append(o)
        self.assertEqual(len(obs), 2)
        f = loc.fuse(obs, prior)
        self.assertLess(np.linalg.norm(f.p - p), 0.05)
        self.assertLess(abs(math.degrees(f.yaw)), 1.0)

    def test_aisle_two_sides(self):
        """В проходе: метка слева и метка справа (шахматка) — две камеры, курс по базе поперёк прохода."""
        a1 = self.wh.aisles[0]
        right = next(m for m in self.mm.markers if m.kind == "path" and m.aisle == "1" and m.normal[1] > 0
                     and m.x > 5)
        left = min((m for m in self.mm.markers if m.kind == "path" and m.aisle == "1" and m.normal[1] < 0),
                   key=lambda m: abs(m.x - right.x - 1.0))
        p = np.array([(right.x + left.x) / 2, a1.y, 1.5])
        nees, errs, yerrs = self.fused_errors([("right", right), ("left", left)], p, 0.0)
        self.assertTrue(2.0 < nees < 6.5, nees)
        self.assertLess(errs.mean(), 0.03)
        self.assertLess(np.degrees(yerrs.mean()), 1.5)


class TestRoute(unittest.TestCase):
    def test_route_demo(self):
        wh = Warehouse.load(DEMO)
        _, edges = aisle_graph(wh)
        self.assertEqual(shortest_path(edges, "DOCK", "E2")[-1], "E2")
        wps = plan_task(wh, ["B-05-2-2", "C-03-1-1"], TaskParams())
        kinds = [w.kind for w in wps]
        self.assertEqual((kinds[0], kinds[-1]), ("takeoff", "land"))
        self.assertEqual(kinds.count("capture"), 2)
        cap = [w for w in wps if w.kind == "capture"]
        b = wh.row("B")
        self.assertAlmostEqual(cap[0].y, b.face_y + b.normal_y * 1.0)
        i0, i1 = kinds.index("capture"), len(kinds) - 1 - kinds[::-1].index("capture")
        self.assertIn("exit", kinds[i0:i1])                                     # между проходами — через магистраль
        self.assertIn("enter", kinds[i0:i1])
        self.assertTrue(all(w.x >= wh.cross_aisle_x - 1e-9 for w in wps))
        self.assertAlmostEqual(braking_distance(1.5, 2.0), 0.5625)             # документ, разд. 8.2

    def test_grid_warehouse_route(self):
        wh = make_grid_warehouse(n_aisles=3, bays=6)
        wps = plan_task(wh, ["E-02-1-1", "A-06-2-3"])
        self.assertEqual(sum(w.kind == "capture" for w in wps), 2)


if __name__ == "__main__":
    unittest.main()

"""Тесты: параметры миссии (FLT-02), перескан (MIS-03), запись вылета и неизвестная занятость (PER-07)."""
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from invent_core import frames as fr
from invent_core.association import locate_in_warehouse, project_to_image
from invent_core.camera import Camera
from invent_core.coverage import PlanningError, ScanParams, plan
from invent_core.layout import make_grid_warehouse
from invent_core.reconcile import (Observation, SortieRecorder, Status, load_observations_csv, reconcile,
                                   write_observations_csv)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class TestParams(unittest.TestCase):
    def test_example_file_matches_defaults(self):
        self.assertEqual(ScanParams.load(EXAMPLES / "scan_params.json"), ScanParams())

    def test_camera_file_and_typos(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "cam.json").write_text(json.dumps({
                "image_width": 1920, "image_height": 1080,
                "camera_matrix": {"data": [1700, 0, 960, 0, 1700, 540, 0, 0, 1]},
                "distortion_coefficients": {"data": [0, 0, 0, 0, 0]}}), encoding="utf-8")
            (d / "p.json").write_text(json.dumps({"camera_file": "cam.json", "v_scan": 0.3}), encoding="utf-8")
            p = ScanParams.load(d / "p.json")
            self.assertEqual((p.camera.fx, p.v_scan), (1700, 0.3))
            (d / "bad.json").write_text(json.dumps({"v_sacn": 0.3}), encoding="utf-8")
            with self.assertRaises(ValueError):
                ScanParams.load(d / "bad.json")


class TestRescan(unittest.TestCase):
    def test_only_addresses(self):
        wh = make_grid_warehouse(n_aisles=3, bays=12, levels=5)
        only = {"A-03-2-1", "D-11-5-3", "F-01-1-2"}
        full, part = plan(wh), plan(wh, ScanParams(hold_s=1.5), only=only)
        self.assertEqual(part.covered, only)
        self.assertEqual(part.warnings, [])
        self.assertLess(sum(len(s.waypoints) for s in part.sorties), sum(len(s.waypoints) for s in full.sorties))
        self.assertTrue(all(w.hold_s == 1.5 for s in part.sorties for w in s.waypoints if w.kind == "scan"))

    def test_unknown_address(self):
        with self.assertRaises(PlanningError):
            plan(make_grid_warehouse(), only={"Z-99-9-9"})


class TestRecorderAndOccupancy(unittest.TestCase):
    def test_recorder(self):
        rec = SortieRecorder()
        rec.expect(["A1", "A2", "A3"])
        rec.read("A1", "S1", 0.9)
        rec.read("", "S9")  # не привязано к адресу — игнорируем
        obs = rec.observations()
        self.assertEqual([(o.address, o.code, o.occupied) for o in obs],
                         [("A1", "S1", True), ("A2", None, None), ("A3", None, None)])

    def test_unknown_occupancy_statuses(self):
        rep = reconcile({"A1": "S1", "A2": None}, [Observation("A1", None, None), Observation("A2", None, None)])
        st = {r.address: (r.status, r.note) for r in rep.results}
        self.assertEqual(st["A1"][0], Status.UNREADABLE)   # паллета ожидалась, не прочитали: не «пропала»
        self.assertEqual(st["A2"][0], Status.OK)
        self.assertIn("не определялась", st["A2"][1])

    def test_seen_elsewhere_is_missing_without_occupancy(self):
        """Ожидаемую паллету прочитали в другой ячейке -> здесь MISSING, даже без датчика занятости."""
        rep = reconcile({"A1": "S1", "B1": None}, [Observation("A1", None, None), Observation("B1", "S1")])
        st = {r.address: r for r in rep.results}
        self.assertEqual(st["A1"].status, Status.MISSING)
        self.assertIn("B1", st["A1"].note)
        self.assertEqual(st["B1"].status, Status.UNEXPECTED)

    def test_csv_roundtrip(self):
        obs = [Observation("A1", "S1", True, 0.9), Observation("A2", None, None, 0.0), Observation("A3", None, False)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "obs.csv"
            write_observations_csv(obs, p)
            back = load_observations_csv(p)
        self.assertEqual([(o.address, o.code, o.occupied) for o in back], [(o.address, o.code, o.occupied) for o in obs])


class TestQuaternionAndWarehouseLookup(unittest.TestCase):
    def test_quat_roundtrip(self):
        rng = random.Random(3)
        for _ in range(50):
            r = fr.euler_zyx(rng.uniform(-3, 3), rng.uniform(-1.4, 1.4), rng.uniform(-3, 3))
            back = fr.matrix_from_quat(*fr.quat_from_matrix(r))
            self.assertTrue(all(abs(r[i][j] - back[i][j]) < 1e-9 for i in range(3) for j in range(3)))

    def test_locate_in_warehouse(self):
        wh = make_grid_warehouse()
        cam = Camera.from_intrinsics(1920, 1080, 1700, 1700, 960, 540)
        aisle = wh.aisles[1]                       # ряды C и D
        for look, row_id in (("right", "C"), ("left", "D")):
            r, t = fr.camera_pose((10.0, aisle.y, 2.4), 0.0, look)
            row = wh.row(row_id)
            uv = project_to_image((10.3, row.face_y, 2.5), cam, r, t)
            self.assertEqual(locate_in_warehouse(uv[0], uv[1], cam, r, t, wh), row.locate(10.3, 2.5))
        r, t = fr.camera_pose((10.0, aisle.y, 2.4), math.pi / 2, "left")  # смотрит вдоль прохода
        self.assertIsNone(locate_in_warehouse(960, 540, cam, r, t, wh))


if __name__ == "__main__":
    unittest.main()


class TestAddressLabelsInFrame(unittest.TestCase):
    """Портретная камера держит в кадре и адресную, и паллетную этикетку; альбомная — нет."""

    def test_required_vertical_px(self):
        from invent_core import barcode
        self.assertAlmostEqual(barcode.required_vertical_px(0.56, 0.5), 3500.0)
        self.assertAlmostEqual(barcode.required_vertical_px(0.56, 1.0), 1750.0)

    def test_portrait_vs_landscape(self):
        wh = make_grid_warehouse()
        def warn(cam):
            pl = plan(wh, ScanParams(camera=cam, cover_address_labels=True, min_z=0.5))
            return [w for w in pl.warnings if "адресные этикетки не в кадре" in w]
        landscape, portrait = Camera(4056, 3040, 29.4, 22.2), Camera(3040, 4056, 22.2, 29.4)
        self.assertGreater(len(warn(landscape)), len(wh.aisles))       # не хватает на верхних ярусах
        self.assertEqual(len(warn(portrait)), len(wh.aisles))          # только напольный ярус (0.1 м)

    def test_address_z(self):
        row = make_grid_warehouse().rows[0]
        self.assertAlmostEqual(row.address_z(1), 0.10)
        self.assertAlmostEqual(row.address_z(2), 1.74)

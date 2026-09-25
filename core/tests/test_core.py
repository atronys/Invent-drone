"""Тесты ядра. Запуск: python -m unittest discover -s tests  (или pytest)."""
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from invent_core import barcode
from invent_core.association import locate_label, pixel_to_face
from invent_core.camera import Camera
from invent_core.coverage import PlanningError, ScanParams, pass_heights, plan
from invent_core.layout import Aisle, RackRow, Warehouse, make_grid_warehouse
from invent_core.reconcile import (Observation, Status, load_expected_csv, reconcile)


class TestLayout(unittest.TestCase):
    def test_grid_counts(self):
        wh = make_grid_warehouse(n_aisles=2, bays=10, levels=4)
        self.assertEqual(len(wh.rows), 4)
        self.assertEqual(len(wh.locations()), 4 * 10 * 4 * 3)
        self.assertEqual(wh.locations()[0].address, "A-01-1-1")

    def test_locate_roundtrip(self):
        wh = make_grid_warehouse()
        for loc in wh.locations():
            self.assertEqual(wh.row(loc.row).locate(loc.x, loc.z), loc.address)

    def test_row_must_face_aisle(self):
        row = RackRow(id="A", origin_x=0, face_y=1.0, normal_y=-1, bays=2)
        with self.assertRaises(ValueError):
            Warehouse("bad", [row], [Aisle("1", 2.5, ("A",))])

    def test_save_load(self):
        wh = make_grid_warehouse()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "wh.json"
            wh.save(p)
            self.assertEqual(Warehouse.load(p).to_dict(), wh.to_dict())


class TestCoverage(unittest.TestCase):
    def test_pass_heights_cover_all(self):
        zs = [0.5, 0.9, 2.3, 2.4, 4.1, 5.9, 6.0]
        groups = pass_heights(zs, band=0.6)
        covered = [z for _, g in groups for z in g]
        self.assertEqual(sorted(covered), sorted(set(zs)))
        for center, g in groups:
            self.assertTrue(all(abs(z - center) <= 0.3 + 1e-9 for z in g))

    def test_plan_covers_everything_within_budget(self):
        wh = make_grid_warehouse(n_aisles=3, bays=12, levels=5)
        p = ScanParams()
        pl = plan(wh, p)
        self.assertEqual(pl.warnings, [])
        self.assertEqual(pl.covered, {loc.address for loc in wh.locations()})
        for s in pl.sorties:
            self.assertLessEqual(s.duration_s, p.battery_s * (1 - p.reserve) + 1e-6)
            self.assertAlmostEqual(s.waypoints[0].x, wh.dock[0])
            self.assertAlmostEqual(s.waypoints[-1].y, wh.dock[1])

    def test_scan_waypoints_stay_in_aisle(self):
        wh = make_grid_warehouse()
        ys = {a.y for a in wh.aisles}
        for s in plan(wh).sorties:
            for w in s.waypoints:
                if w.kind == "scan":
                    self.assertIn(w.y, ys)
                    self.assertGreaterEqual(w.z, ScanParams().min_z)

    def test_tiny_battery_raises(self):
        with self.assertRaises(PlanningError):
            plan(make_grid_warehouse(bays=30), replace(ScanParams(), battery_s=60))

    def test_low_label_warning(self):
        p = replace(ScanParams(), min_z=2.0)
        self.assertTrue(plan(make_grid_warehouse(), p).warnings)


class TestAssociation(unittest.TestCase):
    def setUp(self):
        self.cam = Camera()
        self.row = RackRow(id="A", origin_x=0.0, face_y=1.1, normal_y=1, bays=10)
        self.cam_pos = (4.05, 2.6, 2.3)  # центр секции 2, 1.5 м от фасада, высота этикетки 2-го яруса

    def test_center_pixel(self):
        self.assertEqual(locate_label(self.cam.cx, self.cam.cy, self.cam, self.cam_pos, self.row), "A-02-2-2")

    def test_right_of_image_is_minus_x_for_plus_normal(self):
        u = self.cam.cx + self.cam.fx * 0.9 / 1.5  # 0.9 м вправо в кадре
        x, _ = pixel_to_face(u, self.cam.cy, self.cam, self.cam_pos, self.row)
        self.assertAlmostEqual(x, 4.05 - 0.9)
        self.assertEqual(locate_label(u, self.cam.cy, self.cam, self.cam_pos, self.row), "A-02-2-1")

    def test_opposite_row(self):
        row_b = RackRow(id="B", origin_x=0.0, face_y=4.1, normal_y=-1, bays=10)
        u = self.cam.cx + self.cam.fx * 0.9 / 1.5
        x, _ = pixel_to_face(u, self.cam.cy, self.cam, self.cam_pos, row_b)
        self.assertAlmostEqual(x, 4.05 + 0.9)

    def test_up_in_image_is_higher(self):
        _, z = pixel_to_face(self.cam.cx, 0, self.cam, self.cam_pos, self.row)
        self.assertGreater(z, self.cam_pos[2])

    def test_wrong_side_raises(self):
        with self.assertRaises(ValueError):
            pixel_to_face(0, 0, self.cam, (4.0, 0.5, 2.0), self.row)


class TestBarcode(unittest.TestCase):
    def test_check_digit(self):
        self.assertEqual(barcode.gs1_check_digit("10614141123456789"), 7)   # SSCC из спецификации GS1
        self.assertEqual(barcode.gs1_check_digit("400638133393"), 1)        # EAN-13 4006381333931
        self.assertTrue(barcode.is_valid_sscc("106141411234567897"))
        self.assertFalse(barcode.is_valid_sscc("106141411234567898"))

    def test_parse_hri(self):
        d = barcode.parse_gs1("(00)106141411234567897(37)24")
        self.assertEqual(d, {"00": "106141411234567897", "37": "24"})

    def test_parse_raw(self):
        raw = "]C1" + "00106141411234567897" + "10ABC123" + barcode.GS + "17261231" + "3103001250"
        d = barcode.parse_gs1(raw)
        self.assertEqual(d, {"00": "106141411234567897", "10": "ABC123", "17": "261231", "3103": "001250"})

    def test_read_sscc(self):
        r = barcode.Read("(00)106141411234567897", "Code128", ((0, 0), (10, 0), (10, 5), (0, 5)))
        self.assertEqual(r.sscc, "106141411234567897")
        self.assertEqual(r.center, (5.0, 2.5))

    def test_optics_consistency(self):
        cam = Camera()
        d = barcode.max_read_distance(cam, 1.0)
        self.assertAlmostEqual(barcode.px_per_module(cam, d, 1.0), barcode.MIN_PPM)
        self.assertGreater(barcode.max_read_distance(cam, 1.0), barcode.max_read_distance(cam, 0.5))


class TestReconcile(unittest.TestCase):
    def test_statuses(self):
        expected = {"A1": "S1", "A2": "S2", "A3": "S3", "A4": None, "A5": "S5", "A6": "S6", "A7": None}
        obs = [
            Observation("A1", "S1"),                 # OK
            Observation("A2", "S9"),                 # MISMATCH
            Observation("A3", None, occupied=False),  # MISSING, но S3 нашли в A7
            Observation("A4", None, occupied=False),  # OK (пусто = пусто)
            Observation("A5", None, occupied=True),   # UNREADABLE
            Observation("A7", "S3"),                  # UNEXPECTED (паллета из A3)
        ]                                             # A6 — NOT_SCANNED
        rep = reconcile(expected, obs)
        st = {r.address: r.status for r in rep.results}
        self.assertEqual(st, {"A1": Status.OK, "A2": Status.MISMATCH, "A3": Status.MISSING, "A4": Status.OK,
                              "A5": Status.UNREADABLE, "A6": Status.NOT_SCANNED, "A7": Status.UNEXPECTED})
        notes = {r.address: r.note for r in rep.results}
        self.assertIn("A7", notes["A3"])
        self.assertIn("A3", notes["A7"])
        self.assertTrue(math.isclose(rep.accuracy, 2 / 5))

    def test_majority_vote(self):
        obs = [Observation("A1", "S1", confidence=0.4), Observation("A1", "S2", confidence=0.9),
               Observation("A1", "S1", confidence=0.3)]
        res = reconcile({"A1": "S2"}, obs).results[0]
        self.assertEqual(res.status, Status.OK)
        self.assertIn("конфликт", res.note)

    def test_csv_cp1251_semicolon(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "wms.csv"
            p.write_bytes("Адрес;Код\nA-01-1-1;106141411234567897\nA-01-1-2;\n".encode("cp1251"))
            self.assertEqual(load_expected_csv(p), {"A-01-1-1": "106141411234567897", "A-01-1-2": None})


if __name__ == "__main__":
    unittest.main()

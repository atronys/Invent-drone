"""SIM-09: весь конвейер на маленьком складе — рендер -> zxing -> привязка -> запись -> сверка."""
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "core"))

try:
    import barcode  # noqa: F401  (python-barcode — для текстур)
    import cv2  # noqa: F401
    import zxingcpp  # noqa: F401
    from PIL import Image  # noqa: F401
    DEPS = True
except ImportError:
    DEPS = False


@unittest.skipUnless(DEPS, "нужны opencv, zxing-cpp, python-barcode, pillow (requirements-dev.txt)")
class TestVirtualFlight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import generate_world as gw
        from invent_core.camera import Camera
        from invent_core.coverage import ScanParams
        from invent_core.layout import make_grid_warehouse
        cls.tmp = tempfile.TemporaryDirectory()
        out = Path(cls.tmp.name)
        (out / "labels").mkdir()
        cls.wh = make_grid_warehouse(n_aisles=1, bays=2, levels=2)
        _, cls.truth, cls.wms, _, ok = gw.build(cls.wh, gw.Options(module_mm=1.0, wms_errors=0.15, seed=3), out)
        assert ok
        vfov = math.degrees(2 * math.atan(math.tan(math.radians(10)) * 1200 / 1600))
        cls.p = ScanParams(camera=Camera(1600, 1200, 20.0, vfov), min_z=0.5)
        cls.labels = out / "labels"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_pipeline_reads_and_attributes(self):
        import virtual_flight as vf
        r = vf.run_flight(self.wh, self.p, self.truth, self.wms, self.labels)
        self.assertGreaterEqual(r.metrics["read_rate"], 0.9)
        self.assertEqual(r.metrics["attribution"], 1.0)
        self.assertEqual(r.metrics["garbage"], 0)

    def test_unreadable_label_means_occupied(self):
        """PER-09: все этикетки испорчены — не прочитано ничего, но паллета, которой нет в WMS, находится
        (кандидат этикетки -> «ячейка занята» -> UNEXPECTED), а не ложно считается OK."""
        import generate_world as gw
        import virtual_flight as vf
        from invent_core.layout import make_grid_warehouse
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "labels").mkdir()
            wh = make_grid_warehouse(n_aisles=1, bays=2, levels=2)
            _, truth, wms, _, _ = gw.build(wh, gw.Options(module_mm=1.0, wms_errors=0.15, damaged=1.0, seed=5), out)
            unrecorded = [a for a in truth if truth[a] and not wms[a]]
            self.assertEqual(len(unrecorded), 1)                  # сценарий: 1 паллета не учтена в WMS
            r = vf.run_flight(wh, self.p, truth, wms, out / "labels")
        self.assertEqual(r.metrics["reads"], 0)
        self.assertGreaterEqual(r.reconcile_summary["UNEXPECTED"], 1)
        self.assertEqual(r.injected_found, r.injected_in_scope)

    def test_pose_error_breaks_attribution(self):
        """Ошибка позы 0.5 м (больше половины места 0.9 м) должна портить привязку — тест на чувствительность."""
        import virtual_flight as vf
        r = vf.run_flight(self.wh, self.p, self.truth, self.wms, self.labels, pose_noise=0.5, seed=1)
        self.assertLess(r.metrics["attribution"], 0.95)


@unittest.skipUnless(DEPS and os.environ.get("INVENT_SLOW_TESTS"), "медленный (~30 с): INVENT_SLOW_TESTS=1")
class TestPortraitCameraSeesAddressLabels(unittest.TestCase):
    """12 Мп портретом + объектив 12 мм: читаются модули 0.5 мм и адресные этикетки попадают в кадр."""

    def test_address_labels_are_read_and_attributed(self):
        import generate_world as gw
        import virtual_flight as vf
        from invent_core.camera import Camera
        from invent_core.coverage import ScanParams
        from invent_core.layout import make_grid_warehouse
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "labels").mkdir()
            wh = make_grid_warehouse(n_aisles=1, bays=2, levels=2)
            _, truth, wms, _, ok = gw.build(wh, gw.Options(module_mm=0.5, seed=4), out)
            p = ScanParams(camera=Camera(3040, 4056, 22.2, 29.4), cover_address_labels=True, min_z=0.5)
            r = vf.run_flight(wh, p, truth, wms, out / "labels")
        self.assertGreaterEqual(r.metrics["read_rate"], 0.9)
        self.assertGreater(r.address_reads, 0)
        self.assertGreaterEqual(r.address_attribution, 0.95)


if __name__ == "__main__":
    unittest.main()

"""Тесты генератора мира (SIM-06/07), метрики чтения (SIM-04) и сквозной сверки (DATA-02)."""
import random
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "core"))

import eval_read_rate as ev  # noqa: E402
import generate_world as gw  # noqa: E402
from invent_core.layout import make_grid_warehouse  # noqa: E402
from invent_core.reconcile import Observation, Status, reconcile  # noqa: E402


class TestWorld(unittest.TestCase):
    def setUp(self):
        self.wh = make_grid_warehouse()

    def test_structure(self):
        sdf, truth, wms, addr, _ = gw.build(self.wh, gw.Options())
        root = ET.fromstring(sdf)
        names = [v.get("name") for v in root.iter("visual")]
        occupied = sum(1 for c in truth.values() if c)
        self.assertEqual(sum(n.startswith("pal_") for n in names), occupied)
        self.assertEqual(sum(n.startswith("lbl_") for n in names), occupied)
        self.assertEqual(sum(n.startswith("addr_") for n in names), len(self.wh.locations()))
        self.assertEqual(len(addr), len(self.wh.locations()))
        self.assertTrue(all(gw.gs1_check_digit(c[:17]) == int(c[17]) for c in truth.values() if c))

    def test_end_to_end_reconcile_finds_all_injected_errors(self):
        """DATA-02: «идеальный дрон» видит истину -> сверка находит все ошибки, внесённые в WMS."""
        _, truth, wms, _, _ = gw.build(self.wh, gw.Options(wms_errors=0.05, seed=7))
        obs = [Observation(a, c, occupied=bool(c)) for a, c in truth.items()]
        rep = reconcile(wms, obs)
        not_ok = {r.address for r in rep.results if r.status != Status.OK}
        self.assertEqual(not_ok, {a for a in truth if truth[a] != wms[a]})
        self.assertGreater(len(not_ok), 5)

    def test_scenarios(self):
        sdf, *_ = gw.build(self.wh, gw.Options(obstacle=True, rotate_max=10, ambient=0.2))
        root = ET.fromstring(sdf)
        self.assertIn("obstacle", [m.get("name") for m in root.iter("model")])
        poses = [v.find("pose").text for v in root.iter("visual") if v.get("name", "").startswith("lbl_")]
        self.assertTrue(any(p.split()[4] != "0.0000" for p in poses))   # есть наклонённые этикетки
        self.assertIn("0.2 0.2 0.2 1", sdf)

    def test_textures_decode(self):
        try:
            import numpy as np
            import zxingcpp
            from PIL import Image
        except ImportError:
            self.skipTest("нужны python-barcode, pillow, zxing-cpp")
        from invent_core import barcode
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.png"
            if not gw.render_label("B-07-3-2", p, 0.5, gw.ADDRESS_LABEL):
                self.skipTest("нет python-barcode")
            self.assertEqual([r.text for r in barcode.decode(np.array(Image.open(p).convert("L")))], ["B-07-3-2"])
            q = Path(d) / "dmg.png"
            gw.render_label("00" + gw.make_sscc(random.Random(1)), q, 0.5, gw.PALLET_LABEL, random.Random(2))
            self.assertIsNotNone(zxingcpp)


class TestReadRate(unittest.TestCase):
    def test_metrics(self):
        truth = {"A": "S1", "B": "S2", "C": None, "D": "S4"}
        obs = [Observation("A", "S1"), Observation("B", "S4"),     # B: код D приписан B
               Observation("C", "XXX"), Observation("D", None, None)]
        m = ev.evaluate(truth, obs, targeted={"A", "B", "C", "D"})
        self.assertAlmostEqual(m["read_rate"], 1 / 3)
        self.assertAlmostEqual(m["attribution"], 1 / 3)
        self.assertEqual((m["misattributed"], m["garbage"]), (1, 1))


if __name__ == "__main__":
    unittest.main()

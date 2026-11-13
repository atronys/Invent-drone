"""Сквозные проверки: автомат сканирования в замкнутом цикле, сервер оператора, CLI."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rackscan import config, selftest
from rackscan.layout import Layout
from rackscan.optics import camera_from_preset
from rackscan.scanner import FINISHED, IDLE, NAVIGATE, SEARCH, Frame, ScanConfig, Scanner
from rackscan.wms.demo_data import build_demo

SMALL = (1152, 648)          # уменьшенный предпросмотр — чтобы тесты шли быстрее
STILL = (2304, 1296)


def pick_cases(expected: dict) -> list[str]:
    """По одной ячейке на каждый вид исхода учебного склада."""
    seen, out = set(), []
    for addr, e in sorted(expected.items()):
        key = (e["scan"], e["verdict"], tuple(e["flags"]))
        if key not in seen:
            seen.add(key)
            out.append(addr)
    return out


class ClosedLoopTest(unittest.TestCase):
    def test_every_outcome_kind(self):
        """Имитатор носителя выполняет команды стенда; исход каждой ячейки совпадает с заложенным."""
        _, _, expected = build_demo(seed=7)
        cases = pick_cases(expected)
        self.assertGreaterEqual(len(cases), 8)
        with tempfile.TemporaryDirectory() as d:
            res = selftest.run(seed=7, preview_size=SMALL, still_size=STILL, addresses=cases, out_dir=d,
                               max_steps=60 * len(cases))
            self.assertEqual(res["mismatches"], [])
            self.assertTrue(res["ok"])
            session = next(Path(d).iterdir())
            photos = [p for p in session.glob("*.jpg") if "_annot" not in p.name]
            self.assertEqual(len(photos), len(cases))                    # фото на каждую ячейку, включая пустые
            kinds = [json.loads(x)["kind"] for x in (session / "journal.jsonl").read_text("utf-8").splitlines()]
            self.assertEqual(kinds[0], "task_started")
            self.assertEqual(kinds[-1], "task_finished")
            self.assertEqual(kinds.count("result"), len(cases))
        s = res["summary"]
        self.assertEqual(s["scanned"], len(cases))
        self.assertEqual(s["counts"]["MANUAL"], 2)
        # среднее число шагов на ячейку: наведение сходится, а не блуждает
        self.assertLess(res["steps"] / len(cases), 30)


class ScannerUnitTest(unittest.TestCase):
    def setUp(self):
        self.lay = Layout.demo()
        self.wms, self.scene, self.expected = build_demo(self.lay, seed=7)
        self.cam = camera_from_preset("imx708", SMALL)
        self.sc = Scanner(self.lay, self.cam, self.cam, self.wms, None, ScanConfig(done_hold_s=0.0))

    def _frame(self, p, yaw=0.0, t=0.0, **kw):
        return Frame(self.scene.render(self.cam, p, yaw), t, **kw)

    def test_idle_until_started(self):
        snap = self.sc.step(self._frame((0.6, 0.6, 1.0)))
        self.assertEqual((snap["phase"], snap["status"]), (IDLE, "IDLE"))
        self.assertGreater(len(snap["visible"]), 3)

    def test_search_when_no_markers(self):
        self.sc.start_task(["01-01-01"])
        blank = Frame(np.full((SMALL[1], SMALL[0], 3), 128, np.uint8), 0.0)
        snap = self.sc.step(blank)
        self.assertEqual(snap["status"], "RED")
        self.assertIn("НАВЕДИТЕ КАМЕРУ НА СТЕЛЛАЖ 01", snap["headline"])

    def test_first_command_points_to_target(self):
        self.sc.start_task(["01-02-02"])                 # центр ячейки: x = 0.635 м, y = 0.72 м
        snap = self.sc.step(self._frame((0.2, 0.72, 0.78)))
        self.assertEqual(snap["phase"], NAVIGATE)
        self.assertEqual(snap["commands"][0]["text"], "ВПРАВО")
        self.assertAlmostEqual(snap["commands"][0]["value"], 43.5, delta=2.0)
        self.assertGreater(snap["velocity"]["vy_right"], 0.25)

    def test_blind_count_hides_expected(self):
        self.sc.start_task(["01-02-02"])
        self.assertIsNone(self.sc.step(self._frame((0.2, 0.72, 0.78)))["expected"])
        self.sc.cfg.blind = False
        self.assertEqual(len(self.sc.step(self._frame((0.2, 0.72, 0.78), t=0.2))["expected"]), 1)

    def test_skip_and_rescan(self):
        self.sc.start_task(["01-02-02", "01-02-01"])
        self.sc.step(self._frame((0.2, 0.72, 0.78)))
        first = self.sc.target
        self.sc.skip()
        self.assertEqual(self.sc.results[str(first)]["scan_status"], "SKIPPED")
        self.assertEqual(self.wms.report(self.sc.task.id)["summary"]["counts"]["NOT_SCANNED"], 2)
        self.sc.rescan(str(first))
        self.assertNotIn(str(first), self.sc.results)
        self.assertEqual(self.sc.queue[0], first)

    def test_timeout_skips_cell(self):
        self.sc.cfg.cell_timeout_s = 5.0
        self.sc.start_task(["01-02-02"])
        self.sc.step(self._frame((0.2, 0.72, 0.78), t=0.0))
        snap = self.sc.step(self._frame((0.2, 0.72, 0.78), t=6.0))
        self.assertEqual(snap["results"]["01-02-02"]["scan_status"], "SKIPPED")
        self.assertEqual(self.sc.step(self._frame((0.2, 0.72, 0.78), t=6.2))["phase"], FINISHED)

    def test_wms_failure_does_not_lose_result(self):
        class Broken:
            def cells(self):
                return [str(a) for a in Layout.demo().addresses()]

            def create_task(self, addresses=None):
                from rackscan.wms.base import InventoryTask, TaskLine
                return InventoryTask("7", "", "open", [TaskLine(a) for a in addresses])

            def post_result(self, task_id, result):
                raise ConnectionError("сеть упала")

            def close_task(self, task_id):
                return {}
        sc = Scanner(self.lay, self.cam, self.cam, Broken(), None, ScanConfig(done_hold_s=0.0))
        sc.start_task(["01-02-02"])
        sc.step(self._frame((0.2, 0.72, 0.78)))
        sc.skip()
        r = sc.results["01-02-02"]
        self.assertIsNone(r["verdict"])
        self.assertIn("WMS недоступна", r["verdict_note"])

    def test_free_mode_takes_cell_in_view(self):
        self.sc.start_task(mode="free")
        snap = self.sc.step(self._frame((1.035, 1.12, 0.78)))      # напротив ячейки 01-03-03
        self.assertEqual(snap["target"], "01-03-03")

    def test_other_row_is_reported(self):
        from rackscan.layout import Rack
        lay = Layout.build([Rack(id=1), Rack(id=2, row=2)])
        sc = Scanner(lay, self.cam, self.cam, None, None)
        sc.start_task(["02-01-01"])
        snap = sc.step(self._frame((0.6, 0.6, 1.0)))               # сцена показывает метки стеллажа 01
        self.assertEqual(snap["phase"], SEARCH)
        self.assertIn("ПЕРЕЙДИТЕ К СТЕЛЛАЖУ 02", snap["headline"])


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rackscan.server import Runtime, create_app
        cls.tmp = tempfile.TemporaryDirectory()
        cfg = config.load(None, {
            "camera": {"source": "synthetic", "preview_size": list(SMALL), "still_size": list(STILL), "fps": 100},
            "wms": {"db": ":memory:"}, "evidence_dir": cls.tmp.name + "/ev", "references_dir": cls.tmp.name + "/ref",
            "scan": {"done_hold_s": 0.0}})
        cls.rt = Runtime(cfg)
        cls.rt.source.period = 0.0
        cls.client = create_app(cls.rt).test_client()

    @classmethod
    def tearDownClass(cls):
        cls.rt.stop()
        cls.tmp.cleanup()

    def test_1_page_and_state(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("RackScan — пульт оператора", r.get_data(as_text=True))
        r.close()
        self.assertTrue(self.rt.step_once())
        s = self.client.get("/api/state").get_json()
        self.assertEqual((s["phase"], s["sim"], s["source"]), ("IDLE", True, "SyntheticSource"))
        lay = self.client.get("/api/layout").get_json()
        self.assertEqual(lay["racks"][0]["addresses"][0][0], "01-03-01")         # верхний ярус — первой строкой
        self.assertEqual(self.client.get("/frame.jpg").data[:2], b"\xff\xd8")
        self.assertEqual(self.client.get("/report").status_code, 404)

    def test_2_run_task_with_autopilot(self):
        self.assertEqual(self.client.post("/api/start", json={"addresses": ["01-02-01", "02-03-03"]}).get_json()["cells"], 2)
        self.assertTrue(self.client.post("/api/sim", json={"autopilot": True}).get_json()["autopilot"])
        for _ in range(150):
            self.rt.step_once()
            if self.rt.snap["phase"] == FINISHED:
                break
        s = self.client.get("/api/state").get_json()
        self.assertEqual(s["phase"], FINISHED)
        self.assertEqual(s["status"], "GREEN")
        self.assertEqual({k: v["scan_status"] for k, v in s["results"].items()},
                         {a: self.rt.expected[a]["scan"] for a in ("01-02-01", "02-03-03")})
        rep = self.client.get("/api/report").get_json()
        self.assertEqual(rep["summary"]["scanned"], 2)
        self.assertEqual(rep["status"], "closed")
        self.assertIn("Сличительная ведомость", self.client.get("/report").get_data(as_text=True))
        csv = self.client.get("/report.csv")
        self.assertIn("attachment", csv.headers["Content-Disposition"])
        photo = s["last_result"]["photo"]
        r = self.client.get("/evidence/" + photo)
        self.assertEqual(r.data[:2], b"\xff\xd8")
        r.close()
        self.assertEqual(self.client.get("/evidence/../../secret").status_code, 404)
        kinds = [e["kind"] for e in self.client.get("/api/journal?last=50").get_json()]
        self.assertIn("task_finished", kinds)

    def test_3_wms_service_is_mounted(self):
        r = self.client.get("/wms/hs/rackscan/v1/cells").get_json()
        self.assertEqual(len(r["cells"]), 18)

    def test_4_bad_commands(self):
        self.assertEqual(self.client.post("/api/nope").status_code, 404)
        self.assertEqual(self.client.post("/api/rescan", json={"address": "абв"}).status_code, 400)


class ConfigCliTest(unittest.TestCase):
    def test_scan_config_overrides(self):
        cfg = config.load(None, {"label": {"code_mm": 30, "module_mm": 1.2},
                                 "scan": {"always_approach": False, "fine": {"xy_m": 0.01},
                                          "occ": {"max_gap_mm": 50, "cardboard_h": [3, 30]}}})
        sc = config.scan_config(cfg)
        self.assertEqual((sc.label_code_mm, sc.label_module_mm, sc.always_approach), (30.0, 1.2, False))
        self.assertEqual((sc.fine.xy_m, sc.occ.max_gap_mm, sc.occ.cardboard_h), (0.01, 50, (3, 30)))
        with self.assertRaises(KeyError):
            config.scan_config(config.load(None, {"scan": {"нет_такого": 1}}))

    def test_cli_gen_layout_and_wms(self):
        from rackscan.cli import main
        with tempfile.TemporaryDirectory() as d:
            lay = f"{d}/lay.json"
            self.assertEqual(main(["gen-layout", "--out", lay, "--racks", "1", "--levels", "2", "--cells", "4",
                                   "--cell-w", "300", "--marker", "40"]), 0)
            got = Layout.load(lay)
            self.assertEqual((len(got.addresses()), got.marker_size_mm, got.racks[1].cell_width_mm), (8, 40.0, 300.0))
            db, csv = f"{d}/w.sqlite", f"{d}/stock.csv"
            self.assertEqual(main(["wms", "init-demo", "--db", db]), 0)
            self.assertEqual(main(["wms", "export", "--db", db, "--file", csv]), 0)
            self.assertIn("01-01-01;", Path(csv).read_text("utf-8-sig"))
            self.assertEqual(main(["optics", "--camera", "imx708"]), 0)


if __name__ == "__main__":
    unittest.main()

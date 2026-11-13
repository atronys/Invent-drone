"""Эмулятор склада, сверка, HTTP-сервис в стиле 1С, клиент 1С и чтение OData."""
import base64
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from flask import Flask
from werkzeug.serving import make_server

from rackscan.wms import reconcile
from rackscan.wms.demo_data import build_demo
from rackscan.wms.emulator import EmulatorBackend
from rackscan.wms.http_api import make_blueprint
from rackscan.wms.onec import ODataStockReader, OneCHttpBackend, WmsError
from rackscan.wms.reconcile import ScanStatus as S
from rackscan.wms.reconcile import Verdict as V


def small_wms() -> EmulatorBackend:
    w = EmulatorBackend()
    for i, a in enumerate(("01-01-01", "01-01-02", "01-01-03"), 1):
        w.add_cell(a, 1, 1, i)
    w.add_item("АРТ-1", "Чай")
    w.put("SSCC-A", "АРТ-1", "01-01-01")
    w.put("SSCC-B", "АРТ-1", "01-01-02")
    return w


class VerdictTest(unittest.TestCase):
    def test_table(self):
        cases = [
            (S.READ, ["A"], ["A"], V.MATCH), (S.EMPTY, [], [], V.MATCH),
            (S.EMPTY, [], ["A"], V.SHORTAGE), (S.READ, ["A"], [], V.SURPLUS),
            (S.READ, ["B"], ["A"], V.MISMATCH), (S.READ, ["A", "C"], ["A", "B"], V.PARTIAL),
            (S.NO_LABEL, [], ["A"], V.MANUAL), (S.READ_ERROR, [], [], V.MANUAL),
            (S.UNCERTAIN, [], [], V.MANUAL), (S.SKIPPED, [], ["A"], V.NOT_SCANNED), (None, [], [], V.NOT_SCANNED),
        ]
        for status, found, expected, want in cases:
            got, _ = reconcile.verdict_for(status.value if status else None, found, expected)
            self.assertEqual(got, want, (status, found, expected))

    def test_partial_note_names_both_sides(self):
        _, note = reconcile.verdict_for("READ", ["A", "C"], ["A", "B"])
        self.assertIn("лишние: C", note)
        self.assertIn("не найдены: B", note)

    def test_ira_ignores_undecided(self):
        lines = [{"verdict": v} for v in ("MATCH", "MATCH", "MATCH", "SHORTAGE", "MANUAL", "NOT_SCANNED")]
        s = reconcile.summarize(lines)
        self.assertEqual((s["decided"], s["scanned"]), (4, 5))
        self.assertAlmostEqual(s["ira"], 0.75)
        self.assertIsNone(reconcile.summarize([{"verdict": "MANUAL"}])["ira"])


class EmulatorTest(unittest.TestCase):
    def test_task_snapshot_and_results(self):
        w = small_wms()
        task = w.create_task()
        self.assertEqual([ln.address for ln in task.lines], ["01-01-01", "01-01-02", "01-01-03"])
        self.assertEqual(task.lines[0].expected[0].sscc, "SSCC-A")
        # учёт изменился после создания задания — сверка идёт со снимком на момент создания
        w.put("SSCC-NEW", "АРТ-1", "01-01-03")
        r = w.post_result(task.id, {"address": "01-01-03", "scan_status": "EMPTY", "codes": []})
        self.assertEqual(r["verdict"], "MATCH")
        r = w.post_result(task.id, {"address": "01-01-01", "scan_status": "READ", "codes": ["SSCC-B"]})
        self.assertEqual(r["verdict"], "MISMATCH")
        self.assertIn("SSCC-B по учёту в 01-01-02", r["note"])
        rep = w.report(task.id)
        self.assertEqual(rep["summary"]["counts"]["NOT_SCANNED"], 1)
        self.assertEqual(rep["lines"][0]["found"], ["SSCC-B"])

    def test_errors(self):
        w = small_wms()
        with self.assertRaises(KeyError):
            w.create_task(["09-09-09"])
        task = w.create_task(["01-01-01"])
        with self.assertRaises(KeyError):
            w.post_result(task.id, {"address": "01-01-02", "scan_status": "EMPTY"})
        w.close_task(task.id)
        with self.assertRaises(ValueError):
            w.post_result(task.id, {"address": "01-01-01", "scan_status": "EMPTY"})

    def test_csv_roundtrip(self):
        w = small_wms()
        text = w.export_stock_csv()
        self.assertTrue(text.startswith("﻿адрес;sscc"))
        w2 = EmulatorBackend()
        self.assertEqual(w2.import_stock_csv(text), 2)
        self.assertEqual(w2.cells(), w.cells())
        self.assertEqual([(e.address, e.sscc, e.name) for e in w2.stock()],
                         [(e.address, e.sscc, e.name) for e in w.stock()])

    def test_report_exports(self):
        w = small_wms()
        task = w.create_task()
        w.post_result(task.id, {"address": "01-01-01", "scan_status": "READ", "codes": ["SSCC-A"],
                                "flags": ["GAP"], "photo": "s/01.jpg"})
        rep = w.report(task.id)
        csv = reconcile.to_csv(rep["lines"])
        self.assertIn("Адрес ячейки;По учёту (SSCC)", csv)
        self.assertIn("01-01-01;SSCC-A;Чай;SSCC-A;Код прочитан;Совпадает;большой зазор между коробками", csv)
        html = reconcile.to_html(rep)
        self.assertIn("Сличительная ведомость", html)
        self.assertIn('href="/evidence/s/01.jpg"', html)

    def test_demo_is_self_consistent(self):
        wms, scene, expect = build_demo(seed=7)
        truth = scene.truth()
        book = {}
        for e in wms.stock():
            book.setdefault(e.address, []).append(e.sscc)
        kinds = {(v["scan"], v["verdict"]) for v in expect.values()}
        self.assertEqual(kinds, {("READ", "MATCH"), ("EMPTY", "MATCH"), ("EMPTY", "SHORTAGE"), ("READ", "SURPLUS"),
                                 ("READ", "MISMATCH"), ("NO_LABEL", "MANUAL"), ("READ_ERROR", "MANUAL")})
        for addr, e in expect.items():
            # ожидаемый вердикт выводится из физической истины и учёта теми же правилами сверки
            t = truth[addr]
            v, _ = reconcile.verdict_for(e["scan"], t["codes"] if e["scan"] == "READ" else [], book.get(addr, []))
            self.assertEqual(v.value, e["verdict"], addr)
            self.assertEqual(t["occupied"], e["scan"] != "EMPTY", addr)


class HttpServiceTest(unittest.TestCase):
    """Клиент 1С против эмулятора, поднятого как настоящий HTTP-сервер."""

    @classmethod
    def setUpClass(cls):
        cls.wms = small_wms()
        app = Flask(__name__)
        app.register_blueprint(make_blueprint(cls.wms, "robot", "секрет"), url_prefix="/base/hs/rackscan/v1")
        cls.srv = make_server("127.0.0.1", 0, app)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_port}/base/hs/rackscan/v1"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_full_exchange(self):
        c = OneCHttpBackend(self.url, "robot", "секрет")
        self.assertEqual(c.ping()["service"], "rackscan")
        self.assertEqual(c.cells(), ["01-01-01", "01-01-02", "01-01-03"])
        self.assertEqual([e.sscc for e in c.stock("01-01-02")], ["SSCC-B"])
        task = c.create_task(["01-01-01", "01-01-03"])
        self.assertEqual(task.lines[0].expected[0].name, "Чай")
        r = c.post_result(task.id, {"address": "01-01-01", "scan_status": "EMPTY", "codes": []})
        self.assertEqual(r["verdict"], "SHORTAGE")
        self.assertEqual(c.report(task.id)["summary"]["counts"]["SHORTAGE"], 1)
        self.assertEqual(c.close_task(task.id)["status"], "closed")
        with self.assertRaises(WmsError) as cm:
            c.post_result(task.id, {"address": "01-01-01", "scan_status": "EMPTY"})
        self.assertIn("409", str(cm.exception))

    def test_auth_required(self):
        for user, pw in ((None, None), ("robot", "не тот")):
            with self.assertRaises(WmsError) as cm:
                OneCHttpBackend(self.url, user, pw).cells()
            self.assertIn("401", str(cm.exception))

    def test_bad_requests(self):
        c = OneCHttpBackend(self.url, "robot", "секрет")
        with self.assertRaises(WmsError) as cm:
            c.create_task(["77-77-77"])
        self.assertIn("400", str(cm.exception))
        with self.assertRaises(WmsError) as cm:
            c.report("999")
        self.assertIn("404", str(cm.exception))

    def test_unreachable(self):
        with self.assertRaises(WmsError) as cm:
            OneCHttpBackend("http://127.0.0.1:9/x", timeout=0.5).cells()
        self.assertIn("нет связи", str(cm.exception))


class ODataTest(unittest.TestCase):
    """Чтение остатков через стандартный интерфейс OData — против поддельного сервера."""

    @classmethod
    def setUpClass(cls):
        cls.seen = {}

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                u = urlparse(self.path)
                cls.seen.update(path=unquote(u.path), query=parse_qs(u.query), auth=self.headers.get("Authorization"))
                body = json.dumps({"odata.metadata": "…", "value": [
                    {"Ячейка": {"Description": "01-01-01 "}, "Контейнер": {"Code": "SSCC-A"},
                     "Номенклатура": {"Артикул": "АРТ-1", "Description": "Чай"}, "КоличествоBalance": 3},
                    {"Ячейка": {"Description": "01-01-02"}, "Контейнер": None, "КоличествоBalance": 1},
                ]}, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"\xef\xbb\xbf" + body)          # 1С отдаёт JSON с BOM

            def log_message(self, *a):
                pass

        cls.srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_read_stock(self):
        from rackscan.wms.onec import ODATA_MAPPING_EXAMPLE
        r = ODataStockReader(f"http://127.0.0.1:{self.srv.server_port}/base/odata/standard.odata",
                             ODATA_MAPPING_EXAMPLE, "user", "pw")
        stock = r.stock()
        self.assertEqual(len(stock), 1)                             # строка без контейнера пропущена
        e = stock[0]
        self.assertEqual((e.address, e.sscc, e.sku, e.name, e.qty), ("01-01-01", "SSCC-A", "АРТ-1", "Чай", 3.0))
        self.assertEqual(self.seen["path"], "/base/odata/standard.odata/AccumulationRegister_ТоварыВЯчейках/Balance")
        self.assertEqual(self.seen["query"]["$format"], ["json"])
        self.assertIn("Ячейка", self.seen["query"]["$expand"][0])
        self.assertEqual(self.seen["auth"], "Basic " + base64.b64encode(b"user:pw").decode())


if __name__ == "__main__":
    unittest.main()

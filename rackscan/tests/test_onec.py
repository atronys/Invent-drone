"""Подключение к 1С: перевод адресов ячеек и поиск объектов базы по $metadata."""
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from flask import Flask
from werkzeug.serving import make_server

from rackscan.wms.emulator import EmulatorBackend
from rackscan.wms.http_api import make_blueprint
from rackscan.wms.onec import AddressMap, OneCHttpBackend, discover, parse_odata_metadata


class AddressMapTest(unittest.TestCase):
    def test_roundtrip_with_prefix_and_aisle(self):
        m = AddressMap("A-01-{rack:02d}-{level:02d}-{pos:02d}")
        self.assertEqual(m.to_wms("02-03-01"), "A-01-02-03-01")
        self.assertEqual(m.to_stand("A-01-02-03-01"), "02-03-01")
        self.assertIsNone(m.to_stand("A-02-02-03-01"))          # другой проезд — не наш
        self.assertIsNone(m.to_stand("B-01-02-03-01"))

    def test_default_is_identity(self):
        m = AddressMap()
        self.assertEqual((m.to_wms("01-02-03"), m.to_stand("01-02-03")), ("01-02-03", "01-02-03"))

    def test_other_separators(self):
        m = AddressMap("{rack}.{level}.{pos}")
        self.assertEqual((m.to_wms("01-02-03"), m.to_stand("1.2.3")), ("1.2.3", "01-02-03"))
        with self.assertRaises(ValueError):
            AddressMap("{rack}-{level}")

    def test_client_translates_addresses(self):
        """WMS хранит адреса со своим префиксом; стенд видит только свои ячейки и свои адреса."""
        w = EmulatorBackend()
        for a in ("A-01-01-01-01", "A-01-01-01-02", "B-07-01-01-01"):
            w.add_cell(a, 1, 1, 1)
        w.put("SSCC-A", "", "A-01-01-01-01")
        w.put("SSCC-Z", "", "B-07-01-01-01")
        app = Flask(__name__)
        app.register_blueprint(make_blueprint(w), url_prefix="/hs/rackscan/v1")
        srv = make_server("127.0.0.1", 0, app)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            c = OneCHttpBackend(f"http://127.0.0.1:{srv.server_port}/hs/rackscan/v1",
                                address_template="A-01-{rack:02d}-{level:02d}-{pos:02d}")
            self.assertEqual(c.cells(), ["01-01-01", "01-01-02"])
            self.assertEqual([(e.address, e.sscc) for e in c.stock()], [("01-01-01", "SSCC-A")])
            self.assertEqual([e.sscc for e in c.stock("01-01-01")], ["SSCC-A"])
            task = c.create_task()
            self.assertEqual([ln.address for ln in task.lines], ["01-01-01", "01-01-02"])
            self.assertEqual(task.lines[0].expected[0].address, "01-01-01")
            r = c.post_result(task.id, {"address": "01-01-02", "scan_status": "READ", "codes": ["SSCC-Z"]})
            self.assertEqual((r["address"], r["verdict"]), ("01-01-02", "SURPLUS"))
            self.assertIn("B-07-01-01-01", r["note"])            # «где числится» — в адресах WMS
            rep = c.report(task.id)
            self.assertEqual([ln["address"] for ln in rep["lines"]], ["01-01-01", "01-01-02"])
            self.assertEqual(w.report(task.id)["lines"][1]["address"], "A-01-01-01-02")
        finally:
            srv.shutdown()

    def test_task_carries_wms_terms(self):
        w = EmulatorBackend()
        w.add_cell("01-01-01", 1, 1, 1)
        t = w.create_task().to_dict()
        self.assertEqual((t["document"], t["task_type"]), ("Плановая инвентаризация", "Пересчет ячейки"))


META = """<?xml version="1.0" encoding="UTF-8"?>
<edmx:Edmx Version="1.0" xmlns:edmx="http://schemas.microsoft.com/ado/2007/06/edmx">
 <edmx:DataServices><Schema Namespace="StandardODATA" xmlns="http://schemas.microsoft.com/ado/2009/11/edm">
  <EntityType Name="Catalog_Ячейки"><Key><PropertyRef Name="Ref_Key"/></Key>
    <Property Name="Ref_Key" Type="Edm.Guid"/><Property Name="Description" Type="Edm.String"/>
    <NavigationProperty Name="Зона"/></EntityType>
  <EntityType Name="Catalog_Контейнеры"><Property Name="Code" Type="Edm.String"/></EntityType>
  <EntityType Name="AccumulationRegister_ОстаткиТоваров_Balance"><Property Name="КоличествоBalance" Type="Edm.Double"/></EntityType>
  <EntityType Name="Document_ПлановаяИнвентаризация"><Property Name="Number" Type="Edm.String"/></EntityType>
  <EntityType Name="Catalog_Пользователи"><Property Name="Description" Type="Edm.String"/></EntityType>
 </Schema></edmx:DataServices></edmx:Edmx>"""


class DiscoverTest(unittest.TestCase):
    """Имена сущностей в образце придуманы для теста; настоящие даёт только сама база."""

    def test_parse(self):
        ents = parse_odata_metadata(META)
        self.assertEqual(ents["Catalog_Ячейки"], ["Ref_Key", "Description", "Зона"])
        self.assertEqual(len(ents), 5)

    def test_discover_groups_by_keywords(self):
        meta = META.encode("utf-8")

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                ok = self.path.endswith("/$metadata")
                self.send_response(200 if ok else 404)
                self.end_headers()
                if ok:
                    self.wfile.write(b"\xef\xbb\xbf" + meta)

            def log_message(self, *a):
                pass
        srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            found = discover(f"http://127.0.0.1:{srv.server_port}/base/odata/standard.odata")
        finally:
            srv.shutdown()
        self.assertEqual(found["всего сущностей"], 5)
        self.assertEqual([x["entity"] for x in found["ячейки"]], ["Catalog_Ячейки"])
        self.assertEqual([x["entity"] for x in found["контейнеры"]], ["Catalog_Контейнеры"])
        self.assertEqual([x["entity"] for x in found["остатки"]], ["AccumulationRegister_ОстаткиТоваров_Balance"])
        self.assertEqual([x["entity"] for x in found["инвентаризация"]], ["Document_ПлановаяИнвентаризация"])
        self.assertEqual(found["номенклатура"], [])


if __name__ == "__main__":
    unittest.main()

"""Коды, занятость ячейки, датчики, фото-доказательства, печать."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from rackscan import codes, occupancy, printing, sensors
from rackscan.evidence import EvidenceStore
from rackscan.layout import Layout
from rackscan.synth import CARDBOARD, VOID


class PayloadTest(unittest.TestCase):
    def test_check_digit(self):
        self.assertEqual(codes.gs1_check_digit("460123456789"), 3)      # EAN-13 4601234567893
        self.assertTrue(codes.is_valid_gtin("4601234567893"))
        self.assertFalse(codes.is_valid_gtin("4601234567890"))

    def test_make_sscc(self):
        s = codes.make_sscc(1, "4609999", 42)
        self.assertEqual(len(s), 18)
        self.assertTrue(codes.is_valid_sscc(s))
        self.assertFalse(codes.is_valid_sscc(s[:-1] + str((int(s[-1]) + 1) % 10)))

    def test_parse_variants(self):
        s = codes.make_sscc(1, "4609999", 7)
        for text in (s, "00" + s, f"(00){s}", f"]C100{s}"):
            p = codes.parse_payload(text)
            self.assertEqual((p.kind, p.key), ("sscc", s), text)
        p = codes.parse_payload("(01)04601234567893(21)ABC123")
        self.assertEqual((p.kind, p.key), ("gtin", "04601234567893/ABC123"))
        p = codes.parse_payload("0104601234567893" + "21ABC" + codes.GS + "10LOT7")
        self.assertEqual(p.fields, {"01": "04601234567893", "21": "ABC", "10": "LOT7"})
        self.assertEqual(codes.parse_payload("привет").kind, "text")
        self.assertEqual(codes.parse_payload("12345").kind, "text")       # цифры, но не GS1


class CodeImageTest(unittest.TestCase):
    def _label(self, text, fmt, scale=6, pad=30, bg=246):
        sym = codes.encode_symbol(text, fmt)
        if sym.shape[0] == 1:
            sym = np.repeat(sym, 30, axis=0)
        big = np.repeat(np.repeat(sym, scale, axis=0), scale, axis=1)
        return cv2.copyMakeBorder(big, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg)

    def test_roundtrip_formats(self):
        s = "00" + codes.make_sscc(1, "4609999", 5)
        for fmt, text in (("QRCode", s), ("Code128", s[2:]), ("DataMatrix", "0104601234567893")):
            r = codes.read_codes(self._label(text, fmt))
            self.assertEqual([x.text for x in r if x.decoded], [text], fmt)

    def test_offset_and_scale(self):
        img = self._label("00" + codes.make_sscc(1, "4609999", 5), "QRCode")
        base = codes.read_codes(img)[0].center
        up = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        shifted = codes.read_codes(up, offset=(100, 50), scale=2.0)[0].center
        np.testing.assert_allclose(shifted, base + [100, 50], atol=1.5)

    def test_qr_modules_known(self):
        r = codes.read_codes(self._label("00" + codes.make_sscc(1, "4609999", 5), "QRCode"))[0]
        self.assertEqual(r.modules, 21)
        self.assertAlmostEqual(r.px_per_module(), 6.0, delta=0.6)

    def test_no_false_reads_on_texture(self):
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 255, (400, 600), dtype=np.uint8)
        stripes = np.tile((np.arange(600) // 7 % 2 * 255).astype(np.uint8), (400, 1))
        for img in (noise, stripes):
            self.assertEqual(codes.read_codes(img), [])

    def test_sticker_candidate_without_decode(self):
        """Залитый код не читается, но наклейка находится как кандидат; чистый картон — нет."""
        lab = self._label("00" + codes.make_sscc(1, "4609999", 5), "QRCode", scale=5, pad=18)
        h, w = lab.shape
        lab[18:18 + int(0.62 * (h - 36)), 18:18 + int(0.7 * (w - 36))] = 120
        scene = np.full((360, 520, 3), CARDBOARD, np.uint8)
        scene[100:100 + h, 180:180 + w] = lab[..., None]
        self.assertFalse([r for r in codes.read_codes(scene) if r.decoded])
        cands = codes.find_candidates(scene, expected_side_px=w)
        self.assertEqual(len(cands), 1)
        np.testing.assert_allclose(cands[0].center, (180 + w / 2, 100 + h / 2), atol=6)
        plain = np.full((360, 520, 3), CARDBOARD, np.uint8)
        self.assertEqual(codes.find_candidates(plain, expected_side_px=w), [])

    def test_merge_drops_candidate_over_decoded(self):
        img = cv2.cvtColor(self._label("00" + codes.make_sscc(1, "4609999", 5), "QRCode"), cv2.COLOR_GRAY2BGR)
        reads = codes.read_codes(img)
        merged = codes.merge_reads(reads, codes.find_candidates(img))
        self.assertEqual(len(merged), 1)
        self.assertTrue(merged[0].decoded)


def _cell(boxes=(), w_mm=400, h_mm=340, ppm=0.6, void=VOID):
    """Выпрямленный вид ячейки: тёмный проём и картонные коробки (x0, x1, высота, мм)."""
    img = np.full((int(h_mm * ppm), int(w_mm * ppm), 3), void, np.uint8)
    for x0, x1, hh in boxes:
        img[int((h_mm - hh) * ppm):, int(x0 * ppm):int(x1 * ppm)] = CARDBOARD
    return img


class OccupancyTest(unittest.TestCase):
    def test_empty(self):
        r = occupancy.analyze(_cell(), 0.6)
        self.assertIs(r.occupied, False)
        self.assertEqual(r.spans_mm, [])
        self.assertEqual(r.flags, [])

    def test_single_box(self):
        r = occupancy.analyze(_cell([(60, 340, 250)]), 0.6)
        self.assertIs(r.occupied, True)
        self.assertEqual(len(r.spans_mm), 1)
        self.assertAlmostEqual(r.spans_mm[0][0], 60, delta=6)
        self.assertAlmostEqual(r.spans_mm[0][1], 340, delta=6)
        self.assertEqual(r.flags, [])

    def test_gap_between_boxes(self):
        r = occupancy.analyze(_cell([(10, 135, 220), (265, 390, 220)]), 0.6)
        self.assertIs(r.occupied, True)
        self.assertEqual(len(r.spans_mm), 2)
        self.assertAlmostEqual(r.max_gap_between_mm, 130, delta=8)
        self.assertIn("GAP", r.flags)
        tight = occupancy.analyze(_cell([(10, 190, 220), (210, 390, 220)]), 0.6)
        self.assertNotIn("GAP", tight.flags)

    def test_underfilled(self):
        r = occupancy.analyze(_cell([(20, 120, 200)]), 0.6)
        self.assertIs(r.occupied, True)
        self.assertIn("UNDERFILLED", r.flags)

    def test_reference_handles_bright_background(self):
        """Пустая ячейка на фоне светлой стены: эвристика по цвету бессильна, эталон решает."""
        wall = (200, 200, 200)
        ref = cv2.cvtColor(_cell(void=wall), cv2.COLOR_BGR2GRAY)
        empty = occupancy.analyze(_cell(void=wall), 0.6, reference=ref)
        self.assertIs(empty.occupied, False)
        self.assertEqual(empty.method, "reference")
        box = occupancy.analyze(_cell([(60, 340, 250)], void=wall), 0.6, reference=ref)
        self.assertIs(box.occupied, True)
        # освещение стало на 20 % ярче — пустая ячейка остаётся пустой
        brighter = np.clip(_cell(void=wall).astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
        self.assertIs(occupancy.analyze(brighter, 0.6, reference=ref).occupied, False)

    def test_code_wins(self):
        self.assertIs(occupancy.analyze(_cell(), 0.6, has_code=True).occupied, True)

    def test_tof_fusion(self):
        gray = _cell([(60, 340, 250)])
        gray[gray.sum(axis=2) > 300] = (128, 128, 128)          # серая коробка: по цвету её не видно
        alone = occupancy.analyze(gray, 0.6)
        self.assertIs(alone.occupied, False)                    # известное ограничение эвристики…
        self.assertLessEqual(alone.confidence, 0.6)             # …поэтому уверенность низкая
        near = occupancy.analyze(gray, 0.6, tof_delta_mm=20, cell_depth_mm=300)
        self.assertEqual((near.occupied, near.method), (True, "tof"))    # дальномер исправляет
        self.assertIs(occupancy.analyze(gray, 0.6, tof_delta_mm=290, cell_depth_mm=300).occupied, False)
        # с эталоном спор «камера: пусто, дальномер: близко» не решается — честное «не знаю»
        import cv2 as _cv
        ref = _cv.cvtColor(_cell(), _cv.COLOR_BGR2GRAY)
        self.assertIsNone(occupancy.analyze(_cell(), 0.6, reference=ref, tof_delta_mm=20).occupied)
        # коробка узкая, луч дальномера прошёл мимо: камере верим
        self.assertIs(occupancy.analyze(_cell([(20, 120, 200)]), 0.6, tof_delta_mm=290).occupied, True)


class SensorsTest(unittest.TestCase):
    BODY = "RSI,123456,12,-980,150,250,-1200,30,-150,275,812,1"

    def test_parse(self):
        r = sensors.parse_line(sensors.frame(self.BODY))
        self.assertEqual(r.t_ms, 123456)
        self.assertEqual(r.accel_g, (0.012, -0.98, 0.15))
        self.assertEqual(r.gyro_dps, (2.5, -12.0, 0.3))
        self.assertEqual((r.roll_deg, r.pitch_deg, r.tof_mm, r.button), (-1.5, 2.75, 812.0, True))
        self.assertAlmostEqual(r.gyro_norm, (2.5 ** 2 + 12 ** 2 + 0.3 ** 2) ** 0.5)

    def test_out_of_range_distance(self):
        r = sensors.parse_line(sensors.frame("RSI,1,0,0,1000,0,0,0,0,0,-1,0"))
        self.assertIsNone(r.tof_mm)

    def test_rejects_garbage(self):
        good = sensors.frame(self.BODY)
        bad = [good.replace("812", "813"), good.replace("$", ""), "$RSI,1,2*00", "", "ets Jul 29 2019 12:21:46",
               sensors.frame("RSX,1,2,3"), sensors.frame("RSI,1,2,3,4,5,6,7,8,9,x,0")]
        for line in bad:
            self.assertIsNone(sensors.parse_line(line), line)

    def test_command_frame(self):
        self.assertEqual(sensors.frame("RSC,LED,255,0,0"), "$RSC,LED,255,0,0*%s\r\n" % sensors.checksum("RSC,LED,255,0,0"))

    def test_hub_without_port_is_harmless(self):
        hub = sensors.SensorHub("НЕТ_ТАКОГО_ПОРТА")
        self.assertFalse(hub.available)
        self.assertIsNone(hub.latest())
        hub.set_status("RED")
        hub.close()


class EvidenceTest(unittest.TestCase):
    def test_save_and_journal(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            ev = EvidenceStore(d, session="s1")
            img = np.full((120, 160, 3), 90, np.uint8)
            out = ev.save_result({"address": "01-01-01", "scan_status": "EMPTY", "codes": [], "flags": []}, img, img)
            raw = (Path(d) / out["photo"]).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), out["photo_sha256"])
            self.assertTrue((Path(d) / out["photo_annotated"]).exists())
            passport = json.loads((Path(d) / out["photo"]).with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(passport["photo_sha256"], out["photo_sha256"])
            ev.journal("note", text="проверка")
            j = ev.read_journal()
            self.assertEqual([e["seq"] for e in j], [1, 2])
            self.assertEqual((j[0]["kind"], j[1]["text"]), ("result", "проверка"))


class PrintingTest(unittest.TestCase):
    def test_marker_printed_at_true_size_and_readable(self):
        from rackscan import markers
        lay = Layout.demo()
        pages = printing.address_cards(lay, dpi=150)
        self.assertEqual(pages[0].size, (1240, 1754))                    # A4 при 150 dpi
        dets = markers.detect(np.asarray(pages[0]), lay.marker_dictionary, max_width=None)
        ids = {d.id for d in dets}
        self.assertTrue(ids <= set(lay.marker_ids))
        self.assertEqual(len(ids), 6)                                    # шесть карточек на листе
        self.assertEqual(len(pages), 3)
        side_mm = np.mean([d.side_px for d in dets]) / 150 * 25.4
        self.assertAlmostEqual(side_mm, lay.marker_size_mm, delta=0.6)

    def test_box_labels_decode(self):
        s = codes.make_sscc(1, "4609999", 9)
        pages = printing.box_labels([{"text": "00" + s, "title": printing.sscc_hri(s)}], dpi=150)
        got = [r.text for r in codes.read_codes(np.asarray(pages[0])) if r.decoded]
        self.assertEqual(got, ["00" + s])
        self.assertEqual(printing.sscc_hri(s)[:6], "(00) 1")


if __name__ == "__main__":
    unittest.main()

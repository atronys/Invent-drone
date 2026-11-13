"""Синтетический стеллаж: виртуальная камера для разработки и испытаний без железа.

Сцена рисуется заново для каждого положения камеры: все элементы — многоугольники в системе
фасада, которые проецируются той же пинхол-моделью, что и в pose.py. Поэтому истинная поза
камеры известна точно, а ошибки оценки позы и наведения можно измерять.

Упрощения (важно помнить при переносе на реальный стенд): нет бликов плёнки, теней,
перекрытий; фронт коробок утоплен на постоянную величину; свет равномерный.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import codes, markers
from .layout import Address, Layout
from .optics import CameraModel
from .pose import FacadePose

# цвета BGR
WALL = (206, 204, 198)
VOID = (40, 38, 36)
UPRIGHT = (120, 84, 44)
BEAM = (122, 96, 74)
CARDBOARD = (98, 142, 186)
TAPE = (150, 185, 215)
STICKER = (246, 246, 246)
CARD = (250, 250, 250)


@dataclass
class LabelSpec:
    text: str
    fmt: str = "QRCode"          # QRCode | Code128 | DataMatrix
    code_mm: float = 40.0        # сторона символа (2D) или ширина (линейный код)
    damaged: bool = False        # залита часть символа: наклейка видна, код не читается
    dx_mm: float = 0.0           # смещение центра наклейки от центра фронта коробки
    dy_mm: float = 0.0


@dataclass
class BoxSpec:
    x_mm: float                  # левый край коробки от левого края ячейки
    width_mm: float
    height_mm: float
    label: LabelSpec | None = None
    setback_mm: float = 20.0     # насколько фронт коробки утоплен от кромки полки


@dataclass
class Scene:
    layout: Layout
    cells: dict[Address, list[BoxSpec]] = field(default_factory=dict)
    seed: int = 1

    # --- поза виртуальной камеры --------------------------------------------------
    @staticmethod
    def camera_pose(cam: CameraModel, position_m, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
                    rack_id: int = 0) -> FacadePose:
        """Истинная поза по положению (x, y, z в системе фасада, м) и углам (см. pose.py)."""
        y, p, r = (math.radians(a) for a in (yaw_deg, pitch_deg, roll_deg))
        Ry = np.array([[math.cos(-y), 0, math.sin(-y)], [0, 1, 0], [-math.sin(-y), 0, math.cos(-y)]])
        Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
        Rz = np.array([[math.cos(-r), -math.sin(-r), 0], [math.sin(-r), math.cos(-r), 0], [0, 0, 1]])
        C = Ry @ np.diag([1.0, -1.0, -1.0]) @ Rx @ Rz      # столбцы — оси камеры в системе фасада
        R = C.T
        t = -R @ np.asarray(position_m, dtype=np.float64)
        return FacadePose(rack_id, R, t, cam, 0, 0.0)

    # --- рисование ----------------------------------------------------------------
    def render(self, cam: CameraModel, position_m, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
               noise: float = 2.5, blur_px: float = 0.6, motion_px: float = 0.0, gain: float = 1.0) -> np.ndarray:
        pose = self.camera_pose(cam, position_m, yaw_deg, pitch_deg, roll_deg)
        img = np.empty((cam.height, cam.width, 3), np.uint8)
        img[:] = WALL
        rng = np.random.default_rng(self.seed)

        def proj(pts_mm):
            pts = np.asarray(pts_mm, dtype=np.float64).reshape(-1, 3) / 1000.0
            depth = (pose.R @ pts.T).T[:, 2] + pose.t[2]
            if depth.min() < 0.05:
                return None
            px = pose.project(pts)
            if px[:, 0].max() < -50 or px[:, 0].min() > cam.width + 50 or \
                    px[:, 1].max() < -50 or px[:, 1].min() > cam.height + 50:
                return None
            return px

        def rect(x0, y0, x1, y1, z=0.0):
            return [[x0, y1, z], [x1, y1, z], [x1, y0, z], [x0, y0, z]]   # ЛВ, ПВ, ПН, ЛН

        def fill(pts_mm, color):
            px = proj(pts_mm)
            if px is not None:
                cv2.fillConvexPoly(img, np.round(px * 16).astype(np.int32), color, cv2.LINE_AA, 4)
            return px

        def paste(tex, pts_mm):
            """Наложить текстуру (серую или BGR) на четырёхугольник фасада."""
            px = proj(pts_mm)
            if px is None:
                return
            x0, y0 = np.floor(px.min(axis=0)).astype(int) - 2
            x1, y1 = np.ceil(px.max(axis=0)).astype(int) + 2
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, cam.width), min(y1, cam.height)
            if x1 - x0 < 2 or y1 - y0 < 2:
                return
            h, w = tex.shape[:2]
            src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
            M = cv2.getPerspectiveTransform(src, (px - [x0, y0]).astype(np.float32))
            side = float(np.linalg.norm(px[0] - px[1]))
            interp = cv2.INTER_AREA if side < w else cv2.INTER_NEAREST if side > 3 * w else cv2.INTER_LINEAR
            if tex.ndim == 2:
                tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
            warped = cv2.warpPerspective(tex, M, (x1 - x0, y1 - y0), flags=interp)
            mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
            a = (mask.astype(np.float32) / 255.0)[..., None]
            region = img[y0:y1, x0:x1].astype(np.float32)
            img[y0:y1, x0:x1] = (region * (1 - a) + warped.astype(np.float32) * a).astype(np.uint8)

        lay = self.layout
        ms = lay.marker_size_mm
        for rack in lay.racks.values():
            ox = rack.origin_x_mm
            # проёмы ячеек: тёмная «пустота»
            for a in rack.cells():
                x0, y0, x1, y1 = rack.cell_rect_mm(a)
                fill(rect(x0, y0, x1, y1), VOID)
            # коробки и наклейки
            for a in rack.cells():
                x0, y0, x1, y1 = rack.cell_rect_mm(a)
                for b in self.cells.get(a, []):
                    z = -b.setback_mm
                    bx0, bx1 = x0 + b.x_mm, x0 + b.x_mm + b.width_mm
                    by1 = y0 + b.height_mm
                    fill(rect(bx0, y0, bx1, by1, z), CARDBOARD)
                    fill(rect(bx0, by1 - 0.52 * b.height_mm, bx1, by1 - 0.46 * b.height_mm, z), TAPE)
                    if b.label is not None:
                        self._draw_label(b.label, (bx0 + bx1) / 2 + b.label.dx_mm,
                                         y0 + b.height_mm * 0.45 + b.label.dy_mm, z, fill, paste, rect, rng)
            # стойки и кромки полок — поверх коробок
            fill(rect(ox, 0, ox + rack.upright_width_mm, rack.height_mm), UPRIGHT)
            fill(rect(ox + rack.width_mm - rack.upright_width_mm, 0, ox + rack.width_mm, rack.height_mm), UPRIGHT)
            for sy in rack.shelf_y_mm:
                fill(rect(ox, sy - rack.beam_height_mm, ox + rack.width_mm, sy), BEAM)
            top = rack.shelf_y_mm[-1] + rack.cell_height_mm
            fill(rect(ox, top, ox + rack.width_mm, top + rack.beam_height_mm), BEAM)
            # адресные карточки с метками
            bits = markers.dictionary_bits(lay.marker_dictionary)
            for a in rack.cells():
                cx, cy = rack.marker_center_mm(a)
                pad = ms / (bits + 2) * 1.2                       # белое поле вокруг метки
                fill(rect(cx - ms / 2 - pad, cy - ms / 2 - pad, cx + ms / 2 + pad, cy + ms / 2 + pad), CARD)
                tex = markers.render(lay.marker_dictionary, lay.marker_id(a), (bits + 2) * 12)
                paste(tex, rect(cx - ms / 2, cy - ms / 2, cx + ms / 2, cy + ms / 2))

        out = img.astype(np.float32) * gain
        if blur_px > 0:
            out = cv2.GaussianBlur(out, (0, 0), blur_px)
        if motion_px >= 1.0:
            k = np.zeros((1, int(round(motion_px)) | 1), np.float32)
            k[:] = 1.0 / k.size
            out = cv2.filter2D(out, -1, k)
        if noise > 0:
            out += np.random.default_rng().normal(0, noise, out.shape).astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _draw_label(lab: LabelSpec, cx: float, cy: float, z: float, fill, paste, rect, rng) -> None:
        sym = codes.encode_symbol(lab.text, lab.fmt)
        linear = sym.shape[0] == 1
        if linear:
            cw, ch = lab.code_mm, lab.code_mm * 0.32
            quiet = cw / sym.shape[1] * 10
            tex = np.repeat(np.repeat(sym, 4, axis=1), 40, axis=0)
        else:
            cw = ch = lab.code_mm
            quiet = cw / sym.shape[1] * 4
            tex = np.repeat(np.repeat(sym, 8, axis=0), 8, axis=1)
        if lab.damaged:   # залить половину символа: наклейка остаётся, код не декодируется
            tex = tex.copy()
            h, w = tex.shape
            tex[: int(h * 0.62), : int(w * 0.7)] = 120
            for _ in range(6):
                x, y = int(rng.integers(0, w - w // 4)), int(rng.integers(0, h - h // 4))
                tex[y:y + h // 4, x:x + w // 4] = int(rng.integers(60, 200))
        sw, sh = cw + 2 * quiet + 6, ch + 2 * quiet + 16
        fill(rect(cx - sw / 2, cy - sh / 2, cx + sw / 2, cy + sh / 2, z), STICKER)
        top = cy + sh / 2 - quiet - 3
        paste(tex, rect(cx - cw / 2, top - ch, cx + cw / 2, top, z))
        # две «строки текста» под кодом
        for i, frac in enumerate((0.8, 0.55)):
            y = top - ch - 4 - i * 4.5
            fill(rect(cx - cw / 2, y - 2.2, cx - cw / 2 + cw * frac, y, z), (70, 70, 70))

    # --- дальномер ----------------------------------------------------------------
    def tof_mm(self, position_m, yaw_deg=0.0, pitch_deg=0.0, max_range_mm: float = 4000.0) -> float | None:
        """Имитация дальномера ToF, смотрящего вдоль оптической оси. None — вне диапазона."""
        y, p = math.radians(yaw_deg), math.radians(pitch_deg)
        a = np.array([math.sin(y) * math.cos(p), math.sin(p), -math.cos(y) * math.cos(p)])
        c = np.asarray(position_m, dtype=np.float64) * 1000.0
        if a[2] >= -1e-6:
            return None

        def hit(z_mm):
            s = (z_mm - c[2]) / a[2]
            return s, c + s * a

        s0, q = hit(0.0)
        dist = None
        for rack in self.layout.racks.values():
            if not (rack.origin_x_mm <= q[0] <= rack.origin_x_mm + rack.width_mm and 0 <= q[1] <= rack.height_mm):
                continue
            dist = s0   # по умолчанию — конструкция стеллажа
            for adr in rack.cells():
                x0, y0, x1, y1 = rack.cell_rect_mm(adr)
                if x0 <= q[0] <= x1 and y0 <= q[1] <= y1:
                    dist = hit(-rack.cell_depth_mm)[0]
                    for b in self.cells.get(adr, []):
                        sb, qb = hit(-b.setback_mm)
                        if x0 + b.x_mm <= qb[0] <= x0 + b.x_mm + b.width_mm and y0 <= qb[1] <= y0 + b.height_mm:
                            dist = sb
                    break
        return None if dist is None or dist > max_range_mm else float(dist)

    # --- истина ------------------------------------------------------------------
    def truth(self) -> dict[str, dict]:
        """Что физически стоит в каждой ячейке (для сравнения с результатом сканирования)."""
        out = {}
        for a in self.layout.addresses():
            boxes = self.cells.get(a, [])
            readable = [codes.parse_payload(b.label.text).key for b in boxes if b.label and not b.label.damaged]
            out[str(a)] = {
                "occupied": bool(boxes),
                "codes": sorted(readable),
                "damaged": sum(1 for b in boxes if b.label and b.label.damaged),
                "unlabeled": sum(1 for b in boxes if b.label is None),
            }
        return out

    def label_point_mm(self, a: Address, index: int = 0) -> tuple[float, float]:
        """Центр наклейки index-й коробки ячейки в системе фасада, мм."""
        rack = self.layout.rack_of(a)
        x0, y0, _, _ = rack.cell_rect_mm(a)
        b = self.cells[a][index]
        return x0 + b.x_mm + b.width_mm / 2 + (b.label.dx_mm if b.label else 0), \
            y0 + b.height_mm * 0.45 + (b.label.dy_mm if b.label else 0)

"""Занятость ячейки и зазоры между коробками по выпрямленному виду проёма.

Вход — фронтальный вид ячейки (pose.rectify), масштаб известен в px/мм, поэтому ширина
зазоров получается сразу в миллиметрах.

Признаки занятости, от сильного к слабому:
    1. в ячейке прочитан или найден код                       -> занято;
    2. дальномер ToF: до препятствия ближе глубины ячейки      -> занято, дальше -> пусто;
    3. отличие от эталона пустой ячейки (снят при настройке)   -> маска «не фон»;
    4. без эталона — цвет картона/наклейки (грубая эвристика, уверенность низкая: коробку
       не картонного цвета она примет за пустоту, поэтому на реальном стенде нужен эталон
       или дальномер).
Если признаки не дают уверенного ответа, занятость остаётся неизвестной (None): ячейку
без прочитанной этикетки нельзя честно назвать пустой.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class OccupancyConfig:
    occupied_fill: float = 0.04      # ячейка занята, если найден занятый участок и «не фона» не меньше этой доли
    empty_fill: float = 0.06         # ниже — пусто
    diff_threshold: int = 28         # порог отличия от эталона, уровни серого
    column_fill: float = 0.45        # столбец занят, если заполнен на эту долю высоты зоны анализа
    min_span_mm: float = 25.0        # короче — шум, не коробка
    max_gap_mm: float = 90.0         # зазор между соседними коробками больше — флаг GAP
    max_free_ratio: float = 0.65     # свободная доля ширины занятой ячейки больше — флаг UNDERFILLED
    bottom_part: float = 0.7         # анализируется нижняя часть проёма (коробки стоят на настиле)
    cardboard_h: tuple[int, int] = (4, 32)   # диапазон тона картона (OpenCV: 0..179)
    cardboard_s_min: int = 45
    cardboard_v_min: int = 70


@dataclass
class OccupancyResult:
    occupied: bool | None
    confidence: float
    fill: float
    method: str
    spans_mm: list[tuple[float, float]] = field(default_factory=list)   # занятые участки по ширине
    gaps_mm: list[tuple[float, float]] = field(default_factory=list)    # свободные участки
    max_gap_between_mm: float = 0.0
    free_ratio: float = 1.0
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "occupied": self.occupied, "confidence": round(self.confidence, 2), "fill": round(self.fill, 3),
            "method": self.method, "spans_mm": [[round(a), round(b)] for a, b in self.spans_mm],
            "max_gap_between_mm": round(self.max_gap_between_mm), "free_ratio": round(self.free_ratio, 2),
            "flags": list(self.flags),
        }


def foreground_mask(roi_bgr: np.ndarray, cfg: OccupancyConfig, reference: np.ndarray | None = None) -> np.ndarray:
    """Маска «не фон» (uint8 0/255) того же размера, что и вырез."""
    if reference is not None:
        g = cv2.GaussianBlur(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
        ref = reference if reference.ndim == 2 else cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        ref = cv2.resize(ref, (g.shape[1], g.shape[0])).astype(np.float32)
        ref = cv2.GaussianBlur(ref, (5, 5), 0)
        # компенсация общего изменения освещённости: приводим медианы
        gain = (np.median(g) + 1.0) / (np.median(ref) + 1.0)
        if 0.5 < gain < 2.0:
            ref = ref * gain
        mask = (np.abs(g - ref) > cfg.diff_threshold).astype(np.uint8) * 255
    else:
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        cardboard = (h >= cfg.cardboard_h[0]) & (h <= cfg.cardboard_h[1]) & \
                    (s >= cfg.cardboard_s_min) & (v >= cfg.cardboard_v_min)
        sticker = (s < 40) & (v > 185)
        mask = (cardboard | sticker).astype(np.uint8) * 255
    k = max(3, (min(mask.shape) // 40) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    return mask


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Отрезки подряд идущих True: [(начало, конец_не_включая)]."""
    runs, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def analyze(roi_bgr: np.ndarray, px_per_mm: float, cfg: OccupancyConfig | None = None,
            reference: np.ndarray | None = None, has_code: bool = False,
            tof_delta_mm: float | None = None, cell_depth_mm: float = 300.0) -> OccupancyResult:
    """Занятость и зазоры.

    tof_delta_mm — насколько показание дальномера больше дистанции до плоскости фасада
    (0 — препятствие на кромке полки, ≈ глубина ячейки — видна задняя стенка). Передаётся,
    только когда камера наведена на центр ячейки.
    """
    cfg = cfg or OccupancyConfig()
    mask = foreground_mask(roi_bgr, cfg, reference)
    h, w = mask.shape
    zone = mask[int(h * (1 - cfg.bottom_part)):, :] > 0
    fill = float(zone.mean())
    col_occ = zone.mean(axis=0) > cfg.column_fill
    min_cols = max(1, int(cfg.min_span_mm * px_per_mm))
    spans = [(a, b) for a, b in _runs(col_occ) if b - a >= min_cols]
    spans_mm = [(a / px_per_mm, b / px_per_mm) for a, b in spans]
    width_mm = w / px_per_mm

    gaps_mm: list[tuple[float, float]] = []
    cursor = 0.0
    for a, b in spans_mm:
        if a - cursor > 1.0:
            gaps_mm.append((cursor, a))
        cursor = b
    if width_mm - cursor > 1.0:
        gaps_mm.append((cursor, width_mm))
    between = [spans_mm[i + 1][0] - spans_mm[i][1] for i in range(len(spans_mm) - 1)]
    occupied_w = sum(b - a for a, b in spans_mm)
    free_ratio = 1.0 - occupied_w / width_mm if width_mm > 0 else 1.0

    # --- решение ---------------------------------------------------------------
    by_vision: bool | None
    if fill >= cfg.occupied_fill and spans:
        by_vision = True
    elif fill <= cfg.empty_fill and not spans:
        by_vision = False
    else:
        by_vision = None
    vis_conf = 0.9 if reference is not None else 0.6
    method = "reference" if reference is not None else "heuristic"

    by_tof: bool | None = None
    if tof_delta_mm is not None:
        if tof_delta_mm < 0.5 * cell_depth_mm:
            by_tof = True
        elif tof_delta_mm > 0.8 * cell_depth_mm:
            by_tof = False

    if has_code:
        occupied, conf, method = True, 0.99, "code"
    elif by_vision is True:
        # дальномер может «промахнуться» мимо узкой коробки, поэтому «пусто» от него здесь не решает
        occupied, conf = True, (0.97 if by_tof is True else vis_conf)
        method += "+tof" if by_tof is True else ""
    elif by_vision is False:
        if by_tof is True and reference is None:
            # без эталона «пусто» означает лишь «не похоже на картон»; близкое препятствие по
            # дальномеру — значит, в ячейке стоит что-то другого цвета
            occupied, conf, method = True, 0.8, "tof"
        elif by_tof is True:    # эталон говорит «пусто», дальномер — «близко»: спор, не решаем
            occupied, conf, method = None, 0.3, method + "≠tof"
        else:
            occupied, conf = False, (0.97 if by_tof is False else vis_conf)
            method += "+tof" if by_tof is False else ""
    elif by_tof is not None:
        occupied, conf, method = by_tof, 0.8, "tof"
    else:
        occupied, conf = None, 0.3

    res = OccupancyResult(occupied, conf, fill, method, spans_mm, gaps_mm,
                          max(between) if between else 0.0, free_ratio)
    if occupied:
        if res.max_gap_between_mm > cfg.max_gap_mm:
            res.flags.append("GAP")
        if spans and free_ratio > cfg.max_free_ratio:
            res.flags.append("UNDERFILLED")
    return res

"""Автомат сканирования: что делать с очередным кадром и что сказать оператору.

Фазы одной ячейки:
    SEARCH    адресных меток нужного стеллажа в кадре нет;
    NAVIGATE  выход на «дистанцию обзора»: ячейка целиком в кадре, камера по нормали;
    ASSESS    несколько кадров подряд: занятость, зазоры, поиск этикеток;
    APPROACH  подход к этикетке: её центр — в центре кадра, дистанция чтения;
    READ      съёмка полного кадра и декодирование;
    DONE      результат записан (фото, журнал, WMS), короткая пауза — и следующая ячейка.

Исходы ячейки (ScanStatus): READ, EMPTY, NO_LABEL, READ_ERROR, UNCERTAIN, SKIPPED.
Автомат не знает, кто двигает камеру. Сегодня это человек, читающий команды с экрана;
завтра — дрон, получающий ту же ошибку позы как уставку (guidance.velocity_command).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from . import codes as codes_mod
from . import guidance as gd
from . import markers, occupancy
from .codes import CodeRead
from .evidence import EvidenceStore
from .layout import Address, Layout
from .optics import CameraModel, plan_distances
from .pose import FacadePose, polygon_visible_fraction, rectify, solve_facade_pose
from .wms.base import InventoryTask, WmsBackend
from .wms.reconcile import SCAN_RU, VERDICT_RU, ScanStatus

IDLE, SEARCH, NAVIGATE, ASSESS, APPROACH, READ, DONE, FINISHED = (
    "IDLE", "SEARCH", "NAVIGATE", "ASSESS", "APPROACH", "READ", "DONE", "FINISHED")


@dataclass
class Frame:
    """Кадр предпросмотра с показаниями датчиков на момент съёмки."""

    image: np.ndarray
    t: float                              # секунды, монотонные
    gyro_dps: float | None = None         # модуль угловой скорости, °/с
    tof_mm: float | None = None           # дальномер вдоль оптической оси
    tilt_deg: tuple[float, float] | None = None   # крен, тангаж по акселерометру
    meta: dict = field(default_factory=dict)


@dataclass
class ScanConfig:
    label_code_mm: float = 40.0           # сторона символа кода на этикетке
    label_module_mm: float = 40.0 / 21    # ширина модуля (QR версии 1: 21 модуль)
    label_sticker_mm: float = 62.0        # размер наклейки (для поиска кандидатов)
    min_ppm: float = 3.0                  # пикселей на модуль для уверенного чтения
    view_margin: float = 0.15             # поля кадра вокруг ячейки на дистанции обзора
    d_label_cap_m: float = 0.55           # не дальше этого при чтении этикетки (для паллет — 1.0)
    d_near_m: float = 0.30                # ближе не подходить (фокус, безопасность)
    always_approach: bool = True          # подходить к этикетке, даже если код уже прочитан издалека
    coarse: gd.Tolerances = field(default_factory=lambda: gd.Tolerances(0.05, 0.12, 7.0, 10.0, 8.0))
    fine: gd.Tolerances = field(default_factory=lambda: gd.Tolerances(0.025, 0.06, 5.0, 8.0, 6.0))
    assess_frames: int = 4
    steady_frames: int = 3
    read_attempts: int = 3
    max_labels_per_cell: int = 3
    max_reads_per_cell: int = 8           # всего съёмок полного кадра на ячейку
    done_hold_s: float = 1.5
    lost_timeout_s: float = 1.5
    cell_timeout_s: float = 120.0
    gyro_limit_dps: float = 8.0
    roi_px_per_mm: float = 0.6
    track_radius_mm: float = 40.0
    blind: bool = True                    # «слепой» пересчёт: учётные данные показываются после скана
    occ: occupancy.OccupancyConfig = field(default_factory=occupancy.OccupancyConfig)


@dataclass
class LabelTrack:
    """Этикетка-кандидат в ячейке, сопровождаемая между кадрами в координатах фасада."""

    xy_mm: np.ndarray
    hits: int = 1
    text: str = ""           # непусто — код прочитан
    fmt: str = ""
    failed: bool = False     # исчерпаны попытки чтения
    attempts: int = 0
    side_mm: float = 0.0

    @property
    def done(self) -> bool:
        return bool(self.text) or self.failed

    @property
    def stable(self) -> bool:
        """Прочитана или замечена не меньше двух раз (одиночное срабатывание — возможно, шум)."""
        return bool(self.text) or self.hits >= 2


class Scanner:
    def __init__(self, layout: Layout, preview_cam: CameraModel, still_cam: CameraModel | None = None,
                 wms: WmsBackend | None = None, evidence: EvidenceStore | None = None,
                 cfg: ScanConfig | None = None, capture_still: Callable[[], np.ndarray | None] | None = None,
                 references: dict[str, np.ndarray] | None = None):
        self.layout = layout
        self.cam = preview_cam
        self.still_cam = still_cam or preview_cam
        self.wms = wms
        self.evidence = evidence
        self.cfg = cfg or ScanConfig()
        self.capture_still = capture_still
        self.references = references or {}
        self.plans = {rid: plan_distances(self.cam, self.still_cam, r.cell_width_mm / 1000,
                                          (r.cell_height_mm + r.beam_height_mm) / 1000, self.cfg.label_module_mm,
                                          self.cfg.min_ppm, self.cfg.view_margin, self.cfg.d_near_m)
                      for rid, r in layout.racks.items()}
        self.phase = IDLE
        self.mode = "task"
        self.task: InventoryTask | None = None
        self.queue: list[Address] = []
        self.results: dict[str, dict] = {}
        self.target: Address | None = None
        self.pose: FacadePose | None = None
        self.guidance = gd.Guidance(status=gd.IDLE, headline="ЗАДАНИЕ НЕ ЗАПУЩЕНО")
        self.last_result: dict | None = None
        self.overlay: dict = {}
        self._last_frame: Frame | None = None
        self._reset_cell(0.0)
        self._pose_t = -1e9
        self._done_until = 0.0
        self._prev_pos: np.ndarray | None = None

    # --- управление заданием ---------------------------------------------------------
    def start_task(self, addresses: list[str] | None = None, mode: str = "task") -> dict:
        """Начать инвентаризацию: получить задание из WMS и построить порядок обхода."""
        self.mode = mode
        wanted = {str(a) for a in self.layout.addresses()} if addresses is None else set(addresses)
        if self.wms is not None:
            known = set(self.wms.cells())
            self.task = self.wms.create_task(sorted(wanted & known))
            wanted &= known
        else:
            self.task = None
        self.queue = [a for a in self.layout.serpentine() if str(a) in wanted]
        self.results = {}
        self.last_result = None
        self.target = None
        self.phase = SEARCH if self.queue else FINISHED
        self._log("task_started", task_id=self.task.id if self.task else None, mode=mode, cells=len(self.queue))
        return {"task_id": self.task.id if self.task else None, "cells": len(self.queue)}

    def stop(self) -> None:
        if self.task is not None and self.wms is not None:
            self.wms.close_task(self.task.id)
        self._log("task_stopped", done=len(self.results))
        self.phase, self.target, self.queue = IDLE, None, []
        self.guidance = gd.Guidance(status=gd.IDLE, headline="ЗАДАНИЕ ОСТАНОВЛЕНО")

    def skip(self, reason: str = "пропущено оператором") -> None:
        if self.target is not None and self.phase not in (IDLE, FINISHED, DONE):
            self._finalize(ScanStatus.SKIPPED, self._last_frame, note=reason)

    def rescan(self, address: str) -> None:
        a = Address.parse(address)
        self.results.pop(str(a), None)
        if a not in self.queue:
            self.queue.insert(0, a)
        if self.phase in (FINISHED, IDLE) and self.task is not None:
            self.phase = SEARCH
        if self.phase not in (ASSESS, APPROACH, READ):
            self.target = None
        self._log("rescan", address=str(a))

    # --- главный шаг ----------------------------------------------------------------
    def step(self, frame: Frame) -> dict:
        """Обработать кадр предпросмотра. Возвращает снимок состояния для экрана оператора."""
        self._last_frame = frame
        dets = markers.detect(frame.image, self.layout.marker_dictionary)
        pose = solve_facade_pose(dets, self.layout, self.cam, prior=self.pose)
        if pose is not None:
            self.pose, self._pose_t = pose, frame.t
        self.overlay = {"markers": dets, "pose": pose, "codes": [], "cell_poly": None, "target_px": None,
                        "visible": []}
        visible = sorted({str(a) for d in dets if (a := self.layout.address_of(d.id))})
        self.overlay["visible"] = visible

        if self.phase in (IDLE, FINISHED):
            pass
        elif self.phase == DONE:
            if frame.t >= self._done_until:
                self._next_target(frame)
        else:
            if self.target is None:
                self._choose_target(pose, frame)
            if self.target is not None:
                if frame.t - self._cell_t0 > self.cfg.cell_timeout_s:
                    self._finalize(ScanStatus.SKIPPED, frame, note="тайм-аут ячейки")
                else:
                    self._step_cell(frame, pose, dets)
        return self.snapshot(frame)

    # --- выбор цели -----------------------------------------------------------------
    def _choose_target(self, pose: FacadePose | None, frame: Frame) -> None:
        if not self.queue:
            self._finish()
            return
        if self.mode == "free":
            if pose is None:
                self.phase = SEARCH
                self.guidance = gd.search("НАВЕДИТЕ КАМЕРУ НА СТЕЛЛАЖ", "адресные метки не видны")
                return
            centre = np.array([self.cam.cx, self.cam.cy])
            cands = [a for a in self.queue if self.layout.rack_of(a).row == pose.row_id]
            if not cands:
                self.guidance = gd.search("ЭТОТ СТЕЛЛАЖ УЖЕ ОТСКАНИРОВАН", "перейдите к следующему")
                return
            def dist(a):
                cx, cy = self.layout.rack_of(a).cell_center_mm(a)
                return np.linalg.norm(pose.project(np.array([[cx / 1000, cy / 1000, 0]]))[0] - centre)
            self.target = min(cands, key=dist)
        else:
            self.target = self.queue[0]
        self._reset_cell(frame.t)
        self.phase = NAVIGATE
        self._log("target", address=str(self.target))

    def _next_target(self, frame: Frame) -> None:
        self.target = None
        if not self.queue:
            self._finish()
        else:
            self.phase = SEARCH
            self._choose_target(self.pose if frame.t - self._pose_t < 0.5 else None, frame)

    def _finish(self) -> None:
        self.phase, self.target = FINISHED, None
        self.guidance = gd.Guidance(status=gd.GREEN, headline="ЗАДАНИЕ ВЫПОЛНЕНО",
                                    hint=f"отсканировано ячеек: {len(self.results)}")
        if self.task is not None and self.wms is not None:
            self.wms.close_task(self.task.id)
        self._log("task_finished", done=len(self.results))

    def _reset_cell(self, t: float) -> None:
        self._cell_t0 = t
        self._assess_n = 0
        self._occ_votes: list[bool | None] = []
        self._occ_last: occupancy.OccupancyResult | None = None
        self._flags: set[str] = set()
        self._tracks: list[LabelTrack] = []
        self._label: LabelTrack | None = None
        self._steady = 0
        self._closer_look = False
        self._d_label = None
        self._reads_total = 0

    # --- обработка ячейки -----------------------------------------------------------
    def _step_cell(self, frame: Frame, pose: FacadePose | None, dets) -> None:
        a = self.target
        rack = self.layout.rack_of(a)
        plan = self.plans[a.rack]
        d_label = self._d_label or max(self.cfg.d_near_m, min(plan.d_label, self.cfg.d_label_cap_m))
        self._d_label = d_label

        if pose is None or pose.row_id != rack.row:
            self._steady = 0
            lost_for = frame.t - self._pose_t
            if self.phase in (APPROACH, READ):
                if self._guide_by_label(frame, d_label):
                    return
                if pose is None:
                    # подошли слишком близко: адресная метка вышла из кадра, а этикетка не читается
                    self.guidance = gd.Guidance(status=gd.RED, headline="ОТОЙДИТЕ ДАЛЬШЕ на 15 см",
                                                hint="адресная метка вне кадра", errors={"ez": -0.15},
                                                body=gd.body_setpoint(0.0, 0.0, -0.15, 0.0))
                    return
            if pose is not None:
                if self.phase in (NAVIGATE, ASSESS):
                    self.phase = SEARCH
                self.guidance = gd.search(f"ПЕРЕЙДИТЕ К СТЕЛЛАЖУ {a.rack:02d}",
                                          f"сейчас в кадре стеллаж {pose.rack_id:02d} другого ряда")
            elif lost_for > self.cfg.lost_timeout_s or self.phase == SEARCH:
                self.guidance = gd.search(f"НАВЕДИТЕ КАМЕРУ НА СТЕЛЛАЖ {a.rack:02d}",
                                          "адресные метки не видны — отойдите дальше или добавьте света")
            if self.phase in (ASSESS,) and lost_for > self.cfg.lost_timeout_s:
                self.phase = NAVIGATE
            return
        if self.phase == SEARCH:
            self.phase = NAVIGATE

        rect = rack.cell_rect_mm(a)
        poly = pose.rect_polygon(rect)
        self.overlay["cell_poly"] = poly
        cx, cy = rack.cell_center_mm(a)

        if self.phase == NAVIGATE:
            g = gd.compute(pose, (cx / 1000, cy / 1000), plan.d_cell, self.cfg.coarse)
            self._target_px(pose, cx, cy)
            if g.aligned and polygon_visible_fraction(poly, self.cam.width, self.cam.height) > 0.97:
                self._steady += 1
                if self._steady >= 2:
                    self.phase, self._steady = ASSESS, 0
                    g.headline = "ДЕРЖИТЕ РОВНО — осмотр ячейки"
            else:
                self._steady = 0
            self.guidance = g

        elif self.phase == ASSESS:
            g = gd.compute(pose, (cx / 1000, cy / 1000), plan.d_cell, self.cfg.coarse.scaled(1.6))
            self._target_px(pose, cx, cy)
            if not g.aligned or polygon_visible_fraction(poly, self.cam.width, self.cam.height) < 0.9:
                self.phase, self._steady = NAVIGATE, 0     # увели камеру — возвращаемся к наведению
                self.guidance = g
                return
            self._assess_frame(frame, pose, rect, poly)
            g.status, g.headline = gd.YELLOW, f"ДЕРЖИТЕ РОВНО — осмотр ячейки {self._assess_n}/{self.cfg.assess_frames}"
            self.guidance = g
            if self._assess_n >= self.cfg.assess_frames:
                self._decide_after_assess(frame)

        elif self.phase in (APPROACH, READ):
            if self._label is not None:
                tx, ty = self._label.xy_mm
            else:
                tx, ty = cx, cy
            d_label = max(d_label, self._marker_floor(a, tx, ty))
            g = gd.compute(pose, (tx / 1000, ty / 1000), d_label, self.cfg.fine)
            self._target_px(pose, tx, ty)
            self._refresh_tracks(frame, pose, poly)
            steady = self._is_steady(frame, pose)
            if g.aligned and steady:
                self._steady += 1
                g.headline = "ДЕРЖИТЕ РОВНО — съёмка"
            else:
                self._steady = 0
                if g.aligned:
                    g.headline, g.hint = "ДЕРЖИТЕ РОВНО", "камера движется"
                elif not g.hint:
                    g.hint = "подход к этикетке" if self._label else "осмотр ячейки вблизи"
            self.guidance = g
            if self._steady >= self.cfg.steady_frames:
                self._steady = 0
                self.phase = READ
                self._read(frame, pose)

    def _marker_floor(self, a: Address, x_mm: float, y_mm: float) -> float:
        """Наименьшая дистанция, с которой адресная метка ячейки ещё целиком в кадре, когда
        камера стоит напротив точки (x, y). Ближе подходить нельзя: пропадёт опора для позы."""
        mx, my = self.layout.rack_of(a).marker_center_mm(a)
        half = self.layout.marker_size_mm / 2 + 12.0          # метка + белое поле
        need_x = (abs(mx - x_mm) + half) / 1000.0
        need_y = (abs(my - y_mm) + half) / 1000.0
        k = 0.94                                              # небольшой запас от края кадра
        return max(need_x * 2 * self.cam.fx / (self.cam.width * k), need_y * 2 * self.cam.fy / (self.cam.height * k))

    def _target_px(self, pose: FacadePose, x_mm: float, y_mm: float) -> None:
        self.overlay["target_px"] = pose.project(np.array([[x_mm / 1000, y_mm / 1000, 0.0]]))[0]

    def _is_steady(self, frame: Frame, pose: FacadePose) -> bool:
        pos = pose.position
        moved = np.linalg.norm(pos - self._prev_pos) if self._prev_pos is not None else 0.0
        self._prev_pos = pos
        if frame.gyro_dps is not None and frame.gyro_dps > self.cfg.gyro_limit_dps:
            return False
        return moved < 0.02

    # --- осмотр ячейки ----------------------------------------------------------------
    def _cell_reads(self, image: np.ndarray, pose: FacadePose, poly: np.ndarray, robust: bool = False,
                    candidates: bool = True) -> list[CodeRead]:
        """Коды и кандидаты внутри проёма ячейки на данном изображении."""
        h, w = image.shape[:2]
        x0, y0 = np.floor(poly.min(axis=0)).astype(int)
        x1, y1 = np.ceil(poly.max(axis=0)).astype(int)
        mx, my = int(0.06 * (x1 - x0)) + 4, int(0.06 * (y1 - y0)) + 4
        x0, y0, x1, y1 = max(x0 - mx, 0), max(y0 - my, 0), min(x1 + mx, w), min(y1 + my, h)
        if x1 - x0 < 16 or y1 - y0 < 16:
            return []
        crop = image[y0:y1, x0:x1]
        if robust:
            reads = codes_mod.read_codes_robust(crop, (x0, y0))
        else:
            scale = 2.0 if max(crop.shape[:2]) < 900 else 1.0
            src = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else crop
            reads = codes_mod.read_codes(src, (x0, y0), scale)
        inside = [r for r in reads if cv2.pointPolygonTest(poly.astype(np.float32), tuple(map(float, r.center)), True) > -8]
        if candidates:
            mask = np.zeros(crop.shape[:2], np.uint8)
            cv2.fillConvexPoly(mask, np.round(poly - [x0, y0]).astype(np.int32), 255)
            ppm = pose.cam.fx / max(pose.distance, 0.05) / 1000.0          # px/мм на плоскости фасада
            cands = codes_mod.find_candidates(crop, (x0, y0), expected_side_px=self.cfg.label_sticker_mm * ppm,
                                              mask=mask)
            inside = codes_mod.merge_reads(inside, cands)
        return inside

    def _update_tracks(self, reads: list[CodeRead], pose: FacadePose) -> None:
        ppm = pose.cam.fx / max(pose.distance, 0.05) / 1000.0
        for r in reads:
            xy = np.array(pose.pixel_to_facade(*r.center)) * 1000.0
            tr = next((t for t in self._tracks if np.linalg.norm(t.xy_mm - xy) < self.cfg.track_radius_mm), None)
            if tr is None:
                if len(self._tracks) >= 8:
                    continue
                tr = LabelTrack(xy, hits=0)
                self._tracks.append(tr)
            tr.hits += 1
            tr.side_mm = r.side_px / ppm
            if r.decoded:
                tr.xy_mm = xy if not tr.text else 0.7 * tr.xy_mm + 0.3 * xy
                tr.text, tr.fmt, tr.failed = r.text, r.fmt, False
            elif not tr.text:
                tr.xy_mm = 0.7 * tr.xy_mm + 0.3 * xy

    def _assess_frame(self, frame: Frame, pose: FacadePose, rect, poly) -> None:
        cfg = self.cfg
        reads = self._cell_reads(frame.image, pose, poly)
        self.overlay["codes"] = reads
        self._update_tracks(reads, pose)
        roi = rectify(frame.image, pose, rect, cfg.roi_px_per_mm)
        tof_delta = None
        if frame.tof_mm is not None:
            tof_delta = frame.tof_mm - pose.distance * 1000.0
        rack = self.layout.rack_of(self.target)
        occ = occupancy.analyze(roi, cfg.roi_px_per_mm, cfg.occ, self.references.get(str(self.target)),
                                has_code=any(r.decoded for r in reads), tof_delta_mm=tof_delta,
                                cell_depth_mm=rack.cell_depth_mm)
        self._occ_votes.append(occ.occupied)
        self._occ_last = occ
        self._flags.update(occ.flags)
        self._assess_n += 1

    def _refresh_tracks(self, frame: Frame, pose: FacadePose, poly) -> None:
        """На подходе к этикетке продолжаем читать предпросмотр: вблизи код может прочитаться сам."""
        reads = self._cell_reads(frame.image, pose, poly, candidates=False)
        self.overlay["codes"] = reads
        self._update_tracks([r for r in reads if r.decoded], pose)

    def _vote_occupied(self) -> bool | None:
        v = self._occ_votes
        if not v:
            return None
        if sum(x is True for x in v) > len(v) / 2:
            return True
        if sum(x is False for x in v) > len(v) / 2:
            return False
        return None

    def _stable_tracks(self) -> list[LabelTrack]:
        need = max(2, self.cfg.assess_frames // 2)
        return [t for t in self._tracks if t.hits >= need or t.text][: self.cfg.max_labels_per_cell]

    def _decide_after_assess(self, frame: Frame) -> None:
        occ = self._vote_occupied()
        tracks = self._stable_tracks()
        self._tracks = tracks
        if occ is False and not tracks:
            self._finalize(ScanStatus.EMPTY, frame)
            return
        pending = [t for t in tracks if not t.done]
        if not pending and tracks and not self.cfg.always_approach:
            self._finalize(ScanStatus.READ, frame)
            return
        self._pick_label()
        self._closer_look = self._label is None        # коробка без найденной этикетки — смотрим вблизи
        self.phase, self._steady = APPROACH, 0
        self._log("approach", address=str(self.target), labels=len(tracks),
                  target=None if self._label is None else [round(float(v)) for v in self._label.xy_mm])

    def _pick_label(self) -> None:
        """Следующая этикетка для подхода: сначала непрочитанные, ближайшая к центру ячейки."""
        rack = self.layout.rack_of(self.target)
        c = np.array(rack.cell_center_mm(self.target))
        todo = [t for t in self._tracks if not t.done and t.stable]
        if not todo and self.cfg.always_approach and self._label is None:
            todo = [t for t in self._tracks if t.text]      # уже прочитана издалека — подходим ради фото
        self._label = min(todo, key=lambda t: np.linalg.norm(t.xy_mm - c)) if todo else None

    # --- съёмка и чтение --------------------------------------------------------------
    def _read(self, frame: Frame, pose: FacadePose) -> None:
        still = self.capture_still() if self.capture_still else None
        if still is None:
            still, spose = frame.image, pose
        else:
            spose = self._still_pose(still, pose)
        rack = self.layout.rack_of(self.target)
        rect = rack.cell_rect_mm(self.target)
        poly = spose.rect_polygon(rect)
        reads = self._cell_reads(still, spose, poly, robust=True, candidates=True)
        self._update_tracks(reads, spose)
        self._still, self._still_pose_used, self._still_reads = still, spose, reads

        # занятость по крупному плану, если издалека решить не удалось
        if self._vote_occupied() is None:
            roi = rectify(still, spose, rect, self.cfg.roi_px_per_mm)
            occ = occupancy.analyze(roi, self.cfg.roi_px_per_mm, self.cfg.occ, self.references.get(str(self.target)),
                                    has_code=any(r.decoded for r in reads),
                                    tof_delta_mm=None if frame.tof_mm is None else frame.tof_mm - pose.distance * 1000,
                                    cell_depth_mm=rack.cell_depth_mm)
            self._occ_votes = [occ.occupied]
            self._occ_last = occ
            self._flags.update(occ.flags)

        lab = self._label
        self._reads_total += 1
        if lab is not None and not lab.text:
            lab.attempts += 1
            if lab.attempts >= self.cfg.read_attempts or self._reads_total >= self.cfg.max_reads_per_cell:
                lab.failed = True
                self._log("read_failed", address=str(self.target), attempts=lab.attempts)
            else:
                # ещё попытка: ближе на 15 %, но не ближе предела (и не теряя адресную метку)
                floor = self._marker_floor(self.target, float(lab.xy_mm[0]), float(lab.xy_mm[1]))
                self._d_label = max(self.cfg.d_near_m, floor, self._d_label * 0.85)
                self.phase = APPROACH
                self.guidance.headline, self.guidance.hint = "КОД НЕ ПРОЧИТАН — ПОДОЙДИТЕ БЛИЖЕ", \
                    f"попытка {lab.attempts + 1} из {self.cfg.read_attempts}"
                return
        if self._closer_look and self._label is None:
            new = [t for t in self._tracks if not t.done and t.stable]
            if new:                                     # вблизи нашлась этикетка — подходим к ней
                self._closer_look = False
        self._pick_label_after_read()
        if self._label is not None:
            self.phase = APPROACH
            return
        self._finalize(self._outcome(), frame, still=still, spose=spose, reads=reads)

    def _pick_label_after_read(self) -> None:
        todo = [t for t in self._tracks if not t.done and t.stable]
        rack = self.layout.rack_of(self.target)
        c = np.array(rack.cell_center_mm(self.target))
        self._label = min(todo, key=lambda t: np.linalg.norm(t.xy_mm - c)) if todo else None

    def _outcome(self) -> ScanStatus:
        if any(t.text for t in self._tracks):
            return ScanStatus.READ
        if any(t.failed for t in self._tracks):
            return ScanStatus.READ_ERROR
        occ = self._vote_occupied()
        if occ is True:
            return ScanStatus.NO_LABEL
        if occ is False:
            return ScanStatus.EMPTY
        return ScanStatus.UNCERTAIN

    def _still_pose(self, still: np.ndarray, fallback: FacadePose) -> FacadePose:
        """Поза в момент съёмки полного кадра. Метки ищутся на уменьшенной копии — так быстрее."""
        k = max(1.0, still.shape[1] / 1600.0)
        small = cv2.resize(still, None, fx=1 / k, fy=1 / k, interpolation=cv2.INTER_AREA) if k > 1 else still
        dets = [markers.MarkerDetection(d.id, d.corners * k) for d in markers.detect(small, self.layout.marker_dictionary)]
        cam = self.still_cam if still.shape[1] == self.still_cam.width else self.cam.scaled(still.shape[1], still.shape[0])
        p = solve_facade_pose(dets, self.layout, cam, prior=fallback)
        if p is not None and p.row_id == fallback.row_id:
            return p
        return FacadePose(fallback.rack_id, fallback.R, fallback.t, cam, fallback.n_markers, fallback.reproj_px,
                          fallback.ambiguous, fallback.row_id)

    def _guide_by_label(self, frame: Frame, d_label: float) -> bool:
        """Меток стеллажа не видно, но, возможно, видна сама этикетка — наводимся по ней."""
        reads = codes_mod.read_codes(frame.image)
        if not reads:
            return False
        centre = np.array([self.cam.cx, self.cam.cy])
        r = min(reads, key=lambda x: np.linalg.norm(x.center - centre))
        self.overlay["codes"] = reads
        self.overlay["target_px"] = r.center
        g = gd.from_label_only(float(r.center[0]), float(r.center[1]), r.side_px, self.cam, self.cfg.label_code_mm,
                               d_label, self.cfg.fine)
        self.guidance = g
        if g.aligned and (frame.gyro_dps is None or frame.gyro_dps < self.cfg.gyro_limit_dps):
            self._steady += 1
            if self._steady >= self.cfg.steady_frames and r.decoded:
                still = self.capture_still() if self.capture_still else None
                img = frame.image if still is None else still
                got = codes_mod.read_codes_robust(img)
                for x in got:
                    if x.decoded and self._label is not None and not self._label.text:
                        self._label.text, self._label.fmt = x.text, x.fmt
                if self._label is not None and self._label.text:
                    self._still, self._still_pose_used, self._still_reads = img, None, got
                    self._pick_label_after_read()
                    if self._label is None:
                        self._finalize(self._outcome(), frame, still=img, reads=got)
                    else:
                        self.phase = APPROACH
        else:
            self._steady = 0
        return True

    # --- запись результата ------------------------------------------------------------
    def _finalize(self, status: ScanStatus, frame: Frame | None, still: np.ndarray | None = None,
                  spose: FacadePose | None = None, reads: list[CodeRead] | None = None, note: str = "") -> None:
        a = self.target
        addr = str(a)
        keys, raw_texts = [], []
        for t in self._tracks:
            if t.text:
                p = codes_mod.parse_payload(t.text)
                if p.key not in keys:
                    keys.append(p.key)
                    raw_texts.append({"text": t.text, "format": t.fmt, "kind": p.kind, "key": p.key})
        flags = set(self._flags)
        if len(keys) > 1:
            flags.add("MULTI")
        if status == ScanStatus.READ and any(t.failed for t in self._tracks):
            flags.add("UNREAD_LABEL")
        if status in (ScanStatus.EMPTY, ScanStatus.SKIPPED):
            flags -= {"GAP", "UNDERFILLED"}
        pose = spose or self.pose
        if pose is not None and pose.ambiguous:
            flags.add("POSE_AMBIGUOUS")
        occ = self._occ_last
        result = {
            "address": addr, "address_human": a.human, "scan_status": status.value,
            "codes": keys, "codes_raw": raw_texts, "flags": sorted(flags), "note": note,
            "occupancy": occ.to_dict() if occ else None,
            "confidence": 0.99 if keys else (occ.confidence if occ else 0.0),
            "pose": None if pose is None else {
                "x_m": round(float(pose.position[0]), 3), "y_m": round(float(pose.position[1]), 3),
                "distance_m": round(pose.distance, 3), "yaw_deg": round(pose.yaw_deg, 1),
                "pitch_deg": round(pose.pitch_deg, 1), "roll_deg": round(pose.roll_deg, 1),
                "markers": pose.n_markers, "reproj_px": round(pose.reproj_px, 2)},
            "sensors": None if frame is None else {"gyro_dps": frame.gyro_dps, "tof_mm": frame.tof_mm,
                                                   "tilt_deg": frame.tilt_deg},
            "task_id": self.task.id if self.task else None,
            "duration_s": None if frame is None else round(frame.t - self._cell_t0, 1),
        }
        image = still if still is not None else (frame.image if frame is not None else None)
        if self.evidence is not None:
            annotated = None
            if image is not None:
                from .overlay import annotate_evidence
                annotated = annotate_evidence(image, result, pose if (spose is not None or still is None) else None,
                                              self.layout, reads or self.overlay.get("codes") or [])
            result = self.evidence.save_result(result, image, annotated)
        if self.wms is not None and self.task is not None:
            try:
                verdict = self.wms.post_result(self.task.id, {
                    "address": addr, "scan_status": status.value, "codes": keys, "flags": result["flags"],
                    "photo": result.get("photo", ""), "confidence": result["confidence"], "ts": result.get("ts")})
                result.update({"verdict": verdict["verdict"], "verdict_note": verdict.get("note", ""),
                               "expected": verdict.get("expected", [])})
            except Exception as e:  # сеть/1С недоступны — результат не теряем, он в журнале и на диске
                result.update({"verdict": None, "verdict_note": f"WMS недоступна: {e}"})
                self._log("wms_error", address=addr, error=str(e))
        self.results[addr] = result
        self.last_result = result
        if a in self.queue:
            self.queue.remove(a)
        ok = status in (ScanStatus.READ, ScanStatus.EMPTY)
        text = SCAN_RU[status.value].lower()
        if keys:
            text += ": " + ", ".join(keys)
        if result.get("verdict"):
            text += " — " + VERDICT_RU.get(result["verdict"], "").lower()
        self.guidance = gd.Guidance(status=gd.GREEN if ok else gd.YELLOW,
                                    headline=("ГОТОВО: " if ok else "ЗАФИКСИРОВАНО: ") + addr,
                                    hint=text + ("" if ok else " — фото сохранено"))
        self.phase = DONE
        self._done_until = (frame.t if frame is not None else 0.0) + self.cfg.done_hold_s

    def _log(self, kind: str, **data) -> None:
        if self.evidence is not None:
            self.evidence.journal(kind, **data)

    # --- снимок состояния для интерфейса ----------------------------------------------
    def snapshot(self, frame: Frame | None = None) -> dict:
        g = self.guidance
        pose = self.pose if (frame is None or frame.t - self._pose_t < 0.5) else None
        total = len(self.results) + len(self.queue)
        expected = None
        if self.target is not None and self.task is not None and not self.cfg.blind:
            line = next((ln for ln in self.task.lines if ln.address == str(self.target)), None)
            expected = [e.to_dict() for e in line.expected] if line else []
        d_label = self._d_label
        plan = self.plans.get(self.target.rack) if self.target else None
        return {
            "phase": self.phase, "mode": self.mode,
            "status": g.status, "headline": g.headline, "hint": g.hint,
            "commands": [c.to_dict() for c in g.commands], "errors": {k: round(v, 3) for k, v in g.errors.items()},
            "body": {k: round(v, 3) for k, v in g.body.items()},
            "velocity": {k: round(v, 3) for k, v in gd.velocity_command(g.body).items()} if g.body else {},
            "target": str(self.target) if self.target else None,
            "target_human": self.target.human if self.target else "",
            "target_distance_m": None if plan is None else (plan.d_cell if self.phase in (NAVIGATE, ASSESS, SEARCH)
                                                            else d_label),
            "expected": expected,
            "pose": None if pose is None else {
                "rack": pose.rack_id, "x_m": round(float(pose.position[0]), 3),
                "y_m": round(float(pose.position[1]), 3), "distance_m": round(pose.distance, 3),
                "yaw_deg": round(pose.yaw_deg, 1), "pitch_deg": round(pose.pitch_deg, 1),
                "roll_deg": round(pose.roll_deg, 1), "markers": pose.n_markers, "ambiguous": pose.ambiguous},
            "visible": self.overlay.get("visible", []),
            "occupancy": self._occ_last.to_dict() if self._occ_last and self.target else None,
            "labels": [{"x_mm": round(float(t.xy_mm[0])), "y_mm": round(float(t.xy_mm[1])), "text": t.text,
                        "failed": t.failed} for t in self._tracks] if self.target else [],
            "progress": {"done": len(self.results), "total": total},
            "results": {k: {"scan_status": v["scan_status"], "verdict": v.get("verdict"), "flags": v["flags"],
                            "codes": v["codes"]} for k, v in self.results.items()},
            "queue": [str(a) for a in self.queue],
            "last_result": self.last_result,
            "task_id": self.task.id if self.task else None,
            "sensors": None if frame is None else {"gyro_dps": frame.gyro_dps, "tof_mm": frame.tof_mm,
                                                   "tilt_deg": frame.tilt_deg},
        }


def follow_commands(position: np.ndarray, yaw_deg: float, snap: dict, gain: float = 0.7,
                    max_step_m: float = 0.25, max_yaw_deg: float = 12.0) -> tuple[np.ndarray, float]:
    """Имитатор носителя: выполнить долю показанной команды (для испытаний и демонстрации).

    Работает от той же ошибки позы, что и команды оператору, поэтому сходимость этого
    цикла на синтетике — проверка закона наведения до появления дрона.
    """
    e = snap.get("errors") or {}
    if not e:
        return position, yaw_deg

    def lim(v, m):
        return max(-m, min(m, v))
    dx = lim(gain * e.get("ex", 0.0), max_step_m)
    dy = lim(gain * e.get("ey", 0.0), max_step_m)
    dz = lim(-gain * e.get("ez", 0.0), max_step_m)
    dyaw = lim(-gain * e.get("yaw", 0.0), max_yaw_deg)
    return position + np.array([dx, dy, dz]), yaw_deg + dyaw


"""Поза камеры относительно фасада стеллажа по адресным меткам (PnP) и выпрямление ячейки.

Все видимые метки одного стеллажа решаются совместно, как одна «доска»: положение каждой
известно из модели стенда. Это снимает главную слабость одиночной метки — неоднозначность
ориентации плоского квадрата (два решения IPPE) и шум курса 2–3° (эксперимент LEARN-07
старого проекта; статья A: «две метки точнее одной»).

Минимизируется ошибка репроекции (Gao & Zhang, ф. 6.36):
    T* = argmin ½ Σ ‖u_i − π(K, T·P_i)‖²,   P_i — углы меток в системе фасада.

Углы камеры относительно фасада (камера: x вправо, y вниз, z вперёд):
    курс ψ   > 0 — камера смотрит правее нормали к фасаду;
    тангаж θ > 0 — камера смотрит выше горизонтали;
    крен φ   > 0 — правый край камеры поднят.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .layout import Layout
from .markers import MarkerDetection
from .optics import CameraModel


@dataclass
class FacadePose:
    rack_id: int
    R: np.ndarray            # 3×3, фасад -> камера
    t: np.ndarray            # (3,), м
    cam: CameraModel
    n_markers: int
    reproj_px: float
    ambiguous: bool = False
    row_id: int = 1          # ряд стеллажей (общая плоскость фасада), к которому привязана поза

    # --- положение и углы -------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        """Центр камеры в системе фасада, м: (x вдоль фасада, y высота, z дистанция до фасада)."""
        return -self.R.T @ self.t

    @property
    def distance(self) -> float:
        return float(self.position[2])

    @property
    def yaw_deg(self) -> float:
        a = self.R[2]  # оптическая ось камеры в системе фасада
        return math.degrees(math.atan2(a[0], -a[2]))

    @property
    def pitch_deg(self) -> float:
        return math.degrees(math.asin(max(-1.0, min(1.0, float(self.R[2][1])))))

    @property
    def roll_deg(self) -> float:
        r = self.R[0]  # ось x камеры в системе фасада
        return math.degrees(math.atan2(r[1], math.hypot(r[0], r[2])))

    # --- проекции ---------------------------------------------------------------
    def project(self, pts_m: np.ndarray) -> np.ndarray:
        """Точки фасада (N,3) в метрах -> пиксели (N,2)."""
        rvec, _ = cv2.Rodrigues(self.R)
        px, _ = cv2.projectPoints(np.asarray(pts_m, dtype=np.float64).reshape(-1, 3), rvec, self.t,
                                  self.cam.K, self.cam.D)
        return px.reshape(-1, 2)

    def pixel_to_facade(self, u: float, v: float, z_m: float = 0.0) -> tuple[float, float]:
        """Пиксель -> точка (X, Y) на плоскости Z = z_m фасада, м (луч из центра камеры)."""
        und = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), self.cam.K, self.cam.D).reshape(2)
        ray = self.R.T @ np.array([und[0], und[1], 1.0])
        c = self.position
        s = (z_m - c[2]) / ray[2]
        p = c + s * ray
        return float(p[0]), float(p[1])

    def rect_polygon(self, rect_mm: tuple[float, float, float, float]) -> np.ndarray:
        """Углы прямоугольника фасада (x0,y0,x1,y1 мм) в пикселях: ЛВ, ПВ, ПН, ЛН."""
        x0, y0, x1, y1 = (v / 1000.0 for v in rect_mm)
        return self.project(np.array([[x0, y1, 0], [x1, y1, 0], [x1, y0, 0], [x0, y0, 0]]))


def _angles_ok(R: np.ndarray, t: np.ndarray) -> bool:
    p = -R.T @ t
    yaw = abs(math.degrees(math.atan2(R[2][0], -R[2][2])))
    return p[2] > 0.08 and yaw < 80.0


def _reproj(obj: np.ndarray, img: np.ndarray, rvec, tvec, cam: CameraModel) -> float:
    px, _ = cv2.projectPoints(obj, rvec, tvec, cam.K, cam.D)
    return float(np.sqrt(np.mean(np.sum((px.reshape(-1, 2) - img) ** 2, axis=1))))


def solve_facade_pose(detections: list[MarkerDetection], layout: Layout, cam: CameraModel,
                      prior: FacadePose | None = None, max_reproj_px: float | None = None) -> FacadePose | None:
    """Поза камеры по всем меткам самого «видимого» ряда стеллажей. None — меток стенда в кадре нет.

    Стеллажи одного ряда стоят в одну линию и имеют общую систему фасада, поэтому их метки
    решаются вместе: опорных точек больше, и наведение не обрывается на стыке стеллажей.
    """
    by_row: dict[int, list[tuple[MarkerDetection, np.ndarray, int]]] = {}
    for d in detections:
        a = layout.address_of(d.id)
        if a is not None:
            by_row.setdefault(layout.rack_of(a).row, []).append((d, layout.marker_corners_m(a), a.rack))
    if not by_row:
        return None
    centre = np.array([cam.cx, cam.cy])
    # ряд с наибольшим числом меток; при равенстве — тот, чьи метки ближе к центру кадра
    row_id = max(by_row, key=lambda r: (len(by_row[r]),
                                        -min(np.linalg.norm(d.center - centre) for d, _, _ in by_row[r])))
    triples = by_row[row_id]
    pairs = [(d, o) for d, o, _ in triples]
    rack_id = min(triples, key=lambda x: np.linalg.norm(x[0].center - centre))[2]
    obj = np.concatenate([o for _, o in pairs]).astype(np.float64)
    img = np.concatenate([d.corners for d, _ in pairs]).astype(np.float64)
    try:
        n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, cam.K, cam.D, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None
    cands = []
    for rvec, tvec in zip(rvecs, tvecs):
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, cam.K, cam.D, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)
        if _angles_ok(R, t):
            cands.append((_reproj(obj, img, rvec, tvec, cam), R, t))
    if not cands:
        return None
    cands.sort(key=lambda c: c[0])
    err, R, t = cands[0]
    ambiguous = False
    if len(cands) > 1:
        # Два решения плоской задачи различимы, если ошибка второго заметно больше.
        ambiguous = cands[1][0] < max(1.5 * err, err + 0.3)
        if ambiguous and prior is not None:
            # при неоднозначности берём решение, ближайшее к предыдущему по ориентации
            def rot_dist(Rc):
                return np.linalg.norm(cv2.Rodrigues(Rc @ prior.R.T)[0])
            err, R, t = min(cands[:2], key=lambda c: rot_dist(c[1]))
    limit = max_reproj_px if max_reproj_px is not None else max(3.0, cam.width / 400.0)
    if err > limit:
        return None
    return FacadePose(rack_id, R, t, cam, len(pairs), err, ambiguous and len(pairs) == 1, row_id)


def rectify(image: np.ndarray, pose: FacadePose, rect_mm: tuple[float, float, float, float],
            px_per_mm: float = 0.5) -> np.ndarray:
    """Выпрямленный (фронтальный) вид прямоугольника фасада. Строка 0 — верх ячейки."""
    x0, y0, x1, y1 = rect_mm
    w = max(8, int(round((x1 - x0) * px_per_mm)))
    h = max(8, int(round((y1 - y0) * px_per_mm)))
    src = pose.rect_polygon(rect_mm).astype(np.float32)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def polygon_visible_fraction(poly: np.ndarray, width: int, height: int) -> float:
    """Доля прямоугольника (по площади ограничивающей рамки), попавшая в кадр."""
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    area = max(1e-6, (x1 - x0) * (y1 - y0))
    ix = max(0.0, min(x1, width) - max(x0, 0.0))
    iy = max(0.0, min(y1, height) - max(y0, 0.0))
    return float(ix * iy / area)

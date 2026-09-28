"""Поза этикетки по четырём углам — PnP (Gao & Zhang, гл. 6.7; уточнение — минимизация
ошибки репроекции, гл. 6.7.3). Основа упражнения LEARN-07 и наблюдений для LOC-08.

Кадр этикетки: начало в центре, x вправо, y вниз, z вглубь (от камеры) — при фронтальном
взгляде поза ≈ (R = I, t = (0, 0, d)), как в оптическом кадре камеры книги (гл. 4.1.1).

Зависимости (необязательные для остального ядра): numpy, opencv-python.
"""
from __future__ import annotations

from dataclasses import dataclass

from .camera import Camera

A6 = (0.148, 0.105)   # логистическая этикетка A6, альбомная, м


def _np_cv():
    import cv2
    import numpy as np
    return np, cv2


def object_points(size: tuple[float, float] = A6):
    np, _ = _np_cv()
    w, h = size
    return np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]], dtype=np.float64)


def camera_matrix(cam: Camera):
    np, _ = _np_cv()
    return np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]], dtype=np.float64)


def order_corners(pts):
    """Порядок TL, TR, BR, BL (по сумме и разности координат)."""
    np, _ = _np_cv()
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    s, d = pts.sum(axis=1), pts[:, 1] - pts[:, 0]
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


@dataclass
class LabelPose:
    R: object              # 3x3 numpy: кадр этикетки -> кадр камеры
    t: object              # (3,) numpy: центр этикетки в кадре камеры, м
    reproj_rmse_px: float

    @property
    def distance(self) -> float:
        return float((self.t ** 2).sum() ** 0.5)


def label_pose(corners_px, cam: Camera, size: tuple[float, float] = A6) -> LabelPose:
    """PnP: IPPE для плоской цели + уточнение Левенбергом–Марквардтом (гл. 6.7.3)."""
    np, cv2 = _np_cv()
    img = order_corners(corners_px)
    obj, k, dist = object_points(size), camera_matrix(cam), np.array(cam.dist, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, k, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise ValueError("solvePnP не сошёлся")
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, k, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
    rmse = float(np.sqrt(((proj.reshape(4, 2) - img) ** 2).sum(axis=1).mean()))
    return LabelPose(cv2.Rodrigues(rvec)[0], tvec.reshape(3), rmse)


def find_label_quads(gray, aspect_min: float = 1.0, aspect_max: float = 2.5, min_area_frac: float = 0.0005,
                     min_brightness: int | None = None, close_px: int = 15):
    """Все светлые выпуклые четырёхугольники с пропорциями этикеток -> [(углы TL,TR,BR,BL, площадь px)].

    Кандидаты этикеток без декодирования (PER-09): «этикетка есть, но не прочиталась» — это «ячейка занята,
    нужен перескан», а не «пусто». min_brightness — нижняя граница порога: этикетки — самые светлые объекты,
    а автопорог Оцу при трёх уровнях (фон, паллета, этикетка) может отделить паллету вместо этикетки.
    """
    np, cv2 = _np_cv()
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(otsu, min_brightness) if min_brightness is not None else otsu
    _, bw = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))  # склеить штрихи
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out, min_area = [], min_area_frac * gray.shape[0] * gray.shape[1]
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        q = order_corners(approx.reshape(4, 2))
        w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
        h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
        if h > 0 and aspect_min <= w / h <= aspect_max:
            out.append((q, area))
    return out


def detect_label_quad(gray, aspect: float = A6[0] / A6[1], aspect_tol: float = 0.45, min_area_frac: float = 0.001):
    """Самый крупный светлый четырёхугольник с пропорциями этикетки -> 4 угла (TL, TR, BR, BL) или None."""
    np, cv2 = _np_cv()
    quads = find_label_quads(gray, aspect * (1 - aspect_tol), aspect * (1 + aspect_tol), min_area_frac)
    if not quads:
        return None
    best = max(quads, key=lambda qa: qa[1])[0]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
    refined = cv2.cornerSubPix(gray, best.astype(np.float32).reshape(-1, 1, 2), (7, 7), (-1, -1), crit)
    return refined.reshape(4, 2).astype(np.float64)


def unread_label_addresses(gray, reads, cam: Camera, r_map_cam, t_map_cam, wh, label_width: float = A6[0],
                           tol: float = 0.3, min_brightness: int = 180) -> list[str]:
    """PER-09: адреса ячеек, где в кадре есть этикетка шириной label_width, но ни один код внутри не прочитан.

    Кандидат — светлый четырёхугольник (find_label_quads); паллетный он или адресный, решаем по ширине на
    фасаде (обратная проекция углов на плоскость ряда): A6 — 148 мм, адресная — 100 мм. reads — результаты
    barcode.decode для этого же кадра. «Видели, но не прочитали» = ячейка занята, нужен перескан.
    """
    from .association import face_point_in_warehouse, pixel_to_face_pose
    np, cv2 = _np_cv()
    out = []
    centers = [rd.center for rd in reads]
    for q, _ in find_label_quads(gray, 1.1, 2.4, 0.0002, min_brightness=min_brightness):
        cu, cv_ = q.mean(axis=0)
        hit = face_point_in_warehouse(cu, cv_, cam, r_map_cam, t_map_cam, wh)
        if not hit:
            continue
        row = hit[0]
        tl = pixel_to_face_pose(*q[0], cam, r_map_cam, t_map_cam, row)
        tr = pixel_to_face_pose(*q[1], cam, r_map_cam, t_map_cam, row)
        if not tl or not tr or abs(abs(tr[0] - tl[0]) - label_width) > tol * label_width:
            continue
        poly = q.astype(np.float32).reshape(-1, 1, 2)
        if any(cv2.pointPolygonTest(poly, (float(u), float(v)), False) >= 0 for u, v in centers):
            continue
        addr = row.locate(hit[1], hit[2])
        if addr:
            out.append(addr)
    return out

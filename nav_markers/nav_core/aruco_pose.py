"""Поза дрона по ArUco-меткам с известными координатами (документ, разд. 5 и 7).

Цепочка (Gao & Zhang, гл. 2–3, 6.7):  T_wb = T_wm · T_cm⁻¹ · T_bc⁻¹.

PnP по одной метке даёт положение точно, а ориентацию грубо (разд. 5.3: ошибка курса растёт как d²/S²,
и через неё поперечная ошибка положения — как d·σψ). Поэтому положение считаем НЕ из ориентации PnP,
а из вектора «камера → центр метки» t_cm (направление — σpx/f, дальность — d²σpx/(f·S)) и ориентации
дрона, которую уже держит EKF2 по гироскопу:  p_wb = p_wm − R_wb·(R_bc·t_cm + t_bc).
Курс по метке тоже отдаём в EKF2 — с честной дисперсией, чтобы гироскоп не уплывал.

Повторяющиеся ID (словарь 4×4_50): из всех меток с этим ID берём ту, что согласуется с априорной позой
по расстоянию Махаланобиса (χ², 3 ст. св.); если согласуются две — наблюдение отбрасываем.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from invent_core.camera import Camera
from invent_core.localization import MountedCamera

from .markers import Marker, MarkerMap, object_corners

CHI2_3DOF_99 = 11.34
CHI2_1DOF_99 = 6.63


# --- детектор (OpenCV 4.6 в Ubuntu 24.04 и ≥4.7 на Windows: разные API) ------------------------------

def make_detector(dictionary: str, refine: str = "SUBPIX"):
    """-> detect(gray) -> list[(id, corners(4,2))] в порядке TL, TR, BR, BL.
    refine: NONE | SUBPIX | CONTOUR | APRILTAG — метод уточнения углов OpenCV."""
    import cv2
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    method = getattr(cv2.aruco, f"CORNER_REFINE_{refine}")
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = method
        det = cv2.aruco.ArucoDetector(d, params)
        run = det.detectMarkers
    else:
        params = cv2.aruco.DetectorParameters_create()
        params.cornerRefinementMethod = method
        run = lambda g: cv2.aruco.detectMarkers(g, d, parameters=params)  # noqa: E731

    def detect(gray):
        corners, ids, _ = run(gray)
        if ids is None:
            return []
        return [(int(i), np.asarray(c, dtype=float).reshape(4, 2)) for i, c in zip(ids.ravel(), corners)]
    return detect


def solve_marker(corners_px, cam: Camera, size: float):
    """PnP одной метки. -> (R_cm, t_cm, rms_px) — метка в оптическом кадре камеры.

    SQPnP (глобальный минимум ошибки репроекции, ф. 6.36). Не IPPE_SQUARE: в OpenCV 4.13/5.0 на точных
    углах он возвращает позу с ошибкой ~1 px и смещением ~1 см, когда метка видна почти в лоб
    (проверено в tests/test_nav_core.py), — а это обычный ракурс боковой камеры в проходе."""
    import cv2
    k = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]], dtype=float)
    dist = np.array(cam.dist, dtype=float)
    obj = object_corners(size)
    img = np.asarray(corners_px, dtype=float).reshape(4, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, img, k, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    r, _ = cv2.Rodrigues(rvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - img.reshape(4, 2)) ** 2, axis=1))))
    return r, tvec.ravel(), rms


# --- модель ошибок (документ, разд. 5.3) ---------------------------------------------------------------

def marker_sigmas(d: float, f: float, size: float, sigma_px: float) -> tuple[float, float, float]:
    """-> (σ поперёк луча, σ по дальности, σ курса по одной метке) для метки стороной size на дальности d."""
    lateral = d * sigma_px / f
    depth = d * d * sigma_px / (f * size)
    yaw = math.sqrt(2) * sigma_px * d * d / (f * size * size)   # из ракурса: разница длин боковых рёбер
    return lateral, depth, yaw


def expand_corners(corners_px, delta: float) -> np.ndarray:
    """Сдвинуть каждую сторону четырёхугольника наружу на delta пикселей (угол — по сумме нормалей двух сторон)."""
    c = np.asarray(corners_px, dtype=float).reshape(4, 2)
    if not delta:
        return c
    out = c.copy()
    center = c.mean(axis=0)
    for i in range(4):
        shift = np.zeros(2)
        for j in (i - 1, i):                       # стороны (i-1, i) и (i, i+1), сходящиеся в углу i
            a, b = c[j % 4], c[(j + 1) % 4]
            n = np.array([b[1] - a[1], a[0] - b[0]])
            n /= np.linalg.norm(n)
            if n @ (a - center) < 0:              # нормаль — наружу
                n = -n
            shift += n
        # для угла между сторонами с нормалями n1, n2: сдвиг δ·(n1 + n2)/(1 + n1·n2) держит обе стороны на δ
        out[i] = c[i] + delta * shift / max(0.2, 1 + 0.5 * (shift @ shift - 2))
    return out


def yaw_of(r: np.ndarray) -> float:
    return math.atan2(r[1, 0], r[0, 0])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


@dataclass
class Observation:
    """Одна распознанная и привязанная метка."""
    marker: Marker
    camera: str
    distance: float
    p_wb: np.ndarray          # положение дрона по этой метке
    cov_p: np.ndarray         # 3×3
    yaw: float                # курс дрона по ориентации PnP
    var_yaw: float
    rms_px: float
    corners: np.ndarray | None = None   # пиксели углов (4×2) — для совместного уточнения позы


@dataclass
class Fix:
    """Поза дрона по всем меткам кадра (или нескольких камер одного момента)."""
    p: np.ndarray
    cov_p: np.ndarray
    yaw: float
    var_yaw: float
    observations: list[Observation] = field(default_factory=list)


@dataclass
class Prior:
    """Априорная поза от EKF2: положение, его ковариация, ориентация тела (крен/тангаж/курс)."""
    p: np.ndarray
    cov_p: np.ndarray
    r_wb: np.ndarray
    var_yaw: float = math.radians(2.0) ** 2
    lost: bool = False        # позы нет или она недостоверна: принимаем только уникальные (якорные) метки


@dataclass
class Rejection:
    marker_id: int
    camera: str
    reason: str


class MarkerLocalizer:
    """Детекции → привязка к карте → положение и курс дрона с ковариацией."""

    def __init__(self, mm: MarkerMap, cameras: dict[str, MountedCamera], sigma_px: float = 0.5,
                 max_range: float = 4.0, max_view_angle_deg: float = 65.0, max_rms_px: float = 2.0,
                 corner_bias_px: float = 0.0):
        """corner_bias_px — систематический сдвиг найденных углов внутрь метки (размытие + порог детектора).
        Без поправки метка «меньше» настоящей и дальность завышена в 1 + 2δ/w раз (w — сторона в пикселях).
        Калибруется по меткам с известным расстоянием: в рендере Gazebo δ ≈ 0.48 px
        (gazebo/check_marker_texture.py), на реальной камере — так же на стенде."""
        self.mm = mm
        self.corner_bias = corner_bias_px
        self.cameras = cameras
        self.sigma_px = sigma_px
        self.max_range = max_range
        self.cos_view = math.cos(math.radians(max_view_angle_deg))
        self.max_rms = max_rms_px
        self.unique = {m.id for m in mm.markers if len(mm.by_id(m.id)) == 1}

    def observe(self, camera: str, marker_id: int, corners_px, prior: Prior):
        """-> Observation | Rejection."""
        mc = self.cameras[camera]
        cands = self.mm.by_id(marker_id)
        if not cands:
            return Rejection(marker_id, camera, "нет в карте")
        corners_px = expand_corners(corners_px, self.corner_bias)
        sol = solve_marker(corners_px, mc.cam, self.mm.size)
        if sol is None:
            return Rejection(marker_id, camera, "PnP не сошёлся")
        r_cm, t_cm, rms = sol
        d = float(np.linalg.norm(t_cm))
        if rms > self.max_rms:
            return Rejection(marker_id, camera, f"репроекция {rms:.1f} px")
        if d > self.max_range:
            return Rejection(marker_id, camera, f"далеко {d:.1f} м")
        if -t_cm @ r_cm[:, 2] / d < self.cos_view:      # угол между нормалью метки и лучом на камеру
            return Rejection(marker_id, camera, "острый ракурс")

        lat, dep, s_yaw = marker_sigmas(d, mc.cam.fx, self.mm.size, self.sigma_px)
        r_wc = prior.r_wb @ mc.r_bc
        ray = r_wc @ (t_cm / d)
        cov_cam = np.diag([lat ** 2, lat ** 2, dep ** 2])
        # оптический кадр с z вдоль луча: ковариация по лучу/поперёк в map
        z = ray
        x = np.cross([0.0, 0.0, 1.0], z)
        x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0.0, 0.0])
        y = np.cross(z, x)
        basis = np.column_stack([x, y, z])
        horiz_perp = np.array([-ray[1], ray[0], 0.0])
        n = np.linalg.norm(horiz_perp)
        horiz_perp = horiz_perp / n if n > 1e-6 else horiz_perp
        cov = basis @ cov_cam @ basis.T + prior.var_yaw * d * d * np.outer(horiz_perp, horiz_perp)
        v_body = mc.r_bc @ t_cm + mc.t_bc

        best, second = None, None
        for m in cands:
            t_wm = m.pose()
            r_wb_pnp = t_wm[:3, :3] @ r_cm.T @ mc.r_bc.T
            # поза потеряна — курсу EKF2 не верим (например, до первой метки без магнитометра): берём курс PnP
            r_use = rot_z(yaw_of(r_wb_pnp)) @ rot_z(-yaw_of(prior.r_wb)) @ prior.r_wb if prior.lost else prior.r_wb
            p_wb = t_wm[:3, 3] - r_use @ v_body
            if prior.lost:
                d2 = 0.0 if marker_id in self.unique else math.inf
            else:
                e = p_wb - prior.p
                d2 = float(e @ np.linalg.solve(prior.cov_p + cov, e))
            item = (d2, m, p_wb, r_wb_pnp)
            if best is None or d2 < best[0]:
                best, second = item, best
            elif second is None or d2 < second[0]:
                second = item
        if best[0] > CHI2_3DOF_99:
            why = "якоря не видно, поза потеряна" if prior.lost else f"не согласуется с позой (χ²={best[0]:.0f})"
            return Rejection(marker_id, camera, why)
        if second is not None and second[0] <= CHI2_3DOF_99:
            return Rejection(marker_id, camera, "неоднозначный ID")
        _, m, p_wb, r_wb_pnp = best
        # в совместное уточнение — пиксели идеального пинхола (дисторсия снята, ф. 4.10–4.12)
        ideal = np.array([[mc.cam.fx * xn + mc.cam.cx, mc.cam.fy * yn + mc.cam.cy]
                          for xn, yn in (mc.cam.pixel_to_normalized(u, v) for u, v in np.asarray(corners_px).reshape(4, 2))])
        return Observation(m, camera, d, p_wb, cov, yaw_of(r_wb_pnp), s_yaw ** 2, rms, ideal)

    def fuse(self, observations: list[Observation], prior: Prior, iters: int = 8) -> Fix | None:
        """Совместная поза по ВСЕМ углам всех меток кадра (всех камер): минимум ошибки репроекции
        по θ = (x, y, z, ψ) методом Гаусса–Ньютона (Gao & Zhang, ф. 6.36, 6.46). Крен и тангаж — от EKF2:
        по меткам они определяются хуже, чем по акселерометру и гироскопу.

        Ковариация — σpx²·(JᵀJ)⁻¹ (граница Крамера–Рао при гауссовом шуме углов). Одна метка даёт
        курс грубо (разд. 5.3) — это видно по var_yaw; две метки на разных сторонах прохода дают курс по базе
        между ними. Начальное приближение — среднее положений по отдельным меткам и курс EKF2."""
        if not observations:
            return None
        r_tilt = rot_z(-yaw_of(prior.r_wb)) @ prior.r_wb         # крен и тангаж без курса
        if prior.lost:   # начальный курс — средний по PnP отдельных меток (взвешенно)
            w = np.array([1.0 / o.var_yaw for o in observations])
            psi0 = math.atan2(float(w @ np.sin([o.yaw for o in observations])),
                              float(w @ np.cos([o.yaw for o in observations])))
        else:
            psi0 = yaw_of(prior.r_wb)
        w0 = [1.0 / np.trace(o.cov_p) for o in observations]
        p0 = sum(w * o.p_wb for w, o in zip(w0, observations)) / sum(w0)
        theta = np.array([*p0, psi0])
        world = [(o.marker.pose()[:3, :3] @ object_corners(self.mm.size).T).T + o.marker.pose()[:3, 3]
                 for o in observations]
        meas = np.concatenate([o.corners.ravel() for o in observations])

        def residual(th):
            r_wb = rot_z(th[3]) @ r_tilt
            out = []
            for o, pw in zip(observations, world):
                mc = self.cameras[o.camera]
                r_wc = r_wb @ mc.r_bc
                t_wc = th[:3] + r_wb @ mc.t_bc
                pc = (pw - t_wc) @ r_wc                           # (R_wcᵀ (P − t))ᵀ построчно
                if np.any(pc[:, 2] <= 1e-3):
                    return None
                out.append(np.column_stack([mc.cam.fx * pc[:, 0] / pc[:, 2] + mc.cam.cx,
                                            mc.cam.fy * pc[:, 1] / pc[:, 2] + mc.cam.cy]).ravel())
            return np.concatenate(out) - meas

        eps = np.array([1e-5, 1e-5, 1e-5, 1e-6])
        jac = None
        for _ in range(iters):
            e = residual(theta)
            if e is None:
                return None
            jac = np.empty((e.size, 4))
            for k in range(4):
                dt = np.zeros(4)
                dt[k] = eps[k]
                ek = residual(theta + dt)
                if ek is None:
                    return None
                jac[:, k] = (ek - e) / eps[k]
            step = np.linalg.lstsq(jac, -e, rcond=None)[0]
            theta = theta + step
            if np.linalg.norm(step[:3]) < 1e-6 and abs(step[3]) < 1e-7:
                break
        info = jac.T @ jac / self.sigma_px ** 2
        try:
            cov = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            return None
        yaw = math.atan2(math.sin(theta[3]), math.cos(theta[3]))
        return Fix(theta[:3].copy(), cov[:3, :3], yaw, float(cov[3, 3]), observations)

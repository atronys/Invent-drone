"""LOC-11: EKF по пикселям адресных этикеток — якобиан, сходимость, отсев выбросов."""
import math
import random
import unittest

try:
    import numpy as np
    from invent_core.localization import LandmarkLocalizer, MountedCamera, address_landmarks, observe
except ImportError:
    np = None

from invent_core import frames as fr
from invent_core.association import project_to_image
from invent_core.camera import Camera
from invent_core.layout import make_grid_warehouse


@unittest.skipIf(np is None, "нужен numpy")
class TestObservationModel(unittest.TestCase):
    def setUp(self):
        self.cam = Camera(3040, 4056, 22.2, 29.4)
        self.mc = MountedCamera.side(self.cam, "left")

    def test_matches_core_projection(self):
        """Модель наблюдения EKF совпадает с проекцией association.project_to_image (ф. 4.8)."""
        state = np.array([5.0, 2.6, 1.9, 0.1])
        L = np.array([5.3, 4.104, 1.74])
        uv, _ = observe(state, L, self.mc)
        r, t = fr.camera_pose(tuple(state[:3]), state[3], "left", t_body_cam=(0.0, 0.0, 0.12))
        ref = project_to_image(tuple(L), self.cam, r, t)
        self.assertTrue(np.allclose(uv, ref, atol=1e-6))

    def test_jacobian_vs_numeric(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            state = np.array([rng.uniform(2, 20), rng.uniform(2.3, 2.9), rng.uniform(1, 3), rng.uniform(-0.2, 0.2)])
            L = np.array([state[0] + rng.uniform(-0.3, 0.3), 4.104, state[2] + rng.uniform(-0.3, 0.3)])
            roll, pitch = rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)
            _, J = observe(state, L, self.mc, roll, pitch)
            num = np.zeros((2, 4))
            for k in range(4):
                e = np.zeros(4)
                e[k] = 1e-6
                num[:, k] = (observe(state + e, L, self.mc, roll, pitch)[0]
                             - observe(state - e, L, self.mc, roll, pitch)[0]) / 2e-6
            self.assertTrue(np.allclose(J, num, rtol=1e-4, atol=1e-2), (J, num))


@unittest.skipIf(np is None, "нужен numpy")
class TestLocalizerOnPass(unittest.TestCase):
    """Пролёт вдоль прохода 27 м с дрейфом VIO: адресные этикетки держат ошибку в пределах KPI."""

    def run_pass(self, outlier_rate=0.0, gate=True, seed=0):
        rng = random.Random(seed)
        wh = make_grid_warehouse()
        lms = address_landmarks(wh)
        cam = Camera(3040, 4056, 22.2, 29.4)
        mcs = {look: MountedCamera.side(cam, look) for look in ("left", "right")}
        aisle = wh.aisles[0]
        z, dt, v = 1.9, 0.1, 0.4
        truth = np.array([0.3, aisle.y, z, 0.0])
        loc = LandmarkLocalizer(truth.copy(), gate=9.21 if gate else None)
        scale, yaw_bias = 1.02, math.radians(1.0) / 60 * dt     # «плохой VIO»: масштаб 2%, курс 1°/мин
        errs = []
        while truth[0] < 26.7:
            step = v * dt
            truth[0] += step
            loc.predict((step * scale + rng.gauss(0, 0.002), rng.gauss(0, 0.002)), 0.0, yaw_bias,
                        sigma_xy=0.03 * step + 0.002, sigma_z=0.002, sigma_psi=math.radians(0.03))
            if round(truth[0] / dt) % 2 == 0:                      # 5 Гц декодирование
                for look, mc in mcs.items():
                    r, t = fr.camera_pose(tuple(truth[:3]), truth[3], look, t_body_cam=(0.0, 0.0, 0.12))
                    for addr, L in lms.items():
                        if abs(L[0] - truth[0]) > 0.5 or abs(L[1] - truth[1]) > 2.0:
                            continue
                        uv = project_to_image(tuple(L), cam, r, t)
                        if not cam.in_image(uv):
                            continue
                        u, vv = uv[0] + rng.gauss(0, 1.5), uv[1] + rng.gauss(0, 1.5)
                        if rng.random() < outlier_rate:               # прочитали соседнюю ячейку
                            L = L + np.array([0.9, 0.0, 0.0])
                        loc.update_pixel((u, vv), L, mc, sigma_px=2.0)
            errs.append(float(np.hypot(*(loc.mu[:2] - truth[:2]))))
        return max(errs), loc

    def test_bounded_error(self):
        worst, loc = self.run_pass()
        self.assertLess(worst, 0.10)            # KPI Ф4: <= 10 см
        self.assertGreater(loc.accepted, 50)

    def test_gating_handles_misassociation(self):
        worst_g, loc = self.run_pass(outlier_rate=0.1)
        worst_raw, _ = self.run_pass(outlier_rate=0.1, gate=False)
        self.assertGreater(loc.rejected, 0)
        self.assertLess(worst_g, worst_raw)
        self.assertLess(worst_g, 0.10)


if __name__ == "__main__":
    unittest.main()

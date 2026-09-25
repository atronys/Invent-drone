"""Быстрые проверки прототипа EKF-локализации (LOC-08) на маленьком складе."""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "landmark_ekf"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core"))

try:
    import numpy  # noqa: F401
except ImportError:  # необязательная зависимость: без numpy тесты пропускаются, а не падают
    numpy = None
if numpy is not None:
    import ekf_sim as e  # noqa: E402
from invent_core.coverage import ScanParams, plan  # noqa: E402
from invent_core.layout import make_grid_warehouse  # noqa: E402


@unittest.skipIf(numpy is None, "нужен numpy")
class TestLandmarkEKF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wh = make_grid_warehouse(n_aisles=1, bays=6, levels=2)
        p = ScanParams()
        cls.traj = e.true_trajectory(plan(cls.wh, p).sorties[0].waypoints, p, 0.1)

    def run_scenario(self, name, sm=None, seed=1):
        lms = e.make_landmarks(self.wh, e.SCENARIOS[name], random.Random(seed))
        return e.run(self.traj, 0.1, name, lms, sm or e.SensorModel(), seed)

    def test_labels_beat_odometry(self):
        odom, labels = self.run_scenario("odom"), self.run_scenario("labels")
        self.assertLess(labels.ate, odom.ate / 3)
        self.assertLess(labels.max_err, 0.10)          # KPI фазы 4: <= 10 см

    def test_gating_rejects_misassociations(self):
        sm = e.SensorModel(p_misassoc=0.2)
        gated, raw = self.run_scenario("labels", sm), self.run_scenario("labels_nogate", sm)
        self.assertGreater(gated.rejected, 0)
        self.assertLess(gated.ate, raw.ate)

    def test_perfect_sensors(self):
        sm = e.SensorModel(scale_err=0, yaw_bias_deg_min=0, trans_noise=0, trans_noise_min=0, yaw_noise_deg=0,
                           obs_sigma0=1e-4, obs_sigma_k=0, p_misassoc=0)
        self.assertLess(self.run_scenario("odom", sm).ate, 1e-6)


if __name__ == "__main__":
    unittest.main()

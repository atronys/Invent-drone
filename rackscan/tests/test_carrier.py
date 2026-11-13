"""Шлюз безопасности носителя: что запрещено исполнять, даже если автомат просит."""
import unittest

from rackscan.carrier import CarrierLimits, gate


def snap(phase="APPROACH", distance=1.0, yaw=0.0, markers=3, ambiguous=False, tof=None,
         vx=0.2, vy=0.1, vz=0.05, wz=5.0):
    return {"phase": phase,
            "pose": {"distance_m": distance, "yaw_deg": yaw, "markers": markers, "ambiguous": ambiguous},
            "velocity": {"vx_forward": vx, "vy_right": vy, "vz_up": vz, "yaw_rate_dps": wz},
            "sensors": {"tof_mm": tof}}


class GateTest(unittest.TestCase):
    def test_passes_normal_setpoint(self):
        s = gate(snap(), 0.05)
        self.assertEqual((s.vx_forward, s.vy_right, s.vz_up, s.yaw_rate_dps, s.hold), (0.2, 0.1, 0.05, 5.0, False))

    def test_saturates(self):
        s = gate(snap(vx=3.0, vy=-3.0, vz=1.0, wz=90.0), 0.05)
        self.assertEqual((s.vx_forward, s.vy_right, s.vz_up, s.yaw_rate_dps), (0.3, -0.3, 0.2, 20.0))

    def test_holds_outside_moving_phases(self):
        for phase in ("IDLE", "SEARCH", "ASSESS", "READ", "DONE", "FINISHED"):
            s = gate(snap(phase=phase), 0.05)
            self.assertTrue(s.hold, phase)
            self.assertEqual((s.vx_forward, s.vy_right, s.vz_up, s.yaw_rate_dps), (0, 0, 0, 0))

    def test_holds_without_fresh_pose(self):
        self.assertTrue(gate(snap(), 0.5).hold)
        lost = snap()
        lost["pose"] = None
        self.assertTrue(gate(lost, 0.0).hold)

    def test_never_closer_than_minimum(self):
        s = gate(snap(distance=0.44), 0.05)
        self.assertEqual(s.vx_forward, 0.0)
        self.assertEqual(s.vy_right, 0.1)                       # вдоль фасада двигаться можно
        self.assertEqual(gate(snap(distance=0.44, vx=-0.2), 0.05).vx_forward, -0.2)   # отход разрешён

    def test_single_marker_forbids_approach(self):
        for kw in ({"markers": 1}, {"ambiguous": True}):
            s = gate(snap(**kw), 0.05)
            self.assertEqual((s.vx_forward, s.vy_right), (0.0, 0.05), kw)
            self.assertIn("одной метке", s.reason)

    def test_large_yaw_only_rotates(self):
        s = gate(snap(yaw=50.0), 0.05)
        self.assertEqual((s.vx_forward, s.vy_right, s.vz_up), (0.0, 0.0, 0.0))
        self.assertEqual(s.yaw_rate_dps, 5.0)
        self.assertFalse(s.hold)

    def test_tof_obstacle_blocks_forward(self):
        self.assertEqual(gate(snap(distance=1.2, tof=700), 0.05).vx_forward, 0.0)
        self.assertEqual(gate(snap(distance=1.2, tof=1150), 0.05).vx_forward, 0.2)   # дальномер согласен с камерой

    def test_far_limit_blocks_retreat(self):
        self.assertEqual(gate(snap(distance=2.6, vx=-0.2), 0.05).vx_forward, 0.0)

    def test_real_snapshot_shape(self):
        """Шлюз принимает настоящий снимок автомата, а не только рукописный."""
        from rackscan.layout import Layout
        from rackscan.optics import camera_from_preset
        from rackscan.scanner import Frame, Scanner
        from rackscan.wms.demo_data import build_demo
        lay = Layout.demo()
        _, scene, _ = build_demo(lay)
        cam = camera_from_preset("imx708", (1152, 648))
        sc = Scanner(lay, cam)
        sc.start_task(["01-02-02"])
        s = gate(sc.step(Frame(scene.render(cam, (0.2, 0.72, 1.3)), 0.0)), 0.0, CarrierLimits())
        self.assertFalse(s.hold)
        self.assertGreater(s.vy_right, 0.2)
        self.assertGreater(s.vx_forward, 0.2)


if __name__ == "__main__":
    unittest.main()

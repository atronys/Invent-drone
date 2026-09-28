"""SAFE-01/02: узел safety_monitor на заглушках rclpy — depth-кадр + позиция PX4 -> safety/state."""
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ros_stubs as rs  # noqa: E402

try:
    import numpy as np
    DEPS = True
except ImportError:
    DEPS = False

PKG = str(ROOT / "ros2_ws" / "src" / "invent_mission")
W, H, HFOV = 640, 480, 73.0     # OakD-Lite в модели PX4


@unittest.skipUnless(DEPS, "нужен numpy")
class SafetyNodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from invent_core.layout import make_grid_warehouse
        cls.tmp = tempfile.TemporaryDirectory()
        cls.wh = make_grid_warehouse(n_aisles=1, bays=4, levels=2)
        cls.layout = str(Path(cls.tmp.name) / "layout.json")
        cls.wh.save(cls.layout)
        cls.fx = W / (2 * math.tan(math.radians(HFOV) / 2))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def floor(self, cam_h):
        yn = (np.arange(H)[:, None] - H / 2) / self.fx
        return (np.where(yn > 1e-6, cam_h / np.maximum(yn, 1e-9), np.inf) * np.ones((1, W))).astype(np.float32)

    def run_frames(self, frames, pos_map=None, encoding="32FC1"):
        """frames — список depth-кадров (м); -> список состояний safety/state."""
        from invent_core.frames import map_to_px4
        with rs.installed(PKG):
            rs.PARAMS["safety_monitor"]["layout"] = self.layout
            from invent_mission.safety_node import SafetyNode
            from px4_msgs.msg import VehicleLocalPosition
            from sensor_msgs.msg import Image
            node = SafetyNode()
            if pos_map is not None:
                x, y, z = map_to_px4(pos_map, self.wh.dock)
                rs.publish("/fmu/out/vehicle_local_position_v1",
                           VehicleLocalPosition(x=x, y=y, z=z, xy_valid=True, z_valid=True))
            for d in frames:
                arr = (np.nan_to_num(d, posinf=0.0) * 1000).astype(np.uint16) if encoding == "16UC1" else d
                rs.publish("depth", Image(height=H, width=W, encoding=encoding, array=arr))
            return [m.data for m in rs.SENT["safety/state"]], node

    def test_floor_on_dock_is_go(self):
        """Регрессия: пол перед камерой на доке раньше давал STOP — дрон не взлетал."""
        states, _ = self.run_frames([self.floor(0.30)] * 3, pos_map=self.wh.dock)
        self.assertEqual(states, ["GO"] * 3)

    def test_box_ahead_stops_and_clears_after_5_frames(self):
        dock = self.wh.dock
        box = self.floor(0.30)
        box[200:280, 300:340] = 0.8
        states, _ = self.run_frames([box] + [self.floor(0.30)] * 6, pos_map=dock)
        self.assertEqual(states[0], "STOP")
        self.assertEqual(states[1:5], ["STOP"] * 4)
        self.assertEqual(states[5:], ["GO"] * 2)

    def test_millimetre_depth(self):
        box = np.full((H, W), np.inf, np.float32)
        box[200:280, 300:340] = 0.8
        states, _ = self.run_frames([box], pos_map=(0.0, self.wh.aisles[0].y, 1.5), encoding="16UC1")
        self.assertEqual(states, ["STOP"])

    def test_outside_geofence_stops(self):
        row = self.wh.row(self.wh.aisles[0].rows[0])
        inside_rack = (2.0, row.face_y + row.normal_y * 0.5, 1.5)
        states, node = self.run_frames([np.full((H, W), np.inf, np.float32)], pos_map=inside_rack)
        self.assertEqual(states, ["STOP"])
        self.assertIn("геозоны", node.get_logger().text())


if __name__ == "__main__":
    unittest.main()

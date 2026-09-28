"""SAFE-01/02: пузырь безопасности и геозона."""
import math
import unittest

from invent_core.camera import Camera
from invent_core.coverage import ScanParams, plan
from invent_core.layout import make_grid_warehouse
from invent_core.safety import GO, SLOW, STOP, Geofence, SafetyMonitor, corridor_min_distance, stopping_distance

try:
    import numpy as np
except ImportError:
    np = None


class TestGeofence(unittest.TestCase):
    def test_every_plan_stays_inside(self):
        """Свойство: никакой маршрут планировщика не выходит из геозоны (при разных складах и камерах)."""
        for n_aisles, bays, levels in [(1, 4, 2), (2, 10, 4), (3, 12, 5)]:
            wh = make_grid_warehouse(n_aisles=n_aisles, bays=bays, levels=levels)
            gf = Geofence(wh)
            for cam in (Camera(), Camera(3040, 4056, 22.2, 29.4)):
                pl = plan(wh, ScanParams(camera=cam, min_z=0.5, cover_address_labels=True))
                for s in pl.sorties:
                    self.assertEqual(gf.check_route(s.waypoints), [])

    def test_inside_racks_is_forbidden(self):
        wh = make_grid_warehouse()
        gf = Geofence(wh)
        row = wh.row("B")                                          # фасад y=4.1, стеллаж глубже (+y)
        self.assertFalse(gf.contains((5.0, row.face_y + 0.5, 2.0)))  # внутри стеллажа
        self.assertFalse(gf.contains((5.0, row.face_y - 0.2, 2.0)))  # вплотную к фасаду (нет запаса)
        self.assertTrue(gf.contains((5.0, wh.aisles[0].y, 2.0)))     # ось прохода
        self.assertFalse(gf.contains((5.0, wh.aisles[0].y, wh.ceiling_z)))  # под потолком


@unittest.skipIf(np is None, "нужен numpy")
class TestSafetyBubble(unittest.TestCase):
    def setUp(self):
        self.cam = Camera.from_intrinsics(640, 480, 450, 450, 320, 240)

    def test_obstacle_in_corridor(self):
        depth = np.full((480, 640), 7.0, np.float32)             # стена далеко
        depth[200:280, 300:340] = 1.2                             # коробка по курсу
        self.assertAlmostEqual(corridor_min_distance(depth, self.cam), 1.2, places=5)

    def test_obstacle_outside_corridor_ignored(self):
        depth = np.full((480, 640), np.nan, np.float32)
        depth[:, :60] = 1.0                                       # стеллаж сбоку: на 1 м x ≈ -0.6 м — вне коридора
        self.assertEqual(corridor_min_distance(depth, self.cam), math.inf)

    def floor(self, h):
        """Depth-кадр пустого пола для камеры на высоте h без наклона."""
        yn = (np.arange(480)[:, None] - self.cam.cy) / self.cam.fy
        return np.where(yn > 1e-6, h / np.maximum(yn, 1e-9), np.inf) * np.ones((1, 640))

    def test_floor_is_not_an_obstacle(self):
        """Камера на доке ~0.3 м над полом: без высоты пол даёт STOP (< 1 м), с высотой — нет."""
        depth = self.floor(0.3)
        self.assertLess(corridor_min_distance(depth, self.cam), 1.0)
        self.assertEqual(corridor_min_distance(depth, self.cam, cam_height=0.3), math.inf)
        # недооценка высоты (опоры шасси не учтены) — пол всё равно отброшен
        self.assertEqual(corridor_min_distance(depth, self.cam, cam_height=0.25), math.inf)

    def test_box_on_floor_still_detected(self):
        depth = self.floor(0.3)
        box = np.zeros_like(depth, dtype=bool)
        box[240 - int(450 * 0.1 / 1.5):, 300:340] = True           # коробка высотой 0.4 м в 1.5 м: верх на 0.1 м выше оси
        depth = np.where(box & (depth > 1.5), 1.5, depth)
        self.assertAlmostEqual(corridor_min_distance(depth, self.cam, cam_height=0.3), 1.5, places=5)

    def test_monitor_hysteresis(self):
        m = SafetyMonitor(clear_frames=3)
        self.assertEqual(m.update(5.0, 0.4), GO)
        self.assertEqual(m.update(0.9, 0.4), STOP)                # ближе 1 м
        self.assertEqual(m.update(5.0, 0.0), STOP)                # не сразу
        self.assertEqual(m.update(5.0, 0.0), STOP)
        self.assertEqual(m.update(5.0, 0.0), GO)
        self.assertEqual(m.update(stopping_distance(1.0) * 1.5, 1.0), SLOW)


if __name__ == "__main__":
    unittest.main()

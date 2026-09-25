"""LEARN-05: калибровка по синтетическим снимкам шахматки с известной матрицей K."""
import sys
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core"))


def render_board(k, rvec, tvec, board=(9, 6), square=0.03, size=(1280, 720)):
    px = 50                                               # пикселей на клетку в текстуре
    cols, rows = board[0] + 1, board[1] + 1
    tex = np.full((rows * px + 2 * px, cols * px + 2 * px), 255, np.uint8)
    for j in range(rows):
        for i in range(cols):
            if (i + j) % 2 == 0:
                tex[px + j * px: px + (j + 1) * px, px + i * px: px + (i + 1) * px] = 0
    # текстурный пиксель (u, v) <-> метрическая точка ((u - 2px)/px * square, (v - 2px)/px * square)
    th, tw = tex.shape
    corners_t = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], np.float32)
    corners_m = np.array([[(u - 2 * px) / px * square, (v - 2 * px) / px * square, 0] for u, v in corners_t])
    proj, _ = cv2.projectPoints(corners_m, rvec, tvec, k, None)
    h = cv2.getPerspectiveTransform(corners_t, proj.reshape(4, 2).astype(np.float32))
    img = cv2.warpPerspective(tex, h, size, borderValue=128)
    return cv2.GaussianBlur(img, (3, 3), 0)


@unittest.skipIf(cv2 is None, "нужны numpy и opencv-python")
class TestCalibration(unittest.TestCase):
    def test_recovers_intrinsics(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import calibrate_camera as cc
        from invent_core.camera import Camera
        k = np.array([[1200.0, 0, 640], [0, 1200.0, 360], [0, 0, 1]])
        rng = np.random.default_rng(0)
        images = []
        for _ in range(14):
            rvec = np.radians(rng.uniform([-30, -30, -10], [30, 30, 10]))
            tvec = np.array([rng.uniform(-0.2, -0.05), rng.uniform(-0.15, -0.05), rng.uniform(0.55, 0.8)])
            images.append(render_board(k, rvec, tvec))
        rms, k_est, dist, size, n = cc.calibrate(images, (9, 6), 0.03)
        self.assertGreaterEqual(n, 10)
        self.assertLess(rms, 0.5)
        self.assertLess(abs(k_est[0, 0] - 1200) / 1200, 0.01)
        self.assertLess(abs(k_est[0, 2] - 640), 8)
        # результат читается ядром
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cam.yaml"
            p.write_text(cc.to_camera_info(rms, k_est, dist, size, n), encoding="utf-8")
            cam = Camera.load(p)
        self.assertAlmostEqual(cam.fx, float(k_est[0, 0]), places=3)


if __name__ == "__main__":
    unittest.main()

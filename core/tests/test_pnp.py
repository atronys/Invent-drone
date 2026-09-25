"""LEARN-07: поза этикетки по 4 углам (PnP, гл. 6.7) на синтетическом снимке с известной истиной."""
import math
import unittest

try:
    import cv2
    import numpy as np
except ImportError:  # необязательные зависимости ядра
    cv2 = None

from invent_core.camera import Camera


def render_label(cam, rvec, tvec, size=(0.148, 0.105), seed=0):
    """Этикетка (белая, со «штрихкодом») на тёмном фоне паллеты в заданной позе."""
    from invent_core.pnp import camera_matrix, object_points
    rng = np.random.default_rng(seed)
    tex = np.full((1050, 1480), 250, np.uint8)
    x = 250
    while x < 1230:
        w = int(rng.integers(4, 16))
        if rng.random() < 0.5:
            tex[300:700, x:x + w] = 10
        x += w
    k, dist = camera_matrix(cam), np.array(cam.dist)
    proj, _ = cv2.projectPoints(object_points(size), rvec, tvec, k, dist)
    src = np.array([[0, 0], [1480, 0], [1480, 1050], [0, 1050]], np.float32)
    h = cv2.getPerspectiveTransform(src, proj.reshape(4, 2).astype(np.float32))
    wh = (cam.width, cam.height)
    warped = cv2.warpPerspective(tex, h, wh)
    mask = cv2.warpPerspective(np.full_like(tex, 255), h, wh)
    img = np.where(mask > 127, warped, 80).astype(np.float32) + rng.normal(0, 3, (cam.height, cam.width))
    return np.clip(img, 0, 255).astype(np.uint8), proj.reshape(4, 2)


@unittest.skipIf(cv2 is None, "нужны numpy и opencv-python")
class TestLabelPose(unittest.TestCase):
    def setUp(self):
        self.cam = Camera.from_intrinsics(1920, 1080, 1500, 1500, 960, 540)

    def check(self, rot_deg, tvec):
        from invent_core.pnp import detect_label_quad, label_pose
        rvec = np.radians(np.array(rot_deg, dtype=np.float64))
        tvec = np.array(tvec, dtype=np.float64)
        img, truth_px = render_label(self.cam, rvec, tvec)
        quad = detect_label_quad(img)
        self.assertIsNotNone(quad)
        self.assertLess(np.abs(quad - truth_px).max(), 2.0)           # углы найдены с точностью ~px
        pose = label_pose(quad, self.cam)
        d_true = float(np.linalg.norm(tvec))
        self.assertLess(abs(pose.distance - d_true) / d_true, 0.01)                # дистанция < 1%
        self.assertLess(float(np.linalg.norm(pose.t - tvec)) / d_true, 0.01)       # положение < 1%
        # Ориентация маленькой плоской цели при почти фронтальном взгляде определяется плохо:
        # 0.5 px ошибки углов -> до ~2.5°. Поэтому в LOC-08 наблюдение — положение этикетки, не поворот.
        r_true = cv2.Rodrigues(rvec)[0]
        ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r_true.T @ pose.R) - 1) / 2))))
        self.assertLess(ang, 4.0)
        self.assertLess(pose.reproj_rmse_px, 1.0)

    def test_frontal(self):
        self.check((0, 0, 0), (0.05, -0.03, 1.2))

    def test_oblique(self):
        self.check((-10, 25, 3), (0.15, -0.10, 1.5))

    def test_far_small(self):
        self.check((5, -15, 0), (-0.3, 0.2, 2.5))


if __name__ == "__main__":
    unittest.main()

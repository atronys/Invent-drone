"""Источники кадров: камера Raspberry Pi, USB-камера, папка с кадрами, синтетическая сцена.

У каждого источника два режима: поток предпросмотра (быстро, для наведения) и полный кадр
(медленно, для чтения кода и фото-доказательства).

ВАЖНО: PiCameraSource написан по документации Picamera2 и на реальной плате в рамках этой
работы не запускался — первым делом проверьте его по разд. «Проверка камеры» руководства.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .optics import CAMERA_PRESETS, CameraModel, camera_from_preset
from .scanner import APPROACH, NAVIGATE, SEARCH, Frame, follow_commands


def load_calibration(path: str | Path) -> CameraModel:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return CameraModel(int(d["width"]), int(d["height"]), float(d["fx"]), float(d["fy"]), float(d["cx"]),
                       float(d["cy"]), tuple(float(x) for x in d.get("dist", [])))


def camera_models(cam_cfg: dict) -> tuple[CameraModel, CameraModel]:
    """(модель предпросмотра, модель полного кадра) по разделу camera настроек."""
    preset = cam_cfg.get("preset", "imx708")
    pw, ph = cam_cfg.get("preview_size") or CAMERA_PRESETS[preset]["full"]
    sw, sh = cam_cfg.get("still_size") or CAMERA_PRESETS[preset]["full"]
    if cam_cfg.get("calibration"):
        base = load_calibration(cam_cfg["calibration"])
        return base.scaled(pw, ph), base.scaled(sw, sh)
    return camera_from_preset(preset, (pw, ph)), camera_from_preset(preset, (sw, sh))


class BaseSource:
    preview_cam: CameraModel
    still_cam: CameraModel

    def read(self) -> Frame | None:
        raise NotImplementedError

    def capture_still(self) -> np.ndarray | None:
        return None

    def after_step(self, snap: dict) -> None:
        """Вызывается после обработки кадра (нужно только имитатору)."""

    def close(self) -> None:
        pass


class SyntheticSource(BaseSource):
    """Виртуальная камера в синтетической сцене; умеет сама выполнять команды (автопилот-имитатор)."""

    def __init__(self, scene, preview_cam: CameraModel, still_cam: CameraModel, fps: float = 8.0,
                 autopilot: bool = False, seed: int = 0):
        self.scene, self.preview_cam, self.still_cam = scene, preview_cam, still_cam
        self.position = np.array([0.3, 0.6, 1.6])
        self.yaw = 12.0
        self.pitch = 0.0
        self.roll = 0.0
        self.autopilot = autopilot
        self.period = 1.0 / fps
        self._rng = np.random.default_rng(seed)
        self._t0 = time.monotonic()
        self._next = 0.0

    def read(self) -> Frame:
        now = time.monotonic() - self._t0
        if now < self._next:
            time.sleep(self._next - now)
        self._next = max(now, self._next) + self.period
        img = self.scene.render(self.preview_cam, self.position, self.yaw, self.pitch, self.roll)
        return Frame(img, time.monotonic(), gyro_dps=0.5,
                     tof_mm=self.scene.tof_mm(self.position, self.yaw, self.pitch), tilt_deg=(self.roll, self.pitch))

    def capture_still(self) -> np.ndarray:
        return self.scene.render(self.still_cam, self.position, self.yaw, self.pitch, self.roll)

    def move(self, dx=0.0, dy=0.0, dz=0.0, dyaw=0.0) -> None:
        self.position = self.position + np.array([dx, dy, dz])
        self.position[2] = max(0.15, self.position[2])
        self.yaw = float(np.clip(self.yaw + dyaw, -60, 60))

    def after_step(self, snap: dict) -> None:
        if self.autopilot and snap["phase"] in (NAVIGATE, APPROACH, SEARCH) and snap["errors"]:
            self.position, self.yaw = follow_commands(self.position, self.yaw, snap, gain=0.45, max_step_m=0.12)
            self.position = self.position + self._rng.normal(0, 0.003, 3)
            self.yaw += float(self._rng.normal(0, 0.2))


class OpenCVSource(BaseSource):
    """USB-камера или видеофайл через OpenCV. Полный кадр — тот же кадр предпросмотра."""

    def __init__(self, device, preview_cam: CameraModel):
        self.preview_cam = self.still_cam = preview_cam
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, preview_cam.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, preview_cam.height)
        if not self.cap.isOpened():
            raise RuntimeError(f"камера {device!r} не открылась")
        self._last: np.ndarray | None = None

    def read(self) -> Frame | None:
        ok, img = self.cap.read()
        if not ok:
            return None
        if img.shape[1] != self.preview_cam.width:
            img = cv2.resize(img, (self.preview_cam.width, self.preview_cam.height))
        self._last = img
        return Frame(img, time.monotonic())

    def capture_still(self) -> np.ndarray | None:
        return None if self._last is None else self._last.copy()

    def close(self) -> None:
        self.cap.release()


class FileSource(BaseSource):
    """Кадры из папки по кругу — для разбора записей со стенда."""

    def __init__(self, folder: str | Path, preview_cam: CameraModel, fps: float = 5.0):
        self.files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not self.files:
            raise RuntimeError(f"в папке {folder} нет кадров")
        self.preview_cam = self.still_cam = preview_cam
        self.period, self._i, self._last = 1.0 / fps, 0, None

    def read(self) -> Frame | None:
        time.sleep(self.period)
        img = cv2.imread(str(self.files[self._i % len(self.files)]))
        self._i += 1
        if img is None:
            return None
        self._last = img
        if img.shape[1] != self.preview_cam.width:
            img = cv2.resize(img, (self.preview_cam.width, self.preview_cam.height), interpolation=cv2.INTER_AREA)
        return Frame(img, time.monotonic(), meta={"file": self.files[(self._i - 1) % len(self.files)].name})

    def capture_still(self) -> np.ndarray | None:
        return self._last


class PiCameraSource(BaseSource):
    """Камера Raspberry Pi через Picamera2 (libcamera).

    Предпросмотр идёт в режиме сенсора с полным полем зрения (для IMX708 — 2304×1296 с
    биннингом 2×2), уменьшенном до preview_size; полный кадр снимается переключением режима.
    Формат «RGB888» в Picamera2 даёт порядок байт B, G, R — то, что ждёт OpenCV.
    """

    def __init__(self, preview_cam: CameraModel, still_cam: CameraModel, fps: int = 15,
                 lens_position: float | None = None, short_exposure: bool = True, full_fov_raw=(2304, 1296)):
        from picamera2 import Picamera2  # импорт здесь: модуль есть только на Raspberry Pi
        self.preview_cam, self.still_cam = preview_cam, still_cam
        self.cam = Picamera2()
        video = {"main": {"size": (preview_cam.width, preview_cam.height), "format": "RGB888"},
                 "controls": {"FrameRate": fps}}
        if full_fov_raw:
            video["raw"] = {"size": tuple(full_fov_raw)}
        self._video = self.cam.create_video_configuration(**video)
        self._still = self.cam.create_still_configuration(
            main={"size": (still_cam.width, still_cam.height), "format": "RGB888"})
        self.cam.configure(self._video)
        ctrl: dict = {}
        if lens_position is not None:
            ctrl.update({"AfMode": 0, "LensPosition": float(lens_position)})     # ручной фокус
        else:
            ctrl.update({"AfMode": 2})                                           # непрерывный автофокус
        if short_exposure:
            ctrl["AeExposureMode"] = 1                                           # короткие выдержки
        try:
            self.cam.set_controls(ctrl)
        except Exception:   # у камер без автофокуса (HQ Camera) этих регуляторов нет
            pass
        self.cam.start()
        time.sleep(1.0)

    def read(self) -> Frame | None:
        img = self.cam.capture_array("main")
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return Frame(img, time.monotonic())

    def capture_still(self) -> np.ndarray | None:
        img = self.cam.switch_mode_and_capture_array(self._still, "main")
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def close(self) -> None:
        self.cam.stop()
        self.cam.close()


def make_source(cfg: dict, scene=None) -> BaseSource:
    cam = cfg["camera"]
    preview, still = camera_models(cam)
    kind = cam.get("source", "synthetic")
    if kind == "synthetic":
        if scene is None:
            raise ValueError("для синтетического источника нужна сцена")
        return SyntheticSource(scene, preview, still, fps=min(float(cam.get("fps", 8)), 8.0), seed=cfg.get("seed", 0))
    if kind == "picamera":
        raw = (2304, 1296) if cam.get("preset", "").startswith("imx708") else None
        return PiCameraSource(preview, still, int(cam.get("fps", 15)), cam.get("lens_position"),
                              bool(cam.get("short_exposure", True)), raw)
    if kind == "opencv":
        return OpenCVSource(cam.get("device", 0), preview)
    if kind == "files":
        return FileSource(cam["path"], preview, float(cam.get("fps", 5)))
    raise ValueError(f"неизвестный источник кадров: {kind}")

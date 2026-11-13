"""Адресные метки ArUco: обнаружение и отрисовка.

Обёртка скрывает разницу API OpenCV: до 4.7 — функции cv2.aruco.detectMarkers / drawMarker,
с 4.7 — класс cv2.aruco.ArucoDetector / generateImageMarker. В Raspberry Pi OS Bookworm
системный OpenCV 4.6, в Trixie — 4.10, на ПК разработчика может стоять 5.x.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

aruco = cv2.aruco


@dataclass(frozen=True)
class MarkerDetection:
    id: int
    corners: np.ndarray  # (4, 2) float32: ЛВ, ПВ, ПН, ЛН

    @property
    def center(self) -> np.ndarray:
        return self.corners.mean(axis=0)

    @property
    def side_px(self) -> float:
        c = self.corners
        return float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))


def dictionary_bits(name: str) -> int:
    """Число бит на сторону: DICT_5X5_1000 -> 5."""
    return int(re.search(r"DICT_(\d)X\d", name).group(1))


@lru_cache(maxsize=None)
def get_dictionary(name: str):
    code = getattr(aruco, name)
    if hasattr(aruco, "getPredefinedDictionary"):
        return aruco.getPredefinedDictionary(code)
    return aruco.Dictionary_get(code)  # OpenCV < 4.7


@lru_cache(maxsize=None)
def _detector(name: str):
    params = aruco.DetectorParameters() if hasattr(aruco, "DetectorParameters") and \
        not hasattr(aruco, "DetectorParameters_create") else aruco.DetectorParameters_create()
    # Субпиксельное уточнение углов: без него поза по PnP шумит на сантиметры.
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.minMarkerPerimeterRate = 0.02
    if hasattr(aruco, "ArucoDetector"):
        return aruco.ArucoDetector(get_dictionary(name), params), None
    return None, params


def detect(image: np.ndarray, dictionary: str, max_width: int | None = 960) -> list[MarkerDetection]:
    """Найти метки на кадре (BGR или серый).

    Поиск идёт на уменьшенной копии (не шире max_width) — это в 2–4 раза быстрее, — после чего
    углы уточняются до субпикселя уже на исходном кадре. max_width=None — искать в полном размере.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    k = 1.0
    small = gray
    if max_width and gray.shape[1] > max_width:
        k = gray.shape[1] / float(max_width)
        small = cv2.resize(gray, (max_width, int(round(gray.shape[0] / k))), interpolation=cv2.INTER_AREA)
    det, params = _detector(dictionary)
    if det is not None:
        corners, ids, _ = det.detectMarkers(small)
    else:
        corners, ids, _ = aruco.detectMarkers(small, get_dictionary(dictionary), parameters=params)
    if ids is None:
        return []
    out = []
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    for c, i in zip(corners, np.asarray(ids).ravel()):
        pts = np.asarray(c, dtype=np.float32).reshape(4, 2)
        if k > 1.0:
            pts = (pts + 0.5) * k - 0.5
            side = float(np.mean([np.linalg.norm(pts[j] - pts[(j + 1) % 4]) for j in range(4)]))
            win = int(max(2, min(8, side / 12)))
            pts = cv2.cornerSubPix(gray, pts.reshape(-1, 1, 2).copy(), (win, win), (-1, -1), crit).reshape(4, 2)
        out.append(MarkerDetection(int(i), pts))
    return out


def render(dictionary: str, marker_id: int, side_px: int) -> np.ndarray:
    """Изображение метки (серое, без белого поля) со стороной side_px."""
    d = get_dictionary(dictionary)
    if hasattr(aruco, "generateImageMarker"):
        return aruco.generateImageMarker(d, int(marker_id), int(side_px))
    return aruco.drawMarker(d, int(marker_id), int(side_px))

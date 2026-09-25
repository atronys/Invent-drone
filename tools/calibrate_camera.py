"""Калибровка камеры по шахматке (LEARN-05, PER-02) -> YAML в формате ROS camera_info.

Модель — пинхол + дисторсия rad-tan (Gao & Zhang, гл. 4.1, ф. 4.10–4.12); метод — Чжана
(книга ссылается на него в гл. 4.1.1, реализация — cv2.calibrateCamera).

    python tools/calibrate_camera.py "calib/*.jpg" --board 9x6 --square 0.025 -o camera.yaml

--board — число ВНУТРЕННИХ углов (столбцы x строки), --square — сторона клетки, м.
Советы: 15–30 снимков, доска во всех углах кадра и под наклонами до ~45°, без смаза,
фокус и зум не менять. Цель — RMS репроекции < 0.5 px (LEARN-05).
"""
from __future__ import annotations

import argparse
import glob
import sys

import cv2
import numpy as np

CAMERA_INFO = """image_width: {w}
image_height: {h}
camera_name: {name}
camera_matrix:
  rows: 3
  cols: 3
  data: [{k}]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [{d}]
# RMS репроекции: {rms:.3f} px по {n} снимкам
"""


def find_corners(gray, board: tuple[int, int]):
    ok, corners = cv2.findChessboardCorners(gray, board, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)


def calibrate(images, board: tuple[int, int], square: float):
    """images — список серых изображений. -> (rms, K, dist, (w, h), n_used)."""
    obj = np.zeros((board[0] * board[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * square
    obj_pts, img_pts, size = [], [], None
    for g in images:
        size = (g.shape[1], g.shape[0])
        c = find_corners(g, board)
        if c is not None:
            obj_pts.append(obj)
            img_pts.append(c)
    if len(obj_pts) < 3:
        raise ValueError(f"доска найдена только на {len(obj_pts)} снимках (нужно >= 3, лучше 15+)")
    rms, k, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    return rms, k, dist.ravel()[:5], size, len(obj_pts)


def to_camera_info(rms, k, dist, size, n, name: str = "camera") -> str:
    return CAMERA_INFO.format(w=size[0], h=size[1], name=name, rms=rms, n=n,
                              k=", ".join(f"{v:.6f}" for v in k.ravel()),
                              d=", ".join(f"{v:.8f}" for v in dist))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", help='маска снимков, например "calib/*.jpg"')
    ap.add_argument("--board", default="9x6")
    ap.add_argument("--square", type=float, required=True)
    ap.add_argument("--name", default="camera")
    ap.add_argument("-o", "--out", default="camera.yaml")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    board = tuple(int(v) for v in a.board.lower().split("x"))
    files = sorted(glob.glob(a.pattern))
    images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in files]
    rms, k, dist, size, n = calibrate([g for g in images if g is not None], board, a.square)
    open(a.out, "w", encoding="utf-8").write(to_camera_info(rms, k, dist, size, n, a.name))
    print(f"снимков: {len(files)}, доска найдена: {n}; RMS = {rms:.3f} px "
          f"({'OK' if rms < 0.5 else 'больше цели 0.5 px — переснимите'})")
    print(f"fx={k[0, 0]:.1f} fy={k[1, 1]:.1f} cx={k[0, 2]:.1f} cy={k[1, 2]:.1f} dist={np.round(dist, 5).tolist()}")
    print(f"сохранено: {a.out}  (читается invent_core.camera.Camera.load)")


if __name__ == "__main__":
    main()

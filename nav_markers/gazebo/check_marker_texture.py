"""Проверка в Gazebo: метка в мире лежит так, как её описывает карта (кадр OpenCV), — по реальному рендеру.

Ошибку ориентации текстуры (поворот на 90°, зеркало) синтетические тесты не ловят: ID детектор найдёт и в
повёрнутой метке, но поза выйдет повёрнутой. Здесь: мир из нескольких меток с разными нормалями и неподвижные
камеры, которые пишут кадры на диск (<save>); затем по кадрам — PnP и сравнение позы камеры с истиной.

    python3 nav_markers/gazebo/check_marker_texture.py            # в WSL/Linux с Gazebo Harmonic
    -> «OK» и ошибки положения камеры по каждой метке, либо описание расхождения
"""
from __future__ import annotations

import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "nav_markers"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nav_core  # noqa: E402,F401
from invent_core.camera import Camera  # noqa: E402
from invent_core.frames import R_LINK_OPTICAL, euler_zyx  # noqa: E402
from make_world import BORDER, marker_models  # noqa: E402
from nav_core.aruco_pose import make_detector, solve_marker  # noqa: E402
from nav_core.markers import Marker, MarkerMap, render_marker  # noqa: E402

W, H, HFOV = 640, 480, math.radians(70)
# (метка, положение камеры, курс камеры): камеры смотрят на метку под углом, чтобы зеркало/поворот были видны
CASES = [
    (Marker(3, "path", "1", 2.0, 0.0, 1.5, (-1.0, 0.0, 0.0)), (0.0, 0.5, 1.4), 0.0),
    (Marker(7, "path", "1", 10.0, 3.0, 1.5, (0.0, -1.0, 0.0)), (10.6, 1.4, 1.6), math.pi / 2),
    (Marker(11, "path", "1", 20.0, -3.0, 1.5, (0.0, 1.0, 0.0)), (19.5, -1.5, 1.5), -math.pi / 2),
    (Marker(19, "anchor", "1", -10.0, 0.0, 1.5, (1.0, 0.0, 0.0)), (-8.3, -0.6, 1.5), math.pi),
]


def camera_model(i: int, pos, yaw: float, save_dir: Path) -> str:
    return f"""    <model name="cam{i}"><static>true</static><pose>{pos[0]} {pos[1]} {pos[2]} 0 0 {yaw}</pose>
      <link name="link"><sensor name="c" type="camera">
        <camera><horizontal_fov>{HFOV}</horizontal_fov><image><width>{W}</width><height>{H}</height></image>
          <clip><near>0.05</near><far>30</far></clip>
          <save enabled="true"><path>{(save_dir / f"cam{i}").as_posix()}</path></save></camera>
        <always_on>1</always_on><update_rate>5</update_rate><topic>/check/cam{i}</topic>
      </sensor></link></model>
"""


def world(tex_dir: Path, save_dir: Path) -> str:
    mm = MarkerMap("DICT_4X4_50", 0.20, [c[0] for c in CASES])
    cams = "".join(camera_model(i, pos, yaw, save_dir) for i, (_, pos, yaw) in enumerate(CASES))
    return f"""<?xml version="1.0"?>
<sdf version="1.9"><world name="check">
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <scene><ambient>0.8 0.8 0.8 1</ambient><background>0.5 0.5 0.5 1</background></scene>
    <light type="directional" name="sun"><pose>0 0 20 0 0 0</pose><diffuse>1 1 1 1</diffuse><direction>-0.3 0.2 -0.9</direction></light>
{marker_models(mm, tex_dir)}{cams}</world></sdf>
"""


def main() -> int:
    import cv2
    tmp = Path(tempfile.mkdtemp(prefix="nav_tex_"))
    tex_dir, save_dir = tmp / "markers", tmp / "frames"
    tex_dir.mkdir()
    for m, _, _ in CASES:
        cv2.imwrite(str(tex_dir / f"aruco_{m.id}.png"), render_marker(m.id, 600, "DICT_4X4_50", BORDER))
    (tmp / "check.sdf").write_text(world(tex_dir, save_dir), encoding="utf-8")
    print(f"мир {tmp / 'check.sdf'}; Gazebo без окна, до 60 с...")
    # Кадр рендерится (и сохраняется <save>) только при подписчике на топик камеры — поэтому `gz topic -e`.
    # Не --iterations: сервер выходит раньше, чем поток рендера успевает сохранить кадр (gz-sim 8.15, WSL).
    quiet = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    server = subprocess.Popen(["gz", "sim", "-s", "-r", "--headless-rendering", str(tmp / "check.sdf")], **quiet)
    subs = [subprocess.Popen(["gz", "topic", "-e", "-t", f"/check/cam{i}"], **quiet) for i in range(len(CASES))]
    try:
        for _ in range(60):
            time.sleep(1)
            if all(len(list((save_dir / f"cam{i}").glob("*.png"))) >= 3 for i in range(len(CASES))):
                break
    finally:
        for p in subs + [server]:
            p.terminate()
        subprocess.run(["pkill", "-f", str(tmp / "check.sdf")], check=False)   # gz sim порождает дочерний сервер
    fx = W / (2 * math.tan(HFOV / 2))
    cam = Camera.from_intrinsics(W, H, fx, fx, W / 2, H / 2)
    detect = make_detector("DICT_4X4_50")
    ok = True
    for i, (m, pos, yaw) in enumerate(CASES):
        frames = sorted((save_dir / f"cam{i}").glob("*.png"), key=lambda f: f.stat().st_mtime)
        if not frames:
            print(f"cam{i}: нет кадров (Gazebo не отрендерил: проверьте `gz sim` и ogre2)")
            ok = False
            continue
        grays = [g for g in (cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in frames[-3:]) if g is not None]
        gray = grays[-2] if len(grays) > 1 else grays[-1]       # последний кадр мог остаться недописанным
        dets = [d for d in detect(gray) if d[0] == m.id]
        if not dets:
            other = [d[0] for d in detect(gray)]
            print(f"cam{i}: метка {m.id} не найдена (найдены {other}) — текстура зеркальна или не загрузилась; "
                  f"кадр {frames[-1]}")
            ok = False
            continue
        r_cm, t_cm, rms = solve_marker(dets[0][1], cam, 0.20)
        t_wm = m.pose()
        r_wc = t_wm[:3, :3] @ r_cm.T
        p_cam = t_wm[:3, 3] - r_wc @ t_cm
        r_true = np.array(euler_zyx(yaw)) @ np.array(R_LINK_OPTICAL)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r_true.T @ r_wc) - 1) / 2))))
        err = float(np.linalg.norm(p_cam - np.array(pos)))
        d_true = float(np.linalg.norm(np.array(pos) - t_wm[:3, 3]))
        ray = (t_wm[:3, 3] - np.array(pos)) / d_true
        along = float((p_cam - np.array(pos)) @ ray)
        good = err < 0.08 and ang < 5.0
        ok &= good
        print(f"cam{i}: метка {m.id} нормаль {m.normal}, d={d_true:.2f} м: ошибка положения камеры {err * 100:.1f} см "
              f"(вдоль луча {along * 100:+.1f} см, дальность {np.linalg.norm(t_cm) / d_true:.4f}×), ориентации {ang:.1f}°, "
              f"репроекция {rms:.2f} px — {'OK' if good else 'РАСХОЖДЕНИЕ'}")
    print("OK: текстуры меток в Gazebo совпадают с кадром карты" if ok else "ОШИБКА — см. выше")

    # Смещение углов: дальность d' = d·w/(w − 2δ), w = f·S/d — сторона метки в пикселях, δ — сдвиг каждого угла
    # внутрь (размытие + порог). δ = (1 − d/d')·w/2. Сравнение методов уточнения углов OpenCV.
    print("\nметод уточнения: δ (px) по камерам; среднее")
    for refine in ("NONE", "SUBPIX", "CONTOUR", "APRILTAG"):
        det = make_detector("DICT_4X4_50", refine)
        deltas = []
        for i, (m, pos, _) in enumerate(CASES):
            frames = sorted((save_dir / f"cam{i}").glob("*.png"), key=lambda f: f.stat().st_mtime)
            grays = [g for g in (cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in frames[-3:]) if g is not None]
            dets = [d for d in det(grays[0]) if d[0] == m.id] if grays else []
            if not dets:
                continue
            _, t_cm, _ = solve_marker(dets[0][1], cam, 0.20)
            d_true = float(np.linalg.norm(np.array(pos) - m.pose()[:3, 3]))
            w = fx * 0.20 / d_true
            deltas.append((1 - d_true / float(np.linalg.norm(t_cm))) * w / 2)
        print(f"  {refine:9s}: " + " ".join(f"{d:+.2f}" for d in deltas) + (f";  {np.mean(deltas):+.2f}" if deltas else ""))
    if ok:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

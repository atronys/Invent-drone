"""Модель дрона x500_nav = PX4 x500_flow (оптический поток вниз + дальномер LW20, GPS выключен в airframe 4021)
+ три камеры меток: фронтальная (якоря на торцах рядов) и две боковые (путевые метки в проходе).

    python3 nav_markers/gazebo/make_drone.py -o nav_markers/runs/models [--width 800 --height 600 --hfov 90 --rate 15]

Создаёт <out>/x500_nav/: model.config, model.sdf, camera_{front,left,right}.yaml, mounts.json, bridge.yaml.
Шаблоны звена камеры, camera_info и плагин истинной позы — из sim/make_drone_model.py (импорт, файл не меняется).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nav_markers"))
import nav_core  # noqa: E402,F401
from invent_core.camera import Camera  # noqa: E402
from sim.make_drone_model import CAMERA_INFO, CAMERA_LINK, POSE_PUBLISHER, TRUE_POSE_TOPIC  # noqa: E402

MODEL = "x500_nav"


def mounts(z: float = 0.06) -> list[dict]:
    """Установка камер на base_link (FLU). Звено камеры Gazebo смотрит вдоль своей +x."""
    return [
        {"look": "front", "name": "nav_cam_front", "frame": "nav_cam_front_link", "topic": "/nav/camera/front/image_raw",
         "xyz": [0.14, 0.0, z], "rpy": [0.0, 0.0, 0.0], "camera_info": "camera_front.yaml"},
        {"look": "left", "name": "nav_cam_left", "frame": "nav_cam_left_link", "topic": "/nav/camera/left/image_raw",
         "xyz": [0.0, 0.09, z], "rpy": [0.0, 0.0, math.pi / 2], "camera_info": "camera_left.yaml"},
        {"look": "right", "name": "nav_cam_right", "frame": "nav_cam_right_link", "topic": "/nav/camera/right/image_raw",
         "xyz": [0.0, -0.09, z], "rpy": [0.0, 0.0, -math.pi / 2], "camera_info": "camera_right.yaml"},
    ]


def build(width: int, height: int, hfov_deg: float, rate: float) -> dict[str, str]:
    fx = width / (2 * math.tan(math.radians(hfov_deg) / 2))
    cam = Camera.from_intrinsics(width, height, fx, fx, width / 2, height / 2)
    ms = mounts()
    links = "".join(CAMERA_LINK.format(frame=m["frame"], name=m["name"], topic=m["topic"], x=m["xyz"][0],
                                       y=m["xyz"][1], z=m["xyz"][2], pitch=m["rpy"][1], yaw=m["rpy"][2],
                                       hfov=math.radians(hfov_deg), w=width, h=height, rate=rate) for m in ms)
    sdf = f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- Сгенерировано nav_markers/gazebo/make_drone.py. Камеры меток: {width}x{height}, HFOV {hfov_deg:.0f}°, {rate} Гц. -->
<sdf version='1.9'>
  <model name='{MODEL}'>
    <include merge='true'>
      <uri>x500_flow</uri>
    </include>
{links}{POSE_PUBLISHER}  </model>
</sdf>
"""
    config = f"""<?xml version="1.0"?>
<model>
  <name>{MODEL}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>PX4 x500_flow + три камеры ArUco-меток (INVENT_DRONE / nav_markers)</description>
</model>
"""
    bridge = ["# ros_gz_bridge: ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=bridge.yaml"]
    for ros_topic, gz_topic, ros_t, gz_t in [
            ("/clock", "/clock", "rosgraph_msgs/msg/Clock", "gz.msgs.Clock"),
            *[(m["topic"], m["topic"], "sensor_msgs/msg/Image", "gz.msgs.Image") for m in ms],
            ("/gz/true_pose", TRUE_POSE_TOPIC, "tf2_msgs/msg/TFMessage", "gz.msgs.Pose_V")]:
        bridge.append(f'- ros_topic_name: "{ros_topic}"\n  gz_topic_name: "{gz_topic}"\n  ros_type_name: "{ros_t}"\n'
                      f'  gz_type_name: "{gz_t}"\n  direction: GZ_TO_ROS')
    files = {"model.sdf": sdf, "model.config": config, "bridge.yaml": "\n".join(bridge) + "\n",
             "mounts.json": json.dumps({"base_frame": "base_link", "cameras": ms}, ensure_ascii=False, indent=1)}
    for m in ms:
        files[m["camera_info"]] = CAMERA_INFO.format(w=width, h=height, name=m["name"], fx=cam.fx, cx=cam.cx, cy=cam.cy)
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="nav_markers/runs/models")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--rate", type=float, default=5.0, help="частота кадров камер меток, Гц")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    out = Path(a.out) / MODEL
    out.mkdir(parents=True, exist_ok=True)
    for name, text in build(a.width, a.height, a.hfov, a.rate).items():
        (out / name).write_text(text, encoding="utf-8")
    fx = a.width / (2 * math.tan(math.radians(a.hfov) / 2))
    print(f"{out}: три камеры {a.width}x{a.height}, HFOV {a.hfov:.0f}°, f = {fx:.0f} px, {a.rate} Гц; "
          f"метка 20 см на 3 м = {fx * 0.2 / 3:.0f} px")


if __name__ == "__main__":
    main()

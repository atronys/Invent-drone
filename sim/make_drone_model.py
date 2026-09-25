"""SIM-03: модель дрона x500_invent для Gazebo = PX4 x500 + фронтальная OakD-Lite (depth) + две боковые
камеры этикеток. Параметры камер берутся из той же `Camera`, что и в планировщике, — оптика в симуляции
совпадает с расчётной.

    python sim/make_drone_model.py -o sim/models --params core/examples/scan_params.json [--rate 2]

Создаёт sim/models/x500_invent/:
  model.config, model.sdf   — модель (по образцу PX4 x500_depth: include merge='true' + фиксированные шарниры)
  camera_left.yaml, camera_right.yaml — калибровка в формате ROS camera_info (идеальный пинхол Gazebo)
  mounts.json               — установка камер на base_link (для статического TF, узел odom_tf)
  bridge.yaml               — конфиг ros_gz_bridge: изображения, depth, /clock

Камера Gazebo смотрит вдоль +x своего звена; звено левой камеры повёрнуто на +90° по рысканию
(смотрит в +y тела), правой — на −90°; tilt_down — наклон вниз (тангаж звена).
Запуск в PX4 — sim/README.md.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from invent_core.camera import Camera  # noqa: E402
from invent_core.coverage import ScanParams  # noqa: E402

MODEL = "x500_invent"

CAMERA_LINK = """    <link name="{frame}">
      <pose relative_to="base_link">{x} {y} {z} 0 {pitch:.6f} {yaw:.6f}</pose>
      <inertial><mass>0.03</mass><inertia><ixx>1e-5</ixx><ixy>0</ixy><ixz>0</ixz><iyy>1e-5</iyy><iyz>0</iyz><izz>1e-5</izz></inertia></inertial>
      <visual name="{frame}_visual"><geometry><box><size>0.03 0.04 0.03</size></box></geometry>
        <material><diffuse>0.1 0.1 0.1 1</diffuse></material></visual>
      <sensor name="{name}" type="camera">
        <gz_frame_id>{frame}</gz_frame_id>
        <camera>
          <horizontal_fov>{hfov:.6f}</horizontal_fov>
          <image><width>{w}</width><height>{h}</height></image>
          <clip><near>0.1</near><far>15</far></clip>
        </camera>
        <always_on>1</always_on>
        <update_rate>{rate}</update_rate>
        <visualize>false</visualize>
        <topic>{topic}</topic>
      </sensor>
    </link>
    <joint name="{frame}_joint" type="fixed">
      <parent>base_link</parent>
      <child>{frame}</child>
    </joint>
"""

CAMERA_INFO = """image_width: {w}
image_height: {h}
camera_name: {name}
camera_matrix:
  rows: 3
  cols: 3
  data: [{fx:.6f}, 0, {cx:.3f}, 0, {fx:.6f}, {cy:.3f}, 0, 0, 1]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [0, 0, 0, 0, 0]
"""


def gazebo_camera(cam: Camera) -> Camera:
    """Gazebo: квадратные пиксели, главная точка в центре, VFOV выводится из HFOV и пропорций кадра."""
    fx = cam.width / (2 * math.tan(math.radians(cam.hfov_deg) / 2))
    return Camera.from_intrinsics(cam.width, cam.height, fx, fx, cam.width / 2, cam.height / 2)


def mounts(offset_y: float = 0.08, z: float = 0.12, tilt_down_deg: float = 0.0) -> list[dict]:
    pitch = math.radians(tilt_down_deg)
    return [
        {"look": "left", "name": "label_cam_left", "frame": "label_cam_left_link", "topic": "/camera/left/image_raw",
         "xyz": [0.0, offset_y, z], "rpy": [0.0, pitch, math.pi / 2], "camera_info": "camera_left.yaml"},
        {"look": "right", "name": "label_cam_right", "frame": "label_cam_right_link", "topic": "/camera/right/image_raw",
         "xyz": [0.0, -offset_y, z], "rpy": [0.0, pitch, -math.pi / 2], "camera_info": "camera_right.yaml"},
    ]


def build(cam: Camera, rate: float = 2.0, tilt_down_deg: float = 0.0) -> dict[str, str]:
    """-> {имя файла: содержимое}."""
    g = gazebo_camera(cam)
    ms = mounts(tilt_down_deg=tilt_down_deg)
    links = "".join(CAMERA_LINK.format(frame=m["frame"], name=m["name"], topic=m["topic"], x=m["xyz"][0], y=m["xyz"][1],
                                       z=m["xyz"][2], pitch=m["rpy"][1], yaw=m["rpy"][2], hfov=math.radians(g.hfov_deg),
                                       w=g.width, h=g.height, rate=rate) for m in ms)
    sdf = f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- Сгенерировано sim/make_drone_model.py. Камеры этикеток: {g.width}x{g.height}, HFOV {g.hfov_deg:.1f}°. -->
<sdf version='1.9'>
  <model name='{MODEL}'>
    <include merge='true'>
      <uri>x500</uri>
    </include>
    <include merge='true'>
      <uri>model://OakD-Lite</uri>
      <pose>.12 .03 .242 0 0 0</pose>
    </include>
    <joint name="CameraJoint" type="fixed">
      <parent>base_link</parent>
      <child>camera_link</child>
      <pose relative_to="base_link">.12 .03 .242 0 0 0</pose>
    </joint>
{links}  </model>
</sdf>
"""
    config = f"""<?xml version="1.0"?>
<model>
  <name>{MODEL}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>PX4 x500 + OakD-Lite (depth) + две боковые камеры этикеток (INVENT_DRONE)</description>
</model>
"""
    bridge = ["# ros_gz_bridge: ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=bridge.yaml"]
    for topic, ros_t, gz_t in [("/clock", "rosgraph_msgs/msg/Clock", "gz.msgs.Clock"),
                               *[(m["topic"], "sensor_msgs/msg/Image", "gz.msgs.Image") for m in ms],
                               ("/depth_camera", "sensor_msgs/msg/Image", "gz.msgs.Image")]:
        bridge.append(f'- ros_topic_name: "{topic}"\n  gz_topic_name: "{topic}"\n  ros_type_name: "{ros_t}"\n'
                      f'  gz_type_name: "{gz_t}"\n  direction: GZ_TO_ROS')
    files = {"model.sdf": sdf, "model.config": config, "bridge.yaml": "\n".join(bridge) + "\n",
             "mounts.json": json.dumps({"base_frame": "base_link", "cameras": ms}, ensure_ascii=False, indent=1)}
    for m in ms:
        files[m["camera_info"]] = CAMERA_INFO.format(w=g.width, h=g.height, name=m["name"], fx=g.fx, cx=g.cx, cy=g.cy)
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="sim/models")
    ap.add_argument("--params", help="JSON ScanParams (камера); иначе — Camera() по умолчанию")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--hfov", type=float)
    ap.add_argument("--rate", type=float, default=2.0, help="частота кадров камер этикеток, Гц")
    ap.add_argument("--tilt-down", type=float, default=0.0, help="наклон боковых камер вниз, град")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cam = ScanParams.load(a.params).camera if a.params else Camera()
    if a.width or a.height or a.hfov:
        cam = Camera(a.width or cam.width, a.height or cam.height, a.hfov or cam.hfov_deg, cam.vfov_deg)
    out = Path(a.out) / MODEL
    out.mkdir(parents=True, exist_ok=True)
    for name, text in build(cam, a.rate, a.tilt_down).items():
        (out / name).write_text(text, encoding="utf-8")
    g = gazebo_camera(cam)
    print(f"{out}: камеры {g.width}x{g.height}, HFOV {g.hfov_deg:.1f}° (VFOV {g.vfov_deg:.1f}°), {a.rate} Гц")
    print(f"скопируйте каталог в $PX4_DIR/Tools/simulation/gz/models/ (см. sim/README.md)")


if __name__ == "__main__":
    main()

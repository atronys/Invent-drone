"""Сквозной запуск в симуляции (PX4 SITL + Gazebo с моделью x500_invent уже запущены, см. sim/README.md).

    ros2 launch invent_bringup sim_mission.launch.py \
        layout:=$HOME/INVENT_DRONE/core/examples/warehouse_demo.json \
        scan_params:=$HOME/INVENT_DRONE/core/examples/scan_params.json

Запускает: MicroXRCEAgent, мост ros_gz (камеры, depth, /clock), odom_tf (TF дрона и камер),
safety_monitor (depth + геозона -> GO/SLOW/STOP), mission_node, два label_reader, scan_recorder.
Восприятие и TF работают на времени симуляции (/clock), чтобы метки времени кадров и TF совпадали.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # строки явно типизируем: иначе launch парсит значение как YAML и пустая строка становится None
    text = lambda name: ParameterValue(LaunchConfiguration(name), value_type=str)  # noqa: E731
    arg = lambda name, default, doc: DeclareLaunchArgument(name, default_value=default, description=doc)  # noqa: E731
    layout = text("layout")
    model_dir = LaunchConfiguration("model_dir")
    in_model = lambda name: ParameterValue(PathJoinSubstitution([model_dir, name]), value_type=str)  # noqa: E731
    sim_time = {"use_sim_time": True}

    readers = [
        Node(package="invent_perception", executable="label_reader", name=f"label_reader_{side}", output="screen",
             remappings=[("image", f"/camera/{side}/image_raw")],
             parameters=[{"layout": layout, "camera_info": in_model(f"camera_{side}.yaml"),
                          "frame_is_optical": False, "save_frames_dir": text("save_frames_dir")}, sim_time])
        for side in ("left", "right")
    ]
    return LaunchDescription([
        DeclareLaunchArgument("layout", description="JSON-раскладка склада (invent_core)"),
        arg("scan_params", "", "JSON ScanParams; пусто — по умолчанию"),
        arg("sortie", "0", "номер вылета в плане"),
        arg("series", "false", "FLT-03: после посадки — следующий вылет плана"),
        arg("continue_after_s", "-1.0", "пауза «замена АКБ», с; < 0 — ждать /mission/continue"),
        arg("model_dir", os.path.expanduser("~/INVENT_DRONE/sim/models/x500_invent"),
            "каталог модели от sim/make_drone_model.py (bridge.yaml, mounts.json, camera_*.yaml)"),
        arg("agent", "true", "запускать MicroXRCEAgent"),
        arg("save_frames_dir", "", "отладка чтения: сохранять кадры камер этикеток в каталог"),
        arg("localization", "false", "SIM-05/LOC-11: фальшивый VIO + локализация по этикеткам -> поза в PX4 (GPS выключить)"),
        ExecuteProcess(cmd=["MicroXRCEAgent", "udp4", "-p", "8888"], output="screen",
                       condition=IfCondition(LaunchConfiguration("agent"))),
        Node(package="ros_gz_bridge", executable="parameter_bridge", name="gz_bridge", output="screen",
             parameters=[{"config_file": in_model("bridge.yaml")}]),
        Node(package="invent_mission", executable="odom_tf", output="screen",
             parameters=[{"layout": layout, "mounts_file": in_model("mounts.json")}, sim_time]),
        Node(package="invent_mission", executable="safety_monitor", output="screen",
             remappings=[("depth", "/depth_camera")], parameters=[{"layout": layout}, sim_time]),
        Node(package="invent_mission", executable="mission_node", output="screen",
             parameters=[{"layout": layout, "scan_params": text("scan_params"),
                          "sortie": ParameterValue(LaunchConfiguration("sortie"), value_type=int),
                          "series": ParameterValue(LaunchConfiguration("series"), value_type=bool),
                          "continue_after_s": ParameterValue(LaunchConfiguration("continue_after_s"),
                                                             value_type=float)},
                         # время симуляции: в WSL она идёт ~0.3× реального, и на системных часах «зависание 1 с»
                         # длилось 0.3 с симуляции, а уставка ползла в 3 раза быстрее v_scan (SITL 12.10.2026)
                         sim_time]),
        *readers,
        Node(package="invent_localization", executable="fake_vio", output="screen", parameters=[sim_time],
             condition=IfCondition(LaunchConfiguration("localization"))),
        Node(package="invent_localization", executable="landmark_localizer", output="screen",
             condition=IfCondition(LaunchConfiguration("localization")),
             parameters=[{"layout": layout, "mounts_file": in_model("mounts.json"),
                          "camera_info_left": in_model("camera_left.yaml"),
                          "camera_info_right": in_model("camera_right.yaml"), "forward_to_px4": True}, sim_time]),
        Node(package="invent_perception", executable="scan_recorder", output="screen"),
    ])

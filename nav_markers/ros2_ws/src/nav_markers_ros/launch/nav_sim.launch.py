"""Навигация по меткам в SITL (PX4 + Gazebo с моделью x500_nav и миром warehouse_nav уже запущены).

    ros2 launch nav_markers_ros nav_sim.launch.py layout:=.../warehouse_demo.json \
        marker_map:=.../marker_map.json model_dir:=.../models/x500_nav targets:="B-05-2-2,C-03-1-1" out_dir:=...

Запускает: MicroXRCEAgent, мост ros_gz (камеры меток, /clock, истинная поза), marker_localizer, nav_mission,
nav_logger. Все узлы — на времени симуляции. PYTHONPATH должен содержать nav_markers, core и OpenCV ≥ 4.7
(это делает nav_markers/gazebo/run_nav_sitl.sh).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    text = lambda name: ParameterValue(LaunchConfiguration(name), value_type=str)  # noqa: E731
    sim_time = {"use_sim_time": True}
    return LaunchDescription([
        DeclareLaunchArgument("layout"),
        DeclareLaunchArgument("marker_map"),
        DeclareLaunchArgument("model_dir"),
        DeclareLaunchArgument("targets", default_value=""),
        DeclareLaunchArgument("out_dir", default_value="."),
        DeclareLaunchArgument("agent", default_value="true"),
        DeclareLaunchArgument("v_aisle", default_value="1.5"),
        ExecuteProcess(cmd=["MicroXRCEAgent", "udp4", "-p", "8888"], output="screen",
                       condition=IfCondition(LaunchConfiguration("agent"))),
        Node(package="ros_gz_bridge", executable="parameter_bridge", name="gz_bridge", output="screen",
             parameters=[{"config_file": ParameterValue(PathJoinSubstitution([LaunchConfiguration("model_dir"),
                                                                              "bridge.yaml"]), value_type=str)}]),
        Node(package="nav_markers_ros", executable="marker_localizer", output="screen",
             # corner_bias_px: сдвиг углов в рендере Gazebo, откалиброван gazebo/check_marker_texture.py (SUBPIX)
             parameters=[{"layout": text("layout"), "marker_map": text("marker_map"), "model_dir": text("model_dir"),
                          "corner_bias_px": 0.48},
                         sim_time]),
        Node(package="nav_markers_ros", executable="nav_mission", output="screen",
             parameters=[{"layout": text("layout"), "targets": text("targets"),
                          "v_aisle": ParameterValue(LaunchConfiguration("v_aisle"), value_type=float)}, sim_time]),
        Node(package="nav_markers_ros", executable="nav_logger", output="screen",
             parameters=[{"layout": text("layout"), "out_dir": text("out_dir")}, sim_time]),
    ])

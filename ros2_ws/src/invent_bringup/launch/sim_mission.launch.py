"""Сквозной запуск в симуляции (PX4 SITL + Gazebo уже запущены, см. sim/README.md).

    ros2 launch invent_bringup sim_mission.launch.py \
        layout:=$HOME/INVENT_DRONE/core/examples/warehouse_demo.json \
        scan_params:=$HOME/INVENT_DRONE/core/examples/scan_params.json
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # строки явно типизируем: иначе launch парсит значение как YAML и пустая строка становится None
    text = lambda name: ParameterValue(LaunchConfiguration(name), value_type=str)  # noqa: E731
    layout = text("layout")
    arg = lambda name, default, doc: DeclareLaunchArgument(name, default_value=default, description=doc)  # noqa: E731
    return LaunchDescription([
        DeclareLaunchArgument("layout", description="JSON-раскладка склада (invent_core)"),
        arg("scan_params", "", "JSON ScanParams; пусто — по умолчанию"),
        arg("sortie", "0", "номер вылета в плане"),
        arg("image", "/camera/left/image_raw", "топик боковой камеры"),
        arg("camera_info", "", "YAML калибровки боковой камеры"),
        arg("frame_is_optical", "false", "кадр изображения оптический? (камеры Gazebo — нет)"),
        arg("agent", "true", "запускать MicroXRCEAgent"),
        ExecuteProcess(cmd=["MicroXRCEAgent", "udp4", "-p", "8888"], output="screen",
                       condition=IfCondition(LaunchConfiguration("agent"))),
        Node(
            package="invent_mission", executable="mission_node", output="screen",
            parameters=[{"layout": layout, "scan_params": text("scan_params"),
                         "sortie": ParameterValue(LaunchConfiguration("sortie"), value_type=int)}],
        ),
        Node(
            package="invent_perception", executable="label_reader", output="screen",
            remappings=[("image", LaunchConfiguration("image"))],
            parameters=[{"layout": layout, "camera_info": text("camera_info"),
                         "frame_is_optical": ParameterValue(LaunchConfiguration("frame_is_optical"), value_type=bool)}],
        ),
        Node(package="invent_perception", executable="scan_recorder", output="screen"),
    ])

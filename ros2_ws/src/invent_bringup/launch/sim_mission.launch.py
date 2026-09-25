"""Сквозной запуск в симуляции (PX4 SITL + Gazebo уже запущены, см. sim/README.md).

    ros2 launch invent_bringup sim_mission.launch.py layout:=$HOME/INVENT_DRONE/core/examples/warehouse_demo.json
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    layout = LaunchConfiguration("layout")
    sortie = LaunchConfiguration("sortie")
    image = LaunchConfiguration("image")
    agent = LaunchConfiguration("agent")
    return LaunchDescription([
        DeclareLaunchArgument("layout", description="JSON-раскладка склада (invent_core)"),
        DeclareLaunchArgument("sortie", default_value="0", description="номер вылета в плане"),
        DeclareLaunchArgument("image", default_value="/camera/left/image_raw", description="топик боковой камеры"),
        DeclareLaunchArgument("agent", default_value="true", description="запускать MicroXRCEAgent"),
        ExecuteProcess(cmd=["MicroXRCEAgent", "udp4", "-p", "8888"], output="screen", condition=IfCondition(agent)),
        Node(
            package="invent_mission",
            executable="mission_node",
            parameters=[{"layout": layout, "sortie": ParameterValue(sortie, value_type=int)}],
            output="screen",
        ),
        Node(
            package="invent_perception",
            executable="label_reader",
            remappings=[("image", image)],
            output="screen",
        ),
    ])

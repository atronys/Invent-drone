from setuptools import setup

package_name = "invent_mission"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TODO",
    maintainer_email="todo@example.com",
    description="Миссия облёта склада: план invent_core -> PX4 Offboard",
    license="Proprietary",
    entry_points={"console_scripts": [
        "mission_node = invent_mission.mission_node:main",
        "odom_tf = invent_mission.odom_tf_node:main",
        "safety_monitor = invent_mission.safety_node:main",
    ]},
)

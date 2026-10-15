from glob import glob

from setuptools import setup

package_name = "nav_markers_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TODO",
    maintainer_email="todo@example.com",
    description="Навигация БПЛА на складе по ArUco-меткам",
    license="Proprietary",
    entry_points={"console_scripts": [
        "marker_localizer = nav_markers_ros.marker_localizer_node:main",
        "nav_mission = nav_markers_ros.nav_mission_node:main",
        "nav_logger = nav_markers_ros.nav_logger_node:main",
    ]},
)

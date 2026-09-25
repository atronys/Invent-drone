from setuptools import setup

package_name = "invent_perception"

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
    description="Чтение складских этикеток (zxing-cpp) и привязка к ячейкам",
    license="Proprietary",
    entry_points={"console_scripts": [
        "label_reader = invent_perception.label_reader_node:main",
        "scan_recorder = invent_perception.scan_recorder_node:main",
    ]},
)

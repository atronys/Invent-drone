from setuptools import setup

package_name = "invent_localization"

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
    description="LOC-11: EKF-локализация по адресным этикеткам поверх VIO",
    license="Proprietary",
    entry_points={"console_scripts": ["landmark_localizer = invent_localization.landmark_localizer_node:main"]},
)

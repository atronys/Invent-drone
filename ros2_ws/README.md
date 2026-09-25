# ROS 2 workspace (Jazzy)

| Пакет | Что делает | Статус |
|---|---|---|
| `invent_msgs` | `LabelRead`, `MissionStatus` | готов |
| `invent_mission` | `mission_node`: план `invent_core.coverage` -> PX4 Offboard (uXRCE-DDS), остановка/замедление по `safety/state`; `odom_tf`: TF дрона и камер; `safety_monitor`: depth + геозона | код готов, в SITL не запускался |
| `invent_perception` | `label_reader`: кадр -> zxing-cpp -> `LabelRead` (+ SSCC, + адрес ячейки по TF); `scan_recorder`: CSV вылета для сверки | код готов, в SITL не запускался |
| `invent_localization` | `landmark_localizer`: EKF по адресным этикеткам поверх VIO -> `localization/odom`, TF map->odom, опционально поза в PX4 | код готов, в SITL не запускался |
| `invent_bringup` | `sim_mission.launch.py`: агент, мост ros_gz, odom_tf, safety_monitor, миссия, 2 × label_reader, scan_recorder | готов для SITL |
| `px4_msgs` | клонируется скриптом установки под версию PX4 | внешний |

Логика (планирование, геометрия, сверка) живёт в `core/invent_core` — узлы только обёртки.
Узлы импортируют её через `PYTHONPATH` (прописывается в `~/.bashrc` скриптом установки).

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

Версии: PX4 >= 1.16 добавляет к топику суффикс `_v<N>` для сообщений с ненулевой версией
(проверено по исходникам PX4 v1.17.0). Узел миссии по умолчанию слушает
`/fmu/out/vehicle_status_v1` и `/fmu/out/vehicle_local_position_v1`. Если позиция или статус не
приходят — `ros2 topic list | grep fmu` и параметры `vehicle_status_topic` / `local_position_topic`.
`px4_msgs` должен быть из ветки под версию PX4 (`release/1.17` для v1.17.x).

Следующее по backlog: «фальшивый VIO» из ground truth Gazebo (SIM-05), затем OpenVINS на записи (LOC-03).

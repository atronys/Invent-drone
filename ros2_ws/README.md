# ROS 2 workspace (Jazzy)

| Пакет | Что делает | Статус |
|---|---|---|
| `invent_msgs` | `LabelRead`, `MissionStatus` | готов |
| `invent_mission` | план `invent_core.coverage` -> PX4 Offboard (uXRCE-DDS): взлёт, маршрут с ограничением скорости и разворота, посадка | скелет, работает в SITL |
| `invent_perception` | кадр -> zxing-cpp -> `LabelRead` (+ SSCC) | скелет; привязка к адресу — PER-06 |
| `invent_bringup` | `sim_mission.launch.py` | готов для SITL |
| `px4_msgs` | клонируется скриптом установки под версию PX4 | внешний |

Логика (планирование, геометрия, сверка) живёт в `core/invent_core` — узлы только обёртки.
Узлы импортируют её через `PYTHONPATH` (прописывается в `~/.bashrc` скриптом установки).

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

Версии: PX4 >= 1.16 версионирует часть сообщений (например, `/fmu/out/vehicle_status_v1`).
Если узел миссии «не видит» статус — проверь `ros2 topic list` и задай параметр
`vehicle_status_topic`.

Следующие узлы по backlog: `safety_monitor` (SAFE-01), `landmark_localizer` (LOC-06/LOC-08),
агрегатор наблюдений вылета (PER-07).

# Симуляция: PX4 SITL + Gazebo Harmonic + мир склада

## 1. Сгенерировать мир из раскладки

```bash
python3 sim/generate_world.py core/examples/warehouse_demo.json -o sim/worlds/demo
# -> warehouse.sdf, truth.csv (истина), wms_expected.csv (WMS с внесёнными ошибками), labels/*.png
```

Параметры: `--occupancy 0.8` (заполненность), `--wms-errors 0.03` (ошибки учёта), `--module-mm 0.5`
(ширина модуля штрихкода — как на реальных SSCC-этикетках), `--seed`.

## 2. Запустить PX4 SITL в этом мире

```bash
cp sim/worlds/demo/warehouse.sdf $PX4_DIR/Tools/simulation/gz/worlds/warehouse.sdf
cd $PX4_DIR
PX4_GZ_WORLD=warehouse PX4_GZ_MODEL_POSE="-2,2.6,0.1,0,0,0" make px4_sitl gz_x500_depth
```

`PX4_GZ_MODEL_POSE` = док из раскладки (`dock` в JSON). Кадр Gazebo world = наш кадр `map`,
поэтому координаты плана и симуляции совпадают — отладка тривиальна.

## 3. Запустить миссию

```bash
# в новом терминале (окружение из ~/.bashrc, см. scripts/setup_ubuntu.sh)
ros2 launch invent_bringup sim_mission.launch.py layout:=$HOME/INVENT_DRONE/core/examples/warehouse_demo.json
```

Проверки: `ros2 topic list | grep fmu` (мост PX4 жив), `ros2 topic echo /mission/status`.

## Что дальше (backlog)

- SIM-03: модель дрона с двумя боковыми камерами + `ros_gz_bridge` изображений в ROS 2;
- SIM-04: метрика read rate по `truth.csv`;
- SIM-05: полёт **без GPS** (`EKF2_GPS_CTRL=0`) на внешней одометрии — сначала ground truth с шумом, потом VIO.

Пока GPS в SITL включён — это нормально только для самой первой отладки маршрута.

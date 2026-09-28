# Симуляция: PX4 SITL + Gazebo Harmonic + мир склада

## 1. Сгенерировать мир из раскладки

```bash
python3 sim/generate_world.py core/examples/warehouse_demo.json -o sim/worlds/demo
# -> warehouse.sdf, truth.csv (истина), wms_expected.csv (WMS с внесёнными ошибками),
#    address_labels.csv (адресные этикетки — карта ориентиров для LOC-08), labels/*.png
```

Параметры: `--occupancy 0.8` (заполненность), `--wms-errors 0.03` (ошибки учёта), `--module-mm 0.5`
(ширина модуля штрихкода — как на реальных SSCC-этикетках), `--seed`.
Сценарии (SIM-06): `--damaged 0.3` (доля повреждённых этикеток; при 0.3 читается ≈70% текстур),
`--rotate-max 8` (наклон этикеток, град), `--obstacle` (препятствие в первом проходе — для SAFE-01),
`--ambient 0.2` (тёмный склад).

## 2. Модель дрона и запуск PX4 SITL (SIM-03)

```bash
# модель x500_invent: PX4 x500 + OakD-Lite (depth) + две боковые камеры этикеток с оптикой из ScanParams
python3 sim/make_drone_model.py -o sim/models --params core/examples/scan_params.json
cp -r sim/models/x500_invent $PX4_DIR/Tools/simulation/gz/models/
cp sim/worlds/demo/warehouse.sdf $PX4_DIR/Tools/simulation/gz/worlds/warehouse.sdf

cd $PX4_DIR && make px4_sitl          # один раз — сборка
PX4_SYS_AUTOSTART=4001 PX4_SIM_MODEL=gz_x500_invent PX4_GZ_WORLD=warehouse PX4_GZ_MODEL_POSE="-2,2.6,0.1,0,0,0" ./build/px4_sitl_default/bin/px4
```

- Airframe 4001 (x500) задаёт `PX4_SIM_MODEL` только если переменная пуста, а `px4-rc.gzsim` берёт модель
  из `models/<имя без gz_>/model.sdf` — поэтому своя модель подключается без правки airframe
  (проверено по исходникам PX4 v1.17.0).
- `PX4_GZ_MODEL_POSE` = док из раскладки (`dock` в JSON). Кадр Gazebo world = наш `map`.
- 12 Мп × 2 камеры тяжелы для рендера: на слабой видеокарте `--width 2028 --height 1520 --hfov 25`
  (px/мм на фасаде почти как у 12 Мп с HFOV 50°) или `--rate 1`.

## 3. Запустить миссию

```bash
# в новом терминале (окружение из ~/.bashrc, см. scripts/setup_ubuntu.sh)
ros2 launch invent_bringup sim_mission.launch.py layout:=$HOME/INVENT_DRONE/core/examples/warehouse_demo.json     model_dir:=$HOME/INVENT_DRONE/sim/models/x500_invent
```

Launch поднимает мост ros_gz (камеры, depth, `/clock`), `odom_tf` (TF `map→base_link` из
`/fmu/out/vehicle_odometry` + статические TF камер), узел миссии, два `label_reader` и `scan_recorder`.
Проверка TF: `ros2 run tf2_ros tf2_echo map label_cam_left_link`.

Проверки: `ros2 topic list | grep fmu` (мост PX4 жив), `ros2 topic echo /mission/status`.
Узел `scan_recorder` по окончании вылета пишет `~/invent_runs/<время>_sortie<N>.csv`.

Весь склад серией вылетов (FLT-03): `series:=true continue_after_s:=10` — после посадки пауза 10 с
(«замена АКБ») и следующий вылет плана; без `continue_after_s` узел ждёт команды
`ros2 service call /mission/continue std_srvs/srv/Trigger`. Каждый вылет — свой CSV; сверка берёт все сразу:
`python3 -m invent_core reconcile wms.csv ~/invent_runs/<серия>_sortie*.csv -o report.csv`.

## 4. Метрики и сверка

```bash
python3 sim/eval_read_rate.py sim/worlds/demo/truth.csv ~/invent_runs/<файл>.csv   # read rate, атрибуция
python3 -m invent_core reconcile sim/worlds/demo/wms_expected.csv ~/invent_runs/<файл>.csv -o report.csv
```

## 5. Полёт без GPS (SIM-05, LOC-11)

```bash
# в консоли PX4 (pxh>) один раз: выключить GPS, включить внешнюю навигацию (позиция + курс), высота — от неё же
param set EKF2_GPS_CTRL 0
param set EKF2_EV_CTRL 11
param set EKF2_HGT_REF 3
param save
# запуск с «фальшивым VIO» (истина Gazebo + дрейф) и локализацией по адресным этикеткам -> /fmu/in/vehicle_visual_odometry
ros2 launch invent_bringup sim_mission.launch.py layout:=... model_dir:=... localization:=true
```

Проверено по исходникам PX4 v1.17.0 (`src/modules/ekf2/module.yaml`): биты `EKF2_EV_CTRL` — 0 горизонтальная
позиция, 1 вертикальная, 2 скорость, 3 курс (11 = 0+1+3); `EKF2_HGT_REF=3` — Vision, **требует перезапуска PX4**.
Имя и маску `EKF2_GPS_CTRL` сверить в `param show EKF2_GPS*`.

## Что дальше (backlog)

- SIM-05: проверить цепочку без GPS в SITL (раздел 5), затем заменить фальшивый VIO на OpenVINS (LOC-03).

Пока GPS в SITL включён — это нормально только для самой первой отладки маршрута.

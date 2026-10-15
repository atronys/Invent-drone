#!/usr/bin/env bash
# Навигация по ArUco-меткам в PX4 SITL + Gazebo: мир с метками, x500_nav (поток + дальномер + 3 камеры),
# EKF2 = IMU + поток + метки (внешнее зрение), задание «долететь до ячеек и захватить их» -> метрики.
#
#   bash nav_markers/gazebo/run_nav_sitl.sh
#   TARGETS="B-05-2-2,C-03-1-1" V_AISLE=1.0 bash nav_markers/gazebo/run_nav_sitl.sh
#   GUI=1 GPU=1 KEEP_OPEN=1 bash nav_markers/gazebo/run_nav_sitl.sh   # с окном Gazebo (WSLg), камера за дроном,
#                                                                    # окно не закрывается после задания (Ctrl+C)
#   STEP_BAYS=4 HEIGHTS="0.5 1.5 2.3" bash nav_markers/gazebo/run_nav_sitl.sh   # реже метки
#
# Нужно: то же, что для scripts/smoke_sitl.sh INVENT_DRONE (ROS 2 Jazzy, PX4 v1.17, Gazebo Harmonic,
# ros2_ws INVENT_DRONE собран — из него берётся px4_msgs) + OpenCV ≥ 4.7 для Python в ~/nav_pylib:
#   python3 -m pip install --target ~/nav_pylib --no-deps opencv-python-headless==4.10.0.84
# Результат: nav_markers/runs/<время>/ — px4.log, ros.log, track.csv, fixes.csv, events.csv, summary.txt, track.png.
set -uo pipefail

NAV="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$NAV/.." && pwd)"
PX4_DIR="${PX4_DIR:-$HOME/src/PX4-Autopilot}"
PYLIB="${PYLIB:-$HOME/nav_pylib}"
LAYOUT="${LAYOUT:-$ROOT/core/examples/warehouse_demo.json}"
TARGETS="${TARGETS:-B-05-2-2,C-03-1-1}"
HEIGHTS="${HEIGHTS:-0.5 1.5 2.3}"
STEP_BAYS="${STEP_BAYS:-2}"
V_AISLE="${V_AISLE:-1.5}"
CAM_RATE="${CAM_RATE:-5}"          # Гц, камеры меток (рендер в WSL дорогой: 3 камеры × 15 Гц -> 0.02 реального времени)
TIMEOUT="${TIMEOUT:-3600}"
# GPU=1 — рендер Gazebo на видеокарте через Mesa d3d12 (WSL2 + /dev/dxg); иначе программный llvmpipe
if [ -n "${GPU:-}" ]; then export GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME="${GPU_NAME:-NVIDIA}"; fi
OUT="$NAV/runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
log() { echo "[nav] $*" | tee -a "$OUT/summary.txt"; }

set +u; source /opt/ros/jazzy/setup.bash; source "$ROOT/ros2_ws/install/setup.bash"; set -u
export PYTHONPATH="$PYLIB:$NAV:$ROOT/core:$ROOT:${PYTHONPATH:-}"
CV=$(python3 -c "import cv2; print(cv2.__version__)")
python3 - <<EOF || { log "ОШИБКА: OpenCV $CV < 4.7 (в 4.6 детектор ArUco теряет метки). См. шапку скрипта"; exit 1; }
v = tuple(int(x) for x in "$CV".split(".")[:2]); raise SystemExit(0 if v >= (4, 7) else 1)
EOF
if ! grep -q "time_us != 0" "$PX4_DIR/src/modules/sensors/vehicle_optical_flow/RingBuffer.hpp"; then
  log "ОШИБКА: в PX4 не наложен патч потока — x500_flow зависает на первом сообщении потока (EKF2 не стартует)."
  log "  cd $PX4_DIR && git apply $NAV/patches/px4_optical_flow_ringbuffer.patch && make px4_sitl"
  exit 1
fi
log "OpenCV $CV, задание: $TARGETS, метки на высотах $HEIGHTS через $STEP_BAYS секц., v_aisle $V_AISLE м/с"

kill_sim() {
  pkill -INT -f "ros2 launch nav_markers_ros" 2>/dev/null; sleep 3
  pkill -f "$NAV/ros2_ws/install/" 2>/dev/null; pkill -f "ros_gz_bridge/parameter_bridge" 2>/dev/null
  pkill -f "ros2 launch nav_markers_ros" 2>/dev/null
  pkill -f "bin/px4" 2>/dev/null; pkill -f "gz sim" 2>/dev/null; pkill -f MicroXRCEAgent 2>/dev/null
  sleep 2
  # сервер Gazebo иногда не реагирует на SIGTERM и остаётся с тем же миром: окно Gazebo нового прогона берёт сцену
  # у старого сервера, и дрона в нём не видно (13.10.2026) — добить
  pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f "bin/px4" 2>/dev/null
  sleep 1
}
if pgrep -f "$NAV/ros2_ws/install/|$ROOT/ros2_ws/install/|bin/px4|gz sim" >/dev/null; then
  log "остались процессы прошлого прогона — останавливаю"; kill_sim
fi
trap 'log "остановка процессов"; kill_sim' EXIT

# 1. Мир, модель, сборка пакета
cp "$LAYOUT" "$OUT/layout.json"
python3 "$NAV/gazebo/make_world.py" "$OUT/layout.json" -o "$OUT/world" --heights $HEIGHTS --step-bays "$STEP_BAYS" \
  | tee -a "$OUT/summary.txt" || exit 1
cp "$OUT/world/marker_map.json" "$OUT/"
python3 "$NAV/gazebo/make_drone.py" -o "$OUT/models" --rate "$CAM_RATE" | tee -a "$OUT/summary.txt" || exit 1
cp -r "$OUT/models/x500_nav" "$PX4_DIR/Tools/simulation/gz/models/"
cp "$OUT/world/warehouse_nav.sdf" "$PX4_DIR/Tools/simulation/gz/worlds/warehouse_nav.sdf"
( cd "$NAV/ros2_ws" && colcon build --symlink-install > "$OUT/colcon.log" 2>&1 ) || { log "ОШИБКА сборки (colcon.log)"; exit 1; }
set +u; source "$NAV/ros2_ws/install/setup.bash"; set -u

# 2. PX4 SITL: airframe 4021 (x500_flow, GPS выключен), своя модель; EKF2 — до старта, через PX4_PARAM_*
#    (rcS v1.17 применяет их до запуска ekf2; EKF2_MAG_TYPE и EKF2_EV_DELAY иначе требуют перезагрузки)
DOCK=$(python3 -c "import json;d=json.load(open('$OUT/layout.json'))['dock'];print(f'{d[0]},{d[1]},0.1,0,0,0')")
EKF_ENV=(
  PX4_PARAM_EKF2_EV_CTRL=9        # внешнее зрение: горизонтальное положение (бит 0) + курс (бит 3)
  PX4_PARAM_EKF2_MAG_TYPE=5       # магнитометр не используем: в складе он врёт (металл стеллажей), курс — по меткам
  PX4_PARAM_SYS_HAS_MAG=0
  PX4_PARAM_EKF2_EV_DELAY=20      # мс: кадр -> поза; измерено в SITL (latency_ms в status.jsonl, прогон 11)
  PX4_PARAM_EKF2_EVP_NOISE=0.01   # нижние границы шума метки; основное — ковариация из узла
  PX4_PARAM_EKF2_EVA_NOISE=0.01
  PX4_PARAM_EKF2_OF_CTRL=1        # оптический поток вниз
  PX4_PARAM_EKF2_RNG_CTRL=1       # дальномер вниз (условно)
)
log "PX4 SITL: airframe 4021, модель x500_nav, мир warehouse_nav, док $DOCK"
# PX4 всегда без окна: сам PX4 открывает окно Gazebo ДО появления дрона в мире (px4-rc.gzsim v1.17), и окно,
# занятое загрузкой большой сцены склада, его пропускает — дрона нет ни в кадре, ни в дереве (13.10.2026).
# Окно открываем сами, когда дрон уже в мире.
( cd "$PX4_DIR" && env HEADLESS=1 "${EKF_ENV[@]}" PX4_SYS_AUTOSTART=4021 PX4_SIM_MODEL=gz_x500_nav \
    PX4_GZ_WORLD=warehouse_nav PX4_GZ_MODEL_POSE="$DOCK" ./build/px4_sitl_default/bin/px4 -d > "$OUT/px4.log" 2>&1 & )
if ! timeout 240 bash -c "until grep -q 'Startup script returned successfully' '$OUT/px4.log'; do sleep 2; done"; then
  log "ОШИБКА: PX4 не стартовал за 240 с (px4.log)"; exit 1
fi
log "PX4 запущен"
if [ -n "${GUI:-}" ]; then
  timeout 60 bash -c "until gz model --list 2>/dev/null | grep -q x500_nav_0; do sleep 2; done" \
    || log "ВНИМАНИЕ: дрон x500_nav_0 не появился в мире за 60 с"
  # окно — без принудительного d3d12 (GPU=1): с ним окно Gazebo падает в драйвере NVIDIA при создании
  # OpenGL-контекста (segfault в libnvwgf2umx.so, 13.10.2026), а серверу d3d12 не мешает
  ( set +u; source "$PX4_DIR/build/px4_sitl_default/rootfs/gz_env.sh"; unset GALLIUM_DRIVER MESA_D3D12_DEFAULT_ADAPTER_NAME
    gz sim -g > "$OUT/gui.log" 2>&1 & )
  log "окно Gazebo открыто (дрон уже в мире)"
fi
n_ok=0
while read -r name value _; do
  case "$name" in ''|\#*) continue ;; esac
  ( cd "$PX4_DIR/build/px4_sitl_default" && ./bin/px4-param set "$name" "$value" ) >> "$OUT/px4_params.log" 2>&1 \
    && n_ok=$((n_ok + 1))
done < "$ROOT/sim/px4_params_indoor.txt"
log "параметры полёта в помещении (sim/px4_params_indoor.txt INVENT_DRONE): задано $n_ok"
( cd "$PX4_DIR/build/px4_sitl_default" && for p in EKF2_EV_CTRL EKF2_MAG_TYPE EKF2_OF_CTRL EKF2_GPS_CTRL MPC_XY_VEL_MAX; do
    ./bin/px4-param show "$p"; done ) 2>/dev/null | grep -E "^\s*[x+ ]" | tee -a "$OUT/summary.txt"

# 3. Узлы ROS 2 и ожидание конца задания
ros2 launch nav_markers_ros nav_sim.launch.py layout:="$OUT/layout.json" marker_map:="$OUT/marker_map.json" \
  model_dir:="$OUT/models/x500_nav" targets:="$TARGETS" out_dir:="$OUT" v_aisle:="$V_AISLE" > "$OUT/ros.log" 2>&1 &
log "узлы запущены, жду конца задания (не дольше $TIMEOUT с)"
if [ -n "${GUI:-}" ]; then   # окно Gazebo открывается общим планом склада — привязать камеру к дрону (3 м сзади)
  ( for _ in $(seq 30); do
      gz service -s /gui/follow --reqtype gz.msgs.StringMsg --reptype gz.msgs.Boolean --timeout 2000 \
        --req 'data: "x500_nav_0"' 2>/dev/null | grep -q true && break
      sleep 2
    done
    gz service -s /gui/follow/offset --reqtype gz.msgs.Vector3d --reptype gz.msgs.Boolean --timeout 2000 \
      --req 'x: -3, y: 0, z: 1.5' >/dev/null 2>&1 ) &
fi
t0=$(date +%s)
while true; do
  sleep 20
  # в CSV кавычки JSON удвоены: ""event"": ""done""
  if grep -qE 'event"+: "+(done|lost_land|abort)' "$OUT/events.csv" 2>/dev/null; then break; fi
  if (( $(date +%s) - t0 > TIMEOUT )); then log "ТАЙМАУТ"; break; fi
  last=$(tail -1 "$OUT/events.csv" 2>/dev/null | cut -c1-160)
  rtf=$(timeout 5 gz topic -e -t /world/warehouse_nav/stats -n 1 2>/dev/null | grep -oP "real_time_factor: \K[0-9.]+" | cut -c1-5)
  echo "[nav] $(( $(date +%s) - t0 )) с (RTF ${rtf:-?}): ${last:-нет событий}"
done
sleep 3
log "конец: $(grep -oE 'event"+: "+[a-z_]+'  "$OUT/events.csv" | sort | uniq -c | tr -s ' ' | tr '\n' ',')"

# 4. Метрики
python3 "$NAV/gazebo/eval_nav.py" "$OUT" || log "eval_nav.py: ошибка (track.csv пуст? см. ros.log)"
log "результаты: $OUT"
if [ -n "${KEEP_OPEN:-}" ]; then   # оставить Gazebo открытым, чтобы посмотреть сцену; закрытие — Ctrl+C
  log "симуляция оставлена открытой (KEEP_OPEN), Ctrl+C — закрыть"
  while true; do sleep 3600; done
fi

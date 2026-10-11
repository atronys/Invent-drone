#!/usr/bin/env bash
# SIM-01/02 дымовой тест: маленький склад + дрон x500_invent + PX4 SITL без окна + наши узлы ROS 2
# -> вылет 0 -> логи, метрики чтения и сверка в одном каталоге.
#
#   bash scripts/smoke_sitl.sh               # после scripts/setup_ubuntu.sh, в новом терминале
#   GUI=1 bash scripts/smoke_sitl.sh         # с окном Gazebo (WSLg / обычный Linux)
#   TIMEOUT=1200 bash scripts/smoke_sitl.sh
#
# Результат: sim/runs/<время>/ — px4.log, ros.log, summary.txt (read rate, атрибуция, сверка).
# Если что-то упало — пришлите summary.txt и хвосты логов (tail -100 ros.log px4.log).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PX4_DIR="${PX4_DIR:-$HOME/src/PX4-Autopilot}"
TIMEOUT="${TIMEOUT:-900}"
OUT="$ROOT/sim/runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
set +u; source /opt/ros/jazzy/setup.bash; source "$ROOT/ros2_ws/install/setup.bash"; set -u
export PYTHONPATH="$ROOT/core:${PYTHONPATH:-}"
log() { echo "[smoke] $*" | tee -a "$OUT/summary.txt"; }

kill_sim() {
  # SIGINT для ros2 launch не всегда доходит до узлов: остаются старые mission_node/safety_monitor и
  # командуют PX4 следующего прогона (12.10.2026: узлы трёх прогонов сразу -> ABORT). Добиваем по пути установки.
  pkill -INT -f "ros2 launch invent_bringup" 2>/dev/null; sleep 3
  pkill -f "$ROOT/ros2_ws/install/" 2>/dev/null; pkill -f "ros_gz_bridge/parameter_bridge" 2>/dev/null
  pkill -f "ros2 launch invent_bringup" 2>/dev/null
  pkill -f "bin/px4" 2>/dev/null; pkill -f "gz sim" 2>/dev/null; pkill -f MicroXRCEAgent 2>/dev/null
  sleep 2
}
cleanup() {
  log "остановка процессов"
  kill_sim
}
if pgrep -f "$ROOT/ros2_ws/install/|bin/px4|gz sim" >/dev/null; then
  log "остались процессы прошлого прогона — останавливаю"
  kill_sim
fi
trap cleanup EXIT

# 1. Маленький склад (1 проход × 4 секции × 2 яруса) и лёгкие камеры: дымовой тест, а не эксперимент.
#    Модуль этикеток 0.8 мм и HFOV 20° при 2028 px: ~3 px на модуль на 1.5 м — читаемо и быстро рендерится.
python3 -m invent_core make-layout "$OUT/layout.json" --aisles 1 --bays 4 --levels 2 --name smoke >/dev/null
cat > "$OUT/scan_params.json" <<'EOF'
{"camera": {"width": 2028, "height": 1520, "hfov_deg": 20.0, "vfov_deg": 15.1}, "min_z": 0.5, "hold_s": 1.0}
EOF
python3 "$ROOT/sim/generate_world.py" "$OUT/layout.json" -o "$OUT/world" --module-mm 0.8 | tee -a "$OUT/summary.txt"
python3 "$ROOT/sim/make_drone_model.py" -o "$OUT/models" --params "$OUT/scan_params.json" --rate 2 | tee -a "$OUT/summary.txt"
cp -r "$OUT/models/x500_invent" "$PX4_DIR/Tools/simulation/gz/models/"
cp "$OUT/world/warehouse.sdf" "$PX4_DIR/Tools/simulation/gz/worlds/warehouse.sdf"
python3 -m invent_core plan "$OUT/layout.json" --params "$OUT/scan_params.json" | tee -a "$OUT/summary.txt"

# 2. PX4 SITL (демон, без интерактивной консоли); док — из раскладки
DOCK=$(python3 -c "import json;d=json.load(open('$OUT/layout.json'))['dock'];print(f'{d[0]},{d[1]},0.1,0,0,0')")
log "PX4 SITL: модель x500_invent, мир warehouse, док $DOCK"
HEADLESS_ENV="HEADLESS=1"
[ -n "${GUI:-}" ] && HEADLESS_ENV=""
( cd "$PX4_DIR" && env $HEADLESS_ENV PX4_SYS_AUTOSTART=4001 PX4_SIM_MODEL=gz_x500_invent \
    PX4_GZ_WORLD=warehouse PX4_GZ_MODEL_POSE="$DOCK" ./build/px4_sitl_default/bin/px4 -d > "$OUT/px4.log" 2>&1 & )
if ! timeout 180 bash -c "until grep -qiE 'Ready for takeoff|home set' '$OUT/px4.log'; do sleep 2; done"; then
  log "ОШИБКА: PX4 не готов за 180 с (см. px4.log)"; exit 1
fi
log "PX4 готов"

# 2a. SAFE-03: параметры для полётов в помещении (docs/safety_failsafe.md); PX4_PARAMS=none — заводские
PX4_PARAMS="${PX4_PARAMS:-$ROOT/sim/px4_params_indoor.txt}"
if [ "$PX4_PARAMS" != none ]; then
  n_ok=0; n_bad=0
  while read -r name value _; do
    case "$name" in ''|\#*) continue ;; esac
    if ( cd "$PX4_DIR/build/px4_sitl_default" && ./bin/px4-param set "$name" "$value" ) >> "$OUT/px4_params.log" 2>&1; then
      n_ok=$((n_ok + 1))
    else
      n_bad=$((n_bad + 1)); log "ВНИМАНИЕ: не удалось задать $name=$value"
    fi
  done < "$PX4_PARAMS"
  log "параметры PX4 из $(basename "$PX4_PARAMS"): задано $n_ok, ошибок $n_bad"
fi

# 3. Наши узлы
ros2 launch invent_bringup sim_mission.launch.py layout:="$OUT/layout.json" scan_params:="$OUT/scan_params.json" \
    model_dir:="$OUT/models/x500_invent" > "$OUT/ros.log" 2>&1 &
ROS_PID=$!
START=$(date +%s)
if timeout "$TIMEOUT" bash -c "until grep -qE 'посадка завершена|ABORT|вышли из Offboard|Traceback' '$OUT/ros.log'; do sleep 5; done"; then
  log "миссия: $(grep -oE 'посадка завершена|ABORT|вышли из Offboard[^\n]*|Traceback' "$OUT/ros.log" | head -1) за $(( $(date +%s) - START )) с"
else
  log "ОШИБКА: миссия не завершилась за $TIMEOUT с"
fi
sleep 5   # scan_recorder сохраняет CSV по DONE

# 4. Метрики и сверка
CSV=$(ls -t "$HOME/invent_runs/"*.csv 2>/dev/null | head -1)
if [ -n "$CSV" ] && [ "$(stat -c %Y "$CSV")" -ge "$START" ]; then
  cp "$CSV" "$OUT/observations.csv"
  python3 "$ROOT/sim/eval_read_rate.py" "$OUT/world/truth.csv" "$OUT/observations.csv" | tee -a "$OUT/summary.txt"
  python3 -m invent_core reconcile "$OUT/world/wms_expected.csv" "$OUT/observations.csv" -o "$OUT/report.csv" \
    | tee -a "$OUT/summary.txt"
else
  log "ОШИБКА: CSV вылета не найден (scan_recorder)"
fi
log "чтений этикеток в логе: $(grep -c 'label_reader' "$OUT/ros.log" || true); ошибок Python: $(grep -c Traceback "$OUT/ros.log" || true)"
log "итог: $OUT"

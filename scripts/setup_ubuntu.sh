#!/usr/bin/env bash
# Среда разработки INVENT_DRONE:
#   Ubuntu 24.04 + ROS 2 Jazzy + PX4 SITL (Gazebo Harmonic) + Micro XRCE-DDS Agent + px4_msgs.
# Запуск из корня проекта:  bash scripts/setup_ubuntu.sh
# Время ~1–2 ч, место ~25 ГБ, нужен sudo. Скрипт можно перезапускать: готовые шаги пропускаются.
set -eo pipefail

PX4_VERSION="${PX4_VERSION:-v1.17.0}"            # последний стабильный на 26.09.2026; px4_msgs берётся из release/1.17
XRCE_AGENT_VERSION="${XRCE_AGENT_VERSION:-v2.4.3}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${SRC:-$HOME/src}"
# Параллельная сборка: ~1 ГБ ОЗУ на компилятор C++ (в WSL по умолчанию памяти мало — не больше ОЗУ/1.2 ГБ)
MEM_GB=$(awk '/MemTotal/ {print int($2/1024/1024)}' /proc/meminfo)
JOBS="${JOBS:-$(( MEM_GB * 10 / 12 < $(nproc) ? MEM_GB * 10 / 12 : $(nproc) ))}"
[ "$JOBS" -ge 1 ] || JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" MAKEFLAGS="-j$JOBS"
echo "Сборка в $JOBS потоков (ОЗУ ${MEM_GB} ГБ)"

. /etc/os-release
if [ "$VERSION_ID" != "24.04" ]; then
  echo "Нужна Ubuntu 24.04 (обнаружена $VERSION_ID). ROS 2 Jazzy официально поддерживает только её."
  exit 1
fi

# WSL: проект на диске Windows (/mnt/c/...) собирать нельзя — colcon --symlink-install и сборка там
# медленные и ненадёжные. Клонируйте репозиторий в файловую систему Linux и запускайте оттуда:
#   git clone /mnt/c/INVENT_DRONE ~/INVENT_DRONE && cd ~/INVENT_DRONE && bash scripts/setup_ubuntu.sh
if grep -qi microsoft /proc/version 2>/dev/null && [[ "$PROJECT_DIR" == /mnt/* ]] && [ -z "$FORCE_MNT" ]; then
  echo "WSL: проект лежит на диске Windows ($PROJECT_DIR)."
  echo "Сделайте: git clone $PROJECT_DIR ~/INVENT_DRONE && cd ~/INVENT_DRONE && bash scripts/setup_ubuntu.sh"
  echo "(изменения с Windows забирайте в WSL командой git pull; принудительно — FORCE_MNT=1)"
  exit 1
fi

# --- Зеркала -------------------------------------------------------------------------------
# В РФ и в WSL при VPN на Windows официальные репозитории часто недоступны или рвутся (проверено 28.09.2026:
# packages.ros.org и pypi.org не отвечали, archive.ubuntu.com по HTTP — через раз). MIRRORS=auto (по умолчанию)
# проверяет доступность и при необходимости переключает: Ubuntu -> mirror.yandex.ru, ROS 2 -> mirrors.ustc.edu.cn,
# Gazebo -> HTTPS, pip -> mirrors.aliyun.com. MIRRORS=off — не трогать; MIRRORS=on — переключить сразу.
MIRRORS="${MIRRORS:-auto}"
UBUNTU_MIRROR="${UBUNTU_MIRROR:-https://mirror.yandex.ru/ubuntu/}"
ROS_MIRROR="${ROS_MIRROR:-https://mirrors.ustc.edu.cn/ros2/ubuntu}"
PIP_MIRROR="${PIP_MIRROR:-https://mirrors.aliyun.com/pypi/simple/}"
reachable() { curl -s -o /dev/null -m 10 -f "$1"; }
# Сеть может молча рвать соединения (WSL при VPN на Windows): таймауты и повторы для всего, что качается
CURL=(curl -fsSL --connect-timeout 15 --max-time 300 --retry 8 --retry-delay 5 --retry-all-errors)
GIT=(git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60)   # обрывать «зависшие» передачи
retry() { local n=0; until "$@"; do n=$((n + 1)); [ "$n" -ge 6 ] && return 1; echo "повтор $n: $*"; sleep 10; done; }
clone() {   # clone <ветка> <url> <каталог>: неудачную попытку удаляем и повторяем
  local n=0
  until "${GIT[@]}" clone -b "$1" "$2" "$3"; do
    rm -rf "$3"; n=$((n + 1)); [ "$n" -ge 6 ] && return 1; echo "повтор clone $n: $2"; sleep 10
  done
}
use_mirrors=false
if [ "$MIRRORS" = "on" ]; then use_mirrors=true
elif [ "$MIRRORS" = "auto" ]; then
  reachable "http://archive.ubuntu.com/ubuntu/dists/noble/Release" && reachable "https://pypi.org/simple/pip/" \
    && reachable "http://packages.ros.org/ros2/ubuntu/dists/noble/Release" || use_mirrors=true
fi
configure_ros_mirror() {   # вызывается после установки ros2-apt-source
  [ -f /etc/apt/sources.list.d/ros2.sources ] && ! grep -q "$ROS_MIRROR" /etc/apt/sources.list.d/ros2.sources && {
    sudo cp -n /etc/apt/sources.list.d/ros2.sources /etc/apt/sources.list.d/ros2.sources.bak
    sudo sed -i "s#^URIs:.*#URIs: $ROS_MIRROR#" /etc/apt/sources.list.d/ros2.sources; }
  # на зеркале нет исходных пакетов (deb-src -> 404 и падение apt update) — берём только бинарные
  [ -f /etc/apt/sources.list.d/ros2.sources ] && sudo sed -i "s#^Types:.*#Types: deb#" /etc/apt/sources.list.d/ros2.sources
  [ -f /etc/apt/sources.list.d/gazebo-stable.list ] && sudo sed -i "s#http://packages.osrfoundation.org#https://packages.osrfoundation.org#" /etc/apt/sources.list.d/gazebo-stable.list
  return 0
}
if $use_mirrors; then
  echo "== 0/5 Зеркала: Ubuntu $UBUNTU_MIRROR, ROS 2 $ROS_MIRROR, pip $PIP_MIRROR"
  echo 'Acquire::Retries "10";' | sudo tee /etc/apt/apt.conf.d/80-retries >/dev/null
  if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then
    sudo cp -n /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list.d/ubuntu.sources.bak
    sudo sed -i "s#^URIs:.*#URIs: $UBUNTU_MIRROR#" /etc/apt/sources.list.d/ubuntu.sources
  fi
  configure_ros_mirror
  mkdir -p ~/.config/pip
  printf '[global]\nindex-url = %s\ntimeout = 60\nretries = 10\n' "$PIP_MIRROR" > ~/.config/pip/pip.conf
fi

echo "== 1/5 ROS 2 Jazzy"
if [ ! -f /opt/ros/jazzy/setup.bash ]; then
  sudo apt update && sudo apt install -y software-properties-common curl git
  sudo add-apt-repository -y universe
  ROS_APT_SOURCE_VERSION=$("${CURL[@]}" https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
    | grep -F "tag_name" | awk -F\" '{print $4}') || true
  ROS_APT_SOURCE_VERSION="${ROS_APT_SOURCE_VERSION:-1.3.0}"   # последняя на 28.09.2026 — если API GitHub недоступен
  "${CURL[@]}" -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${VERSION_CODENAME}_all.deb"
  sudo dpkg -i /tmp/ros2-apt-source.deb
fi
$use_mirrors && configure_ros_mirror
sudo apt update
sudo apt install -y ros-jazzy-desktop ros-dev-tools ros-jazzy-ros-gz ros-jazzy-cv-bridge python3-pip

echo "== 2/5 PX4-Autopilot $PX4_VERSION (SITL + Gazebo Harmonic)"
mkdir -p "$SRC"
if [ ! -d "$SRC/PX4-Autopilot" ]; then
  clone "$PX4_VERSION" https://github.com/PX4/PX4-Autopilot.git "$SRC/PX4-Autopilot"
fi
# подмодули отдельно и с повторами: так обрыв посреди загрузки не заставляет качать всё заново
(cd "$SRC/PX4-Autopilot" && retry "${GIT[@]}" submodule update --init --recursive --jobs 4)
PX4_SETUP="$SRC/PX4-Autopilot/Tools/setup/ubuntu.sh"
if $use_mirrors; then
  # скрипт PX4 каждый раз заново пишет репозиторий Gazebo по HTTP (у нас недоступен) — запускаем копию с HTTPS;
  # копия лежит рядом с оригиналом, чтобы нашёлся его requirements.txt
  sed 's#http://packages.osrfoundation.org#https://packages.osrfoundation.org#g' "$PX4_SETUP" > "${PX4_SETUP%.sh}_https.sh"
  PX4_SETUP="${PX4_SETUP%.sh}_https.sh"
fi
bash "$PX4_SETUP" --no-nuttx
(cd "$SRC/PX4-Autopilot" && make px4_sitl)

echo "== 3/5 Micro XRCE-DDS Agent $XRCE_AGENT_VERSION"
if ! command -v MicroXRCEAgent >/dev/null; then
  [ -d "$SRC/Micro-XRCE-DDS-Agent" ] || \
    clone "$XRCE_AGENT_VERSION" https://github.com/eProsima/Micro-XRCE-DDS-Agent.git "$SRC/Micro-XRCE-DDS-Agent"
  cmake -S "$SRC/Micro-XRCE-DDS-Agent" -B "$SRC/Micro-XRCE-DDS-Agent/build"
  cmake --build "$SRC/Micro-XRCE-DDS-Agent/build" -j"$JOBS"
  sudo cmake --install "$SRC/Micro-XRCE-DDS-Agent/build"
  sudo ldconfig /usr/local/lib/
fi

echo "== 4/5 Рабочее пространство ROS 2"
# px4_msgs ДОЛЖЕН соответствовать версии PX4, иначе сообщения не совпадут побайтно.
PX4_MSGS_BRANCH="release/$(echo "$PX4_VERSION" | sed -E 's/^v([0-9]+\.[0-9]+).*/\1/')"
if [ ! -d "$PROJECT_DIR/ros2_ws/src/px4_msgs" ]; then
  clone "$PX4_MSGS_BRANCH" https://github.com/PX4/px4_msgs.git "$PROJECT_DIR/ros2_ws/src/px4_msgs" || {
    echo "ВНИМАНИЕ: ветки $PX4_MSGS_BRANCH нет — беру main и копирую msg из PX4 (сверь вручную)"
    git clone https://github.com/PX4/px4_msgs.git "$PROJECT_DIR/ros2_ws/src/px4_msgs"
    cp "$SRC/PX4-Autopilot/msg/"*.msg "$PROJECT_DIR/ros2_ws/src/px4_msgs/msg/" 2>/dev/null || true
  }
fi
# Ubuntu 24.04 запрещает pip в системный Python (PEP 668); пакеты чисто пользовательские, это безопасно.
sudo apt install -y python3-numpy python3-opencv python3-matplotlib
pip3 install --user --break-system-packages zxing-cpp python-barcode pillow
set +u; source /opt/ros/jazzy/setup.bash
(cd "$PROJECT_DIR/ros2_ws" && colcon build --symlink-install --parallel-workers "$JOBS")

echo "== 5/5 Переменные окружения (~/.bashrc)"
if ! grep -q "# INVENT_DRONE" ~/.bashrc; then
  cat >> ~/.bashrc <<EOF
# INVENT_DRONE
source /opt/ros/jazzy/setup.bash
[ -f $PROJECT_DIR/ros2_ws/install/setup.bash ] && source $PROJECT_DIR/ros2_ws/install/setup.bash
export PYTHONPATH=$PROJECT_DIR/core:\$PYTHONPATH   # invent_core без установки
export PX4_DIR=$SRC/PX4-Autopilot
EOF
fi
echo "Готово. Откройте новый терминал и пройдите проверку из sim/README.md"

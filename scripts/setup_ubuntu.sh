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

echo "== 1/5 ROS 2 Jazzy"
if [ ! -f /opt/ros/jazzy/setup.bash ]; then
  sudo apt update && sudo apt install -y software-properties-common curl git
  sudo add-apt-repository -y universe
  ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
  curl -L -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${VERSION_CODENAME}_all.deb"
  sudo dpkg -i /tmp/ros2-apt-source.deb
  sudo apt update
fi
sudo apt install -y ros-jazzy-desktop ros-dev-tools ros-jazzy-ros-gz ros-jazzy-cv-bridge python3-pip

echo "== 2/5 PX4-Autopilot $PX4_VERSION (SITL + Gazebo Harmonic)"
mkdir -p "$SRC"
if [ ! -d "$SRC/PX4-Autopilot" ]; then
  git clone --recursive -b "$PX4_VERSION" https://github.com/PX4/PX4-Autopilot.git "$SRC/PX4-Autopilot"
fi
bash "$SRC/PX4-Autopilot/Tools/setup/ubuntu.sh" --no-nuttx
(cd "$SRC/PX4-Autopilot" && make px4_sitl)

echo "== 3/5 Micro XRCE-DDS Agent $XRCE_AGENT_VERSION"
if ! command -v MicroXRCEAgent >/dev/null; then
  [ -d "$SRC/Micro-XRCE-DDS-Agent" ] || \
    git clone -b "$XRCE_AGENT_VERSION" https://github.com/eProsima/Micro-XRCE-DDS-Agent.git "$SRC/Micro-XRCE-DDS-Agent"
  cmake -S "$SRC/Micro-XRCE-DDS-Agent" -B "$SRC/Micro-XRCE-DDS-Agent/build"
  cmake --build "$SRC/Micro-XRCE-DDS-Agent/build" -j"$(nproc)"
  sudo cmake --install "$SRC/Micro-XRCE-DDS-Agent/build"
  sudo ldconfig /usr/local/lib/
fi

echo "== 4/5 Рабочее пространство ROS 2"
# px4_msgs ДОЛЖЕН соответствовать версии PX4, иначе сообщения не совпадут побайтно.
PX4_MSGS_BRANCH="release/$(echo "$PX4_VERSION" | sed -E 's/^v([0-9]+\.[0-9]+).*/\1/')"
if [ ! -d "$PROJECT_DIR/ros2_ws/src/px4_msgs" ]; then
  git clone -b "$PX4_MSGS_BRANCH" https://github.com/PX4/px4_msgs.git "$PROJECT_DIR/ros2_ws/src/px4_msgs" || {
    echo "ВНИМАНИЕ: ветки $PX4_MSGS_BRANCH нет — беру main и копирую msg из PX4 (сверь вручную)"
    git clone https://github.com/PX4/px4_msgs.git "$PROJECT_DIR/ros2_ws/src/px4_msgs"
    cp "$SRC/PX4-Autopilot/msg/"*.msg "$PROJECT_DIR/ros2_ws/src/px4_msgs/msg/" 2>/dev/null || true
  }
fi
# Ubuntu 24.04 запрещает pip в системный Python (PEP 668); пакеты чисто пользовательские, это безопасно.
sudo apt install -y python3-numpy python3-opencv python3-matplotlib
pip3 install --user --break-system-packages zxing-cpp python-barcode pillow
set +u; source /opt/ros/jazzy/setup.bash
(cd "$PROJECT_DIR/ros2_ws" && colcon build --symlink-install)

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

#!/usr/bin/env bash
# Установка RackScan на Raspberry Pi 5 / 4 с Raspberry Pi OS (64-bit), Bookworm или Trixie.
# Запуск из папки проекта:   bash deploy/install_pi.sh
#
# Системные пакеты ставятся через apt (Picamera2, OpenCV и numpy собраны друг под друга),
# в виртуальное окружение добавляется только декодер кодов. Не ставьте opencv-python и numpy
# через pip поверх системных: Picamera2 перестанет импортироваться из-за другой версии numpy.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo "== 1/5 системные пакеты =="
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip python3-numpy python3-opencv python3-picamera2 \
    python3-flask python3-serial python3-pil fonts-dejavu-core

echo "== 2/5 виртуальное окружение (с доступом к системным пакетам) =="
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-pi.txt

echo "== 3/5 проверка импорта =="
.venv/bin/python - <<'PY'
import cv2, numpy, zxingcpp, flask
print("OpenCV", cv2.__version__, "| numpy", numpy.__version__, "| ArUco:", hasattr(cv2, "aruco"))
try:
    import picamera2
    print("Picamera2: есть")
except Exception as e:
    print("Picamera2: НЕТ —", e)
PY

echo "== 4/5 доступ к последовательному порту (сенсорная головка) =="
sudo usermod -aG dialout "$USER" || true

echo "== 5/5 самопроверка на синтетической сцене (1–3 минуты) =="
.venv/bin/python -m rackscan selftest --quiet

cat <<EOF

Готово. Дальше:
  1. Скопируйте config/pi.example.json в config/pi.json и поправьте под свой стенд.
  2. Проверка узлов:   .venv/bin/python -m rackscan check -c config/pi.json
  3. Запуск:           .venv/bin/python -m rackscan run -c config/pi.json
  4. Автозапуск:       sudo cp deploy/rackscan.service /etc/systemd/system/ && \\
                       sudo sed -i "s|__ROOT__|$ROOT|g; s|__USER__|$USER|g" /etc/systemd/system/rackscan.service && \\
                       sudo systemctl daemon-reload && sudo systemctl enable --now rackscan
Если пользователь только что добавлен в группу dialout — перезайдите в систему.
EOF

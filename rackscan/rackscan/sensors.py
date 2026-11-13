"""Сенсорная головка на ESP32: инерциальный модуль, дальномер ToF, кнопка, светодиоды.

Обмен — текстовые строки по USB-UART, 115200 бод, в духе NMEA 0183: «$…*КС», где КС —
исключающее ИЛИ всех символов между «$» и «*», две шестнадцатеричные цифры.

ESP32 -> Raspberry Pi, 50 раз в секунду:
    $RSI,<мс>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>,<крен>,<тангаж>,<дальность>,<кнопка>*КС
    ускорения — в мg, угловые скорости — в сотых °/с, углы — в сотых градуса,
    дальность — в мм (−1 — вне диапазона), кнопка — 0/1.
Raspberry Pi -> ESP32:
    $RSC,LED,<r>,<g>,<b>*КС      цвет индикатора состояния (0..255)
    $RSC,LIGHT,<0..255>*КС       яркость подсветки
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass


def checksum(body: str) -> str:
    cs = 0
    for ch in body:
        cs ^= ord(ch)
    return f"{cs:02X}"


def frame(body: str) -> str:
    """Собрать строку протокола из тела: 'RSC,LED,255,0,0' -> '$RSC,LED,255,0,0*5B\\r\\n'."""
    return f"${body}*{checksum(body)}\r\n"


@dataclass(frozen=True)
class SensorReading:
    t_ms: int
    accel_g: tuple[float, float, float]
    gyro_dps: tuple[float, float, float]
    roll_deg: float
    pitch_deg: float
    tof_mm: float | None
    button: bool

    @property
    def gyro_norm(self) -> float:
        return math.sqrt(sum(g * g for g in self.gyro_dps))


def parse_line(line: str) -> SensorReading | None:
    """Разобрать строку $RSI. Повреждённые строки (нет «*», не сошлась сумма, не то число полей)
    отбрасываются молча: на линии бывает мусор при включении ESP32."""
    line = line.strip()
    if not line.startswith("$") or "*" not in line:
        return None
    body, _, cs = line[1:].rpartition("*")
    if checksum(body) != cs.upper():
        return None
    p = body.split(",")
    if p[0] != "RSI" or len(p) != 12:
        return None
    try:
        v = [int(x) for x in p[1:]]
    except ValueError:
        return None
    return SensorReading(
        t_ms=v[0], accel_g=(v[1] / 1000, v[2] / 1000, v[3] / 1000),
        gyro_dps=(v[4] / 100, v[5] / 100, v[6] / 100), roll_deg=v[7] / 100, pitch_deg=v[8] / 100,
        tof_mm=None if v[9] < 0 else float(v[9]), button=bool(v[10]))


LED_COLORS = {"RED": (255, 0, 0), "YELLOW": (255, 150, 0), "GREEN": (0, 255, 0), "IDLE": (0, 0, 40)}


class SensorHub:
    """Фоновое чтение сенсорной головки. Если порта или pyserial нет — available = False,
    стенд работает без датчиков (контроль неподвижности тогда только по изображению)."""

    def __init__(self, port: str, baud: int = 115200, stale_s: float = 0.5):
        self.port, self.baud, self.stale_s = port, baud, stale_s
        self.available = False
        self.error = ""
        self._last: SensorReading | None = None
        self._last_t = 0.0
        self._ser = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._led: tuple[int, int, int] | None = None
        self.button_presses = 0
        self._btn_prev = False
        try:
            import serial
            self._ser = serial.Serial(port, baud, timeout=0.2)
            self.available = True
            threading.Thread(target=self._run, name="sensors", daemon=True).start()
        except Exception as e:   # нет pyserial, нет порта, порт занят
            self.error = str(e)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self._ser.readline().decode("ascii", "ignore")
            except Exception as e:
                self.error, self.available = str(e), False
                return
            r = parse_line(raw)
            if r is not None:
                with self._lock:
                    self._last, self._last_t = r, time.monotonic()
                    if r.button and not self._btn_prev:
                        self.button_presses += 1
                    self._btn_prev = r.button

    def latest(self) -> SensorReading | None:
        """Последнее показание, если оно свежее stale_s."""
        with self._lock:
            if self._last is None or time.monotonic() - self._last_t > self.stale_s:
                return None
            return self._last

    def send(self, body: str) -> None:
        if self._ser is not None and self.available:
            try:
                self._ser.write(frame(body).encode("ascii"))
            except Exception as e:
                self.error = str(e)

    def set_status(self, status: str) -> None:
        """Продублировать цвет состояния на светодиоде головки (шлётся только при смене)."""
        rgb = LED_COLORS.get(status, LED_COLORS["IDLE"])
        if rgb != self._led:
            self._led = rgb
            self.send(f"RSC,LED,{rgb[0]},{rgb[1]},{rgb[2]}")

    def set_light(self, level: int) -> None:
        self.send(f"RSC,LIGHT,{max(0, min(255, int(level)))}")

    def close(self) -> None:
        self._stop.set()
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass

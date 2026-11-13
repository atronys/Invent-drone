/*
 * RackScan — сенсорная головка на ESP32 (плата ESP32 DevKit V1, 30 или 38 выводов).
 *
 * Датчики на общей шине I2C (3,3 В):
 *   MPU-6050 (GY-521)  — акселерометр + гироскоп, адрес 0x68
 *   VL53L1X            — лазерный дальномер ToF до 4 м, адрес 0x29
 * Исполнительные:
 *   WS2812B, 8 светодиодов — индикатор состояния (красный / жёлтый / зелёный)
 *   MOSFET-ключ             — подсветка 5 В, ШИМ
 *   Кнопка                  — «снять кадр / подтвердить»
 *
 * Выводы (совпадают со схемой РСКН.467239.001 Э4):
 *   GPIO21 SDA, GPIO22 SCL, GPIO4 DIN ленты WS2812B, GPIO25 затвор MOSFET, GPIO27 кнопка на GND.
 *
 * Протокол (см. rackscan/sensors.py): 115200 бод, строки «$тело*КС\r\n», КС — XOR тела.
 *   выход, 50 Гц:  $RSI,мс,ax,ay,az,gx,gy,gz,крен,тангаж,дальность,кнопка*КС
 *                  (мg; сотые °/с; сотые градуса; мм, −1 — нет цели; 0/1)
 *   вход:          $RSC,LED,r,g,b*КС    $RSC,LIGHT,0..255*КС
 *
 * Библиотеки (Менеджер библиотек Arduino IDE): «Adafruit NeoPixel», «VL53L1X» (Pololu).
 * Плата: пакет «esp32» от Espressif, версия 3.x.
 *
 * ВНИМАНИЕ: прошивка собрана по документации на микросхемы и на реальной плате в рамках
 * этой работы не запускалась. Порядок первой проверки — в руководстве, разд. «Сенсорная головка».
 */
#include <Wire.h>
#include <Adafruit_NeoPixel.h>
#include <VL53L1X.h>

const int PIN_SDA = 21, PIN_SCL = 22, PIN_LED = 4, PIN_LIGHT = 25, PIN_BUTTON = 27;
const int LED_COUNT = 8;
const uint8_t MPU_ADDR = 0x68;
const uint32_t PERIOD_MS = 20;              // 50 Гц
const float GYRO_LSB = 65.5f;               // ±500 °/с
const float ACC_LSB = 8192.0f;              // ±4 g
const float ALPHA = 0.98f;                  // комплементарный фильтр: доля гироскопа

Adafruit_NeoPixel strip(LED_COUNT, PIN_LED, NEO_GRB + NEO_KHZ800);
VL53L1X tof;
bool tofOk = false, mpuOk = false;
float gyroBias[3] = {0, 0, 0};
float roll = 0, pitch = 0;
uint32_t lastTick = 0;
int lastDistance = -1;
char rx[64];
uint8_t rxLen = 0;

void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

bool mpuRead(int16_t *acc, int16_t *gyr) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);                          // ACCEL_XOUT_H
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)MPU_ADDR, 14) != 14) return false;
  int16_t v[7];
  for (int i = 0; i < 7; i++) {
    uint8_t hi = Wire.read();
    v[i] = (int16_t)((hi << 8) | Wire.read());
  }
  acc[0] = v[0]; acc[1] = v[1]; acc[2] = v[2];
  gyr[0] = v[4]; gyr[1] = v[5]; gyr[2] = v[6];   // v[3] — температура
  return true;
}

bool mpuInit() {
  Wire.beginTransmission(MPU_ADDR);
  if (Wire.endTransmission() != 0) return false;
  mpuWrite(0x6B, 0x01);                      // выход из сна, тактирование от гироскопа X
  mpuWrite(0x1A, 0x03);                      // цифровой фильтр 44 Гц
  mpuWrite(0x1B, 0x08);                      // гироскоп ±500 °/с
  mpuWrite(0x1C, 0x08);                      // акселерометр ±4 g
  delay(100);
  // смещение нуля гироскопа: головка должна лежать неподвижно первую секунду
  long sum[3] = {0, 0, 0};
  int n = 0;
  for (int i = 0; i < 200; i++) {
    int16_t a[3], g[3];
    if (mpuRead(a, g)) { sum[0] += g[0]; sum[1] += g[1]; sum[2] += g[2]; n++; }
    delay(5);
  }
  if (n == 0) return false;
  for (int i = 0; i < 3; i++) gyroBias[i] = (float)sum[i] / n;
  return true;
}

void setAll(uint8_t r, uint8_t g, uint8_t b) {
  for (int i = 0; i < LED_COUNT; i++) strip.setPixelColor(i, strip.Color(r, g, b));
  strip.show();
}

uint8_t checksum(const char *s, size_t n) {
  uint8_t cs = 0;
  for (size_t i = 0; i < n; i++) cs ^= (uint8_t)s[i];
  return cs;
}

void handleCommand(char *body) {             // тело без «$» и «*КС»
  int a, b, c;
  if (sscanf(body, "RSC,LED,%d,%d,%d", &a, &b, &c) == 3) {
    setAll(constrain(a, 0, 255), constrain(b, 0, 255), constrain(c, 0, 255));
  } else if (sscanf(body, "RSC,LIGHT,%d", &a) == 1) {
    analogWrite(PIN_LIGHT, constrain(a, 0, 255));
  }
}

void pollSerial() {
  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch == '$') { rxLen = 0; continue; }
    if (ch == '\n' || ch == '\r') {
      rx[rxLen] = 0;
      char *star = strrchr(rx, '*');
      if (star && strlen(star) >= 3) {
        uint8_t want = (uint8_t)strtol(star + 1, NULL, 16);
        if (checksum(rx, star - rx) == want) { *star = 0; handleCommand(rx); }
      }
      rxLen = 0;
      continue;
    }
    if (rxLen < sizeof(rx) - 1) rx[rxLen++] = ch;
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_BUTTON, INPUT_PULLUP);
  pinMode(PIN_LIGHT, OUTPUT);
  analogWrite(PIN_LIGHT, 0);
  strip.begin();
  strip.setBrightness(60);                   // ток ленты ≤ 0,2 А: хватает питания от USB
  setAll(0, 0, 40);
  Wire.begin(PIN_SDA, PIN_SCL, 400000);
  mpuOk = mpuInit();
  tof.setTimeout(100);
  tofOk = tof.init();
  if (tofOk) {
    tof.setDistanceMode(VL53L1X::Long);
    tof.setMeasurementTimingBudget(33000);   // 33 мс на измерение
    tof.startContinuous(40);
  }
  lastTick = millis();
}

void loop() {
  pollSerial();
  uint32_t now = millis();
  if (now - lastTick < PERIOD_MS) return;
  float dt = (now - lastTick) / 1000.0f;
  lastTick = now;

  int16_t a[3] = {0, 0, 0}, g[3] = {0, 0, 0};
  float ax = 0, ay = 0, az = 0, gx = 0, gy = 0, gz = 0;
  if (mpuOk && mpuRead(a, g)) {
    ax = a[0] / ACC_LSB; ay = a[1] / ACC_LSB; az = a[2] / ACC_LSB;
    gx = (g[0] - gyroBias[0]) / GYRO_LSB;
    gy = (g[1] - gyroBias[1]) / GYRO_LSB;
    gz = (g[2] - gyroBias[2]) / GYRO_LSB;
    // углы по акселерометру (верны в покое) и по гироскопу (верны в движении) — смешиваем
    float rollAcc = atan2f(ay, az) * 57.29578f;
    float pitchAcc = atan2f(-ax, sqrtf(ay * ay + az * az)) * 57.29578f;
    roll = ALPHA * (roll + gx * dt) + (1 - ALPHA) * rollAcc;
    pitch = ALPHA * (pitch + gy * dt) + (1 - ALPHA) * pitchAcc;
  }
  if (tofOk && tof.dataReady()) {
    int d = tof.read(false);
    lastDistance = (tof.ranging_data.range_status == VL53L1X::RangeValid) ? d : -1;
  }
  int button = digitalRead(PIN_BUTTON) == LOW ? 1 : 0;

  char body[128];
  int n = snprintf(body, sizeof(body), "RSI,%lu,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
                   (unsigned long)now, (int)lroundf(ax * 1000), (int)lroundf(ay * 1000), (int)lroundf(az * 1000),
                   (int)lroundf(gx * 100), (int)lroundf(gy * 100), (int)lroundf(gz * 100),
                   (int)lroundf(roll * 100), (int)lroundf(pitch * 100), lastDistance, button);
  Serial.printf("$%s*%02X\r\n", body, checksum(body, n));
}

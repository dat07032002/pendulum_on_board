/*
 * Furuta Pendulum — Torque-Source LQR Balance (GM3506 + TMC6300 + AS5048A + AS5600)
 *
 * Ben Katz's architecture: LQR outputs motor voltage directly. No velocity
 * integration, no dither — the gimbal motor produces smooth torque at any speed.
 *
 * Wiring:
 *   Motor:    25->UH 26->UL 27->VH 14->VL 4->WH 13->WL (TMC6300)
 *   Arm enc:  5->CS 18->SCK 23->MOSI 19->MISO (AS5048A SPI, 14-bit)
 *   Pend enc: 21->SDA 22->SCL (AS5600 I2C, 12-bit, theta=0 upright)
 *   Power:    TMC6300 VIO->3V3, VIN->11V
 *
 * Commands (Serial 921600):
 *   bal           arm balance (lift the rod to upright)
 *   t <volts>     direct torque/voltage
 *   s / 0         stop
 *   k <4 gains>   set LQR gains
 *   tr <deg>      theta trim
 *   hand <deg>    handoff window
 *   log / nolog   toggle per-tick system-ID stream (log=[t_ms,phi,theta,phi_dot,theta_dot,V,theta_raw])
 *   params        print all parameters
 *   calhang/calup calibrate AS5600 (hanging / upright); saved to flash
 *   calfoc        re-run FOC calibration and save; clearcal wipes all cal
 *   raw           print AS5600 raw angle
 */
#include <SPI.h>
#include <Wire.h>
#include <math.h>
#include <Preferences.h>   // NVS flash storage for FOC + AS5600 calibration
#include "policy_weights.h" // RL actor weights (rl/export_policy.py) -> MODE_RL on-chip MLP

// === Pin definitions ===
#define UH 25
#define UL 26
#define VH 27
#define VL 14
#define WH  4
#define WL 13
#define CS_PIN 5
#define SDA_PIN 21
#define SCL_PIN 22

// === Motor parameters ===
const int POLE_PAIRS = 11;
const float V_SUPPLY = 11.0f;
const int PWM_MAX = 255;
const float V_LIMIT_MAX = 10.0f;

// === FOC calibration (found by auto-sweep, persisted to NVS) ===
int foc_dir = -1;
float foc_offset = 0.0f;
bool foc_valid = false;            // true once FOC has been calibrated (or loaded)
Preferences prefs;                 // NVS namespace "furuta"

// === AS5600 (pendulum theta, I2C) ===
#define AS5600_ADDR 0x36
#define RAW_ANGLE_H 0x0C
int UPRIGHT_RAW = 0;
const int AS5600_MAX_RAW_STEP = 768;

// === Sensor bus speeds (raised from defaults for lower read latency) ===
const uint32_t SPI_HZ = 1000000;   // AS5048A SPI clock — 4 MHz corrupted drive reads; back to known-good 1 MHz
const uint32_t I2C_HZ = 400000;    // AS5600 I2C clock (was 100 kHz; AS5600 not in the arm-torque path)

// === LQR + Observer (from balance_torque.py) ===
const float AD[16] = {1.000000f, -0.001050f, 0.004855f, -0.000002f,
                      0.000000f, 1.003084f, 0.000087f, 0.005005f,
                      0.000000f, -0.416259f, 0.942374f, -0.001050f,
                      0.000000f, 1.231765f, 0.034604f, 1.003084f};
const float BD[4] = {0.002139f, -0.001284f, 0.847440f, -0.508886f};
const float LOBS[12] = {0.990201f, -0.001070f, 0.004634f,
                        -0.000021f, 1.064660f, -0.000054f,
                        0.000928f, -0.428129f, 0.899271f,
                        -0.004199f, 15.391941f, 0.005799f};
float Kgain[4] = {-4.656349f, -62.347487f, -1.531077f, -4.535903f};

// === Runtime state ===
enum RunMode : uint8_t { MODE_IDLE, MODE_MANUAL, MODE_BAL_ARMED, MODE_BALANCING, MODE_RL,
                         MODE_RL_RECOVER };
RunMode run_mode = MODE_IDLE;

// RL on-chip policy state (MODE_RL): prev_action goes into the obs; phi_ref recenters the
// arm at engage so the policy's arm-centering matches the sim (which starts arm near 0).
float rl_prev_action = 0.0f;
float rl_phi_ref = 0.0f;
const float RL_ARM_LIMIT = 160.0f * PI / 180.0f;   // PC-in-loop used the same cable guard
// Boot auto-start: a few seconds after power-on, engage MODE_RL automatically (swing-up from
// hanging + balance), so it's plug-and-play. Any serial command cancels it (so you can stop it).
bool rl_autostart_pending = true;
unsigned long boot_ms = 0;
const unsigned long RL_AUTOSTART_MS = 4000;        // delay before auto-engage
// Auto-recovery: if the policy winds the arm to the cable guard, unwind it back to center and
// re-engage instead of dying in IDLE. So a failed swing-up self-retries (plug-and-play).
unsigned long rl_recover_ms = 0;
const unsigned long RL_RECOVER_TIMEOUT_MS = 6000;  // give up homing after this -> IDLE

float bal_theta_ref = 0.0f;
float bal_handoff_rad = 5.0f * PI / 180.0f;
float bal_handoff_thd = 2.0f;
const float BAL_FALL_RAD = 30.0f * PI / 180.0f;
const float BAL_V_MAX = 400.0f;
float voltage_limit = 6.0f;
float manual_voltage = 0.0f;

float xhat[4] = {0, 0, 0, 0};
float bal_prev_V = 0.0f;

// Last voltage actually commanded to the motor (set in driveVoltage, for logging)
float last_V = 0.0f;

// System-ID streaming: when on, print one log=[...] line per control tick.
bool stream_log = false;

// Velocity-filter EMA weights (new-sample fraction). Higher = lower lag, more noise.
// phi_dot was 0.15 (~28 ms lag) — too laggy given the clean 14-bit arm encoder + the
// Kalman observer that already smooths. Set to 0.50 (~5 ms lag, matches theta_dot;
// noise 0.08 rad/s << operating 5-19 rad/s). Runtime-tunable via 'pdf <a>'.
float phi_dot_alpha = 0.50f;

// Sensor state
float phi_angle = 0.0f;          // arm angle from AS5048A [rad]
float phi_prev = 0.0f;
float phi_full = 0.0f;           // unwrapped arm angle
float phi_dot_filt = 0.0f;
float theta_angle = 0.0f;        // pendulum from upright [rad]
float theta_prev_raw = 0.0f;
float theta_dot_filt = 0.0f;
bool sensor_init = false;
uint16_t as5600_last_raw = 0;
bool as5600_valid = false;

unsigned long loop_us = 0;
const unsigned long LOOP_PERIOD_US = 5000;  // 200 Hz

char cmd_buf[64];
uint8_t cmd_len = 0;

// ============================================================
// AS5048A (arm encoder, SPI)
// ============================================================
uint16_t readAS5048A() {
  uint16_t command = 0x3FFF | (1 << 14);
  uint16_t temp = command; int parity = 0;
  while (temp) { parity ^= (temp & 1); temp >>= 1; }
  if (parity) command |= (1 << 15);
  SPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE1));
  digitalWrite(CS_PIN, LOW); SPI.transfer16(command); digitalWrite(CS_PIN, HIGH);
  delayMicroseconds(1);
  digitalWrite(CS_PIN, LOW); uint16_t r = SPI.transfer16(0xFFFF); digitalWrite(CS_PIN, HIGH);
  SPI.endTransaction();
  return r & 0x3FFF;
}

// Read any AS5048A register (14-bit data). Used for diagnostics (DIAAGC, MAGNITUDE).
uint16_t readAS5048Reg(uint16_t addr) {
  uint16_t cmd = 0x4000 | (addr & 0x3FFF);              // bit14 = read
  uint16_t t = cmd; int p = 0; while (t) { p ^= (t & 1); t >>= 1; }
  if (p) cmd |= 0x8000;
  uint16_t nop = 0x4000 | 0x3FFF; t = nop; p = 0; while (t) { p ^= (t & 1); t >>= 1; }
  if (p) nop |= 0x8000;
  SPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE1));
  digitalWrite(CS_PIN, LOW); SPI.transfer16(cmd); digitalWrite(CS_PIN, HIGH);
  delayMicroseconds(1);
  digitalWrite(CS_PIN, LOW); uint16_t r = SPI.transfer16(nop); digitalWrite(CS_PIN, HIGH);
  SPI.endTransaction();
  return r & 0x3FFF;
}

// ============================================================
// AS5600 (pendulum encoder, I2C)
// ============================================================
bool readAS5600(uint16_t &raw) {
  Wire.beginTransmission(AS5600_ADDR);
  Wire.write(RAW_ANGLE_H);
  if (Wire.endTransmission(false) != 0) return false;
  Wire.requestFrom(AS5600_ADDR, 2);
  if (Wire.available() < 2) return false;
  uint8_t h = Wire.read(), l = Wire.read();
  raw = ((h & 0x0F) << 8) | l;
  return true;
}

// AS5600 register reads (for STATUS/AGC/MAGNITUDE health check).
uint8_t as5600Reg8(uint8_t reg) {
  Wire.beginTransmission(AS5600_ADDR); Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return 0xFF;
  Wire.requestFrom(AS5600_ADDR, 1);
  return Wire.available() ? Wire.read() : 0xFF;
}
uint16_t as5600Reg16(uint8_t reg) {
  Wire.beginTransmission(AS5600_ADDR); Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return 0xFFFF;
  Wire.requestFrom(AS5600_ADDR, 2);
  if (Wire.available() < 2) return 0xFFFF;
  uint8_t h = Wire.read(), l = Wire.read();
  return ((h & 0x0F) << 8) | l;
}

// Set the AS5600 internal slow filter (CONF SF bits[9:8]). 0=16x..3=2x. Lower = less lag.
void setAS5600SlowFilter(uint8_t sf) {
  Wire.beginTransmission(AS5600_ADDR);
  Wire.write(0x07);                 // CONF high byte
  Wire.write(sf & 0x03);            // SF in bits[1:0] of the high byte (=bits 8-9)
  Wire.write(0x00);                 // CONF low byte: PM/HYST/OUTS/PWMF all default
  Wire.endTransmission();
}

float wrapToPi(float x) {
  while (x > PI) x -= 2.0f * PI;
  while (x < -PI) x += 2.0f * PI;
  return x;
}

// ============================================================
// Motor drive (FOC)
// ============================================================
void setFOC(float elec_angle, float voltage) {
  float duty = fabsf(voltage) / V_SUPPLY;
  if (duty > 1.0f) duty = 1.0f;
  float q = elec_angle + ((voltage >= 0) ? (PI/2.0f) : (-PI/2.0f));
  float a = duty * (sinf(q)*0.5f+0.5f);
  float b = duty * (sinf(q-2.094f)*0.5f+0.5f);
  float c = duty * (sinf(q+2.094f)*0.5f+0.5f);
  float mn = min(a, min(b, c));
  float an = a-mn, bn = b-mn, cn = c-mn;
  analogWrite(UH, (int)(an*PWM_MAX));
  analogWrite(VH, (int)(bn*PWM_MAX));
  analogWrite(WH, (int)(cn*PWM_MAX));
  digitalWrite(UL, (an<0.01f)?HIGH:LOW);
  digitalWrite(VL, (bn<0.01f)?HIGH:LOW);
  digitalWrite(WL, (cn<0.01f)?HIGH:LOW);
}

void motorOff() {
  analogWrite(UH, 0); analogWrite(VH, 0); analogWrite(WH, 0);
  digitalWrite(UL, LOW); digitalWrite(VL, LOW); digitalWrite(WL, LOW);
  last_V = 0.0f;
}

void driveVoltage(float V) {
  float sa = readAS5048A() * 2.0f * PI / 16384.0f;
  float ea = foc_dir * sa * POLE_PAIRS - foc_offset;
  V = constrain(V, -voltage_limit, voltage_limit);
  last_V = V;
  setFOC(ea, V);
}

// ============================================================
// Sensor update
// ============================================================
void updateSensors(float dt) {
  // AS5048A: arm angle (phi)
  float sa = readAS5048A() * 2.0f * PI / 16384.0f;
  if (!sensor_init) {
    phi_prev = sa;
    sensor_init = true;
  }
  float d_phi = sa - phi_prev;
  if (d_phi > PI) d_phi -= 2.0f*PI;
  if (d_phi < -PI) d_phi += 2.0f*PI;
  phi_full += d_phi * foc_dir;
  phi_prev = sa;
  phi_angle = phi_full;
  if (dt > 0) phi_dot_filt = (1.0f - phi_dot_alpha) * phi_dot_filt + phi_dot_alpha * (d_phi * foc_dir / dt);

  // AS5600: pendulum angle (theta)
  uint16_t raw;
  if (readAS5600(raw)) {
    if (as5600_valid) {
      int step = (int)raw - (int)as5600_last_raw;
      if (step > 2048) step -= 4096;
      if (step < -2048) step += 4096;
      if (abs(step) < AS5600_MAX_RAW_STEP) {
        as5600_last_raw = raw;
      }
    } else {
      as5600_last_raw = raw;
      as5600_valid = true;
    }
  }
  float new_theta = -wrapToPi((as5600_last_raw - UPRIGHT_RAW) * 2.0f * PI / 4096.0f);
  if (dt > 0) {
    float d_theta = wrapToPi(new_theta - theta_angle);
    theta_dot_filt = 0.5f * theta_dot_filt + 0.5f * (d_theta / dt);
  }
  theta_angle = new_theta;
}

// ============================================================
// FOC auto-calibration
// ============================================================
// Bidirectional sweep calibration: for each (dir, offset) drive BOTH +V and -V and score
// by min(|+motion|, |-motion|) with opposite signs — i.e. pick the offset that gives strong
// torque BOTH ways (symmetric). The old version maximized +motion only, so it could land on
// an asymmetric offset (the 30-deg lottery: 300 symmetric, 150/270 not). 2.0 V / 120 ms/step.
const float CAL_V = 2.0f;
const unsigned long CAL_STEP_MS = 120;

float calMotion(int dir, float offset, float V) {
  float start = readAS5048A() * 2.0f * PI / 16384.0f;
  unsigned long t0 = millis();
  while (millis() - t0 < CAL_STEP_MS) {
    float sa = readAS5048A() * 2.0f * PI / 16384.0f;
    setFOC(dir * sa * POLE_PAIRS - offset, V);
    delayMicroseconds(200);
  }
  float end = readAS5048A() * 2.0f * PI / 16384.0f;
  motorOff(); delay(120);
  float m = end - start;
  if (m > PI) m -= 2.0f * PI;
  if (m < -PI) m += 2.0f * PI;
  return m;
}

void calibrateFOC() {
  Serial.println("# FOC calibrating (bidirectional sweep)...");
  float best_score = -1.0f, best_offset = 0.0f;
  int best_dir = 1;
  for (int dir = -1; dir <= 1; dir += 2) {
    for (int k = 0; k < 12; k++) {
      float offset = k * (2.0f * PI / 12.0f);
      float mp = calMotion(dir, offset, CAL_V);
      float mm = calMotion(dir, offset, -CAL_V);
      // strong BOTH ways and opposite signs -> symmetric torque
      float score = (mp * mm < 0.0f) ? fminf(fabsf(mp), fabsf(mm)) : 0.0f;
      if (score > best_score) { best_score = score; best_offset = offset; best_dir = dir; }
    }
  }
  foc_dir = best_dir;
  foc_offset = best_offset;
  motorOff();
  Serial.print("# FOC: dir="); Serial.print(foc_dir);
  Serial.print(" offset="); Serial.print(foc_offset * 57.3f, 0);
  Serial.print(" sym_motion="); Serial.println(best_score * 57.3f, 1);
}

// ============================================================
// Calibration persistence (NVS / flash)
// ============================================================
void saveCal() {
  prefs.putInt("foc_dir", foc_dir);
  prefs.putFloat("foc_off", foc_offset);
  prefs.putInt("up_raw", UPRIGHT_RAW);
  prefs.putBool("foc_ok", foc_valid);
}

void loadCal() {
  foc_dir    = prefs.getInt("foc_dir", -1);
  foc_offset = prefs.getFloat("foc_off", 0.0f);
  UPRIGHT_RAW = prefs.getInt("up_raw", 0);
  foc_valid  = prefs.getBool("foc_ok", false);
}

// ============================================================
// Balance step
// ============================================================
void balanceStep(float dt) {
  if (run_mode == MODE_BAL_ARMED) {
    motorOff();
    if (fabsf(theta_angle - bal_theta_ref) < bal_handoff_rad
        && fabsf(theta_dot_filt) < bal_handoff_thd) {
      xhat[0] = phi_angle; xhat[1] = theta_angle;
      xhat[2] = phi_dot_filt; xhat[3] = theta_dot_filt;
      bal_prev_V = 0.0f;
      run_mode = MODE_BALANCING;
      Serial.println("# BALANCE engaged");
    }
    return;
  }

  if (fabsf(theta_angle - bal_theta_ref) > BAL_FALL_RAD) {
    run_mode = MODE_BAL_ARMED;
    motorOff();
    Serial.println("# balance lost -> re-arming");
    return;
  }

  // LQR FIRST: V = -K * xhat, using the CURRENT estimate. Computing the control
  // before propagating the observer (and then propagating with THIS V) is the
  // correct predictor-form order. Propagating first with the previous step's
  // voltage -- as this code did before -- makes the combined observer+LQR loop
  // unstable (augmented |eig| ~1.17) even though K and L are each stable. See sim.py.
  float th_err = xhat[1] - bal_theta_ref;
  float V = -(Kgain[0]*xhat[0] + Kgain[1]*th_err + Kgain[2]*xhat[2] + Kgain[3]*xhat[3]);
  V = constrain(V, -voltage_limit, voltage_limit);
  driveVoltage(V);

  // Observer: xhat[k+1] = AD*xhat[k] + BD*V[k] + L*(y - C*xhat[k])
  float in0 = phi_angle - xhat[0];
  float in1 = wrapToPi(theta_angle - xhat[1]);
  float in2 = phi_dot_filt - xhat[2];
  float xn[4];
  for (int i = 0; i < 4; i++) {
    float ax = AD[i*4]*xhat[0] + AD[i*4+1]*xhat[1] + AD[i*4+2]*xhat[2] + AD[i*4+3]*xhat[3];
    float lx = LOBS[i*3]*in0 + LOBS[i*3+1]*in1 + LOBS[i*3+2]*in2;
    xn[i] = ax + BD[i]*V + lx;
  }
  for (int i = 0; i < 4; i++) xhat[i] = xn[i];

  bal_prev_V = V;

  // Telemetry (suppressed while the ID log stream is active so it stays parseable)
  static unsigned long last_print = 0;
  if (!stream_log && millis() - last_print > 250) {
    last_print = millis();
    Serial.print("# bal th="); Serial.print(theta_angle * 57.3f, 1);
    Serial.print(" phi="); Serial.print(phi_angle * 57.3f, 0);
    Serial.print(" V="); Serial.print(V, 2);
    Serial.print(" thd="); Serial.println(xhat[3], 1);
  }
}

// ============================================================
// Command handling
// ============================================================
void handleCommand(char *cmd) {
  while (*cmd == ' ') cmd++;
  size_t len = strlen(cmd);
  while (len > 0 && cmd[len-1] == ' ') cmd[--len] = '\0';
  for (size_t i = 0; cmd[i]; i++) cmd[i] = tolower(cmd[i]);
  if (!*cmd) return;
  rl_autostart_pending = false;     // any command cancels the boot auto-start

  if (!strcmp(cmd, "s") || !strcmp(cmd, "0")) {
    run_mode = MODE_IDLE; manual_voltage = 0; motorOff();
    Serial.println("# STOP"); return;
  }
  if (!strcmp(cmd, "bal")) {
    run_mode = MODE_BAL_ARMED;
    Serial.println("# balance armed: lift the rod upright"); return;
  }
  if (!strcmp(cmd, "rl")) {
    rl_prev_action = 0.0f; rl_phi_ref = phi_full;   // recenter arm at engage
    run_mode = MODE_RL;
    Serial.println("# RL policy on (on-chip MLP). hold rod upright; 's' to stop"); return;
  }
  if (!strcmp(cmd, "log")) {
    stream_log = true;
    Serial.println("# log on: log=[t_ms,phi,theta,phi_dot,theta_dot,V,theta_raw]"); return;
  }
  if (!strcmp(cmd, "nolog")) {
    stream_log = false;
    Serial.println("# log off"); return;
  }
  if (!strcmp(cmd, "params") || !strcmp(cmd, "?")) {
    Serial.print("# K="); for (int i=0;i<4;i++) { Serial.print(Kgain[i],2); Serial.print(i<3?",":"|"); }
    Serial.print(" vlim="); Serial.print(voltage_limit, 1);
    Serial.print(" tr="); Serial.print(bal_theta_ref*57.3f, 1);
    Serial.print(" hand="); Serial.print(bal_handoff_rad*57.3f, 1);
    Serial.print(" mode="); Serial.println(run_mode);
    return;
  }
  if (!strcmp(cmd, "raw")) {
    uint16_t r; if (readAS5600(r)) {
      float th = -wrapToPi((r - UPRIGHT_RAW) * 2.0f * PI / 4096.0f);
      Serial.print("# raw="); Serial.print(r);
      Serial.print(" UP="); Serial.print(UPRIGHT_RAW);
      Serial.print(" th="); Serial.println(th * 57.3f, 1);
    } return;
  }
  if (!strcmp(cmd, "calhang")) {
    uint16_t r; if (readAS5600(r)) {
      UPRIGHT_RAW = (r + 2048) % 4096;
      saveCal();
      Serial.print("# UPRIGHT_RAW="); Serial.print(UPRIGHT_RAW);
      Serial.println(" (saved)");
    } return;
  }
  if (!strcmp(cmd, "calup")) {
    uint16_t r; if (readAS5600(r)) {
      UPRIGHT_RAW = r;
      saveCal();
      Serial.print("# UPRIGHT_RAW="); Serial.print(UPRIGHT_RAW);
      Serial.println(" (saved)");
    } return;
  }
  if (!strcmp(cmd, "health")) {
    uint8_t st = as5600Reg8(0x0B), agc = as5600Reg8(0x1A);
    uint16_t mag = as5600Reg16(0x1B);
    Serial.print("# AS5600 STATUS=0x"); Serial.print(st, HEX);
    Serial.print(" MD="); Serial.print((st >> 5) & 1);
    Serial.print(" ML(too-strong)="); Serial.print((st >> 4) & 1);
    Serial.print(" MH(too-weak)="); Serial.print((st >> 3) & 1);
    Serial.print(" AGC="); Serial.print(agc); Serial.print("/128(3V3) MAG="); Serial.println(mag);
    uint16_t d = readAS5048Reg(0x3FFD), m2 = readAS5048Reg(0x3FFE);
    Serial.print("# AS5048A AGC="); Serial.print(d & 0xFF);
    Serial.print("/255 OCF="); Serial.print((d >> 8) & 1);
    Serial.print(" COF="); Serial.print((d >> 9) & 1);
    Serial.print(" CompLow(weak)="); Serial.print((d >> 10) & 1);
    Serial.print(" CompHigh(strong)="); Serial.print((d >> 11) & 1);
    Serial.print(" MAG="); Serial.println(m2);
    return;
  }
  if (!strcmp(cmd, "calfoc")) {
    Serial.println("# re-running FOC calibration (arm will move)...");
    calibrateFOC();
    foc_valid = true; saveCal();
    Serial.println("# FOC cal saved"); return;
  }
  if (!strcmp(cmd, "clearcal")) {
    prefs.clear();
    foc_valid = false;
    Serial.println("# all calibration cleared; reboot to re-calibrate"); return;
  }

  float val;
  if (sscanf(cmd, "t %f", &val) == 1) {
    manual_voltage = val; run_mode = MODE_MANUAL;
    Serial.print("# torque "); Serial.println(val, 1); return;
  }
  if (sscanf(cmd, "tr %f", &val) == 1) {
    bal_theta_ref = val * PI / 180.0f;
    Serial.print("# tr="); Serial.println(val, 1); return;
  }
  if (sscanf(cmd, "hand %f", &val) == 1) {
    bal_handoff_rad = val * PI / 180.0f;
    Serial.print("# hand="); Serial.println(val, 1); return;
  }
  if (sscanf(cmd, "pdf %f", &val) == 1) {
    phi_dot_alpha = constrain(val, 0.05f, 1.0f);
    Serial.print("# phi_dot_alpha="); Serial.println(phi_dot_alpha, 2); return;
  }
  if (sscanf(cmd, "vlim %f", &val) == 1) {
    voltage_limit = constrain(val, 0.5f, V_LIMIT_MAX);
    Serial.print("# vlim="); Serial.println(voltage_limit, 1); return;
  }
  if (!strncmp(cmd, "k ", 2)) {
    float v[4]; char *p = cmd+2; bool ok = true;
    for (int i=0; i<4; i++) {
      char *end; v[i] = strtof(p, &end);
      if (p == end) { ok = false; break; }
      p = end;
    }
    if (ok) {
      for (int i=0;i<4;i++) Kgain[i] = v[i];
      Serial.print("# K="); for (int i=0;i<4;i++) { Serial.print(Kgain[i],2); Serial.print(i<3?",":" "); }
      Serial.println();
    } else Serial.println("# usage: k <phi> <th> <phid> <thd>");
    return;
  }
  Serial.println("# cmds: bal s t<V> k<4> tr<d> hand<d> vlim<V> log nolog raw health calhang calup calfoc clearcal params");
}

// ============================================================
// Setup & Loop
// ============================================================
void setup() {
  Serial.begin(921600);
  pinMode(UH, OUTPUT); pinMode(UL, OUTPUT);
  pinMode(VH, OUTPUT); pinMode(VL, OUTPUT);
  pinMode(WH, OUTPUT); pinMode(WL, OUTPUT);

  // NOTE: 20 kHz PWM was tried (quieter) but appears to weaken torque ~4x with this
  // high-side-PWM/low-side-static drive scheme — reverted to the default to restore torque.
  // analogWriteFrequency(UH, 20000); analogWriteFrequency(VH, 20000); analogWriteFrequency(WH, 20000);
  motorOff();

  // SPI (AS5048A)
  pinMode(CS_PIN, OUTPUT); digitalWrite(CS_PIN, HIGH);
  SPI.begin(18, 19, 23, CS_PIN);

  // I2C (AS5600)
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(I2C_HZ);
  Wire.setTimeOut(5);
  setAS5600SlowFilter(3);   // 2x slow filter (was default 16x) for lower pendulum latency

  delay(500);

  // Load persisted calibration; only sweep if we've never calibrated. This keeps
  // the cable from re-winding on every boot (the sweep moves the arm).
  prefs.begin("furuta", false);
  loadCal();
  if (foc_valid) {
    Serial.print("# FOC cal loaded from flash: dir="); Serial.print(foc_dir);
    Serial.print(" offset="); Serial.print(foc_offset * 57.3f, 0);
    Serial.print("  UPRIGHT_RAW="); Serial.println(UPRIGHT_RAW);
    Serial.println("#   (run 'calfoc' to re-calibrate, 'clearcal' to wipe)");
  } else {
    calibrateFOC();
    foc_valid = true;
    saveCal();
    Serial.println("# FOC cal saved to flash (won't sweep again on boot)");
  }

  // Init sensors
  updateSensors(0);
  phi_full = 0;
  loop_us = micros();

  Serial.println("# Furuta FOC Balance (GM3506 + AS5048A + AS5600)");
  Serial.println("# cmds: rl | bal | t <V> | s | k <4> | tr <deg> | log | nolog | calhang | calup | calfoc | clearcal | params");
  boot_ms = millis();
  Serial.print("# RL auto-start in "); Serial.print(RL_AUTOSTART_MS / 1000);
  Serial.println("s (any command cancels; 's' to stop)");
}

// ============================================================
// RL on-chip policy: obs[6] -> H0 (relu) -> H1 (relu) -> mu -> clip -> tanh -> action
// obs MUST match rl/furuta_env.py: [cos(th),sin(th),thd/15,clip(phi/pi,-2,2),phid/25,prev_a]
// ============================================================
float rlForward(const float *obs) {
  static float h0[RL_H0], h1[RL_H1];
  for (int i = 0; i < RL_H0; i++) {                 // layer 0: (H0 x 6)
    float s = RL_B0[i];
    for (int j = 0; j < RL_OBS; j++) s += RL_W0[i * RL_OBS + j] * obs[j];
    h0[i] = s > 0.0f ? s : 0.0f;
  }
  for (int i = 0; i < RL_H1; i++) {                 // layer 1: (H1 x H0)
    float s = RL_B1[i];
    for (int j = 0; j < RL_H0; j++) s += RL_W1[i * RL_H0 + j] * h0[j];
    h1[i] = s > 0.0f ? s : 0.0f;
  }
  float mu = RL_BM[0];                              // mu: (1 x H1)
  for (int j = 0; j < RL_H1; j++) mu += RL_WM[j] * h1[j];
  if (mu >  RL_CLIP_MEAN) mu =  RL_CLIP_MEAN;       // gSDE mean clamp (matches export)
  if (mu < -RL_CLIP_MEAN) mu = -RL_CLIP_MEAN;
  return tanhf(mu);                                 // action in [-1,1]
}

void rlStep(float dt) {
  float phi = phi_full - rl_phi_ref;               // arm relative to engage point
  if (fabsf(phi) > RL_ARM_LIMIT) {                  // cable guard -> unwind + retry
    run_mode = MODE_RL_RECOVER; rl_recover_ms = millis(); motorOff();
    Serial.println("# RL: arm limit -> homing to recenter, will retry"); return;
  }
  float obs[RL_OBS] = {
    cosf(theta_angle), sinf(theta_angle),
    theta_dot_filt / 15.0f,
    constrain(phi / PI, -2.0f, 2.0f),
    phi_dot_filt / 25.0f,
    rl_prev_action,
  };
  float a = rlForward(obs);
  rl_prev_action = a;
  driveVoltage(a * 6.0f);                            // V=6*action, driveVoltage clips to vlim
}

// Homing after a cable-guard hit: gently drive the arm back to the engage center (phi->0),
// then re-engage the policy. Keeps rl_phi_ref so the cable actually unwinds. Times out to IDLE.
void rlRecoverStep(float dt) {
  float phi = phi_full - rl_phi_ref;
  if (fabsf(phi) < 0.35f && fabsf(phi_dot_filt) < 1.0f) {   // recentered + slow -> retry
    motorOff(); rl_prev_action = 0.0f; run_mode = MODE_RL;
    Serial.println("# recentered -> re-engaging RL"); return;
  }
  if (millis() - rl_recover_ms > RL_RECOVER_TIMEOUT_MS) {   // stuck -> give up
    run_mode = MODE_IDLE; motorOff();
    Serial.println("# recover timeout -> idle ('rl' to retry)"); return;
  }
  driveVoltage(constrain(-1.5f * phi, -3.0f, 3.0f));        // P-home toward center (gentle)
}

void loop() {
  unsigned long now = micros();
  float dt = (now - loop_us) * 1e-6f;
  if (dt < LOOP_PERIOD_US * 1e-6f) return;
  loop_us = now;

  updateSensors(dt);

  // Boot auto-start: engage the RL policy a few seconds after power-on (plug-and-play).
  if (rl_autostart_pending && millis() - boot_ms > RL_AUTOSTART_MS) {
    rl_autostart_pending = false;
    rl_prev_action = 0.0f; rl_phi_ref = phi_full;     // recenter arm at engage
    run_mode = MODE_RL;
    Serial.println("# auto-start: RL swing-up + balance");
  }

  switch (run_mode) {
    case MODE_BAL_ARMED:
    case MODE_BALANCING:
      balanceStep(dt);
      break;
    case MODE_MANUAL:
      driveVoltage(manual_voltage);
      break;
    case MODE_RL:
      rlStep(dt);
      break;
    case MODE_RL_RECOVER:
      rlRecoverStep(dt);
      break;
    default:
      motorOff();
      break;
  }

  // System-ID telemetry: one CSV-in-brackets line per tick at the loop rate.
  if (stream_log) {
    Serial.print("log=["); Serial.print(now / 1000);
    Serial.print(","); Serial.print(phi_angle, 5);
    Serial.print(","); Serial.print(theta_angle, 5);
    Serial.print(","); Serial.print(phi_dot_filt, 4);
    Serial.print(","); Serial.print(theta_dot_filt, 4);
    Serial.print(","); Serial.print(last_V, 4);
    Serial.print(","); Serial.print(as5600_last_raw);
    Serial.println("]");
  }

  // Serial commands
  while (Serial.available()) {
    int ch = Serial.read();
    if (ch == '\r') continue;
    if (ch == '\n') {
      cmd_buf[cmd_len] = '\0';
      handleCommand(cmd_buf);
      cmd_len = 0;
    } else if (ch >= 32 && ch <= 126 && cmd_len < sizeof(cmd_buf)-1) {
      cmd_buf[cmd_len++] = (char)ch;
    }
  }
}

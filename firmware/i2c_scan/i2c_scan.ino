// Quick I2C scanner — checks if AS5600 (0x36) is on the bus
#include <Wire.h>

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  Wire.setClock(100000);
  delay(1000);
  Serial.println("Scanning I2C bus...");
  int found = 0;
  for (int addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.print("  Found device at 0x");
      Serial.print(addr, HEX);
      if (addr == 0x36) Serial.print(" <- AS5600");
      Serial.println();
      found++;
    }
  }
  if (found == 0) Serial.println("  No I2C devices found! Check SDA/SCL wiring.");
  else { Serial.print(found); Serial.println(" device(s) found."); }
}

void loop() { delay(1000); }

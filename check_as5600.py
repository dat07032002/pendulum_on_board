"""
check_as5600.py — live AS5600 readout to verify the pendulum angle scaling.

Run it, then move the rod to KNOWN positions and watch the numbers:
  - It zeroes 'delta' at the position where you start.
  - Rotate the rod by a known amount (e.g. exactly 90 deg) and read 'delta'.
    If delta reads ~90, scaling is correct. If it reads ~180, the AS5600 is
    double-counting (2x scaling) -> a real fault to fix.
  - 'raw' is the 12-bit count (0..4095); a full 360 deg mechanical turn should
    sweep raw by 4096 (i.e. 4096/360 = 11.38 counts per degree).

Press Ctrl-C to stop. Run in your own terminal so you see it update live:
    python check_as5600.py [--port COM5]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import config


def main():
    ap = argparse.ArgumentParser(description="Live AS5600 angle check.")
    ap.add_argument("--port", default=config.PORT)
    args = ap.parse_args()

    print("Move the rod to a known angle and watch 'delta'.")
    print("  full 360 deg turn  -> raw sweeps 4096 (11.38 counts/deg)")
    print("  rotate exactly 90 deg -> delta should read ~90 (not ~180)\n")

    with config.Link(args.port) as link:
        link.stop_motor()
        link.log_on()
        link.drain_until_logging(3.0)
        ref = None
        last_raw = None
        unwrapped = 0.0          # continuous raw, unwrapped across the 0/4095 seam
        try:
            while True:
                d = link.read_log(timeout=0.5)
                if d is None:
                    continue
                raw = d["theta_raw"]
                if last_raw is not None:
                    step = raw - last_raw
                    if step > 2048:
                        step -= 4096
                    elif step < -2048:
                        step += 4096
                    unwrapped += step
                last_raw = raw
                if ref is None:
                    ref = unwrapped
                delta_deg = (unwrapped - ref) * 360.0 / config.AS5600_CPR
                abs_deg = raw * 360.0 / config.AS5600_CPR
                theta_deg = np.rad2deg(d["theta"])     # firmware angle from upright
                sys.stdout.write(
                    f"\r raw={raw:4d}  abs={abs_deg:6.1f}deg  "
                    f"delta={delta_deg:+7.1f}deg  theta(from upright)={theta_deg:+7.1f}deg   "
                )
                sys.stdout.flush()
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        import serial
        if isinstance(e, serial.SerialException):
            print(f"serial error: {e}\nIs the ESP32 on {config.PORT}? Close EncoderServer/monitors.")
            sys.exit(1)
        raise

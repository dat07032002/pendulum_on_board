"""
console.py — interactive serial console for the Furuta firmware (like the old
balance_chip.py). Type firmware commands, watch telemetry live.

    python console.py [--port COM5]

Once it says "ready", type commands and press Enter:
    bal        arm balance, then LIFT the rod to upright to hand off
    s   or 0   STOP the motor (also sent automatically on quit)
    t <V>      manual voltage
    k <4 g>    set LQR gains
    vlim <V>   voltage limit          tr <deg>   theta trim
    hand <deg> handoff window         params     print parameters
    log/nolog  toggle full-rate stream (off by default; on burdens the loop)
    calhang/calup/calfoc/clearcal     calibration
    q          quit (sends stop first)

Safety: 's' is sent on quit and on Ctrl-C. Keep a hand near Enter to stop.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time

import serial

import config


def main():
    ap = argparse.ArgumentParser(description="Interactive serial console.")
    ap.add_argument("--port", default=config.PORT)
    ap.add_argument("--baud", type=int, default=config.BAUD)
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=0.1)
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                raw = ser.readline()
            except Exception:
                break
            if raw:
                sys.stdout.write(raw.decode("ascii", "replace"))
                sys.stdout.flush()

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    print(f"# opening {args.port} @ {args.baud}; waiting out ESP32 boot + FOC calibration...")
    print("# (the motor may twitch during the boot sweep ONLY on first-ever calibration)")
    print("# commands: bal | s | t <V> | k <4> | vlim <V> | tr <d> | hand <d> | params | log | q\n")

    try:
        while True:
            cmd = input()
            if cmd.strip().lower() in ("q", "quit", "exit"):
                break
            ser.write((cmd.strip() + "\n").encode("ascii"))
            ser.flush()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        # always stop the motor on the way out
        try:
            ser.write(b"s\n"); ser.flush(); time.sleep(0.05)
            ser.write(b"s\n"); ser.flush()
        finally:
            stop.set()
            time.sleep(0.2)
            ser.close()
            print("\n# stopped, port closed.")


if __name__ == "__main__":
    try:
        main()
    except serial.SerialException as e:
        print(f"serial error: {e}\nIs the ESP32 on {config.PORT}? Close EncoderServer/other monitors.")
        sys.exit(1)

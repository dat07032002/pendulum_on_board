"""
sign_check.py — Step 3 of system ID: verify the arm->pendulum coupling sign.
** PULSES THE ARM GENTLY (pendulum hanging). ** Ported from main:sign_check.py.

If balance "falls right away", the control sign is often flipped: the arm pushes the
pendulum the wrong way. This pulses a small +arm voltage at the hanging rest and
compares the measured sign of the pendulum's response to what plant_torque predicts.

The expected sign is computed from plant_torque.dynamics() itself (simulate a +V step
from hanging), so this check stays correct if the model changes. If hardware disagrees,
the coupling is flipped -> negate COUPLING in plant_torque.py (the LQR and observer
pick it up automatically when you re-run balance_torque.py).

Usage: python sign_check.py [--port COM5] [--volts 3.0]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import config
import plant_torque as plant

PULSE_S = 0.12          # short: capture the INITIAL response, before the pendulum swings back
EARLY_S = 0.06          # window for the initial-response sign


def predicted_theta_dot_sign(V: float) -> float:
    """Model prediction: sign of the INITIAL theta acceleration at hanging (theta=pi).

    Use the instantaneous theta_ddot (one dynamics eval), NOT an integrated window:
    at hanging the pendulum oscillates (~0.44 s period), so averaging theta_dot over a
    longer pulse flips sign mid-swing and gives the wrong answer.
    """
    xd = plant.dynamics(np.array([0.0, np.pi, 0.0, 0.0]), V)
    return float(np.sign(xd[3]))    # theta_ddot


def main():
    ap = argparse.ArgumentParser(description="Check coupling sign (pulses the arm gently).")
    ap.add_argument("--port", default=config.PORT)
    ap.add_argument("--volts", type=float, default=3.0, help="pulse voltage magnitude")
    args = ap.parse_args()

    pred = predicted_theta_dot_sign(args.volts)
    print("** sign_check pulses the arm. Let the pendulum HANG still first. **")
    print(f"   model predicts: +{args.volts:.1f}V -> theta_dot {'+' if pred>0 else '-'}")

    with config.Link(args.port) as link:
        link.stop_motor()
        link.log_on()
        if not link.drain_until_logging():
            print("No log lines. Is furuta_foc.ino (with 'log') flashed at 921600?")
            return

        # confirm hanging: theta (from upright) sits near +-pi at rest
        base = None
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 1.0:
            d = link.read_log()
            if d is not None:
                base = d
        if base is None or abs(base["theta"]) < 2.8:
            th = None if base is None else np.rad2deg(base["theta"])
            print(f"Pendulum not hanging (theta={th}). Let it settle near +-180 deg and retry.")
            return

        # +V pulse, log the response (short; we only want the INITIAL deflection)
        ts, tds, pds = [], [], []
        link.ser.reset_input_buffer()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < PULSE_S:
            link.torque(+args.volts)
            d = link.read_log()
            if d is not None:
                ts.append(time.perf_counter() - t0)
                tds.append(d["theta_dot"])
                pds.append(d["phi_dot"])
        link.torque(0.0)

    if not tds:
        print("No data captured."); return
    ts = np.array(ts); tds = np.array(tds); pds = np.array(pds)
    early = ts < EARLY_S                          # initial response only
    sel = early if early.sum() >= 3 else np.ones(len(ts), bool)
    td = float(np.mean(tds[sel]))
    pd = float(np.mean(pds[sel]))
    print(f"\n+V command -> mean phi_dot   = {pd:+.2f} rad/s  (motor direction)")
    print(f"+V command -> mean theta_dot = {td:+.2f} rad/s  (coupling response)")

    if abs(td) < 0.15:
        print("=> pendulum barely responded; increase --volts and retry.")
    elif np.sign(td) == pred:
        print("=> coupling sign MATCHES the model. Keep COUPLING positive in plant_torque.py.")
        print("   (so 'falls right away' is NOT a coupling flip -- look at gains/handoff.)")
    else:
        print("=> coupling sign is FLIPPED vs the model. Fix: negate COUPLING in")
        print("   plant_torque.py, then re-run balance_torque.py and sim.py.")
    if pd < 0:
        print("   NOTE: +V gave -phi_dot -> motor/encoder direction is reversed (foc_dir).")

    config.save_sysid({"sign_check": {"theta_dot": td, "phi_dot": pd,
                                      "volts": args.volts, "predicted_sign": pred}})


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")
    except Exception as e:  # noqa: BLE001
        import serial
        if isinstance(e, serial.SerialException):
            print(f"serial error: {e}\nIs the ESP32 on {config.PORT}? Close other monitors.")
            sys.exit(1)
        raise

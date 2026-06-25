"""
actuator_id.py — Step 1-B1: map the commanded-voltage -> torque actuator. ** MOVES MOTOR. **
Run with the arm FREE (rod + AS5600 decoupled).

Two measurements on the free arm:
  1. Low-V terminal-velocity sweep (<=2 V, both directions): steady speed vs V gives
     KM/DAMPING (slope), the deadzone / breakaway voltage (x-intercept), linearity, and
     direction symmetry.
  2. Initial-acceleration pulses (up to +-6 V, both directions, speed-aborted): a0 vs V gives
     KM/J_arm and confirms linearity holds at the higher voltages used for swing-up.

Safety: free arm spins fast at high V, so accel pulses stop the instant |phi_dot| exceeds
SPEED_CAP. Sends 's' on every exit.

Usage: python actuator_id.py [--port COM5]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import config

SPEED_CAP = 30.0   # rad/s — abort accel pulses above this (safety on the free arm)


def terminal(link, V, secs=1.0):
    link.ser.reset_input_buffer()
    pds = []
    t0 = time.time()
    while time.time() - t0 < secs:
        link.torque(V)
        d = link.read_log()
        if d:
            pds.append(abs(d["phi_dot"]))
    link.torque(0.0)
    pds = np.array(pds)
    n = len(pds)
    return float(np.median(pds[int(0.6 * n):])) if n > 8 else 0.0


def initial_accel(link, V, max_t=0.3):
    """Pulse V; capture phi_dot until SPEED_CAP or max_t; fit a0 = slope after breakaway."""
    link.ser.reset_input_buffer()
    ts, pds = [], []
    t0 = time.time()
    while time.time() - t0 < max_t:
        link.torque(V)
        d = link.read_log()
        if d is None:
            continue
        ts.append(time.time() - t0)
        pds.append(abs(d["phi_dot"]))
        if abs(d["phi_dot"]) > SPEED_CAP:
            break
    link.torque(0.0)
    ts = np.array(ts); pds = np.array(pds)
    bk = ts[np.argmax(pds > 0.5)] if np.any(pds > 0.5) else -1
    if bk < 0:
        return 0.0
    mk = (ts >= bk) & (ts < bk + 0.1)
    return float(np.polyfit(ts[mk], pds[mk], 1)[0]) if mk.sum() >= 4 else 0.0


def main():
    ap = argparse.ArgumentParser(description="Actuator torque map (MOVES MOTOR; arm free).")
    ap.add_argument("--port", default=config.PORT)
    args = ap.parse_args()

    print("** actuator_id MOVES THE MOTOR. Arm must be FREE (rod/AS5600 off). **")
    with config.Link(args.port) as link:
        link.stop_motor(); link.log_on(); link.drain_until_logging(2.0)

        # 1) terminal-velocity sweep (low V, both dirs)
        print("\n  terminal velocity:")
        term = {}
        for mag in [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]:
            for sgn in (1, -1):
                V = sgn * mag
                term[V] = terminal(link, V)
                time.sleep(0.6)
            p, m = term[mag], term[-mag]
            a = abs(p - m) / ((p + m) / 2 + 1e-6) * 100
            print(f"    +/-{mag:.1f}V:  +{p:5.2f} / -{m:5.2f} rad/s   asym={a:.0f}%")

        # 2) initial-accel sweep (full range, speed-aborted)
        print("\n  initial acceleration (a0):")
        acc = {}
        for mag in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
            for sgn in (1, -1):
                V = sgn * mag
                acc[V] = initial_accel(link, V)
                time.sleep(0.7)
            print(f"    +/-{mag:.1f}V:  a0 +{acc[mag]:6.0f} / -{acc[-mag]:6.0f} rad/s^2")
        link.torque(0.0); link.log_off()

    # fits (use the average of both directions)
    mags_t = [0.6, 0.8, 1.0, 1.5, 2.0]
    Vt = np.array(mags_t)
    St = np.array([(term[m] + term[-m]) / 2 for m in mags_t])
    moving = St > 0.5
    print("\n===== actuator fit =====")
    if moving.sum() >= 2:
        slope, intc = np.polyfit(Vt[moving], St[moving], 1)
        vdead = -intc / slope
        print(f"  terminal: omega = {slope:.2f}*V {intc:+.2f}  -> KM/DAMPING={slope:.2f} rad/s/V")
        print(f"  deadzone / breakaway V_th = {vdead:.2f} V   (KE_eff = {1/slope:.4f} V*s/rad)")
    mags_a = [2.0, 3.0, 4.0, 5.0, 6.0]
    Va = np.array(mags_a)
    Aa = np.array([(acc[m] + acc[-m]) / 2 for m in mags_a])
    am = Aa > 1
    if am.sum() >= 2:
        sa, ia = np.polyfit(Va[am], Aa[am], 1)
        print(f"  accel: a0 = {sa:.0f}*V {ia:+.0f}  -> KM/J_arm={sa:.0f} rad/s^2/V")
        # linearity check: R^2
        pred = sa * Va[am] + ia
        ss = 1 - np.sum((Aa[am]-pred)**2)/np.sum((Aa[am]-Aa[am].mean())**2)
        print(f"  KM(V) linearity R^2 = {ss:.3f}  (close to 1 = linear, good)")

    config.save_sysid({"actuator_id": {"terminal": {str(k): v for k, v in term.items()},
                                       "accel": {str(k): v for k, v in acc.items()}}})
    print("\nsaved -> sysid.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")

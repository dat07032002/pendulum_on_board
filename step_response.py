"""
step_response.py — measure the arm's actuator response time. ** MOVES THE ARM. **

Settles whether the swing-up's "instant torque" assumption is realistic. Applies a voltage
STEP from 0 and logs phi_dot at full rate, then reports:
  - dead time : delay from command to first motion (includes friction breakaway)
  - tau       : first-order rise time (phi_dot to 63% of its short-term plateau)
  - total settle ~ dead + 3*tau

If dead+tau is ~1-2 control steps (5-10 ms), the actuator is effectively instant at 200 Hz
(the modeled 1-step delay + EMA filter already cover it) -> the 20-50 ms lag / slew ablations
were pessimistic. If it's tens of ms, we must add that lag to the sim before trusting swing-up.

Run with the arm free to move ~90 deg. Pendulum hanging is fine.
    python step_response.py [--port COM5] [--volts 3.0]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import config


def step_once(link, V, max_t=0.4, abort_deg=90.0):
    link.ser.reset_input_buffer()
    s0 = link.capture(0.2)
    phi0 = np.mean([d["phi"] for d in s0])
    tms, pd = [], []
    t0 = time.time()
    while time.time() - t0 < max_t:
        link.torque(V)
        d = link.read_log()
        if d is None:
            continue
        tms.append(d["t_ms"]); pd.append(d["phi_dot"])
        if abs(d["phi"] - phi0) > np.deg2rad(abort_deg):
            break
    link.torque(0.0)
    t = (np.array(tms, float) - tms[0]) * 1e-3
    pd = np.abs(np.array(pd))
    return t, pd


def analyze(t, pd):
    if len(pd) < 8:
        return None
    plateau = np.median(pd[max(1, len(pd) // 2):])      # short-term steady |phi_dot|
    if plateau < 0.5:
        return None
    moved = np.where(pd > 0.1 * plateau)[0]
    dead = t[moved[0]] if len(moved) else float("nan")  # time to first motion
    rise_idx = np.where(pd > 0.63 * plateau)[0]
    tau = (t[rise_idx[0]] - dead) if len(rise_idx) else float("nan")
    return dead, tau, plateau


def main():
    ap = argparse.ArgumentParser(description="Arm actuator step response (MOVES ARM).")
    ap.add_argument("--port", default=config.PORT)
    ap.add_argument("--volts", type=float, default=3.0)
    args = ap.parse_args()

    print(f"** step_response MOVES THE ARM. step to {args.volts} V, x3. **")
    with config.Link(args.port) as link:
        link.stop_motor(); link.log_on(); link.drain_until_logging(3.0)
        res = []
        for i in range(3):
            V = args.volts * (1 if i % 2 == 0 else -1)       # alternate to avoid net wind
            t, pd = step_once(link, V)
            a = analyze(t, pd)
            if a:
                dead, tau, plat = a
                res.append((dead, tau))
                print(f"  step {V:+.1f}V: dead={dead*1e3:4.0f} ms  tau={tau*1e3:4.0f} ms  "
                      f"plateau={plat:.1f} rad/s")
            time.sleep(0.6)
        link.stop_motor(); link.log_off()

    if res:
        dead = np.median([r[0] for r in res]); tau = np.median([r[1] for r in res])
        total = dead + 3 * tau
        print(f"\n  median dead time = {dead*1e3:.0f} ms, tau = {tau*1e3:.0f} ms, "
              f"~settle = {total*1e3:.0f} ms ({total/0.005:.1f} control steps)")
        if total < 0.015:
            print("  => actuator effectively INSTANT at 200 Hz -> lag/slew ablations were pessimistic;")
            print("     the modeled 1-step delay + EMA filter already cover it.")
        else:
            print(f"  => real lag ~{total*1e3:.0f} ms is NOT negligible -> add it to the sim before swing-up.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")

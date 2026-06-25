"""
arm_friction_id.py — Step 1-B2: separate arm Coulomb vs viscous friction. ** MOVES MOTOR. **
Run with the arm FREE (rod + AS5600 decoupled).

Coast-down: spin the arm up, cut power, and watch it decelerate. During coast (omega>0):
    J*dω/dt = -DAMPING*ω - Tc   ->   dω/dt = -(DAMPING/J)*ω - (Tc/J)
So a linear fit of dω/dt vs ω gives slope -(DAMPING/J) [viscous] and intercept -(Tc/J)
[Coulomb], cleanly separating the two. With J_arm from geometry -> DAMPING and Tc.
A slow breakaway ramp cross-checks Tc (= KM * V_breakaway).

Usage: python arm_friction_id.py [--port COM5]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import config
import plant_torque as plant

SPIN_V = 1.5          # spin-up voltage
SPIN_TARGET = 16.0    # rad/s to reach before coasting


def coast_down(link, sgn):
    """Spin up in direction sgn, cut power, capture (t, |omega|) during the coast."""
    # spin up
    t0 = time.time()
    while time.time() - t0 < 0.8:
        link.torque(sgn * SPIN_V)
        d = link.read_log()
        if d and abs(d["phi_dot"]) > SPIN_TARGET:
            break
    # cut power, capture coast (use device t_ms for an accurate, monotonic time axis)
    link.torque(0.0)
    link.ser.reset_input_buffer()
    tms, w = [], []
    t0 = time.time()
    while time.time() - t0 < 2.0:
        d = link.read_log()
        if d is None:
            continue
        tms.append(d["t_ms"])
        w.append(abs(d["phi_dot"]))
        if abs(d["phi_dot"]) < 0.5 and len(w) > 10:
            break
    tms = np.array(tms, float)
    ts = (tms - tms[0]) * 1e-3
    # drop any duplicate timestamps (keeps np.gradient well-defined)
    keep = np.concatenate(([True], np.diff(ts) > 0))
    return ts[keep], np.array(w)[keep]


def fit_coast(ts, w):
    """Fit dω/dt = -a·ω - b over the coast. Returns (a=DAMPING/J, b=Tc/J)."""
    # light smoothing then central difference
    if len(w) < 12:
        return None
    ws = np.convolve(w, np.ones(3) / 3, mode="same")
    dwdt = np.gradient(ws, ts)
    # use the clean decelerating region: ω between 2 and ~max, dω/dt<0
    m = (ws > 2.0) & (ws < ws.max() * 0.95) & (dwdt < 0)
    if m.sum() < 6:
        return None
    slope, intc = np.polyfit(ws[m], dwdt[m], 1)   # dwdt = slope*ω + intc
    return -slope, -intc                          # a = -slope, b = -intc


def breakaway(link, sgn):
    """Slowly ramp |V| from 0; return the V where the arm first sustains motion."""
    link.ser.reset_input_buffer()
    t0 = time.time(); T = 3.0
    while time.time() - t0 < T:
        V = sgn * 1.0 * (time.time() - t0) / T     # ramp 0 -> 1 V over T
        link.torque(V)
        d = link.read_log()
        if d and abs(d["phi_dot"]) > 1.0:
            link.torque(0.0)
            return abs(V)
    link.torque(0.0)
    return float("nan")


def main():
    ap = argparse.ArgumentParser(description="Arm friction coast-down (MOVES MOTOR; arm free).")
    ap.add_argument("--port", default=config.PORT)
    args = ap.parse_args()

    print("** arm_friction_id MOVES THE MOTOR. Arm must be FREE. **")
    with config.Link(args.port) as link:
        link.stop_motor(); link.log_on(); link.drain_until_logging(2.0)
        results = []
        for sgn in (1, -1, 1, -1):
            ts, w = coast_down(link, sgn)
            fit = fit_coast(ts, w)
            if fit:
                results.append(fit)
                print(f"  coast {'+' if sgn>0 else '-'}: peak={w.max():.1f} rad/s  "
                      f"DAMPING/J={fit[0]:.2f} 1/s   Tc/J={fit[1]:.1f} rad/s^2")
            time.sleep(0.5)
        print("  breakaway ramp:")
        bk = [breakaway(link, 1), breakaway(link, -1)]
        for s, v in zip("+-", bk):
            print(f"    {s}: V_break = {v:.2f} V")
        link.torque(0.0); link.log_off()

    if results:
        a = np.median([r[0] for r in results])     # DAMPING/J
        b = np.median([r[1] for r in results])     # Tc/J
        J = plant.J_ARM
        DAMPING = a * J
        Tc = b * J
        print("\n===== arm friction fit =====")
        print(f"  DAMPING/J = {a:.2f} 1/s   Tc/J = {b:.1f} rad/s^2")
        print(f"  with J_arm={J:.2e}: DAMPING = {DAMPING:.3e} N*m*s/rad   Tc = {Tc*1e3:.2f} mN*m")
        print(f"  model DAMPING = {plant.DAMPING:.3e}")
        vb = np.nanmedian(bk)
        if np.isfinite(vb):
            print(f"  breakaway V={vb:.2f}V -> Tc(cross-check) = KM*V = {plant.KM*vb*1e3:.2f} mN*m")
        config.save_sysid({"arm_friction": {"DAMPING_over_J": float(a), "Tc_over_J": float(b),
                                            "DAMPING": float(DAMPING), "Tc": float(Tc),
                                            "V_break": float(vb)}})
        print("\nsaved -> sysid.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")

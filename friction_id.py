"""
friction_id.py — Step 1-A1: robust pendulum friction model (NO MOTOR). Run it yourself:

    python friction_id.py [--port COM5] [--swings 3]

For each swing it prompts you to press Enter, then you lift the rod, hold ~1 s, and release
cleanly (hands off until it settles). Release from a SPREAD of small angles — about
20, 35, 45 deg — and stay under ~50 deg (linear regime).

It auto-detects the release, saves the raw decay to swingN.csv, extracts the alternating
peak amplitudes, and fits across all CLEAN swings:
    A_{n+1} = rho * A_n - C        rho -> VISCOUS (ratio), C -> COULOMB (decrement)
Contaminated swings (non-alternating / non-decreasing peaks, e.g. a re-grab) are rejected
so they can't bias the fit.
"""
from __future__ import annotations

import argparse
import csv
import time

import numpy as np

import config
import freeswing
import plant_torque as plant

HANG = np.pi


def dev_from_hang(theta):
    return (theta - HANG + np.pi) % (2 * np.pi) - np.pi


def capture_one(link, timeout=40.0):
    """After Enter: wait for a steady hold, then a release; return (t, theta_unwrapped)."""
    link.ser.reset_input_buffer()
    hold = None
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = link.read_log()
        if d is None:
            continue
        if abs(np.rad2deg(dev_from_hang(d["theta"]))) > 12 and abs(d["theta_dot"]) < 0.4:
            hold = hold or time.time()
            if time.time() - hold > 0.5:
                break
        else:
            hold = None
    else:
        return None
    t1 = time.time()
    while time.time() - t1 < 15:
        d = link.read_log()
        if d and abs(d["theta_dot"]) > 1.2:
            break
    else:
        return None
    s = link.capture(4.0)
    tms = np.array([x["t_ms"] for x in s], float)
    t = (tms - tms[0]) * 1e-3
    y = freeswing.unwrap_raw(np.array([x["theta_raw"] for x in s], float))
    return t, y


def alternating_peaks(t, y):
    """Extract peaks enforcing sign alternation (rejects spurious same-side detections)."""
    hang = np.median(y[-200:])
    d = np.rad2deg(y - hang)
    pk = []
    for i in range(1, len(d) - 1):
        is_ext = (d[i] > d[i-1] and d[i] >= d[i+1]) or (d[i] < d[i-1] and d[i] <= d[i+1])
        if not is_ext or abs(d[i]) < 1.0:
            continue
        if pk and t[i] - pk[-1][0] < 0.06:
            continue
        if pk and np.sign(d[i]) == np.sign(pk[-1][1]):
            if abs(d[i]) > abs(pk[-1][1]):      # missed a peak: keep the larger same-side
                pk[-1] = (t[i], d[i])
            continue
        pk.append((t[i], d[i]))
    return [tp for tp, _ in pk], [abs(a) for _, a in pk]


def main():
    ap = argparse.ArgumentParser(description="Robust pendulum friction (motor off; run yourself).")
    ap.add_argument("--port", default=config.PORT)
    ap.add_argument("--swings", type=int, default=3)
    args = ap.parse_args()

    targets = [20, 35, 45, 30, 40]
    print("** friction_id: MOTOR STAYS OFF. Clean single releases, under ~50 deg. **")
    pairs, half_periods = [], []
    with config.Link(args.port) as link:
        link.stop_motor(); link.log_on(); link.drain_until_logging(3.0)
        for k in range(args.swings):
            tgt = targets[k] if k < len(targets) else 35
            input(f"\n  swing {k+1}/{args.swings}: press Enter, then lift ~{tgt} deg, hold, release...")
            cap = capture_one(link)
            if cap is None:
                print("    no clean release detected; repeating this swing.");
                continue
            t, y = cap
            with open(f"swing{k+1}.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["t", "theta"]); w.writerows(zip(t, y))
            tms, amps = alternating_peaks(t, y)
            # validate: need >=4 alternating peaks, monotonically decreasing
            ok = len(amps) >= 4 and all(amps[i] > amps[i+1] for i in range(len(amps)-1))
            print(f"    release={amps[0]:.0f} deg  peaks={[round(a,1) for a in amps[:6]]}  "
                  f"{'OK' if ok else 'REJECTED (not clean) -> redo this angle'}")
            if not ok:
                continue
            for i in range(len(amps)-1):
                pairs.append((amps[i], amps[i+1]))
            half_periods += list(np.diff(tms))

    if len(pairs) < 4:
        print("\nNot enough clean peaks. Rerun and release cleanly (hold still, let go, hands off).")
        return
    A = np.array([p[0] for p in pairs]); An = np.array([p[1] for p in pairs])
    rho, negC = np.polyfit(A, An, 1); C = -negC
    per = 2 * np.median([h for h in half_periods if 0.1 < h < 0.5])
    wn = 2 * np.pi / per
    zeta = -np.log(min(max(rho, 1e-3), 0.999)) / np.pi
    b_theta = 2 * zeta * wn * plant.J_P
    Tf = (2 * np.deg2rad(max(C, 0.0))) * plant.M_ROD * plant.L_P * plant.G / 4

    print("\n===== pendulum friction fit =====")
    print(f"  clean pairs: {len(pairs)}   A_(n+1) = {rho:.3f}*A_n - {C:.2f} deg")
    print(f"  period={per*1e3:.0f} ms  alpha={wn**2:.0f}")
    print(f"  VISCOUS: rho={rho:.3f}  zeta={zeta:.4f}  b_theta={b_theta:.3e} N*m*s/rad")
    print(f"  COULOMB: C={C:.2f} deg/half-swing  Tf={Tf*1e3:.2f} mN*m  (deadband ~{np.rad2deg(np.arcsin(min(1,Tf/(plant.M_ROD*plant.L_P*plant.G)))):.1f} deg)")
    config.save_sysid({"friction_id": {"rho": float(rho), "C_deg": float(C), "zeta": float(zeta),
                                       "b_theta": float(b_theta), "Tf": float(Tf), "alpha": float(wn**2),
                                       "n_pairs": len(pairs)}})
    print("saved -> sysid.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")

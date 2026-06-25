"""
arm_id.py — Step 2 of system ID: voltage-step test.  ** MOVES THE MOTOR. **

Identifies the arm subsystem of plant_torque.py from short voltage steps:
    J_arm_eff * phi_ddot = KM * V - DAMPING * phi_dot
First-order step response:  phi_dot(t) = phi_dot_ss * (1 - exp(-t/tau))
  phi_dot_ss = KM/DAMPING * V      (slope of steady speed vs V)
  tau        = J_arm_eff / DAMPING (rise time constant)
  a0         = phi_dot_ss/tau = KM/J_arm * V   (initial accel, robust on a short capture)

The pendulum stays hanging (adds a little inertia but stays ~down at low arm speed).
The arm is fast and cable-wrap-limited, so each step is brief and aborts before the
soft limit; direction alternates so the cable doesn't net-wind.

Separating KM from DAMPING needs one extra fact (the data only gives the ratios).
Measure winding resistance with the bench PSU -- locked rotor, low DC volts across
two phases, R = V/I (phase-pair, ~2x a single winding) -- and pass --R, with the
datasheet --kt. Then DAMPING = kt*kt/R, KM = kt/R, cross-checked against the fits.

Safety: type "go" to start; soft-limit abort; alternate direction; "s" on every exit.

Usage:
    python arm_id.py [--port COM5] [--levels 0.4,0.8,1.2,1.6] [--soft 100]
                     [--step-t 0.8] [--R 5.6] [--kt 0.068] [--yes]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import config


def recenter(link: config.Link, soft_rad: float, drive_V: float = 0.8,
             tol_rad=np.deg2rad(10.0), timeout=6.0) -> bool:
    """Bring the arm back toward phi=0 with a gentle bang-bang, then coast to stop."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        d = link.read_log()
        if d is None:
            continue
        phi = d["phi"]
        if abs(phi) < tol_rad:
            link.torque(0.0)
            time.sleep(0.2)
            return True
        if abs(phi) > soft_rad * 1.5:        # way out: still pull inward, harder
            link.torque(-np.sign(phi) * drive_V * 1.5)
        else:
            link.torque(-np.sign(phi) * drive_V)
        time.sleep(config.CONTROL_DT)
    link.torque(0.0)
    return False


def step_capture(link: config.Link, V: float, soft_rad: float, max_t: float):
    """Apply a constant voltage V; capture (t, phi, phi_dot) until max_t or soft limit.

    Returns arrays; stops early (and commands 0) if |phi| exceeds soft_rad."""
    link.ser.reset_input_buffer()
    ts, phis, pds = [], [], []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < max_t:
        link.torque(V)
        d = link.read_log()
        if d is None:
            continue
        ts.append(time.perf_counter() - t0)
        phis.append(d["phi"])
        pds.append(d["phi_dot"])
        if abs(d["phi"]) > soft_rad:          # cable-wrap safety
            break
    link.torque(0.0)
    return np.array(ts), np.array(phis), np.array(pds)


def fit_step(t: np.ndarray, phi_dot: np.ndarray):
    """Fit phi_dot = vss*(1 - exp(-t/tau)). Returns (vss, tau, a0, ok).

    Falls back to an initial-slope estimate of a0 if the capture is too short to
    pin tau (arm wound out before the speed curved over)."""
    pd = np.abs(phi_dot)
    if len(t) < 6 or t[-1] <= 0:
        return None
    # initial acceleration from the first ~80 ms (robust even on a short capture)
    early = t < min(0.08, t[-1])
    a0_fd = float(np.polyfit(t[early], pd[early], 1)[0]) if early.sum() >= 3 else np.nan
    try:
        from scipy.optimize import curve_fit
        vss0 = float(np.median(pd[len(pd) // 2:])) or float(pd.max())
        popt, _ = curve_fit(lambda tt, vss, tau: vss * (1 - np.exp(-tt / tau)),
                            t, pd, p0=[vss0, max(t[-1] / 3, 0.05)],
                            bounds=([0, 1e-3], [np.inf, 10.0]), maxfev=20000)
        vss, tau = float(popt[0]), float(popt[1])
        a0 = vss / tau
        # trust the steady value only if we actually captured the knee (t spans >~2 tau)
        reached = t[-1] > 1.5 * tau
        return dict(vss=vss, tau=tau, a0=a0, a0_fd=a0_fd, reached=reached)
    except Exception:
        return dict(vss=np.nan, tau=np.nan, a0=a0_fd, a0_fd=a0_fd, reached=False)


def main():
    ap = argparse.ArgumentParser(description="Arm voltage-step ID (MOVES THE MOTOR).")
    ap.add_argument("--port", default=config.PORT)
    ap.add_argument("--levels", default="0.4,0.8,1.2,1.6",
                    help="comma-separated voltage magnitudes to test")
    ap.add_argument("--soft", type=float, default=100.0, help="arm soft limit [deg]")
    ap.add_argument("--step-t", type=float, default=0.8, help="max step duration [s]")
    ap.add_argument("--R", type=float, default=None, help="winding R [ohm] (bench PSU V/I)")
    ap.add_argument("--kt", type=float, default=0.068, help="datasheet torque constant")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    levels = [abs(float(x)) for x in args.levels.split(",") if x.strip()]
    soft_rad = np.deg2rad(args.soft)

    print("** arm_id.py MOVES THE MOTOR. Keep clear; let the pendulum hang. **")
    print("   The arm encoder zero is the boot position; the script keeps the arm")
    print(f"   within +-{args.soft:.0f} deg of it and recenters between steps.")
    print(f"   port={args.port}  test V={levels}")
    if not args.yes and input('   type "go" to start: ').strip().lower() != "go":
        print("aborted."); return

    rows = []  # (V_signed, fit)
    with config.Link(args.port) as link:
        link.stop_motor()
        link.log_on()
        if not link.drain_until_logging():
            print("No log lines. Is furuta_foc.ino (with 'log') flashed at 921600?")
            return
        try:
            for i, mag in enumerate(levels):
                sign = 1.0 if i % 2 == 0 else -1.0
                V = sign * mag
                if not recenter(link, soft_rad):
                    print(f"   V={V:+.2f}  ->  could not recenter; skipping")
                    continue
                t, phi, pd = step_capture(link, V, soft_rad, args.step_t)
                fit = fit_step(t, pd)
                if fit is None:
                    print(f"   V={V:+.2f}  ->  too few samples")
                    continue
                tag = "ss" if fit["reached"] else "accel-only"
                print(f"   V={V:+.2f}  ->  a0={fit['a0']:7.1f} rad/s^2  "
                      f"vss={fit['vss']:6.2f} rad/s  tau={fit['tau']*1e3:5.1f} ms  [{tag}]")
                rows.append((mag, fit))
            recenter(link, soft_rad)
        except KeyboardInterrupt:
            print("\ninterrupted.")

    if len(rows) < 2:
        print("\nNot enough levels captured to fit. Retry attended with more --levels.")
        return

    V = np.array([m for m, _ in rows])
    a0 = np.array([f["a0"] for _, f in rows])
    km_over_J = float(np.polyfit(V, a0, 1)[0])     # a0 = (KM/J_arm) * V

    print("\n===== arm ID fit =====")
    print(f"KM / J_arm  (from a0 vs V)      : {km_over_J:.1f}  (rad/s^2 per V)")

    reached = [(m, f) for m, f in rows if f["reached"]]
    km_over_D = tau_med = None
    if len(reached) >= 2:
        Vr = np.array([m for m, _ in reached])
        vss = np.array([f["vss"] for _, f in reached])
        km_over_D = float(np.polyfit(Vr, vss, 1)[0])    # vss = (KM/DAMPING) * V
        tau_med = float(np.median([f["tau"] for _, f in reached]))
        print(f"KM / DAMPING (from vss vs V)    : {km_over_D:.2f}  (rad/s per V)")
        print(f"tau (median, = J_arm/DAMPING)   : {tau_med*1e3:.1f} ms")
    else:
        print("KM / DAMPING                    : (no steady-state captures; lower --levels")
        print("                                   or raise --soft to let speed settle)")

    out = {"km_over_J": km_over_J}
    if args.R:
        DAMPING = args.kt * args.kt / args.R
        KM = args.kt / args.R
        print(f"\nwith R={args.R} ohm, kt={args.kt}:")
        print(f"  DAMPING = kt^2/R = {DAMPING:.3e} N*m*s/rad")
        print(f"  KM      = kt/R   = {KM:.4f} N*m/V")
        J_from_a0 = KM / km_over_J
        print(f"  J_arm_eff (= KM / (KM/J_arm)) = {J_from_a0:.3e} kg*m^2")
        out.update({"R": args.R, "kt": args.kt, "DAMPING": DAMPING, "KM": KM,
                    "J_arm_eff": J_from_a0})
        if km_over_D:
            print(f"  cross-check KM = (KM/DAMPING)*DAMPING = {km_over_D*DAMPING:.4f} N*m/V "
                  f"(vs {KM:.4f})")
    else:
        print("\nPass --R (bench-PSU winding resistance) and --kt to separate KM and DAMPING.")
    if km_over_D:
        out["km_over_D"] = km_over_D
    if tau_med:
        out["tau"] = tau_med

    config.save_sysid({"arm_id": out})
    print("\nsaved -> sysid.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 - surface serial errors cleanly
        import serial
        if isinstance(e, serial.SerialException):
            print(f"serial error: {e}\nIs the ESP32 on {config.PORT}? Close other monitors.")
            sys.exit(1)
        raise

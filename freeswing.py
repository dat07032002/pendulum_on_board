"""
freeswing.py — Step 1 of system ID: pendulum free-swing test (NO MOTOR).

Measures the two pendulum parameters that set the unstable upright pole:
  alpha = M_ROD*L_P*G / J_P   [1/s^2]  -- gravity / inertia (the dominant A-matrix term)
  b_theta                      [N*m*s/rad] -- viscous damping (absent from the model today)

Procedure (motor stays OFF the whole time):
  1. Let the rod hang dead still.
  2. Run this script; when it says so, lift the rod ~25 deg off hanging and release.
  3. It captures ~6 s of the AS5600 raw angle and fits a damped sinusoid
        y(t) = c + A * exp(-zeta*wn*t) * cos(wd*t + phi)
     giving wn (-> alpha = wn^2), the period, zeta, and b_theta = 2*zeta*wn*J_P.

Why raw counts: near hanging theta (referenced to upright) sits at +-pi, right on
the wrap boundary, so we stream the raw 12-bit count and unwrap it here instead.

Usage:
    python freeswing.py [--port COM5] [--seconds 6] [--save swing.csv]
    python freeswing.py --file swing.csv         # re-fit a saved capture, no hardware
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

import config
import plant_torque as plant


def unwrap_raw(raw: np.ndarray) -> np.ndarray:
    """AS5600 raw counts (0..4095) -> continuous radians, unwrapped."""
    ang = raw.astype(float) * (2.0 * np.pi / config.AS5600_CPR)
    return np.unwrap(ang)


def _initial_guess(t: np.ndarray, y: np.ndarray):
    c = float(np.mean(y))
    yc = y - c
    A = float(np.max(np.abs(yc))) or 1e-3
    # dominant frequency via FFT on the (roughly) uniform samples
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        dt = config.CONTROL_DT
    freqs = np.fft.rfftfreq(len(yc), d=dt)
    mag = np.abs(np.fft.rfft(yc))
    mag[0] = 0.0                       # ignore the DC bin
    f0 = freqs[int(np.argmax(mag))] or 1.0
    wd = 2.0 * np.pi * f0
    return c, A, wd


def fit_damped_sine(t: np.ndarray, y: np.ndarray):
    """Fit y = c + A exp(-zeta*wn*t) cos(wd*t + phi). Returns dict of params."""
    from scipy.optimize import curve_fit

    c0, A0, wd0 = _initial_guess(t, y)

    def model(tt, c, A, sigma, wd, phi):     # sigma = zeta*wn (decay rate)
        return c + A * np.exp(-sigma * tt) * np.cos(wd * tt + phi)

    p0 = [c0, A0, 0.1, wd0, 0.0]
    bounds = ([-np.inf, 0.0, 0.0, 0.1 * wd0, -np.pi],
              [np.inf, 10 * abs(A0) + 1, 50.0, 5.0 * wd0, np.pi])
    popt, _ = curve_fit(model, t, y, p0=p0, bounds=bounds, maxfev=20000)
    c, A, sigma, wd, phi = popt
    wn = float(np.hypot(wd, sigma))          # wn^2 = wd^2 + sigma^2
    zeta = float(sigma / wn) if wn > 0 else 0.0
    resid = y - model(t, *popt)
    rms = float(np.sqrt(np.mean(resid**2)))
    return dict(c=float(c), A=float(A), sigma=float(sigma), wd=float(wd),
                phi=float(phi), wn=wn, zeta=zeta, period=2 * np.pi / wd, rms=rms)


def capture(port: str, seconds: float):
    """Stream the free-swing; returns (t[s], theta_unwrapped[rad])."""
    print("** freeswing: MOTOR STAYS OFF. Let the rod hang dead still. **")
    with config.Link(port) as link:
        link.stop_motor()
        link.log_on()
        if not link.drain_until_logging():
            print("No log lines. Is furuta_foc.ino (with 'log') flashed at 921600?")
            sys.exit(1)
        input("   Rod hanging still? Press Enter, then LIFT ~25 deg and RELEASE...")
        link.ser.reset_input_buffer()
        print(f"   capturing {seconds:.0f}s — release now!")
        samples = link.capture(seconds)
    if len(samples) < 50:
        print(f"Only {len(samples)} samples captured — check the link.")
        sys.exit(1)
    t = np.array([s["t_ms"] for s in samples], dtype=float) * 1e-3
    t -= t[0]
    raw = np.array([s["theta_raw"] for s in samples], dtype=float)
    return t, unwrap_raw(raw)


def load_csv(path: str):
    t, y = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            t.append(float(row["t"]))
            y.append(float(row["theta"]))
    return np.array(t), np.array(y)


def save_csv(path: str, t, y):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "theta"])
        w.writerows(zip(t, y))
    print(f"   raw capture saved -> {path}")


def main():
    ap = argparse.ArgumentParser(description="Pendulum free-swing ID (motor off).")
    ap.add_argument("--port", default=config.PORT)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--save", default=None, help="write the raw capture to a CSV")
    ap.add_argument("--file", default=None, help="re-fit a saved CSV (no hardware)")
    args = ap.parse_args()

    if args.file:
        t, y = load_csv(args.file)
    else:
        t, y = capture(args.port, args.seconds)
        if args.save:
            save_csv(args.save, t, y)

    fit = fit_damped_sine(t, y)
    alpha_meas = fit["wn"] ** 2
    alpha_model = plant.M_ROD * plant.L_P * plant.G / plant.J_P
    b_theta = 2.0 * fit["zeta"] * fit["wn"] * plant.J_P

    print("\n===== free-swing fit =====")
    print(f"samples            : {len(t)} over {t[-1]:.2f}s "
          f"({len(t)/t[-1]:.0f} Hz)   fit RMS = {np.rad2deg(fit['rms']):.2f} deg")
    print(f"period T           : {fit['period']:.4f} s")
    print(f"damped wd          : {fit['wd']:.3f} rad/s")
    print(f"natural wn         : {fit['wn']:.3f} rad/s")
    print(f"damping ratio zeta : {fit['zeta']:.4f}")
    print()
    print(f"alpha (measured)   : {alpha_meas:.2f} 1/s^2   (= wn^2)")
    print(f"alpha (model now)  : {alpha_model:.2f} 1/s^2   (M_ROD*L_P*G / J_P)")
    ratio = alpha_meas / alpha_model if alpha_model else float('nan')
    print(f"  ratio meas/model : {ratio:.3f}")
    if abs(ratio - 1.0) > 0.1:
        J_P_eff = plant.M_ROD * plant.L_P * plant.G / alpha_meas
        print(f"  -> off by >10%. To match, set effective J_P = {J_P_eff:.3e} kg*m^2")
        print(f"     (model J_P = {plant.J_P:.3e}); likely magnet/hub adds inertia.")
    else:
        print("  -> within 10%: geometry-derived J_P is good.")
    print()
    print(f"b_theta (viscous)  : {b_theta:.3e} N*m*s/rad   (= 2*zeta*wn*J_P)")
    print("   model has no pendulum damping yet; add this as B_THETA in plant_torque.py.")

    config.save_sysid({
        "freeswing": {
            "alpha_meas": alpha_meas, "wn": fit["wn"], "wd": fit["wd"],
            "period": fit["period"], "zeta": fit["zeta"], "b_theta": b_theta,
            "fit_rms_deg": float(np.rad2deg(fit["rms"])), "n_samples": len(t),
        }
    })
    print(f"\nsaved -> sysid.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")

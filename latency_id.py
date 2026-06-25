"""
latency_id.py — Step 1-A2: control-loop timing + velocity-filter lag. Run it yourself:

    python latency_id.py [--port COM5]

Two measurements, both from the device clock (t_ms), so no serial-timing confound:
  1. Loop period + jitter: how steadily the firmware runs its 200 Hz (5 ms) tick.
  2. Velocity-filter lag: applies a brief arm step, then compares the RAW position-derivative
     d(phi)/dt to the firmware's filtered phi_dot. The time shift between them (peak cross-
     correlation) is the EMA filter lag. -> the env applies the same delay + EMA so its
     observations match the ESP32.

The pendulum can stay hanging. The motor step is brief and small (cable-limited); you press
Enter to allow it, and it aborts if the arm travels too far.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import config

STEP_V = 1.5
STEP_T = 0.35
ABORT_DEG = 70.0


def loop_timing(link, secs=2.5):
    s = link.capture(secs)
    tms = np.array([d["t_ms"] for d in s], float)
    dt = np.diff(tms)              # ms between consecutive ticks
    dt = dt[dt > 0]
    return dt, len(s) / secs


def step_capture(link):
    link.ser.reset_input_buffer()
    s0 = link.capture(0.2)
    phi0 = np.mean([d["phi"] for d in s0])
    ts, phi, vf = [], [], []
    t0 = time.time()
    while time.time() - t0 < STEP_T:
        link.torque(STEP_V)
        d = link.read_log()
        if d is None:
            continue
        ts.append(d["t_ms"])
        phi.append(d["phi"])
        vf.append(d["phi_dot"])
        if abs(d["phi"] - phi0) > np.deg2rad(ABORT_DEG):
            break
    link.torque(0.0)
    t = (np.array(ts, float) - ts[0]) * 1e-3
    return t, np.array(phi), np.array(vf)


def filter_lag(t, phi, vf):
    """Lag (ms) between raw d(phi)/dt and the firmware's filtered phi_dot."""
    if len(t) < 12:
        return None
    dt = np.median(np.diff(t))
    raw = np.gradient(phi, t)
    # normalize and cross-correlate over the active (moving) region
    m = np.abs(raw) > 0.5 * np.max(np.abs(raw))
    if m.sum() < 6:
        return None
    a = raw - raw.mean(); b = vf - vf.mean()
    xc = np.correlate(b, a, mode="full")
    lag = (np.argmax(xc) - (len(a) - 1))      # samples that vf lags raw
    return lag * dt * 1e3, dt * 1e3           # (lag_ms, sample_period_ms)


def main():
    ap = argparse.ArgumentParser(description="Loop timing + filter lag (brief motor step).")
    ap.add_argument("--port", default=config.PORT)
    args = ap.parse_args()

    with config.Link(args.port) as link:
        link.stop_motor(); link.log_on(); link.drain_until_logging(3.0)

        print("\n  [1] loop timing (idle, no motion)...")
        dt, rate = loop_timing(link)
        print(f"    rate={rate:.0f} Hz   period: mean={dt.mean():.2f}ms  "
              f"std={dt.std():.2f}ms  p95={np.percentile(dt,95):.2f}ms  max={dt.max():.2f}ms")

        input("\n  [2] filter lag: press Enter to apply a brief arm step (it will move)...")
        t, phi, vf = step_capture(link)
        link.log_off()

    res = filter_lag(t, phi, vf)
    print("\n===== latency / filter =====")
    if res:
        lag_ms, per_ms = res
        print(f"  velocity-filter lag = {lag_ms:.1f} ms  (~{lag_ms/per_ms:.1f} control steps)")
        print(f"  (firmware EMA: phi_dot 0.85/0.15, theta_dot 0.5/0.5)")
        config.save_sysid({"latency": {"loop_ms_mean": float(np.median(np.diff((t*1e3)))) if len(t)>2 else 5.0,
                                       "filter_lag_ms": float(lag_ms),
                                       "filter_lag_steps": float(lag_ms/per_ms)}})
        print("  saved -> sysid.json")
    else:
        print("  not enough motion to measure lag; rerun (arm must move during the step).")
    print("\n  -> env model: ~1-step action delay + the firmware EMA filters on the velocities.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")

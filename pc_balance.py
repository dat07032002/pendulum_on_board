"""
pc_balance.py — PC-in-the-loop deployment of the trained TQC policy. ** DRIVES THE MOTOR. **

Reads the firmware log at 200 Hz, builds the SAME obs the policy was trained on, runs the
policy, and sends the voltage back as `t <V>`. Use --dry FIRST (motor stays off) to verify
the obs/action signs match before going live.

Obs (must match rl/furuta_env.py exactly):
    [cos(theta), sin(theta), theta_dot/15, phi/pi (clip +-2), phi_dot/25, prev_action]
    theta = 0 at upright (firmware calhang/calup), action in [-1,1] -> V = 6*action.

Sign convention (verify in --dry): tilt the pole one way and the policy should command the
voltage that would drive the arm UNDER it. Flags --sflip_theta / --sflip_act let you flip if
the hardware convention is mirrored vs the sim (cos is sign-immune; sin/theta_dot/action care).

Safety: --vlim caps voltage (default 4 V); PC-side arm-limit abort at +-160 deg; `s` on exit.
For balance-only de-risk: run --dry, then live with the pole already held near upright.

    python pc_balance.py --dry              # sign check, motor OFF
    python pc_balance.py --vlim 4           # live, capped at 4 V
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import config
from sb3_contrib import TQC

TH_SCALE, PHI_SCALE = 15.0, 25.0
ARM_ABORT = np.deg2rad(160.0)


def main():
    ap = argparse.ArgumentParser(description="PC-in-loop policy deploy (drives motor).")
    ap.add_argument("--port", default=config.PORT)
    ap.add_argument("--model", default="rl/models/fix_sde/best_model.zip")
    ap.add_argument("--vlim", type=float, default=4.0)
    ap.add_argument("--dry", action="store_true", help="motor OFF; just print obs + would-be action")
    ap.add_argument("--sflip_theta", action="store_true", help="flip theta sign (sin & theta_dot)")
    ap.add_argument("--sflip_act", action="store_true", help="flip action/voltage sign")
    args = ap.parse_args()

    s_th = -1.0 if args.sflip_theta else 1.0
    s_a = -1.0 if args.sflip_act else 1.0
    policy = TQC.load(args.model)
    print(f"loaded {args.model}; {'DRY (motor off)' if args.dry else f'LIVE vlim={args.vlim}V'}")

    with config.Link(args.port) as link:
        link.stop_motor(); link.log_on(); link.drain_until_logging(3.0)
        prev_a = 0.0
        t_print = 0.0
        try:
            while True:
                d = link.read_log(timeout=0.1)
                if d is None:
                    continue
                th = s_th * d["theta"]; thd = s_th * d["theta_dot"]
                phi = d["phi"]; phid = d["phi_dot"]
                obs = np.array([np.cos(th), np.sin(th), thd / TH_SCALE,
                                float(np.clip(phi / np.pi, -2, 2)), phid / PHI_SCALE,
                                prev_a], dtype=np.float32)
                a = float(policy.predict(obs, deterministic=True)[0][0])
                a = float(np.clip(a, -1, 1)) * s_a
                V = float(np.clip(a * 6.0, -args.vlim, args.vlim))

                if abs(phi) > ARM_ABORT:
                    link.stop_motor()
                    print(f"\n!! arm {np.rad2deg(phi):.0f} deg > limit -> STOP"); break

                if args.dry:
                    link.torque(0.0)                 # sign-check: do NOT drive
                else:
                    link.torque(V)
                prev_a = a

                if time.time() - t_print > 0.1:      # ~10 Hz console
                    t_print = time.time()
                    print(f"\r th={np.rad2deg(d['theta']):+6.1f} phi={np.rad2deg(phi):+6.0f} "
                          f"thd={d['theta_dot']:+5.1f} | action={a:+.2f} V={V:+.2f}"
                          f"{'  (DRY)' if args.dry else ''}   ", end="", flush=True)
        except KeyboardInterrupt:
            print("\nstopped by user")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        import serial
        if isinstance(e, serial.SerialException):
            print(f"serial error: {e}\nClose EncoderServer/other monitors on {config.PORT}.")
        else:
            raise

"""
view_tilt.py — MuJoCo viewer for the tilting-base Furuta model (confirm physics).

Opens the interactive viewer and runs the hand-coded LQR (balancing to TRUE vertical) while the
board tilts +-30 deg. Watch: the board (grey stand) rocks about its base; the red pole stays
~vertical against gravity; the blue arm swings to compensate. Close the window to stop.

    python rl/view_tilt.py [--betadot 1.0] [--mode random|triangle] [--nopolicy]
"""
from __future__ import annotations

import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer

import feasibility_tilt as F      # reuses the model (M, D), addresses, wrap, SUB, K
from tilt import TiltGenerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betadot", type=float, default=1.0, help="tilt rate cap [rad/s]")
    ap.add_argument("--mode", default="random", choices=["random", "triangle"])
    ap.add_argument("--nopolicy", action="store_true", help="no balance control (raw physics)")
    ap.add_argument("--theta0", type=float, default=None,
                    help="initial pole angle from upright [deg] (0=upright, 180=hanging)")
    ap.add_argument("--phi0", type=float, default=None, help="initial arm angle [deg]")
    args = ap.parse_args()

    # raw-physics default: hanging pole at phi=90 (where the y-tilt strongly drives it) so motion
    # is obvious. NOTE: at phi=0 the tilt is perpendicular to the swing plane -> nothing moves.
    theta0 = args.theta0 if args.theta0 is not None else (180.0 if args.nopolicy else 0.0)
    phi0 = args.phi0 if args.phi0 is not None else (90.0 if args.nopolicy else 0.0)

    M, D, K = F.M, F.D, F.K
    mujoco.mj_resetData(M, D)
    D.qpos[F.PQ] = np.pi - np.deg2rad(theta0)  # pole: pi=upright, 0=hanging
    D.qpos[F.AQ] = np.deg2rad(phi0)            # arm orientation
    mujoco.mj_forward(M, D)
    gen = TiltGenerator(beta_max=np.deg2rad(30.0), betadot_max=args.betadot, dt=F.DT,
                        mode=args.mode, rng=np.random.default_rng(0))
    n_settle = 200
    print(f"viewer: tilt +-30 deg @ {args.betadot} rad/s ({args.mode}), "
          f"{'NO policy (raw physics)' if args.nopolicy else 'LQR balancing to true vertical'}")
    print("close the window to stop.")

    with mujoco.viewer.launch_passive(M, D) as v:
        i = 0
        while v.is_running():
            t0 = time.time()
            phi = D.qpos[F.AQ]; phid = D.qvel[F.AV]
            th = F.wrap(D.qpos[F.PQ] - np.pi); thd = D.qvel[F.PV]
            beta = D.qpos[F.TQ]; betad = D.qvel[F.TV]
            if args.nopolicy:
                D.ctrl[F.MOT] = 0.0
            else:
                th_ref = beta * np.sin(phi); thd_ref = betad * np.sin(phi)
                V = -(K[0] * phi + K[1] * (th - th_ref) + K[2] * phid + K[3] * (thd - thd_ref))
                D.ctrl[F.MOT] = float(np.clip(V, -6.0, 6.0))
            D.ctrl[F.TILT] = 0.0 if i < n_settle else gen.step()
            for _ in range(F.SUB):
                mujoco.mj_step(M, D)
            v.sync()
            i += 1
            dt = F.DT - (time.time() - t0)      # ~real-time playback
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()

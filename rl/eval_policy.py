"""eval_policy.py — INDEPENDENT verification of a trained policy over fresh episodes.

USE THIS to verify any policy. The SB3 EvalCallback success_rate AGREES with this for nominal
(no DR) but badly OVER-REPORTS under DR+tilt (e.g. it claimed 0.82 where this gives 0.27). Trust
this, not the training-log number, for any tilt/DR result. (See SESSION_2026-06-26.md sec 2.1.)

  python rl/eval_policy.py models/nomB_fa_s0/best_model.zip                  # nominal swing-up
  python rl/eval_policy.py models/tilt_fa20_s4/best_model.zip --tilt_deg 20 --dr -n 100  # deployment

Reports success over N fresh-seed episodes for BOTH free-arm and cable (±180°) configs.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from furuta_env import FurutaEnv          # noqa: E402
from sb3_contrib import TQC                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tilt_deg", type=float, default=0.0)   # 0 = nominal; 20/30 = tilt
    ap.add_argument("--dr", action="store_true")             # domain randomization on
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=9000)       # base seed for the fresh episodes
    args = ap.parse_args()

    model = TQC.load(args.model, device="cpu")
    cond = f"tilt={args.tilt_deg}deg, DR={'on' if args.dr else 'off'}, full swing-up"
    print(f"{args.model}  |  N={args.n}  |  {cond}")
    for arm_name, arm_limit in (("free-arm", None), ("cable+-180", np.pi)):
        env = FurutaEnv(randomize=args.dr)
        env.init_angle_max = np.pi
        env.tilt_amp = float(np.deg2rad(args.tilt_deg))
        env.arm_limit = arm_limit
        rews, catches, succ, lens = [], [], [], []
        for ep in range(args.n):
            obs, _ = env.reset(seed=args.seed0 + ep)
            done = trunc = False; R = L = catch = s = 0
            while not (done or trunc):
                a, _ = model.predict(obs, deterministic=True)
                obs, r, done, trunc, info = env.step(a)
                R += r; L += 1
                if done or trunc:
                    catch = int(info.get("is_catch_success", False))
                    s = int(info.get("is_success", False))
            rews.append(R); catches.append(catch); succ.append(s); lens.append(L)
        rews, catches, succ, lens = map(np.array, (rews, catches, succ, lens))
        print(f"  [{arm_name:10s}] sustained={succ.mean():.2f} ({int(succ.sum())}/{args.n})  "
              f"catch={catches.mean():.2f}  "
              f"rew={rews.mean():.0f}+/-{rews.std():.0f}  eplen={lens.mean():.0f}/2000")


if __name__ == "__main__":
    main()

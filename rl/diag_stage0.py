"""
diag_stage0.py — controlled bisect: WHY did the v2 (hardened-env) retrain stall at stage 0?

Stage 0 is balance from +-10 deg with randomize=False, so the ONLY reward difference vs the
working v1 is the arm-envelope penalty (corner-DR / delay live behind `if randomize`). This
trains stage-0-only TQC (same hyperparams as train_tqc.py) for a short budget at a given
arm_envelope_w and prints the rolling eval success every 10k steps. If w=0.5 stays ~0 while
w=0.0 climbs, the arm-envelope is the cause.

    python rl/diag_stage0.py --w 0.5   # v2 (hardened)
    python rl/diag_stage0.py --w 0.0   # v1 baseline
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from furuta_env import FurutaEnv  # noqa: E402
from sb3_contrib import TQC  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback  # noqa: E402


def make_env(w, randomize=False, init_angle=0.17, assist=0.0, free_arm=False):
    def _f():
        e = FurutaEnv(randomize=randomize)    # stage 0: near-nominal; --randomize -> DR like stage 1+
        e.init_angle_max = init_angle         # 0.17 = +-10 deg (stage 0); 0.79 = +-45 deg (stage 1)
        e.init_vel_assist = assist
        e.arm_envelope_w = w
        if free_arm:
            e.arm_limit = None                # remove the +-180deg cable termination (free hinge)
        return Monitor(e, info_keywords=("is_success",))
    return _f


class RollingReport(BaseCallback):
    def __init__(self, every=10000):
        super().__init__(); self.every = every; self._last = 0
    def _on_step(self):
        if self.num_timesteps - self._last < self.every:
            return True
        self._last = self.num_timesteps
        buf = [e for e in self.model.ep_info_buffer if "is_success" in e]
        if buf:
            succ = np.mean([e["is_success"] for e in buf])
            rew = np.mean([e["r"] for e in buf]); ln = np.mean([e["l"] for e in buf])
            print(f"  t={self.num_timesteps:6d}  success={succ:.2f}  ep_rew={rew:7.1f}  "
                  f"ep_len={ln:6.1f}  (n={len(buf)})", flush=True)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--nenv", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_sde", action="store_true")
    ap.add_argument("--ent_coef", default="auto")
    ap.add_argument("--target_entropy", default="auto")
    ap.add_argument("--randomize", action="store_true")          # DR on (mimic stage 1+)
    ap.add_argument("--init_angle", type=float, default=0.17)    # 0.17=stage0, 0.79=stage1
    ap.add_argument("--assist", type=float, default=0.0)
    ap.add_argument("--eval", action="store_true")               # add trainer-style EvalCallback
    ap.add_argument("--eval_tilt_deg", type=float, default=30.0)
    ap.add_argument("--free_arm", action="store_true")           # remove +-180deg cable termination
    args = ap.parse_args()
    ec = args.ent_coef if args.ent_coef == "auto" else float(args.ent_coef)
    te = args.target_entropy if args.target_entropy == "auto" else float(args.target_entropy)
    print(f"=== stage-0 bisect: w={args.w}, {args.steps} steps, nenv={args.nenv}, seed={args.seed}, "
          f"sde={not args.no_sde}, ent_coef={args.ent_coef}, target_entropy={args.target_entropy} ===",
          flush=True)

    venv = VecMonitor(SubprocVecEnv([make_env(args.w, args.randomize, args.init_angle, args.assist,
                                              args.free_arm)
                                     for _ in range(args.nenv)]),
                      info_keywords=("is_success",))
    model = TQC(
        "MlpPolicy", venv,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], qf=[256, 256])),
        learning_rate=3e-4, buffer_size=400_000, batch_size=512,
        gamma=0.998, tau=0.005, train_freq=1, gradient_steps=max(4, args.nenv // 2),
        learning_starts=10_000, ent_coef=ec, target_entropy=te,
        use_sde=not args.no_sde, sde_sample_freq=64, top_quantiles_to_drop_per_net=2,
        device="cuda", verbose=0, seed=args.seed,
    )
    callbacks = [RollingReport()]
    if args.eval:   # mirror train_tqc.py's EvalCallback (eval at +-eval_tilt_deg full swing-up + DR)
        import numpy as _np
        eval_env = VecMonitor(SubprocVecEnv([make_env(args.w, True, _np.pi, 0.0, args.free_arm)]),
                              info_keywords=("is_success",))
        eval_env.set_attr("tilt_amp", float(_np.deg2rad(args.eval_tilt_deg)))
        callbacks.append(EvalCallback(eval_env, eval_freq=20_000 // args.nenv,
                                      n_eval_episodes=20, deterministic=True, verbose=0))
        print(f"  [eval callback ON: +-{args.eval_tilt_deg}deg tilt, every {20_000//args.nenv} steps]",
              flush=True)
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=False)
    print(f"=== done w={args.w} ===", flush=True)


if __name__ == "__main__":
    main()

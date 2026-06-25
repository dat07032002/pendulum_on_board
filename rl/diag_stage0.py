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
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402


def make_env(w):
    def _f():
        e = FurutaEnv(randomize=False)        # stage 0: near-nominal
        e.init_angle_max = 0.17               # +-10 deg
        e.init_vel_assist = 0.0
        e.arm_envelope_w = w
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
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--nenv", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(f"=== stage-0 bisect: arm_envelope_w={args.w}, {args.steps} steps, nenv={args.nenv}, "
          f"seed={args.seed} ===", flush=True)

    venv = VecMonitor(SubprocVecEnv([make_env(args.w) for _ in range(args.nenv)]),
                      info_keywords=("is_success",))
    model = TQC(
        "MlpPolicy", venv,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], qf=[256, 256])),
        learning_rate=3e-4, buffer_size=400_000, batch_size=512,
        gamma=0.998, tau=0.005, train_freq=1, gradient_steps=max(4, args.nenv // 2),
        learning_starts=10_000, ent_coef="auto",
        use_sde=True, sde_sample_freq=64, top_quantiles_to_drop_per_net=2,
        device="cuda", verbose=0, seed=args.seed,
    )
    model.learn(total_timesteps=args.steps, callback=RollingReport(), progress_bar=False)
    print(f"=== done w={args.w} ===", flush=True)


if __name__ == "__main__":
    main()

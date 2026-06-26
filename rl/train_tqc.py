"""
train_tqc.py — Step 5: train a TQC policy for Furuta swing-up + balance, with a curriculum.

  python rl/train_tqc.py [--steps 2000000] [--nenv 8]

Curriculum (advances when rolling success rate > 0.7): start balancing near upright, then
widen the initial tilt, then add energy-pumping from near hanging, then full swing-up from
rest. Domain randomization is on from stage 1 onward (stage 0 near-nominal so balance is
learned cleanly first). Actor net is [64,64] (small enough to port to the ESP32); the TQC
critics can be larger since they're training-only. Best model (by eval success) -> rl/models/.
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
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback  # noqa: E402

HERE = os.path.dirname(__file__)

# (init_angle_max [rad], init_vel_assist, randomize, tilt_amp_deg)
# Stages 0-4 learn the full task on LEVEL ground (as deployed v1), then 5-7 ramp the board tilt in.
STAGES = [
    (0.17,  0.0, False, 0),    # 0: balance ~10 deg, level, near-nominal
    (0.79,  0.0, True,  0),    # 1: catch ~45 deg + DR, level
    (1.57,  0.0, True,  0),    # 2: ~90 deg, level
    (2.60,  1.5, True,  0),    # 3: near hanging + energy-pump assist, level
    (np.pi, 0.0, True,  0),    # 4: full swing-up from rest, level
    (np.pi, 0.0, True,  10),   # 5: full task + gentle +-10 deg random tilt
    (np.pi, 0.0, True,  20),   # 6: + moderate +-20 deg tilt
    (np.pi, 0.0, True,  30),   # 7: + full +-30 deg random tilt (target)
]


def make_env(randomize=True):
    def _f():
        return Monitor(FurutaEnv(randomize=randomize), info_keywords=("is_success",))
    return _f


class Curriculum(BaseCallback):
    # soft gate (0.6) + a per-stage step timeout: a slow-but-fine run can't get TRAPPED below the
    # threshold and then diverge (the v2 post-mortem failure mode). Seeded run for reproducibility.
    def __init__(self, check_every=20000, success_thresh=0.6, min_eps=60, stage_timeout=700_000):
        super().__init__()
        self.check_every = check_every
        self.thresh = success_thresh
        self.min_eps = min_eps
        self.stage_timeout = stage_timeout
        self.stage = 0
        self._last = 0
        self._stage_start = 0

    def _apply(self):
        amax, assist, rand, tilt_deg = STAGES[self.stage]
        self.training_env.set_attr("init_angle_max", amax)
        self.training_env.set_attr("init_vel_assist", assist)
        self.training_env.set_attr("randomize", rand)
        self.training_env.set_attr("tilt_amp", float(np.deg2rad(tilt_deg)))
        print(f"[curriculum] -> stage {self.stage}: init_angle_max={amax:.2f} "
              f"assist={assist} randomize={rand} tilt={tilt_deg}deg", flush=True)

    def _advance(self):
        if self.stage < len(STAGES) - 1:
            self.stage += 1
            self._stage_start = self.num_timesteps
            self._apply()

    def _on_training_start(self):
        self._apply()

    def _on_step(self):
        if self.num_timesteps - self._last < self.check_every:
            return True
        self._last = self.num_timesteps
        buf = [e for e in self.model.ep_info_buffer if "is_success" in e]
        if len(buf) >= self.min_eps:
            rate = np.mean([e["is_success"] for e in buf[-self.min_eps:]])
            stuck = self.num_timesteps - self._stage_start
            print(f"[curriculum] stage {self.stage} success={rate:.2f} "
                  f"(t={self.num_timesteps}, in_stage={stuck})", flush=True)
            if rate > self.thresh:
                self._advance()
            elif stuck > self.stage_timeout:
                print(f"[curriculum] stage {self.stage} TIMEOUT -> advancing anyway", flush=True)
                self._advance()
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--nenv", type=int, default=8)
    ap.add_argument("--tag", default="run")          # per-run output subdir (parallel runs)
    ap.add_argument("--no_sde", action="store_true") # disable gSDE (ablation)
    ap.add_argument("--seed", type=int, default=0)   # reproducibility (post-mortem fix)
    args = ap.parse_args()
    MODELS = os.path.join(HERE, "models", args.tag)
    os.makedirs(MODELS, exist_ok=True)

    venv = VecMonitor(SubprocVecEnv([make_env(True) for _ in range(args.nenv)]),
                      info_keywords=("is_success",))
    eval_env = VecMonitor(SubprocVecEnv([make_env(True)]), info_keywords=("is_success",))

    model = TQC(
        "MlpPolicy", venv,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], qf=[256, 256])),
        learning_rate=3e-4, buffer_size=400_000, batch_size=512,
        gamma=0.998, tau=0.005, train_freq=1, gradient_steps=max(4, args.nenv // 2),  # gamma: ~2.5s horizon @200Hz
        learning_starts=10_000, ent_coef="auto",
        use_sde=not args.no_sde, sde_sample_freq=64,   # gSDE: smoother exploration -> sim-to-real
        top_quantiles_to_drop_per_net=2, seed=args.seed,
        device="cuda", verbose=1, tensorboard_log=os.path.join(HERE, "tb", args.tag),
    )
    callbacks = [
        Curriculum(),
        EvalCallback(eval_env, best_model_save_path=MODELS, log_path=MODELS,
                     eval_freq=20_000 // args.nenv, n_eval_episodes=20, deterministic=True),
        # periodic checkpoints so we can recover the PEAK policy if a run later regresses
        CheckpointCallback(save_freq=max(100_000 // args.nenv, 1), save_path=MODELS,
                           name_prefix="ckpt"),
    ]
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=False)
    model.save(os.path.join(MODELS, "tqc_final"))
    print("done; saved rl/models/tqc_final.zip and best_model.zip", flush=True)


if __name__ == "__main__":
    main()

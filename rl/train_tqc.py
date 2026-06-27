"""
train_tqc.py — Step 5: train a TQC policy for Furuta swing-up + balance, with a curriculum.

  python rl/train_tqc.py [--steps 2000000] [--nenv 8]

IMPORTANT — use nenv=8 (NOT 16). Confirmed twice (v2 post-mortem + a 2026-06-26 stage-0 bisect):
at the same step budget nenv=8 learns markedly faster/more stably. `gradient_steps=max(4,nenv//2)`
means nenv=16 does 8 consecutive updates on a staler buffer snapshot vs 4 at nenv=8 (same 0.5
updates/sample, but bigger/staler blocks -> worse sample efficiency per env-step). Higher nenv is
faster wall-clock data collection but the wrong trade for this small task. (Off-policy quirk: the
PPO "more envs = better" intuition fails because gradient_steps scales with nenv.)

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


def make_env(randomize=True, free_arm=False):
    def _f():
        e = FurutaEnv(randomize=randomize)
        if free_arm:
            e.arm_limit = None          # remove the +-180deg cable termination (sim-only ceiling probe)
        return Monitor(e, info_keywords=("is_success",))
    return _f


class Curriculum(BaseCallback):
    # soft gate (0.6) + a per-stage step timeout: a slow-but-fine run can't get TRAPPED below the
    # threshold and then diverge (the v2 post-mortem failure mode). Seeded run for reproducibility.
    def __init__(self, check_every=20000, success_thresh=0.6, min_eps=60, stage_timeout=700_000,
                 start_stage=0, max_stage=None, force_no_dr=False, no_plant_dr=False):
        super().__init__()
        self.check_every = check_every
        self.thresh = success_thresh
        self.min_eps = min_eps
        self.stage_timeout = stage_timeout
        self.stage = start_stage
        self.max_stage = (len(STAGES) - 1) if max_stage is None else max_stage
        self.force_no_dr = force_no_dr   # Phase A: DR + tilt OFF for ALL stages (nominal pretrain)
        self.no_plant_dr = no_plant_dr   # tilt phase: keep TILT but force plant-DR off (no randomize)
        self._last = 0
        self._stage_start = 0

    def _apply(self):
        amax, assist, rand, tilt_deg = STAGES[self.stage]
        if self.force_no_dr:             # nominal pretrain: never enable DR/tilt
            rand, tilt_deg = False, 0
        elif self.no_plant_dr:           # tilt without plant-DR: keep tilt_deg, drop randomize
            rand = False
        # env_method (NOT set_attr — set_attr writes to the Monitor wrapper, never reaches FurutaEnv)
        self.training_env.env_method("set_params", init_angle_max=amax, init_vel_assist=assist,
                                     randomize=rand, tilt_amp=float(np.deg2rad(tilt_deg)))
        print(f"[curriculum] -> stage {self.stage}: init_angle_max={amax:.2f} "
              f"assist={assist} randomize={rand} tilt={tilt_deg}deg", flush=True)

    def _advance(self):
        if self.stage < self.max_stage:
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


class StopOnSuccess(BaseCallback):
    """Stop training once the EvalCallback's eval success_rate reaches a threshold -> capture the
    peak and skip the post-peak collapse. Used as EvalCallback(callback_after_eval=...), so
    self.parent is the EvalCallback and self.parent._is_success_buffer holds the last eval's flags.
    Only fires at the FINAL curriculum stage (don't stop early on an easy intermediate stage)."""
    def __init__(self, threshold, max_stage):
        super().__init__()
        self.threshold = threshold
        self.max_stage = max_stage
        self.curriculum = None        # set in main() so we can check the current stage

    def _on_step(self):
        if self.curriculum is not None and self.curriculum.stage < self.max_stage:
            return True               # not at the target stage yet -> keep training
        buf = getattr(self.parent, "_is_success_buffer", [])
        if len(buf) > 0:
            sr = float(np.mean(buf))
            if sr >= self.threshold:
                print(f"[stop_success] eval success_rate {sr:.2f} >= {self.threshold} at "
                      f"{self.num_timesteps} steps (stage {getattr(self.curriculum,'stage','?')}) "
                      f"-> stopping", flush=True)
                return False          # stops training
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--nenv", type=int, default=8)
    ap.add_argument("--tag", default="run")          # per-run output subdir (parallel runs)
    # gSDE OFF by default: with auto target-entropy it let ent_coef run away (~0.77) -> entropy
    # collapse -> stage-0 stalled below the 0.6 gate. The 2026-06-26 diag sweep confirmed --no_sde
    # crosses 0.6 @130k and holds 1.00. `--sde` re-enables it (the gSDE contrast seed).
    ap.add_argument("--no_sde", dest="use_sde", action="store_false", default=False)  # default
    ap.add_argument("--sde", dest="use_sde", action="store_true")  # enable gSDE (contrast run)
    ap.add_argument("--seed", type=int, default=0)   # reproducibility (post-mortem fix)
    ap.add_argument("--ent_coef", default="auto")    # "auto" or a float (Phase B: fixed e.g. 0.05)
    ap.add_argument("--target_entropy", default="auto")  # "auto"(=-act_dim) or a float (e.g. -2)
    ap.add_argument("--eval_tilt_deg", type=float, default=30.0)  # eval at deployment tilt
    ap.add_argument("--lr", type=float, default=3e-4)        # Phase B fine-tune: lower (1e-4)
    # --- two-phase warm-start ---
    ap.add_argument("--no_dr", action="store_true")         # Phase A: DR+tilt OFF, stages 0-4 only
    ap.add_argument("--no_plant_dr", action="store_true")   # tilt phase: TILT on, plant-DR off
    ap.add_argument("--warmstart", default=None)            # Phase B: load policy weights from this .zip
    ap.add_argument("--start_stage", type=int, default=0)   # Phase B: resume curriculum at this stage
    ap.add_argument("--max_stage", type=int, default=None)  # last stage to reach (no_dr -> 4)
    ap.add_argument("--free_arm", action="store_true")      # remove +-180 cable limit (ceiling probe)
    ap.add_argument("--arm_center_w", type=float, default=0.20)  # arm-centering weight; LOW (~0.02)
    ap.add_argument("--stop_success", type=float, default=None)  # early-stop at this eval success rate
    ap.add_argument("--n_eval", type=int, default=50)            # eval episodes (more = less noisy stop)
    ap.add_argument("--tqd", type=int, default=2)               # top_quantiles_to_drop (3=more conservative
                                                                # critic, fights overestimation collapse)
    args = ap.parse_args()                                       # frees the arm to pump for swing-up
    ent_coef = args.ent_coef if args.ent_coef == "auto" else float(args.ent_coef)
    target_entropy = args.target_entropy if args.target_entropy == "auto" else float(args.target_entropy)
    max_stage = args.max_stage if args.max_stage is not None else (4 if args.no_dr else len(STAGES) - 1)
    MODELS = os.path.join(HERE, "models", args.tag)
    os.makedirs(MODELS, exist_ok=True)

    venv = VecMonitor(SubprocVecEnv([make_env(True, args.free_arm) for _ in range(args.nenv)]),
                      info_keywords=("is_success",))
    # EVAL CONDITION: Phase B -> deployment (full swing-up + +-eval_tilt_deg random tilt + DR);
    # Phase A (--no_dr) -> nominal full swing-up, level, no DR. best_model.zip selected on this.
    # eval plant-DR: off for nominal (--no_dr) and tilt-no-DR (--no_plant_dr); on otherwise.
    eval_dr = not (args.no_dr or args.no_plant_dr)
    eval_env = VecMonitor(SubprocVecEnv([make_env(eval_dr, args.free_arm)]),
                          info_keywords=("is_success",))
    # env_method (NOT set_attr — see Curriculum._apply / FurutaEnv.set_params)
    eval_env.env_method("set_params", init_angle_max=float(np.pi),
                        tilt_amp=float(np.deg2rad(0.0 if args.no_dr else args.eval_tilt_deg)),
                        arm_center_w=args.arm_center_w)
    venv.env_method("set_params", arm_center_w=args.arm_center_w)

    model = TQC(
        "MlpPolicy", venv,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], qf=[256, 256])),
        learning_rate=args.lr, buffer_size=400_000, batch_size=512,
        gamma=0.998, tau=0.005, train_freq=1, gradient_steps=max(4, args.nenv // 2),  # gamma: ~2.5s horizon @200Hz
        learning_starts=10_000, ent_coef=ent_coef, target_entropy=target_entropy,
        use_sde=args.use_sde, sde_sample_freq=64,   # default OFF (entropy-collapse fix; see args)
        top_quantiles_to_drop_per_net=args.tqd, seed=args.seed,
        device="cuda", verbose=1, tensorboard_log=os.path.join(HERE, "tb", args.tag),
    )
    if args.warmstart:   # Phase B: copy actor+critic weights from the Phase-A nominal master
        src = TQC.load(args.warmstart, device="cuda")
        model.policy.load_state_dict(src.policy.state_dict())
        del src
        print(f"[warmstart] loaded policy weights from {args.warmstart} "
              f"(lr={args.lr}, ent_coef={ent_coef}, start_stage={args.start_stage})", flush=True)
    curriculum = Curriculum(start_stage=args.start_stage, max_stage=max_stage,
                            force_no_dr=args.no_dr, no_plant_dr=args.no_plant_dr)
    stop_cb = None
    if args.stop_success is not None:
        stop_cb = StopOnSuccess(args.stop_success, max_stage)
        stop_cb.curriculum = curriculum     # so it only fires at the final stage
        print(f"[stop_success] will stop when eval success_rate >= {args.stop_success} "
              f"at stage {max_stage} (n_eval={args.n_eval})", flush=True)
    callbacks = [
        curriculum,
        EvalCallback(eval_env, best_model_save_path=MODELS, log_path=MODELS,
                     eval_freq=20_000 // args.nenv, n_eval_episodes=args.n_eval,
                     deterministic=True, callback_after_eval=stop_cb),
        # periodic checkpoints so we can recover the PEAK policy if a run later regresses
        CheckpointCallback(save_freq=max(100_000 // args.nenv, 1), save_path=MODELS,
                           name_prefix="ckpt"),
    ]
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=False)
    model.save(os.path.join(MODELS, "tqc_final"))
    print("done; saved rl/models/tqc_final.zip and best_model.zip", flush=True)


if __name__ == "__main__":
    main()

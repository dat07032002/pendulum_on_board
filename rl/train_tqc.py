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
critics can be larger since they're training-only. `best_success_model.zip` is selected by
sustained eval success; SB3's `best_model.zip` remains the best by mean reward.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from furuta_env import FurutaEnv  # noqa: E402
from residual_env import ResidualActionWrapper  # noqa: E402

from sb3_contrib import TQC  # noqa: E402
from retention_tqc import RetentionTQC  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback  # noqa: E402

HERE = os.path.dirname(__file__)

# (init_angle_max [rad], init_vel_assist, randomize, tilt_deg, dr_probability, dr_scale)
# Phase C starts at stage 4 with the target +-20deg tilt already active, then ramps only plant DR.
# Non-DR episodes remain clean +-20deg rehearsal, so robustification cannot erase the tilt skill.
STAGES = [
    (0.17,  0.0, False, 0,  0.00, 0.00),
    (0.79,  0.0, True,  0,  1.00, 1.00),
    (1.57,  0.0, True,  0,  1.00, 1.00),
    (2.60,  1.5, True,  0,  1.00, 1.00),
    (np.pi, 0.0, False, 20, 0.00, 0.00),  # 4: clean target tilt
    (np.pi, 0.0, True,  20, 0.10, 0.25),  # 5: light DR, 90% clean rehearsal
    (np.pi, 0.0, True,  20, 0.25, 0.50),  # 6: moderate DR
    (np.pi, 0.0, True,  20, 0.50, 0.75),  # 7: strong DR
    (np.pi, 0.0, True,  20, 0.75, 1.00),  # 8: mostly full-range DR
    (np.pi, 0.0, True,  20, 0.90, 1.00),  # 9: target mix, 10% clean rehearsal
]


def make_env(randomize=True, free_arm=False, residual_base=None, residual_scale=0.05):
    def _f():
        e = FurutaEnv(randomize=randomize)
        if free_arm:
            e.arm_limit = None          # remove the +-180deg cable termination (sim-only ceiling probe)
        if residual_base:
            e = ResidualActionWrapper(e, residual_base, residual_scale)
        return Monitor(e, info_keywords=("is_success", "is_catch_success"))
    return _f


class Curriculum(BaseCallback):
    # soft gate (0.6) + a per-stage step timeout: a slow-but-fine run can't get TRAPPED below the
    # threshold and then diverge (the v2 post-mortem failure mode). Seeded run for reproducibility.
    def __init__(self, check_every=20000, success_thresh=0.8, min_eps=60, stage_timeout=700_000,
                 min_stage_steps=100_000, passes_required=2, start_stage=0, max_stage=None,
                 force_no_dr=False, no_plant_dr=False):
        super().__init__()
        self.check_every = check_every
        self.thresh = success_thresh
        self.min_eps = min_eps
        self.stage_timeout = stage_timeout
        self.min_stage_steps = min_stage_steps
        self.passes_required = passes_required
        self.stage = start_stage
        self.max_stage = (len(STAGES) - 1) if max_stage is None else max_stage
        self.force_no_dr = force_no_dr   # Phase A: DR + tilt OFF for ALL stages (nominal pretrain)
        self.no_plant_dr = no_plant_dr   # tilt phase: keep TILT but force plant-DR off (no randomize)
        self._last = 0
        self._stage_start = 0
        self._consecutive_passes = 0

    def _apply(self):
        amax, assist, rand, tilt_deg, dr_probability, dr_scale = STAGES[self.stage]
        if self.force_no_dr:             # nominal pretrain: never enable DR/tilt
            rand, tilt_deg, dr_probability, dr_scale = False, 0, 0.0, 0.0
        elif self.no_plant_dr:           # tilt without plant-DR: keep tilt_deg, drop randomize
            rand, dr_probability, dr_scale = False, 0.0, 0.0
        # env_method (NOT set_attr — set_attr writes to the Monitor wrapper, never reaches FurutaEnv)
        self.training_env.env_method("set_params", init_angle_max=amax, init_vel_assist=assist,
                                     randomize=rand, tilt_amp=float(np.deg2rad(tilt_deg)),
                                     dr_probability=dr_probability, dr_scale=dr_scale)
        print(f"[curriculum] -> stage {self.stage}: init_angle_max={amax:.2f} "
              f"assist={assist} randomize={rand} tilt={tilt_deg}deg "
              f"dr_prob={dr_probability:.2f} dr_scale={dr_scale:.2f}", flush=True)

    def _advance(self):
        if self.stage < self.max_stage:
            self.stage += 1
            self._stage_start = self.num_timesteps
            self._consecutive_passes = 0
            self.model.ep_info_buffer.clear()
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
                  f"(t={self.num_timesteps}, in_stage={stuck}, "
                  f"passes={self._consecutive_passes}/{self.passes_required})", flush=True)
            if stuck < self.min_stage_steps:
                return True
            if rate >= self.thresh:
                self._consecutive_passes += 1
            else:
                self._consecutive_passes = 0
            if self._consecutive_passes >= self.passes_required:
                self._advance()
            elif stuck > self.stage_timeout:
                print(f"[curriculum] stage {self.stage} TIMEOUT -> stopping seed", flush=True)
                return False
        return True


class SaveBestSuccessAndStop(BaseCallback):
    """Save by sustained success after every eval; stop at threshold only on the final stage."""
    def __init__(self, save_path, threshold, max_stage, min_timesteps=0):
        super().__init__()
        self.save_path = save_path
        self.threshold = threshold
        self.max_stage = max_stage
        self.min_timesteps = int(min_timesteps)
        self.best_success = -np.inf
        self.curriculum = None        # set in main() so we can check the current stage

    def _on_step(self):
        buf = getattr(self.parent, "_is_success_buffer", [])
        if len(buf) > 0:
            sr = float(np.mean(buf))
            eligible = self.num_timesteps >= self.min_timesteps
            if eligible and sr > self.best_success:
                self.best_success = sr
                self.model.save(os.path.join(self.save_path, "best_success_model"))
                print(f"[best_success] saved success_rate={sr:.2f} at "
                      f"{self.num_timesteps} steps", flush=True)
            if self.curriculum is not None and self.curriculum.stage < self.max_stage:
                return True
            if not eligible:
                return True
            if sr >= self.threshold:
                print(f"[stop_success] eval success_rate {sr:.2f} >= {self.threshold} at "
                      f"{self.num_timesteps} steps (stage {getattr(self.curriculum,'stage','?')}) "
                      f"-> stopping", flush=True)
                return False          # stops training
        return True


class StopOnRetentionDrop(BaseCallback):
    """Abort a fine-tune before it overwrites the clean-tilt skill."""
    def __init__(self, clean_floor):
        super().__init__()
        self.clean_floor = float(clean_floor)

    def _on_step(self):
        buf = getattr(self.parent, "_is_success_buffer", [])
        if buf:
            success = float(np.mean(buf))
            print(f"[retention_eval] clean_success={success:.2f}", flush=True)
            if success < self.clean_floor:
                print(
                    f"[retention_guard] clean_success={success:.2f} < "
                    f"{self.clean_floor:.2f} -> stopping",
                    flush=True,
                )
                return False
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
    ap.add_argument("--retention", action="store_true")         # full-model resume + balanced teacher replay
    ap.add_argument("--residual_base", default=None,
                    help="Freeze this policy and train only a bounded action residual")
    ap.add_argument("--residual_scale", type=float, default=0.05,
                    help="Maximum absolute residual added to the frozen base action")
    ap.add_argument("--teacher_data", default=None)             # fixed clean-tilt replay .npz
    ap.add_argument("--learning_starts", type=int, default=10_000)
    ap.add_argument("--actor_start", type=int, default=100_000) # critic-only adaptation before this step
    ap.add_argument("--actor_lr", type=float, default=1e-6)
    ap.add_argument("--critic_lr", type=float, default=3e-5)
    ap.add_argument("--teacher_coef", type=float, default=100.0)
    ap.add_argument("--teacher_fraction", type=float, default=0.5)
    ap.add_argument("--clean_floor", type=float, default=0.60)
    ap.add_argument("--p_corner", type=float, default=0.10,
                    help="Probability that each DR draw is forced to a range endpoint")
    ap.add_argument("--tilt_amp_min_fraction", type=float, default=0.30,
                    help="Training-only minimum episode tilt amplitude as a fraction of the cap")
    ap.add_argument("--tilt_rate_min", type=float, default=0.50,
                    help="Training-only minimum episode tilt-rate cap [rad/s]")
    args = ap.parse_args()                                       # frees the arm to pump for swing-up
    if args.retention and (not args.warmstart or not args.teacher_data):
        ap.error("--retention requires --warmstart and --teacher_data")
    if args.retention and args.residual_base:
        ap.error("--retention and --residual_base are separate methods; choose one")
    if args.residual_base and args.warmstart:
        ap.error("--residual_base freezes the master; do not also pass --warmstart")
    if not 0.0 < args.residual_scale <= 0.15:
        ap.error("--residual_scale must be in (0, 0.15]")
    if not 0.0 <= args.p_corner <= 1.0:
        ap.error("--p_corner must be between 0 and 1")
    if not 0.0 <= args.tilt_amp_min_fraction <= 1.0:
        ap.error("--tilt_amp_min_fraction must be between 0 and 1")
    if not 0.0 <= args.tilt_rate_min <= 2.0:
        ap.error("--tilt_rate_min must be between 0 and 2 rad/s")
    ent_coef = args.ent_coef if args.ent_coef == "auto" else float(args.ent_coef)
    target_entropy = args.target_entropy if args.target_entropy == "auto" else float(args.target_entropy)
    max_stage = args.max_stage if args.max_stage is not None else (4 if args.no_dr else len(STAGES) - 1)
    MODELS = os.path.join(HERE, "models", args.tag)
    os.makedirs(MODELS, exist_ok=True)

    venv = VecMonitor(SubprocVecEnv([
        make_env(True, args.free_arm, args.residual_base, args.residual_scale)
        for _ in range(args.nenv)
    ]),
                      info_keywords=("is_success", "is_catch_success"))
    # EVAL CONDITION: Phase B -> deployment (full swing-up + +-eval_tilt_deg random tilt + DR);
    # Phase A (--no_dr) -> nominal full swing-up, level, no DR. best_model.zip selected on this.
    # eval plant-DR: off for nominal (--no_dr) and tilt-no-DR (--no_plant_dr); on otherwise.
    eval_dr = not (args.no_dr or args.no_plant_dr)
    eval_env = VecMonitor(SubprocVecEnv([
        make_env(eval_dr, args.free_arm, args.residual_base, args.residual_scale)
    ]),
                          info_keywords=("is_success", "is_catch_success"))
    # env_method (NOT set_attr — see Curriculum._apply / FurutaEnv.set_params)
    eval_env.env_method("set_params", init_angle_max=float(np.pi),
                        tilt_amp=float(np.deg2rad(0.0 if args.no_dr else args.eval_tilt_deg)),
                        arm_center_w=args.arm_center_w,
                        dr_probability=1.0, dr_scale=1.0, p_corner=args.p_corner)
    venv.env_method(
        "set_params",
        arm_center_w=args.arm_center_w,
        p_corner=args.p_corner,
        tilt_amp_min_fraction=args.tilt_amp_min_fraction,
        tilt_rate_min=args.tilt_rate_min,
    )
    print(f"[domain_randomization] p_corner={args.p_corner:.2f}", flush=True)
    print(
        f"[tilt_training] amp_fraction={args.tilt_amp_min_fraction:.2f}-1.00 "
        f"rate_cap={args.tilt_rate_min:.2f}-2.00 rad/s",
        flush=True,
    )

    if args.retention:
        # Load the complete model so Adam moments and entropy state survive. The old warm-start
        # path copied only neural-network weights and silently discarded optimizer state.
        model = RetentionTQC.load(args.warmstart, env=venv, device="cuda")
        model.tensorboard_log = os.path.join(HERE, "tb", args.tag)
        model.verbose = 1
        model.learning_starts = args.learning_starts
        model.batch_size = 512
        model.gradient_steps = max(4, args.nenv // 2)
        model.seed = args.seed
        model.set_random_seed(args.seed)
        model.configure_retention(
            args.teacher_data,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            actor_start_steps=args.actor_start,
            teacher_coef=args.teacher_coef,
            teacher_fraction=args.teacher_fraction,
        )
        print(
            f"[warmstart] resumed full model+optimizers from {args.warmstart} "
            f"(learning_starts={model.learning_starts}, start_stage={args.start_stage})",
            flush=True,
        )
    else:
        model = TQC(
            "MlpPolicy", venv,
            policy_kwargs=dict(net_arch=dict(pi=[64, 64], qf=[256, 256])),
            learning_rate=args.lr, buffer_size=400_000, batch_size=512,
            gamma=0.998, tau=0.005, train_freq=1, gradient_steps=max(4, args.nenv // 2),
            learning_starts=args.learning_starts, ent_coef=ent_coef, target_entropy=target_entropy,
            use_sde=args.use_sde, sde_sample_freq=64,
            top_quantiles_to_drop_per_net=args.tqd, seed=args.seed,
            device="cuda", verbose=1, tensorboard_log=os.path.join(HERE, "tb", args.tag),
        )
        if args.warmstart:
            src = TQC.load(args.warmstart, device="cuda")
            model.policy.load_state_dict(src.policy.state_dict())
            del src
            print(f"[warmstart] loaded policy weights from {args.warmstart} "
                  f"(lr={args.lr}, ent_coef={ent_coef}, start_stage={args.start_stage})", flush=True)
    curriculum = Curriculum(start_stage=args.start_stage, max_stage=max_stage,
                            force_no_dr=args.no_dr, no_plant_dr=args.no_plant_dr)
    stop_cb = None
    if args.stop_success is not None:
        min_save_steps = args.actor_start if args.retention else 0
        if args.residual_base:
            min_save_steps = args.learning_starts + 20_000
        stop_cb = SaveBestSuccessAndStop(
            MODELS,
            args.stop_success,
            max_stage,
            min_timesteps=min_save_steps,
        )
        stop_cb.curriculum = curriculum     # so it only fires at the final stage
        print(f"[stop_success] will stop when eval success_rate >= {args.stop_success} "
              f"at stage {max_stage} (n_eval={args.n_eval})", flush=True)
    callbacks = [
        curriculum,
        EvalCallback(eval_env, best_model_save_path=MODELS, log_path=MODELS,
                     eval_freq=20_000 // args.nenv, n_eval_episodes=args.n_eval,
                     deterministic=True, callback_after_eval=stop_cb),
    ]
    # In a clean no-plant-DR continuation run, the primary evaluator already is the clean
    # retention condition. Running a second identical callback doubles evaluation cost and can
    # falsely stop a healthy seed on a noisy duplicate sample.
    if (args.retention and not args.no_plant_dr) or args.residual_base:
        clean_eval_env = VecMonitor(
            SubprocVecEnv([
                make_env(False, args.free_arm, args.residual_base, args.residual_scale)
            ]),
            info_keywords=("is_success", "is_catch_success"),
        )
        clean_eval_env.env_method(
            "set_params",
            init_angle_max=float(np.pi),
            tilt_amp=float(np.deg2rad(20.0)),
            arm_center_w=args.arm_center_w,
            randomize=False,
            dr_probability=0.0,
            dr_scale=0.0,
        )
        clean_guard = StopOnRetentionDrop(args.clean_floor)
        callbacks.append(
            EvalCallback(
                clean_eval_env,
                best_model_save_path=os.path.join(MODELS, "clean"),
                log_path=os.path.join(MODELS, "clean"),
                eval_freq=20_000 // args.nenv,
                n_eval_episodes=args.n_eval,
                deterministic=True,
                callback_after_eval=clean_guard,
            )
        )
    elif args.retention:
        print(
            "[retention_eval] skipped duplicate clean evaluator; primary eval is already no-plant-DR",
            flush=True,
        )
    callbacks.append(
        CheckpointCallback(
            save_freq=max(100_000 // args.nenv, 1),
            save_path=MODELS,
            name_prefix="ckpt",
        )
    )
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=False)
    model.save(os.path.join(MODELS, "tqc_final"))
    print("done; saved tqc_final.zip, best_model.zip, and best_success_model.zip", flush=True)


if __name__ == "__main__":
    main()

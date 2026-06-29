"""Independent, deterministic verification of a trained tilt-policy checkpoint.

This evaluator deliberately reports more than a binary success rate. It records the evidence
needed by POLICY_RUBRIC.md: sustained/catch success, true-vertical quality, action smoothness and
saturation, arm/cable margin, exposure near the difficult phi=+-90 degree orientation, realized
tilt motion, and the actual randomized plant parameters.

Use at least 500 fresh episodes for model-selection claims. ``--compare`` evaluates two models on
identical seeds and reports the paired sustained-success difference. ``--save_npz`` preserves all
per-episode measurements for later inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from furuta_env import DT, FurutaEnv  # noqa: E402
from residual_env import ResidualActionWrapper  # noqa: E402
from sb3_contrib import TQC  # noqa: E402


FINAL_WINDOW_STEPS = int(round(2.0 / DT))


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_interval(successes: int, episodes: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Two-sided 95% Wilson score interval without a SciPy dependency."""
    if episodes <= 0:
        return np.nan, np.nan
    p = successes / episodes
    denom = 1.0 + z * z / episodes
    center = (p + z * z / (2.0 * episodes)) / denom
    half = z * np.sqrt(p * (1.0 - p) / episodes + z * z / (4.0 * episodes**2)) / denom
    return center - half, center + half


def mean_sd(values) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return (float(np.mean(a)), float(np.std(a))) if len(a) else (np.nan, np.nan)


def first_catch_start(balanced: np.ndarray) -> int | None:
    """Start of the first run satisfying the environment's strict >0.5 s catch definition."""
    run = 0
    for index, value in enumerate(balanced):
        run = run + 1 if value else 0
        if run * DT > 0.5:
            return index - run + 1
    return None


def evaluate_arm(model: TQC, args, arm_name: str, arm_limit: float | None) -> dict[str, np.ndarray]:
    base_env = FurutaEnv(randomize=args.dr)
    base_env.init_angle_max = np.pi
    base_env.tilt_amp = float(np.deg2rad(args.tilt_deg))
    base_env.tilt_betadot_max = args.tilt_rate_max
    base_env.p_corner = args.p_corner
    base_env.arm_limit = arm_limit
    env = (
        ResidualActionWrapper(base_env, args.residual_base, args.residual_scale)
        if args.residual_base
        else base_env
    )
    rows: dict[str, list] = defaultdict(list)

    for episode in range(args.n):
        seed = args.seed0 + episode
        obs, _ = env.reset(seed=seed)
        terminated = truncated = False
        total_reward = 0.0
        action_prev = 0.0
        actions, delta_actions, residual_actions = [], [], []
        true_ups, balanced_ticks, phi90_ticks, phi90_balanced = [], [], [], []
        phis, betas, betadots = [], [], []
        info = {}

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            action_scalar = (
                env.last_combined_action
                if args.residual_base
                else float(np.asarray(action).reshape(-1)[0])
            )
            total_reward += reward

            up = base_env._true_up()
            thd = float(base_env.data.qvel[base_env.dadr_p])
            phi = float(base_env.data.qpos[base_env.qadr_a])
            beta = float(base_env.data.qpos[base_env.qadr_t])
            betadot = float(base_env.data.qvel[base_env.dadr_t])
            physics_balanced = up > 0.9 and abs(thd) < 4.0
            arm_ok = arm_limit is None or abs(phi) < np.pi / 2
            balanced = physics_balanced and arm_ok
            # Maximum tilt coupling repeats every pi; abs(cos(phi)) is zero at +-90 deg.
            near_phi90 = abs(np.cos(phi)) <= np.sin(np.deg2rad(args.phi90_band_deg))

            actions.append(action_scalar)
            if args.residual_base:
                residual_actions.append(env.last_residual_action)
            delta_actions.append(abs(action_scalar - action_prev))
            action_prev = action_scalar
            true_ups.append(up)
            balanced_ticks.append(balanced)
            phi90_ticks.append(near_phi90)
            phi90_balanced.append(near_phi90 and physics_balanced)
            phis.append(phi)
            betas.append(beta)
            betadots.append(betadot)

        actions_a = np.asarray(actions)
        true_ups_a = np.asarray(true_ups)
        balanced_a = np.asarray(balanced_ticks, dtype=bool)
        phi90_a = np.asarray(phi90_ticks, dtype=bool)
        final_slice = slice(max(0, len(true_ups_a) - FINAL_WINDOW_STEPS), None)
        catch_start = first_catch_start(balanced_a)
        post_catch = float(np.mean(balanced_a[catch_start:])) if catch_start is not None else 0.0
        max_abs_phi = float(np.max(np.abs(phis)))

        rows["seed"].append(seed)
        rows["sustained_success"].append(bool(info.get("is_success", False)))
        rows["catch_success"].append(bool(info.get("is_catch_success", False)))
        rows["final_balance_occupancy"].append(float(info.get("final_balance_occupancy", 0.0)))
        rows["return"].append(total_reward)
        rows["episode_length"].append(len(actions))
        rows["balance_fraction"].append(float(np.mean(balanced_a)))
        rows["post_catch_balance_fraction"].append(post_catch)
        rows["final_true_up_mean"].append(float(np.mean(true_ups_a[final_slice])))
        rows["final_true_up_min"].append(float(np.min(true_ups_a[final_slice])))
        rows["action_mean_abs_delta"].append(float(np.mean(delta_actions)))
        rows["action_saturation_fraction"].append(
            float(np.mean(np.abs(actions_a) >= args.sat_threshold))
        )
        rows["action_rms"].append(float(np.sqrt(np.mean(actions_a**2))))
        if args.residual_base:
            residual_a = np.asarray(residual_actions)
            rows["residual_mean_abs"].append(float(np.mean(np.abs(residual_a))))
            rows["residual_max_abs"].append(float(np.max(np.abs(residual_a))))
        rows["max_abs_phi_rad"].append(max_abs_phi)
        rows["cable_margin_rad"].append(float(np.pi - max_abs_phi))
        rows["phi90_exposure_fraction"].append(float(np.mean(phi90_a)))
        rows["phi90_balance_fraction"].append(
            float(np.sum(phi90_balanced) / np.sum(phi90_a)) if np.any(phi90_a) else np.nan
        )
        rows["max_abs_beta_rad"].append(float(np.max(np.abs(betas))))
        rows["max_abs_betadot"].append(float(np.max(np.abs(betadots))))

        # Preserve the realized DR draw, not merely the requested ranges.
        rows["dr_active"].append(bool(base_env._dr_active))
        rows["dr_motor_gear"].append(float(base_env.model.actuator_gear[0, 0]))
        rows["dr_arm_damping"].append(float(base_env.model.dof_damping[base_env.dadr_a]))
        rows["dr_pole_damping"].append(float(base_env.model.dof_damping[base_env.dadr_p]))
        rows["dr_arm_friction"].append(float(base_env.model.dof_frictionloss[base_env.dadr_a]))
        rows["dr_pole_friction"].append(float(base_env.model.dof_frictionloss[base_env.dadr_p]))
        rows["dr_pole_inertia_scale"].append(
            float(base_env.model.body_inertia[base_env.bid_pole][0] / base_env.nom["inertia_p"][0])
        )
        rows["dr_obs_noise"].append(float(base_env._obs_noise))
        rows["dr_action_delay_steps"].append(int(base_env._delay))
        rows["tilt_cap_rad"].append(
            float(base_env.tilt_gen.beta_max) if base_env.tilt_gen is not None else 0.0
        )
        rows["tilt_rate_cap"].append(
            float(base_env.tilt_gen.betadot_max) if base_env.tilt_gen is not None else 0.0
        )

    env.close()
    return {key: np.asarray(value) for key, value in rows.items()}


def print_arm_summary(arm_name: str, result: dict[str, np.ndarray], args) -> None:
    n = len(result["seed"])
    sustained = int(np.sum(result["sustained_success"]))
    catches = int(np.sum(result["catch_success"]))
    low, high = wilson_interval(sustained, n)
    reward_mean, reward_sd = mean_sd(result["return"])
    length_mean, _ = mean_sd(result["episode_length"])
    da_mean, da_sd = mean_sd(result["action_mean_abs_delta"])
    sat_mean, sat_sd = mean_sd(result["action_saturation_fraction"])
    up_mean, up_sd = mean_sd(result["final_true_up_mean"])
    up_min_mean, _ = mean_sd(result["final_true_up_min"])
    phi_mean, phi_sd = mean_sd(np.rad2deg(result["max_abs_phi_rad"]))
    margin_mean, margin_sd = mean_sd(np.rad2deg(result["cable_margin_rad"]))
    phi90_exposure, _ = mean_sd(result["phi90_exposure_fraction"])
    phi90_balance, _ = mean_sd(result["phi90_balance_fraction"])
    beta_max_mean, beta_max_sd = mean_sd(np.rad2deg(result["max_abs_beta_rad"]))
    betadot_mean, betadot_sd = mean_sd(result["max_abs_betadot"])
    post_catch_mean, post_catch_sd = mean_sd(result["post_catch_balance_fraction"])

    print(
        f"  [{arm_name:10s}] sustained={sustained/n:.3f} ({sustained}/{n}, "
        f"95% CI {low:.3f}-{high:.3f})  catch={catches/n:.3f} ({catches}/{n})"
    )
    print(
        f"    return={reward_mean:.0f}+/-{reward_sd:.0f}  "
        f"eplen={length_mean:.0f}/2000  post-catch-balance={post_catch_mean:.3f}"
        f"+/-{post_catch_sd:.3f}"
    )
    print(
        f"    action: mean|da|={da_mean:.4f}+/-{da_sd:.4f}  "
        f"sat(|a|>={args.sat_threshold:g})={sat_mean:.3f}+/-{sat_sd:.3f}"
    )
    if args.residual_base:
        residual_mean, residual_sd = mean_sd(result["residual_mean_abs"])
        residual_max, _ = mean_sd(result["residual_max_abs"])
        print(
            f"    residual: mean|r|={residual_mean:.4f}+/-{residual_sd:.4f}  "
            f"mean episode max|r|={residual_max:.4f}"
        )
    print(
        f"    true-up(final 2s): mean={up_mean:.3f}+/-{up_sd:.3f}  "
        f"mean episode-min={up_min_mean:.3f}"
    )
    print(
        f"    arm: mean max|phi|={phi_mean:.1f}+/-{phi_sd:.1f}deg  "
        f"mean cable margin={margin_mean:.1f}+/-{margin_sd:.1f}deg"
    )
    print(
        f"    phi~90: exposure={phi90_exposure:.3f}  balance-while-exposed={phi90_balance:.3f}"
    )
    print(
        f"    realized motion: mean max|beta|={beta_max_mean:.1f}+/-{beta_max_sd:.1f}deg  "
        f"mean max|betadot|={betadot_mean:.2f}+/-{betadot_sd:.2f}rad/s"
    )
    if args.dr:
        gear = result["dr_motor_gear"]
        delay = result["dr_action_delay_steps"]
        obs_noise = result["dr_obs_noise"]
        print(
            f"    realized DR: gear={gear.min():.5f}-{gear.max():.5f}, "
            f"delay={delay.min()}-{delay.max()} steps, "
            f"obs_noise={obs_noise.min():.4f}-{obs_noise.max():.4f}"
        )


def evaluate_model(path: str, args) -> dict[str, dict[str, np.ndarray]]:
    model = TQC.load(path, device="cpu")
    print(
        f"{path}  | sha256={sha256(path)}\n"
        f"N={args.n}, seeds={args.seed0}..{args.seed0 + args.n - 1}, "
        f"tilt={args.tilt_deg:g}deg, tilt_rate_cap={args.tilt_rate_max:g}rad/s, "
        f"DR={'on' if args.dr else 'off'}, p_corner={args.p_corner:g}, deterministic"
    )
    if args.residual_base:
        print(
            f"frozen_base={args.residual_base} | sha256={sha256(args.residual_base)} | "
            f"residual_scale={args.residual_scale:g}"
        )
    arms = []
    if args.arm in ("both", "free"):
        arms.append(("free-arm", None))
    if args.arm in ("both", "cable"):
        arms.append(("cable+-180", np.pi))
    results = {}
    for arm_name, arm_limit in arms:
        result = evaluate_arm(model, args, arm_name, arm_limit)
        results[arm_name] = result
        print_arm_summary(arm_name, result, args)
    return results


def paired_summary(
    first: dict[str, dict[str, np.ndarray]],
    second: dict[str, dict[str, np.ndarray]],
    first_path: str,
    second_path: str,
) -> None:
    print("PAIRED SUSTAINED-SUCCESS COMPARISON (second - first)")
    for arm_name in first:
        if arm_name not in second:
            continue
        a = first[arm_name]["sustained_success"].astype(float)
        b = second[arm_name]["sustained_success"].astype(float)
        delta = b - a
        rng = np.random.default_rng(0)
        indices = rng.integers(0, len(delta), size=(20_000, len(delta)))
        bootstrap = np.mean(delta[indices], axis=1)
        low, high = np.percentile(bootstrap, [2.5, 97.5])
        second_only = int(np.sum((a == 0) & (b == 1)))
        first_only = int(np.sum((a == 1) & (b == 0)))
        print(
            f"  [{arm_name:10s}] delta={np.mean(delta):+.3f} "
            f"(paired bootstrap 95% CI {low:+.3f}..{high:+.3f}); "
            f"second-only wins={second_only}, first-only wins={first_only}"
        )
    print(f"  first={first_path}\n  second={second_path}")


def save_npz(path: str, model_results, model_paths: list[str], args) -> None:
    payload = {
        "model_paths": np.asarray(model_paths),
        "model_sha256": np.asarray([sha256(p) for p in model_paths]),
        "tilt_deg": np.asarray(args.tilt_deg),
        "tilt_rate_max": np.asarray(args.tilt_rate_max),
        "dr": np.asarray(args.dr),
        "p_corner": np.asarray(args.p_corner),
        "seed0": np.asarray(args.seed0),
        "residual_scale": np.asarray(args.residual_scale),
    }
    if args.residual_base:
        payload["residual_base"] = np.asarray(args.residual_base)
        payload["residual_base_sha256"] = np.asarray(sha256(args.residual_base))
    for model_index, results in enumerate(model_results):
        for arm_name, result in results.items():
            prefix = f"model{model_index}_{arm_name.replace('-', '_').replace('+', 'plus')}_"
            for key, values in result.items():
                payload[prefix + key] = values
    np.savez_compressed(path, **payload)
    print(f"saved per-episode evidence: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--compare", help="Optional second checkpoint; evaluated on identical seeds")
    ap.add_argument("--residual_base", help="Frozen base policy for evaluating a residual checkpoint")
    ap.add_argument("--residual_scale", type=float, default=0.05)
    ap.add_argument("--tilt_deg", type=float, default=0.0)
    ap.add_argument("--tilt_rate_max", type=float, default=2.0)
    ap.add_argument("--dr", action="store_true")
    ap.add_argument("--p_corner", type=float, default=0.10)
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--arm", choices=("both", "free", "cable"), default="both")
    ap.add_argument("--sat_threshold", type=float, default=0.95)
    ap.add_argument("--phi90_band_deg", type=float, default=15.0)
    ap.add_argument("--save_npz", help="Write all per-episode measurements to this .npz")
    args = ap.parse_args()
    if args.n <= 0:
        ap.error("-n must be positive")
    if not 0.0 <= args.p_corner <= 1.0:
        ap.error("--p_corner must be between 0 and 1")
    if args.tilt_rate_max <= 0.0:
        ap.error("--tilt_rate_max must be positive")
    if not 0.0 < args.sat_threshold <= 1.0:
        ap.error("--sat_threshold must be in (0, 1]")
    if not 0.0 < args.phi90_band_deg < 90.0:
        ap.error("--phi90_band_deg must be in (0, 90)")
    if not 0.0 < args.residual_scale <= 0.15:
        ap.error("--residual_scale must be in (0, 0.15]")

    paths = [args.model] + ([args.compare] if args.compare else [])
    all_results = [evaluate_model(path, args) for path in paths]
    if len(all_results) == 2:
        paired_summary(all_results[0], all_results[1], paths[0], paths[1])
    if args.save_npz:
        save_npz(args.save_npz, all_results, paths, args)


if __name__ == "__main__":
    main()

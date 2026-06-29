"""Collect successful clean-tilt teacher trajectories into a fixed replay dataset."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from furuta_env import FurutaEnv  # noqa: E402
from sb3_contrib import TQC  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("output")
    ap.add_argument("--transitions", type=int, default=200_000)
    ap.add_argument("--tilt_deg", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=20_000)
    args = ap.parse_args()

    model = TQC.load(args.model, device="cpu")
    accepted = 0
    attempts = 0
    observations, actions, next_observations, rewards, dones = [], [], [], [], []

    while accepted < args.transitions:
        env = FurutaEnv(randomize=False)
        env.init_angle_max = np.pi
        env.tilt_amp = float(np.deg2rad(args.tilt_deg))
        env.arm_limit = None
        env.arm_center_w = 0.0
        obs, _ = env.reset(seed=args.seed + attempts)
        attempts += 1
        episode = []
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode.append((
                obs.copy(),
                np.asarray(action, dtype=np.float32).copy(),
                next_obs.copy(),
                np.float32(reward),
            ))
            obs = next_obs

        if not info.get("is_success", False):
            continue
        for obs, action, next_obs, reward in episode:
            observations.append(obs)
            actions.append(action)
            next_observations.append(next_obs)
            rewards.append([reward])
            # Successful episodes end only because of the time limit; bootstrap through it.
            dones.append([0.0])
        accepted += len(episode)
        if accepted // 20_000 != (accepted - len(episode)) // 20_000:
            print(f"[teacher] accepted={accepted} attempts={attempts}", flush=True)

    arrays = dict(
        observations=np.asarray(observations[:args.transitions], dtype=np.float32),
        actions=np.asarray(actions[:args.transitions], dtype=np.float32),
        next_observations=np.asarray(next_observations[:args.transitions], dtype=np.float32),
        rewards=np.asarray(rewards[:args.transitions], dtype=np.float32),
        dones=np.asarray(dones[:args.transitions], dtype=np.float32),
    )
    np.savez_compressed(args.output, **arrays)
    print(
        f"[teacher] saved {args.output}: transitions={args.transitions}, "
        f"successful_episodes={accepted // 2000}, attempts={attempts}",
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Render one deterministic Furuta policy episode to a GIF."""
from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np
from PIL import Image
from sb3_contrib import TQC

sys.path.insert(0, os.path.dirname(__file__))
from furuta_env import FurutaEnv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("output")
    ap.add_argument("--seed", type=int, default=40003)
    ap.add_argument("--tilt_deg", type=float, default=20.0)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    model = TQC.load(args.model, device="cpu")
    env = FurutaEnv(randomize=False)
    env.init_angle_max = np.pi
    env.tilt_amp = float(np.deg2rad(args.tilt_deg))
    env.arm_limit = None
    obs, _ = env.reset(seed=args.seed)

    renderer = mujoco.Renderer(env.model, height=480, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = (0.0, 0.0, 0.04)
    camera.distance = 0.38
    camera.azimuth = 135
    camera.elevation = -18

    frames = []
    frame_interval = max(1, int(round(200 / args.fps)))
    terminated = truncated = False
    step = 0
    info = {}
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        if step % frame_interval == 0:
            renderer.update_scene(env.data, camera=camera)
            frames.append(renderer.render().copy())
        step += 1

    renderer.close()
    env.close()
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        args.output,
        save_all=True,
        append_images=images[1:],
        duration=int(round(1000 / args.fps)),
        loop=0,
        optimize=False,
    )
    print(
        f"wrote {args.output}: seed={args.seed}, frames={len(frames)}, "
        f"sustained={bool(info.get('is_success', False))}"
    )


if __name__ == "__main__":
    main()

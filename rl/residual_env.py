"""Frozen-master action-residual wrapper for safe DR adaptation."""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from sb3_contrib import TQC


def _linear(state_dict, prefix):
    for weight_key, bias_key in (
        (f"{prefix}.weight", f"{prefix}.bias"),
        (f"{prefix}.0.weight", f"{prefix}.0.bias"),
    ):
        if weight_key in state_dict:
            return state_dict[weight_key], state_dict[bias_key]
    raise KeyError(f"actor layer {prefix!r} not found")


class FrozenNumpyActor:
    """Load an SB3 actor once, then run its deterministic mean with NumPy."""

    def __init__(self, model_path: str):
        model = TQC.load(model_path, device="cpu")
        state = {key: value.detach().cpu().numpy() for key, value in model.policy.actor.state_dict().items()}
        self.w0, self.b0 = _linear(state, "latent_pi.0")
        self.w1, self.b1 = _linear(state, "latent_pi.2")
        self.wm, self.bm = _linear(state, "mu")
        self.clip_mean = (
            float(getattr(model.policy.actor, "clip_mean", 0.0))
            if "mu.0.weight" in state
            else 0.0
        )
        del model

    def predict(self, observation: np.ndarray) -> float:
        obs = np.asarray(observation, dtype=np.float32)
        hidden = np.maximum(0.0, self.w0 @ obs + self.b0)
        hidden = np.maximum(0.0, self.w1 @ hidden + self.b1)
        mean = self.wm @ hidden + self.bm
        if self.clip_mean > 0.0:
            mean = np.clip(mean, -self.clip_mean, self.clip_mean)
        return float(np.tanh(mean[0]))


class ResidualActionWrapper(gym.Wrapper):
    """Action = frozen master action + bounded learned residual."""

    def __init__(self, env: gym.Env, base_model_path: str, residual_scale: float = 0.05):
        super().__init__(env)
        if not 0.0 < residual_scale <= 0.15:
            raise ValueError("residual_scale must be in (0, 0.15]")
        self.base_actor = FrozenNumpyActor(base_model_path)
        self.residual_scale = float(residual_scale)
        self._current_obs = None
        self.last_base_action = 0.0
        self.last_residual_action = 0.0
        self.last_combined_action = 0.0

    def set_params(self, **kwargs):
        return self.env.unwrapped.set_params(**kwargs)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._current_obs = observation
        self.last_base_action = 0.0
        self.last_residual_action = 0.0
        self.last_combined_action = 0.0
        return observation, info

    def step(self, action):
        if self._current_obs is None:
            raise RuntimeError("reset() must be called before step()")
        residual_normalized = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        base_action = self.base_actor.predict(self._current_obs)
        residual_action = self.residual_scale * residual_normalized
        combined_action = float(np.clip(base_action + residual_action, -1.0, 1.0))
        result = self.env.step(np.asarray([combined_action], dtype=np.float32))
        self._current_obs = result[0]
        self.last_base_action = base_action
        self.last_residual_action = residual_action
        self.last_combined_action = combined_action
        return result

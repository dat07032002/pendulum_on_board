"""
furuta_env.py — Gymnasium environment for the GM3506 Furuta pendulum (swing-up + balance).

Wraps the MuJoCo model (furuta.xml) and matches the real firmware:
  - 200 Hz control (10 physics substeps of 0.5 ms)
  - action = motor voltage in [-6, 6] V (action in [-1,1] x 6), with a 1-2 step delay
  - velocities EMA-filtered (alpha=0.5, like the firmware) before going into the obs
  - obs = [cos(theta_up), sin(theta_up), theta_dot/15, phi/pi, phi_dot/25]  (+ sensor noise)
    where theta_up = pole - pi  (0 = upright, +-pi = hanging)
  - hard +-180 deg arm limit -> terminate + penalty (the cable)

Domain randomization (per episode) covers the measured sim-to-real uncertainty (esp. the
~2x motor-param spread) so the policy is robust. Curriculum is driven externally by setting
init_angle_max / init_vel_assist (see train_tqc.py).
"""
from __future__ import annotations

import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

HERE = os.path.dirname(__file__)
V_MAX = 6.0
DT = 0.005
TH_SCALE = 15.0      # rad/s normalizers
PHI_SCALE = 25.0
ARM_LIMIT = np.pi    # +-180 deg


class FurutaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 200}

    def __init__(self, randomize=True, render_mode=None, max_seconds=10.0):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(os.path.join(HERE, "furuta.xml"))
        self.data = mujoco.MjData(self.model)
        self.sub = int(round(DT / self.model.opt.timestep))
        self.randomize = randomize
        self.render_mode = render_mode
        self.max_steps = int(max_seconds / DT)
        self._viewer = None

        jp = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pole")
        ja = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "arm")
        self.qadr_p, self.qadr_a = self.model.jnt_qposadr[jp], self.model.jnt_qposadr[ja]
        self.dadr_p, self.dadr_a = self.model.jnt_dofadr[jp], self.model.jnt_dofadr[ja]

        # nominal params (randomize relative to these)
        self.nom = dict(
            gear=float(self.model.actuator_gear[0, 0]),
            dmp_a=float(self.model.dof_damping[self.dadr_a]),
            dmp_p=float(self.model.dof_damping[self.dadr_p]),
            fr_a=float(self.model.dof_frictionloss[self.dadr_a]),
            fr_p=float(self.model.dof_frictionloss[self.dadr_p]),
            inertia_p=self.model.body_inertia[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "pole")].copy(),
        )
        # curriculum (set externally): initial |theta_up| range + optional velocity assist
        self.init_angle_max = np.pi      # default: full swing-up from hanging
        self.init_vel_assist = 0.0
        self.p_corner = 0.3              # fraction of DR draws pushed to a min/max extreme
        self.arm_envelope_w = 0.5        # weight of the >90deg arm penalty (0 = off, v1 behavior)

        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        # obs includes prev_action (6th) so the memoryless policy can handle the action delay
        self.observation_space = spaces.Box(-np.inf, np.inf, (6,), np.float32)

    # ---- domain randomization ----
    def _randomize(self):
        rng = self.np_random
        if not self.randomize:
            return
        def u(lo, hi):   # uniform, but with prob p_corner sample a min/max extreme
            if rng.random() < self.p_corner:            # -> trains worst-case corners, not just center
                return lo if rng.random() < 0.5 else hi
            return rng.uniform(lo, hi)
        self.model.actuator_gear[0, 0] = u(0.008, 0.020)                 # KM (~2x spread)
        self.model.dof_damping[self.dadr_a] = u(3e-4, 10e-4)
        self.model.dof_damping[self.dadr_p] = u(2e-5, 1.0e-4)
        self.model.dof_frictionloss[self.dadr_a] = u(4e-3, 8e-3)
        self.model.dof_frictionloss[self.dadr_p] = u(0.2e-3, 0.6e-3)
        bp = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pole")
        self.model.body_inertia[bp] = self.nom["inertia_p"] * u(0.92, 1.08)  # alpha +-8%
        self._obs_noise = rng.uniform(0.0, 0.01)        # rad / (rad/s) scale
        self._delay = int(rng.integers(1, 4))           # 1-3 step action latency (was 1-2)

    def _restore_nominal(self):
        self.model.actuator_gear[0, 0] = self.nom["gear"]
        self.model.dof_damping[self.dadr_a] = self.nom["dmp_a"]
        self.model.dof_damping[self.dadr_p] = self.nom["dmp_p"]
        self.model.dof_frictionloss[self.dadr_a] = self.nom["fr_a"]
        self.model.dof_frictionloss[self.dadr_p] = self.nom["fr_p"]
        bp = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pole")
        self.model.body_inertia[bp] = self.nom["inertia_p"]

    # ---- gym API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._restore_nominal()
        self._obs_noise = 0.0
        self._delay = 1
        self._randomize()
        mujoco.mj_resetData(self.model, self.data)

        # initial state from curriculum: theta_up in +-init_angle_max about upright/hanging
        rng = self.np_random
        th_up = rng.uniform(-self.init_angle_max, self.init_angle_max)
        self.data.qpos[self.qadr_p] = np.pi + th_up          # pole = pi + theta_up
        self.data.qpos[self.qadr_a] = rng.uniform(-0.3, 0.3)
        self.data.qvel[self.dadr_p] = rng.uniform(-1, 1) * self.init_vel_assist
        self.data.qvel[self.dadr_a] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.thd_f = 0.0
        self.phid_f = 0.0
        self.prev_action = 0.0
        self.steps = 0
        self._up_streak = 0
        self._best_up_streak = 0
        self._was_up = False
        self.act_buf = [0.0] * self._delay
        return self._obs(), {}

    def _obs(self):
        q = self.data.qpos[self.qadr_p]
        th_up = q - np.pi                                    # 0 at upright
        phi = self.data.qpos[self.qadr_a]
        n = self._obs_noise * self.np_random.standard_normal(2)
        o = np.array([
            np.cos(th_up),
            np.sin(th_up),
            self.thd_f / TH_SCALE + n[0],
            np.clip(phi / np.pi, -2, 2),
            self.phid_f / PHI_SCALE + n[1],
            self.prev_action,                                # in-flight action (handles delay)
        ], dtype=np.float32)
        return o

    def step(self, action):
        a = float(np.clip(action[0], -1, 1))
        self.act_buf.append(a)
        a_eff = self.act_buf.pop(0)                          # delayed action
        self.data.ctrl[0] = a_eff * V_MAX
        for _ in range(self.sub):
            mujoco.mj_step(self.model, self.data)
        # EMA-filter velocities (match firmware alpha=0.5)
        self.thd_f = 0.5 * self.thd_f + 0.5 * self.data.qvel[self.dadr_p]
        self.phid_f = 0.5 * self.phid_f + 0.5 * self.data.qvel[self.dadr_a]

        q = self.data.qpos[self.qadr_p]
        th_up = (q - np.pi + np.pi) % (2 * np.pi) - np.pi
        phi = self.data.qpos[self.qadr_a]
        thd = self.data.qvel[self.dadr_p]
        phid = self.data.qvel[self.dadr_a]

        # reward: cos(theta) drives swing-up AND balance; the velocity penalty is GATED to
        # the upper half so it doesn't discourage the pumping that swing-up needs, plus a
        # strong bonus for the actual balanced state.
        up = np.cos(th_up)                                   # +1 upright, -1 hanging
        # arm-centering weight raised 0.03->0.2: the policy MUST keep the arm from winding to
        # the +-180 limit (the LQR's failure mode, which the sim reproduced).
        r = up - 0.20 * (phi / np.pi)**2 - 0.005 * a**2 - 0.002 * phid**2
        r -= 0.02 * (a - self.prev_action)**2               # mild smoothness (don't block corrective wiggle)
        # steep arm-envelope past 90 deg: discourage using the arm as a 180-deg flywheel
        # (the realism/cable risk seen in eval). Still allows transient pumping near 90-120 deg.
        r -= self.arm_envelope_w * max(0.0, abs(phi) - np.pi / 2) ** 2
        if up > 0.5:                                         # upper half: settle the pole
            r -= 0.01 * thd**2
        # bonus only when balanced AND the arm is bounded -> can't earn it by drifting
        if up > 0.92 and abs(thd) < 3.0 and abs(phi) < np.pi / 2:
            r += 2.0
        self.prev_action = a

        # success = pendulum upright (cos>0.9 ~25 deg, |thd|<4) AND arm bounded (<90 deg)
        if up > 0.9 and abs(thd) < 4.0 and abs(phi) < np.pi / 2:
            self._up_streak += 1
            self._best_up_streak = max(self._best_up_streak, self._up_streak)
        else:
            self._up_streak = 0

        if up > 0.9:
            self._was_up = True                              # reached upright at least once

        self.steps += 1
        terminated = False
        if abs(phi) > ARM_LIMIT:                             # cable limit
            r -= 10.0
            terminated = True
        # fall-termination: once it's been up, ending the episode when it falls past 90 deg
        # removes post-fall garbage AND the "wind-the-arm-to-quit" fail-fast incentive.
        # (Only fires after was_up, so swing-up from hanging isn't terminated prematurely.)
        if self._was_up and up < 0.0:
            terminated = True
        truncated = self.steps >= self.max_steps
        info = {}
        if terminated or truncated:
            info["is_success"] = bool(self._best_up_streak * DT > 0.5)   # held upright >0.5 s

        if self.render_mode == "human":
            self._render_human()
        return self._obs(), float(r), terminated, truncated, info

    # ---- rendering (eval only) ----
    def _render_human(self):
        if self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close(); self._viewer = None


if __name__ == "__main__":
    # sanity: random policy doesn't crash; reward + obs are finite and sane
    env = FurutaEnv(randomize=True)
    o, _ = env.reset(seed=0)
    print("obs0:", np.round(o, 3), "shape", o.shape)
    tot = 0.0
    for _ in range(2000):
        o, r, term, trunc, _ = env.step(env.action_space.sample())
        tot += r
        assert np.all(np.isfinite(o)) and np.isfinite(r)
        if term or trunc:
            o, _ = env.reset()
    print(f"random-policy 2000 steps OK; finite obs/reward; sample return ~{tot:.0f}")

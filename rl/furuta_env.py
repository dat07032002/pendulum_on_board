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
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from tilt import TiltGenerator  # noqa: E402

V_MAX = 6.0
DT = 0.005
TH_SCALE = 15.0      # rad/s normalizers
PHI_SCALE = 25.0
BETA_SCALE = 0.6     # board-tilt normalizer (~just above +-30 deg = 0.52 rad)
BETADOT_SCALE = 3.0  # board-tilt-rate normalizer
ARM_LIMIT = np.pi    # +-180 deg
IMU_DECIM = 2        # BNO086 fusion ~100 Hz -> update beta every 2 ticks (200 Hz loop)


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
        jt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "tilt")
        self.qadr_p, self.qadr_a = self.model.jnt_qposadr[jp], self.model.jnt_qposadr[ja]
        self.dadr_p, self.dadr_a = self.model.jnt_dofadr[jp], self.model.jnt_dofadr[ja]
        self.qadr_t, self.dadr_t = self.model.jnt_qposadr[jt], self.model.jnt_dofadr[jt]
        self.act_motor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor")
        self.act_tilt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tilt")
        self.bid_pole = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pole")

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
        self.arm_envelope_w = 0.0        # >90deg arm penalty (0 = deployed-v1 behavior; off here)
        # tilt curriculum (set externally): max board-tilt amplitude this stage (0 = level ground)
        self.tilt_amp = 0.0
        self.tilt_betadot_max = 2.0      # cap on random tilt rate [rad/s] (Phase-0 feasible bound)
        self.beta_noise = 0.005          # IMU fusion noise on beta [rad] (~0.3 deg)

        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        # obs (8): [cos th, sin th, thd/15, phi/pi, phid/25, prev_action, beta/0.6, betad/3]
        # theta/phi are base-frame (AS5600/AS5048A); beta/betad = board tilt vs gravity (BNO086 IMU).
        self.observation_space = spaces.Box(-np.inf, np.inf, (8,), np.float32)

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

        # tilt: per-episode random board motion (None = level ground). Amplitude + rate randomized
        # up to the curriculum's tilt_amp / betadot cap. Driven each step via the position actuator.
        if self.tilt_amp > 1e-4 and self.randomize:
            amp = rng.uniform(0.3, 1.0) * self.tilt_amp
            rate = rng.uniform(0.5, self.tilt_betadot_max)
            self.tilt_gen = TiltGenerator(beta_max=amp, betadot_max=rate, dt=DT,
                                          mode="random", rng=rng)
        else:
            self.tilt_gen = None
        self._beta_meas = 0.0      # IMU-measured board tilt (BNO086, ~100 Hz, held between updates)
        self._betad_meas = 0.0
        self._imu_ctr = 0
        return self._obs(), {}

    def _true_up(self):
        """cos of the pole's tilt from TRUE (gravity) vertical (+1 up). Frame-independent — this is
        the real balance objective once the base tilts (base-frame 'up' != gravity 'up')."""
        pole_dir = self.data.xmat[self.bid_pole].reshape(3, 3) @ np.array([0.0, 0.0, -1.0])
        return float(pole_dir[2])

    def _obs(self):
        q = self.data.qpos[self.qadr_p]
        th_up = q - np.pi                                    # 0 at upright (base/board frame, AS5600)
        phi = self.data.qpos[self.qadr_a]
        n = self._obs_noise * self.np_random.standard_normal(2)
        o = np.array([
            np.cos(th_up),
            np.sin(th_up),
            self.thd_f / TH_SCALE + n[0],
            np.clip(phi / np.pi, -2, 2),
            self.phid_f / PHI_SCALE + n[1],
            self.prev_action,                                # in-flight action (handles delay)
            np.clip(self._beta_meas / BETA_SCALE, -2, 2),    # board tilt (BNO086 IMU vs gravity)
            self._betad_meas / BETADOT_SCALE,                # board tilt rate (IMU gyro)
        ], dtype=np.float32)
        return o

    def step(self, action):
        a = float(np.clip(action[0], -1, 1))
        self.act_buf.append(a)
        a_eff = self.act_buf.pop(0)                          # delayed action
        self.data.ctrl[self.act_motor] = a_eff * V_MAX
        # drive the board tilt: the servo (position actuator) tracks beta_ref(t)
        beta_ref = self.tilt_gen.step() if self.tilt_gen is not None else 0.0
        self.data.ctrl[self.act_tilt] = beta_ref
        for _ in range(self.sub):
            mujoco.mj_step(self.model, self.data)
        # EMA-filter velocities (match firmware alpha=0.5)
        self.thd_f = 0.5 * self.thd_f + 0.5 * self.data.qvel[self.dadr_p]
        self.phid_f = 0.5 * self.phid_f + 0.5 * self.data.qvel[self.dadr_a]
        # BNO086 IMU read of board tilt: ~100 Hz fusion -> refresh every IMU_DECIM ticks (+ noise)
        self._imu_ctr += 1
        if self._imu_ctr >= IMU_DECIM:
            self._imu_ctr = 0
            nb = self.beta_noise * self.np_random.standard_normal() if self.randomize else 0.0
            self._beta_meas = float(self.data.qpos[self.qadr_t]) + nb
            self._betad_meas = float(self.data.qvel[self.dadr_t])

        q = self.data.qpos[self.qadr_p]
        th_up = (q - np.pi + np.pi) % (2 * np.pi) - np.pi
        phi = self.data.qpos[self.qadr_a]
        thd = self.data.qvel[self.dadr_p]
        phid = self.data.qvel[self.dadr_a]

        # reward: cos(theta) drives swing-up AND balance; the velocity penalty is GATED to
        # the upper half so it doesn't discourage the pumping that swing-up needs, plus a
        # strong bonus for the actual balanced state.
        up = self._true_up()                                 # +1 true-vertical, -1 inverted (gravity)
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

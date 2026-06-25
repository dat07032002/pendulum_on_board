"""
tilt.py — bounded tilt reference beta_ref(t) for the tilting base (LX-16A servo).

Two modes, sharing one rate/amplitude bound so sim and firmware match:
  - "triangle": deterministic triangle wave at a fixed rate -> for the Phase-0 feasibility SWEEP
    (clean worst-case: continuous motion at exactly beta_dot, reversing at +-beta_max).
  - "random":   rate-limited moves to random targets within +-beta_max, random per-segment speed
    and dwell -> for training domain randomization.

beta in radians. Deterministic given the rng. Ported to firmware as the same logic.
"""
from __future__ import annotations

import numpy as np


class TiltGenerator:
    def __init__(self, beta_max=np.deg2rad(30.0), betadot_max=2.0, dt=0.005,
                 mode="random", rng=None, dwell_s=(0.1, 0.6)):
        self.beta_max = float(beta_max)
        self.betadot_max = float(betadot_max)
        self.dt = float(dt)
        self.mode = mode
        self.rng = rng if rng is not None else np.random.default_rng()
        self.dwell_s = dwell_s
        self.reset()

    def reset(self):
        self.beta = 0.0
        self.betadot = 0.0
        self._dir = 1
        self._new_target()
        return self.beta

    def _new_target(self):
        self.target = float(self.rng.uniform(-self.beta_max, self.beta_max))
        self._rate = float(self.rng.uniform(0.3 * self.betadot_max, self.betadot_max))
        self._dwell = int(self.rng.uniform(*self.dwell_s) / self.dt)

    def step(self):
        if self.mode == "triangle":
            self.beta += self._dir * self.betadot_max * self.dt
            if self.beta >= self.beta_max:
                self.beta = self.beta_max; self._dir = -1
            elif self.beta <= -self.beta_max:
                self.beta = -self.beta_max; self._dir = 1
            self.betadot = self._dir * self.betadot_max
            return self.beta
        # "random": rate-limited toward target, dwell on arrival, then re-target
        err = self.target - self.beta
        if abs(err) <= self._rate * self.dt:
            self.beta = self.target; self.betadot = 0.0
            self._dwell -= 1
            if self._dwell <= 0:
                self._new_target()
        else:
            self.betadot = np.sign(err) * self._rate
            self.beta += self.betadot * self.dt
        self.beta = float(np.clip(self.beta, -self.beta_max, self.beta_max))
        return self.beta


if __name__ == "__main__":
    import numpy as np
    g = TiltGenerator(betadot_max=2.0, mode="random", rng=np.random.default_rng(0))
    bs = [g.step() for _ in range(2000)]
    bs = np.array(bs)
    print(f"random: range [{np.rad2deg(bs.min()):.1f}, {np.rad2deg(bs.max()):.1f}] deg, "
          f"max |betadot| {np.max(np.abs(np.diff(bs))/0.005):.2f} rad/s (cap 2.0)")
    g = TiltGenerator(betadot_max=1.0, mode="triangle")
    bs = np.array([g.step() for _ in range(2000)])
    print(f"triangle@1.0: range [{np.rad2deg(bs.min()):.1f}, {np.rad2deg(bs.max()):.1f}] deg")

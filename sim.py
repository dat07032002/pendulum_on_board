"""
sim.py — minimal simulator + closed-loop catch test for the TORQUE-SOURCE plant.

Ported from the proven velocity-source sim.py on `main`, adapted to the GM3506
torque model in plant_torque.py. The control input is the motor VOLTAGE V (not an
arm acceleration), held constant over each control step (zero-order hold), exactly
as the ESP32 runs it.

The catch test runs the SAME algorithm as firmware/furuta_foc/furuta_foc.ino
balanceStep(): a predictor observer updates xhat from the position measurements,
then the LQR computes V = -K xhat, clipped to voltage_limit. This validates that
the GM3506 LQR actually catches a falling rod -- and tells you the peak voltage and
arm excursion -- BEFORE you trust the real motor.

    python sim.py
"""
from __future__ import annotations

import numpy as np

import plant_torque as plant
from balance_torque import LQRBalance


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class FurutaSim:
    """RK4 rollout of the full nonlinear torque-source dynamics."""

    def __init__(self, dt: float = plant.DT, substeps: int = 10):
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.x = np.zeros(plant.N_STATE)
        self.t = 0.0

    def reset(self, theta0: float = 0.0, theta_dot0: float = 0.0,
              phi0: float = 0.0, phi_dot0: float = 0.0) -> np.ndarray:
        self.x = np.array([phi0, theta0, phi_dot0, theta_dot0], dtype=float)
        self.t = 0.0
        return self.x.copy()

    def step(self, V: float) -> np.ndarray:
        """Advance one control step under constant motor voltage V (RK4)."""
        h = self.dt / self.substeps
        x = self.x
        for _ in range(self.substeps):
            k1 = plant.dynamics(x, V)
            k2 = plant.dynamics(x + 0.5 * h * k1, V)
            k3 = plant.dynamics(x + 0.5 * h * k2, V)
            k4 = plant.dynamics(x + h * k3, V)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.x = x
        self.t += self.dt
        return self.x.copy()

    @property
    def theta(self) -> float:
        return float(self.x[plant.THETA])


class FirmwareController:
    """Mirror of balanceStep() in furuta_foc.ino: predictor observer -> LQR.

    Uses the exact K, Ad, Bd, L, C from balance_torque.LQRBalance, and applies them
    in the same order and form as the firmware so the sim is faithful to what runs
    on the ESP32. Measures y = [phi, theta, phi_dot]; estimates theta_dot.
    """

    def __init__(self, ctrl: LQRBalance, voltage_limit: float = 6.0):
        self.K = ctrl.K.ravel()
        self.Ad = ctrl.Ad
        self.Bd = ctrl.Bd.ravel()
        self.L = ctrl.L
        self.C = ctrl.C
        self.voltage_limit = float(voltage_limit)
        self.reset()

    def reset(self, x0: np.ndarray | None = None):
        self.xhat = np.zeros(4) if x0 is None else np.asarray(x0, float).copy()
        self.prev_V = 0.0
        self._init = x0 is not None

    def __call__(self, x: np.ndarray) -> float:
        # On the first call after handoff, seed xhat from the measurement (firmware
        # copies the live state into xhat when BALANCE engages).
        meas = np.array([x[plant.PHI], _wrap(x[plant.THETA]),
                         x[plant.PHI_DOT], x[plant.THETA_DOT]])
        if not self._init:
            self.xhat = meas.copy()
            self._init = True

        # LQR FIRST: V = -K xhat from the current estimate, clipped (firmware order).
        V = float(-(self.K @ self.xhat))
        V = float(np.clip(V, -self.voltage_limit, self.voltage_limit))

        # Observer: propagate with THIS V (predictor form): correct order, or the
        # combined loop is unstable. See balanceStep() in furuta_foc.ino.
        y = meas[:3]
        innov = y - self.C @ self.xhat
        innov[1] = _wrap(innov[1])
        self.xhat = self.Ad @ self.xhat + self.Bd * V + self.L @ innov

        self.prev_V = V
        return V


def _catch(sim, ctrl, theta0, theta_dot0=0.0, seconds=3.0):
    """Run a catch from a tilt; return (held, stats). 'held' = ended upright+slow."""
    x = sim.reset(theta0=theta0, theta_dot0=theta_dot0)
    ctrl.reset()
    n = int(seconds / sim.dt)
    max_V = max_phidot = max_phi = 0.0
    held_streak = best_streak = 0
    for _ in range(n):
        V = ctrl(x)
        x = sim.step(V)
        max_V = max(max_V, abs(V))
        max_phidot = max(max_phidot, abs(x[plant.PHI_DOT]))
        max_phi = max(max_phi, abs(x[plant.PHI]))
        if abs(_wrap(x[plant.THETA])) < np.deg2rad(8.0) and abs(x[plant.THETA_DOT]) < 2.0:
            held_streak += 1
            best_streak = max(best_streak, held_streak)
        else:
            held_streak = 0
    held = abs(_wrap(x[plant.THETA])) < np.deg2rad(5.0) and abs(x[plant.THETA_DOT]) < 1.0
    return held, dict(max_V=max_V, max_phidot=max_phidot, max_phi=max_phi,
                      best_hold=best_streak * sim.dt)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # Sanity: uncontrolled, a tiny tilt should diverge (upright unstable).
    sim = FurutaSim()
    sim.reset(theta0=np.deg2rad(1.0))
    for _ in range(50):
        sim.step(0.0)
    print(f"uncontrolled from 1 deg after 50 steps: theta = {np.degrees(sim.theta):+.1f} deg "
          "(should grow -> upright unstable)\n")

    bal = LQRBalance()
    print("LQR gain K [phi, theta, phi_dot, theta_dot] =", np.round(bal.K.ravel(), 3))
    cl = bal.Ad - bal.Bd @ bal.K
    print("Closed-loop discrete |eigenvalues| =", np.round(np.abs(np.linalg.eigvals(cl)), 4))
    print("  -> all < 1 means the LQR stabilizes upright.\n")

    VLIM = 6.0   # must match `voltage_limit` in firmware
    ctrl = FirmwareController(bal, voltage_limit=VLIM)

    print(f"Catch test (firmware algorithm, voltage_limit = {VLIM:.1f} V):")
    print(f"{'tilt deg':>8} | {'held?':>5} | {'hold s':>6} | {'max V':>6} | "
          f"{'max phidot':>10} | {'max phi deg':>11}")
    print("-" * 64)
    for deg in (2, 5, 8, 10, 15, 20):
        held, s = _catch(sim, ctrl, np.deg2rad(deg))
        print(f"{deg:8d} | {('YES' if held else 'no'):>5} | {s['best_hold']:6.2f} | "
              f"{s['max_V']:6.2f} | {s['max_phidot']:10.2f} | "
              f"{np.degrees(s['max_phi']):11.1f}")

    print(f"\nNote: max V must stay under voltage_limit ({VLIM:.1f} V) or the catch is")
    print("saturating -- raise vlim or soften the catch (lower R / lower q_theta).")
    print("The handoff window in firmware is +-5 deg, so the 2-8 deg rows matter most.")

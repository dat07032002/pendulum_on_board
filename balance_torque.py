"""
balance_torque.py — LQR balance controller for the torque-source Furuta plant.

Designs the discrete LQR gain K and Kalman observer gain L for the GM3506
gimbal motor. The LQR output is a motor VOLTAGE (not acceleration), which
goes directly to the FOC driver — no velocity integration, no dither.

Outputs the matrices needed for the firmware: AD, BD, LOBS, KGAIN.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg

import plant_torque as plant


def design_lqr(Q: np.ndarray, R: np.ndarray, dt: float = plant.DT):
    """Discrete LQR about upright. Returns (K, Ad, Bd)."""
    A, B = plant.linearize()
    Ad, Bd = plant.discretize(A, B, dt)
    P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, Ad, Bd


def design_observer(Ad, C, Qn, Rn):
    """Steady-state Kalman predictor gain L."""
    P = scipy.linalg.solve_discrete_are(Ad.T, C.T, Qn, Rn)
    L = (Ad @ P @ C.T) @ np.linalg.inv(C @ P @ C.T + Rn)
    return L


class LQRBalance:
    # Q weights: [phi, theta, phi_dot, theta_dot]
    # With torque-source motor, centering (q_phi) should work directly
    # since the motor can apply any torque at any speed.
    DEFAULT_Q = np.diag([500.0, 5000.0, 10.0, 50.0])
    DEFAULT_R = np.array([[0.01]])

    # Observer: measure [phi, theta, phi_dot]; estimate theta_dot
    DEFAULT_QN = np.diag([1e-3, 1e-3, 1e-1, 2e-1])
    DEFAULT_RN = np.diag([1e-5, 1e-5, 5e-3])

    def __init__(self, Q=None, R=None, dt=plant.DT):
        self.Q = np.asarray(self.DEFAULT_Q if Q is None else Q, dtype=float)
        self.R = np.asarray(self.DEFAULT_R if R is None else R, dtype=float)
        self.dt = float(dt)
        self.K, self.Ad, self.Bd = design_lqr(self.Q, self.R, self.dt)
        self.C = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
        self.L = design_observer(self.Ad, self.C, self.DEFAULT_QN, self.DEFAULT_RN)


def format_for_firmware(arr, name, cols=None):
    """Format a numpy array as a C float array initializer."""
    flat = arr.ravel()
    vals = ", ".join(f"{v:.6f}f" for v in flat)
    return f"const float {name}[{len(flat)}] = {{{vals}}};"


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)

    ctrl = LQRBalance()
    K = ctrl.K
    Ad = ctrl.Ad
    Bd = ctrl.Bd
    L = ctrl.L

    cl = Ad - Bd @ K
    eig = np.linalg.eigvals(cl)

    print("=== Torque-Source LQR Balance (GM3506) ===\n")
    print(f"Q = diag{np.diag(ctrl.Q).tolist()}")
    print(f"R = {ctrl.R.ravel().tolist()}\n")

    print(f"LQR gain K [phi, theta, phi_dot, theta_dot] = {np.round(K.ravel(), 5)}")
    print(f"Closed-loop |eigenvalues| = {np.round(np.abs(eig), 4)}")
    print(f"  -> all < 1 means stable\n")

    print(f"Observer L (4x3):")
    print(np.round(L, 6))

    print(f"\nVoltage for 1 deg tilt: {abs(K[0,1]) * np.deg2rad(1):.2f} V")
    print(f"Voltage for 10 deg arm displacement: {abs(K[0,0]) * np.deg2rad(10):.2f} V")

    # Firmware output
    print("\n=== Copy to firmware ===\n")
    print(format_for_firmware(Ad, "AD"))
    print(format_for_firmware(Bd, "BD"))
    print(format_for_firmware(L, "LOBS"))
    print(format_for_firmware(K, "KGAIN_DEFAULT"))

    print(f"\n// V_max at 5 deg tilt = {abs(K[0,1]) * np.deg2rad(5):.1f} V")
    print(f"// V_max at 30 deg arm = {abs(K[0,0]) * np.deg2rad(30):.1f} V")

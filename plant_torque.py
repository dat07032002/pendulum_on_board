"""
plant_torque.py — Furuta pendulum physics for a TORQUE-SOURCE motor (GM3506).

Full 2-DOF coupled Lagrangian dynamics. The motor applies a torque (voltage)
to the arm, and the arm and pendulum are both free bodies coupled by their
shared pivot. This replaces the velocity-source model in furuta_lqr/plant.py.

State:
    x = [phi, theta, phi_dot, theta_dot]
Input:
    u = V  (motor voltage, proportional to torque at low speed)

The motor model: tau = Km * V - Km*Ke/R * phi_dot
where Km = Kt/R (torque per volt at stall), and the back-EMF term provides
natural velocity damping. At low speed (balance), the damping is small.

Linearized about upright (theta=0):
    M * [phi_ddot; theta_ddot] = [0; m_p*l_p*g] * [phi; theta] + [Km; 0]*V + damping

    Rearranged to xdot = A*x + B*u form by inverting the mass matrix M.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg

# --- physical parameters ---
G = 9.81              # gravity [m/s^2]

# Pendulum (uniform rod, no tip mass)
L_ROD = 0.075         # rod length [m]
M_ROD = 0.027         # rod mass [kg]
L_P = L_ROD / 2.0     # center of mass from pivot [m]
J_P = M_ROD * L_ROD**2 / 3.0   # moment of inertia about pivot [kg*m^2]

# Arm (point mass at L_ARM from motor axis, plus motor rotor)
L_ARM = 0.035         # arm pivot radius [m] (measured)
M_ARM = 0.015         # arm assembly mass [kg]
J_ROTOR = 5e-5        # motor rotor inertia [kg*m^2] (typical GM3506)
J_ARM = M_ARM * L_ARM**2 + J_ROTOR   # total arm inertia about motor axis

# Motor (GM3506 via TMC6300)
R_MOTOR = 5.6         # winding resistance [Ohm]
KT = 0.068            # torque constant [N*m/A] (≈ Ke for BLDC)
KM = KT / R_MOTOR     # torque per volt at stall [N*m/V]
KE = KT               # back-EMF constant [V*s/rad]
DAMPING = KT * KE / R_MOTOR   # electrical damping coefficient [N*m*s/rad]

# Coupling term
COUPLING = M_ROD * L_P * L_ARM   # m_p * l_p * L_arm [kg*m^2]

# --- control timing ---
CONTROL_HZ = 200.0
DT = 1.0 / CONTROL_HZ

# state indices
PHI, THETA, PHI_DOT, THETA_DOT = 0, 1, 2, 3
N_STATE = 4


def _mass_matrix():
    """2x2 mass matrix M for [phi_ddot, theta_ddot]."""
    M = np.array([
        [J_ARM + M_ROD * L_ARM**2,  COUPLING],
        [COUPLING,                   J_P],
    ])
    return M


def dynamics(x: np.ndarray, V: float) -> np.ndarray:
    """Continuous-time state derivative xdot = f(x, V). Full nonlinear."""
    phi, theta, phi_dot, theta_dot = x
    M = np.array([
        [J_ARM + M_ROD * L_ARM**2,  COUPLING * np.cos(theta)],
        [COUPLING * np.cos(theta),   J_P],
    ])
    # Right-hand side: gravity + motor torque + coriolis/centrifugal
    tau = KM * V - DAMPING * phi_dot
    rhs = np.array([
        tau + COUPLING * np.sin(theta) * theta_dot**2,
        M_ROD * L_P * G * np.sin(theta) - COUPLING * np.sin(theta) * phi_dot * theta_dot,
    ])
    # Solve M * [phi_ddot, theta_ddot] = rhs
    acc = np.linalg.solve(M, rhs)
    return np.array([phi_dot, theta_dot, acc[0], acc[1]], dtype=float)


def linearize() -> tuple[np.ndarray, np.ndarray]:
    """Continuous-time linearization about upright (x*=0, V*=0): xdot = A x + B V.

    At theta=0: sin(theta)≈theta, cos(theta)≈1, theta_dot^2≈0, phi_dot*theta_dot≈0.

    M * [phi_ddot; theta_ddot] = [0, 0; 0, m_p*l_p*g] * [phi; theta]
                                 + [-DAMPING; 0] * phi_dot
                                 + [KM; 0] * V

    Invert M to get accelerations as a function of state and input.
    """
    M = _mass_matrix()
    M_inv = np.linalg.inv(M)

    # Gravity matrix (only theta has a restoring/destabilizing force)
    G_mat = np.array([
        [0.0, 0.0],
        [0.0, M_ROD * L_P * G],
    ])

    # Damping matrix (only phi_dot has electrical damping)
    D_mat = np.array([
        [-DAMPING, 0.0],
        [0.0,      0.0],
    ])

    # Input vector (torque only on phi)
    B_tau = np.array([[KM], [0.0]])

    # A matrix: xdot = [phi_dot, theta_dot, M_inv @ (G_mat @ [phi,theta] + D_mat @ [phi_dot,theta_dot])]
    A = np.zeros((4, 4))
    A[0, 2] = 1.0   # phi_dot
    A[1, 3] = 1.0   # theta_dot
    A[2:, :2] = M_inv @ G_mat       # position -> acceleration
    A[2:, 2:] = M_inv @ D_mat       # velocity -> acceleration (damping)

    # B matrix: effect of voltage on acceleration
    B = np.zeros((4, 1))
    B[2:, :] = M_inv @ B_tau

    return A, B


def discretize(A: np.ndarray, B: np.ndarray, dt: float = DT) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretization."""
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = scipy.linalg.expm(M * dt)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd


def pendulum_energy(theta: float, theta_dot: float) -> float:
    """Pendulum mechanical energy (for swing-up), referenced so upright E=1."""
    alpha = M_ROD * L_P * G / J_P
    return 0.5 * theta_dot**2 / alpha + np.cos(theta)


E_UPRIGHT = 1.0


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)

    print("=== Torque-Source Furuta Plant (GM3506) ===\n")
    print(f"Pendulum: L={L_ROD}m, m={M_ROD}kg, l_cm={L_P}m, J_p={J_P:.6f} kg*m^2")
    print(f"Arm:      L={L_ARM}m, m_total={M_ARM}kg, J_arm={J_ARM:.6f} kg*m^2")
    print(f"Motor:    R={R_MOTOR}ohm, Kt={KT}, Km={KM:.4f} N*m/V, damping={DAMPING:.5f}")
    print(f"Coupling: {COUPLING:.6f} kg*m^2")
    print(f"DT:       {DT:.4f}s ({CONTROL_HZ:.0f} Hz)\n")

    M = _mass_matrix()
    print("Mass matrix M:")
    print(M)
    print(f"det(M) = {np.linalg.det(M):.2e}  (must be > 0)\n")

    A, B = linearize()
    print("Continuous A:")
    print(A)
    print(f"\nB: {B.ravel()}")

    eig_c = np.linalg.eigvals(A)
    print(f"\nContinuous eigenvalues: {np.round(eig_c, 3)}")
    print("  -> positive real eigenvalue = upright is unstable (expected)")

    Ad, Bd = discretize(A, B)
    print(f"\nDiscrete Ad:")
    print(Ad)
    print(f"Bd: {Bd.ravel()}")

    eig_d = np.linalg.eigvals(Ad)
    print(f"\nDiscrete |eigenvalues|: {np.round(np.abs(eig_d), 4)}")
    print("  -> any > 1 = open-loop unstable (LQR must fix)")

    # Controllability
    C = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(N_STATE)])
    rank = np.linalg.matrix_rank(C)
    print(f"\nControllability rank: {rank} / {N_STATE}  {'OK' if rank == N_STATE else 'FAIL'}")

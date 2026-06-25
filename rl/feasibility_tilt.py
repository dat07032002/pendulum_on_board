"""
feasibility_tilt.py — Phase 0: can a HAND-CODED controller hold the pendulum upright while the
base tilts +-30 deg? Uses the classical LQR (no trained policy), with a true-vertical feedforward
(theta_ref = s*beta*cos(phi)) so the balance target tracks gravity as the board tilts.

Sweeps the tilt RATE (triangle wave at +-30 deg) and reports the max beta_dot the LQR survives.
That sets a conservative beta_dot_max for the random tilt + a go/no-go before any training/hardware.

    python rl/feasibility_tilt.py
"""
from __future__ import annotations

import os
import numpy as np
import mujoco

from tilt import TiltGenerator

HERE = os.path.dirname(__file__)
M = mujoco.MjModel.from_xml_path(os.path.join(HERE, "furuta.xml"))
D = mujoco.MjData(M)


def _jadr(n):
    j = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_JOINT, n)
    return M.jnt_qposadr[j], M.jnt_dofadr[j]


PQ, PV = _jadr("pole"); AQ, AV = _jadr("arm"); TQ, TV = _jadr("tilt")
POLE_JNT = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_JOINT, "pole")
MOT = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor")
TILT = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_ACTUATOR, "tilt")
POLE_BODY = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, "pole")
DT = 0.005
SUB = int(round(DT / M.opt.timestep))
K = np.array([-4.66, -62.35, -1.53, -4.54])     # LQR [phi, theta_up, phidot, thetadot]
ARM_GUARD = np.deg2rad(160.0)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def true_up():
    """cos of the pole's tilt from TRUE vertical (+1 upright, -1 hanging) — frame-independent."""
    R = D.xmat[POLE_BODY].reshape(3, 3)
    return float((R @ np.array([0.0, 0.0, -1.0]))[2])


UP = np.array([0.0, 0.0, 1.0])


def true_theta():
    """SIGNED pole angle from true vertical, about the pole's hinge axis (0 = true upright).
    Geometric -> the real zero-torque equilibrium, so a controller on this won't wind the arm."""
    pole_dir = D.xmat[POLE_BODY].reshape(3, 3) @ np.array([0.0, 0.0, -1.0])  # pivot->tip, +z up
    h = D.xaxis[POLE_JNT]                                                    # hinge axis (world)
    return float(np.arctan2(np.dot(np.cross(UP, pole_dir), h), np.dot(UP, pole_dir)))


def run(beta_fn, phi_target=0.0, seconds=12.0, settle=1.0):
    """beta_fn(i) -> commanded board tilt [rad] each step (after settle). phi_target = arm
    orientation to hold (worst tilt-disturbance is ~90 deg, swing plane aligned with the tilt)."""
    mujoco.mj_resetData(M, D)
    D.qpos[PQ] = np.pi                     # start upright
    D.qpos[AQ] = phi_target                # arm at the test orientation
    mujoco.mj_forward(M, D)
    n, n_settle = int(seconds / DT), int(settle / DT)
    min_up, max_arm, failed = 1.0, 0.0, False
    for i in range(n):
        phi = D.qpos[AQ]; phid = D.qvel[AV]
        th = wrap(D.qpos[PQ] - np.pi); thd = D.qvel[PV]    # base-frame pole angle (consistent sign)
        beta = D.qpos[TQ]; betad = D.qvel[TV]
        th_ref = beta * np.sin(phi)                        # base-frame angle of TRUE vertical
        thd_ref = betad * np.sin(phi)                      # (drop the phi_dot term -> no runaway)
        V = -(K[0] * (phi - phi_target) + K[1] * (th - th_ref) + K[2] * phid + K[3] * (thd - thd_ref))
        if abs(phi - phi_target) > ARM_GUARD:
            V = -np.sign(phi - phi_target) * 6.0
        D.ctrl[MOT] = float(np.clip(V, -6.0, 6.0))
        D.ctrl[TILT] = 0.0 if i < n_settle else beta_fn(i - n_settle)
        for _ in range(SUB):
            mujoco.mj_step(M, D)
        if i > n_settle:
            u = true_up()
            min_up = min(min_up, u); max_arm = max(max_arm, abs(D.qpos[AQ] - phi_target))
            if u < 0.0 or abs(D.qpos[AQ] - phi_target) > np.pi:
                failed = True; break
    return dict(min_up=min_up, max_arm=np.rad2deg(max_arm),
                held=(not failed and min_up > 0.82))   # >0.82 = within ~35 deg of true vertical


def static_hold(target, ramp=1.0):
    """beta_fn: ramp to a fixed tilt over `ramp` s, then hold (for sign calibration)."""
    nr = ramp / DT
    return lambda i: target * min(1.0, i / nr)


def triangle(betadot, beta_max=np.deg2rad(30.0)):
    gen = TiltGenerator(beta_max=beta_max, betadot_max=betadot, dt=DT, mode="triangle")
    return lambda i: gen.step()


def main():
    # the y-tilt's disturbance is arm-orientation dependent: ~0 at phi=0 (swing plane _|_ tilt),
    # max at phi=90 (swing plane aligned). Test across phi, and sweep rate at the WORST (90).
    print("Static +-30 deg hold vs arm orientation (phi=0 benign, phi=90 worst):")
    for pd in (0, 45, 90):
        r = run(static_hold(np.deg2rad(30.0)), phi_target=np.deg2rad(pd), seconds=5.0)
        print(f"  phi={pd:>2} deg: held={r['held']}  min_up={r['min_up']:.2f}  "
              f"arm excursion max={r['max_arm']:.0f} deg")

    print("\nTilt-rate sweep at WORST arm orientation (phi=90 deg), +-30 deg triangle:")
    print(f"{'betadot[rad/s]':>14} {'~deg/s':>8} {'held':>6} {'min_up':>8} {'arm_exc[deg]':>13}")
    max_ok = 0.0
    for bd in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
        r = run(triangle(bd), phi_target=np.deg2rad(90.0))
        print(f"{bd:>14.1f} {np.rad2deg(bd):>8.0f} {str(r['held']):>6} "
              f"{r['min_up']:>8.2f} {r['max_arm']:>13.0f}")
        if r["held"]:
            max_ok = bd
    print(f"\n=> max survivable tilt rate at worst phi (LQR, conservative): ~{max_ok:.1f} rad/s "
          f"({np.rad2deg(max_ok):.0f} deg/s) at +-30 deg.")
    print("   RL should match/exceed this; use it to set beta_dot_max for training.")


if __name__ == "__main__":
    main()

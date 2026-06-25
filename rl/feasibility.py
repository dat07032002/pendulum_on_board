"""
feasibility.py — can a HAND-CODED controller swing up + balance within +-180 deg / +-6 V?

Decouples "is it physically possible" from "can RL learn it":
  - Energy-pump swing-up (Astrom-style bang-bang) until near upright,
  - then hand off to the LQR gains (from balance_torque.py),
  - all clipped to +-6 V, arm guarded at +-180 deg.
If this reaches and HOLDS upright, the constraints are feasible -> the RL 0% is a learning
problem, not a hardware wall. If it can't swing up at 6 V, the voltage is too tight (raise it).

    python rl/feasibility.py [--vmax 6] [--seconds 12]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import mujoco

HERE = os.path.dirname(__file__)
M = mujoco.MjModel.from_xml_path(os.path.join(HERE, "furuta.xml"))
D = mujoco.MjData(M)
PA = M.jnt_qposadr[mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_JOINT, "pole")]
AA = M.jnt_qposadr[mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_JOINT, "arm")]
PV = M.jnt_dofadr[mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_JOINT, "pole")]
AV = M.jnt_dofadr[mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_JOINT, "arm")]
DT = 0.005
SUB = int(round(DT / M.opt.timestep))
ALPHA = 214.0                      # m g l / J_p (measured)
K = np.array([-4.66, -62.35, -1.53, -4.54])   # LQR [phi, theta_up, phidot, thetadot]
ARM_GUARD = np.deg2rad(160.0)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run(vmax, pump_sign, kE=3.0, seconds=12.0, catch_deg=30.0):
    mujoco.mj_resetData(M, D)            # pole=0 -> hanging
    catch = np.deg2rad(catch_deg)
    t = 0.0
    max_up = -1.0
    held = 0
    log = []
    n = int(seconds / DT)
    for i in range(n):
        q = D.qpos[PA]; phi = D.qpos[AA]
        thd = D.qvel[PV]; phid = D.qvel[AV]
        th_up = wrap(q - np.pi)          # 0 at upright
        up = np.cos(th_up)
        # energy relative to upright (0 at top, negative below)
        E = 0.5 * thd**2 + ALPHA * (up - 1.0)
        if abs(th_up) < catch and abs(thd) < 8.0:
            V = -(K[0]*phi + K[1]*th_up + K[2]*phid + K[3]*thd)   # LQR catch/balance
            mode = "BAL"
        else:
            s = np.sign(thd * np.cos(th_up))     # energy-pump direction
            if s == 0:                            # at rest -> kick to break symmetry
                s = 1.0
            # proportional to energy deficit so it TAPERS near the top (catchable handoff),
            # saturates far away (max pumping). E<0 below upright; use |E|.
            V = pump_sign * min(0.12 * abs(E), vmax) * s
            V -= 2.5 * phi            # arm-centering: keep the arm near 0 so there's catch room
            mode = "PUMP"
        V = float(np.clip(V, -vmax, vmax))
        if abs(phi) > ARM_GUARD:         # arm guard: push back inward
            V = -np.sign(phi) * vmax
        D.ctrl[0] = V
        for _ in range(SUB):
            mujoco.mj_step(M, D)
        max_up = max(max_up, up)
        if up > 0.95 and abs(D.qvel[PV]) < 3.0:
            held += 1
        else:
            held = 0
        if i % 40 == 0:
            log.append((t, np.degrees(th_up), np.degrees(phi), V, mode))
        t += DT
    final_up = np.cos(wrap(D.qpos[PA] - np.pi))
    return dict(max_up=max_up, final_up=final_up, best_hold_s=held*DT,
                final_arm=np.degrees(wrap(D.qpos[AA])), log=log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vmax", type=float, default=6.0)
    ap.add_argument("--seconds", type=float, default=12.0)
    args = ap.parse_args()

    print(f"Feasibility: energy-pump swing-up + LQR, vmax={args.vmax} V, arm +-180 deg\n")
    best = None
    for s in (+1.0, -1.0):
        r = run(args.vmax, s, seconds=args.seconds)
        print(f"  pump_sign={s:+.0f}: max cos(theta)={r['max_up']:+.2f} "
              f"(1=upright), best_hold={r['best_hold_s']:.1f}s, final cos={r['final_up']:+.2f}, "
              f"final_arm={r['final_arm']:+.0f}deg")
        score = r['best_hold_s'] * 100 + r['max_up']     # prioritize hold, tiebreak by uprightness
        if best is None or score > best_score:
            best = r; best_s = s; best_score = score
    print()
    if best['best_hold_s'] > 1.0:
        print(f"=> FEASIBLE at {args.vmax} V: swings up and HOLDS {best['best_hold_s']:.1f}s "
              f"(pump_sign={best_s:+.0f}). The RL 0% is a learning problem, not the constraints.")
    elif best['max_up'] > 0.5:
        print(f"=> swings UP (max cos={best['max_up']:.2f}) but doesn't hold -> catch/handoff "
              f"tuning, swing-up energy is reachable at {args.vmax} V.")
    else:
        print(f"=> CANNOT swing up at {args.vmax} V (max cos={best['max_up']:.2f}). "
              f"Voltage too tight -> raise vmax (rail allows ~10 V) and retry.")
    print("\n  best-run trace (t, theta_up deg, arm deg, V, mode):")
    for row in best['log'][:18]:
        print(f"   {row[0]:4.1f}  th={row[1]:+6.0f}  arm={row[2]:+5.0f}  V={row[3]:+5.2f}  {row[4]}")


if __name__ == "__main__":
    main()

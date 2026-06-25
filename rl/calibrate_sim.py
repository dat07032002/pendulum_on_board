"""
calibrate_sim.py — Step 3: check the MuJoCo model reproduces the measured rig behavior.

Runs three experiments in sim and compares to the hardware system-ID:
  1. Free-swing (pole released from ~42 deg): period + amplitude-decay fit
     -> compare to alpha=214 (period 0.43 s) and the swing3.csv decay (rho~0.90, C~4 deg).
  2. Arm coast-down (spin to 16 rad/s, cut power): dw/dt vs w fit
     -> compare to DAMPING/J=13.8, Tc/J=100.
  3. Free-spin terminal velocity vs voltage -> compare KM/DAMPING ~ 19 rad/s/V.

No hardware needed. Gate: sim matches the real curves within ~10%.
"""
from __future__ import annotations

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


def step_ctrl(v=0.0, lock_pole=False):
    D.ctrl[0] = v
    for _ in range(SUB):
        mujoco.mj_step(M, D)
        if lock_pole:                    # hold pole at hanging -> arm-only (matches the
            D.qpos[PA] = 0.0             # decoupled hardware coast-down / terminal tests)
            D.qvel[PV] = 0.0


def free_swing(theta0_deg, secs=4.0):
    mujoco.mj_resetData(M, D)
    D.qpos[PA] = np.deg2rad(theta0_deg)
    t, th = [], []
    for i in range(int(secs / DT)):
        step_ctrl(0.0)
        t.append(i * DT); th.append(D.qpos[PA])
    return np.array(t), np.rad2deg(np.array(th))


def peaks(t, d):
    pk = []
    for i in range(1, len(d) - 1):
        ext = (d[i] > d[i-1] and d[i] >= d[i+1]) or (d[i] < d[i-1] and d[i] <= d[i+1])
        if not ext or abs(d[i]) < 1.0:
            continue
        if pk and (np.sign(d[i]) == np.sign(pk[-1][1])):
            continue
        pk.append((t[i], d[i]))
    return [tp for tp, _ in pk], [abs(a) for _, a in pk]


def coast(w0=16.0, secs=1.5):
    mujoco.mj_resetData(M, D)
    D.qvel[AV] = w0
    t, w = [], []
    for i in range(int(secs / DT)):
        step_ctrl(0.0, lock_pole=True)   # arm alone (decoupled), as measured
        t.append(i * DT); w.append(abs(D.qvel[AV]))
        if abs(D.qvel[AV]) < 0.3:
            break
    return np.array(t), np.array(w)


def terminal(v, secs=1.5):
    mujoco.mj_resetData(M, D)
    for _ in range(int(secs / DT)):
        step_ctrl(v, lock_pole=True)     # arm alone (decoupled), as measured
    return abs(D.qvel[AV])


print("===== MuJoCo model validation =====\n")

# 1) free-swing
t, th = free_swing(42.0)
tp, amps = peaks(t, th)
half = np.diff(tp)
per = 2 * np.median(half[half > 0.05])
pairs = [(amps[i], amps[i+1]) for i in range(len(amps)-1) if amps[i] > amps[i+1]]
A = np.array([p[0] for p in pairs]); An = np.array([p[1] for p in pairs])
rho, negC = np.polyfit(A, An, 1)
print("1) FREE-SWING")
print(f"   sim period   = {per*1e3:.0f} ms   (real ~430 ms, alpha {(2*np.pi/per)**2:.0f} vs 214)")
print(f"   sim decay    = A_(n+1) = {rho:.3f}*A_n - {-negC:.2f} deg   (real swing3: 0.898, 4.07)")
print(f"   sim peaks    = {[round(a,1) for a in amps[:6]]}")

# 2) coast-down
t, w = coast()
dwdt = np.gradient(w, t)
m = (w > 2) & (w < w.max()*0.95) & (dwdt < 0)
sl, ic = np.polyfit(w[m], dwdt[m], 1)
print("\n2) ARM COAST-DOWN")
print(f"   sim DAMPING/J = {-sl:.1f} 1/s   (real 13.8)")
print(f"   sim Tc/J      = {-ic:.0f} rad/s^2   (real 100)")

# 3) terminal velocity
print("\n3) TERMINAL VELOCITY")
vs = [0.5, 1.0, 1.5]
tv = [terminal(v) for v in vs]
for v, w_ in zip(vs, tv):
    print(f"   {v:.1f}V -> {w_:.1f} rad/s")
slope = np.polyfit(vs, tv, 1)[0]
print(f"   sim KM/DAMPING = {slope:.1f} rad/s/V   (real ~19)")

# sanity: upright unstable, hanging stable
mujoco.mj_resetData(M, D); D.qpos[PA] = np.pi - np.deg2rad(2)
for _ in range(int(1.0 / DT)): step_ctrl(0.0)
print(f"\n4) SANITY: from 2 deg below upright -> {np.rad2deg(D.qpos[PA]):.0f} deg "
      f"({'falls (unstable) OK' if abs(np.rad2deg(D.qpos[PA])) < 150 else 'check'})")

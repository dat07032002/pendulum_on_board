"""
view.py — open the MuJoCo viewer and run scripted motions to confirm the physics.

    python rl/view.py

A window opens and cycles through (real-time):
  1. FREE-SWING: pole released ~45 deg, no motor -> should oscillate ~0.43 s period and
     damp out in a few swings (matches the rig).
  2. COUPLING: a slow sine voltage on the arm -> arm rocks, and the pendulum reacts
     (the Furuta coupling). Confirms +V tips the pendulum the right way.
  3. SWING-UP-ish: a few stronger pumps to see it can come up toward vertical.
Close the window (or Ctrl-C) to exit. Drag with the mouse to orbit the camera.
"""
from __future__ import annotations

import os
import time

import numpy as np
import mujoco
import mujoco.viewer

HERE = os.path.dirname(__file__)
m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "furuta.xml"))
d = mujoco.MjData(m)
PA = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "pole")]
AA = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "arm")]
DT = m.opt.timestep


def settle_hanging():
    mujoco.mj_resetData(m, d)          # pole=0 is hanging
    d.ctrl[0] = 0.0


with mujoco.viewer.launch_passive(m, d) as v:
    print("viewer open. phases: free-swing -> coupling -> pumps. Ctrl-C / close to quit.")
    while v.is_running():
        # 1) free-swing from 45 deg
        mujoco.mj_resetData(m, d)
        d.qpos[PA] = np.deg2rad(45.0)
        t0 = time.time()
        while v.is_running() and time.time() - t0 < 4.0:
            d.ctrl[0] = 0.0
            mujoco.mj_step(m, d); v.sync(); time.sleep(DT)

        # 2) coupling: slow sine voltage on the arm
        settle_hanging()
        t0 = time.time()
        while v.is_running() and time.time() - t0 < 6.0:
            d.ctrl[0] = 3.0 * np.sin(2 * np.pi * 0.6 * (time.time() - t0))
            mujoco.mj_step(m, d); v.sync(); time.sleep(DT)

        # 3) pumps toward upright (open-loop energy pumping, bang-bang on pole velocity)
        settle_hanging()
        t0 = time.time()
        while v.is_running() and time.time() - t0 < 6.0:
            thd = d.qvel[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "pole")]]
            d.ctrl[0] = 6.0 * np.sign(thd) if abs(thd) > 0.1 else 6.0
            mujoco.mj_step(m, d); v.sync(); time.sleep(DT)
            if abs(d.qpos[AA]) > np.pi:        # respect the +-180 arm limit
                d.ctrl[0] = 0.0
        time.sleep(0.5)

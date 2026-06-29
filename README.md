# Pendulum on a tilting board

Furuta pendulum simulation, reinforcement-learning training, evaluation, and ESP32 firmware for a
base that tilts about one axis.

## Start here

Read these files in order:

1. `SESSION_2026-06-29_CLEAN91_RESIDUAL_DR.md` — current state and exact next actions.
2. `HANDOFF.md` — complete project context.
3. `rl/POLICY_RUBRIC.md` — evaluation and deployment gates.

## Current verified model

The clean ±20° free-arm master is:

```text
rl/models/clean20_master_verified91p5.zip
```

SHA-256:

```text
775afbb5cf1553becb347c422edc6c03300990134d45c7d2b567f7a9db849d3a
```

It achieved 915/1000 = 91.5% corrected sustained success across two independent 500-episode
blocks. It is not yet plant-DR-, cable-, or hardware-ready.

## Local setup

Use Python 3.12. Install PyTorch for the computer's CPU/GPU first, then:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Quick checks:

```bash
python -m py_compile rl/*.py
python rl/tilt.py
```

MuJoCo rendering on a headless Linux machine may require:

```bash
export MUJOCO_GL=egl
```

## Evaluate the clean master

```bash
python rl/eval_policy.py rl/models/clean20_master_verified91p5.zip \
  --tilt_deg 20 -n 500 --seed0 30000 --arm free \
  --save_npz clean_master_eval_500.npz
```

## Residual DR training

The current method freezes the clean master and trains a bounded action correction:

```text
combined_action = clip(master_action + 0.05 * residual_action, -1, 1)
```

Launch five server seeds with:

```bash
bash rl/launch_residual_dr5.sh
```

Evaluate a residual checkpoint with:

```bash
python rl/eval_policy.py RESIDUAL_CHECKPOINT.zip \
  --residual_base rl/models/clean20_master_verified91p5.zip \
  --residual_scale 0.05 --tilt_deg 20 --dr \
  -n 500 --seed0 30000 --arm free --save_npz residual_eval_500.npz
```

## Important limitations

- The live residual training artifacts remain on the training server until a candidate is verified.
- Firmware still uses the inherited 6-D level-ground policy. Do not deploy the 8-D tilt policy yet.
- Do not select a model from a 50-episode training peak; independently verify with at least 500
  episodes.
- The server environment lives at `~/furuta_rl/.venv`; the project lives at `~/furuta_tilt`.

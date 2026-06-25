# HANDOFF — Furuta Pendulum (GM3506) RL swing-up + balance

Last updated 2026-06-24 22:00 CDT. This is the state + context so another agent can continue.
The full step-plan lives at `~/.claude/plans/async-beaming-teapot.md`.

## ✅ RESULT — WORKING on hardware (on-chip)

**The sim-trained TQC policy runs standalone on the ESP32 (`MODE_RL`) and does the full task:
swing-up + balance + arm-centering + disturbance rejection** (user-confirmed 2026-06-24). This
is the goal — a friction-robust nonlinear controller the classical LQR could not achieve.
Deployed model: `rl/models/fix_sde/best_model.zip` → `rl/policy_weights.h` (also copied into the
sketch as `firmware/furuta_foc/policy_weights.h`).

Path that worked: actuator step-test (real lag ~15 ms = already modeled, so the lag/slew realism
fears were pessimistic) → hardened env (arm-envelope + corner DR) → PC-in-loop sign-check +
balance (signs correct) → **on-chip MLP**. Removing the PC-in-loop serial latency is what made
balance reliable. See "On-chip deployment" + "Tooling added this session" below.

## Goal

A self-balancing Furuta (rotary inverted) pendulum on a **GM3506 gimbal motor**. Classical
LQR balances in sim but **fails on hardware because of pendulum-pivot friction** (the arm
winds to its cable limit). So the current plan: train a **single TQC policy** (MuJoCo +
domain randomization + curriculum) for **swing-up AND balance**, robust to that friction,
and deploy it to the ESP32.

Repo: branch `foc` (GM3506, torque-source). `main` branch = an earlier *working* build on a
different motor (Nidec, velocity-source) — useful reference for proven structure/tooling.

## Hardware

- **MCU:** ESP32 on **COM5 @ 921600**. Control loop 200 Hz, measured zero jitter.
- **Driver:** TMC6300, **open-loop FOC, NO current sensing**, ~11 V rail. This is the root of
  most pain (torque ≈ voltage only approximately; cogging; the FOC offset matters a lot).
- **Motor:** GM3506, 11 pole pairs, R≈5.6 Ω, Kt≈0.068 (datasheet-confirmed).
- **Arm encoder:** AS5048A, SPI, 14-bit. **Pendulum encoder:** AS5600, I2C, 12-bit, addr 0x36.
- **Cable limit:** the arm is limited to ~**±180°** by the **AS5600 cable**. Unplugging the
  AS5600 frees the arm to spin (used that for decoupled motor-only ID).

## Firmware — `firmware/furuta_foc/furuta_foc.ino` (FLASHED, current)

Build/flash: `arduino-cli compile --fqbn esp32:esp32:esp32 --upload -p COM5 firmware/furuta_foc`

- **`balanceStep()` observer-order fix** — compute LQR `V` from the current estimate FIRST,
  then propagate the observer with that `V`. The original order made the combined loop
  unstable (augmented |eig|≈1.17) even though K and L are each stable. (See `sim.py`.)
- **Serial 921600.** `log`/`nolog` stream: `log=[t_ms,phi,theta,phi_dot,theta_dot,V,theta_raw]`
  every tick. Commands: `bal s t<V> k<4> tr<d> hand<d> vlim<V> log nolog raw health
  calhang calup calfoc clearcal pdf<a> params`.
- **NVS persistence** (flash): FOC `dir`/`offset` + AS5600 `UPRIGHT_RAW` survive reboot, so
  it no longer re-sweeps on every boot (cable-safe). `calfoc` re-runs FOC cal, `clearcal` wipes.
- **FOC calibration = bidirectional sweep** → reliably finds the **symmetric** offset (≈300°).
  (The old +only sweep landed on asymmetric offsets, e.g. 270° → 2× torque asymmetry.)
- **Sensors:** I2C 400 kHz, AS5600 slow-filter 2× (CONF SF, ~0.29 ms vs 2.2 ms). **SPI kept at
  1 MHz** (4 MHz corrupted AS5048A reads). `health` prints STATUS/AGC/MAGNITUDE — both
  magnets confirmed healthy (so pendulum friction is the bearing, NOT a rubbing magnet).
- **Velocity filters:** both `phi_dot` and `theta_dot` EMA alpha = **0.5** (≈5 ms lag).
  `phi_dot` was 0.15 (~28 ms) — lightened (runtime-tunable via `pdf <a>`).
- **PWM = default (~1 kHz).** ⚠️ Setting 20 kHz **cut torque ~4×** with this high-side-PWM
  drive scheme — DO NOT re-enable without switching to complementary 6-PWM.
- **Arm soft-limit ±180°** during balance (re-arms instead of winding the cable).
- LQR gains in firmware `Kgain={-4.66,-62.35,-1.53,-4.54}` (correct, but balance fails on the
  friction — that's why we're going RL).

### Firmware gotchas
- Opening the serial port **resets the ESP32** → it loads cal from NVS (no sweep now) and is
  ready in ~2–3 s. `config.Link` waits for the banner.
- **`EncoderServer.exe`** (a separate elevated Windows app) grabs COM5 and **auto-restarts** —
  you must close it (Task Manager, maybe as admin) before flashing or running serial scripts.
- After reconnecting the AS5600 / remounting the magnet, re-verify it reads ~±180° at hang
  (`calhang` if it shifted; `calup` to set true vertical).

## System ID — measured values in `sysid.json`

| Quantity | Value | Notes |
|---|---|---|
| `alpha` = m·l·g/J_p (pendulum pole) | **214** (period 0.43 s) | validated 3–4×; matches geometry |
| Pendulum Coulomb `Tf` | **0.35 mN·m** (~2° deadband) | mixed friction |
| Pendulum viscous | **ζ=0.034**, b_theta=5.06e-5 | viscous-dominant |
| Arm Coulomb | **~6.5 mN·m** (deadzone ~0.3 V) | coast-down + breakaway |
| Arm viscous DAMPING | **9.4e-4** | coast-down dω/dt vs ω |
| Motor `KM` | **0.0127 N·m/V**, J_arm=6.84e-5 | datasheet/geometry |
| Coupling sign | **+V → +θ̇ and +φ̇** | confirmed; model matches |
| Latency | 1 control step (5 ms) | zero jitter |

**KEY findings:**
1. The original FOC offset was 30° off → **2× direction-asymmetric torque** (would've made
   balance impossible). Fixed with bidirectional calibration.
2. **Pendulum-pivot friction is the classical-control killer** — the LQR balances the pole but
   the arm drifts to the ±180° limit in ~1–2 s (no integral action to fight the friction
   offset). The RL approach exists specifically to learn a friction-robust nonlinear policy.
3. Motor electrical params (KM/DAMPING) are **internally inconsistent ~2×** across tests
   (open-loop FOC nonlinearity) → we **domain-randomize widely** rather than trust one value.

### Hardware ID tools (run on the rig, need COM5 free)
`config.py` (serial `Link` + `log` parser), `freeswing.py`, `friction_id.py` (interactive,
self-run: clean releases ~20/35/45°), `actuator_id.py`, `arm_friction_id.py`, `latency_id.py`,
`sign_check.py`, `check_as5600.py`, `console.py` (interactive serial REPL). `plant_torque.py` /
`balance_torque.py` / `sim.py` = the torque-model LQR design + closed-loop catch sim.

## RL — `rl/`

- **`furuta.xml`** — MuJoCo model from the measured params. Validated by `calibrate_sim.py`
  (free-swing period/decay, arm coast-down match real within ~10–15%). **Pole joint axis is
  `-1 0 0`** so the sim's `+V→+θ̇` matches the firmware (sim↔hardware obs/action map 1:1).
- **`furuta_env.py`** — Gymnasium env. **Obs `[cosθ, sinθ, θ̇/15, φ/π, φ̇/25, prev_action]`**,
  θ=0 at upright. `prev_action` is included so the memoryless actor can cope with the modeled
  1–2 step action delay; deployment must use the same 6D input/order. Action ∈[-1,1]→±6 V.
  EMA filters α=0.5 (match firmware). **No VecNormalize** (obs pre-normalized → simple
  deployment). `check_env` passes.
  - **Reward:** `cos(θ)` backbone (swing-up + balance); velocity penalty **gated to the upper
    half** (don't punish pumping); **+2 bonus AND `is_success` gated on arm `<90°`** (so it
    can't "succeed" by balancing at the cable edge — kills the arm-drift loophole); CAPS-style
    action-smoothness `−0.02(Δa)²`; arm-center `−0.2(φ/π)²`; control `−0.005a²`; **−10 +
    terminate at |φ|>180°** (cable).
  - **Domain randomization (per episode):** KM[0.008–0.020], arm damping[3e-4–10e-4], arm
    frictionloss[4e-3–8e-3], pole frictionloss[0.2e-3–0.6e-3], pole damping[2e-5–1e-4],
    inertia ±8%, obs noise, action latency 1–2 steps.
- **`train_tqc.py`** — **TQC** (sb3-contrib), actor **[64,64]** (ESP32-portable), critic
  [256,256], **gSDE by default** (`use_sde`, disabled with `--no_sde`),
  `gradient_steps=nenv//2`. **5-stage curriculum** (balance ±10° → ±45° → ±90° →
  near-hanging+assist → full swing-up), advances when rolling success >0.7. EvalCallback saves
  `models/best_model.zip` by deterministic eval mean reward.
- **`calibrate_sim.py`** (model validation), **`view.py`** (MuJoCo viewer demo — local only).

### Why TQC + gSDE + CAPS (researched, peer-reviewed)
- **CAPS action smoothness** (ICRA 2021) — smooth control transfers far better to real motors
  (reduces the vibration/limit-cycle we measured). Applied as the `−0.02(Δa)²` reward term.
- **gSDE** (CoRL 2021, in SB3) — smoother exploration → more deployable policies.
- Cosine reward + `[cos,sin,θ̇,φ,φ̇]` obs are the validated standard for Furuta RL; this env adds
  `prev_action` as a practical delay-compensation input.
- TQC ≈ SAC for this small task but a free upgrade (only the actor is deployed, so no ESP32 cost).

## Training server (UT)

- `ssh -i ~/.ssh/aere_codex_ed25519 tn22833@aere-a83514.ae.utexas.edu` — **requires UT VPN**
  (hostname won't resolve otherwise). Key already authorized.
- **5× RTX 6000 Ada (48 GB), 256 cores, no SLURM.** Be a good citizen: use **one GPU**
  (`CUDA_VISIBLE_DEVICES=0`), modest `nenv`; it's shared.
- Our project: **`~/furuta_rl/`** (venv `.venv`, code in `rl/`, logs `train.log`). **Do NOT
  touch `~/pendulum/`** — that's prior PPO work from a different session.
- **Env gotcha:** the driver is CUDA 12.7, so torch must be a **cu124** build (`pip install
  torch --index-url https://download.pytorch.org/whl/cu124`); `tensorboard` must be installed.
- Launch example: `cd ~/furuta_rl && CUDA_VISIBLE_DEVICES=0 nohup ./.venv/bin/python
  rl/train_tqc.py --steps 5000000 --nenv 16 --tag fix_sde > train_fix_sde.log 2>&1 &`.
  Monitor: `grep curriculum train_fix_sde.log`,
  `grep -E 'success_rate|ep_rew_mean' train_fix_sde.log | tail`.

## CURRENT STATE (in progress)

- **Two TQC A/B runs are live on the UT server** (`~/furuta_rl/`, 5M target, `nenv=16`):
  - `fix_sde` (PID 2391852): `./.venv/bin/python rl/train_tqc.py --steps 5000000 --nenv 16
    --tag fix_sde`
  - `fix_nosde` (PID 2344750): `./.venv/bin/python rl/train_tqc.py --steps 5000000 --nenv 16
    --no_sde --tag fix_nosde`
- Both reached **stage 4 = full swing-up**. The theoretical max episode return is ~**6000**
  (`3.0` reward/step × 2000 steps).
- Latest check (2026-06-24 19:34 CDT):
  - `fix_sde`: ~848k steps, rollout reward ~1.6k, rollout success ~0.42. It previously looked
    much better around 600–760k steps (rollout reward ~4.8k, success ~0.89; deterministic eval
    at 740k: reward ~4.13k, success 80%) but has since regressed. **Use best_model/eval
    artifacts, not simply the final checkpoint. Watch for instability.**
  - `fix_nosde`: ~1.09M steps, rollout reward ~0.55–0.61k, rollout success ~0.32–0.34; latest
    deterministic eval at 1.08M: reward ~986, success 50%. It is generally weaker and more
    inconsistent than SDE.
- Earlier local run revealed the **arm-drift loophole** (policy balanced but arm wound to
  limit); fixed by gating bonus/success on arm<90°. That fix is in the current env.

### Validation snapshot — `rl/models/fix_sde/best_model.zip`

Validated on the server 2026-06-24 19:xx CDT with deterministic actions, full swing-up initial
conditions, 10 s episodes:

- **Nominal/no-DR:** 20/20 success, mean reward **5055/6000**, full 2000-step episodes,
  saturation ≈0%, smooth actions (`mean|Δa|≈0.02`). Good.
- **Random DR:** 100 episodes, **88% success**, mean reward **4452/6000**, median reward 5157,
  but 12 failures and some high-action episodes (`p90` saturation ≈32%). This clears the
  rubric's ≥80% randomized-DR success bar, but with visible tail risk.
- **Corner sweep** over low/high KM, arm friction, pole friction, and 1/2-step delay: **11/16
  success = 68.8%**, mean reward 3751. This does **not** pass worst-case robustness.
  Worst failures cluster around high KM + 2-step delay and one low-KM/low-arm-friction/high-pole-
  friction case.
- **Arm margin warning:** even successful nominal/random runs often used arm excursions above the
  90° success/bonus gate (nominal mean max |φ|≈110°, random mean max |φ|≈112°, worst random hit
  the ±180° cable limit). It can succeed, but not yet with comfortable cable margin.

Verdict: `fix_sde/best_model.zip` is the best candidate so far and is **sim-good on average DR**,
but **not deploy-ready** until it passes a harder worst-case sweep and shows better arm/cable
margin. Consider adding checkpoint validation, stronger arm-margin shaping, or narrower/safer
hardware initial tests before PC-in-loop.

### Validation snapshot — `rl/models/fix_nosde/best_model.zip`

Same deterministic validation battery, full swing-up initial conditions, 10 s episodes:

- **Nominal/no-DR:** 19/20 success, mean reward **5185/6000**, smooth and essentially no
  saturation. One early failure; successful cases often looked very good.
- **Random DR:** 100 episodes, **74% success**, mean reward **3846/6000**, median reward 5117,
  26 failures. This **does not pass** the rubric's ≥80% randomized-DR success bar.
- **Corner sweep:** 12/16 success = **75%**, mean reward 2668. Slightly higher corner success
  count than SDE in this small grid, but much lower reward/episode length and still not
  worst-case robust. Delay=2 is a recurring failure mode.
- **Arm margin warning:** similar cable-margin issue as SDE. Random max |φ| mean ≈112.5° and
  worst cases hit/exceeded the ±180° cable limit.

Comparison: `fix_sde/best_model.zip` is still the preferred candidate because it passes average
randomized DR (88% vs 74%) and has higher mean reward. `fix_nosde` is smoother/less saturated in
some nominal cases but less robust across randomized DR. Neither is deploy-ready because both
fail worst-case robustness and both use too much arm travel.

### Fast swing-up realism ablation — `fix_sde/best_model.zip`

Ran read-only ablation tests after the GIF looked suspiciously fast. Baseline deterministic rollout
from hard full-swing-up starts (`θ0 = ±180°, ±150°, ±120°`) catches upright extremely quickly:
median first upright ≈ **0.23 s**, median first 0.5 s stable streak ≈ **0.29 s**, mean max arm
travel ≈ **115°**, max ≈ **133°**.

Key ablation results:

- **Actuator dynamics are the biggest realism suspect.**
  - Baseline: **100%** success.
  - Add 20 ms first-order motor lag: **33%** success.
  - Add 50 ms or 100 ms lag: **0%** success.
  - Add voltage slew limits of 40/20/10 V/s: **0%** success.
  - Interpretation: the policy relies on near-instant, clean torque. Real open-loop FOC voltage
    control will likely be slower/nonlinear.
- **Arm envelope is too permissive.**
  - Hard arm stop at ±150°: **100%** success.
  - Hard arm stop at ±120°: **50%** success.
  - Hard arm stop at ±90°: **0%** success.
  - Interpretation: the policy is not a polite ±90° swing-up; it uses the arm as a big flywheel.
- **Motor authority matters.**
  - Corrected KM scaling: KM×1.00 → **100%**, KM×0.75 → **17%**, KM×0.50 → **83%** but slower/
    near cable limit, KM×0.35 and below → **0%**.
  - Interpretation: if real effective torque is lower than sim, the fast swing-up will not transfer.
- **Voltage cap matters less than lag/slew.**
  - Clamp to ±4 V or ±3 V: **100%** success but slower catch.
  - Clamp to ±2 V: **50%** success.
  - Interpretation: the issue is not just high voltage; it is ideal instantaneous torque.
- **Deadzone sensitivity.**
  - 0.3 V deadzone: **100%** success.
  - 0.6 V deadzone: **50%** success.
  - 1.0 V deadzone: **17%** success.
- **Reward is too forgiving of wide/fast swing-up.**
  - Counterfactual rescoring with stronger arm penalty / soft >90° penalty only modestly reduces
    the baseline rollout score. Current reward mainly pays for early upright time.
- **Hand-coded `feasibility.py` sanity check:** energy-pump + LQR can fling the pole upright
  quickly in the same MuJoCo model, though it does not hold. This supports the idea that the model
  permits aggressive energy injection.

Narrowed diagnosis: **primary causes = ideal/instant actuator model + weak arm-envelope shaping**;
secondary causes = motor-gain sensitivity + reward overvaluing early upright time. SDE itself is
not the main culprit. The balance behavior is more believable than the swing-up. Next training
iteration should add actuator lag/slew/deadzone/back-EMF-ish effects and enforce a safer arm
envelope (e.g. strong penalty past 90°, hard/soft limit before 120°) before hardware swing-up.

> **NOTE:** the snapshot above (regression/realism worries) is the *pre-deployment* analysis.
> It was largely resolved by what follows. In particular the swing-up **did transfer** — the
> lag/slew ablations were pessimistic (the step-test below measured the real lag at ~15 ms,
> already modeled). Keep the analysis for context, but the on-chip result is the ground truth.

## Step 1 — actuator step-test (settled the "fast swing-up unrealistic?" worry)

`step_response.py` (drives the arm, ±3 V step ×3): **real response = dead 15 ms, τ 10 ms**, very
consistent. Running the *same* step in sim gives the **same 15 ms dead time** (the modeled 1-step
delay + α=0.5 EMA filter reproduce it). So there is **no large unmodeled lag** → the ablation's
20–50 ms lag and 40 V/s slew scenarios were worse-than-reality (a PWM driver slews far faster).
The real transfer risks were KM-sensitivity + arm-flywheel, not lag — addressed by the hardening.

## Env hardening (this session, in `rl/furuta_env.py`)

- **Arm-envelope shaping:** added `−0.5·max(0,|φ|−90°)²` so the policy can't use the arm as a
  180° flywheel (kills the cable risk; balanced reward now +3 centered → −0.4 at 180°).
- **Corner-weighted DR:** `p_corner=0.3` — 30% of DR draws are pushed to a min/max extreme so
  worst-case corners (the failed corner-sweep cells) are actually trained, not just the center.
- **Response-time DR:** action delay widened 1–2 → **1–3 steps**.
- `train_tqc.py`: added `CheckpointCallback` (periodic `ckpt_*.zip`) to recover the peak if a run
  regresses. The v2 retrain (`v2_sde` GPU2, `v2_nosde` GPU3) uses this hardened env.

## On-chip deployment (`MODE_RL`) — the working path

- **`rl/export_policy.py`** dumps the actor MLP (`6→64→64→1`, ReLU, **tanh(clip(mu,±2))**) to
  `rl/policy_weights.h` and verifies the numpy forward matches SB3 to <1e-6, then it's copied to
  `firmware/furuta_foc/policy_weights.h`.
  - **GOTCHA (cost an hour):** gSDE actors have `clip_mean=2.0` → `mu = Hardtanh(±2)` before tanh.
    Miss it and the export only diverges at action saturation. The C forward replicates the clip
    (`RL_CLIP_MEAN`). `pc_balance.py` was fine because it calls `policy.predict` (clip is internal).
  - Also: SB3 `predict` on **CUDA** differs from CPU by ~0.03 at saturation (matmul order); verify
    the export on **CPU** (`TQC.load(..., device="cpu")`), which is the ESP32's float32 path.
- **Firmware `MODE_RL`** (`rlForward` + `rlStep`, `rl` command): builds obs
  `[cosθ,sinθ,θ̇/15,clip(φ/π,±2),φ̇/25,prev_action]` from the same sensors, runs the MLP at 200 Hz,
  `V=6·action` (clipped to `vlim`). `rl` recenters the arm (`rl_phi_ref=phi_full`) at engage so
  arm-centering matches the sim. Cable guard at ±160°. C-indexed forward verified vs SB3 to 6.6e-7.
- **`pc_balance.py`** (PC-in-loop, `--dry` sign-check / live with `--sflip_*`, `--vlim`): used to
  confirm sign alignment + first balance before going on-chip. Kept for future PC-in-loop tests.

**Test recipe:** `console.py` → confirm `raw` th≈0 at upright (`calup` if not) → `vlim 5` → hold
rod upright → `rl` → `s` to stop. (PC-in-loop equivalent: `python pc_balance.py --vlim N`.)

## NEXT STEPS (goal met; these are polish/robustness)

1. **Pull the v2 model when the retrains finish** (`v2_sde`/`v2_nosde` on the UT server, hardened
   env). Re-export + flash; it should be *gentler* (arm-envelope) and more corner-robust than the
   deployed `fix_sde`. Select by `best_model.zip`/checkpoints, not final reward.
2. **Quantify on hardware:** swing-up success rate over N trials, disturbance-rejection limit,
   steady-state arm drift, action smoothness/buzz at `vlim 5` vs `vlim 6`. Judge vs
   `rl/POLICY_RUBRIC.md`.
3. **Tune if needed:** if buzzy → CAPS weight up / retrain; if arm drifts → arm-center weight up.
4. **Robustness/iterate (`Step 9`):** if v2 underperforms, re-tune sim friction from fresh `log=`
   data, widen DR, retrain. Server runs/logs in `~/furuta_rl/`.

## Open design questions
- Arm success/bonus gate at **90°** (margin from the 180° cable). Could relax to 120° if
  swing-up+recenter is too hard. (User aware; default 90°.)
- 5M steps may not be enough for full swing-up within ±180°/±6 V — extend if stage 4 stalls.
  If swing-up is infeasible at the low-KM end of the DR range, that's a real authority finding
  (need more arm travel or voltage, or accept balance-only + manual lift).

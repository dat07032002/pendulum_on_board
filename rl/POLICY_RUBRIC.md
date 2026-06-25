# Policy acceptance rubric — Furuta swing-up + balance (TQC)

How we decide a trained policy is good enough to deploy. Sim passes are **necessary but not
sufficient** — the real judge is the hardware. Use this for Step 6 (validate) before Step 7
(deploy). Evaluate the **deterministic** policy (mean action, no exploration noise).

## Pass 1 — deterministic eval (run ACROSS the domain-randomization range, not just nominal)
- **Success rate ≥ 80%** across randomized friction / KM / latency (not just nominal sim).
- **Hold time ≈ full episode** (sustained balance, not just clearing the 0.5 s success bar).
- **Action not always saturated** (headroom; not bang-bang ±6 V).
- **Action smooth** — low high-frequency content / small `mean|Δa|` (CAPS metric). A jerky-but-
  unsaturated policy still excites the open-loop-motor vibration / limit-cycle and won't transfer.

## Pass 2 — multiple seeds
- Consistent across **3–5 seeds** (and/or parallel configs). Not one lucky run. RL is high-variance.

## Pass 3 — region-of-attraction (RoA) handoff
- The **states swing-up delivers** the pole into must lie **inside the balance controller's
  catchable region** (otherwise the two work alone but fail combined).
- **Arm angle and action stay realistic/physical** throughout (no exploiting sim quirks).

## Pass 4 — sim-to-real readiness (the gap the sim passes don't cover)
- **DR robustness** = Pass 1 re-stated: success must hold across the measured ~2× motor-param
  spread, not at nominal only. This is the #1 predictor of real-world success.
- **Latency robustness** — survives the real 1-step + EMA-filter lag (modeled in the env; verify
  it isn't relying on instantaneous reaction).
- **Sign alignment check (before trusting hardware)** — cos(θ) is sign-immune, but `sinθ`, `θ̇`,
  and the action sign must match the firmware (analog of the LQR sign check). Verify at PC-in-loop.

## Pass 5 — the final judge: real hardware
- **PC-in-loop** on the rig: swings up / holds (latency-limited, for validation).
- **On-chip MLP** (`MODE_RL`): standalone swing-up + balance, arm soft-limit intact; compare vs
  the LQR baseline (`bal`).

## Quick reference
sim-good = Pass 1–3 (under DR) + smoothness. deploy-ready = + Pass 4. done = Pass 5 on hardware.

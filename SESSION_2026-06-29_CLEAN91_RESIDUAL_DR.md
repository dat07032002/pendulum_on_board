# Clean ±20° master and residual-DR session — 2026-06-29

This is the newest project record. It supersedes the immediate actions in `HANDOFF.md` and
`SESSION_2026-06-27_TO_29.md`.

## Current state

- Clean ±20°, free-arm sustained success improved from **73.2% (366/500)** to a verified
  **91.5% (915/1000)**.
- Verified server-side master:
  `rl/models/clean20_master_verified91p5.zip`
- SHA-256:
  `775afbb5cf1553becb347c422edc6c03300990134d45c7d2b567f7a9db849d3a`
- Verification blocks:
  - seeds 30000–30499: **90.4% (452/500)** sustained;
  - seeds 40000–40499: **92.6% (463/500)** sustained;
  - combined catch success: **95.5%**.
- The master is free-arm and clean-plant only. It is not cable- or DR-ready.
- Five bounded-residual DR runs are currently active on server GPUs 0–4.

## Clean-master work

The original clean warm start scored 73.2% sustained and 90.0% catch over 500 fresh episodes.
Failures were dominated by caught-then-lost episodes and high-amplitude/high-rate tilt.

Gentle retention fine-tuning improved one checkpoint to 88.6%. A targeted continuation that
oversampled 70–100% tilt amplitude and 1.2–2.0 rad/s tilt-rate caps produced the 91.5% master.

Visual artifact:

- `clean20_verified91p5.gif`

## Evaluation and export improvements

`rl/eval_policy.py` now reports and can save:

- sustained/catch success with Wilson 95% intervals;
- action smoothness and saturation;
- true-vertical quality;
- arm excursion and cable margin;
- performance near `phi=±90°`;
- realized tilt and DR parameters;
- paired checkpoint comparisons and per-episode NPZ evidence;
- combined frozen-base plus residual-policy evaluation.

`rl/export_policy.py` now supports current non-gSDE and legacy gSDE actor layouts, handles mean
clipping correctly, verifies NumPy inference against SB3, detects 6-D/8-D input, and embeds model
SHA-256 provenance. The verified 8-D clean master matched SB3 within `2.15e-6`.

No 8-D firmware header has been deployed. Firmware remains the inherited 6-D implementation.

## Direct retention DR attempt — stopped

Five seeds used the verified master, fresh successful teacher replay, gradual stage-5 DR, actor LR
`1e-6`, and 25% teacher replay. Four stopped on clean-retention loss; the fifth was manually
stopped after showing the same trend.

The main failure was not the DR range. The actor-retention objective was ineffective:

- RL actor loss was approximately 1,300;
- `teacher_coef × teacher_loss` was approximately 1;
- retention therefore contributed less than 0.1% of the actor objective;
- the actor received roughly 10,000 updates from 100k to 120k environment steps.

Final action mean-absolute deviation from the master reached 0.20–0.64 depending on seed. Clean
success collapsed before DR success improved. Do not resume these runs.

Artifacts/logs:

- `train_clean20_dr91_s0.log` through `train_clean20_dr91_s4.log`
- `rl/models/clean20_dr91_s0/` through `s4/`
- `rl/teacher_clean20_master91_200k.npz`

## Bounded residual method

`rl/residual_env.py` freezes the verified master and trains a separate policy whose normalized
action is multiplied by a hard residual bound:

```text
combined_action = clip(master_action + 0.05 * residual_action, -1, 1)
```

The master weights cannot drift. The current bound limits action correction to ±0.05.
`FrozenNumpyActor` matched SB3 inference within `3.8e-6`.

Training configuration:

- five seeds, GPUs 0–4;
- tags `residual_dr_s0` … `residual_dr_s4`;
- start stage 5, max stage 9;
- 2M-step ceiling, `nenv=8`;
- residual scale 0.05;
- residual TQC LR `1e-4`, fixed entropy coefficient 0.01;
- hard tilt training: amplitude fraction 0.70–1.00, rate cap 1.2–2.0 rad/s;
- clean guard 0.70;
- full-DR target evaluation every 20k steps.

Launch script: `rl/launch_residual_dr5.sh`.

## Live residual snapshot

Snapshot around 120k steps, all at stage 5:

| Seed | Latest full-DR | Best trained full-DR | Clean |
|---|---:|---:|---:|
| 0 | 42% | 44% | 98% |
| 1 | 52% | 52% | 90% |
| 2 | 46% | 46% | 86% |
| 3 | 38% | 40% | 90% |
| 4 | 42% | 58% | 92% |

The residual method has so far solved the forgetting symptom: clean performance remains 86–98%.
DR improvement is modest and no seed has advanced beyond stage 5 yet. Seed 4 has the highest
training-eval peak, but no residual checkpoint has passed independent 500-episode verification.

## Exact next actions

1. Continue monitoring the five residual runs.
2. Do not select from a 50-episode peak.
3. If a checkpoint shows a sustained improvement, independently evaluate it with:

   ```bash
   python rl/eval_policy.py RESIDUAL.zip \
     --residual_base rl/models/clean20_master_verified91p5.zip \
     --residual_scale 0.05 --tilt_deg 20 --dr \
     -n 500 --seed0 30000 --arm free --save_npz residual_verify_500.npz
   ```

4. Verify clean ±20° retention separately.
5. If all seeds remain stuck at stage 5, do not increase the residual bound blindly. First test
   individual DR groups (actuator/delay, mechanical loss, inertia, sensing) to identify which
   uncertainty requires more correction authority.

## Local working tree

The session changes are currently uncommitted. They include evaluator/exporter fixes, residual
training/evaluation support, launch scripts, the GIF, and this documentation. Do not discard them.

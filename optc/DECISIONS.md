# Option C decisions

Choices the plan did not settle, the more conservative option taken each time. Newest last.

## C0 (2026-11-09)

1. **D1, real falls too early for the plan's window.** The plan gives the simulator "the true 32-step history before the last 50 steps", which needs 81 steps. Most failed-policy falls come sooner (Step 2's stored episodes: 44 of 80 for `A_s0__p0`, 72 for `B_s0__p0`, 62 for `A_s0__u50__p0`). They are not dropped: `full` is the plan's window (81 steps or more); `short` (33–80 steps) starts the forecast straight after the first 32 rows; `too_short` (under 33 steps) has no history to give and is only counted. The table reports each group separately.
2. **D1, "within 0.2 s".** A rollout has no joint commands after the real fall, so a hit is a predicted fall (the lite threshold, 0.5) from 10 steps before the fall to the fall itself. Predictions earlier than that are counted as early alarms. Each is reported for the ensemble mean, the average over members (what imagination sees) and any member.
3. **D1, imagined actions.** "As policy training does" means sampled actions, using the final checkpoint's noise level; this is the primary figure and the source of D2's pairs. The noise-free action, which is what MuJoCo measures, is reported beside it.
4. **D1 and D2 use the rebuilt route for `REL`** (`common.load` and the adapter). Step 1 showed the adapter equals the upstream class on the released weights.
5. **D2 sample size:** 10,000 points per set, drawn with a fixed seed. The plan gave no number.
6. **D4 commands** are the 21 regime means of the recording, which are reconstructions rather than logged commands. Coverage is the exact union volume, with a Monte Carlo cross-check. The eight training episodes are reported beside all ten.
7. **D3 outputs** go to `runs/c/` and `results/c/` by setting the module paths of Step 2's unchanged `train_policy` and `evaluate`, including the lite log directory (`runs/c/d3_lite/`).

**Interpretation limits (not choices):** the MuJoCo world is a hand-built model with a fitted PD motor; D2's authors' pairs are MuJoCo pairs measured against data recorded in a different simulator, so that distance includes the sim-to-sim gap.

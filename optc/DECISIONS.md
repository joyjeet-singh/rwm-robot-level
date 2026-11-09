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

## C1 (2026-11-09)

1. **Batch size is 256, not 1,024.** The plan says "the rebuild's batch size". `A5` was trained with 256 (`results/step5_armA_seed0_ens5.json`, `hyperparameters.batch`; `step5_train.py --batch` defaults to 256). 1,024 is the lite config's value, which `optc/notes/C0_facts.md` lists; that note is wrong for the rebuild. The reproduction test (first 20 losses) confirms 256.
2. **Rows per episode.** 999 steps plus the reset row give the recording's 1,000 rows. A row counts as one transition, as the recording's "10,000 rows" do. Row 0 is the reset state with a zero action; its torques are zero (the recording's reset rows carry the previous step's torques and contacts, which do not exist here).
3. **Foot friction** applies to the foot sphere and shank tube (the URDF foot link). MuJoCo uses the larger friction of two touching shapes, so the feet get contact priority 1; otherwise values below the floor's 0.6 would do nothing. A test confirms the drawn value reaches the contacts.
4. **Payload** adds to the base mass and scales its inertia in proportion. **Motor gains:** Kp and Kd are drawn independently, each ±10%, one pair for all 12 joints (as in the fitted motor).
5. **Pushes:** a horizontal velocity change in a random direction, its size uniform from 0 to 0.5 m/s, so it never exceeds 0.5 m/s.
6. **Action noise** is added to the joint command, not clipped, and is what the robot receives and what the next observation reports.
7. **Smooth random joint commands:** each joint an independent first-order filtered noise around the standing pose, time constant 0.5 s, standard deviation 0.5 action units (0.25 rad).
8. **Commands for the expert test set and D6** are the generator's uniform commands. "Like the paper's recorded data" is taken to mean the policy, calibrated physics and no pushes.
9. **Rungs and shares.** A rung takes whole episodes in one fixed random order per group, cut at the whole-episode boundary nearest its target, so rungs are nested. The 22 trained policies form one group whose next episode comes from the policy with the fewest transitions so far. At 10k, whole episodes mean not all 22 appear, and the rung's size is only approximately 10,000. Generation slightly exceeds each source's target (under about 2%); the 6M rung uses the nearest cut and the surplus stays unused.
10. **Mixed test set:** 40 episodes split by the shares (12, 5, 5, 10, 4, 4); each trained-policy episode uses a randomly chosen policy.
11. **Divergence.** If MuJoCo warns or a state is not finite, the episode ends there with no fall label, and the bad step is not stored.
12. **Checks** use the paper's 400-row windows, so episodes shorter than 400 rows give none; they are counted (`episodes_too_short`) and excluded. Normalisation and residual scales come from each simulator's rung.
13. **Probe stand-in.** The 1M rung does not exist yet, so the training probe tiled about 196,000 generated rows to 1.18M, using training shards only. Timing and memory do not depend on the values.
14. **Training layout: one at a time, 4 threads.** Measured peak for two trainings at once on the 1M stand-in was 3.56 GB, under the 6 GB limit. `results/c/probe.json` holds a further bound for the 6M rung (6.09 GB) that adds the larger file's resident pages once per process; it counts the shared file twice, so it overstates physical memory. The probe's conclusion was left as written rather than revised after seeing it, as the more conservative choice. It should cost little: one training alone runs about twice as fast as two at once (`A5` took 5.8 s per step alone; two at once took 11.2 s here), so S and total hours are about the same. S = 2,572 by the design's rule, from the two-at-once N = 8 step time.

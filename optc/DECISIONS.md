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

## C2 (2026-11-10)

1. **Start states, "at most 100,000 rows".** Read as a cap on the rung's rows, which also caps the windows. A rung of 100,000 rows or fewer is used whole. Otherwise whole episodes are taken in a fixed random order that depends only on the rung (not the policy seed), until the next would not fit. Shares hold only approximately (random episodes, not drawn within source).
2. **"End at least 1 s before any fall"** means the fall row is at least 50 rows after the window's last row. Episodes that end without a fall are usable to their last row. Tilt is the angle between the base's up axis and vertical, from projected gravity.
3. **Which command scores a step.** The command the step was taken under, as the task's own reward uses it. Step 2's speed and turn errors keep their own alignment (the next row's command). The two differ on one step of 1,000.
4. **The step on which a fall happens is scored; every later step scores 0.**
5. **Divergence.** A non-finite state or a MuJoCo divergence warning (which would silently restart the robot) ends the episode and counts as a fall; the count is reported. On normal episodes the loop equals Step 2's (test).
6. **Protocol seeds.** Schedules and starts use a new seed, 20275, not Step 2's. The standing-still reference on these 24 schedules is 0.069; the plan's "about 0.09" is the average over all uniform commands, and 48 draws differ from it by about one standard error.
7. **"The ten recorded schedules"** is Step 2's protocol unchanged: 10 schedules × 8 starts, 80 episodes.
8. **Intervals.** "95% intervals over episodes" resamples the 96 episodes, and Gate C uses that. Step 2's version (starts within schedules) is reported beside it and is not used by any gate.
9. **Gate C compares** the 10k policy (weight 1.0; the pilot trains no other) with the kept 6M policy, whatever its weight.
10. **Retry, "provided it passes Gate B"** is read as the highest scorer among the weights that pass.
11. **Saves** are taken after exactly 500, 1,000, 1,500 and 2,000 finished iterations, through a hook on the lite log. The lite code's own save at iteration index 500 comes one update later.
12. **An interrupted policy restarts from iteration 0,** as in Step 2, so each policy is one uninterrupted, seeded run. Finished policies and measured saves are skipped.
13. **A BLOCKED pilot** is recorded by `scripts/c/pilot_rule.py` in this file and in `optc/LOG.md`, and the job exits 3, which stops the launcher.

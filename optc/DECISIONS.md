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

## C2R (2026-11-11)

1. **Amendment 1's account, one partial disagreement.** In the first 6M run the summed penalties outweighed tracking in every iteration with a finished episode, as the amendment says. But vertical speed plus tilt exceeded the uncertainty charge in only 71 of 1,994 such iterations, 52 of them before iteration 100. From about iteration 10 onwards, the uncertainty charge was usually the largest single penalty. Also, after iteration 1 the lite log shows disagreement of 2.41 (released model), 2.02 (6M), 2.71 (10k) and 2.67 (`A5_s0`), not quite the amendment's 2.0–2.2, 2.3 and about 2.8. Neither fact makes a change impossible; recorded, not blocking (`optc/notes/C2R.md`).
2. **Calibration ran in the session** (about 1.5 min, under the 10-minute limit). Closed-loop imagination worked, so the open-loop fallback was not used. ū averages the lite disagreement over every step each of 1,024 robots spends in its first episode (seed 20278, policy mean).
3. **Start states in calibration.** For our simulators: the amendment's rule (source 0, own rung). For the released model and `A5_s0`: the authors' recorded data, as the lite code starts. `A5_s0` was trained on those data, so it keeps Step 2's hold-out.
4. **Gate A2's MuJoCo reference** is a re-run of the authors' policy on `results/c_authors.json`'s protocol, with the reward recorded per term (`results/c/authors_terms.json`). Its walking score equals `c_authors.json`'s exactly. Tracking is compared per 0.02 s step: imagination counts the steps before each episode ends.
5. **Gate A2 failed** (falls 15.5%, tracking 31% of MuJoCo's). The job retrains first.
   - It retrains from scratch: same seed and code, 2 × S = 5,144 steps, one at a time on 4 threads (C1 item 14).
   - The new simulators are named `sim_6M_N8_s0__2S` and `sim_10k_N8_s0__2S`. The S-step simulators are kept.
   - Checks go to `results/c/pilot_sims_2S.json`, and calibration to `results/c/calibration_2S.json`, which re-measures the released model.
   - λ for the retrained simulators is read from that file.
   - If Gate A2 passes there, C3 must use 2 × S for the whole pool.
6. **Noise floor:** a clamp to 0.05 after every optimiser step, and at the start. The lite policy uses a scalar standard deviation; the log-scale case is handled the same way.
7. **Fall cost:** subtracted once, at the step an episode ends in a predicted fall. A time-out at the 256-step limit is not charged. Each iteration logs the share of predicted falls and the charge per step.
8. **Early stop.**
   - A mean length logged after iteration n counts for iteration n, and the stop fires when iterations n−99 to n all come after 300 and are all under 64 (strictly). So the earliest stop is at 400.
   - An iteration with no finished episode breaks the streak, the conservative reading.
   - The stopped actor is kept as `actor_stopped_i<n>.pt` and is not measured; only regular saves are.
   - The rule does not fire on `REL__p0`'s metrics. On the first pilot it would have fired at 1,549 (6M) and 400 (10k).
9. **Names:** `<sim>__cal__p0`, `__cal_fall4__p0`, `__w0__p0` and `__w0_fall4__p0`, added to the evaluator's pilot list (with the `__2S` simulators). The guard is otherwise unchanged.
10. **Gate B':** an arm's best save is its highest score, and a tie goes to the earlier save. An arm stopped before iteration 500 has no save and fails.
11. **Gate C''s 10k policy** runs without the early stop, which the amendment defines for the pilot's arms. It trains alone on 4 threads and is measured at the chosen iteration count. Its intervals are the episode bootstrap (C2 item 8).
12. **The 10k rung has only 3 source-0 episodes** (2,907 start windows). The rule is applied as written. This narrows the 10k policy's starts; keep it in view at Gate C'.
13. **`results/c_pilot1.*`** is written once by the session. The job adds only the measured surviving saves, after checking that every other value is unchanged; the plan names this addition. The weight-2.0 run crashed before the lite code wrote its metrics, so it has none.
14. **The "nonzero uncertainty" test** checks the disagreement (> 0) and the penalty weight. After one iteration the logged penalty term is 0, because terms are logged when episodes reset and none had ended.
15. **Layout and time.** P1 and P2 run together on 2 threads each, then P3 and P4. Evaluations use 1 torch thread each.
    - Estimates from measured rates: 2 × 7.9 h to retrain the simulators (5.5 s per step), 2 × 6.7 h for the arm pairs (16 s per iteration two at once), and about 4 h for the 10k policy, plus measuring.
    - Total: about 35–40 h. These are estimates, not measured on this job.
16. **The first pilot's results are committed** with this session: `results/c/pilot_decision.json`, `results/c/eval/` and `results/c/train/`. The job left them uncommitted, and `results/c_pilot1.*` is built from them. `results/c/progress.json` and `results/s2/progress.json` are status files and stay uncommitted.

## Pilot job: BLOCKED at Gate A2 (2026-11-11 15:38)

**What is blocked:** Gate A2 failed twice (sim_6M_N8_s0: fell 15.5%, tracking 31% of MuJoCo's; after retraining with 2 x S steps, sim_6M_N8_s0__2S: fell 26.6%, tracking 49% of MuJoCo's; limits: at most 10% falls and at least 80% of MuJoCo's tracking). The simulator cannot hold the authors' walk in imagination; the plan says to stop.

**Options:** (a) stop Option C and report Gate A2 as the finding (the simulators, even with twice the training, do not hold the authors' walk); (b) a further change to the simulator's training (more steps, a different loss weighting or data mix), which is a change to the fixed design and needs the author.

**Recommendation:** (a), unless the author chooses a further simulator change.

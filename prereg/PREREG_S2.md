# Pre-registration, Step 2: do the checks predict how well policies trained inside learned simulators walk in physics?

Written by session M4 on 2026-11-08, after Gate 1 and the four-policy pilot, and **before any registered policy
exists**: no policy with seed 1 or 2 has been trained in any pool simulator. One policy trained in a pool simulator
does exist and was measured: the pilot's `REL__p0` (seed 0, in `REL`), which is validation and never a result. This file and
`scripts/s2_stats.py` are committed together, hashed in `results/prereg_hash_s2.json`, and made read-only. Problems
found later go to `step2/DECISIONS.md` and are reported as deviations; analyses added later are labelled
exploratory or descriptive.

**Terms.** *Joint commands* are the 12 joint position targets a policy sends at every step (the actions). The
*velocity command* is the walking speed and turning rate the policy is asked to follow. A *simulator* is a learned
model that predicts the robot's next state from its recent states and joint commands. *MuJoCo model* is the
ANYmal D model built by hand in this round from ANYbotics' published robot description, with a fitted PD motor
model; it is **not** the simulator that produced the recorded data, and it is **not** hardware. *E_h* is the mean
normalised error of the simulator's predictions h steps (of 0.02 s) ahead, against recorded trajectories. The
*growth rate* λ1 is the leading Lyapunov exponent (1/s): how fast small differences between two predicted
trajectories grow. The *kinematic residual* r̄J is how far the predicted joint velocities are from the change of the
predicted joint angles, scaled by the recording's own spread. λ1 and r̄J need no recorded future ("label-free").
*ρ* is the Spearman rank correlation.

## 1. The question

The paper ranks learned simulators by their long-horizon error, and asks whether label-free checks rank them as
well as error checks do. Step 2 asks whether these rankings also predict something a roboticist cares about: how
well a walking policy trained inside each simulator walks in a physics simulator. Here the physics simulator is the
hand-built MuJoCo model, so every answer is about that model, not about the data's own simulator or a robot.

## 2. Gate 2 and the pool

**Gate 2** (fixed in `STEP2_PLAN.md` before any MuJoCo number; read from `results/s2_pilot.json`, one seed each,
80 episodes):

| pilot policy | trained in | falls (95% interval) | physics return (95% interval) |
|---|---|---|---|
| REL__p0 | the released five-network model | 0.100 [0.062, 0.150] | 18.59 [17.37, 19.66] |
| A_s0__p0 | rebuilt, released settings, seed 0 | 1.000 [1.000, 1.000] | −10.37 [−12.71, −8.28] |
| B_s0__p0 | rebuilt, teacher-forced, seed 0 | 1.000 [1.000, 1.000] | −3.99 [−5.27, −3.17] |
| A_s0__u50__p0 | A_s0 with 50% of weights pruned | 1.000 [1.000, 1.000] | −6.56 [−8.71, −4.75] |

- **P1** (released-model policy falls in fewer than 25% of episodes): 10.0%. **Passed.**
- **P2** (at least one of the three rebuilt-simulator policies falls in fewer than 50%): all three fell in every
  episode. **Failed.**
- **P3** (the four returns are not all inside each other's 95% intervals): the released model's interval overlaps
  none of the others. **Passed.**

P2 failed, so the plan's rule registers **the fallback pool**, not the 25-simulator main pool: "the released model
and its eight compressed copies, two policy seeds each, all five-network, so the penalty is active alike."

**The pool: nine simulators** (ids as used by every script):

| id | what it is |
|---|---|
| `REL` | the authors' released checkpoint `robotic_world_model_lite/assets/models/pretrain_rnn_ens.pt` (sha256 `2ac8686c8c7736287853c0fe1438c379320068adf97be71609b5f1afb52c6e5a`), five networks, trained on all ten recorded episodes |
| `REL__fp16` | every parameter rounded to 16-bit floating point and back |
| `REL__int8w` | every weight matrix rounded to a symmetric per-tensor 8-bit grid (biases untouched) |
| `REL__u10`, `REL__u20`, `REL__u30`, `REL__u50` | 10, 20, 30 and 50% of weights removed by global magnitude pruning |
| `REL__s192`, `REL__s128` | GRU hidden units removed down to width 192 and 128, by an L2 importance score |

The copies are made by `scripts/compress.py` (`compress.apply`, unchanged) from the released weights, after
training and without fine-tuning. Every member has five networks, so the lite code's uncertainty penalty is active
for all nine alike.

## 3. Training settings (quoted from the plan, with this pool's specifics)

> two seeds per simulator; the authors' lite code, unchanged apart from Step 1's shims: 500 iterations, 4,096
> robots; start states only from the simulator's own training episodes.

- **Seeds:** policy seeds **1 and 2** for every simulator, 18 policies. Seed 0 is not used, so the pilot's measured
  policy `REL__p0` is never part of the results.
- **Start states:** the released model was trained on all ten recorded episodes, so its "own training episodes"
  are all ten; no episode is held out for any pool member (as for the pilot's `REL__p0`).
- **Route:** all nine simulators enter the lite trainer the same way: `common.load` of the released weights, the
  compression variant applied (none for `REL`), wrapped by `scripts/lite/adapter.py`
  (`scripts/s2/train_policy.py --route rebuilt`). The adapter equals the authors' `SystemDynamicsEnsemble` on the
  released weights to a relative 1e-6 (`scripts/lite/test_lite.py`, test 2).
- Run by `scripts/s2_main.py` (job `step2/jobs/main.zsh`); each policy is one seeded, uninterrupted run (an
  interrupted run restarts from iteration 0); the actor of the last iteration is exported.

## 4. The MuJoCo protocol (quoted from the plan; code `scripts/s2/evaluate.py`, unchanged since the pilot)

> **Command schedules:** the ten recorded episodes' velocity-command schedules, 20 s each. **Starts:** eight
> perturbed starts per schedule, seeded. Joint angles move by up to ±0.05 rad, and heading is random. **Falls:** an
> episode ends at a fall, by Isaac Lab's rule for this task.

The fall rule is Isaac Lab's `illegal_contact` on the trunk (net contact force above 1 N). The model is the frozen
calibrated `step2/model/anymal_d_cal.xml` (friction 0.6, contact time constant 0.01 s, armature 0.02 kg·m², damping
0), with the PD motor fallback (the ANYdrive network file could not be obtained).

## 5. Measures

- **Primary: physics return** — the lite code's own reward terms and weights, computed from MuJoCo states and foot
  contacts, averaged over the 80 episodes.
- **Secondary:** fall rate (share of the 80 episodes ending in a fall), speed-tracking error (m/s) and
  turn-tracking error (rad/s).
- **A simulator's score** is the mean over its two policies.

## 6. Checks (all stored before registration)

All seven simulator checks are in `results/s2/checks.json` (sha256
`23e9daaf219febe57b25b01924f3fa6eba29a312f48d31d7832591e550a351c2`, fixed in `scripts/s2_stats.py`), computed by
`scripts/s2/fallback_checks.py` with the registered functions only (`compute.build`, `compute.per_model`,
`s1_rollout.score_gru`), before any pool policy existed and without any physics number. The compressed copies had
no stored checks, so they were computed now, with the released model recomputed in the same run (one code path);
the released model's recomputed values equal the stored ones (E_h and λ1 to a relative 5e-7; r̄J to 0.13%).

- **Under recorded joint commands:** E1, E8, E100, λ1 and r̄J, each the mean over trajectories.
- **Under policy commands** (the authors' released actor in closed loop inside the simulator, as in Step 1): λ1 and
  r̄J; a simulator with any diverged trajectory gets +∞ (ranked worst).
- **Arena:** all 20 recorded trajectories (both halves of all ten episodes), the released model's own arena for its
  stored values. **These checks are in-sample:** the released model was trained on all ten episodes, so no
  held-out episode exists for this pool. Stated as a limitation wherever the result is reported.
- **Imagined return:** each policy's mean `Train/mean_reward` over its last 25 training iterations (the lite log),
  averaged over the simulator's two policies. It exists only after training; it is written by the training script
  and is not read by any person before the analysis.

## 7. Statistics (`scripts/s2_stats.py`, committed with this file)

- **Unit:** the simulator (nine), with the two policy seeds averaged.
- **ρ** is Spearman's rank correlation (`stats.spearman`, imported). Every check is made "higher is better" through
  its rank: E1, E8, E100, λ1 and r̄J (both command sources) are negated as −rank, so a non-finite value ranks worst;
  the imagined return is used as it is (a non-finite value also ranks worst). ρ(−x, y) is written for the negated
  check.
- **Intervals:** paired percentile bootstrap over simulators, 20,000 draws, numpy `default_rng(0)` (the same draws
  for every analysis), by `stats.boot_rho` and `stats.boot_diff`, imported, never copied. Draws in which a
  correlation is undefined (all drawn values tied) are dropped and counted.
- **Labels:** an interval wholly above 0 gives the first label, wholly below 0 the second, otherwise NOT SETTLED
  (`common.label3`). An outcome with no spread (for example every policy falling in every episode) makes that
  analysis NOT COMPUTABLE.
- **NOT COMPUTABLE** (reported with its reason) when: the outcome has fewer than 3 simulators or fewer than 3
  distinct values; a check has all values tied, or its ρ is undefined on the full sample; or more than 10% of the
  20,000 draws are undefined. Dropped draws are counted and printed for every analysis.
- **A policy that fails.** Every policy whose training finishes is measured, whatever its losses. A step that fails
  twice with a non-finite error (nan, inf, "not finite") in its traceback was failed by its simulator
  (`results/s2/failed/<policy>.json`): that policy is scored as falling in every episode, with the lowest physics
  return among all measured policies, and is listed. Any other failure (killed, out of memory, a code error) is not
  a result: the job stops for diagnosis and is re-run.
- **Missing data:** a simulator's score uses its measured or failed-by-simulator policies; a simulator with none is
  dropped and listed. A policy without a training record or imagined-return value is missing for the imagined
  return; a simulator with no imagined return leaves the imagined-return analyses (ρ and selection), listed; a
  non-finite imagined return is kept and ranks worst. All 18 are expected.

**Premise (95%).**
- **R0:** ρ(−E100, physics return). SUPPORTED if the interval lies above 0, CONTRADICTED if below, NOT SETTLED
  otherwise.

**Primary family (97.5% intervals, two questions).**
- **R1:** ρ(−λ1, return) − ρ(−E1, return). LABEL-FREE BETTER / ONE-STEP BETTER / NOT SETTLED.
- **R2:** ρ(−r̄J, return) − ρ(−E8, return). LABEL-FREE BETTER / TRAINING-HORIZON BETTER / NOT SETTLED.

**Secondary (95% intervals; labelled secondary wherever reported).**
- R0–R2 with the fall rate as the outcome, used as −(fall rate) so that a positive ρ still means "the check picks
  simulators whose policies walk better".
- R1 and R2 with λ1 and r̄J computed under the policy's commands (λ1_pol, r̄J_pol), against the same E1 and E8.
- ρ(imagined return, physics return): POSITIVE / NEGATIVE / NOT SETTLED.
- **Selection:** for each check (E1, E8, E100, λ1, r̄J, λ1_pol, r̄J_pol, imagined return), the three simulators it
  ranks best; the *shortfall* is the mean physics return of the best three simulators by physics return minus the
  mean of the check's three. Each shortfall has a 95% interval from the same bootstrap draws. In a draw, the entries
  are the drawn simulators, so a simulator drawn twice is two entries; ties go to the lower pool position (the
  table order of Section 2). Selection is NOT COMPUTABLE with fewer than 6 simulators. SEL1 = shortfall(E1) − shortfall(λ1) and SEL2 = shortfall(E8) −
  shortfall(r̄J), labelled LABEL-FREE BETTER / ONE-STEP or TRAINING-HORIZON BETTER / NOT SETTLED.
- **The rollout-trained subset alone:** in this pool only `REL` is itself rollout-trained (the other eight are
  compressed copies, not trained), so this analysis is **NOT APPLICABLE** and is reported as such.

**Exploratory.** Two-level intervals for R0–R2: simulators resampled, then each drawn simulator's policy seeds
resampled within it (20,000 draws, `default_rng(0)`).

**Descriptive.** Per simulator: physics return, falls, speed and turn errors, imagined return and every check.

## 8. Expected outcome

The plan expected about 25 simulators; this pool has nine. With nine simulators, the bootstrap intervals of the
differences R1 and R2 will be wide, and **they may well end NOT SETTLED**; that would be reported as it is. The
pool's spread comes from compression only (one training run), and one copy (`REL__fp16`) is almost the released
model itself: its E1, E8 and E100 differ from `REL`'s by less than 0.1%, so the order of these two on those checks
is arbitrary. It is a narrow test. The pilot's rebuilt single-network simulators produced policies that fell
in every episode; if the compressed copies do the same, the fall-rate analyses may be NOT COMPUTABLE and the returns
will differ mainly in how quickly the robot falls.

## 9. What was known at registration

- **Gate 1** (`results/s2_fidelity.json`, the authors' policy in the calibrated MuJoCo model, 80 episodes): falls
  1.25% (G1 < 10%: passed); share of time with two feet down 0.789 against 0.844 recorded (G2, gap ≤ 10 points:
  passed); speed-tracking error 0.069 against 0.072 m/s recorded (G3, ratio ≤ 1.5: passed). Turn error 0.061 against
  0.138 rad/s recorded (not gated). Replay of recorded joint commands on held-out episodes: E1 0.085, E8 0.150,
  E100 1.039 for the calibrated model (descriptive).
- **The pilot** (Section 2), including that the released-model policy walks (10% falls) and that the three
  single-network rebuilt simulators' policies fall in every episode. The pilot's imagined returns were known too,
  and ran opposite to the physics returns: `REL__p0` 2.07 (best in physics), `A_s0__p0`
  5.13, `B_s0__p0` 3.95, `A_s0__u50__p0` 3.79 (different
  simulators from this pool, one seed each).
- **The nine simulators' checks** (`results/s2/checks.json`): for example E100 ranges from 0.40 (`REL`,
  `REL__fp16`) to 7.56 (`REL__u50`), and `REL__u30` diverges on one trajectory under policy commands. No physics
  number exists for any of them.
- **All earlier results of the project:** the paper's registered studies (`prereg/PREREG.md`, `PREREG_FT.md`,
  `PREREG_S1.md`) and their results, including: at ranking simulators by held-out long-horizon error, λ1 did
  better than the one-step error E1 (Q1, LABEL-FREE BETTER) and was not settled against the 8-step error E8 (Q2);
  under the policy's commands, r̄J ranked worse than under recorded commands (CL1), and λ1's change was not
  settled (CL2).

## 10. Choices made at registration (also in `step2/DECISIONS.md`, M4)

1. The fallback pool's checks were not stored; they were computed before registration by registered functions
   (Section 6). The released model's recomputed values are used for it too, so all nine share one computation.
2. Policy seeds 1 and 2 (not 0), so the pilot's measured policy is not reused.
3. One training route for all nine (the adapter), since the copies cannot enter through the authors' loader.
4. The fall-rate outcome is negated; selection, the NOT COMPUTABLE rules, failed policies and missing data are
   defined in Section 7 (tightened after an independent statistical review, before registration).
5. The rollout-trained subset is not applicable in this pool.

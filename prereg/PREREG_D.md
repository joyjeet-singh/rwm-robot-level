# Pre-registration, Option D: do the paper's checks rank simulators adapted from the released model by how well policies trained inside them walk?

Registered by session D1 before any policy was trained in a ladder simulator (only 2- and 3-iteration `_test*` runs,
deleted, existed). It fixes the design of `OPTIOND_PLAN.md` ("Design fixed by this plan", committed by D0 before any
Option D number existed on this Mac) and writes out the analysis. The analysis code `scripts/d_stats.py` is committed
with this file; both are made read-only after the commit. Problems found later go to `optd/DECISIONS.md`; analyses
added after the results are labelled descriptive or exploratory.

## 1. The question

RWM's released world model (`REL`) is adapted to the MuJoCo ANYmal D by fine-tuning it on MuJoCo data, by different
amounts and in different ways. Policies are trained inside each adapted simulator with the authors' own code and
measured in MuJoCo. **Do the paper's checks (E1, E8, E100, λ1 and r̄J) rank these simulators in the order of how well
their policies walk?** The design is "early-failure": a run is searched for the amount of adaptation at which its
simulators stop producing walking policies, so the tested simulators cluster around that boundary.

## 2. The world

The calibrated MuJoCo ANYmal D of Step 2: a hand-built MuJoCo model of ANYmal D with a fitted PD motor. **It is not
Isaac Lab and not hardware**; every answer here is about this model. It made the fine-tuning data (Option C's rung
`rung_1M`) and the test sets (Option C's `test_expert.npy` and `test_mixed.npy`), and it measures the policies.

## 3. The runs

A run is one fine-tuning of the released model. Every run starts from all of its weights (all five networks), keeps
the authors' fixed scales (`anymal_d_flat_cfg.py`, `state_data_mean` and `state_data_std`), and trains everything.

| Order | Run | Data | Learning rate | Windows | Seed | Starts only if |
|---|---|---|---|---|---|---|
| 1 | `W_s0` | walking | 1e-4 | rollout | 0 | always |
| 2 | `M_s0` | mixed | 1e-4 | rollout | 0 | always |
| 3 | `Wtf_s0` | walking | 1e-4 | teacher-forced | 0 | always |
| 4 | `W_s1` | walking | 1e-4 | rollout | 1 | `W_s0` found a boundary |
| 5 | `M_s1` | mixed | 1e-4 | rollout | 1 | `M_s0` found a boundary |
| 6 | `Wg_s0` | walking | 1e-5 | rollout | 0 | always |
| 7 | `Mg_s0` | mixed | 1e-5 | rollout | 0 | always |

- **Mixed data:** every row of `rung_1M` (999,973 transitions; 45% from weaker policies, heavy action noise and random
  motion, with many falls). **Walking data:** its rows from sources 0, 1 and 2 (RWM's released policy with action noise
  0, 0.1 and 0.3; 549,773 rows).
- **Rollout windows:** 32 + 8 rows; the model predicts 8 steps from its own outputs. **Teacher-forced:** 32 + 1 rows.
  Windows lie inside one episode, with a fall only at the last row (the lite loader's rule).
- **Fine-tuning:** the rebuild's trainer, which equals the authors' loss; Adam with weight decay 1e-5 and a fresh
  optimiser state; batch 256 windows; no gradient clipping. A **step** is one update on one batch. The seed sets the
  batch order. One fine-tuning at a time, on 2 threads (`scripts/d/finetune.py`).
- **Saves:** at 5, 20, 40, 80, 160, 320 and 1,280 steps; each save is a simulator `<run>_d<steps>`. A run is fine-tuned
  only as far as its next test needs.
- **A loss that is not finite ends the run** (also a save whose weights are not finite). Its tested simulators stay in
  the pool.
- **A run found a boundary** when at least one of its tested simulators failed and at least one did not.

## 4. The search, the verdict rule and the stop rules

**A test** of a simulator: compute its checks (Section 9) before its first policy; train and measure its policies; give
its verdict. Three tests per run:

1. test the simulator at 80 steps;
2. if it failed, test 20 steps; otherwise test 320 steps;
3. the third test:

| 80 steps | Second test | Third test |
|---|---|---|
| failed | 20 failed | 5 |
| failed | 20 did not fail | 40 |
| did not fail | 320 failed | 160 |
| did not fail | 320 did not fail | 1,280 |

**Policies per tested simulator (the verdict rule, `ladder.verdict_rule`).** The run's worker trains seeds 1 and 2, one
after the other, measuring each as soon as it is trained. If exactly one of the two failed, it trains seed 3. The
simulator **failed** when at least two of its policies failed; that is its **verdict**, and the search waits for it.
A **policy failed** when its walking score is below 0.15 (a score of exactly 0.15 does not fail), or it was stopped by
the stop rule, or its training failed by its simulator. Every policy counts: a failed policy is measured and kept, and
enters its simulator's mean score.

**The stop rule (collapse in imagination).** From iteration 50 on, training stops once the mean imagined episode length
(`Train/mean_episode_length`) has stayed under 128 steps (half the limit of 256) for 50 iterations in a row. Iterations
1–50 do not count, so the earliest stop is at iteration 100. An iteration with no logged length (no imagined episode
ended in it; about 2% of iterations in Step 2's logs) breaks the run of short iterations. A stopped policy is measured
from its weights at the stop and counts as failed. **No other early stop:** a policy that still trains runs its 500
iterations.

**Only walking scores steer the search.** No check decides which simulator is tested, whether a run stops, or whether
a third policy is trained.

## 5. Limits and the sanity gate

- All seven runs may be done. **No run starts after 56 hours of job time** (the driver's running time, summed over
  resumes). A run that has started finishes its three tests. A run that does not start is recorded with its reason.
- **Two workers**, each taking the next run in the order above whose start condition can be decided, and doing its
  tests and each test's policies one after another. Policy training and fine-tuning use 2 threads each.
- **Sanity gate:** if neither `W_s0` nor `M_s0` can produce a simulator at 80 steps with finite weights, the job stops
  BLOCKED. Work the other worker did before the gate decided is kept and reported as descriptive only.

## 6. Pools

- **Primary pool:** `REL`, with its two registered policies (Step 2's seeds 1 and 2), and every ladder simulator that
  was tested (at most 21). A test counts once it has reached its verdict; a test cut short (the job stopped inside it)
  is left out and listed.
- **Pooled:** the primary pool and Step 2's eight compressed copies of `REL`, with Step 2's two policies each.
- **Descriptive only:** Option C's four simulators, `A5_s0`, and Step 2's three single-network rebuilds.
- The unit is the simulator. Its **outcome** is its mean walking score over its two or three policies.

## 7. Policies

- **Step 2's recipe** (`scripts/d/train_policy.py`): the authors' lite code (`scripts/lite/run_lite.py`, unchanged), as
  `scripts/s2/train_policy.py --route rebuilt` runs it for a simulator with no hold-out: 500 iterations; 4,096 robots;
  start states from the authors' recorded data with no episode held out; the lite config's uncertainty weight; the
  torch seed; 2 threads. No noise floor and no fall cost. The only change is the stop rule's log hook. The simulator
  comes from `scripts/d/sims.load` and is built before the lite code sets its seed, as Step 2's wrapper builds it. On
  `REL` (seed 1, 256 robots, 3 iterations, one thread) the two wrappers log identical losses and rewards.
- **Seeds:** 1 and 2; 3 for a third policy.
- **A training run that fails by its simulator:** a loss or step reward that is not finite, or the same crash inside the
  lite code twice (same exception type). Such a policy gets a walking score of 0 and a fall rate of 1 (Step 2's rule),
  is not measured, and counts as failed. For the physics-return outcome it gets the lowest physics return among the
  measured policies of the pool analysed, the primary pool (Step 2's rule).
- **The crash guard:** before a crash is counted as the simulator's, the same code runs on `REL` for 3 iterations
  (`_test_guard_<policy>`). If that crashes too, the fault is the code's: the job stops with an error and nothing is
  recorded as failed. Two different crashes are not a result either; the job stops.
- **Interrupted training** restarts from iteration 0, so every policy is one uninterrupted, seeded run.

## 8. The walking score and other measures

- **Walking score** (Option C's measure, `scripts/c/evaluate.py` through `scripts/d/evaluate.py`): each step scores
  exp(−(|v_xy − c_xy|² + (ω_z − c_ω)²)/0.25), with v and ω the base-frame speed and turn rate after the step and c the
  velocity command; after a fall every remaining step scores 0. An episode scores the mean over its 1,000 steps (20 s);
  a policy the mean over 96 episodes: 24 fixed velocity-command schedules (uniform in −1 to 1 on each axis, a new
  command every 10 s) × 4 seeded starts, calibrated physics, no pushes, no observation noise. RWM's released policy
  scores 0.942; standing still scores 0.069.
- **Secondary measures:** the fall rate (share of the 96 episodes that end in a fall), and Step 2's physics return
  (Step 2's protocol: the ten recorded schedules × 8 starts, Step 2's reward).

## 9. The checks and the walk test

Computed for a ladder simulator before its first policy is trained (`scripts/d/checks.py`), on the expert test set
(primary: RWM's released policy without noise, 40 windows) and the mixed test set (secondary, 67 windows):
- the paper's 400-row windows (32 history rows, 368 forecast steps) and its rollout from the mean of the five networks;
- **common units:** every prediction converted back to physical units and divided by the authors' scales. For the
  ladder, `REL` and its copies this is the paper's own error;
- **errors** E1, E8, E24, E100 and E368, with the hold-the-last-state floor in the same units;
- **kinematic residual r̄J** (and the attitude residual r̄A, exploratory), scaled by the residuals of the expert test
  set's own true rows;
- **growth rate λ1:** the paper's function and settings (p = 4, float64, seed 0), in the simulator's own coordinates.

**The walk test** (secondary; `scripts/d/walk_test.py`): RWM's released policy run inside the simulator's imagination
(noise-free actions, 1,024 robots, 256 steps, seed 20278, start states from the authors' recorded data). Three numbers:
the share of imagined episodes that end in a fall; the tracking reward per step; ū, the mean disagreement between the
five networks.

## 10. Statistics (`scripts/d_stats.py`, committed with this file)

- **ρ** is Spearman's rank correlation across simulators (`stats.spearman`). Every check is made "higher is better"
  through its rank: errors, λ1, r̄J, r̄A, walk-test falls and ū are negated as −rank, so a non-finite value ranks worst;
  walk-test tracking is used as it is. ρ(−x, y) is written for the negated check.
- **Intervals:** paired percentile bootstrap over simulators, 20,000 draws, numpy `default_rng(0)` (the same draws for
  every analysis), by `stats.boot_rho` and `stats.boot_diff` through Step 2's registered helpers
  (`scripts/s2_stats.py`: `rho_q`, `diff_q`, `better`, `computable`, `top_mean`), imported, never copied. Undefined
  draws are dropped and counted.
- **Labels (Step 2's decision rule):** an interval wholly above 0 gives the first label, wholly below 0 the second,
  otherwise NOT SETTLED (`common.label3`).
- **NOT COMPUTABLE**, with its reason printed: a pool of fewer than 4 simulators; an outcome with fewer than 3
  simulators or 3 distinct values; a check with all values tied, or a ρ undefined on the full sample; more than 10% of
  the draws undefined (Step 2's rules).
- **Missing data and non-finite values:** a check that was computed but is not finite (NaN or infinite) stays and ranks
  worst. A check that is missing (not computed, or stored as null) makes the simulator leave the analyses that use that
  check, and it is listed; the same holds for walk-test values. An outcome that is missing (no seed-1 policy, or a
  measured policy without a physics return) makes the simulator leave the analyses of that outcome, listed. A ladder
  test that did not reach a verdict is left out and listed. A missing policy result is listed.

**Premise (95%). R0:** ρ(−E100, score), expert test set, primary pool. SUPPORTED / CONTRADICTED / NOT SETTLED.

**Primary family (97.5% intervals each), expert test set, primary pool.**
- **R1:** ρ(−λ1, score) − ρ(−E1, score). LABEL-FREE BETTER / ONE-STEP BETTER / NOT SETTLED.
- **R2:** ρ(−r̄J, score) − ρ(−E8, score). LABEL-FREE BETTER / TRAINING-HORIZON BETTER / NOT SETTLED.

**Secondary (95% intervals; labelled secondary wherever reported).**
- R0–R2 on the pooled pool (expert test set); and R0–R2 on the mixed test set (primary pool).
- ρ(−E368, score) − ρ(−E8, score): LONG-HORIZON BETTER / TRAINING-HORIZON BETTER.
- The walk test: ρ(tracking, score), ρ(−falls, score) and ρ(−ū, score) (POSITIVE / NEGATIVE), each also minus
  ρ(−E8, score) (WALK TEST BETTER / TRAINING-HORIZON BETTER).
- R1 and R2 on `REL` with the seed-0 runs alone (`W_s0`, `M_s0`, `Wtf_s0`, `Wg_s0`, `Mg_s0`).
- R0–R2 with each simulator's seed-1 policy alone as the outcome, to show how much the extra policies change the answer.
- "Failed" as the outcome: for each check (E1, E8, E24, E100, E368, λ1, r̄J), the area under the ROC curve for
  separating failed simulators from the others (the chance that a simulator that did not fail has the better check
  value; ties count one half), with a 95% interval from the same draws; ABOVE CHANCE / BELOW CHANCE / NOT SETTLED
  against 0.5. NOT COMPUTABLE with fewer than 2 simulators of either kind, or more than 10% of draws holding one kind.
- The fall rate as the outcome (used as −fall rate), and Step 2's physics return as the outcome: R0–R2.
- The best three simulators chosen by each check (E1, E8, E100, E368, λ1, r̄J), and each check's shortfall: the mean
  score of the best three by score minus the mean score of the check's three, with a 95% interval (Step 2's selection
  method; NOT COMPUTABLE with fewer than 6 simulators).

**Exploratory.** ρ(−r̄A, score) − ρ(−E8, score).

**Descriptive.** The ladder table (per run: each tested step count with its checks, its verdict, and each policy's
score and fall rate); how often a test needed a third policy; the descriptive simulators with their checks and Stage 1
results; the stop rule's stops.

## 11. What was known before registering

- **Step 2's results:** its registered analysis of `REL` and its eight copies (fall rate and physics return), and its
  22 policies.
- **Option C's records:** its from-scratch simulators, its two pilots and Gate A2. Its simulators failed the walk test;
  two design mistakes lay behind it (the data's own scales instead of the authors', and too little training).
- **Everything in `optd/REVIEW.md`** (measured on a cloud copy, not on this Mac), including: walking-data and mixed-data
  fine-tunes of the released model (rung 100k, up to 160 and 320 steps); policies trained in a cloud copy of the
  released model and in a walking-data fine-tune (`FT160`) with three seeds, where `FT160`'s three policies all failed
  (0.06–0.12) while the released model's walked (0.64); the checks of those simulators, where `FT160`'s short-horizon
  errors were better than the released model's and its long-horizon checks a little worse; and a simulated ladder used
  to choose two policies per simulator, a third after a split.
- **Everything in `results/d_record.json`** (Stage 1, measured on this Mac): the walking scores, fall rates and physics
  returns of Step 2's 22 policies; the checks and walk tests of 17 simulators. In particular:
  - `REL`'s policies score 0.610 and 0.573 (the pilot `REL__p0`, released route, 0.769);
  - by this design's rule, 6 of Step 2's nine simulators failed and 3 did not, and the two seeds **never** disagreed
    (0 of 9): Step 2's simulators lie far from the boundary;
  - the stop rule, applied to stored training logs, would have stopped 5 of Step 2's 18 registered policies:
    `REL__s192__p1`, `REL__s192__p2`, `REL__s128__p1`, `REL__s128__p2` at iteration 100 and `REL__u30__p1` at 478. All
    five failed anyway (scores 0.004–0.069). It would have stopped none of Step 2's four pilots, and, among Option C's
    pilot policies (their own recipe, up to 2,000 iterations), `sim_10k_N8_s0__p0`, `A5_s0__p0` and `A5_s0__p1` at
    100 and `sim_6M_N8_s0__p0` and `__w4__p0` at 798 and 776 (`results/d/stage1.json`).

## 12. Limits to state

- **The pool is not a fixed sample.** It is built step by step from the policies' scores, by a search that puts its tests
  near the boundary on purpose. The bootstrap treats the tested simulators as a sample, so its intervals are optimistic.
- **A simulator's score is the mean of only two or three policies**, and a third is trained only after a split, so how
  many policies a simulator has depends on their scores. Step 2's two seeds disagreed for none of its nine simulators,
  but those lay far from the boundary; how often ladder simulators split is unknown.
- **Near the boundary even a majority of three can misjudge a simulator,** and each verdict steers the next test.
- **The simulators are not independent.** Simulators saved from one run share its whole history, and every one descends
  from `REL`.
- **One thing may drive everything:** how far a simulator was moved from `REL` may move every check and the score
  together, so a correlation may say no more than "less adaptation, better policies".
- **500 iterations** is the authors' default and leaves policies immature (in the review, the released model's own
  policy scored 0.08 at 200 iterations and 0.64 at 500).
- **The design was chosen after the review saw one fine-tuned simulator fail** (`FT160`), and the number of policies per
  simulator after it measured Step 2's pairs and three seeds in that simulator.
- **The stop rule applies to the ladder only.** Step 2's policies (in `REL`'s two and in the pooled pool) enter as Step 2
  trained them, for 500 iterations without the stop rule; the five the rule would have stopped (Section 11) are all
  failed policies either way, but they are measured at 500 iterations, not at the stop.
- **The world is a hand-built MuJoCo model with a fitted PD motor**, not Isaac Lab and not hardware.

## 13. Choices made at registration (also in `optd/DECISIONS.md`, D1)

1. "The same crash" means the same exception type; two different crashes stop the job (not a result).
2. Job time for the 56-hour limit is the driver's running time, summed over resumes.
3. A worker skips a run whose start condition cannot yet be decided and takes the next; the skipped run is decided
   later.
4. A stopped policy whose losses are not finite counts as failed by its simulator (score 0, not measured).
5. A ladder simulator whose checks or walk test cannot be computed during the job is retried at the job's end; checks
   never steer the search.
6. `REL` and Step 2's copies take their verdict (for the "failed" outcome) from their two registered policies; a split
   between those two (none happened) would mean "did not fail", since a simulator fails only when two policies fail.
   For a ladder simulator the analysis recomputes the verdict from its policies and stops if it differs from the
   driver's.
7. The labels of the secondary comparisons and the AUC rule above were added by this registration; the design lists the
   quantities.
8. "Every ladder simulator that was tested" is read as every test that reached its verdict (a test cut short by a
   stopped job has no outcome).

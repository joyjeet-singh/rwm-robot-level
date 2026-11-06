# Pre-registration of studies CL and AR (Step 1)

Written in session S3 of AIR 2027 Step 1, 2026-11-06. At the time of writing no trained simulator has been driven by
the policy's joint commands or by noisy joint commands, no trained MLP, RSSM or Transformer simulator has had an
error, a growth rate or a kinematic residual computed by this folder's code, and no new baseline has been trained
beyond 50 iterations (step1/LOG.md, sessions S0–S2). Once committed, this file and `scripts/s1_stats.py` do not
change. A needed change goes to `ANOMALIES.md` in a new commit, and the paper says so. Study TB (training length) and
every analysis not named here are descriptive or exploratory.

**Terms.** *Joint commands* are the 12 joint position targets given at every step (the actions). The *velocity
command* is the walking speed and turning rate (vx, vy, wz) that a policy is asked to follow. A *simulator* is a
learned model that predicts the robot's next state from its recent states and joint commands. The *label-free checks*
need no recorded future: the *growth rate* λ1 (the leading Lyapunov exponent, in 1/s: how fast small differences
between two predicted trajectories grow) and the *kinematic residual* r̄J (how far the predicted joint velocities are
from the change of the predicted joint angles, scaled by the recording's own spread). *E_h* is the mean error at
horizon h steps of 0.02 s, normalised as in the paper. *ρ* is the Spearman rank correlation.

Scope: one robot (ANYmal D), one trotting gait, simulated data. No policy is trained, and no policy is tested in a
physics simulator.

## 1. Questions

- **CL.** When a policy, not the recording, supplies the joint commands, do the label-free checks still rank
  simulators by their 100-step error under recorded commands?
- **AR.** Do the registered comparisons of checks (Q1, and Q4's second question) hold for simulators of other
  architectures: MLP, RSSM and Transformer?

## 2. Simulators

- **CL, primary:** F18's 97 simulators (pools R, T and C of `PREREG.md`): 57 trained models and 40 compressed copies,
  all on held-out episodes 1 and 8. Keys, groups and weights are those of `results/model_table.json` and
  `results/models.json` (key `F18`).
- **CL, secondary:** X5's 255 simulators (five splits of 51), each on its own held-out pair (`model_table.json`,
  key `X5`).
- **Reference only:** the released model (`REL`), on all 20 trajectories. It enters no statistic.
- **AR:** every baseline run S0 listed, **26** (`../rwm_repro/runs/baseline_<arch>_<regime>_<spec>_seed<s>/`: MLP 9,
  RSSM 8, Transformer 9), plus **12 new runs** that the job trains after this registration: seeds 3 and 4 of `tf_s7`
  (trained one step ahead, "teacher-forced") and `ar_s7` (trained on its own rollouts), for each of the MLP, RSSM and
  Transformer. Total 38: MLP 13, RSSM 12, Transformer 13.
  - *Why the new runs.* With them, each architecture has five seeds of both training methods, as the paper's main
    models (`armA`, seeds 0–4) have. With three seeds per method only, a ranking within one architecture would rest
    on 8 or 9 models.
  - The `matched`, `x1v1` and `x1v2` runs are kept as they are, and not extended.
  - The new runs are trained by `scripts/s1_train_baseline.py`, a copy of the reproduction's
    `scripts/train_baseline.py` whose changed lines are marked `S1:` (outputs, progress, thread count). Its training
    is unchanged: F18's split (episodes 1 and 8 held out), 2,500 iterations, batch 256, 2 threads. Test (ix) showed
    that at 2 threads it reproduces the stored loss curves bit for bit for 50 iterations.
  - Every run is scored with its final weights (`weights_2500.pt`) on F18's arena (4 trajectories of 400 rows from
    episodes 1 and 8).

## 3. Policy and commands

- **Actor:** the authors' actor in `../robotic_world_model_lite/assets/models/pretrain_rnn_ens.pt`,
  `model_state_dict` keys `actor.{0,2,4,6}.weight` and `.bias` (48 → 128 → 128 → 128 → 12, ELU). Deterministic: its
  mean output, no action noise (DECISIONS S1-2).
- **Velocity command:** per recorded segment, the mean (vx, vy, wz) of the Step 0 velocity regime that covers the row
  (`../rwm_repro/results/step0_regimes.json`, `regimes`; built as task1c's `command_per_row`). It stays fixed within a
  segment: no resampling, no pushes, no resets.
- **Observation:** as `../robotic_world_model_lite/scripts/envs/anymal_d_flat.py` lines 39–56
  (`get_imagination_observation`) builds it, from the simulator's current state (denormalised with the model's own
  normalisation), the velocity command and the previous joint command; no observation noise (DECISIONS S1-2). Test
  (i) checks it against the authors' own method.
- **Timing (S1):** fed state row r and joint-command row r, the actor's output stands for joint-command row r + 1.
  In `lyap.rollout_map` (action offset 1) that is the command that moves state row r to row r + 1, so the
  registered pairing is kept. Forecast step 1 keeps the recorded command, as `lyap.first_step` does. Task1c's R²
  (0.486 at offset −1, 0.391 at 0) is reproduced by test (i).

## 4. Measures

As defined in S1 and S2, quoted from the docstrings of `scripts/s1_rollout.py` and `scripts/s1_arch.py`:

> Every rollout starts where the registered 368-step rollout starts: after each trajectory's 32 true history rows,
> on the model's own arena, with its own normalisation.
> - rec: the recorded commands (the registered setting; recomputes the stored lambda_1 and r_J)
> - pol: the authors' actor in closed loop (s1_policy.Policy): lambda1_pol, rJ_pol, imagined return
> - cl: the same closed loop, with the policy differentiated too: lambda1_cl (Benettin, forward-mode JVPs)
> - noise: recorded commands plus Gaussian noise: lambda1_noise, rJ_noise
>
> lambda1_pol is lyap.lyapunov with the command array after the history replaced by the closed-loop commands, so the
> policy's own feedback is not differentiated. lambda1_cl differentiates the composed map (policy, then simulator
> step) with the policy's previous joint command in the state; its start directions are lyap.lyapunov's (same seed,
> same draw, zero in the 12 added coordinates).
>
> Divergence (new conditions only): the first forecast step at which any normalised state component is non-finite
> or exceeds 1e3 in absolute value. The checks of a diverged rollout are +inf; the step is recorded (0 = none).
> The imagined return of a diverged rollout is its sum up to that step, flagged.

For the other architectures (`s1_arch.score_arch`, recorded joint commands only):

> E1, E8, E24, E100: evalh.window_error's windows (every position for E1, every 4th after 32 true history rows
> otherwise), each window rolled out from its own 32 true rows, normalised MAE, float32.
> E368: the mean normalised MAE of the single 368-step rollout (compute.per_model's err).
> lam1_rec: Benettin as lyap.lyapunov: float64, forward-mode JVPs, K = 367 products over forecast steps 1 -> 368,
> start direction from numpy default_rng(0) (an (n, 4) draw, QR, first column), 1/s.
> rJ_rec: mean over the 368 steps of common.monitors' scaled joint residual r_J.

Further rules, fixed now:

- **One exponent** (S1-1): every λ1 uses `p = 1`; λ1 is the same for every p (test iii).
- **Noise** (S1-3): 0.1 × the per-joint standard deviation of the recorded joint commands over all 10,000 rows
  (ddof 0), added to rows START+1…399; trajectory b of an arena uses `default_rng(b)`.
- **λ1_cl** (S1-4): start directions are lyap's draw padded with zero in the 12 previous-command coordinates; this
  leaves a small finite-time offset (test ii).
- **Imagined return** (S1-5): the 12 weighted terms of the authors' `_compute_imagination_reward_terms`, times
  step_dt, summed over closed-loop steps 2–368. The uncertainty penalty is reported apart and not added;
  terminations are counted, not acted on. The authors' `dof_acc_l2` reads joint positions where joint velocities are
  meant (`anymal_d_flat.py:105`); it is reproduced as written.
- **Divergence** (S1-6): λ1_pol, λ1_noise and r̄J use the float32 rollout's divergence step; λ1_cl uses its own
  float64 trajectory; a non-finite λ is +inf.
- **RSSM step map** (S2-1): the state is (GRU latent, last predicted state); the categorical latent is the prior's
  mode (argmax), never sampled, so its derivative is zero. The straight-through term is left out of the derivative.
- **Window state** (S2-2): the MLP and Transformer carry 32 states and the 32 joint commands of the same rows.
- **Transformer derivative** (S2-3): inside λ1 only, training mode with every dropout 0 and the plain attention
  kernel; the error measures use the reproduction's evaluation path.
- **Baselines that diverge under recorded commands** (S2-4): their values are kept as computed; non-finite values
  are +inf and rank worst; the count of such runs is reported per architecture. No run is dropped for diverging.
- **A model's value** is the mean over its trajectories (4 per arena, 20 for the released model). One non-finite
  trajectory makes the model's value +inf.
- **Recorded-command values for F18 and X5** are the stored registered values: `results/model_table.json`, fields
  `lam1`, `rJ`, `E1`, `E8` and `E100` (from `results/raw_F18.npz` and `raw_X5.npz`: `lam_<key>[:, 0]`,
  `rJ_<key>`, `E1_<key>`, `E8_<key>`, `E100_<key>`). The job recomputes λ1_rec and r̄J_rec only for check S1-C1.
- **Target.** For CL, the stored E100. For AR, the E100 computed by `score_arch`.

## 5. Statistics

- Spearman correlation with average ranks; non-finite values rank worst (`stats.spearman`, `stats._clean`).
- On F18, intervals from `stats.boot_diff`: a paired model bootstrap, 20,000 draws, `default_rng(0)`; both
  correlations come from the same draw of models. Single correlations use `stats.boot_rho`.
- On X5, `stats.boot_diff_within` over the five splits: the correlation within each split, averaged over splits, with
  models resampled within their own split.
- For AR, `stats.boot_diff_within` over the three architectures: correlations within each architecture, averaged,
  with models resampled within their own architecture. `boot_diff_within` reports a 95% interval; the 97.5% interval
  is taken from the same 20,000 draws.
- These functions are imported from `scripts/stats.py`, never copied. Resampling is of whole models; never of seed ×
  trajectory, except in the exploratory two-level intervals.

## 6. Questions and rules

### Primary family CL (97.5% intervals, 1.25th to 98.75th percentile; on F18)

- **CL1:** d_r = ρ(r̄J_pol, E100) − ρ(r̄J_rec, E100).
- **CL2:** d_λ = ρ(λ1_pol, E100) − ρ(λ1_rec, E100).
- **Labels**, applied in this order:
  1. If the interval lies inside [−0.10, +0.10] (ends included): **UNCHANGED BY THE COMMAND SOURCE**. Add
     "small decrease" or "small increase" if it also excludes 0.
  2. Otherwise, if its upper end is below 0: **WORSE WITH POLICY COMMANDS**.
  3. Otherwise, if its lower end is above 0: **BETTER WITH POLICY COMMANDS**.
  4. Otherwise: **NOT SETTLED**.
- **The margin, 0.10, is fixed by the plan.** It is smaller than the smallest registered difference that earned a
  label (`results/stats_report.txt`):

  > `Q3_vs_E1: ONE-STEP BETTER  delta/mean=-0.1170  ci95=[-0.21635841508055823, -0.0255001157782541]`
  > `Q4_vs_E8: LABEL-FREE BETTER  delta/mean=+0.1150  ci95=[0.022439985843859287, 0.23318466621089803]`

### Primary family AR (97.5% intervals; within architectures and averaged, as in item 5)

- **AR1:** ρ(λ1_rec, E100) − ρ(E1, E100), which is Q1's question. Q1's labels: **LABEL-FREE BETTER** if the
  interval lies above 0, **ONE-STEP BETTER** if below, **NOT SETTLED** otherwise.
- **AR2:** ρ(r̄J_rec, E100) − ρ(E8, E100), which is Q4's second question. Q2's labels: **LABEL-FREE BETTER**,
  **TRAINING-HORIZON BETTER**, **NOT SETTLED**.

### Secondary (95% intervals, not corrected for multiplicity, reported as secondary)

- **CL3:** rank agreement ρ(check_pol, check_rec), for λ1 and for r̄J, on F18.
- **CL4:** CL1 and CL2 on X5 (within splits), with CL1's labels.
- **CL5:** ρ(λ1_cl, E100) − ρ(λ1_rec, E100), on F18, with CL1's labels.
- **CL6:** ρ(imagined return, E100), on F18. A positive value means that less accurate simulators flatter the
  policy. Models with any diverged policy rollout are left out, because their return is a partial sum; their number
  is reported.
- **CL7:** divergence counts (models and trajectories, under policy, closed-loop and noisy commands) by group, on
  F18 and X5: compressed (any compression variant); trained one step ahead (F18 groups "B (teacher forced)" and
  "B 10k iterations", X5 group "B"); rollout-trained (every other trained model, including F18's "sweep M32 N1",
  whose training rollout is one step long).
- **CL8:** CL1 and CL2 with noisy recorded joint commands in place of the policy's, on F18; CL1's labels with
  "NOISY" in place of "POLICY".
- **CL9:** CL1 and CL2 within F18's trained (57) and compressed (40) simulators separately, with CL1's labels.
- **AR3:** AR1 and AR2 pooled across the three architectures (one ranking of all baseline runs), and on the extended
  pool (F18's 97 stored values plus the baseline runs), with Q1's and Q2's labels.
- **AR4:** for each architecture, over seeds 0–4 of `tf_s7` and `ar_s7`, paired by seed: is the teacher-forced run
  better one step ahead (strictly lower E1) and worse at 100 steps (strictly higher E100)? Descriptive: counts out
  of five.

### Exploratory (labelled so)

- Two-level intervals for CL1 and CL2, as in `scripts/revision_analyses.py` (A2a): per draw, the 4 trajectories are
  resampled, then the clusters (a parent with its compressed copies; F18 has 57); 20,000 draws, `default_rng(0)`.
- The released model's values, for reference.
- Anything else computed from the job's output.

## 7. Checks

The job (`scripts/s1_eval.py`) runs them and records them in `results/s1_models.json`; `s1_stats.py` reads them and
refuses to run if one is missing.

- **S1-C1:** the recomputed recorded-command λ1 and r̄J agree with the stored ones, for all F18 models and every
  fifth X5 model in table order (indices 0, 5, …, 250). For each pool and each check separately, at least 95% of
  these models agree: |Δλ1| ≤ 1e-3 per second, and |Δr̄J| ≤ 1e-3 relative to the stored value; both values
  non-finite counts as agreement. The rank
  correlation between recomputed and stored values is at least 0.999 for each pool and check. Models outside the
  tolerance are listed.
- **S1-C2:** test (i) passes at the job's start.
- **S1-C3:** test (vi) passes at the job's start.
- **S1-C4:** every model in the pools was scored. A missing model, or a new run whose training loss became
  non-finite at any iteration or whose final weights are non-finite, is excluded and listed.
- **S1-C5:** test (ix) passes at the job's start. If it fails, no new run is trained; AR uses the existing runs only,
  and says so.
- **If S1-C1, S1-C2 or S1-C3 fails,** the new code does not reproduce the registered measures. `s1_stats.py` then
  gives no verdict and no label (primary or secondary) for the affected family, and says why; S6 stops BLOCKED. A failed S1-C1 affects
  both families (AR's extended pool and its code path share the measures); a failed S1-C2 affects only CL (the
  policy plays no part in AR); a failed S1-C3 affects only AR.

## 8. Time

No time limit: everything registered runs. The launcher stops the job only if it makes no progress for an hour.

- **Probe projection** (`results/s1_probe.json`; random-weight models, 2 threads): CL scoring 0.22 h for F18 and
  0.58 h for X5 (upper bounds, without loading); AR scoring of all 38 runs 6.7 min; training the 12 new runs 9.6 h one
  at a time at 2 threads, or 8.1 h two at a time at one thread each.
- **New runs train one at a time, at 2 threads.** Two at a time would finish only 16% sooner, below the 30% the plan
  requires; and only 2 threads reproduce the stored curves bit for bit (DECISIONS S2-6).

## 9. What was known when this was written

- The paper's values under recorded commands (Table 2, `tab:checks`), and every registered result in
  `results/stats_report.txt`: on F18, ρ(λ1, E100) = 0.796, ρ(r̄J, E100) = 0.951, ρ(E1, E100) = 0.318,
  ρ(E8, E100) = 0.836.
- Task1c (`../rwm_repro/results/task1c_policy.json`): the actor reproduces the recorded joint commands only partly
  (R² 0.486 at offset −1, 0.391 at offset 0; verdict "inconclusive").
- `../rwm_repro/results/action_sensitivity.json` (exploratory, arm A only): the simulators respond strongly to joint
  commands (swapping them raises E8 by 228% at 2,500 iterations and 676% at 10,000).
- The 26 existing baseline runs' errors in the reproduction's own error measure
  (`../rwm_repro/results/baselines_eval.json`) and their training-loss curves (`baseline_run_*.json`).
- ICPAIAS computed the kinematic residual, P_on and E368 for 24 existing baselines (exploratory;
  `../ICPAIAS/results/r2_rank.json`: ρ(mean r_J, E368) = 0.978 over 41 models). Study AR's values are computed fresh
  by this folder's code.
- The 10,000-iteration models are already in F18 (Study TB, descriptive).
- Sessions S1–S2 ran their tests and probes on random-weight models, constructed systems and stored values only;
  test (vi) compared the 26 baselines' one-step predictions with the reproduction's code (predictions only), and
  test (ix) trained each class for 50 iterations into `.cache/`. S3 checked the job driver on random-weight models
  (`s1_eval.py --dry-run`, into `.cache/`).
- **No closed-loop rollout, no noisy-command check, and no baseline growth rate or kinematic residual computed by
  this folder's code existed** before this registration. No baseline growth rate existed anywhere.

## 10. After the commit

Neither this file nor `scripts/s1_stats.py` changes. A needed change goes to `ANOMALIES.md` in a new commit, and the
paper says so.

## 11. Files

- Code: `scripts/s1_policy.py`, `s1_rollout.py`, `s1_arch.py`, `s1_train_baseline.py`, `s1_eval.py` (the job),
  `s1_stats.py` (the analysis), `test_s1.py` (tests i–xi).
- Job outputs: `results/s1_models.json`, `results/raw_S1.npz`, `results/s1/` (per-model arrays, training summary,
  progress), new weights in `runs/s1_baselines/`.
- Analysis outputs: `results/s1_stats.json`, `results/s1_report.txt`.
- Hash: `results/prereg_hash_s1.json`.

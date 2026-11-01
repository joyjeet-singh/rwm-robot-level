# AIR 2027 — pre-registration (written before any model is run through this study's code)

Written 2026-11-01 (IST afternoon) in the cloud workspace, committed to this folder's git, hashed into
`results/prereg_hash.json`, and copied to the Mac (`~/Downloads/PDM/AIR 2027/prereg/PREREG.md`) before any
script of this folder computes a rollout, an exponent or a statistic. The Mac file's date is the independent
timestamp. This file is never edited after that; problems go to `ANOMALIES.md`.

## 1. Question

Learned simulators of legged robots are run forward on their own predictions for tens to hundreds of steps, but
are usually trained and checked one step ahead. Across the simulators this project has trained, plus compressed
copies of them made for on-board use, **which pre-deployment check best ranks how wrong a simulator becomes over
the horizon at which it is used?** The candidates are two labelled checks that need recorded future states
(one-step error; error over the 8-step training horizon) and two label-free checks that need only the network,
a recorded history and the commands (the leading finite-time Lyapunov exponent; a kinematic-consistency residual).

## 2. What is already known (stated so a reader can judge every test below)

The author of this file has read the earlier studies of the same models in `../rwm_repro` (TMLR submission),
`../NGCAPS`, `../ICMAI`, `../ICPAIAS` and `../PerconAI`. Their relevant results:
- NGCAPS (arena F18 below): λ1 ranks the 368-step error over its pool R (48 models), Spearman 0.85 [0.70, 0.93],
  and over R ∪ T (57) 0.90 [0.80, 0.94]; ρ(training horizon N, λ1) = −0.97 over 15 models; rollout-trained
  models have mean λ1 −0.49 /s, one-step/teacher-forced +0.94 /s; 368-step error 1.96 at N = 1, 0.19 at N = 32.
- ICMAI / X5: λ1 ranks the 368-step error over the 55 five-fold models, 0.66 [0.46, 0.79]; within the 40
  rollout-trained ones, not settled (0.21).
- ICPAIAS / X5: the mean rule-J residual ranks the 368-step error over the 55 models, 0.93 [0.84, 0.97],
  much of it by training method.
- TMLR: on short windows, teacher forcing beats rollout training at one step; at 368 steps rollout training wins
  on all ten held-out episodes.
- PerconAI / X5 (relative-L1, 25 models): 8-bit dynamic quantisation keeps 95% of the skill at 368 steps;
  30% magnitude pruning keeps 89% at 1 and 8 steps and is worse than predicting no change at 368 (−59%);
  structured width reduction without fine-tuning fails even at one step.

**Never computed by anyone, and not looked at before this file was written:** the one-step or 8-step error of
any model computed from every position of the held-out data (sections 4 and 5), any error from multi-start
windows, the correlation of any one-step or 8-step error with any long-horizon error, the Lyapunov exponent or
physics residual of any compressed model, any physical-unit error. The raw arrays of NGCAPS
(`../NGCAPS/results/raw_*.npz`) were copied into the workspace for check C1 only; no statistic is taken from them.

## 3. Arenas and pools

- **F18** (primary): held-out episodes 1 and 8, four non-overlapping 400-row trajectories (starts 999, 1399,
  7999, 8399). Every F18 model trained on the other eight episodes.
- **X5** (replication): five folds {1,8}, {0,2}, {3,4}, {5,6}, {7,9}; each model scored only on its own fold's
  four trajectories; 20 independent trajectories in all.

**Pool P on F18 (97 models)** = R ∪ T ∪ C:
- R (48), rollout-trained, NGCAPS's pool unchanged: Arm A M32/N8 seeds 0–4; Arm A ensemble-of-5 seeds 0–2;
  Gaussian-NLL objective seeds 0–4; width 124 seeds 0–4; contaminated windows seeds 0–2; duplicated windows
  seeds 0–2; 10,000-iteration runs seeds 0–2; sweep (M, N) ∈ {(1,8), (2,8), (8,8), (16,8), (32,2), (32,16),
  (32,32)} seeds 0–2.
- T (9), trained one step ahead: Arm B (teacher forcing) seeds 0–2; Arm B 10,000 iterations seeds 0–2;
  sweep (32, 1) seeds 0–2.
- C (40), new: the eight compressed variants below of Arm A M32/N8 seeds 0–4.

**Pool X on X5 (255 models)**: per fold, Arm A seeds 0–4, Arm A ensemble-of-5 seeds 0–2, Arm B seeds 0–2
(55 in all), and the eight variants of each Arm A model (200), i.e. 51 models per fold.

**Compressed variants** (made after training; none fine-tuned; code `scripts/compress.py`, committed with this
file): `fp16` (every parameter rounded to half precision and back); `int8w` (every GRU and linear weight matrix
rounded to a symmetric per-tensor 8-bit grid, biases untouched; weights only); `u10`, `u20`, `u30`, `u50`
(global magnitude pruning of all GRU and linear weights, PerconAI's definition); `s192`, `s128` (removing GRU
hidden units by PerconAI's L2 importance score, no fine-tuning).

**Released checkpoint** (descriptive only): the authors' five-head model on all 20 trajectories; its training
rows are not recorded, so it is never called held out.

## 4. Measures

The simulator's step is the released rollout: ensemble mean of the heads fed back, recorded actions with
action_offset = 1, float32. err = mean over the 45 normalised state numbers of |prediction − recording|.

- **E1** (labelled): one-step error at every position k = 32..399 of the arena's trajectories, each from the 32
  true rows before it; mean over positions and trajectories.
- **E8**, **E24**, **E100** (labelled): from every start k = 32, 36, 40, … (stride 4) with k + h − 1 ≤ 399, the
  model reads the 32 true rows before k and runs h steps on its own predictions; mean error over the h steps,
  the windows and the trajectories.
- **E368**: the project's standard protocol: one 368-step rollout per trajectory from row 32; mean over steps
  and trajectories.
- **λ1** (label-free): the leading finite-time Lyapunov exponent over forecast steps 1 → 368 of that rollout
  (K = 367 Jacobian products, dt = 0.02 s), Benettin's QR method with forward-mode Jacobian–vector products in
  float64, p = 4, start matrix from `numpy.random.default_rng(0)`; code `scripts/lyap.py`, copied unchanged from
  NGCAPS. A model's λ1 is the mean over its trajectories. It uses the history and the recorded commands, never
  a future state.
- **r̄J** (label-free): ICPAIAS's rule-J residual (joint angles change as the joint speeds say, trapezoid rule,
  each joint scaled by its RMS on the recording's training episodes of the model's fold) per forecast step of
  the same rollout, averaged over steps and trajectories.
- The **floor** holds the last history row; its E_h is computed the same way.

Non-finite values are ranked as the largest.

## 5. Questions and decision rules

Spearman correlations use average ranks (`scripts/ranks.py`). Bootstrap: 20,000 draws,
`numpy.random.default_rng(0)`, percentile intervals; both correlations of a difference come from the same draw;
draws with an undefined correlation are dropped and counted.

**Q1 (primary). On pool P, does λ1 rank E100 better than one-step error?**
Δ1 = ρ(λ1, E100) − ρ(E1, E100) over the 97 models; interval by resampling models. Two primaries, so the
verdict uses the **97.5%** interval (Bonferroni); the 95% interval is reported beside it.
**LABEL-FREE BETTER** if the interval lies above 0, **ONE-STEP BETTER** if below, **NOT SETTLED** otherwise.

**Q2 (primary). Same against the training-horizon error.** Δ8 = ρ(λ1, E100) − ρ(E8, E100); 97.5% interval;
**LABEL-FREE BETTER**, **TRAINING-HORIZON BETTER**, **NOT SETTLED**.

Horizon 100 is the length of the imagination rollouts over which the uncertainty-aware follow-up (RWM-U)
optimises its policy. Secondary, same statistics and labels with 95% intervals: Δ1 and Δ8 with E24 and with
E368 as the target.

**Q3 (secondary). Replication on X5.** The same Δ1 and Δ8 (target E100), with ρ computed within each fold over
its 51 models and averaged over the five folds; each draw resamples models within each fold. 95% intervals,
labels as Q1/Q2.

**Q4 (secondary). The physics residual as the label-free check.** On pool P, ρ(r̄J, E100) − ρ(E1, E100) and
ρ(r̄J, E100) − ρ(E8, E100); 95% intervals; labels as Q1/Q2.

**Q5 (secondary). Without the one-step-trained models.** Δ1 and Δ8 (target E100) on R ∪ C (88 models);
95% intervals; labels as Q1/Q2.

**Q6 (secondary). Does pruning without retraining make the simulator less stable?** On X5, the paired
difference λ1(u30) − λ1(parent) over the 25 Arm A parents, mean with an interval from resampling parents.
**MORE EXPANSIVE** if above 0, **MORE CONTRACTIVE** if below, **NOT SETTLED** otherwise.

**Descriptive (no verdicts).**
- D1: per group, mean E1, E8, E24, E100, E368, λ1, r̄J (F18 and X5); every correlation of section 5 also with
  Kendall's τ-b.
- D2: for each variant on X5, skill retained K(h) = 1 − Δ(h)/S(h) at h = 1, 8, 24, 100, 368, where Δ(h) is the
  variant's extra error over its parent and S(h) the parent's advantage over the floor, both averaged over seeds
  within each trajectory; intervals from resampling the 20 trajectories.
- D3: Arm A against Arm B on X5 at each horizon (normalised error; joint-angle error in degrees, base-velocity
  error in m/s and dead-reckoned planar position error in m, beside the distance the recording travels);
  intervals from resampling the 20 trajectories.
- D4: on each pool, how many models are worse than the floor at E100, by the sign of λ1.
- D5: the released checkpoint's E1, E100, E368, λ1, r̄J and physical errors on its 20 trajectories.
- D6: on F18, for each seed 0–2, choosing among the nine (M, N) configurations of seed s (the seven of the
  sweep, Arm A as (32, 8), and (32, 1)) by the lowest E1, E8, λ1 or r̄J: the E100 of the choice minus the lowest
  E100 there.

## 6. Checks before any result is trusted (differences only; no statistic)

- C1: this folder's code reproduces NGCAPS's stored arrays (`../NGCAPS/results/raw_F18.npz`, `raw_X5.npz`) for
  models A_s0, B_s0, mn_M32_N32_s0, A5_s0, w124_s0 (F18) and A_02_s0, A5_34_s1, B_79_s2 (X5): err and floor
  bit for bit, λ1…λ4 within 1e-12.
- C2: u0 and s256 reproduce the parent's rollout bit for bit; fp16 weights equal the half-rounded parent's;
  int8w changes no weight by more than half its scale.
- C3: Jacobian–vector product against a central finite difference on one step, float64, for A_s0's u30 and s128
  variants: relative error below 1e-5.
- C4: the recording obeys rule G to 1e-6 on every consecutive pair inside an episode, and its scaled rule-J
  residual has root-mean-square 1 on the episodes that define the scale.
- C5: the 368-step window error equals the rollout error to 1e-6.
- The number of models with any non-finite quantity is reported; none is excluded.

A failing check is recorded in `ANOMALIES.md` and fixed before any statistic is computed; no question, pool or
rule changes.

## 7. Limits stated in advance

F18 has four independent trajectories, so each model's error carries their sampling noise, which resampling
models ignores; X5 replicates on 20 trajectories across five splits. Models share training data, so model-level
intervals may be too narrow. The pools are this project's models and compressed copies chosen here, not a random
sample of simulators, and the verdicts are about them. λ1 is a finite-time exponent in the network's own
coordinates and depends on the start direction by tenths of 1/s (ICMAI A-01); the start matrix is fixed. int8w
is weight-only. The dead-reckoned position ignores roll and pitch and is a comparison device applied identically
to prediction and recording, not odometry. One robot, one gait, flat ground, simulated data; no controller is
trained.

## 8. Environment

Linux cloud workspace, 2 CPU cores, Python 3.13.18, torch 2.15.0 (CPU), numpy 2.5.3: the versions NGCAPS and
ICMAI ran on. Nothing outside `AIR 2027` is modified.

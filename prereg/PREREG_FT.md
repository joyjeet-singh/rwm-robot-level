# Pre-registration of study FT: fine-tuning the magnitude-pruned copies

Written in session R3 of the AIR 2027 revision, 2026-11-01, before any fine-tuned magnitude-pruned model exists.
Once committed, this file and `scripts/ft_stats.py` do not change. A needed change goes to `ANOMALIES.md` in a new
commit, and the paper says so. Everything else computed in this revision is exploratory; study FT is the one
registered study of the revision.

## 1. Question

Does a short fine-tuning, with the pruned weights held at zero, restore the long-horizon skill that magnitude pruning
removes?

Terms used below. *Magnitude pruning at level p* sets to zero the fraction p of the weight-matrix entries with the
smallest absolute value, over all weight matrices of the model together (`compress.unstructured`). *Skill* is how
much better a model predicts than the floor, the forecast that holds the last true state. *Horizon h* is the number
of 0.02 s steps predicted without seeing the true state.

Scope: one robot (ANYmal D), one trotting gait, simulated data, recorded commands, no policy trained.

## 2. Models

- **Folds.** The five X5 folds: held-out episode pairs {1, 8}, {0, 2}, {3, 4}, {5, 6}, {7, 9}.
- **Parents P.** `rid(f, s) = common.fold_dir("A", f, s)` for seeds s ∈ {0, 1, 2}: 15 rollout-trained models with
  one head and two GRU layers of 256 units. These seeds are used because PerconAI's fine-tuned controls exist for
  them. All 15 controls exist (`../PerconAI/runs/e3_ft/e3ft_<rid>_w256/weights_500.pt`, checked 2026-11-01).
- **Arms, per parent:**
  - **U30** = `compress.unstructured(P, 0.30)`, not trained.
  - **U30F** = U30 fine-tuned for 500 iterations, with the mask held.
  - **U50**, **U50F**: the same at level 0.50. Secondary; they run only under the time rule of §8.
  - **CTRL** = PerconAI's `e3ft_<rid>_w256`: P fine-tuned for 500 iterations by the same protocol, with nothing
    pruned. Read only.

Run ids: `ft_u30_<rid>` and `ft_u50_<rid>`.

## 3. Fine-tuning protocol

That of `../PerconAI/scripts/e3_finetune.py` (sha256 `b4e92994…6802b2`), unchanged except that training starts
from the magnitude-pruned copy, and every pruned entry is set to exactly zero after every optimiser step. In full:

- the rollout loss (`teacher_forcing=False`, loss `"mse"`), weighted by the reference config's loss weights;
- the fold's eight training episodes, in 40-row windows;
- batch 256;
- Adam with fresh state, at the reference learning rate (1e-4) and weight decay (1e-5);
- no gradient clipping;
- 500 iterations;
- the parent's seed for `torch.manual_seed`, `np.random.seed` and the batch generator;
- 2 threads;
- before training, the settings (batch, learning rate, weight decay, loss weights, gradient clipping, training
  episodes, number of training windows, held-out episodes) are asserted equal to the source run's record (the
  `hyperparameters` field of `../rwm_repro/results/step5_<rid>.json` or `x3_run_*.json`).

The controls' records (`../PerconAI/results/raw/e3_ft/e3ft_<rid>_w256.json`) show the same settings for all 15:
batch 256, learning rate 1e-4, weight decay 1e-5, no clipping, 500 iterations, Adam with fresh state, 2 threads, rollout
loss; they differ only in the training episodes and the number of training windows (7688 for fold {0, 2}, 7687
otherwise). They took 0.85–0.96 s per iteration.

Code: `scripts/ft_unstructured.py`.

## 4. Scoring

This paper's measures, computed by `compute.per_model` on each parent's own fold arena: the four non-overlapping
400-step trajectories of the fold's held-out episodes, the first 32 steps given as true history.

- E1, E8, E24 and E100: the mean absolute error of the normalised state over h-step forecasts, from windows that start
  at every fourth position after the 32 history steps (E1 at every position).
- E368: the mean error over the 368-step forecast that follows the history.
- λ1: the largest finite-time Lyapunov exponent along the trajectory. r̄J: the mean joint-rule residual of the
  prediction.
- The floor is holding the last true state, scored the same way.
- Every arm, P and CTRL included, is scored by the same code in one run on this machine (`scripts/ft_eval.py`).

## 5. Unit and estimator

- The unit is the trajectory: the 20 X5 trajectories, ordered by fold as in §2, then by start row. For each arm and
  trajectory, the error is first averaged over the three seeds.
- **K_orig(x, h)** = 1 − mean_t(e_x − e_P) / mean_t(e_floor − e_P), exactly the estimator of `scripts/stats.py`'s
  D2 (computed, as there, at the scores' float32 precision). K = 1: the arm keeps all of P's skill over the floor;
  K = 0: none of it.
- **K_ctrl(x, h)** is the same with CTRL in place of P.
- **Intervals.** 20,000 draws of the 20 trajectories with replacement (whole trajectories), from
  `numpy.random.default_rng(0).integers(0, 20, size=(20000, 20))`; one set of draws for every quantity; each
  statistic recomputed in full on each draw (a ratio of resampled means; every difference paired by trajectory);
  percentile intervals by numpy's default linear rule (numpy ≥ 2), as reimplemented in `ft_stats.percentile` so
  that it runs at the scores' float32 precision, as D2's intervals did in the cloud. Thresholds are compared with
  these float32 endpoints. No draw is dropped; a NaN draw cannot arise and stops the analysis.
- **Undefined draws** (as PerconAI's `PREREG_R2.md` §2.1): if the reference's skill mean_t(e_floor − e_ref) is ≤ 0 on
  a draw, K is undefined there. It counts as K = −∞, sorts below every finite draw, and an interval endpoint that
  falls on it is −∞. In a difference of two K (FT1, the one-step reading), such a draw counts as −∞ for the lower
  endpoint and +∞ for the upper, so it can only widen the interval. The number of such draws is reported.
- "Above" and "below" are strict: an endpoint equal to a threshold is neither.

## 6. Questions and rules

### Primary: a family of two, with 97.5% intervals (1.25th to 98.75th percentile)

- **FT1.** d = K_orig(U30F, 100) − K_orig(U30, 100).
  - Interval above 0: **FINE-TUNING RESTORES SKILL**.
  - Interval below 0: **FINE-TUNING REMOVES SKILL**.
  - Otherwise: **NOT SETTLED**.
  - If P has no skill over the floor on the full sample (mean_t(e_floor − e_P) ≤ 0), FT1 is reported as NOT
    COMPUTABLE.
- **FT2.** K_ctrl(U30F, 100), with PerconAI Q3's thresholds and labels (`../PerconAI/PREREG_R2.md` §3, lines 68–80,
  and §5, lines 95–96; `PREREG_R3.md` applies them unchanged), copied verbatim. Only the horizon (100 steps here; 368
  for PerconAI's primary) and the interval level (97.5% here) differ. In the quoted text, e_w is the fine-tuned
  pruned arm (U30F here, U50F in the secondary reading), so Δ^ctrl_w = mean_t(e_U30F − e_CTRL). Labels, checked in
  this order:
  - "**NO SKILL TO LOSE** if mean(e_floor − e_ctrl) ≤ 0 on the full sample. K is then not read; Δ^ctrl_w =
    mean(e_w − e_ctrl) with its interval is reported instead."
  - "**NEARLY AS SKILLED** if lo > 0.90;"
  - "**KEEPS MOST** if lo > 0.50 (and not above 0.90);"
  - "**LOSES MOST** if hi < 0.50;"
  - "**NOT SETTLED** otherwise."
  - "'Above' and 'below' are strict. An endpoint exactly equal to a threshold (0.90 or 0.50 for Q3 …) does not lie
    above or below it, so it cannot earn the label that needs it."

### Secondary: 95% intervals, not corrected for multiplicity, reported as secondary

The labels of the one-step reading and of the λ1 and r̄J changes are this registration's own; the revision plan
asked for these readings but gave them no labels.

- FT1 and FT2 at h = 24 and at h = 368 (E368).
- FT1 and FT2 for U50 / U50F at h = 100, if U50F runs (§8).
- **One-step reading.** K_orig(U30F, 1) beside K_orig(U30F, 100), and their difference d1 = K_orig(U30F, 1) −
  K_orig(U30F, 100): interval above 0, **ONE-STEP READING OVERSTATES**; below 0, **ONE-STEP READING
  UNDERSTATES**; otherwise **NOT SETTLED**.
- **Mean change in λ1** for U30F − P and for U30 − P, and the same for **r̄J** (trajectory bootstrap, seeds averaged
  within a trajectory): interval above 0, **INCREASES**; below 0, **DECREASES**; otherwise **NOT SETTLED**.

### Descriptive (no verdicts)

K of every arm against P and against CTRL at h = 1, 8, 24, 100 and 368, with 95% intervals, and the mean error of
every arm and of the floor. Any further analysis is exploratory and labelled so.

## 7. Checks before the analysis

A failure is reported, and the affected arm is excluded from every question it touches (if any of an arm's 15 runs
fails C-FT1, C-FT2 or C-FT3, that arm is excluded, and those questions are reported EXCLUDED with the failed check).
A failed check does not stop the study: `ft_unstructured.py` records it in the run's JSON and exits normally; only a
crash stops the job.

- **C-FT1.** After fine-tuning, every pruned entry is exactly zero, and the count of zeros in the weight matrices
  equals the count pruned. Checked at the end of training and again by `ft_eval.py` on the reloaded weights, with
  the mask re-derived by pruning the parent again.
- **C-FT2.** Every loss trace is finite.
- **C-FT3.** The settings equal the source run's record. They are compared before training; on a difference the
  run's JSON records it, the model is not trained, and it is not scored.
- **C-FT4.** This machine's E100 for P and U30 agrees with the stored X5 values (`results/raw_X5.npz`, computed in
  the cloud) within 1e-3 relative, per trajectory. A larger difference is reported as an anomaly; every question
  still uses this machine's scores for every arm. (The test of §10 found 1.3e-7 for P and 2.0e-7 for U30 on fold
  {1, 8}, seed 0.)

`scripts/ft_stats.py` refuses to run if any check result is missing, or if any score it reads is missing for an
arm that is not excluded. If a score used by a question is not finite, that question is reported NOT COMPUTABLE
(non-finite score); the other questions are unaffected. If P has no skill over the floor on the full sample at a
horizon that FT1 or the one-step reading uses, that reading is reported NOT COMPUTABLE.

## 8. Time rule (fixed by the probe before any fine-tuned model exists)

U30F always runs. U50F runs only if the projected total (U30F, plus U50F, plus scoring) is at most 8 hours on this
machine.

Probe (`./py scripts/ft_unstructured.py --probe 20`, `results/ft_probe.json`; fold {1, 8}, seed 0, level 0.30; macOS
13, x86-64, Python 3.11.15, torch 2.2.2, numpy 1.26.4, 2 threads):

- 0.896 s per iteration (mean after the first; maximum 0.959 s); setup 3.4 s per run;
- one `per_model` scoring: 7.4 s;
- projected: U30F 1.88 h (15 runs × 500 iterations), U50F 1.88 h (the pruned matrices stay dense, so the cost is the
  same), scoring of all six arms (90 models) 0.19 h; **total 3.95 h ≤ 8 h**.

**Result: U50F runs.** Arms: P, U30, U30F, CTRL, U50, U50F.

## 9. What was known when this was written

- **Table 1's K for U30 without fine-tuning** (X5, seeds 0–4, 20 trajectories; `results/stats.json`,
  `D2_skill_retained_X5.u30.h<h>`): h = 1: 0.902 [0.876, 0.919]; h = 24: 0.825 [0.801, 0.844]; h = 100: 0.394
  [0.291, 0.472]; h = 368: −0.409 [−0.87, −0.08].
- **PerconAI's registered results** (unit removal, Q3 at h = 368, 95% intervals):
  - width 192 KEEPS MOST: K_ctrl 0.839 [0.768, 0.896] with seed 0 (`../PerconAI/results/r2_verdicts.json`, `Q3.w192`),
    0.896 [0.840, 0.936] with seeds 1–2 (`r3_verdicts.json`, `primary.Q3.w192`);
  - width 128 NOT SETTLED: 0.533 [0.404, 0.645] (seed 0) and 0.444 [0.036, 0.645] (seeds 1–2) (same files, `w128`);
  - fine-tuning alone improved the control: K of CTRL against the original, seeds 0–2 pooled, 1.043 [1.013, 1.082]
    at h = 368 and 1.021 [1.011, 1.033] at h = 100 (`r3_verdicts.json`,
    `descriptive.controls_K_vs_original.pooled`).
- No fine-tuned magnitude-pruned copy existed, and none had been scored. The test of §10 trained 3 iterations, twice,
  on fold {1, 8}, seed 0 (saved to `.cache` only, to test the reload check); the probe trained 20 in memory; a check
  of the run path trained 2 iterations on fold {3, 4}, seed 1, into `.cache`, and deleted them. None was scored.
- K_orig(U30, h) for seeds 0–2 alone, the second term of FT1, could already be computed from `results/raw_X5.npz`.
  It was not computed or looked at; only the seeds 0–4 values above (Table 1) were known.
- `ft_stats.py` was run only on synthetic scores in `.cache/test_ft/`.

## 10. Files and outputs

Committed with this file:

- `scripts/ft_unstructured.py`: fine-tuning and the probe.
- `scripts/ft_eval.py`: scoring of every arm, checks C-FT1 to C-FT4.
- `scripts/ft_stats.py`: §5–§7, exactly.
- `scripts/test_ft.py`: tests (masks held; two runs identical; settings for all 15 parents; C-FT4 on one parent; the
  K estimator reproduces `D2_skill_retained_X5.u30` exactly; rules on constructed inputs; end to end on synthetic
  scores).
- `results/ft_probe.json`: the probe and the time rule's decision.

Written later, by `revision/jobs/R3.zsh` and then `./py scripts/ft_stats.py`:

- `runs/ft/<id>/weights_500.pt` (not committed) and `results/raw/ft/<id>.json` (per run: settings, loss traces,
  `mask_zeros`, `pruned_entries`, `level`, timing, versions);
- `results/raw_FT.npz`, `results/ft_models.json` (per model: arm, parent, fold, seed, E_h, E368, λ1, r̄J, finite
  flag, software versions; check results), `results/ft/progress.json`;
- `results/ft_stats.json` and `results/ft_report.txt`.

# R1 specification (verbatim from REVISION_PLAN.md, block R1)

**Goal:** one script, `scripts/revision_analyses.py`, that answers three reviewer objections from the stored data. It writes `results/revision_exploratory.json` and a summary of at most 60 lines in `results/revision_exploratory.txt`. Nothing here is registered; everything is exploratory.

**Read:**
- `scripts/stats.py` (whole).
- From `scripts/common.py`, only the definitions of `arena`, `ci`, `FOLDS` and `N_BOOT`.
- `per_model` in `scripts/compute.py`, to learn what each stored array holds.
- `results/model_table.json` and `results/stats.json` through `./py -c` only.
- The raw `.npz` files through code only.

**Stored arrays** for a model key k, per held-out trajectory:
- `E1_k`, `E8_k`, `E24_k`, `E100_k`: errors at those horizons.
- `err_k`: trajectory × 368 forecast steps.
- `log1_k`: trajectory × 367; the per-step log stretch of the leading direction.
- `rJ_k`: trajectory × 368.
- `lam_k`: trajectory × 4, the first four exponents. It is **not** λ1, so don't use it as λ1.
- Floors: `floor_*_<fold>`.

The registered λ1 is the mean over trajectories of `sum(log1_k[t]) / (367 × 0.02 s)`. This reproduces `model_table.json`'s `lam1` exactly; assert it.

**Reuse, never copy.** Import from `stats.py`: `spearman`, `boot_rho`, `boot_diff`, `boot_diff_within`, the pools, and the per-split averaging. Before computing anything new, assert that the registered numbers this script recomputes equal `results/stats.json` exactly: Table 2's correlations, `Q1`, `Q2`, `Q3_vs_E1`, `Q3_vs_E8`, `Q4_vs_E1`, `Q4_vs_E8`, `Q5_vs_E1`, `Q5_vs_E8`.

**A1 Horizon-matched label-free checks.**
- For k in {8, 24, 100, 368}, define:
  - λ1(k) = mean over trajectories of the sum of the first min(k, 367) entries of `log1`, divided by min(k, 367) × 0.02 s.
  - r̄J(k) = mean over trajectories of the mean of `rJ` over forecast steps 1..k.
- Compute the Spearman correlation of λ1(k), r̄J(k), E1 and E8 with each target E24, E100 and E368:
  - on F18: all 97 models, the 57 trained models, and the 40 compressed copies;
  - on X5: within each split, then averaged over the five splits.
- 95% intervals as the registered analysis computes them: models resampled, within splits on X5.
- Also compute the differences r̄J(8) − E8 and r̄J(100) − E8, both against E100, with intervals.

**A2 Wider intervals.**
- **(a) Two-level bootstrap.** Each draw resamples clusters and trajectories together.
  - A cluster is a parent with its eight compressed copies. Every other model is its own cluster.
  - On X5, clusters and trajectories are resampled within each split.
  - In a draw, a model's E_h, λ1 and r̄J are means over the drawn trajectories, with λ1 per trajectory from `log1` as above.
  - Recompute Table 2's correlations and the registered differences in every draw.
  - Use 20,000 draws with `numpy.random.default_rng(0)` if the run takes under about 8 minutes, else 5,000; record which.
  - Report the percentile intervals beside the registered ones, with the ratio of widths. For each registered verdict, say whether its interval under this scheme would still exclude zero. This is descriptive only; the registered verdict stands.
- **(b) X5 without split {1, 8}.** Recompute Table 2's X5 column and Q3's differences averaged over the other four splits, with the registered resampling.
- **(c) Holm correction.** The family is the secondary registered tests: `Q3_vs_E1`, `Q3_vs_E8`, `Q4_vs_E1`, `Q4_vs_E8`, `Q5_vs_E1`, `Q5_vs_E8` and `Q6`.
  1. Reproduce the registered bootstrap draws (same generator and the same order as `stats.py`), and assert that their intervals equal `stats.json`.
  2. Take the two-sided p = 2·min(P(d ≤ 0), P(d ≥ 0)).
  3. Apply Holm at family-wise 5%, and list which verdicts survive.

**A3 Split-half.** On each arena, each held-out episode contributes two trajectories; confirm this with `common.arena(fold)["episode"]`.
- Assignment 1: the checks (E1, E8, λ1, r̄J) come from the first trajectory of each episode, and the target E100 from the second. Assignment 2 swaps them.
- Compute Spearman for each assignment and average the two.
- Report F18 (all, trained, compressed) and X5 (within-split average), with model-resampled 95% intervals.

**Not needed:** the per-variant growth-rate shifts already exist in `results/stats.json`, under `lam_shift_X5` (registered output, descriptive). Don't recompute them.

Keep the whole run under about 10 minutes: vectorise the draws, and build the per-trajectory arrays once.


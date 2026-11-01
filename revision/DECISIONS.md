# Decisions

## R0
- poppler: `brew install poppler` was running ~40 min compiling Python 3.15 from source on Intel/macOS 13; stopped it and used the pypdf route the plan allows when brew is unavailable. Conservative: no further system changes. Author may install poppler later; the scripts then use it automatically.
- Git identity: left as the existing global one (the plan sets neutral values only when none exists).

## R1
- λ1 from `log1` cannot reproduce `model_table` lam1 bit for bit: the cloud run summed in torch (`lyap.py:98`), and numpy, torch 2.2 and fsum all differ in the last bit for some models (max 2.2e-16). The script asserts |Δ| < 1e-12 and asserts that every rank-based registered number is identical; the registered numbers themselves are asserted exactly from `model_table.json` through stats.py's own functions.
- Environment: `./py` puts `.pylib` (numpy 2.2.6, pulled in by R0's scipy install) ahead of the venv's numpy 1.26.4. torch 2.2.2 then cannot convert numpy arrays ("Numpy is not available"). R0_env.md records numpy 1.26.4. R1 needed no torch: `common.arena` is called with `C.tensors` stubbed for the episode list. Not changed (outside R1's block). **Any session that trains or scores models (R3) will fail under `./py` until this is resolved.** Options: remove `.pylib/numpy*` (scipy 1.13.1 runs on numpy 1.26), or reinstall scipy into `.pylib` with `--no-deps`. Recommendation: remove `.pylib/numpy` and `numpy-2.2.6.dist-info`, then rerun R0's smoke check.
- Registered draws for Holm are taken by wrapping `common.ci` while stats.py's `boot_diff`/`boot_diff_within` run, so that no loop is copied.
- A2a (reviewer SHOULD FIX): each F18 subset resamples its own clusters (all 57, trained 57, compressed 5, RC 48), so a subset never has zero clusters. With 5 clusters, the compressed intervals are unreliable.

## R3
- Environment (R1's open point): an earlier R3 attempt moved `.pylib/numpy` and `numpy-2.2.6.dist-info` to `.cache/pylib_removed/` (R1's recommended option; inside this folder, reversible). `./py` now uses the venv's numpy 1.26.4; scipy 1.13.1 imports; torch↔numpy works. R0's smoke check was not rerun in R3 (not in R3's block); later sessions that rebuild should note that numpy is now 1.26.4 on this machine.
- FT2 copies PerconAI Q3's undefined-draw rule (PREREG_R2.md §2.1: K = −∞ on a draw where the reference has no skill). The block names D2's estimator, which has no such rule; the copy only matters on such draws and can only widen intervals (FT1's difference: −∞ low, +∞ high). Conservative choice.
- FT1 is reported NOT COMPUTABLE if P has no skill over the floor on the full sample (guard; not expected).
- air-stats-reviewer (1 MUST FIX, 7 SHOULD FIX, 5 NOTE), all addressed before registration. MUST FIX: a failed C-FT1/C-FT3 aborted the run, so "exclude the arm" was unreachable. Now a failed check is recorded and the run exits 0 (C-FT3: recorded, not trained, not scored), ft_eval records it, and ft_stats excludes the arm without needing its scores. Secondary labels (one-step; INCREASES/DECREASES) are kept and declared as this registration's own.

## R4
- All 8 requested records VERIFIED; none left out of refs.bib. Quotes are fragments (tool limit), so text should use them as fragments.
- Manek/Kolter author order follows the plan (arXiv order); NeurIPS page lists Kolter first. Left as in the plan; R8 may check.
- RAMBO refs [75],[76] = COMBO, MOPO were reported by the checker, not confirmed by the reference list; the text should not name them without a check.
- check_bib.py: strict mode now also fails on uncited entries (previously only reported); --allow-uncited makes them warnings.

## R5
- scale_ratio uses the unrounded c2_pct (0.133%) from rwm_repro/results/paper_numbers.json: 100/0.133 = 752, so "about 750". Dividing the printed 0.13% would give "about 770"; R6 should write the sentence next to {{budget_pct}} so the two agree (0.13% is a rounding of 0.133%).
- Plan item (d): lamshift_*, lam_*, lampos_* for all variants already exist in air_numbers.py; R5 added only totals (n_unstable_removed 7 of n_removed_copies 150; n_unstable_wide 7 of n_wide_copies 75). A copy = one parent x one variant.
- A2 "holds" means the two-level interval gives the registered verdict (excludes 0 with the estimate's sign for a directional verdict; includes 0 for NOT SETTLED). a2_median_ratio is over all 29 intervals (16 Table 2 + 13 registered tests); the two sub-medians are also registered.
- stale_ho_h100 (the held-out-pair stale command at h = 100, released checkpoint) prints with two significant figures like stale_h100 (22%, [6.3%, 42%]); the earlier study printed 22.5 [6.3, 41.7]. Source n5_stale_armA_ci.json, results.released.held_out_n4.by_horizon[100].
- hp_*: the optimiser is not in the hyperparameters field; it is read from rwm_repro/src/rwm_train.py (make_optimizer, torch.optim.Adam, asserted in code). All seven loss weights are non-zero. Equality asserted over 25 parents' records (5 folds x 5 seeds; fold 18 from step5_armA_seed0-4.json, the rest x3_run_armA_ho*_seed0-4.json); hidden size/layers only exist in the 20 x3 records.
- perc_*: "first" = seed 0 (PerconAI r2.* keys), "rep" = seeds 1-2 (r3.rep.*), "pool" = all three (r3.pool.*). Q4 is the 3000-iteration, 368-step reading. results/paper_numbers.json source paths are now relative to this Mac's PDM folder (they were the cloud's /mnt path).

## R6
- T6b wording weakened: u10 and u20 growth-rate shifts are small but their intervals exclude zero (lam_shift_X5), so the text says moderate pruning "adds error while barely changing" the growth rate, not "without changing". Conservative option.
- T20 uses the 368-step PerconAI keys (perc_K_pool_*_h368, perc_Q4_pool_*), because R5's Q4 keys are 368-step readings; "three seeds" = pool. The plan's "at 100 steps" is not used.
- T2/T19: RAMBO is cited for keeping the lowest-validation-error members. Its "previous works [75, 76]" are not named (R4 could not confirm them). MOPO and COMBO are cited for building on one-step model ensembles; R4's COMBO quote does not cover this. R8's claim/bib check should confirm, or drop COMBO to the uncertainty clause.
- scripts/air_numbers.py fk(): abs(v) >= 99.95 printed K <= -100% as "100%" (e.g. K_s128_h100 was "100%", source -1.82). Fixed to v >= 99.95. Only unused keys changed (10 keys, diffed). Not a result change.
- Length (T22): beyond the plan's cut order, the run-time Sec. 4.3 text went to R6_moved.md. TorchScript timing, per-head coverage, regret and some physical numbers were dropped from the body and listed in the appendix comments. Reproducibility note shortened and made consistent with Sec. 3.

## R7
- Figure 2 third panel is the kinematic residual (rJ, pool P); its Spearman is `stats.json` D1_rho_F18["rJ~E100"], not typed.
- A3 table uses the existing a1 results (checks over 8/24/100/368 steps against E100 on F18 all, trained, compressed and X5); no new computation.
- A1 settings for teacher-forced, N=1, 10k, width-124, ensemble and likelihood models are read from their rwm_repro run records (new `hp_*` keys); asserted equal to the parent's except as stated.
- A2 quotes PREREG §2 only where a key exists; the TMLR and PerconAI points are stated without numbers.
- Did not place shared_pct, cov1_*, ng_pathB_* or the TorchScript timings in the appendix: their meaning is not documented in the notes, so a wrong caption was the greater risk.
- Fig. 2 caption in the body now names three panels (a few words longer; two-column build still 6 pages).
- figures/fig_horizon.* were regenerated by make_figures.py (same data, binary differences only) and are left unstaged.

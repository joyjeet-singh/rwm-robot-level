# Step 1 log

## S0 status: COMPLETE (2026-11-06 14:00)
Tag air2027-step1-start at 6887672; branch step1; plan committed b96ab1f. Env matches R0 (py3.11.15, torch 2.2.2, numpy 1.26.4, scipy 1.13.1, tensordict 0.3.2; pypdf ok); disk 35 GiB free.
Notes: step1/notes/S0_inventory.md, S0_known.md. Key finding: ICPAIAS already computed rJ/P_on/E368 for 24 baselines (exploratory) — PREREG_S1 must declare it.
Lite: train.py needs wandb stub, rsl_rl via sys.path, CUDA→CPU (torch.load without map_location), and blocks waiting for assets/data CSVs.
Open: baselines_eval.json contents only skimmed.

## S1 status: COMPLETE (2026-11-06 15:10)
Added scripts/s1_policy.py, s1_rollout.py (score_gru), test_s1.py, s1_probe.py, and results/s1_probe.json; commit e52aa65.
All 13 tests in test_s1.py pass in 15 s. task1c R² was reproduced exactly. The linear exponents are within 4e-4 of the true values. A_s0 and B_s0 match the stored λ1 and r̄J within 3.2e-5 relative. The replay closed loop reproduces the open-loop rollout exactly.
Probe: 8.0 s per one-head model and 11.3 s per five-head model. Projected 0.22 h for F18 and 0.58 h for X5, not counting weight loading.
Choices are in DECISIONS S1-1…7, which S3 must register: p=1, no observation noise, the noise seed, the λ1_cl start directions, the imagined-return scope (with the authors' dof_acc quirk), divergence, and the probe scope.
No trained simulator was driven by the policy or by noise.

## S2 status: COMPLETE (2026-11-06 16:58)
Added scripts/s1_arch.py (score_arch; adapters for the MLP, RSSM and Transformer), s1_train_baseline.py (a copy of the reproduction's script, with sha256), s2_probe.py, and tests (vi)–(ix) in test_s1.py. All 22 tests pass in 238 s. Commit ca2bc12.
(vi) All 26 runs held out episodes 1 and 8; their one-step predictions equal the reproduction's code exactly. (vii) JVP matches finite differences within 4e-9. On a GRU, the generic code equals lyap and evalh exactly. (ix) The 50-iteration losses equal the stored curves bit for bit at 2 threads.
Probe: scoring all 38 runs takes 6.7 min. The 12 new runs take 9.6 h one at a time at 2 threads, or 8.1 h two at a time.
DECISIONS S2-1…6. S3 must register the RSSM map (argmax mode, zero derivative) and how a baseline that diverges under recorded commands enters AR.
No trained baseline was scored. Nothing was trained beyond 50 iterations, and only into .cache/.

## S3 status: COMPLETE (2026-11-06 17:33)
Registered studies CL and AR: prereg/PREREG_S1.md plus all the S1 code in commit 2cb7d24 (hash results/prereg_hash_s1.json, 924a57a). PREREG_S1.md and s1_stats.py are chmod a-w.
New: scripts/s1_eval.py (the job driver: checks, CL, training, AR; resumable), s1_stats.py (CL1–CL9, AR1–AR4, exploratory two-level), and tests (x) and (xi). test_s1 passes 24/24.
air-stats-reviewer: no MUST FIX; 7 SHOULD FIX and 5 notes fixed (DECISIONS S3-8). Decisions S3-1…8.
Job: step1/jobs/S3.zsh (zsh -n ok), committed 7c753fc. Expect about 11 h, of which 9.6 h is training one run at a time at 2 threads.
The dry run used random weights only, into .cache/s1_dry. No trained simulator was driven before 2cb7d24.
Open: figures/fig_horizon.{pdf,png} were already modified when S3 started. They are not S3's and were left unstaged.

## S4 status: COMPLETE (2026-11-06 18:05)
Lite policy training runs on this CPU with no install: scripts/lite/run_lite.py, adapter.py, test_lite.py (10/10, 11 s), stubs/wandb/.
Adapter equals the rebuild exactly (200 windows, 1 and 5 heads) and the upstream class with the released weights. Smoke: 1,024 robots, 10 its: released 3.41 s/it, A_s0 2.78 s/it, 0.70 GB; rewards finite.
500 its: 0.47/0.39 h at 1,024 robots, 1.88/1.59 h at 4,096 (released/A_s0). Recommend the CPU.
Step 2: use --holdout (window filter) and 5-head rebuilds (A5_*) so the uncertainty penalty is active. Notes: step1/notes/S4_lite.md; DECISIONS S4-1…3.
No trained simulator was driven by the CL/AR study code; this task touches no registered result. figures/fig_horizon.* left unstaged (not mine).

## S5 status: COMPLETE (2026-11-06 17:44)
Study TB (descriptive): scripts/s1_budget.py -> results/s1_budget.{json,txt}; notes step1/notes/S5.md. All inputs existed before this plan; nothing rescored.
Still falling: 6/6 at 2,500 its, 5/6 at 10,000 (slopes ~10x smaller). At 10,000: TF still better at 1 step, worse at 100, lam1 >0 for TF, <0 for rollout, in all 3 seeds.
FT control: 500 extra its add little (P holds 0.956-0.990 of CTRL skill).
Open: figures/fig_horizon.* still not mine, unstaged.

## S6 status: COMPLETE (2026-11-07 04:56)
Job S3 finished; checks S1-C1…C5 all PASS. s1_stats.py run unchanged: CL1 WORSE WITH POLICY COMMANDS (−0.120 [−0.220, −0.052]), CL2 NOT SETTLED, AR1 LABEL-FREE BETTER (+0.796 [+0.368, +1.025]), AR2 NOT SETTLED.
New: scripts/s1_numbers.py (162 keys), scripts/s1_extra.py → results/s1_extra.json (exploratory X5 noisy and closed-loop ρ, AR divergence list).
Appendix A8 (CL), A9 (AR), A10 (TB, descriptive). Body: Discussion (3) and Conclusion only. Limitations, abstract and intro unchanged (still true; DECISIONS S6-2, S6-3).
check_claims 107/107 (25 new), check_numbers and submission_checks pass. Two-column 6 pages, submission 19.
Commit cf87927 (with raw_S1.npz, s1_models/stats/report/extra, results/s1/). Notes step1/notes/S6.md; DECISIONS S6-1…7.
Open: figures/fig_horizon.* still not mine, unstaged (the PDFs use the working copy, which looks identical).

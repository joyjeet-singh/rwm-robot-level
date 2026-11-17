# Option D log

One status line per session or job (Rule 10).

## D0 started (2026-11-14 18:34)

## D0 status: COMPLETE (2026-11-14 18:56)
Branch optd from optc (tag air2027-optd-start); Option C's leftovers and the Option D plan committed first.
Facts in optd/notes/D0_facts.md: all 22 Step 2 actors exist; the 18 registered match s2_main.json's sha256. No contradiction with the design.
Code in scripts/d/: sims, checks, walk_test, evaluate, record, finetune (+ dprogress). The tests (scripts/d/test_d0.py, 9 groups) all pass in 137 s.
REL on the paper's arena: E8 and λ1 equal model_table; r̄J equals Step 2's recomputation on this Mac (DECISIONS D0-1).
Fine-tuning: 5.0 s per step on the 1M rung (walking, rollout windows), so 1,280 steps take about 1.8 h.
Rehearsal at toy size passed; the record job (optd/jobs/record.zsh) passes zsh -n and has not run. Estimate 1–2 h.
No Option D number exists yet except the toy rehearsal's (under .cache/).

## Record job status: COMPLETE (2026-11-14 20:10)
results/d_record.txt and .json written; gates 1 and 2 passed.

## D1 started (2026-11-14 20:12)

## D1 status: COMPLETE (2026-11-14 20:48)
Stage 1 (optd/notes/D1.md): 6 of Step 2's 9 fail, seeds never disagree (0 of 9); the stop rule would have stopped 5 of the 18 registered policies (all failed anyway).
scripts/d/train_policy.py equals Step 2's recipe (REL, 3 iterations: logs identical); stop rule via the log hook; failed-by-simulator rule.
scripts/d/ladder.py: verdict and search rules, two-worker queue, fine-tune lock, crash guard, toy and rehearsal modes. scripts/d_stats.py: the registered analysis.
Tests scripts/d/test_d1.py: 71 of 71 pass in 231 s. air-stats-reviewer: 2 MUST FIX and 4 SHOULD FIX, all fixed (DECISIONS D1-9).
Registered: 8894501 (prereg/PREREG_D.md, scripts/d_stats.py, now read-only); hash results/prereg_hash_d.json (609a1d1).
Rehearsal with real code at toy size passed (fine-tune 2 steps, checks, walk test, 2 policies, measured, verdict; stopped after one policy and resumed); its files deleted.
optd/jobs/ladder.zsh passes zsh -n and has not run. Estimate 40-55 h. No ladder policy has been trained.
## D2 started (2026-11-17 13:24)

## D2 status: COMPLETE (2026-11-17 14:06)
Registered analysis: results/d_stats.* (already on disk; rerun to .cache byte-identical, DECISIONS D2-1). R0, R1, R2 NOT SETTLED (19 simulators).
Ladder: 6 of 7 runs, 18 saves, 13 failed; every save had lower E1 and E8 than REL, only 2 lower E100. E1/E8 ranked walking in reverse (rho -0.42, -0.43).
Secondary: walk-test tracking and u POSITIVE and better than E8; pooled R0 SUPPORTED, R2 LABEL-FREE BETTER.
scripts/d_numbers.py (called from paper_numbers.py). Paper: Sec. step2 paragraph + tab:ladder; verdict-table PT rows = Option D; A12 root cause; new A13.
Fit: slack.py PASS (6 pages, abstract 199 words) after wording cuts listed in DECISIONS D2-3. Builds identical; check_claims, check_numbers, submission_checks pass.

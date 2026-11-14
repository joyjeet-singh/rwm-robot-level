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

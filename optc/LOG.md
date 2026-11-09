# Option C log

Session status lines, oldest first. Details are in `optc/CHECKPOINT.md`, `optc/DECISIONS.md` and `optc/notes/`.

## C0 status: COMPLETE (2026-11-09 13:01)
Branch `optc` from Step 2 (tag `air2027-step2` on the start commit; `air2027-optc-start` added).
D0: 10,000 rows, 0 with a fall, 1 with thigh contact.
Facts in `optc/notes/C0_facts.md`; paper quotes came through a summariser and are unconfirmed.
Tests passed (about 3 min): D4 full, D1 small, D2 on 1,000 points, D3 route (5 networks, uncertainty 2.67), evaluator route, report on a fixture.
Job `optc/jobs/diag.zsh` written, `zsh -n` passes. It has not run; no D1-D3 result exists yet.

## C1 status: COMPLETE (2026-11-09 16:02)
Root-cause note `optc/notes/C1_rootcause.md`: the penalty alone did not fix it (D3: seed 0 stayed up but did not walk; seed 1 fell in all 80 episodes).
Generator, trainer, checks and tests in `scripts/c/`. Tests passed, on small data only; the full job has not run.
Generator equals Step 2's evaluator (difference 0). First 20 losses equal the rebuild's (difference 0). An interrupted run resumes to identical weights.
Probe `results/c/probe.json`: S = 2,572; 1,249 and 1,234 transitions/s per worker; 11.2 s (N = 8) and 6.1 s (N = 1) per step with two at once.
Layout: one training at a time, 4 threads, from a memory bound that is conservative (6.09 GB against 3.56 GB measured on the 1M stand-in; `optc/DECISIONS.md` C1 item 14).
Batch size is 256, the rebuild's; C0's note of 1,024 was the lite config's value (C1 item 1).
Job `optc/jobs/data.zsh` written, `zsh -n` passes. It has not run.

## C2 status: COMPLETE (2026-11-10 01:07)
Gate A passed: on the expert set the 6M simulator's E8 and E24 are below holding the last state (`results/c/gate_a.json`).
Authors' policy measured with the full protocol (`results/c_authors.json`, Gate B's reference); standing-still reference computed from the same schedules.
Tests passed (`scripts/c/test_c2.py`): evaluator equals Step 2's loop on normal episodes; standing pose scores near the reference; guard refuses main-pool policies; start-state rule; wrapper predictions equal the trainer's on 100 windows of the 6M rung (difference 0, limit 1e-6); one iteration with nonzero uncertainty; pilot rule on fixtures, every branch.
No pilot policy has been trained or measured. Job `optc/jobs/pilot.zsh` written; `zsh -n` passes; it has not run.

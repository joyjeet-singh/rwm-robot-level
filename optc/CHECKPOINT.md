# Option C checkpoint

**Last session:** C2, COMPLETE.

**Next action:** the launcher runs the job `optc/jobs/pilot.zsh`, then starts C3. Sessions must not start the job.

**Blockers:** none.

**State**
- Gate A passed (`results/c/gate_a.json`).
- `scripts/c/evaluate.py` (walking score), `train_policy.py` (wrapper of Step 2's route), `pilot_rule.py` (Gate B, retry, iteration rule, Gate C), `gate_a.py`, `test_c2.py`. All C2 tests passed.
- `results/c_authors.json` is Gate B's reference, measured with the full protocol.
- No pilot policy has been trained or measured. The longest training tested is one iteration with 64 robots. The job's zsh flow has been syntax-checked only; its Python steps were tested separately, and `pilot_rule.py` on fixtures.

**For the job**
- D3's 17 s per iteration (five networks, 4,096 robots, 2 threads) implies about 9.5 h for the two pilot policies, and as much again if the retry runs (about 19.5 h with measuring; the plan said 8–17 h). This is an estimate from D3, not measured on this route.

**For later sessions**
- **C3:** `results/c_pilot.txt` holds the verdicts. If Gate C is not passed, add the 2k rung and record it before registering. Results of the pilot are declared in the registration as known.
- **C3:** `scripts/c/train_policy.py` accepts only 2,000 iterations for non-test names, and `scripts/c/evaluate.py` only the four saved counts. The main job needs the iteration count from `results/c_pilot.json`: extend the wrapper or take that save from a 2,000-iteration run, and record which.
- `results/c_diag.*`, `results/c/d3_*` and `results/c/progress.json` remain uncommitted, as C1 left them. Commit the diagnosis results with the session that uses them.
- Descriptive, not a gate: on the expert set the 6M simulator's kinematic residual r̄J is 9.1, against 0.59 for the real data (`results/c/pilot_sims.json`). Its forecasts beat holding the last state, but its joint motion is much less consistent with the joint speeds it predicts. Keep this in view when interpreting the pilot.
- The tools refuse `mkdir` under `results/`; create result folders from scripts with `os.makedirs`.
- The RWM-U and RWM quotes in `optc/notes/C0_facts.md` are unconfirmed; check them against the papers before citing.

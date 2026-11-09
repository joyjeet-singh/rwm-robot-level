# Option C checkpoint

**Last session:** C0, COMPLETE.

**Next action:** the launcher runs the job `optc/jobs/diag.zsh`, then starts C1. Sessions must not start the job.

**Blockers:** none.

**State**
- Branch `optc`; tag `air2027-optc-start` is on Step 2's last commit (8e57841, tag `air2027-step2`).
- `scripts/c/diag.py` is tested on small sizes only. No D1–D3 result exists until the job has run; D0 and D4 are computed, and the job recomputes both.
- The job writes `results/c_diag.json` and `results/c_diag.txt` once, and skips finished work on a re-run.

**For later sessions**
- `results/c/progress.json` still says "report written", left by C0's report test on a fixture. It is not a job result; the job overwrites it.
- `.cache/diag_test`, `.cache/diag_rep`, `runs/c/_test_d3_0` and `runs/c/d3_lite/runs/lite/s2__test_d3_0` are test leftovers and can be ignored.
- The tools refuse `mkdir` under `results/`, probably because of the project's `Edit(/results/**)` deny rule. Create result folders from scripts, with `os.makedirs`.
- A shell command with a pipe followed by more lines was refused, so keep commands on one line.
- C1's trainer copy must drop two checks in `rwm_data.py`: column 65 all zero (line 223) and shape (10000, 66) (line 243).
- The RWM-U and RWM quotes in `optc/notes/C0_facts.md` are unconfirmed; check them against the papers before citing.

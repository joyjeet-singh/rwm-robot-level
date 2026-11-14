# Option D checkpoint

**Last session:** D1, COMPLETE (see the status line in `optd/LOG.md`).

**Next action:** the launcher runs the ladder job `optd/jobs/ladder.zsh` (about 40–55 hours; resumable), then starts D2. Sessions must not start the job.

**Blockers:** none.

**State**
- Registered at commit 8894501: `prereg/PREREG_D.md` and `scripts/d_stats.py` (read-only, `chmod a-w`); hash `results/prereg_hash_d.json`.
- Code in `scripts/d/` added by D1:
  - `train_policy.py`: Step 2's recipe in a ladder simulator or one of Step 2's nine, the stop rule, the failed-by-simulator rule → `runs/d/policies/<p>/actor.pt`, `results/d/train/<p>.json`;
  - `ladder.py`: `run` (the driver, two workers; state in `results/d/ladder/`, one line per policy in `results/d_ladder.txt`), `fill`, `final` (→ `results/d_ladder.json` and the final table appended to `results/d_ladder.txt`);
  - `stage1.py` → `results/d/stage1.json`; `test_d1.py` (71 tests, `.cache/d_test1/results.json`).
- The job exits 3 at the sanity gate (the driver writes `optd/DECISIONS.md` and `optd/LOG.md`), 1 on any other failure. Logs: `.cache/logs/job_d_ladder.log`, `ladder_*.log`.

**For D2**
- `./py scripts/d_stats.py` writes `results/d_stats.json` and `.txt` once `results/d_ladder.json` exists (about 4–5 minutes).
- Registration choices: §13 of the registration and DECISIONS D1-1 to D1-9.

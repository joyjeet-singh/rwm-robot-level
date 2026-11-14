# Option D checkpoint

**Last session:** D0, COMPLETE (see the status line in `optd/LOG.md`).

**Next action:** the launcher runs the record job `optd/jobs/record.zsh` (about 1–2 hours), then starts D1. Sessions must not start the job.

**Blockers:** none.

**State**
- Branch `optd`, tag `air2027-optd-start` (on Option C's last commit).
- Code in `scripts/d/`:
  - `sims.py`: the 17 simulators and the ladder;
  - `checks.py`: common units, both test sets → `results/d/checks/`;
  - `walk_test.py`: secondary → `results/d/walk/`;
  - `evaluate.py`: `./py-mj` → `results/d/eval/`. Its guard opens once `prereg/PREREG_D.md` is committed;
  - `record.py`: the table → `results/d_record.*`, and gates 1 and 2;
  - `finetune.py`: the seven runs, saves at the design's step counts → `runs/d/<run>_d<steps>/final.pt`, resumable.
- Tests: `scripts/d/test_d0.py`, all passing (`.cache/d_test/results.json`).

**For D1**
- The stop rule must say what a missing `Train/mean_episode_length` does (about 2% of iterations have none). `record.py`'s `stop_rule_at` treats it as breaking the run (DECISIONS D0-3).
- `evaluate.py --policy <name> --actor <file>` measures a ladder policy after registration; for a stopped policy, `iterations` is read from the actor file.
- `finetune.advance(run, to)` is resumable. `status(run)` gives `nonfinite_at`. Fine-tuning takes 5.0 s per step.
- `sims.load("<run>_d<steps>")` loads ladder simulators. `checks.run(name)` and `walk_test.run(name)` work on them unchanged.
- The registration must declare as known: Step 2's results, Option C's records, `optd/REVIEW.md`, and `results/d_record.*`.

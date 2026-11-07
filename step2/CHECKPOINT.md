# Checkpoint
Last session: M4, COMPLETE (2026-11-08). Gate 2: P2 failed -> fallback pool registered (step2/notes/M4.md).
Next: the launcher runs job main (step2/jobs/main.zsh): M4 tests, then scripts/s2_main.py trains and measures 18 policies (REL and its eight compressed copies x seeds 1, 2), 2 workers x 2 threads, about 20 h; writes results/s2_main.json. A re-run resumes. Then M5 runs ./py scripts/s2_stats.py (registered, read-only) once.
Blockers: none. If the job exits 1, a policy failed twice for a reason other than non-finite values: see .cache/s2_failed/ and .cache/logs/s2_main_*.log.

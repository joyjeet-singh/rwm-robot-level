# Checkpoint
Last session: S3, COMPLETE (2026-11-06 17:33). Registration commit 2cb7d24 (prereg/PREREG_S1.md; hash 924a57a); job committed 7c753fc.
Next: the launcher runs job S3 (step1/jobs/S3.zsh → ./py scripts/s1_eval.py; about 11 h; progress in results/s1/progress.json; driver log .cache/logs/job_S3_driver_*.log). Sessions S4 and S5 work alongside it with ≤2 threads and ≤10 min computations. S6 runs ./py scripts/s1_stats.py after the job. It must stop BLOCKED if S1-C1, S1-C2 or S1-C3 failed (see results/s1_models.json "checks").
Never edit: prereg/PREREG_S1.md, scripts/s1_stats.py. Changes go to ANOMALIES.md.
Blockers: none.

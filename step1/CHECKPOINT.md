# Checkpoint
Last session: S4, COMPLETE (2026-11-06 18:05). Lite policy training works on this CPU (scripts/lite/); notes in step1/notes/S4_lite.md; DECISIONS S4-1…3.
Next: S5 (alongside job S3: ≤2 threads, ≤10 min computations). Job S3 (step1/jobs/S3.zsh, about 11 h; results/s1/progress.json belongs to the job). S6 runs ./py scripts/s1_stats.py after the job and must stop BLOCKED if S1-C1, S1-C2 or S1-C3 failed (results/s1_models.json "checks").
Never edit: prereg/PREREG_S1.md, scripts/s1_stats.py.
Blockers: none.
For Step 2: lite runs need --holdout per model fold and 5-head rebuilds; 500 iterations take 0.4-1.9 h on this CPU at 1,024-4,096 robots.

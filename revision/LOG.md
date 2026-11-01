# Revision log

## R0 status: COMPLETE (2026-11-01 19:09)
- Registration protected; env recorded (revision/notes/R0_env.md); git restored, tag air-pre-revision, branch revision-1.
- Smoke test: all printed numbers identical; float32 last-digit diffs in stats/check_numbers recorded; pages 10 and 6.
- poppler absent: pypdf fallback in scripts/pdf_tools.py (new), submission_checks.py, smoke_compare.py (new).
- Commits: dc2fccb (setup); environment commit follows.

## R1 status: COMPLETE (2026-11-01 19:35)
- scripts/revision_analyses.py → results/revision_exploratory.{json,txt}; 178 s, 20,000 two-level draws; all asserts pass.
- Registered Q1–Q6 and Table 2 rhos reproduce exactly; λ1 from log1 matches only to 2.2e-16 (DECISIONS R1).
- Review (air-stats-reviewer): 0 MUST FIX; 5 SHOULD FIX all fixed; 2 NOTES recorded in revision/notes/R1.md.
- Headlines: λ1(8) does not rank E100; r̄J(8) ≈ E8. Two-level: Q3_vs_E1, Q5_vs_E1, Q1_E24 would no longer exclude 0. Holm: only Q3_vs_E8 and Q4_vs_E1 survive.
- Commits: 8786f28 (work); log/checkpoint commit follows.
- OPEN: ./py loads numpy 2.2.6 from .pylib, so torch↔numpy is broken; R3 training will fail until fixed (DECISIONS R1).

## R2 status: COMPLETE (2026-11-01 20:00)
- scripts/revision_recheck.py (independent; numpy/json/spearmanr) → results/revision_recheck.json: all_agree true, 151 point estimates (A1, A3, A2b), max diff 1.1e-16.
- One first-pass mismatch was the recheck's own E368 definition (mean over 368 steps); fixed. R1 needs no change.
- Notes: revision/notes/R2.md. Open: numpy/.pylib issue from R1 still stands for R3.

## R3 status: COMPLETE (2026-11-01 22:10)
- Study FT registered before any fine-tuned model existed: commit 7fc5524 (PREREG_FT.md, ft_unstructured/ft_eval/ft_stats/test_ft.py, ft_probe.json); PREREG_FT.md and ft_stats.py made read-only.
- Tests 7/7 pass (54 s); D2 u30 reproduced exactly; C-FT4 on one parent 1.3e-7/2.0e-7.
- Probe: 3.95 h projected ≤ 8 h, so U50F runs (arms P, U30, U30F, CTRL, U50, U50F). All 15 PerconAI controls exist.
- air-stats-reviewer: 1 MUST FIX + 7 SHOULD FIX fixed before registration (DECISIONS.md R3).
- Job: revision/jobs/R3.zsh (30 runs, then ft_eval.py; ~4 h). After it: ./py scripts/ft_stats.py.
- Env: .pylib numpy 2.2.6 moved to .cache/pylib_removed (earlier R3 attempt); R0 smoke not rerun. figures/*.png show as modified but were not changed by R3 and were left unstaged.

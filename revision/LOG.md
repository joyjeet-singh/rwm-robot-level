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

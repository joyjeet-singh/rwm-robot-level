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

## R4 status: COMPLETE (2026-11-01 22:08)
- 8 new references verified by air-bib-checker (two briefs); paper/bib_verified_air.json written (9 entries incl. RWM venue).
- paper/refs.bib: 7 new keys + mikhaeil2022chaotic; li2025rwm note now workshop poster.
- scripts/check_bib.py: fourth source, --allow-uncited, scans body.template/body/appendix; passes with --allow-uncited.
- Notes: revision/notes/R4.md. Open: quotes are fragments; RAMBO refs [75,76] unconfirmed; Manek/Kolter order (DECISIONS R4).

## R5 status: COMPLETE (2026-11-01 23:00)
- scripts/revision_numbers.py (called from paper_numbers.py after air_numbers): 277 new keys, 790 in all; paper_numbers.py runs.
- Registered: A1, A2 (two-level, X5 without {1,8}, Holm), A3, unstable-copy totals, scale ratio ("about 750"), stale command on the held-out pair at h=100, training settings (equal across 25 parents), pre-registration priors, PerconAI unit-removal K/Q3/Q4.
- Spot checks (printed both sides): a1 X5 λ1(8) −0.02, a2 Q3_vs_E1 two-level CI, perc K width-128, stale 22.49, NGCAPS ρ [0.70, 0.93], Holm p for Q4_vs_E8; all match their sources.
- Key list for R6: revision/notes/R5_keys.md (128 lines, generated from paper_numbers.json). Commit 44235e1. Choices in DECISIONS.md (R5).
- Open: A1/A2/A3 keys are exploratory and must be labelled so; scale_ratio is 750 (from 0.133%), not 770.

## R6 status: COMPLETE (2026-11-01 22:46)
- body.template.tex revised for T1–T22; title kept, two alternatives in revision/notes/R6.md (T23).
- Two-column 6 pages (page 6 full, no spare line), single column 10; abstract 199 words; check_bib (strict), submission_checks, check_claims pass.
- New: paper/appendix.template.tex (A1–A7 skeleton), revision/notes/R6.md, R6_moved.md (old Sec. 4.3 for A6).
- scripts/air_numbers.py fk() fix (K <= −100% printed as 100%; only unused keys changed).
- Commit c0f7c93. Choices in DECISIONS.md (R6): "barely changing", T20 at 368 steps, COMBO citation unverified.
- Open: any added text (R7 appendix ref, R9 FT sentence in abstract) needs matching cuts; the two-column build has no slack.

## R7 status: COMPLETE (2026-11-01 22:52)
- Figure 2 has three panels (Spearman from stats.json); caption in the body names them.
- paper/appendix.template.tex: A1–A6 filled, A7 left for R9; new keys (hp_*, a1t_*) in revision_numbers.py (839 keys).
- build_paper.py fills appendix.tex (refuses on a missing key); main_single inputs it after the references.
- Single column 15 pages, two-column 6 pages; submission_checks, check_claims (46), check_bib pass.
- Commit 8227323. Notes: revision/notes/R7.md; choices in DECISIONS (R7).
- Open: not placed in the appendix: shared_pct, cov1_*, ng_pathB_*, TorchScript timing. fig_horizon.* left unstaged.

## R8 status: COMPLETE (2026-11-02 12:00)
- check_claims.py: 67 assertions (46 before); list in revision/notes/R8_claims.md. All five checks pass; two-column 6 pages, abstract 200 words.
- air-claim-checker run; fixes (Q5 labels, pathB sentence, exploratory/descriptive labels, narrower wording) in revision/notes/R8.md.
- Open for R10: items in R8.md "Not changed". A7 still empty for R9.

## R9 status: COMPLETE (2026-11-02 03:10)
- Job R3 complete: 30 runs (U30F, U50F), 90 models scored; C-FT1–C-FT4 pass for all; no exclusion or anomaly.
- ft_stats.py (unedited): FT1 FINE-TUNING RESTORES SKILL (+63.9 pts [54.4, 76.0]); FT2 NEARLY AS SKILLED (99.9% [99.5, 100.4]); secondary agree; one-step reading not settled.
- revision_numbers.study_ft(): ft_* keys (1017 total). Text: abstract clause, Sec. 4.1 two sentences, Table 1 FT row, Discussion (2), Limitations, Conclusion, appendix A7 (two tables).
- Cuts for 6 pp (T22 order): Sec. 4.3 folded into Discussion (5); synthetic-data sentence; minor wording. Listed in DECISIONS (R9).
- check_claims 78 (11 new), check_bib strict, submission_checks pass; single 16 pp, two-column 6 pp, abstract 198 words.
- Commit e9b295c. Notes: revision/notes/R9.md. figures/fig_horizon.* still unstaged (not mine).

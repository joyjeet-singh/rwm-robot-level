# Revision round 1 — report for the author

All checks pass. The final pipeline ran three times and gave identical PDFs: submission 16 pages, two-column check 6 pages
(no slack), abstract 198 words. The sha256 hashes are in `revision/LOG.md`.

## What changed, by review point
Full table: `revision/notes/R10_traceability.md` (ids P1–P7j).
- **P1, P1b** (prior work): the introduction credits RWM's own ablation and aims the critique at held-out one-step model selection.
- **P2a, P2b** (framing, what was known beforehand): the abstract leads with training choices, pruning and uncertainty; the failed five-split replication is stated; appendix A2 says what was known when.
- **P3a–P3e** (overreach): the residual's lead fails correction for multiple tests; pruning claims limited to 10–30%; the released model, label-free and stale-command claims narrowed.
- **P4a–P4d** (confounds): horizon-matched and split-half checks, replication without the shared split (A3, A4).
- **P4c, P5** (scope): simulated data, recorded commands, one gait, no policy trained, stated in abstract, discussion and limitations.
- **P6a, P6b** (statistics): two-level intervals and Holm correction, both exploratory.
- **P7a–P7j** (presentation): tables, definitions, figure 2, references, the AI statement. Appendix A1–A7 added.

## New results
- **Exploratory** (labelled so in the paper; `results/revision_exploratory.txt`): the growth rate tracks 100-step error well among trained models and less among compressed copies; the kinematic residual ranks well in both. Wider two-level intervals are about 1.5× the registered ones, and 3 of 13 verdicts lose their margin. Registered verdicts are unchanged.
- **Study FT** (registered before any model existed; 30 runs, 90 models):
  - FT1: FINE-TUNING RESTORES SKILL (+63.9 points [54.4, 76.0]).
  - FT2: NEARLY AS SKILLED (99.9% [99.5, 100.4]).
  - The secondary tests agree. The one-step reading is not settled.
  - Scope: one robot, one gait, simulated data, no policy.

## Referee and claim checker
The fresh-eyes referee scored the paper **borderline** (major revision). The claim checker found 12 points, all fixed or answered.

**Declined** (reasons in `revision/DECISIONS.md`, "R10 declined"):
- New calibration analysis, intervals on Table 1 ratios, and a constant-velocity curve.
- Six new references, which need bib-checking and page room. I recommend adding Lohmiller & Slotine and Hwangbo et al. if you can.
- A title change (alternatives are in the R6 notes).
- Restructuring, a glossary and a public registry. The registration commits are in the supplement.

## Supplement
`submission/AIR2027_supplement.zip` (about 33 MB, with `raw_*.npz`) is built locally and scanned: no name, handle, place or account. Two local paths and the name patterns of `submission_checks.py` were neutralised in the copy. `PREREG.md` is byte-identical.

## Only you can do
1. Read the paper end to end, and rewrite the introduction and discussion in your own voice (a two-column page is the limit: one added word makes seven).
2. Check AIR 2027's rules on AI-assisted writing, and edit the AI-use statement.
3. Withdraw the overlapping submissions (TMLR, PeRConAI, ICMAI-MAC, NGCAPS, ICPAIAS) before uploading.
4. Decide whether to upload the supplement.
5. Submit through CMT between 15 Nov and 31 Dec 2026.
6. `figures/fig_horizon.*` differ from the last commit; they were not touched by this round. Decide whether to keep them.

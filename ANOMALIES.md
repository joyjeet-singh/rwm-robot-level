# Anomalies

Problems found after `prereg/PREREG.md` was committed, in the order found. None changes a registered question,
pool, measure or decision rule.

None. All pre-registered checks passed (`results/checks.json`), no model produced a non-finite value, and the
analysis ran exactly as registered. Analyses added afterwards are in `scripts/exploratory.py` and are labelled
exploratory in the paper.

## Environment (revision round 1)

2026-11-01, R0. The stored results were computed in a Linux workspace (Python 3.13, torch 2.15, numpy 2.5); the
smoke test reran the scripts on a copy on this Mac (Python 3.11.15, torch 2.2.2, numpy 1.26.4, scipy 1.13.1).
- **Effect on results:** none on any number the paper prints (`paper_numbers.json`: every value identical). Stored
  float32 bootstrap bounds differ in the last digits: `stats.json` (13 values, at most 5.3e-7 relative) and
  `check_numbers.json` (2 values, 1.9e-8 relative). Both are above the 1e-9 tolerance but far below the printed precision.
- **Tools:** poppler is not installed. `brew install poppler` was started and stopped after about 40 minutes, because
  it was compiling Python 3.15 and other dependencies from source. `scripts/pdf_tools.py` (new) uses pypdf, installed
  in `.pylib`, when the poppler tools are absent; `scripts/submission_checks.py` now calls it. With poppler present
  the behaviour is unchanged. pypdf puts word gaps differently from pdftotext, so PDF text is compared ignoring spaces.
- **Builds:** page counts equal the stored ones (10 single column, 6 two-column); `results/build.json` identical.
  PDF text is identical apart from word spacing; the PDF files differ in size (different TeX Live build).
- **Working tree:** `figures/fig_checks.png` and `figures/fig_horizon.png` already differed from the last commit
  before R0 ran anything; left as they are.

## Revision round 1 (R0-R11)

None of these touches a registered verdict.
- **Study FT:** C-FT1-C-FT4 passed for every model; no exclusion and no anomaly (R9).
- **Exploratory analyses (R1-R2):** written after the registered results; labelled exploratory everywhere. Registered Q1-Q6 and Table 2 reproduce exactly.
- **Two-column length:** exactly six pages with no slack; text was cut in R9 and R10 to keep it (DECISIONS).
- **Declined review points** and their reasons: DECISIONS "R10 declined" (new analyses, new references needing bib checks, title change).
- **Working tree:** `figures/fig_horizon.pdf` and `.png` differ from the last commit; they were already different before the round and were left alone.
- **Supplement (R11):** in the zip copy only, a local path (`stage_out.py`, `common.py`) and the author-name patterns of `submission_checks.py` were replaced by placeholders. `PREREG.md` is kept byte-identical (its hash is evidence); it names a generic download folder, with no name.
- **Registration time zone:** git commit times of the FT registration carry a +05:30 offset; the supplement README gives UTC.

## Step 1 (S0-S8)

None of these departs from `prereg/PREREG_S1.md` or touches a registered verdict (CL1, CL2, AR1, AR2). Details in
`step1/DECISIONS.md` and `step1/notes/`.
- **Check S1-C1 (2026-11-07, S6).** Recomputed on this Mac, 3 of F18's 97 kinematic residuals (the 10,000-iteration
  teacher-forced models B10k_s0-s2) differ from the stored Linux values by up to 0.51% relative, outside the 1e-3
  per-model tolerance. The registered rule (at least 95% within tolerance, rank correlation at least 0.999) passes:
  96.9% within, ρ = 0.99999. **Effect:** none; CL uses the stored recorded-command values, as registered. The paper
  reports the three models and the largest difference (appendix A8).
- **AR runs that diverged under recorded commands (S6).** Three teacher-forced MLP runs (two earlier ones, seeds 0
  and 2, and the new seed 4) left the ±1e3 bound on 2 of their 4 trajectories. Handled as registered (S2-4, S3-2):
  values kept, non-finite ranked worst, count reported; no run dropped.
- **Imagined return (registered, S1-5).** It reproduces the authors' `dof_acc_l2` term as written, which reads joint
  positions where joint velocities are meant (`anymal_d_flat.py:105`). Only CL6 uses the return.
- **Lite code (S4, Task LP; not in the paper).** Runs with two compatibility shims and no install (`torch.load` to
  CPU; `TensorDictBase.__contains__` for tensordict 0.3.2) and a wandb stub. Timings were taken while job S3 used the
  other two cores; the 500-iteration hours are projections, not measured runs.
- **Supplement size (S7).** The full zip is 69.6 MB (round 1: 32.7 MB) because of `raw_S1.npz`; a 1.4 MB slim copy
  without raw arrays was also built. The author picks one against the portal's limit.
- **Body pointers (S6-S8).** The two-column check has no slack, so the body names only appendix A8; A9 and A10 have no
  body pointer, and the Limitations, abstract and introduction were left unchanged (DECISIONS S6-2 to S6-4).
- **Working tree.** `figures/fig_horizon.pdf` and `.png` still differ from the last commit, as before Step 1; the PDFs
  include the working copy, which looks identical. Never staged in Step 1.

## Step 2 (M0-M6)

None of these changes `prereg/PREREG_S2.md` or a registered verdict (R0, R1, R2 all NOT SETTLED). Details in `step2/DECISIONS.md`.
- **ANYdrive actuator network not obtained (M0, M1).** Isaac Lab's asset root is read at run time and the file's full URL
  was never verified; the fetch job's two guesses failed (manifest: FAILED). The fallback fixed in the plan was used: a PD
  controller with a fixed delay, fitted to the recorded torques ($R^2$ in appendix A11). The torques separate the delay only coarsely.
- **Gate 2 failed on P2 (M4).** Policies trained in three rebuilt single-network simulators fell in all 80 pilot episodes
  (the released model's: 10%). By the plan's rule the pool became the released model and its eight compressed copies (9 simulators,
  2 seeds each), not 25 main-pool simulators. So the physics test says nothing about simulators trained in other ways. Why
  the rebuilt simulators' policies fall was not studied (one difference: no spread between heads, so the uncertainty penalty is inactive).
- **Checks for the copies were computed before registration (M4).** They had never been built in earlier rounds; they were
  computed with the registered functions only, before any pool policy existed, in-sample (the released model saw all ten episodes).
- **The pilot's registered-pool overlap (M4).** The first prereg text wrongly said no pool-simulator policy existed (the pilot's
  released-model policy does); the stats reviewer caught it and it was fixed before the commit. That policy is validation, not a result.
- **Model details a reader might doubt (M1, M2).** Contact time constants below 0.01 s are raised to 0.01 s by MuJoCo, so two
  grid points were identical (tie broken towards the larger value by a rule fixed before the grid ran). Collision shapes are
  simplified (no drive housings, no self-collision). The velocity commands used for Gate 1 are segment means read from the data.
- **Physics return rewards standing still (M5, descriptive).** The two policy pairs that stood still (50%-pruned and width-128
  copies, never fell) ranked third and fourth by physics return, above every walking policy that fell. The best policies
  still fell in 31 to 45% of episodes. So a null result here may reflect the outcome and the pilot-level policy quality, not only the checks.
- **Table placement (M5).** The R0 to R2 rows are a block of Table III, not a separate table (a separate one pushed the
  two-column check to seven pages). The Step 2 slot became a run-in paragraph; the Limitations lost two clauses for space.
- **Supplement (M6).** The ANYdrive file is not packed. The MuJoCo version is recorded only in the uncommitted `step2/assets/manifest.json`
  (shipped inside the supplement as `step2/assets_manifest.json`), not printed in the paper.


## Option D (D0-D3)

None of these changes `prereg/PREREG_D.md` or a registered verdict (R0, R1, R2 all NOT SETTLED). Details in `optd/DECISIONS.md`.
- **The registered analysis file existed before D2 (D2-1).** `results/d_stats.json` was on disk before any D2 session ran (three earlier D2
  launches failed at sign-in); no launcher names `d_stats`, so what wrote it is not recorded. Never overwritten: a rerun of the unedited
  `scripts/d_stats.py` into `.cache/` is byte-identical.
- **Option C's from-scratch simulators failed, through two design mistakes of ours.** State scaling (each state divided by its own
  data's spread, not the authors' fixed scales) and training amount (about 0.1 of one pass over the data). They were not repaired; the next
  design adapted the released model. Why the rebuilt single-network simulators' policies fall was diagnosed later (descriptive, A12).
- **Three of the ladder's saves had policies failed by their simulator (D1-5).** Losses not finite, or the same crash twice, all in mixed-data
  saves; they score 0 and count as failed.
- **Error checks ranked walking in reverse.** Every tested save had lower one-step and 8-step error than the released model, yet most of
  them failed. The R1 and R2 point estimates favour the label-free checks only because of that reversal; both intervals include 0.
- **The mixed-data repeat did not start (design rule).** The mixed run at 1e-4 found no boundary, so M_s1 was skipped; 6 of 7 runs ran.
- **Walk-test starts for Option C's four simulators (D0-4)** differ from Option C's calibration, so their walk-test values differ a little from Gate A2's.
- **r-bar-J on the paper's arena (D0-1)** is 0.13% below the value stored on the cloud and equals the Mac recomputation; the gap is floating-point.
- **Option C's five-network simulator's policies were never given a walking score (D0-8);** the record shows their fall rate only.
- **Fit (D2-3).** The new paragraph and table left no space: wording elsewhere was compressed without removing a result.
- **Supplement (D3).** One Option D result file (a crash traceback) held the working copy's absolute path; it is replaced in the staged copy
  only. `build_supplement.py` lists it among the anonymity edits in the supplement's README.

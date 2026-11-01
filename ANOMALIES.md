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

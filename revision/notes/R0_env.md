# R0 — environment and baseline (2026-11-01)

**Registration:** `prereg/PREREG.md` mtime 1793521083, 11808 bytes before and after `chmod a-w`; read-only now.

**Versions:** Python 3.11.15, torch 2.2.2, numpy 1.26.4, scipy 1.13.1 (installed in `.pylib`), matplotlib 3.8.4,
pypdf 6.21.4 (`.pylib`). Mac: Intel i5-7360U, 4 cores, 8 GiB, macOS 13.7.8, 40 GiB free.
**Tools:** pdflatex, bibtex (TeX Live 2026) in /Library/TeX/texbin; brew /usr/local/bin/brew; no pdftotext,
pdfinfo or pdffonts (poppler). Route taken: pypdf fallback (`scripts/pdf_tools.py`), because the brew build of
poppler pulled in a from-source Python 3.15 and was stopped (see DECISIONS.md).

**Git:** history restored from `history.bundle` (head feee991 on master); tag `air-pre-revision`; now on `revision-1`.
Git user is the global identity (already set, so not changed). Working files differ from the last commit only in
`figures/fig_checks.png` and `fig_horizon.png`, plus untracked files. Setup commit dc2fccb.

**Smoke test** (copy in `.cache/smoke`, all nine scripts ran, exit 0): `paper_numbers.json` identical in every value;
`checks`, `exploratory`, `build`, `claims_check`, `bib_check`, `submission_checks` identical. Last-digit float32
differences: `stats.json` 13 values (max 5.3e-7 rel), `check_numbers.json` 2 values (1.9e-8). Page counts
10 single-column and 6 two-column, equal to stored; PDF text identical apart from word spacing in extraction.
Overfull boxes: 0 and 1, as stored. R6 can use the stored six-page budget; the build here gives six.

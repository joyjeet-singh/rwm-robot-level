# AIR 2027 — working rules for this folder

The paper for **AIR 2027** (8th International Conference of The Robotics Society, IIT Gandhinagar, 30 Jun–3 Jul
2027). It draws on all of the PDM project (`../rwm_repro`, `../NGCAPS`, `../ICMAI`, `../ICPAIAS`, `../PerconAI`)
and adds one pre-registered study of its own. The earlier venue papers are being withdrawn; this is the target.

- **Everything outside `AIR 2027/` is read only.** Data, weights and earlier results are read, never written.
- **No number is typed.** `paper/body.template.tex` holds placeholders; `scripts/paper_numbers.py` and
  `scripts/air_numbers.py` read every value from a result file; `scripts/build_paper.py` refuses to build if a
  placeholder has no value. Edit the template, never `paper/body.tex`.
- **`prereg/PREREG.md` is never edited.** It was committed and copied to the Mac before any model was run through
  this folder's code. Problems found later go to `ANOMALIES.md`; analyses added after the results are labelled
  descriptive or exploratory in the paper.
- **Double-blind.** No name, handle, place, account or repository link in the paper or its metadata;
  `scripts/submission_checks.py` scans both PDFs.
- **Length.** The submission is single column with no page limit; the camera-ready must fit six two-column pages,
  which `paper/AIR2027_twocolumn_check.pdf` checks.
- Plain language; define every unavoidable term. State what failed as plainly as what worked.
- No external actions (no submission, upload, push or e-mail) without the author.

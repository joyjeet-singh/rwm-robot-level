# Submitting to AIR 2027 — what only the author can do

**Submission window:** papers are accepted from 15 Nov 2026 to **31 Dec 2026** (Microsoft CMT; the portal link is on
advancesinrobotics.com/2027-IITGN). Notification 28 Feb 2027; camera-ready and registration 30 May 2027.

**Which PDF:** upload `paper/AIR2027_submission.pdf` (single column, anonymous; the call allows any length at this
stage). The appendix (A1-A12) is in this PDF only; cut it for the camera-ready, which must fit six two-column pages.
`paper/AIR2027_twocolumn_check.pdf` is for you only: it shows the main text fits six two-column pages.
The official template is e-mailed to accepted authors; the text will need re-flowing into it.

**Step 2 is in the paper** (Sec. 4.2 "Walking in a physics simulator", a block of Table III, appendices A11 and A12). There
is no blank slot any more; `scripts/slack.py` shows the two-column text fits six pages with no spare room (reserve 0), so any
further addition to the main text needs a cut elsewhere. The result is registered and negative: no check was shown to
predict how the policies walk (R0 to R2 NOT SETTLED). The physics is a model of ANYmal D built by hand for this paper, not the
data's simulator and not hardware; the paper says so wherever the result is stated.

## Before you submit

1. **Withdraw the overlapping submissions first.** This paper reuses results from the TMLR submission
   (`../rwm_repro`) and from the PeRConAI, ICMAI-MAC, NGCAPS and ICPAIAS papers. Submitting the same results to
   two venues at once is dual submission at almost every conference; withdraw (or confirm withdrawn) each of
   them before uploading here, and keep the withdrawal e-mails.
2. **Double-blind.** The PDF carries no name, place, account or link (checked by `scripts/submission_checks.py`,
   result in `results/submission_checks.json`). But the project is public under your name
   (GitHub `rwm` repository, Hugging Face checkpoints), and the talk proposal in `../SciPyIndia` describes it; a
   reviewer searching for the topic could find it. That does not break the rules as written ("author names and identities must not appear
   anywhere in the submitted paper"), but do not link, mention or upload those pages with the submission, and
   avoid publicising this exact title until decisions are out.
3. **Supplementary material (optional; your decision).** Two zips are built and scanned for identifying text by
   `scripts/build_supplement.py` (Step 1, S7; extended in Step 2, M6); neither has ever been uploaded. `submission/AIR2027_supplement.zip`
   (71.0 MB, 345 files) holds round 1's contents plus Step 1's and Step 2's registrations (`PREREG_S1.md`, `PREREG_S2.md`), scripts, lite code, the MuJoCo model and
   result files, including `raw_S1.npz`; `submission/AIR2027_supplement_slim.zip` (2.0 MB) leaves out every
   `raw_*.npz`, so the statistics cannot be rerun from it. **Check CMT's size limit** and pick one.
   `README_SUPPLEMENT.md` inside says what it holds and gives the registration times in UTC. Upload only if CMT offers
   a slot and you choose to; the paper's "will be released" sentences do not depend on it.
   Re-scan it for your name first (`unzip` into an empty folder, then `grep -ri`); never include the git history bundle.
4. **AI-use statement.** The paper ends with a short declaration that an AI assistant wrote code, ran analyses
   and drafted text under your direction. **You must edit it** to describe your own role accurately, and check AIR 2027's rules on AI-assisted writing.
5. **Read the paper once end to end.** Every number is generated from a result file, every reference was traced
   to a record an earlier study checked, and an independent re-computation agrees; but you are the author.
   Re-open each reference link listed in the earlier studies' `bib_verified.json` files.
6. **Keywords and topic.** Suggested CFP themes: "Field Robots: Legged, Flying, and Underwater" and "AI/ML and
   Ethical Implications in Robotics".

## Title, abstract and keywords for the form

`paper/ABSTRACT.txt` has the title, abstract and keywords as plain text, generated from the paper itself.

## After the improvement round (I5)

- **Window:** 15 Nov to 31 Dec 2026 (see the top of this file). The Step 2 slot still fits six pages (`slack.py`: free space 0.30).
- **Abstract text for the form:** copy it from `paper/ABSTRACT.txt` (185 words, limit 200); do not retype it.
- **Step 2:** `STEP2_PLAN.md` describes the robot-level test (policies trained inside scored simulators, tested in a physics simulator).
- **Before submitting:** put the paragraphs flagged in `improve/REPORT.md` into your own words; check the AI-use statement
  (it does not mention the AI-run reviews and reference checks); check the corrected facts against RWM, RWM-U and iKCE;
  check the supplement's version-control history for names (`improve/notes/I4.md`).

## After Step 2 (M6)

- **Step 2 checks:** `paper_numbers.py`, `build_paper.py` (run twice: same text), `slack.py`, `check_claims.py` (181 assertions),
  `check_numbers.py`, `check_bib.py` and `submission_checks.py` all pass. An independent claim check of the Step 2 text against
  `results/s2_*` and `prereg/PREREG_S2.md` is in `step2/review/` (findings fixed).
- **Supplement:** rebuilt with Step 2 (`prereg/PREREG_S2.md`, `scripts/s2/`, the MuJoCo model files with the robot description's
  BSD 3-Clause notice, the `s2` results). Scan: 0 hits. The ANYdrive actuator file is not in it and was never obtained, so there is
  no URL or sha256 to give; the README inside says so. Check the new size against CMT's limit; the slim copy is the safer choice.
- **Look at before submitting:** (1) the new paragraph and appendices A11 and A12 in your own words; (2) the "Limitations"
  sentence on what the physics test covers; (3) that withdrawing the earlier venue papers is done (step 1 above).
- **The main job** ran about 21 hours (`step2/LOG.md`); nothing was pushed, uploaded or sent.

## After Option D (D3)

- **Checks:** `paper_numbers.py`, `build_paper.py` (run twice: same text), `slack.py` (free space 0), `check_claims.py` (210 assertions),
  `check_numbers.py`, `check_bib.py` and `submission_checks.py` pass. An independent claim check of the Option D text against
  `results/d_*`, `results/c_*` and `prereg/PREREG_D.md` is in `optd/review/` (findings fixed or listed in `optd/notes/D3.md`).
- **Supplement:** rebuilt with Options C and D (`prereg/PREREG_D.md`, `scripts/c/`, `scripts/d/`, the `c` and `d` results, Option C's data
  manifest). The data, simulators and policies are not in it. Scan: 0 hits. Full zip 73.4 MB (limit kept at 80 MB), slim 4.4 MB: check the
  size against CMT's limit; the slim copy is the safer choice.
- **Look at before submitting:** (1) the new Sec. 4.2 paragraph, Table "ladder" and appendix A13 in your own words; (2) the main-text
  sentence that the R0-R2 verdict-table rows now give Option D's tests (Step 2's are in A12); (3) that the two-column check is exactly
  six pages with no spare room (any added word needs a cut); (4) that withdrawing the earlier venue papers is done (step 1 above).
- **The ladder job** ran about 53 hours; nothing was pushed, uploaded or sent.

# Submitting to AIR 2027 — what only the author can do

**Deadline:** papers are accepted from 15 Nov 2026 to **31 Dec 2026** (Microsoft CMT; the portal link is on
advancesinrobotics.com/2027-IITGN). Notification 28 Feb 2027; camera-ready and registration 30 May 2027.

**Upload:** `paper/AIR2027_submission.pdf` (single column, anonymous; the call allows any length at this stage).
`paper/AIR2027_twocolumn_check.pdf` is for you only: it shows the same text fits six two-column pages, which the
camera-ready must. The official template is e-mailed to accepted authors; the text will need re-flowing into it.

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
3. **Supplementary material.** None is required. If CMT offers an upload, an anonymised zip of `prereg/`,
   `scripts/` and `results/` (without `raw_*.npz` if size-limited) supports the "will be released" sentence.
   Check it for your name first (`grep -ri joyjeet`), including inside the git history bundle, which must not be
   included.
4. **AI-use statement.** The paper ends with a short declaration that an AI assistant wrote code, ran analyses
   and drafted text under your direction. Keep it, and edit it to describe your own role accurately.
5. **Read the paper once end to end.** Every number is generated from a result file, every reference was traced
   to a record an earlier study checked, and an independent re-computation agrees; but you are the author.
   Re-open each reference link listed in the earlier studies' `bib_verified.json` files.
6. **Keywords and topic.** Suggested CFP themes: "Field Robots: Legged, Flying, and Underwater" and "AI/ML and
   Ethical Implications in Robotics".

## Title, abstract and keywords for the form

See `paper/body.tex` (generated): title, abstract and keywords are at the top.

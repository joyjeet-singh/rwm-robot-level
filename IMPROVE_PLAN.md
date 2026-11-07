# AIR 2027 — Improvement round without new experiments (unattended plan)

This round applies the expert review of the rewritten paper. It trains nothing, runs no simulator and measures nothing new. Its one new computation re-analyses values already stored. It keeps the six-page fit and the Step 2 slot.

What changes:
1. **Two facts about RWM** are traced to their sources and corrected:
   - the 24-step "imagination horizon" (RWM's own settings table lists 100 imagination steps per iteration);
   - the data fraction, "0.13%" and "about 750 times" (RWM's Table 1 gives 6M state transitions).
2. **The abstract** becomes three or four plain claims and one scope sentence, with fewer status labels.
3. **The recommendation** leads with checking at several steps on held-out episodes. The checks that need no recorded future are presented as a fallback.
   - New in this round: how well each check ranks simulators trained the same way.
   - This is an exploratory re-analysis of stored values.
4. **The kinematic residual** is placed against the existing iKCE diagnostic.
5. **The registrations' limits** are stated plainly: they live in the authors' own version control.
6. **Model counts** are reconciled in one place.
7. **The imagined-return signal** is mentioned as the strongest signal that needs no recording, and a candidate for Step 2.
8. **Layout:**
   - Table 3 grouped and with labels that no longer break;
   - Table 1 simplified;
   - appendix tables kept inside their sections;
   - Figure 1 labelled from the corrected facts.

**For the author — how to run it.** In a terminal (the app's terminal or Terminal.app):

```
cd ~/Downloads/PDM/"AIR 2027"
chmod +x go_im.zsh
caffeinate -i ./go_im.zsh auto     # every session in order; safe to re-run after a stop
./go_im.zsh status                 # progress, from another terminal
```

Expect about 5–8 hours, with no background job. If a session stops at Claude's usage limit, the launcher waits for the reset time and then resumes. Keep the Mac plugged in, with the lid open.

**If it stops BLOCKED:**
1. Read `improve/DECISIONS.md`, and write your answer there.
2. Run `./go_im.zsh I<n> resume`, then `auto` again.

When it finishes, read `improve/REPORT.md`.

**How work is split.** The launcher starts one headless Claude Code session per block. Each session receives only the common rules and its own block.

| Session | Model | Effort | Max turns | Work |
|---|---|---|---|---|
| I0 | Sonnet 5.5 | medium | 80 | branch; build; fresh map; facts from the sources, through subagents |
| I1 | Sonnet 5.5 | high | 100 | the re-analysis of how checks rank similar simulators (exploratory), reviewed by Opus |
| I2 | Opus 5.5 | high | 150 | every text change; abstract; framing; six-page fit |
| I3 | Sonnet 5.5 | high | 150 | figure labels, tables, appendix float placement, claim assertions, all checks, supplement |
| I4 | Opus 5.5 | high | 120 | fresh-eyes review of the whole main text, and fixes |
| I5 | Sonnet 5.5 | medium | 60 | change list, documents, report, tag |

Opus writes and reviews the text, where judgement matters most. Sonnet does the source-checking, the re-analysis and the engineering. An Opus subagent checks the statistics.

| Subagent | Model | Used by |
|---|---|---|
| air-bib-checker (web) | Sonnet 5.5 | I0 |
| air-reader (read-only files) | Sonnet 5.5 | I0 |
| air-stats-reviewer | Opus 5.5 | I1, I4 |
| air-claim-checker | Sonnet 5.5 | I3 |
| air-referee | Opus 5.5 | I4 |

<!-- BEGIN COMMON -->
## Common rules (every session)

### C1 What this round is
- Apply the expert review of the rewritten AIR 2027 paper, without new experiments:
  - correct two facts about RWM;
  - rewrite the abstract;
  - lead the recommendation with checking at several steps, adding how well each check ranks simulators trained the same way;
  - place the kinematic residual against iKCE;
  - state the registrations' limits;
  - reconcile the model counts;
  - mention the imagined-return signal;
  - tidy Tables 1 and 3, the appendix float placement and Figure 1.
- The main text must still fit six two-column pages with the Step 2 slot (a box 0.30 of the text height tall, one column wide).
- Sessions I0–I5 work in order; yours is the block after these rules.
  - Earlier sessions' outcomes are in `improve/LOG.md`, `improve/CHECKPOINT.md` and `improve/notes/`.
  - The plan is `IMPROVE_PLAN.md`; its launcher is `go_im.zsh`.
- **Terms.** Keep these two distinct:
  - **joint commands** are the actions, the 12 joint position targets given at every step;
  - the **velocity command** is the walking speed and turning rate a policy is asked to follow.

### C2 Running unattended
- **Never ask the user anything.** When this plan does not settle a choice:
  - take the more conservative option (fewer or weaker claims, less change);
  - record it in `improve/DECISIONS.md` under your session id;
  - go on.

  Stop BLOCKED only when going on would break a rule below.
- **Never start** `claude`, any `go_*.zsh` launcher, or `caffeinate` yourself.
- **On a resumed attempt,** first read `improve/CHECKPOINT.md`, `tail -20 improve/LOG.md`, your notes in `improve/notes/`, and `git log --oneline -15`. Continue from the first unfinished step; never redo a finished one.
- **Before ending,** with any status: commit (C7), append your status line to `improve/LOG.md`, and overwrite `improve/CHECKPOINT.md` (C9).

### C3 What you may change
- **Where you may write.** Write only inside this folder (`AIR 2027`).
  - The launcher compares the rest of `~/Downloads/PDM` before and after every session, and stops the run if anything changed.
  - Never open key files (`../rwm_repro/sshkey*`), and don't read `../rwm_private/`.
- **What this round may change, and nothing else:**
  - `paper/body.template.tex` and `paper/appendix.template.tex`;
  - `paper/refs.bib`, to add or fix a reference that a corrected fact needs (`check_bib.py` must pass);
  - `paper/main_single.tex` and `paper/main_twocol.tex`, only for float placement;
  - new scripts:
    - `scripts/im_facts.py`, which writes `results/external_facts.json`;
    - `scripts/im_within.py`, which writes `results/im_within.*`;
    - `scripts/im_numbers.py`, for number keys;
  - `scripts/paper_numbers.py`, only to call `im_numbers`;
  - `scripts/make_figures.py`, only for Figure 1's horizon lines and labels;
  - `scripts/check_claims.py`, and `scripts/build_supplement.py` only if it needs a fix;
  - `README.md`, `SUBMIT.md`, `ANOMALIES.md`, and the files in `improve/`.
- **Everything else is read-only:** every registration (`prereg/*`), every other script, every existing result file, and the sibling folders.
- **No installs and no sudo. No external actions:** never push, upload, e-mail, submit or post.
  - Reading public web pages through the air-bib-checker subagent is allowed.

### C4 Integrity
- **No number is typed into a template.** Every value comes from a result file through a `{{key}}`.
  - Facts from other papers enter only through `results/external_facts.json`, which `scripts/im_facts.py` writes.
  - For each fact the script records the value, the source's URL and version, and the quoted line it comes from.
- **No new experiments.** Nothing is trained, rolled out or measured.
  - The only new computation is I1's re-analysis of stored values. It is labelled exploratory: added after the registered results were seen.
- **Registered verdicts** are stated as registered. Plain words are fine, but never stronger:
  - NOT SETTLED never becomes "no difference";
  - a verdict keeps its scope.
- **Labels in the abstract.** The abstract may state status once per group of claims rather than once per claim. No non-registered result may read as registered.
- **Claims.** Every claim in the main text is backed by a key or a registered label, and asserted in `scripts/check_claims.py`.
- **Moving material.** No result leaves the paper: material may move to the appendix, not out of the paper.
- **Edit templates only,** never the generated `paper/body.tex` or `paper/appendix.tex`.
- **Never edit a shell script while it runs.**

### C5 Environment
- **Python:** always `./py`. Intel i5, 4 logical cores. A build takes about a minute.
- **PDF text:** comes from pypdf through `scripts/pdf_tools.py`. There is no poppler; Ghostscript (`gs`) may exist, so check before using it.
- **Long output:** send it to `.cache/logs/`, and print only its tail.
- **The pipeline:**
  1. `./py scripts/paper_numbers.py`
  2. `./py scripts/build_paper.py --reserve`
  3. `./py scripts/slack.py`
  4. `./py scripts/check_claims.py`
  5. `./py scripts/check_numbers.py`
  6. `./py scripts/check_bib.py`
  7. `./py scripts/submission_checks.py`
- **Fit rules,** checked by `slack.py`, which must print PASS:
  - both the two-column check and the reserve check have at most six pages;
  - the abstract has at most 200 words.

### C6 Context and token discipline
- **How to read:**
  - read only what your block names;
  - grep first, and read at most about 200 lines at a time;
  - print JSON fields with `./py -c`;
  - print at most about 40 lines of command output.
- **Templates:** work on them section by section, using the line ranges in `improve/notes/I0_map.md`. Never read `paper/appendix.template.tex` whole.
- **Never read these whole:**
  - other rounds' plans;
  - `revision/review/*`;
  - `results/paper_numbers.json` and `results/paper_numbers.md` (grep for a key);
  - any `results/raw_*.npz`;
  - `history.bundle`;
  - logs (use `tail`);
  - any JSON over 100 KB.
- **Subagents:**
  - use only those your block names, and never pass a `model` parameter;
  - give each a self-contained brief: exact files or URLs, the question, and what to return. Ask for no more than you need.

### C7 Git
- Work on branch `improve`, which I0 creates from the rewrite's last commit. Commit subjects are "I<n>: <what>".
- **Staging:** stage only the paths your session changed. Never `git add -A`.
- **Never** push, checkout, restore, reset --hard, stash, clean or rebase.
- **Never commit:** `runs/`, `.cache/`, `.pylib/`, `paper/build/`, `submission/` or `history.bundle`.
- **Tags:** only `air2027-improve-start` (I0) and `air2027-improve` (I5).
- **Figures:** if I3 regenerates figures, it commits all regenerated figure files.

### C8 Style
- **Readers are roboticists, not statisticians.**
  - Use plain language, and define every unavoidable term at first use.
  - One idea per sentence, with at most two numbers-with-intervals in a sentence.
- **Honesty.**
  - Lead with what held, and state what failed just as plainly.
  - Never claim more than the numbers show.
  - Name the scope wherever a claim could be read more widely: one robot, one trotting gait, simulated data, a policy driving the simulators only in imagination, no physics-simulator test.
- **Tone.**
  - No hype.
  - Don't invent the author's motives or experience.
  - Double-blind: no names, places, accounts or links that identify the authors.
- **Notes** in `improve/notes/` are at most 300 words unless your block says otherwise.

### C9 Ending a session
- **Status line.** Append to `improve/LOG.md` the line `## I<n> status: COMPLETE (YYYY-MM-DD HH:MM)`, with PARTIAL or BLOCKED in place of COMPLETE where needed.
  - Then add at most 8 lines: what was done, files, commit hashes, open points.
  - The launcher reads this line, so write it exactly in that form.
- **Checkpoint.** Overwrite `improve/CHECKPOINT.md` with the last session and its status, the next action, and any blockers.
- **BLOCKED:** add to `improve/DECISIONS.md` what is blocked, the options, and your recommendation.
- **PARTIAL:** commit what is done, and write the exact next step in `improve/CHECKPOINT.md`.
<!-- END COMMON -->

<!-- BEGIN I0 -->
## I0 — Branch, build, map and facts (Sonnet 5.5, medium)

**Goal:**
- the branch, with the plan committed;
- a fresh map of the paper;
- the facts the corrections need, quoted from their sources;
- where the paper's current figures came from.

**Read:**
- `rewrite/CHECKPOINT.md`;
- the first 40 lines of `rewrite/REPORT.md`.

**Steps**
1. **Git.**
   - Run `git status --porcelain | head -20`, `git log --oneline -3` and `git tag --points-at HEAD`. HEAD should be tagged `air2027-rewrite`; if not, record what is there.
   - Tag `air2027-improve-start`.
   - Run `git switch -c improve`; on a resumed attempt, `git switch improve`.
   - Commit `IMPROVE_PLAN.md` and `go_im.zsh` alone: "I0: improvement plan".
   - Create in `improve/`: `LOG.md` and `DECISIONS.md` (each with a title line), `CHECKPOINT.md`, and the folders `notes/` and `review/`.
2. **Build and check:** run `paper_numbers.py`, `build_paper.py --reserve` and `slack.py`. Record the output; PASS is expected.
3. **Map:** `improve/notes/I0_map.md` (at most 100 lines).
   - For both templates: every section and subsection, with its line range and word count.
   - Every table and figure, with its label and section.
   - The lines of either template that state RWM's imagination horizon, "24 steps", the data fraction ("0.13%"), "750 times", the model counts (101, 200, 97, 255, 38), the iKCE reference [26], the registration paragraph, and the imagined return. Grep for each.
4. **Facts from the sources.** Ask air-bib-checker for three separate briefs. Each reply is at most 300 words, with exact quotes, URLs and version numbers.
   - **a. RWM** (arXiv 2501.10100: the latest version and v4):
     - the imagination steps per iteration, and the number of imagination environments (Table S11);
     - the history and forecast horizons (Sec. 4.1);
     - the state transitions used for pretraining (Table 1);
     - the teacher-forcing ablation (Sec. 4.3, App. A.4.1).
   - **b. RWM-U** (arXiv 2504.16680):
     - the imagination horizon used for policy training;
     - whether imagined reward is penalised by ensemble disagreement.
   - **c. iKCE** (arXiv 2607.05966): answer from the method section, not the abstract.
     - what it measures and how, in words;
     - on which models and tasks;
     - whether it is used to diagnose a model, or to rank several models;
     - any result on ranking models.
5. **Where the paper's figures came from.**
   - Trace the keys behind the 24-step horizon, "0.13%" and "750 times" through `scripts/*numbers*.py` and their inputs.
   - Ask air-reader whether the released code sets 24 anywhere that matters here, such as imagination steps or steps per environment per update. It should check `../robotic_world_model_lite/scripts/configs/` and `../rsl_rl_rwm/config/`, and return file, line and meaning.
6. **Write `improve/notes/I0_facts.md`** (at most 400 words, plus the quotes):
   - each fact, its source line and URL;
   - for 24, 0.13% and 750: where the paper's value came from, and the corrected value or the right citation.
7. Commit "I0: map and facts".
<!-- END I0 -->

<!-- BEGIN I1 -->
## I1 — How the checks rank simulators trained the same way (Sonnet 5.5, high)

**Goal:** an exploratory re-analysis of stored values. It shows how well each check ranks simulators trained the same way, which is the usual real choice. Nothing is computed from new runs.

**Read:**
- `scripts/stats.py`: `spearman`, `_clean`, and the bootstrap helpers. Import them; never copy them.
- `results/model_table.json`, through `./py -c`: the fields that say how each model was trained.
- How `scripts/revision_analyses.py` forms its "trained" and "compressed" groups (grep).
- The fields of `results/raw_F18.npz`, `raw_X5.npz` and `raw_S1.npz`: keys and shapes only.

**Definitions, fixed by this plan before anything is computed:**
- **Target:** the stored 100-step error E100 under recorded commands.
- **Checks:**
  - E1, E8, λ1 and r̄J under recorded commands;
  - on F18 also λ1 and r̄J under the policy's joint commands, and the negative of the imagined return. Higher return is better, so the sign is flipped to read like the error-type checks.
- **Groups:**
  - **(a)** F18's 48 rollout-trained simulators, whose settings differ;
  - **(b)** X5's rollout-trained simulators: the five seeds and three ensembles of each split, 40 in all. Rank within each split, then average;
  - **(c)** F18's 40 compressed copies;
  - **(d)** for reference, all 97 of F18. These must reproduce Table 2's stored values exactly.
- **Intervals:** 95% percentile intervals from 20,000 bootstrap draws of models, with `numpy.random.default_rng(0)`; on X5, models are resampled within splits.
- **Differences:** ρ(λ1) − ρ(E8) and ρ(r̄J) − ρ(E8), from the same draws.
- **Already known** (appendix A2): an earlier study ranked the 368-step error at +0.85 for (a) and +0.21 for (b), with the growth rate. Say so in the output.

**Steps**
1. **Write `scripts/im_within.py`.** It writes `results/im_within.json`, and `results/im_within.txt` (at most 40 lines, labelled exploratory). It refuses to run if a group's size differs from the definitions.
2. **Write `scripts/im_numbers.py`,** with the `add(...)` pattern of `scripts/revision_numbers.py`, for the values the text will use. Call it from `paper_numbers.py` after `rw_numbers`.
3. **Write `scripts/im_facts.py`.**
   - It writes `results/external_facts.json` from `improve/notes/I0_facts.md`'s facts, as constants in the script, each with its value, URL, version and quote.
   - Add their keys to `im_numbers.py`, including any derived fraction.
4. **Test:** group (d) reproduces Table 2's F18 column; the group sizes are 48, 40 and 40.
5. **Review.** Ask air-stats-reviewer to check `im_within.py` and `im_within.txt` against the definitions above, in at most 300 words. Fix every MUST FIX and SHOULD FIX.
6. **Finish.**
   - Write `improve/notes/I1.md`: the headline readings in plain words, and the keys.
   - Commit "I1: within-method rankings (exploratory) and external facts".
<!-- END I1 -->

<!-- BEGIN I2 -->
## I2 — Every text change (Opus 5.5, high)

**Goal:** the main text, and the appendix where needed, carry all the changes listed in C1, and `slack.py` prints PASS.

**Read:**
- `improve/notes/I0_facts.md`, `I0_map.md` and `I1.md`;
- `results/im_within.txt`;
- `paper/body.template.tex` (whole);
- in `paper/appendix.template.tex`, only the sections the map names for these changes.

**Changes, in priority order:**
1. **Facts.** Correct RWM's imagination horizon and the data fraction everywhere they appear, through keys from `external_facts.json`.
   - If 24 has a real source in the released code (I0), cite it for what it is. Otherwise keep 24 steps only as an intermediate horizon (0.48 s), not as RWM's.
   - Write in `improve/notes/I2.md` exactly what Figure 1's vertical lines and labels should become, for I3.
2. **Abstract** (at most 200 words): three or four plain claims, then one scope sentence.
   - Give status once per group of claims (C4).
   - Keep "no physics-simulator test".
3. **Recommendation.**
   - Lead with checking at several steps (8 or more) on held-out episodes.
   - Present the checks without recorded futures as a fallback that mainly separates training methods.
   - Give I1's result on simulators trained the same way in one or two sentences, labelled exploratory, in Sec. 4.2 and the checklist. Its table goes to appendix A4.
4. **Residual and iKCE.** One sentence, from I0's facts: what the two share, and what differs. Claim nothing the facts don't support.
5. **Registrations.** One sentence in "Statistics and registration":
   - the commits are in the authors' own version control, so their times are self-reported;
   - the supplementary history shows their order, but not independent timestamps;
   - the first question's answer was largely predictable from what was known (appendix A2).
6. **Counts.** One sentence reconciling 101 trained and 200 compressed GRU simulators with the pools: F18 (97), X5 (255) and the 38 runs of other architectures.
7. **Imagined return.** One sentence, labelled secondary:
   - it was the strongest signal under the policy that needs no recording;
   - it requires a policy and a reward, and was not registered as a check;
   - it is a candidate for Step 2.
8. **Fewer labels in Sections 3–4.**
   - Say once which kinds of status exist.
   - Keep small capitals for verdicts in Table 3 only.
   - In the text, use plain words with "(Table 3)".
   - Every non-registered result keeps its label.
9. **AI-use statement.** Compare it with the venue notes in `SUBMIT.md`. Don't change its substance; list any mismatch for the author in `I2.md`.
10. **Fit.** Make room by tightening wording, never by dropping a registered result. `slack.py` must print PASS.

**Finish.**
- Write `improve/notes/I2_claims.md` (at most 120 lines): every new or changed sentence that bears a result, with section, short quote, claim, keys and status.
- Write `improve/notes/I2.md`:
  - what changed;
  - the instructions for I3: Figure 1, Tables 1 and 3;
  - the final `slack.py` output.
- Commit "I2: text changes".
<!-- END I2 -->

<!-- BEGIN I3 -->
## I3 — Figure, tables, floats, claims and checks (Sonnet 5.5, high)

**Read:**
- `improve/notes/I2.md` and `I2_claims.md`;
- `scripts/make_figures.py`, only Figure 1's code (grep);
- in `paper/body.template.tex`, only Tables 1 and 3 (line ranges from the map);
- the start of `scripts/check_claims.py`, and its claim descriptions (grep).

**Steps**
1. **Figure 1:** change the horizon lines and labels as `I2.md` says. Regenerate all figures, and commit them.
2. **Table 3:**
   - Group the rows by registration: the main study (Q1–Q6); study FT; studies CL and AR.
   - Put estimates of different kinds under separate subheadings: differences in rank correlation, and skill.
   - Verdict labels must never break inside a word. Use a ragged-right column with hyphenation off, or short labels with a key in the caption.
3. **Table 1:** move the model and trajectory counts into the caption, and group the rows (training choices; compression; uncertainty). Keep every value.
4. **Appendix floats.** Keep each appendix table inside its own section: `\usepackage{placeins}` in both wrappers, and `\FloatBarrier` before each appendix section. Check in the extracted text that each table's caption falls between its own section heading and the next.
5. **Visual check.** If `gs` is available, render the pages with Tables 1 and 3, in both PDFs, to PNG in `.cache/`, and look at them. Otherwise, check the extracted text for broken verdict words.
6. **Fit.** If a layout change breaks `slack.py`, shorten captions or table text only, never sentences in the main text. If it still fails, undo that layout change, and note it for I4.
7. **Claims.** Update `check_claims.py`:
   - add an assertion for every claim in `I2_claims.md`, including the exploratory values and the external facts;
   - keep the others;
   - all must pass.
8. **Full pipeline (C5),** with every step passing.
   - Then ask air-claim-checker to check the sentences in `I2_claims.md` against `results/im_within.json`, `results/external_facts.json` and the result files they name.
   - Fix each finding minimally, and never strengthen a claim. Log the changes in `improve/notes/I3.md`, then rerun the pipeline.
9. **Supplement.** Rebuild it with `./py scripts/build_supplement.py`, and scan it as before. It must have 0 hits.
10. Commit "I3: figure, tables, claims and checks".
<!-- END I3 -->

<!-- BEGIN I4 -->
## I4 — Fresh-eyes review (Opus 5.5, high)

**Read:** `improve/notes/I2.md` and `I3.md`.

**Steps**
1. **Extract the main text** of the two-column check to `improve/review/body.txt`, with `scripts/pdf_tools.py`.
2. **Get two reviews, in parallel.**
   - air-referee reviews `improve/review/body.txt` only, as a whole paper.
   - air-stats-reviewer checks every sentence of `body.txt` that states a registered verdict, an exploratory result or an external fact. It checks them against:
     - `results/stats_report.txt`, `results/ft_report.txt` and `results/s1_report.txt`;
     - `results/im_within.txt` and `results/external_facts.json`;
     - the verdict rules of the three registrations (grep the label lines).

     It replies in at most 400 words.
3. **Act on the reviews.**
   - Fix unclear text, jargon and overclaims.
   - Run no new analyses.
   - A requested experiment goes to `improve/DECISIONS.md` as a Step 2 candidate.
   - Decline anything else with a reason, under "I4 declined".
4. **Keep the fit.** `slack.py` must still PASS. Update `check_claims.py` for changed sentences, then rerun the full pipeline.
5. **Finish.**
   - Write `improve/notes/I4.md` with the referee's score and main points, and what was done about each.
   - Commit "I4: review fixes".
<!-- END I4 -->

<!-- BEGIN I5 -->
## I5 — Change list, documents and report (Sonnet 5.5, medium)

**Read:**
- `improve/notes/I0_facts.md`, `I1.md`, `I2.md`, `I3.md` and `I4.md`;
- `improve/DECISIONS.md` and `SUBMIT.md`.

**Steps**
1. **Change list for the author.**
   - If `latexdiff` is on PATH, and `git show air2027-improve-start:paper/body.tex` works, build `paper/build/body_diff.pdf`.
   - Otherwise, write `improve/review/section_changes.md` (at most 100 lines).
2. **Update the documents:**
   - `README.md`: this round and its files;
   - `SUBMIT.md`: the submission window (15 Nov–31 Dec 2026); the abstract text for the form, from `paper/ABSTRACT.txt`; that `STEP2_PLAN.md` describes the robot-level test;
   - `ANOMALIES.md`: only if needed.
3. **Write `improve/REPORT.md`** for the author, in plain language and at most 450 words:
   - what changed;
   - the facts corrected, with their sources;
   - the result on simulators trained the same way;
   - the referee's score and main points;
   - what the author must do before submitting: put the flagged paragraphs in your own words; check the AI-use statement; check the corrected facts against the sources;
   - the next step: `STEP2_PLAN.md`.
4. **Close the round.**
   1. Final check: build, `check_claims.py`, and `slack.py` (PASS).
   2. Commit "I5: improvement round complete", and tag it `air2027-improve`.
   3. Run `git bundle create history.bundle --all`, then `git bundle verify history.bundle`.
5. **Overwrite `improve/CHECKPOINT.md`:** "Improvement round complete. Next: read improve/REPORT.md, then decide Step 2 (STEP2_PLAN.md)."
<!-- END I5 -->

## Traceability: review points → sessions

| Review point | Change | Sessions |
|---|---|---|
| RWM's 24-step horizon is wrong (its Table S11: 100) | Trace, correct, cite; fix Figure 1 | I0, I1, I2, I3 |
| Data fraction (0.13%, 750 times) doesn't match RWM's Table 1 (6M) | Trace, correct, cite | I0, I1, I2 |
| The abstract is overloaded with status labels | Three or four claims and one scope sentence | I2 |
| 8-step error is the fair comparison; checks mainly separate training methods | Lead with checking at several steps; show rankings among similar simulators (exploratory) | I1, I2 |
| The residual is close to iKCE | One sentence on what is shared and what differs | I0, I2 |
| Registrations are self-hosted; Q1 was predictable | One sentence | I2 |
| Model counts are hard to reconcile | One sentence | I2 |
| Imagined return is only an aside | One sentence; a candidate for Step 2 | I2 |
| Tables 1 and 3; drifting appendix floats | Regroup; fix labels; keep floats in their sections | I3 |
| No robot-level evidence | Not answerable without experiments: see `STEP2_PLAN.md` | — |

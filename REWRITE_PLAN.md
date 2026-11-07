# AIR 2027 — Rewrite of the main text (unattended plan)

This round rewrites the paper's main text around everything now known: the original study, revision round 1 and Step 1. It runs no new analysis and collects no new data. It ends with a paper ready to submit when the submission site opens (15 November 2026). It also leaves a reserved space, the "Step 2 slot", where Step 2's result can go if it arrives before the deadline (31 December 2026).

What changes:
- **The story.** The abstract, introduction and discussion lead with what held up across all the evidence, and say plainly what did not. Step 1's results move from the appendix into the main text.
- **The space.** The main text must still fit six two-column pages, now with the Step 2 slot left free. Older detail moves to the appendix.
- **Nothing else.** Every number still comes from a result file, and registered verdicts stay as registered.

**For the author — how to run it.** In a terminal (the app's terminal or Terminal.app):

```
cd ~/Downloads/PDM/"AIR 2027"
chmod +x go_rw.zsh
caffeinate -i ./go_rw.zsh auto     # every session in order; safe to re-run after a stop
./go_rw.zsh status                 # progress, from another terminal
```

Expect about 6–9 hours. There is no background job. If a session stops at Claude's usage limit, the launcher waits until the reset time the message gives, then resumes. Keep the Mac plugged in with the lid open.

If it stops BLOCKED:
1. Read `rewrite/DECISIONS.md` and write your answer there.
2. Run `./go_rw.zsh W<n> resume`, then `auto` again.

When it finishes, read `rewrite/REPORT.md` first, then `paper/AIR2027_submission.pdf`. The report lists the paragraphs to put in your own words before you submit.

**How work is split.** The launcher starts one headless Claude Code session per block. Each session receives only the common rules and its own block.

| Session | Model | Effort | Max turns | Work |
|---|---|---|---|---|
| W0 | Sonnet 5.5 | medium | 100 | branch; map of the paper; digest of all the evidence; a tool that measures free space |
| W1 | Opus 5.5 | high | 80 | the new story and outline, with a statistics check of every planned claim |
| W2 | Opus 5.5 | high | 180 | rewrite the main text; move detail to the appendix; fit six pages with the Step 2 slot |
| W3 | Sonnet 5.5 | high | 120 | claim assertions, all build checks, a claim check against the evidence, supplement |
| W4 | Opus 5.5 | high | 120 | fresh-eyes review of the whole main text, and fixes |
| W5 | Sonnet 5.5 | medium | 60 | change list, documents, report, tag |

Opus does the work where judgement matters most: the story, the writing and the final review. Sonnet does the mapping, the checks and the paperwork.

The subagents are the ones in `.claude/agents/`, with their models pinned in their files:

| Subagent | Model | Used by |
|---|---|---|
| air-stats-reviewer | Opus 5.5 | W1, W4 |
| air-claim-checker | Sonnet 5.5 | W3 |
| air-referee | Opus 5.5 | W4 |

<!-- BEGIN COMMON -->
## Common rules (every session)

### C1 What this round is
- **The task.** Rewrite the main text of the AIR 2027 paper (`paper/body.template.tex`) around all the evidence:
  - the original study;
  - revision round 1 (`revision/REPORT.md`);
  - Step 1 (`step1/REPORT.md`).
- **The space.** The result must fit six two-column pages with the **Step 2 slot** inserted: a blank box 0.30 of the text height tall (about 19 lines of one column) and one column wide. Step 2's result can then be added later without more cuts.
- **The sessions.** W0–W5 work in order; yours is the block after these rules.
  - Earlier sessions' outcomes are in `rewrite/LOG.md`, `rewrite/CHECKPOINT.md` and `rewrite/notes/`.
  - The plan is `REWRITE_PLAN.md`; its launcher is `go_rw.zsh`.
- **Terms.** Keep these two distinct:
  - **joint commands** are the actions, the 12 joint position targets given at every step;
  - the **velocity command** is the walking speed and turning rate a policy is asked to follow.

### C2 Running unattended
- **Never ask the user anything.** When this plan does not settle a choice:
  - take the more conservative option (fewer or weaker claims, less change);
  - record it in `rewrite/DECISIONS.md` under your session id;
  - go on.

  Stop BLOCKED only when going on would break a rule below.
- **Never start** `claude`, `go_rw.zsh`, `go_s1.zsh`, `go_rev.zsh` or `caffeinate` yourself.
- **On a resumed attempt,** first read `rewrite/CHECKPOINT.md`, `tail -20 rewrite/LOG.md`, your notes in `rewrite/notes/`, and `git log --oneline -15`. Continue from the first unfinished step; never redo a finished one.
- **Before ending,** with any status: commit (C7), append your status line to `rewrite/LOG.md`, and overwrite `rewrite/CHECKPOINT.md` (C9).

### C3 What you may change
- **Where you may write.** Write only inside this folder (`AIR 2027`).
  - The launcher compares the rest of `~/Downloads/PDM` before and after every session, and stops the run if anything changed.
  - Never open key files (`../rwm_repro/sshkey*`), and don't read `../rwm_private/`.
- **What this round may change, and nothing else:**
  - `paper/body.template.tex` and `paper/appendix.template.tex`;
  - `paper/refs.bib`, only to drop a reference the paper no longer cites or to fix one (`check_bib.py` must pass);
  - `paper/main_single.tex`, `paper/main_twocol.tex` and `scripts/build_paper.py`, only for the Step 2 slot (W0);
  - `scripts/slack.py` (new, W0) and `scripts/rw_numbers.py` (new number keys, W2), plus `scripts/paper_numbers.py` to call it;
  - `scripts/make_figures.py`, only if the outline changes a figure;
  - `scripts/check_claims.py` (W3, W4), and `scripts/build_supplement.py` only if it needs a fix;
  - `README.md`, `SUBMIT.md`, `ANOMALIES.md`, and the files in `rewrite/`.
- **Everything else is read-only:** every registration (`prereg/*`), every analysis script and every result file.
- **No installs and no sudo. No external actions:** never push, upload, e-mail, submit or post.

### C4 Integrity
- **No number is typed into a template.** Every value comes from a result file through a `{{key}}`:
  - `scripts/paper_numbers.py`, with `air_numbers.py`, `revision_numbers.py`, `s1_numbers.py` and, new, `rw_numbers.py`;
  - a new key may only read a value that already exists in a result file.
- **No new analysis, no new data, and no rerun of a registered script.** A question that would need one goes to `rewrite/DECISIONS.md` as a Step 2 candidate.
- **Verdicts.** Registered verdicts are stated as registered. Plain words are fine, but never stronger:
  - NOT SETTLED never becomes "no difference" or "similar";
  - a verdict keeps its scope: pool, command source, architectures.
- **Labels.** Secondary, exploratory and descriptive results are labelled so wherever they appear, the abstract included.
- **Claims.** Every claim in the main text is backed by a key or a registered label, and asserted in `scripts/check_claims.py`.
- **Moving material.** No registered result leaves the paper: material may move from the main text to the appendix, not out of the paper.
- **Generated files.** Edit templates only, never the generated `paper/body.tex` or `paper/appendix.tex`.
- **Scripts.** Never edit a shell script while it runs.

### C5 Environment
- **Python:** always `./py`. Intel i5, 4 logical cores. A build takes about a minute.
- **PDF text:** comes from pypdf through `scripts/pdf_tools.py` (there is no poppler).
- **Long output:** send it to `.cache/logs/`, and print only its tail.
- **The pipeline:**
  1. `./py scripts/paper_numbers.py`
  2. `./py scripts/build_paper.py`
  3. `./py scripts/slack.py` (from W0)
  4. `./py scripts/check_claims.py`
  5. `./py scripts/check_numbers.py`
  6. `./py scripts/check_bib.py`
  7. `./py scripts/submission_checks.py`
- **Fit rules,** checked by `slack.py`:
  - the two-column check has at most six pages;
  - the reserve check (the two-column check with the Step 2 slot) has at most six pages;
  - the abstract has at most 200 words.

### C6 Context and token discipline
- **How to read:**
  - read only what your block names;
  - grep first, and read at most about 200 lines at a time;
  - print JSON fields with `./py -c`;
  - print at most about 40 lines of command output.
- **Templates:** work on them section by section, using the line ranges in `rewrite/notes/W0_map.md`. Never read `paper/appendix.template.tex` whole.
- **Never read these whole:**
  - `REVISION_PLAN.md`, `STEP1_PLAN.md` and `revision/review/*`;
  - `results/paper_numbers.json` and `results/paper_numbers.md` (grep for a key);
  - any `results/raw_*.npz`;
  - `history.bundle`;
  - logs (use `tail`);
  - any JSON over 100 KB.
- **Subagents:**
  - use only those your block names, and never pass a `model` parameter;
  - give each a self-contained brief: exact files, the question, and what to return. Ask for no more than you need.

### C7 Git
- Work on branch `rewrite`, which W0 creates from Step 1's last commit. Commit subjects are "W<n>: <what>".
- **Staging:** stage only the paths your session changed. Never `git add -A`.
- **Never** push, checkout, restore, reset --hard, stash, clean or rebase.
- **Never commit:** `runs/`, `.cache/`, `.pylib/`, `paper/build/`, `submission/` or `history.bundle`.
- **Tags:** only `air2027-rewrite-start` (W0) and `air2027-rewrite` (W5).
- **Figures:** `figures/fig_horizon.*` differ from the last commit because they were rendered on this Mac. Leave them alone, unless W2 regenerates figures; then commit all regenerated figures.

### C8 Style
- **Readers are roboticists, not statisticians.**
  - Use plain language, and define every unavoidable term at first use.
  - One idea per sentence, with at most two numbers-with-intervals in a sentence. Prefer one clear number to three.
- **Honesty.**
  - Lead with what held, and state what failed just as plainly.
  - Never claim more than the numbers show.
  - Name the scope wherever a claim could be read more widely: one robot, one trotting gait, simulated data, a policy driving the simulators only in imagination, no physics-simulator test.
- **Tone.**
  - No hype: no "novel", no "first", and "significant" only in its statistical sense.
  - Don't invent the author's motives or experience.
  - Double-blind: no names, places, accounts or links.
- **Notes** in `rewrite/notes/` are at most 300 words unless your block says otherwise.

### C9 Ending a session
- **Status line.** Append to `rewrite/LOG.md` the line `## W<n> status: COMPLETE (YYYY-MM-DD HH:MM)`, with PARTIAL or BLOCKED in place of COMPLETE where needed.
  - Then add at most 8 lines: what was done, files, commit hashes, open points.
  - The launcher reads this line, so write it exactly in that form.
- **Checkpoint.** Overwrite `rewrite/CHECKPOINT.md` with the last session and its status, the next action, and any blockers.
- **BLOCKED:** add to `rewrite/DECISIONS.md` what is blocked, the options, and your recommendation.
- **PARTIAL:** commit what is done, and write the exact next step in `rewrite/CHECKPOINT.md`.
<!-- END COMMON -->

<!-- BEGIN W0 -->
## W0 — Branch, paper map, evidence digest and free-space tool (Sonnet 5.5, medium)

**Goal:**
- the branch, with the plan committed;
- a map of the current paper;
- a one-line-per-result digest of all the evidence;
- a tool that says how much space is free.

**Read:**
- `step1/CHECKPOINT.md`;
- the first 20 lines of `SUBMIT.md`;
- `scripts/build_paper.py`;
- `paper/main_twocol.tex` and `paper/main_single.tex`;
- `ls paper/`.

**Steps**
1. **Git.**
   - Run `git status --porcelain | head -20`, `git log --oneline -3` and `git tag --points-at HEAD`. HEAD should be Step 1's last commit, tagged `air2027-step1`; if not, record what is there.
   - Tag `air2027-rewrite-start`.
   - Run `git switch -c rewrite`; on a resumed attempt, `git switch rewrite`.
   - Commit `REWRITE_PLAN.md` and `go_rw.zsh` alone: "W0: rewrite plan".
   - Create in `rewrite/`: `LOG.md` and `DECISIONS.md` (each with a title line), `CHECKPOINT.md`, and the folders `notes/` and `review/`.
2. **Build the current paper** (`paper_numbers.py`, `build_paper.py`), and record both PDFs' page counts.
3. **The Step 2 slot.**
   - Add `\providecommand{\stepslot}{}` to both wrappers, so the macro is empty in every normal build.
   - `build_paper.py --reserve [H]` also builds `paper/build/reserve_check.pdf`. That is the two-column check with `\stepslot` drawn as a framed box labelled "Step 2 slot", H\textheight tall (default 0.30) and one column wide.
   - If the body has no `\stepslot`, the reserve build puts the box just before `\section{Discussion}`.
   - The normal outputs must not change.
4. **Write `scripts/slack.py`.** It builds what it needs and prints:
   - the two-column check's pages and the reserve check's pages;
   - if the two-column check runs over six pages: the overflow, as the number of text lines on pages 7 and later;
   - if it fits: the tallest box (in steps of 0.02\textheight) that still fits in six pages, that is, the free space;
   - the abstract's word count, from `build_paper.py`'s own count;
   - **PASS** only if both checks have at most six pages and the abstract has at most 200 words.

   Run it on the current paper and record the result. Expected: no free space, so FAIL.
5. **Paper map:** `rewrite/notes/W0_map.md` (at most 150 lines), covering `body.template.tex` and `appendix.template.tex`:
   - every section, subsection and paragraph heading, with its line range and word count;
   - every table and figure: its label, the first words of its caption, and its section;
   - the keys each section of the main text uses;
   - which `check_claims.py` assertions concern which section (they are keyed by description);
   - the abstract's word count.
6. **Evidence digest:** `rewrite/notes/W0_evidence.md` (at most 180 lines), one line per result the paper states or could state. Each line gives:
   - an id (E1, E2, …);
   - the finding in plain words;
   - its status: registered primary, registered secondary, exploratory, descriptive, or earlier study;
   - the verdict label, if registered;
   - its keys and its source file.

   The sources are:
   - `results/stats_report.txt`, `results/ft_report.txt`, `results/s1_report.txt` and `results/s1_budget.txt`;
   - the headline lines of `results/revision_exploratory.txt`;
   - every result-bearing sentence or table row of the current main text.

   Copy labels and values verbatim from the files; compute nothing.
7. Commit "W0: paper map, evidence digest and free-space tool".
<!-- END W0 -->

<!-- BEGIN W1 -->
## W1 — The new story and outline (Opus 5.5, high)

**Goal:** a plan for the new main text that a robotics reviewer would find clear, honest and convincing. Every later session follows it.

**Read:**
- `rewrite/notes/W0_map.md` and `W0_evidence.md`;
- `step1/REPORT.md`;
- in `paper/body.template.tex`, only the abstract, introduction, Discussion, Limitations and Conclusion (line ranges from the map).

**What earlier internal reviews raised**, and what answers each:
1. No robot-level evidence: no policy trained in a simulator and tested in physics. This is Step 2's job, and is not answered yet.
2. One robot, one gait, simulated data. Not answerable on the CPU.
3. One simulator design. Answered by Step 1's study AR, with the RSSM as an exception.
4. Models not fully trained. Answered by Step 1's study TB (descriptive).
5. Label-free checks tested only with recorded commands. Answered by Step 1's study CL, partly against the residual.
6. Dense, number-heavy writing: five earlier studies packed into six pages, with jargon.
7. An abstract that led with what did not hold.

**A starting point,** from the planning conversation. Check every part against `W0_evidence.md`, and change it wherever the evidence says otherwise.
- **One-step checks pick the wrong training choices** on this simulator: teacher forcing, one-step training, and pruning without fine-tuning. This agrees with RWM's own ablation.
  - The reversal held after four times longer training, and in the MLP and Transformer simulators (descriptive).
  - It did not hold in the RSSM.
- **Two checks need no recorded future states:** the kinematic residual and the growth rate. Both rank simulators by their 2-second error.
  - With recorded commands, the residual ranked highest.
  - With the RWM authors' policy driving the simulators, the residual ranked less well (registered), and the growth rate's change was not settled.
  - Across other architectures, the growth rate beat one-step error (registered). This rests on the MLP and the Transformer.
- **Measure the growth rate along the policy's own rollouts.** Including the policy's reactions in it removed its ranking (secondary).
- **Pruning without retraining** looks harmless one step ahead, but loses most skill by 2 s. Fine-tuning restores it (registered).
- **Scope:** one robot, one trotting gait on flat ground, simulated data; a policy driving the simulators only in imagination; no physics-simulator test.

**Write `rewrite/OUTLINE.md`** (at most 1,600 words), in nine parts:
1. **Story:** 4–6 plain sentences on what a roboticist should take away, strongest evidence first, each tied to digest ids.
2. **Title:** keep it, or change it and give two alternatives. Record either way in `DECISIONS.md`, for the author.
3. **Abstract plan:** its sentences in order, each with keys and status.
   - At most 200 words.
   - The scope sentence says what was tested and what was not.
4. **Section plan:** each section's purpose, and its budget in words. Count about 10 words per two-column line, and floats at their current size from the map. The total must fit six two-column pages with the Step 2 slot, which is 0.30\textheight, about 19 lines of one column.
5. **Moves:**
   - what leaves the main text for the appendix, and where it goes;
   - what is condensed;
   - what is cut from the main text only because the appendix already has it.
6. **Tables and figures:** keep, change or replace each one. New ones may use existing result files only.
7. **Claims:**
   - every claim the main text will make, with its status label and keys;
   - a "must not say" list. For example: "the residual is the best check" without its scope; "no difference" for NOT SETTLED; "holds across architectures" without the RSSM; anything about policies in physics.
8. **Step 2 slot:**
   - where it goes;
   - what it will hold: a paragraph and a small table answering one question, stated here in one sentence;
   - which sentences change when Step 2 arrives.
9. **Author's voice:** the paragraphs the author should put in their own words (introduction, discussion), for the report.

**Check.** Ask air-stats-reviewer to check parts 3 and 7 of `OUTLINE.md` against two things:
- `W0_evidence.md`;
- the verdict rules of the three registrations (grep the label lines of `prereg/PREREG.md`, `PREREG_FT.md` and `PREREG_S1.md`).

It replies in at most 400 words, as MUST FIX / SHOULD FIX / NOTE. Fix every MUST FIX and SHOULD FIX, and record the review in `DECISIONS.md`.

**Finish.** Commit "W1: story and outline".
<!-- END W1 -->

<!-- BEGIN W2 -->
## W2 — Rewrite the main text (Opus 5.5, high)

**Goal:** the main text rewritten as `rewrite/OUTLINE.md` says, fitting six two-column pages with the Step 2 slot, and building cleanly.

**Read:**
- `rewrite/OUTLINE.md` (whole);
- `rewrite/notes/W0_map.md`;
- `paper/body.template.tex` (whole);
- in `paper/appendix.template.tex`, only the sections that receive moved material (line ranges from the map);
- keys, by grepping `results/paper_numbers.md`.

**Steps**
1. **Rewrite `body.template.tex`** section by section, following the outline.
   - Keep labels, `\ref` and `\cite` working.
   - Put `\stepslot` where the outline places the Step 2 slot.
2. **Move material** to `appendix.template.tex` where the outline says.
   - Keep its keys, and update every cross-reference.
   - Keep the appendix numbering coherent.
3. **New keys:** only for values already in result files.
   - Put them in `scripts/rw_numbers.py`, using the `add(...)` pattern of `scripts/revision_numbers.py`.
   - Call it from `paper_numbers.py` after `s1_numbers`.
4. **Tables and figures** as the outline says. Figures are made only through `scripts/make_figures.py`, from existing result files.
5. **Build loop.** Run `paper_numbers.py`, then `build_paper.py --reserve`, then `slack.py`.
   - Repeat until `slack.py` prints PASS, and the build has no undefined references or missing keys.
   - Cut words before cutting content. If content must go, move it to the appendix.
6. **Claims list.** Don't edit `check_claims.py`. Instead, write `rewrite/notes/W2_claims.md` (at most 150 lines).
   - List every result-bearing sentence of the new main text, in order: section, short quote, claim, keys, status label.
   - Mark the sentences that are new or changed.
7. **Notes.** Write `rewrite/notes/W2.md`:
   - what moved where;
   - the final `slack.py` output;
   - anything that departed from the outline, with the reason (also in `DECISIONS.md`).
8. Commit "W2: main text rewritten around the full evidence".
<!-- END W2 -->

<!-- BEGIN W3 -->
## W3 — Claims, checks and supplement (Sonnet 5.5, high)

**Read:**
- `rewrite/notes/W2_claims.md` and `W2.md`;
- the first 80 lines of `scripts/check_claims.py`, and its claim descriptions (grep);
- `results/claims_check.json`, its summary only, through `./py -c`.

**Steps**
1. **Update `check_claims.py`** so it tests the new text.
   - Keep assertions for claims that moved to the appendix.
   - Remove only assertions for claims no longer anywhere in the paper. List each, and why, in `rewrite/notes/W3.md`.
   - Add an assertion for every new or changed claim in `W2_claims.md`: verdict, direction, count or order.
   - All must pass.
2. **Run the full pipeline.** Every step must pass:
   - `paper_numbers.py`;
   - `build_paper.py --reserve`, twice, giving identical text through `pdf_tools.py`;
   - `slack.py` (PASS);
   - `check_claims.py`, `check_numbers.py`, `check_bib.py` (strict) and `submission_checks.py`.

   `paper/ABSTRACT.txt` is rewritten by `build_paper.py`; confirm it changed with the abstract.
3. **Check against the evidence.** Ask air-claim-checker to check `paper/body.tex` against:
   - `rewrite/notes/W0_evidence.md`;
   - part 7 of `rewrite/OUTLINE.md`;
   - the result files they name.

   Fix each finding minimally, and never strengthen a claim. Log every change in `W3.md`, then rerun step 2.
4. **Rebuild the supplement** with `./py scripts/build_supplement.py`, and scan it as Step 1 did. It must have 0 hits.
5. Commit "W3: claims, checks and supplement".
<!-- END W3 -->

<!-- BEGIN W4 -->
## W4 — Fresh-eyes review (Opus 5.5, high)

**Read:**
- `rewrite/OUTLINE.md`, parts 1, 7 and 8;
- `rewrite/notes/W3.md`.

**Steps**
1. **Extract the main text** of the two-column check (no appendix) to `rewrite/review/body.txt`, with `scripts/pdf_tools.py`.
2. **Get two reviews, in parallel.**
   - air-referee reviews `rewrite/review/body.txt` only, as a whole paper.
   - air-stats-reviewer checks every sentence of `body.txt` that states a registered verdict or a labelled result. It checks them against:
     - `results/stats_report.txt`, `results/ft_report.txt` and `results/s1_report.txt`;
     - the verdict rules of the three registrations (grep the label lines).

     It replies in at most 400 words.
3. **Act on the reviews.**
   - Fix unclear text, jargon and overclaims.
   - Run no new analyses; registered verdicts stand.
   - A requested experiment becomes a Step 2 candidate in `DECISIONS.md`.
   - Decline anything else with a reason, under "W4 declined" in `DECISIONS.md`.
4. **Keep the fit.** `slack.py` must still PASS. Update `check_claims.py` for changed sentences, and rerun W3's step 2.
5. **Finish.**
   - Write `rewrite/notes/W4.md` with the referee's score and main points, and what was done about each.
   - Commit "W4: review fixes".
<!-- END W4 -->

<!-- BEGIN W5 -->
## W5 — Change list, documents and report (Sonnet 5.5, medium)

**Read:**
- `rewrite/notes/W2.md`, `W3.md` and `W4.md`;
- parts 1, 2, 8 and 9 of `rewrite/OUTLINE.md`;
- `rewrite/DECISIONS.md` and `SUBMIT.md`.

**Steps**
1. **Change list for the author.**
   - If `latexdiff` is on PATH, and the old generated text can be recovered (`git show air2027-rewrite-start:paper/body.tex`), build `paper/build/body_diff.pdf` comparing it with the current one.
   - Otherwise, write `rewrite/review/section_changes.md` (at most 120 lines): what changed, section by section.
2. **Update the documents:**
   - `README.md`: this round, its files and its run order;
   - `SUBMIT.md`: the submission window (15 Nov–31 Dec 2026); which PDF to submit; the Step 2 slot; replacing the file if Step 2 finishes in time, after checking that CMT allows it;
   - `ANOMALIES.md`: only if something needs recording.
3. **Write `rewrite/REPORT.md`** for the author, in plain language and at most 500 words:
   - the new story in 4–6 sentences, and the title with its alternatives;
   - what moved to the appendix; the fit (`slack.py` output); where the Step 2 slot is;
   - the referee's score and main points, and what was done;
   - the paragraphs to put in your own words before submitting, and the AI-use statement to check;
   - next steps: Step 2's registration should test the main text's claims (outline part 8); and the submission plan.
4. **Close the round.**
   1. Final check: build, `check_claims.py`, and `slack.py` (PASS).
   2. Commit "W5: rewrite complete", and tag it `air2027-rewrite`.
   3. Run `git bundle create history.bundle --all`, then `git bundle verify history.bundle`.
5. **Overwrite `rewrite/CHECKPOINT.md`:** "Rewrite complete. Next: read rewrite/REPORT.md and the PDF; put the flagged paragraphs in your own words; then Step 2."
<!-- END W5 -->

## Traceability: review points → sessions

| Point raised | What this round does | Sessions |
|---|---|---|
| The abstract led with what did not hold | The story leads with what held across all evidence, and states what failed | W1, W2 |
| Step 1's answers sit only in the appendix | They move into the main text: policy in the loop, other architectures, training length | W1, W2 |
| Dense, number-heavy writing | Fewer numbers per sentence; older detail moves to the appendix; terms defined | W2, W4 |
| No robot-level evidence | Not answerable here; the Step 2 slot keeps room for it | W0, W1, W2 |
| Claims must match the evidence | Claim list checked before writing; assertions; a claim check; a statistics review | W1, W3, W4 |

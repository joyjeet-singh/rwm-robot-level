# AIR 2027 — revision round 1: plan for unattended Claude Code sessions

This plan applies the internal review (`revision/REVIEW.md`) to the AIR 2027 submission. It makes these changes:

- Fixes the factual slips.
- Removes the overclaims.
- Credits RWM's own results and adds the missing references.
- Re-centres the abstract and introduction on what held up.
- Answers the reviewer's objections with exploratory analyses of the stored results.
- Runs one new pre-registered experiment: brief fine-tuning of the magnitude-pruned copies.
- Moves detail into an appendix for reviewers.
- Ends with a fresh-eyes review and the full set of checks.

It never submits, uploads, pushes or e-mails anything.

**For the author — how to run it.** You need the Claude Code CLI signed in, and MacTeX in `/Library/TeX/texbin`. In a terminal (the app's terminal or Terminal.app):

```
cd ~/Downloads/PDM/"AIR 2027"
chmod +x go_rev.zsh py revision/boundary.zsh
./go_rev.zsh R0                  # once, interactive: approve the folder, watch setup (about 20 min), then /exit
caffeinate -i ./go_rev.zsh auto  # everything else, unattended; safe to re-run after any stop
./go_rev.zsh status              # progress, from another terminal
```

While it runs, keep the Mac plugged in, and don't change anything under `~/Downloads/PDM`. A boundary check stops the run if any file outside `AIR 2027` changes.

If the run stops BLOCKED:
1. Read `revision/DECISIONS.md` and write your answer there.
2. Run `./go_rev.zsh R<n> resume`, then `caffeinate -i ./go_rev.zsh auto` again.

When the run finishes, start with `revision/REPORT.md`.

**How it works.** `go_rev.zsh` starts one headless Claude Code session per block (R0–R11). Each session gets its own model, effort level and turn limit, and receives only the common rules and its own block (not this preamble).

- **Opus 5.5** handles the high-value work: analysis design, the new study's registration and code, the paper text, the study's results, and the final review.
- **Sonnet 5.5** handles setup, the independent recheck, references, number plumbing, figure and appendix assembly, the checks and packaging.
- **Subagents** in `.claude/agents/` read only the files named in their brief and return capped reports. Their models are pinned in their files. The permissions file (`.claude/settings.json`) and the agent files ship in `revision/claude/`; the launcher copies them into `.claude/` on first run and never overwrites a copy you have edited.
- **The fine-tuning job** runs in the background while the text is revised: about 2–5 hours on this Mac, judging from PerconAI's fine-tuning runs.

| Session | Model | Effort | Max turns | Work |
|---|---|---|---|---|
| R0 | Sonnet 5.5 | medium | 80 | environment, git history, smoke test |
| R1 | Opus 5.5 | high | 100 | exploratory re-analyses of stored results |
| R2 | Sonnet 5.5 | medium | 60 | independent recheck of R1 |
| R3 | Opus 5.5 | high | 150 | register study FT; code, tests, probe, job |
| job R3 | — | — | — | fine-tuning and scoring, in the background |
| R4 | Sonnet 5.5 | medium | 80 | references and verified quotes |
| R5 | Sonnet 5.5 | high | 100 | numbers for the revision |
| R6 | Opus 5.5 | high | 150 | paper text |
| R7 | Sonnet 5.5 | medium | 100 | figure, appendix, build |
| R8 | Sonnet 5.5 | high | 100 | claims, references, checks |
| R9 | Opus 5.5 | high | 100 | study FT results into the paper (waits for job R3) |
| R10 | Opus 5.5 | high | 120 | fresh-eyes review and fixes |
| R11 | Sonnet 5.5 | medium | 80 | final checks, supplement, report |

| Subagent | Model | Used by | Reads |
|---|---|---|---|
| air-reader | Sonnet 5.5 | R3, R5, R6 | the files named in its brief (read-only) |
| air-stats-reviewer | Opus 5.5 | R1, R3 | a specification and the code or registration |
| air-bib-checker | Sonnet 5.5 | R4 | the web records of the references |
| air-claim-checker | Sonnet 5.5 | R8, R10 | paper text, key list, result files |
| air-referee | Opus 5.5 | R10 | the PDF's text only |

<!-- BEGIN COMMON -->
## Common rules (every session)

### C1 What this round is
The paper in `paper/` is being revised in response to an internal review. Sessions R0–R11 do the work in order; yours is the block after these rules. Earlier sessions' outcomes are in `revision/LOG.md`, `revision/CHECKPOINT.md` and `revision/notes/`. The review itself (`revision/REVIEW.md`) is read only by R10; your block already contains what you need from it.

### C2 Running unattended
- Never ask the user anything. When this plan does not settle a choice, take the more conservative option (fewer or weaker claims, no new data), record it in `revision/DECISIONS.md` under your session id, and go on. Stop BLOCKED only when going on would break a rule below or a registered design.
- Never start `claude`, `go_rev.zsh` or a job yourself. Anything that would run longer than about 10 minutes goes into `revision/jobs/<your id>.zsh`, which the launcher runs after your session.
- On a resumed attempt, first read `revision/CHECKPOINT.md`, `tail -20 revision/LOG.md`, your notes in `revision/notes/`, and `git log --oneline -15`. Continue from the first unfinished step and never redo a finished one.
- Before ending, with any status: commit your work (C7), append your status line to `revision/LOG.md`, and overwrite `revision/CHECKPOINT.md` (C9).

### C3 The boundary
- You may read anything under `~/Downloads/PDM`. Create, change or delete files only inside this folder (`AIR 2027`). The launcher compares the rest of `~/Downloads/PDM` before and after each session, and stops the run if anything changed.
- Never write into the sibling folders (`../rwm_repro`, `../robotic_world_model_lite`, `../rsl_rl_rwm`, `../PerconAI`, `../NGCAPS`, `../ICMAI`, `../ICPAIAS` and the rest), the virtual environments, or `../*.md`. Never run their scripts or a `main()`. Instead, import functions (`./py` puts `scripts/` on the path, and `common.py` adds `../rwm_repro/src`), or copy a file into `scripts/` with a header line giving its source path and sha256.
- `prereg/PREREG.md` is never edited, copied over, touched, re-permissioned or checked out. It is read-only, and its file time is evidence. Once R3 commits `prereg/PREREG_FT.md` and `scripts/ft_stats.py`, the same holds for them.
- No sudo. No external actions: never push, upload, e-mail, submit or post. Use the web only to verify references (R4, through air-bib-checker). Install nothing beyond what R0 allows.

### C4 Integrity
- No result is typed into `paper/*.template.tex`. Every value comes from a result file through `scripts/paper_numbers.py` (with `air_numbers.py` and, new in this round, `revision_numbers.py`), and `build_paper.py` refuses to build when a `{{key}}` has no value. Section, figure, table and reference numbers, citation years, and structural facts of the model (two GRU layers, a 45-number state) are not results.
- Registered verdicts stand as registered. Everything computed in this round is exploratory, except study FT, which R3 registers before its data exist. Label exploratory results "(exploratory)" everywhere they appear: abstract, text, tables, captions and appendix.
- Problems found after a registration go to `ANOMALIES.md`: the date, what happened, its effect, and what was done. Result files are written only by scripts, never by hand.
- Resample whole trajectories, never seed × trajectory; seeds are averaged within a trajectory first. F18 has 4 independent held-out trajectories and X5 has 20.
- The stored results were computed in a Linux cloud workspace (Python 3.13, torch 2.15, numpy 2.5). Never recompute them into `results/`: do not run `compute.py`, `checks.py` or `stats.py` there (R0's smoke test runs them on a copy). New measurements are made on this machine, recorded with its versions, and compared with stored values only where a block says so.
- Never edit a shell script while it runs.

### C5 Environment
- Python: always `./py`. It runs the PDM interpreter `../.venv-rwm311` (Python 3.11, torch 2.2.2) with `PDM_ROOT`, `PYTHONPATH`, no bytecode, and LaTeX on `PATH`. Never use `python3`, and never `pip install` into `../.venv-rwm311`.
- Build both PDFs with `./py scripts/build_paper.py`. Training and scoring code calls `torch.set_num_threads(2)`.
- What R0 found about this machine is in `revision/notes/R0_env.md`.

### C6 Context and token discipline
- Read only what your block names. Grep first, read at most about 200 lines at a time, print JSON fields with `./py -c`, and print at most about 40 lines from any command (pipe through `head` or `tail`).
- Never read these whole: `results/paper_numbers.json`, any `results/raw_*.npz`, `history.bundle`, logs (`tail` only), `../rwm_repro/PAPER*.md`, `../rwm_repro/RESULTS.md`, `../rwm_repro/FINDINGS_LEDGER.md`, `../PerconAI/PLAN*.md`, `../PerconAI/SESSION_LOG.md`, `../PerconAI/LEDGER.md`, `../PerconAI/DECISIONS.md`, any JSON over 100 KB, or `revision/REVIEW.md` (R10 excepted).
- Use a subagent only where your block names it, from `.claude/agents/`. Never pass a `model` parameter on an Agent call; the agent files pin the models. Give each subagent a self-contained brief: the exact files, the question, and what to return. Ask for no more than you need.
- Don't re-read a file you have already read in this session unless it has changed.

### C7 Git
- Work on branch `revision-1` (R0 creates it). Commit at the end of your session and wherever your block says "commit", with the subject "R<n>: <what>".
- Stage only the paths your session changed, with `git add <paths>`. Never use `git add -A`, because job R3 may be writing files.
- Never push, checkout, restore, reset --hard, stash, clean or rebase, or rewrite history in any other way. Never commit `runs/`, `.cache/`, `.pylib/`, `paper/build/`, `submission/` or `history.bundle`.

### C8 Style
- The paper's readers are roboticists. Use plain language and define every unavoidable term at first use. One idea per sentence, with at most two numbers-with-intervals in a sentence.
- State failures as plainly as successes.
- Never claim more than the numbers show. Name the scope (one robot, one trotting gait, simulated data, recorded commands, no policy trained) wherever a claim could be read more widely.
- Notes in `revision/notes/` are at most 300 words unless your block says otherwise.

### C9 Ending a session
- Append to `revision/LOG.md` the line `## R<n> status: COMPLETE (YYYY-MM-DD HH:MM)`, or PARTIAL or BLOCKED in place of COMPLETE. Then add at most 8 lines: what was done, files written, commit hashes and open points. The launcher reads this line, so write it exactly in that form.
- Overwrite `revision/CHECKPOINT.md` with: the last session and its status, the next action, and any blockers.
- BLOCKED: add to `revision/DECISIONS.md` what is blocked, the options, and your recommendation.
- PARTIAL (out of turns or context): commit what is done, and write the exact next step in `revision/CHECKPOINT.md`.
<!-- END COMMON -->

<!-- BEGIN R0 -->
## R0 — Setup, environment and baseline (Sonnet 5.5, medium)

**Goal:** a recorded, working environment on this Mac; the git history restored; and proof that the stored results and the paper rebuild identically here.

**Read:** nothing beyond this block. List folders with `ls`.

**Steps**
1. **Protect the registration.**
   1. Record `stat -f '%m %z' prereg/PREREG.md`.
   2. Run `chmod a-w prereg/PREREG.md`.
   3. Confirm the mtime is unchanged.
   4. Note both values in `revision/notes/R0_env.md`.
2. **Environment.** With `./py`, print the versions of Python, torch, numpy, scipy (or "missing") and matplotlib. Also print `sysctl -n machdep.cpu.brand_string hw.ncpu hw.memsize`, `sw_vers` and `df -h .`. Then check which tools the scripts' subprocesses will find: `./py -c "import shutil;[print(t, shutil.which(t)) for t in ('pdflatex','bibtex','pdftotext','pdfinfo','pdffonts','brew')]"`.
   - **scipy missing:** run `../.venv-rwm311/bin/python -m pip install --target .pylib "scipy>=1.11,<1.14"`. This installs beside this folder, not into the environment. Confirm with `./py -c "import scipy"`.
   - **pdftotext, pdfinfo or pdffonts missing:** if `brew` exists, run `brew install poppler`. Otherwise install `pypdf` into `.pylib` the same way, and give `scripts/submission_checks.py` a fallback that uses pypdf for text, page count and embedded fonts only when the poppler tools are absent (behaviour unchanged when they are present). Record which route was taken.
   - **pdflatex or bibtex missing:** stop BLOCKED; the author must install MacTeX.
3. **Git.** This folder has `history.bundle` but no `.git`. Restore the history without touching any working file:
   ```
   git init -q
   git bundle verify history.bundle
   git bundle list-heads history.bundle
   git fetch -q history.bundle "+refs/heads/*:refs/remotes/bundle/*"
   git symbolic-ref HEAD refs/heads/<branch named by list-heads>
   git update-ref refs/heads/<branch> refs/remotes/bundle/<branch>
   git reset -q                    # index only; working files untouched
   git status --porcelain | head -40
   ```
   - Record in `R0_env.md` any differences between the working files and the last commit; none are expected beyond untracked files. Never resolve a difference by checking a file out.
   - If git has no `user.name` or `user.email` here, set neutral repo-local values: `AIR 2027 author` and `author@localhost`.
   - Tag the restored commit `air-pre-revision`, then run `git switch -c revision-1`.
   - Add to `.gitignore`: `runs/`, `.cache/`, `.pylib/`, `submission/`, `history.bundle`, `.DS_Store`.
   - Commit "R0: setup" with `.gitignore`, `py`, `go_rev.zsh`, `REVISION_PLAN.md`, `revision/` (which includes `revision/claude/`) and `.claude/settings.json` and `.claude/agents/`.
4. **Smoke test on a copy, never in place.**
   1. Make the copy: `rm -rf .cache/smoke && mkdir -p .cache/smoke && cp -R scripts paper prereg figures results .cache/smoke/`.
   2. Run `./py .cache/smoke/scripts/<s>.py` for s = stats, check_numbers, exploratory, make_figures, paper_numbers, build_paper, check_claims, check_bib, submission_checks, in that order. Each writes inside `.cache/smoke`.
   3. Write `scripts/smoke_compare.py`. It compares JSON files (numbers equal within 1e-9 relative, strings exactly; for `paper_numbers.json`, every `value`) and both PDFs (page counts, and pdftotext text with whitespace normalised).
   4. Run it on every JSON in `results/` that the smoke run rewrote, and on both PDFs. Expected: identical.

   Handle what the comparison shows:
   - **A script fails for an environment reason** (an API or version problem): fix it minimally in `scripts/`, changing no computed value. Rerun the whole smoke test from a fresh copy, and record the fix in `ANOMALIES.md` under "Environment (revision round 1)".
   - **Only last digits differ** (≤ 1e-9 relative): record it and continue.
   - **A larger difference in any number the paper prints:** stop BLOCKED with the list.
   - **Page counts differ** from the stored `results/build.json` (10 pages single-column, 6 two-column): record it. R6 must meet the six-page limit on this machine's build.
5. **Revision files.**
   - Create `revision/LOG.md` (a title line, if absent), `revision/CHECKPOINT.md`, and `revision/DECISIONS.md` (a title line).
   - Write `revision/notes/R0_env.md` (≤ 300 words): versions, tools and their paths, what was installed, the git result, the smoke test result and the page counts.
6. Commit "R0: environment and smoke test", with `scripts/smoke_compare.py`, any fixed scripts, `ANOMALIES.md` and `revision/`.

**Done when:**
- `R0_env.md` is written.
- HEAD is on `revision-1`, with the tag `air-pre-revision`.
- The smoke test is identical, or has only recorded last-digit differences.
- `PREREG.md` is read-only, with its mtime unchanged.
<!-- END R0 -->

<!-- BEGIN R1 -->
## R1 — Exploratory re-analyses from the stored results (Opus 5.5, high)

**Goal:** one script, `scripts/revision_analyses.py`, that answers three reviewer objections from the stored data. It writes `results/revision_exploratory.json` and a summary of at most 60 lines in `results/revision_exploratory.txt`. Nothing here is registered; everything is exploratory.

**Read:**
- `scripts/stats.py` (whole).
- From `scripts/common.py`, only the definitions of `arena`, `ci`, `FOLDS` and `N_BOOT`.
- `per_model` in `scripts/compute.py`, to learn what each stored array holds.
- `results/model_table.json` and `results/stats.json` through `./py -c` only.
- The raw `.npz` files through code only.

**Stored arrays** for a model key k, per held-out trajectory:
- `E1_k`, `E8_k`, `E24_k`, `E100_k`: errors at those horizons.
- `err_k`: trajectory × 368 forecast steps.
- `log1_k`: trajectory × 367; the per-step log stretch of the leading direction.
- `rJ_k`: trajectory × 368.
- `lam_k`: trajectory × 4, the first four exponents. It is **not** λ1, so don't use it as λ1.
- Floors: `floor_*_<fold>`.

The registered λ1 is the mean over trajectories of `sum(log1_k[t]) / (367 × 0.02 s)`. This reproduces `model_table.json`'s `lam1` exactly; assert it.

**Reuse, never copy.** Import from `stats.py`: `spearman`, `boot_rho`, `boot_diff`, `boot_diff_within`, the pools, and the per-split averaging. Before computing anything new, assert that the registered numbers this script recomputes equal `results/stats.json` exactly: Table 2's correlations, `Q1`, `Q2`, `Q3_vs_E1`, `Q3_vs_E8`, `Q4_vs_E1`, `Q4_vs_E8`, `Q5_vs_E1`, `Q5_vs_E8`.

**A1 Horizon-matched label-free checks.**
- For k in {8, 24, 100, 368}, define:
  - λ1(k) = mean over trajectories of the sum of the first min(k, 367) entries of `log1`, divided by min(k, 367) × 0.02 s.
  - r̄J(k) = mean over trajectories of the mean of `rJ` over forecast steps 1..k.
- Compute the Spearman correlation of λ1(k), r̄J(k), E1 and E8 with each target E24, E100 and E368:
  - on F18: all 97 models, the 57 trained models, and the 40 compressed copies;
  - on X5: within each split, then averaged over the five splits.
- 95% intervals as the registered analysis computes them: models resampled, within splits on X5.
- Also compute the differences r̄J(8) − E8 and r̄J(100) − E8, both against E100, with intervals.

**A2 Wider intervals.**
- **(a) Two-level bootstrap.** Each draw resamples clusters and trajectories together.
  - A cluster is a parent with its eight compressed copies. Every other model is its own cluster.
  - On X5, clusters and trajectories are resampled within each split.
  - In a draw, a model's E_h, λ1 and r̄J are means over the drawn trajectories, with λ1 per trajectory from `log1` as above.
  - Recompute Table 2's correlations and the registered differences in every draw.
  - Use 20,000 draws with `numpy.random.default_rng(0)` if the run takes under about 8 minutes, else 5,000; record which.
  - Report the percentile intervals beside the registered ones, with the ratio of widths. For each registered verdict, say whether its interval under this scheme would still exclude zero. This is descriptive only; the registered verdict stands.
- **(b) X5 without split {1, 8}.** Recompute Table 2's X5 column and Q3's differences averaged over the other four splits, with the registered resampling.
- **(c) Holm correction.** The family is the secondary registered tests: `Q3_vs_E1`, `Q3_vs_E8`, `Q4_vs_E1`, `Q4_vs_E8`, `Q5_vs_E1`, `Q5_vs_E8` and `Q6`.
  1. Reproduce the registered bootstrap draws (same generator and the same order as `stats.py`), and assert that their intervals equal `stats.json`.
  2. Take the two-sided p = 2·min(P(d ≤ 0), P(d ≥ 0)).
  3. Apply Holm at family-wise 5%, and list which verdicts survive.

**A3 Split-half.** On each arena, each held-out episode contributes two trajectories; confirm this with `common.arena(fold)["episode"]`.
- Assignment 1: the checks (E1, E8, λ1, r̄J) come from the first trajectory of each episode, and the target E100 from the second. Assignment 2 swaps them.
- Compute Spearman for each assignment and average the two.
- Report F18 (all, trained, compressed) and X5 (within-split average), with model-resampled 95% intervals.

**Not needed:** the per-variant growth-rate shifts already exist in `results/stats.json`, under `lam_shift_X5` (registered output, descriptive). Don't recompute them.

Keep the whole run under about 10 minutes: vectorise the draws, and build the per-trajectory arrays once.

**Steps**
1. Write the script and run it; the asserts must pass.
2. Get an independent review:
   1. Save the specification above (A1–A3, verbatim) to `revision/notes/R1_spec.md`.
   2. Ask air-stats-reviewer to review `scripts/revision_analyses.py` against `revision/notes/R1_spec.md` and `scripts/stats.py`.
   3. Fix every MUST FIX and SHOULD FIX, then rerun.
   4. Record any NOTE you did not act on, with the reason, in `revision/notes/R1.md`.
3. Write `revision/notes/R1.md` (≤ 300 words): the headline numbers with their JSON paths, the number of draws used, and any departures from this block. Commit "R1: exploratory re-analyses".

**Done when:** the JSON and text files exist, the registered numbers reproduce exactly, and every MUST FIX is fixed.
<!-- END R1 -->

<!-- BEGIN R2 -->
## R2 — Independent recheck of R1 (Sonnet 5.5, medium)

**Goal:** a second implementation of R1's point estimates that shares no code with it.

**Read:**
- `revision/notes/R1_spec.md`.
- The keys of `results/revision_exploratory.json`, printed to depth 3.
- From `scripts/common.py`, only the data helpers `arena`, `FOLDS` and `fid`.
- Do **not** read `scripts/revision_analyses.py` or `scripts/stats.py`.

**Steps**
1. Write `scripts/revision_recheck.py`, using only numpy, json and `scipy.stats.spearmanr` (average ranks). Load `results/model_table.json` and the raw `.npz` files directly. Take pool membership from `model_table.json`'s fields (`pool`, `group`, `variant`, `extra.fold`). Recompute:
   - every A1 point estimate;
   - every A3 point estimate;
   - A2(b)'s correlations for X5 without {1, 8}.
2. Compare with `results/revision_exploratory.json`. If all agree within 1e-9, write `results/revision_recheck.json` with `{"all_agree": true, ...}`. Intervals are not rechecked.
3. On any disagreement, find whose definition departs from `R1_spec.md`.
   - If it is the recheck's, fix it.
   - If it is R1's, stop BLOCKED, with the details in `revision/DECISIONS.md`; R1 must be resumed to fix it.
4. Write `revision/notes/R2.md` and commit "R2: independent recheck".
<!-- END R2 -->

<!-- BEGIN R3 -->
## R3 — Register and prepare study FT: fine-tuning the magnitude-pruned copies (Opus 5.5, high)

**Why:** Table 1 shows that pruning 30% of weights without retraining keeps most of the skill one step ahead and much less at 2 s. Every reviewer will ask whether a short fine-tuning repairs this. An earlier registered study of these models (`../PerconAI`, rounds 1–3) answered the question for unit removal (widths 192 and 128), but not for magnitude pruning.

This session writes the code, registers the study and prepares the job. It must not train or score any fine-tuned model.

**Read:**
- `../PerconAI/scripts/e3_finetune.py` (whole): the protocol to copy.
- `scripts/compress.py`: `unstructured`.
- `scripts/compute.py`: `build`, `per_model`, `setup_arena`.
- The D2 block of `scripts/stats.py` (grep `# D2:`): the estimator of K.
- `scripts/common.py`: `load`, `fold_dir`, `run_path`.

Ask **air-reader** for:
- (a) the exact text, thresholds and labels of PerconAI's Q3 rule, and its definition of K against the control, from `../PerconAI/PREREG_R2.md` and `../PerconAI/PREREG_R3.md`;
- (b) the `hyperparameters`, `s_per_iter` and `wall_clock_s` fields of the 15 control records, `../PerconAI/results/raw/e3_ft/e3ft_<rid>_w256.json`.

Check for yourself that all 15 control weights exist: `ls ../PerconAI/runs/e3_ft/*_w256/weights_500.pt`. Their checkpoint fields are those that `e3_finetune.py` saves. The source-run records that the protocol asserts against are the `hyperparameters` field of `../rwm_repro/results/step5_<rid>.json` and `x3_run_*.json`, read by `e3_finetune.py`'s `source_record`. If any control is missing, FT2 uses only the parents that have one, and the registration says so before the commit.

**Design to register** in `prereg/PREREG_FT.md`. Write it in plain language and fill the bracketed items.
1. **Question.** Does a short fine-tuning, with the pruned weights held at zero, restore the long-horizon skill that magnitude pruning removes?
2. **Models.**
   - Folds: the five X5 folds.
   - Parents P: `rid(f, s)` for seeds s ∈ {0, 1, 2}, which is 15 models. These seeds are used because PerconAI's fine-tuned controls exist for them.
   - Arms, per parent:
     - U30 = `compress.unstructured(P, 0.30)`, not trained.
     - U30F = U30 fine-tuned for 500 iterations, with the mask held.
     - U50 and U50F: the same at 0.50. These are secondary; see item 8.
     - CTRL = PerconAI's `e3ft_<rid>_w256`: P fine-tuned for 500 iterations by the same protocol. Read-only.
3. **Fine-tuning protocol:** that of `../PerconAI/scripts/e3_finetune.py`, unchanged except that training starts from the pruned copy, and every pruned entry is set to exactly zero after every optimiser step. In full:
   - rollout loss (`teacher_forcing=False`, loss `"mse"`);
   - the fold's training episodes, in 40-row windows;
   - batch 256;
   - Adam with fresh state, at the reference learning rate and weight decay;
   - no gradient clipping;
   - the parent's seed for torch, numpy and the batch generator;
   - 2 threads;
   - settings asserted equal to the source run's record before training.
4. **Scoring:** this paper's measures, computed by `compute.per_model` on each parent's own fold arena.
   - E1, E8, E24 and E100 (windows from every fourth position after 32 true history steps; E1 at every position), E368, λ1 and r̄J.
   - The floor is holding the last state.
   - All arms, P and CTRL included, are scored by the same code in one run on this machine.
5. **Unit and estimator.**
   - The unit is the trajectory: the 20 X5 trajectories. For each arm and trajectory, the error is first averaged over the three seeds.
   - K_orig(x, h) = 1 − mean_t(e_x − e_P) / mean_t(e_floor − e_P), exactly the estimator of `stats.py`'s D2. K_ctrl(x, h) is the same with CTRL in place of P.
   - Intervals: 20,000 draws of the 20 trajectories with replacement (whole trajectories), using `numpy.random.default_rng(0).integers(0, 20, size=(20000, 20))`, and percentile intervals.
6. **Questions and rules.**
   - **Primary**, a family of two with 97.5% intervals:
     - **FT1:** d = K_orig(U30F, 100) − K_orig(U30, 100). If the interval lies above 0: FINE-TUNING RESTORES SKILL. If below 0: FINE-TUNING REMOVES SKILL. Otherwise: NOT SETTLED.
     - **FT2:** K_ctrl(U30F, 100), with PerconAI Q3's thresholds and labels copied verbatim from (a).
   - **Secondary**, at 95%, not corrected for multiplicity, and reported as secondary:
     - FT1 and FT2 at h = 24 and h = 368.
     - FT1 and FT2 for U50 at h = 100, if U50F runs.
     - K_orig(U30F, 1) beside K_orig(U30F, 100): does the one-step reading still overstate the skill?
     - The mean change in λ1, for U30F − P and for U30 − P, and the same for r̄J (trajectory bootstrap, seeds averaged).
7. **Checks before the analysis.** A failure is reported, and the affected arm is excluded from the questions it touches.
   - **C-FT1:** every pruned entry is exactly zero after fine-tuning, and the count of zeros equals the count pruned.
   - **C-FT2:** every loss trace is finite.
   - **C-FT3:** the settings equal the source run's record.
   - **C-FT4:** this machine's E100 for P and U30 agrees with the stored X5 values (`results/raw_X5.npz`, computed in the cloud) within 1e-3 relative. A larger difference is reported as an anomaly, and every question still uses this machine's scores for every arm.
8. **Time rule**, fixed by the probe before any fine-tuned model exists. U30F always runs. U50F runs only if the projected total (U30F, plus U50F, plus scoring) is at most 8 hours on this machine. [Fill in the probe's numbers and the arms that result.]
9. **What was known when this was written:**
   - Table 1's K for U30 without fine-tuning, on X5 seeds 0–4, at 1, 24, 100 and 368 steps [values and `results/stats.json` paths].
   - PerconAI's registered results: fine-tuned width-192 copies keep most of an equally fine-tuned control's skill; width 128 is not settled; and fine-tuning alone improved the control [values and file paths].
   - No fine-tuned magnitude-pruned copy existed, and none had been scored.
10. **Files and outputs:** list them. After the commit, neither this file nor `scripts/ft_stats.py` changes. A needed change goes to `ANOMALIES.md` with a new commit, and the paper says so.

**Steps**
1. **Write `scripts/ft_unstructured.py`.**
   - It is a copy of `../PerconAI/scripts/e3_finetune.py`, with a header giving the source path and sha256.
   - Replace PerconAI's env and harness helpers with their equivalents in `common.py` and `compress.py`.
   - Start from `compress.unstructured`, hold the mask after every `opt.step()`, and assert the final count of zeros.
   - Outputs: `runs/ft/<id>/weights_500.pt` and `results/raw/ft/<id>.json`. The JSON has PerconAI's fields plus `mask_zeros`, `pruned_entries` and `level`.
   - Run ids are `ft_u30_<rid>` and `ft_u50_<rid>`.
   - Add a `--probe N` mode. It trains N iterations on fold 1-8, seed 0, at 0.30, without saving or scoring the result. It times one `per_model` scoring of the parent, and writes `results/ft_probe.json` (seconds per iteration, and the projected hours for each arm and for scoring).
2. **Write `scripts/ft_eval.py`.**
   - It scores P, U30, U30F, U50, U50F (those present) and CTRL for the 15 parents, using `compute.per_model` on each fold's arena.
   - It writes `results/raw_FT.npz` and `results/ft_models.json`: per model, the arm, parent, fold, seed, E_h, E368, λ1, r̄J, a finite flag and the software versions.
   - It also runs C-FT4.
   - It is resumable: it skips models already scored, and writes `results/ft/progress.json` after each model.
3. **Write `scripts/ft_stats.py`.** It implements items 5–7 of the design exactly, and refuses to run if a check result is missing. It writes `results/ft_stats.json` and a summary of at most 60 lines in `results/ft_report.txt`.
4. **Write `scripts/test_ft.py` and run it.** It must pass, take at most 10 minutes, and save nothing outside `.cache/`. It tests that:
   - (i) three iterations on fold 1-8, seed 0, keep every pruned entry at zero;
   - (ii) two three-iteration runs give identical weights;
   - (iii) the settings assertion passes for all 15 parents (no training);
   - (iv) `ft_eval`'s scoring of one parent and its U30 reproduces the E100 in `raw_X5.npz` within 1e-3 relative;
   - (v) `ft_stats`'s K estimator, applied to the stored X5 values of U30 for seeds 0–4, reproduces `D2_skill_retained_X5.u30` in `results/stats.json` at every horizon to 1e-12;
   - (vi) `ft_stats`'s rules return the right labels on constructed inputs: intervals above, below and straddling each threshold.
5. **Run the probe:** `./py scripts/ft_unstructured.py --probe 20`. Fill item 8 of the registration.
6. **Write `prereg/PREREG_FT.md`** from the design above. Ask air-stats-reviewer to review it, together with `scripts/ft_stats.py`, `scripts/ft_unstructured.py` and `scripts/test_ft.py`, against this block's design. Fix every MUST FIX and SHOULD FIX, then rerun the tests.
7. **Register.**
   1. Commit `prereg/PREREG_FT.md`, the four scripts and `results/ft_probe.json` together, with the subject "R3: PRE-REGISTER study FT (no fine-tuned model exists yet)".
   2. Then run `chmod a-w prereg/PREREG_FT.md scripts/ft_stats.py`.
   3. Record the commit hash and time in `revision/notes/R3.md`.
8. **Write `revision/jobs/R3.zsh`.** The launcher runs it in the background after this session. It must:
   - `cd` to this folder.
   - For each queued run (U30F for all 15 parents, then U50F if item 8 says so): skip it if `results/raw/ft/<id>.json` exists; otherwise run `./py scripts/ft_unstructured.py <id> <a-b> <seed> <level> 500`.
   - Exit non-zero on the first failure, and print one line per run.
   - Finally, run `./py scripts/ft_eval.py`.
   - Check its syntax with `zsh -n`.
9. Write `revision/notes/R3.md` (≤ 300 words) and commit "R3: job for study FT".

**Done when:** the tests pass; the registration is committed before any fine-tuned model exists; and the job file exists.
<!-- END R3 -->

<!-- BEGIN R4 -->
## R4 — References and verified quotes (Sonnet 5.5, medium)

**Goal:** verified records for every new reference and for each statement the revision makes about another paper, plus an updated `refs.bib` and `check_bib.py`.

**Read:**
- `paper/refs.bib` and `scripts/check_bib.py`.
- The Mikhaeil entry only (by grep) from `../NGCAPS/results/bib_verified.json` and `../NGCAPS/paper/refs.bib`.

**Steps**
1. **Verify the records.** Ask air-bib-checker to verify these records and quote the sentences asked for. Send them in two briefs: the first four, then the rest.
   - Lambert, Amos, Yadan, Calandra, "Objective Mismatch in Model-based Reinforcement Learning", L4DC 2020 (arXiv 2002.04523). Quote the abstract sentence(s) saying that a model's one-step likelihood or accuracy is a poor guide to downstream control performance.
   - Malik, Kuleshov, Song, Nemer, Seymour, Ermon, "Calibrated Model-Based Deep Reinforcement Learning", ICML 2019 (PMLR v97).
   - Kuleshov, Fenner, Ermon, "Accurate Uncertainties for Deep Learning Using Calibrated Regression", ICML 2018 (PMLR v80).
   - Manek, Kolter, "Learning Stable Deep Dynamics Models", NeurIPS 2019.
   - Hess, Monfared, Brenner, Durstewitz, "Generalized Teacher Forcing for Learning Chaotic Dynamics", ICML 2023 (PMLR v202).
   - Rigter, Lacerda, Hawes, "RAMBO-RL: Robust Adversarial Model-Based Offline Reinforcement Learning", NeurIPS 2022 (arXiv 2204.12581). Quote the sentences in Appendix B.1 on training an ensemble and keeping the best members by validation error on held-out transitions, and which works they follow.
   - Yu, Kumar, Rafailov, Rajeswaran, Levine, Finn, "COMBO: Conservative Offline Model-Based Policy Optimization", NeurIPS 2021.
   - Li, Krause, Hutter, "Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics" (arXiv 2501.10100). Confirm the venue: NeurIPS 2025 Workshop on Embodied World Models for Decision Making, poster (https://neurips.cc/virtual/2025/loc/san-diego/123991). From https://arxiv.org/html/2501.10100v4, quote (i) its comparison of autoregressive training with the teacher-forcing counterpart (Sec. 4.3), and (ii) what Appendix A.4.1 says about the forecast horizon N.
2. **Write `paper/bib_verified_air.json`:** `{"note": ..., "checked_on": <date>, "entries": {key: {title, authors, venue, year, url, summary, quotes: [{text, where, url}]}}}`, for VERIFIED entries only. An UNVERIFIED entry is not added to `refs.bib`; record it in `revision/DECISIONS.md` and go on.
3. **Update `paper/refs.bib`.**
   - Add BibTeX entries with the keys `lambert2020objective`, `malik2019calibrated`, `kuleshov2018accurate`, `manek2019stable`, `hess2023generalized`, `rigter2022rambo` and `yu2021combo`.
   - Copy NGCAPS's Mikhaeil entry under NGCAPS's key.
   - Change the RWM entry's note to the workshop poster.
   - Keep the style of the existing entries.
4. **Extend `scripts/check_bib.py`.**
   - Accept `paper/bib_verified_air.json` as a fourth source, under the same rule: the key is present there, and the title is equal after normalisation.
   - Add a flag `--allow-uncited` that reports uncited entries as warnings; R8 runs the check strict.
   - Look for citations in `paper/body.tex` and, when it exists, `paper/appendix.tex`.
5. **Finish.**
   - Run `./py scripts/check_bib.py --allow-uncited`; it must pass.
   - Write `revision/notes/R4.md`: the new keys, with one line each on what the text may say about them, taken from the quotes.
   - Commit "R4: references".
<!-- END R4 -->

<!-- BEGIN R5 -->
## R5 — Numbers for the revision (Sonnet 5.5, high)

**Goal:** every new number the revised paper needs is registered with its source, and R6 gets a short key list in place of `results/paper_numbers.json`.

**Read:**
- `scripts/paper_numbers.py` (whole), and the first 60 lines plus the formatting helpers of `scripts/air_numbers.py`, to copy their conventions.
- `results/revision_exploratory.json` and `results/stats.json` through `./py -c`.
- `revision/notes/R1.md` and `revision/notes/R4.md`.

**Steps**
1. **Create `scripts/revision_numbers.py`** with `add(put, ...)` in the style of `air_numbers.add`. Call it from `paper_numbers.py` (one import, one call, after `air_numbers`). Use the same formatting helpers, so that new numbers print like the old ones.
2. **Register these numbers,** each with its source file and field:
   - **a. A1:** the Spearman correlation with E100 of r̄J(100), r̄J(8), λ1(100) and λ1(8), on F18 (all) and X5, with intervals; and the two differences against E8.
   - **b. A2:**
     - the two-level intervals for Table 2's correlations and for the registered differences;
     - the median ratio of interval widths;
     - for each registered verdict, whether it would hold;
     - the X5 values without {1, 8};
     - the Holm outcome (which secondary verdicts survive).
   - **c. A3:** the split-half Spearman correlation of each check with E100 (F18 all, and X5).
   - **d. Growth-rate shifts per variant,** from `results/stats.json`'s `lam_shift_X5`:
     - the mean and 95% interval for u10, u20, u30, u50, s192 and s128;
     - `n_positive_lam1` for u50, s192 and s128;
     - totals computed in code (for example, the copies that became unstable out of all unit-removed copies).
   - **e. Scale of the released model's data.**
     - Find the key the template uses for the fraction of RWM's training data that the release contains: grep `body.template.tex` for "of what the original model was trained on".
     - Register the ratio 1/fraction, rounded to the nearest 10, printed as "about N".
   - **f. Stale command on the held-out pair at h = 100.** These are the released checkpoint's relative-L1 figures, given in the earlier study for comparison only. Ask air-reader to find the file and field in `../rwm_repro/results/` behind the value +22.5% [+6.3%, +41.7%]. That value is in the held-out-pair action table of `../rwm_repro/PAPER.md`: grep it for "+22.5", and the table's note names its JSON. Register the value and its interval.
   - **g. Training settings of the rollout-trained parents,** from the `hyperparameters` field of `../rwm_repro/results/step5_armA_seed0.json`: optimiser, learning rate, weight decay, batch size, iterations, the non-zero loss weights, and gradient clipping. Assert that they are equal across all X5 parents' records (`x3_run_*.json`), then register them once.
   - **h. What was known before the registration.** Ask air-reader for the files and keys behind the values that `prereg/PREREG.md` §2 quotes: NGCAPS's ρ of λ1 with the 368-step error over R and over R∪T, ICMAI's 0.66, and ICPAIAS's residual value of 0.93. Register them.
   - **i. PerconAI's unit-removal study.** Ask air-reader for the keys in `../PerconAI/results/paper_numbers.json` (it has a `keys` dict) behind these sentences of `../PerconAI/paper/main.tex`: the lines with "KEEPS MOST", "SCRATCH BETTER" and "108%". Register:
     - K of the fine-tuned width-192 and width-128 copies against the equally fine-tuned control, at 100 and 368 steps (first seed, and the replication or pooled reading, as its paper reports);
     - the Q3 and Q4 verdicts;
     - K of the control against the original at 368 steps.
   - **j. Counts:** `n_trained` and `n_compressed` already exist. Add more only if R6 will need them.
3. **Run `./py scripts/paper_numbers.py`;** it must succeed. Spot-check five new values against their sources by printing both.
4. **Write `revision/notes/R5_keys.md`:** one line per new key, in the form `key | printed value | meaning`, at most 150 lines and no other text. Commit "R5: numbers".
<!-- END R5 -->

<!-- BEGIN R6 -->
## R6 — Revise the paper text (Opus 5.5, high)

**Goal:** revise `paper/body.template.tex` so that every review point below is handled, with no overclaim, in plain language, within six two-column pages; and create the skeleton of a reviewers' appendix.

**Read:**
- `paper/body.template.tex` (whole).
- `revision/notes/R5_keys.md` (whole).
- `revision/notes/R1.md` and `revision/notes/R4.md`.
- `paper/bib_verified_air.json` (the quotes only).
- `scripts/build_paper.py`.
- For any other fact, ask air-reader. For example: why the width-124 models exist (`../rwm_repro/MODEL_CARD.md`, or the run name `armA_seed0_m49h124`).

**Rules for this session:**
- Numbers only as `{{key}}`. If a key you need doesn't exist, add it to `scripts/revision_numbers.py` reading its source file (as R5 did) and rerun `paper_numbers.py`. Never type the number.
- Keep every registered verdict and every failure in the text.
- In Table 2's registered rows, change nothing but labels and caption.

**Edits.** The ids are the review's traceability ids (see the table at the end of this file).

- **T1 Credit RWM [P1].** In the Introduction and Sec. 4.1, say that:
  - RWM trains on its own 8-step rollouts and reports multi-step error;
  - it found that autoregressive training beats its teacher-forcing counterpart, and that a longer forecast horizon helps. Cite [1] with the appendix, using the wording of R4's quotes.

  Present Table 1's first two rows as an independent confirmation on a small fraction of RWM's data. This paper adds the one-step view and the growth-rate explanation.
- **T2 Aim the critique [P1].** Replace "Yet such simulators are usually trained on one-step targets … and checked by their one-step error on held-out data" with an accurate statement:
  - many model-based methods train one-step models and keep the ensemble members with the lowest error on held-out single transitions (MOPO, COMBO, RAMBO; R4's quote);
  - compression and uncertainty are judged one step at a time [9–11];
  - Objective Mismatch, for the general gap between a model's accuracy and what the model is used for.
- **T3 Abstract [P2, P3, P5, P7i]**, at most 200 words, in this order:
  1. the problem;
  2. what was done: `{{n_trained}}` trained simulators, `{{n_compressed}}` compressed copies and the released model, with gradients matched;
  3. the findings, most robust first:
     - one-step checks reverse the training choices. Teacher forcing and one-step training are better one step ahead and several times worse at 2 s, consistent with RWM's own ablation;
     - pruning 30% of weights keeps most of the skill one step ahead and far less at 2 s, using the same keys as Sec. 4.1 (`{{K_u30_h1}}` and its 100-step counterpart). R9 adds the fine-tuning result here;
     - the ensemble's spread is too small, by a factor that grows with the horizon, because the implemented loss is lowest at zero spread;
     - in a registered comparison over `{{n_P}}` simulators, a kinematic-consistency residual ranked the 2-second error best. It was level with an 8-step check on a second pool (exploratory);
     - the growth rate beat one-step error, but failed its registered replication on a pool made mostly of compressed copies, because moderate pruning adds error without changing it;
  4. the scope: one robot, one trotting gait, simulated data, recorded commands replayed, no policy trained;
  5. the checklist.
- **T4 Contributions [P2]:** at most four bullets, re-centred on what held. Label each exploratory part, and drop "302".
- **T5 Registration wording [P2].** In the Introduction and Sec. 3, write "committed before this study's measurements were run". Add one sentence: the growth rates of the trained F18 models, and the residual's ranking of the five-split trained models, were known from earlier analyses (R5 h). The open questions were the comparison with one-step and 8-step errors, and the compressed copies (appendix A2).
- **T6 Overclaims [P3].**
  - **a.** Every statement that the residual matched or beat the other checks "on both pools" adds "(exploratory on the five-split pool)". This applies to the abstract, Introduction, Sec. 4.2, Discussion and Conclusion.
  - **b.** "Pruning adds bias rather than instability" (abstract, Sec. 4.1, Sec. 4.2, the Fig. 2 caption, Discussion item 3) becomes:
    - moderate pruning (10–30%) adds error without changing the growth rate;
    - heavier pruning and unit removal do raise it;
    - say how many unit-removed copies become unstable (R5 d).
  - **c.** In the stability bullet and Sec. 4.2: rollout training made every rebuilt model contracting. The released model, rollout-trained on about N times more data (R5 e), is not contracting on these trajectories, which it may have trained on. So the account is shown only for models trained on this data set.
  - **d.** Replace "without using any recorded future" (and "no recorded future") with "without recorded future states (the checks use the recorded commands)".
  - **e.** Stale command (Sec. 3): replace "a defect that only a short-horizon check reveals" with:
    - it raises the one-step error most;
    - at 100 steps, the change is not resolved on the 20 trajectories;
    - but it is resolved on the four held-out trajectories (R5 f).
- **T7 Check comparison [P4a, P4b, P4d].** Add two or three sentences to Sec. 4.2, each marked "(exploratory)":
  - with both label-free checks limited to the target's 100 steps, the residual still ranks well;
  - over 8 steps it is level with the 8-step error on F18, and below it on the five splits;
  - the growth rate needs the long rollout;
  - checks and target use the same trajectories, and every 100-step window begins with an 8-step window; computing them on different trajectories (split-half) gives the A3 result;
  - give the five-split result without split {1, 8}.
- **T8 Use case [P4c].** In Sec. 4.2 and Discussion item 3: the label-free checks were tested only with recorded commands. Whether they rank simulators when a policy supplies the commands is untested.
- **T9 Scope and baseline [P5].**
  - Add a scope sentence to the Introduction, not only to the Limitations.
  - In Sec. 4.1, one plain sentence: for position over 2 s, neither simulator beats assuming the robot keeps its last measured velocity. This baseline is strong here because the commanded velocity changes about once per episode.
  - In the Limitations: recorded commands are replayed, with no controller in the loop.
- **T10 Statistics [P6].** In Sec. 3:
  - intervals on correlations resample models, which ignores that compressed copies share a parent and that every model is scored on the same trajectories;
  - resampling parents and trajectories together widens them (R5 b; appendix A4); say whether this changes any verdict;
  - the secondary verdicts are at 95% without correction; say which survive Holm's correction.
- **T11 Table 1 [P7a, P7c]:**
  - a caption without a row count;
  - one error measure throughout;
  - remove the stale-command row (it is now in Sec. 3);
  - group the rows with midrules: training choices, compression, uncertainty;
  - keep the evidence column;
  - define K and the spread factor in plain words.
- **T12 The 5.0/5.85 mismatch [P7b].** Sec. 4.1 gives the 368-step ratio in this paper's error measure (the Table 1 value). State the registered test as "the registered rule found rollout training better on all ten held-out episodes". Mention the release's relative error only by name, if at all.
- **T13 Verdict labels [P7d].** No small-caps labels in running text; use plain words, for example "the registered rule favoured the label-free check". Table 2 keeps its labels, with a one-line legend in its caption.
- **T14 Density [P7d].**
  - At most two numbers-with-intervals per sentence.
  - A number that is in a table is not repeated in the text unless the sentence is about it.
  - Sec. 4.3 becomes at most three sentences. Move the removed text, unchanged, into `revision/notes/R6_moved.md` for appendix A6.
- **T15 Methods [P7e].** In Sec. 3, Models:
  - say what "teacher-forced" and "N = 1" each mean, one clause each;
  - say why width 124 was used, as its record states (from air-reader);
  - add one sentence of training settings, with keys (R5 g) and a pointer to appendix A1.
- **T16 Earlier analyses [P7g].** One sentence: the earlier registrations and code are in the anonymised supplementary material, and the commit times come from our own version control.
- **T17 Counts [P7i].** Replace "302 versions" and "302 simulators" with "`{{n_trained}}` trained simulators, `{{n_compressed}}` compressed copies and the released model".
- **T18 AI statement [P7j].** Write "the author(s)", which is neutral; otherwise keep the content.
- **T19 Related work [RW].** One clause each:
  - Objective Mismatch, in the compounding-error paragraph;
  - MOPO, COMBO and RAMBO keeping the lowest-error members, in the uncertainty paragraph;
  - Kuleshov and Malik on recalibration, in the uncertainty paragraph (the per-horizon multiplier in Sec. 4.1 is a recalibration);
  - Mikhaeil and Hess on teacher forcing, chaos and Lyapunov exponents in recurrent-network training, and Manek and Kolter on stable learned dynamics, in the stability paragraph.
- **T20 Unit removal [N1].** Replace "Removing units to width 192 fails at every horizon (…) and needs fine-tuning" with:
  - without fine-tuning, it fails;
  - in an earlier registered study, copies fine-tuned for 500 iterations kept most of an equally fine-tuned full-width model's skill at width 192 (three seeds), and were not settled at width 128;
  - narrow models trained from scratch did as well or better at 100 steps.

  Use the R5 i keys, with each statement exactly as strong as its verdict.
- **T21 Appendix skeleton [F5].** Create `paper/appendix.template.tex` with these sections:
  - A1 Training settings
  - A2 What was known before the registration
  - A3 Horizon-matched and split-half checks
  - A4 Wider intervals, the five-split pool without split {1, 8}, and Holm's correction
  - A5 Compression and the growth rate
  - A6 Run-time signals (moved from Sec. 4.3)
  - A7 Study FT: fine-tuning the pruned copies

  Under each heading, a LaTeX comment says what goes there. The body refers to them as "appendix A1" and so on.
- **T22 Length [F5].** Build with `./py scripts/build_paper.py`.
  - The two-column PDF must have at most 6 pages, references included; R7 adds the appendix to the single-column build only.
  - If it is over, cut in this order: Sec. 4.3; Related Work wording; the uncertainty paragraph (keep the zero-spread cause and the per-horizon result); repeated numbers.
  - Never cut a limitation, a failed result or a scope statement.
- **T23 Title.** Keep it. Put two alternative titles that name the actual findings in `revision/notes/R6.md`, for the author.

**Steps**
1. Make the edits.
2. Build until both PDFs build and the two-column one has at most 6 pages.
3. Write `revision/notes/R6.md`: what changed for each id, and anything not done, with the reason.
4. Commit "R6: paper text".
<!-- END R6 -->

<!-- BEGIN R7 -->
## R7 — Figure, appendix and build (Sonnet 5.5, medium)

**Read:**
- `fig_checks` in `scripts/make_figures.py`.
- `paper/appendix.template.tex`.
- `revision/notes/R5_keys.md`, `R6_moved.md` and `R6.md`.
- The §2 section only of `prereg/PREREG.md` (grep the heading, read that section).
- `scripts/build_paper.py`, `paper/main_single.tex` and `paper/main_twocol.tex`.

**Steps**
1. **Figure 2 gets three panels:** one-step error, growth rate and kinematic residual (x), each against the 100-step error (y).
   - Keep the same marker groups, the palette (#2a78d6, #eb6834, #1baf7a), the log axes and the floor line.
   - Read each panel's Spearman value from `results/stats.json`; never type it.
   - Regenerate the figure, and open it with Read to check that labels don't overlap.
   - Keep the figure-width macro, so it fits both builds.
2. **Fill `paper/appendix.template.tex`,** with numbers as `{{keys}}`. Add any missing keys to `scripts/revision_numbers.py`, as R5 did.
   - **A1:** a table of training settings (R5 g) for the rollout-trained, teacher-forced and N = 1 models, saying what differs (the loss target and the iterations).
   - **A2:** a short paragraph listing what `prereg/PREREG.md` §2 says was known, with keys (R5 h).
   - **A3:** a table of check × window (8, 24, 100 and 368 steps) × pool, giving the Spearman correlation with E100 (A1), plus the split-half row (A3). Put "(exploratory)" in the caption.
   - **A4:** Table 2's correlations and the registered differences with two-level intervals, beside the registered ones; the five-split pool without split {1, 8}; and the Holm outcome.
   - **A5:** the growth-rate shift, and the count of unstable copies, for every variant (R5 d).
   - **A6:** the text from `R6_moved.md`, edited only for flow, with numbers unchanged.
   - **A7:** keep the heading and its comment; R9 writes this section.
3. **Build integration.**
   - `build_paper.py` also fills `paper/appendix.template.tex` into `paper/appendix.tex`, refusing on a missing key as it does for the body.
   - `main_single.tex` inputs `appendix.tex` after the references (`\clearpage\appendix`); `main_twocol.tex` does not.
   - Build both. The single-column PDF has the appendix; the two-column one still has at most 6 pages.
4. Write `revision/notes/R7.md` and commit "R7: figure, appendix, build".
<!-- END R7 -->

<!-- BEGIN R8 -->
## R8 — Claims, references and checks (Sonnet 5.5, high)

**Read:**
- `scripts/check_claims.py` (whole).
- `git diff air-pre-revision -- paper/body.template.tex`, in parts of at most 200 lines.
- `paper/appendix.template.tex` and `revision/notes/R5_keys.md`.

**Steps**
1. **Encode the claims.**
   - For every sentence added or changed since `air-pre-revision` that states an order, a sign, a direction, a comparison with a threshold or a verdict, add an assertion to `scripts/check_claims.py`. It reads the same files the numbers come from, in the existing CLAIMS style.
   - Remove assertions whose sentences no longer exist.
   - List each sentence and its assertion in `revision/notes/R8_claims.md`, one line each.
2. **Run, in order:**
   - `./py scripts/paper_numbers.py`
   - `build_paper.py`
   - `check_claims.py`
   - `check_bib.py`, strict: every entry is cited in `body.tex` or `appendix.tex`
   - `submission_checks.py`: anonymity of both PDFs and their metadata, embedded fonts, and at most 6 pages in two columns

   All must pass.
3. **Check against the evidence.** Ask air-claim-checker to check `paper/body.tex` and `paper/appendix.tex` against `revision/notes/R5_keys.md`, `results/stats.json`, `results/revision_exploratory.json` and `prereg/PREREG.md`. For each finding:
   - a wrong key or number: fix the key;
   - a sentence wider than its evidence: narrow the wording minimally, and never strengthen it.

   Log every change in `revision/notes/R8.md`, for R10. Then rerun step 2.
4. Commit "R8: claims and checks".
<!-- END R8 -->

<!-- BEGIN R9 -->
## R9 — Study FT: results into the paper (Opus 5.5, high)

The launcher starts this session only after job R3 has finished.

**Read:**
- `prereg/PREREG_FT.md` (whole) and `results/ft_report.txt`.
- `results/ft_stats.json` and `results/ft_models.json` through `./py -c`.
- `tail -30 .cache/logs/job_R3.log`.
- In `paper/body.template.tex`: the compression paragraph of Sec. 4.1, Table 1, the abstract, and Discussion items 2–3.
- Section A7 of `paper/appendix.template.tex`.

**Steps**
1. **Confirm the job finished.** Every queued run has `results/raw/ft/<id>.json` with finite traces, `ft_eval` wrote `results/ft_models.json`, and checks C-FT1 to C-FT4 are recorded. Handle a failed or missing run as `PREREG_FT.md` §7 says, and record it in `ANOMALIES.md`.
2. **Run `./py scripts/ft_stats.py`.** It is registered; do not edit it. Its verdicts are the study's result, whatever they are.
3. **Register the numbers** in `scripts/revision_numbers.py`:
   - the verdicts;
   - the K values with intervals at 1, 24, 100 and 368 steps, for U30 and U30F (and for U50 and U50F, if they ran);
   - K against the control;
   - the changes in λ1 and r̄J;
   - the wall-clock time per run;
   - the registration commit time, from git.
4. **Text.**
   - **Sec. 4.1, compression paragraph:** one or two sentences with the two registered verdicts, in plain words.
   - **Table 1:** add the row "Pruning 30%, then fine-tuned for 500 iterations": K against the original parent at each horizon, with the evidence "15 + 15 on X5, 20".
   - **Abstract:** add one clause on pruning. If the result is NOT SETTLED, say so.
   - **Discussion item 2 or 3:** update it if the result changes the advice.
   - **Appendix A7:** the design in five sentences, the outcome of the time rule, the checks, a table of all registered and secondary results, and what was known before. Label the secondary results as secondary.
5. **Check.** Add claim assertions for the new sentences to `scripts/check_claims.py`, and run the pipeline of R8 step 2. The two-column PDF must still have at most 6 pages; if needed, cut in R6's T22 order.
6. Write `revision/notes/R9.md`, and commit "R9: study FT results". Include `results/raw/ft/*.json`, `results/raw_FT.npz`, `ft_models.json`, `ft_stats.json` and `ft_report.txt`, but not `runs/`.
<!-- END R9 -->

<!-- BEGIN R10 -->
## R10 — Fresh-eyes review and the fixes it calls for (Opus 5.5, high)

**Read:**
- `revision/notes/R6.md`, `R8.md` and `R9.md`.
- The traceability table: print it with `awk '/BEGIN TRACE/,/END TRACE/' REVISION_PLAN.md`.
- `revision/REVIEW.md` (this session only).
- `paper/body.template.tex` and `paper/appendix.template.tex`, as needed.

**Steps**
1. **Extract the text.** Run `mkdir -p revision/review` and `pdftotext -layout paper/AIR2027_submission.pdf revision/review/paper.txt` (or the pypdf fallback R0 installed).
2. **Get two reviews.** Ask air-referee to review `revision/review/paper.txt`, giving it only that file. In parallel, ask air-claim-checker, with the files listed in R8 step 3.
3. **Triage every point.**
   - Fix it if it concerns text, presentation, consistency, scope or an overclaim.
   - Decline it only with a reason (for example, it needs a new experiment), in `revision/DECISIONS.md` under "R10 declined".
   - No new experiments, and no change to any registered analysis, in this session.
4. **Traceability.** For every row of the table, check that the revised paper handles it. Write `revision/notes/R10_traceability.md`: for each row id, where it is handled, or why not.
5. **Recheck.** Rerun the pipeline of R8 step 2. All checks must pass, and the two-column PDF must have at most 6 pages.
6. **Save and commit.** Save the referee's report as `revision/review/referee.md`, and your triage as `revision/notes/R10.md`. Commit "R10: review fixes".
<!-- END R10 -->

<!-- BEGIN R11 -->
## R11 — Final checks, supplement, report (Sonnet 5.5, medium)

**Read:**
- `README.md`, `SUBMIT.md` and `ANOMALIES.md`.
- The header and first lines of each of `revision/notes/R0`–`R10`, at most 30 lines each.
- The last line of `revision/review/referee.md` (the score).

**Steps**
1. **Final pipeline.** Run the full pipeline, from `paper_numbers.py` to `submission_checks.py`, three times in a row. Every check must pass, and the three builds must give the same pdftotext text. Record the page counts and the sha256 of both PDFs.
2. **Supplement for reviewers.** Build it locally; never upload it.
   - Make `submission/AIR2027_supplement.zip`, containing:
     - `prereg/` (`PREREG.md` and `PREREG_FT.md`);
     - `scripts/`;
     - `results/`, without `raw_*.npz` if the zip would otherwise exceed 50 MB (say so in its README);
     - `paper/*.template.tex` and `paper/refs.bib`;
     - a short `README_SUPPLEMENT.md`: what each folder holds, the run order, and the commit times of both registrations (from `results/timeline.json` and git).
   - Exclude `.git`, `history.bundle`, `revision/`, `.claude/`, `runs/`, `.cache/`, `.pylib/`, `go_rev.zsh` and `py`.
   - Scan every file in the zip (unzip it to `.cache/supp_check`) for the identifying patterns of `scripts/submission_checks.py`, plus "/Users/", "Downloads" and "joyjeetsingh".
   - On any hit: remove or neutralise it in the copy (never in the source), rebuild the zip, rescan, and record what was changed.
3. **Update the documents.**
   - **`README.md`:** the new scripts, result files and run order; study FT; the appendix.
   - **`SUBMIT.md`:**
     - the appendix is for the single-column submission only, and is cut for the camera-ready;
     - the supplement is optional, and its upload is the author's decision;
     - the AI-use statement must be edited by the author.
   - **`ANOMALIES.md`:** anything recorded in R0–R10.
4. **Write `revision/REPORT.md`** for the author: at most 600 words, in plain language. It covers:
   - what changed, by review point, one line each, with the traceability ids;
   - the new results and their verdicts: the exploratory analyses, and study FT with its registered verdicts;
   - the referee's score, and the points declined;
   - what only the author can do:
     - read the paper end to end, and rewrite the introduction and discussion in their own voice;
     - check AIR 2027's rules on AI-assisted writing, and edit the AI-use statement;
     - withdraw the overlapping submissions before uploading;
     - decide whether to upload the supplement;
     - submit through CMT, between 15 Nov and 31 Dec 2026.
5. **Close the round.**
   1. Commit "R11: final" and tag it `air2027-revision-1`.
   2. Regenerate the history bundle with `git bundle create history.bundle --all`, then run `git bundle verify history.bundle`. It is gitignored, so nothing is committed after it.
6. **Overwrite `revision/CHECKPOINT.md`:** "Revision round 1 complete. Next action: the author's steps in `revision/REPORT.md`."
<!-- END R11 -->

## Traceability: review points → sessions → edits

R10 checks this table, and R11 reports against it.

<!-- BEGIN TRACE -->
| Id | Review point | Sessions | Edits |
|---|---|---|---|
| P1 | RWM already reported that autoregressive training beats teacher forcing, and the effect of the horizon; the introduction implied this system is checked one step ahead | R4, R6 | T1, T2 |
| P1b | Missing *Objective Mismatch*; the critique should target real one-step practice (member selection) | R4, R6 | T2, T19 |
| P2a | The abstract led with the part that did not hold | R6 | T3, T4 |
| P2b | What was known before the registration was not disclosed; "before the data existed" | R5, R6, R7 | T5, A2 |
| P3a | The residual "on both pools" lacked the exploratory label | R6 | T6a |
| P3b | "Pruning adds bias rather than instability" was stated too generally | R5, R6, R7 | T6b, A5 |
| P3c | The released model's positive growth rate was not reconciled | R6 | T6c |
| P3d | "No recorded future", although the recorded commands are used | R6 | T6d |
| P3e | Stale command: "only a short-horizon check reveals" | R5, R6 | T6e |
| P4a | Label-free checks see 368 steps; labelled ones see 1 or 8 | R1, R2, R6, R7 | T7, A3 |
| P4b | Checks and target share trajectories; nested windows | R1, R2, R6, R7 | T7, A3 |
| P4c | The use case (commands supplied by a policy) is untested | R6 | T8 |
| P4d | The replication includes the original split | R1, R2, R6, R7 | T7, A4 |
| P5 | Scope stated late; the constant-velocity baseline buried; easy data | R6 | T9 |
| P6a | Correlation intervals ignore clustering and fixed trajectories | R1, R6, R7 | T10, A4 |
| P6b | Secondary tests uncorrected | R1, R6, R7 | T10, A4 |
| P7a | "Six readings" caption | R6 | T11 |
| P7b | 5.0 against 5.85 for the same comparison | R6 | T12 |
| P7c | Table 1 mixes model sets; the stale-command row points the other way | R6 | T11 |
| P7d | Density; Sec. 4.3; verdict labels | R6, R7 | T13, T14, A6 |
| P7e | Training settings; teacher-forced against N = 1; width 124 | R5, R6, R7 | T15, A1 |
| P7f | Figure 2 shows the growth rate, not the best check | R7 | Fig. 2 |
| P7g | Unpublished earlier analyses; self-attested times | R6, R11 | T16, supplement |
| P7h | Reference [1]'s venue | R4 | refs.bib |
| P7i | "302 versions" | R6 | T17 |
| P7j | AI statement in the singular; text in the author's voice | R6, R11 | T18, REPORT |
| RW | Calibration, teacher-forcing/Lyapunov and stable-dynamics references | R4, R6 | T19 |
| N1 | PerconAI's fine-tuned unit-removal results omitted ("needs fine-tuning") | R5, R6 | T20 |
| F5 | Reviewers' appendix; the camera-ready stays at six pages or fewer | R6, R7, R11 | T21, T22 |
| F6 | New experiment: fine-tune the pruned copies | R3, job R3, R9 | study FT |
<!-- END TRACE -->

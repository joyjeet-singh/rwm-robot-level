# AIR 2027 — "One Step Ahead Is Not Enough"

A paper for **Advances in Robotics (AIR 2027)**, the 8th International Conference of The Robotics Society
(IIT Gandhinagar, 30 June–3 July 2027). It brings together the whole PDM project on the Robotic World Model's
quadruped dynamics model and adds one new pre-registered study: which pre-deployment check (one-step error,
error over the 8-step training horizon, the leading Lyapunov exponent, a kinematic residual) best ranks how wrong a
learned simulator becomes over the horizon at which it is used.

| Path | What it is |
|---|---|
| `paper/AIR2027_submission.pdf` | the submission: single column, anonymous (the call's format) |
| `paper/AIR2027_twocolumn_check.pdf` | the same text in two columns, to check the 6-page camera-ready limit |
| `paper/body.template.tex` | the paper's source; every number is a placeholder (edit this, never `body.tex`) |
| `paper/main_single.tex`, `paper/main_twocol.tex` | the two wrappers; `paper/refs.bib` the references |
| `prereg/PREREG.md` | the new study's questions and decision rules, written before any model was run |
| `results/stats.json`, `results/stats_report.txt` | every registered and descriptive result of the new study |
| `results/model_table.json`, `results/raw_*.npz` | per-model and per-step quantities behind them |
| `results/checks.json`, `results/check_numbers.json` | pre-registered checks; independent re-computation |
| `results/exploratory.json` | analyses added after the registered results were seen |
| `results/claims_check.json` | every qualitative sentence of the paper checked against the numbers |
| `results/timeline.json` | commit times of the pre-registration, code and first data |
| `paper/ABSTRACT.txt` | title, abstract and keywords as plain text for the submission form |
| `results/paper_numbers.json` | every number in the paper, with the file and field it came from |
| `results/bib_check.json`, `results/submission_checks.json` | references traced; anonymity and page checks |
| `figures/` | the two figures |
| `scripts/` | everything above (order below) |
| `SUBMIT.md` | what only the author can do before submitting |
| `ANOMALIES.md` | problems found after the pre-registration |
| `history.bundle` | the git history of this folder (`git clone history.bundle air-history`) |

## The new study's registered verdicts (details and numbers: `results/stats_report.txt`, Table 2 of the paper)

| Question | Pool | Verdict |
|---|---|---|
| Q1 (primary): growth rate vs one-step error, ranking 100-step error | F18, 97 simulators | LABEL-FREE BETTER |
| Q2 (primary): growth rate vs 8-step error | F18, 97 | NOT SETTLED |
| Q3: replication of Q1 / Q2 within the five splits | X5, 255 | ONE-STEP BETTER / TRAINING-HORIZON BETTER (replication failed) |
| Q4: kinematic residual vs one-step / 8-step error | F18, 97 | LABEL-FREE BETTER / LABEL-FREE BETTER |
| Q5: Q1 / Q2 without the one-step-trained models | F18, 88 | LABEL-FREE BETTER / NOT SETTLED |
| Q6: does 30% pruning raise the growth rate? | X5, 25 parents | NOT SETTLED |

Exploratory (written after these were seen, `scripts/exploratory.py`): split by what differs between simulators,
one-step error ranks *training choices* backwards and *compressed copies* well; the kinematic residual also ranks the
five-split pool best, level with the 8-step error.

## Reproducing

From `scripts/`, with `PDM_ROOT` pointing at the PDM folder (default: the cloud workspace's copy; on the Mac
`PDM_ROOT=~/Downloads/PDM`), in this order:

```
python checks.py            # pre-registered checks C1-C5 (1 minute)
python compute.py F18 X5 REL   # every model's rollouts, errors, exponents, residuals (about 45 minutes, 2 cores)
python stats.py             # every registered question and descriptive result
python check_numbers.py     # independent re-computation of the registered statistics
python exploratory.py       # exploratory analyses (labelled so in the paper)
python make_figures.py
python check_bib.py         # every reference traced to an earlier verified record
python paper_numbers.py     # every number the paper prints, with its source
python build_paper.py       # fills the template, builds both PDFs, refuses on a missing number
python check_claims.py      # the paper's qualitative sentences against the numbers
python submission_checks.py # anonymity, metadata, fonts, page count
```

Nothing outside this folder is written. The stored results were produced in a Linux cloud workspace (Python
3.13.18, torch 2.15.0 CPU, numpy 2.5.3), the versions the NGCAPS and ICMAI studies used; check C1 reproduces
their stored arrays bit for bit. The Mac's torch 2.2.2 may change last digits.

## Where the paper's numbers come from

New in this folder: the check comparison (Q1-Q6), the skill retained by each compressed variant at every
horizon, the physical-unit errors, the training-horizon table and the released model's exponent. Read from the
earlier studies (their files are cited per number in `results/paper_numbers.json`): the five-fold
teacher-forcing test and the uncertainty measurements (`../rwm_repro`), the training-horizon stability rule and
the run-time growth signal (`../NGCAPS`), the kinematic alarm, repair and re-synchronisation tests (`../ICPAIAS`),
and 8-bit dynamic quantisation and CPU timing (`../PerconAI`).

## Revision round 1 (added)

The paper was revised after an internal review. What is new:

- **Study FT** (registered before any fine-tuned model existed: `prereg/PREREG_FT.md`, `scripts/ft_stats.py`): does a short fine-tune on
  multi-step error restore the skill that a one-step-only model lacks? 30 runs, 90 models. Results: `results/ft_stats.json`,
  `results/ft_report.txt`, `results/raw_FT.npz`, `results/ft_models.json`. Verdicts: FT1 FINE-TUNING RESTORES SKILL, FT2 NEARLY AS SKILLED;
  the one-step reading is not settled.
- **Appendix** (`paper/appendix.template.tex`, A1-A7: settings, statistics, horizon-matched checks, replication, compression, floors, study FT).
  It is for the single-column submission only; it is cut from the six-page camera-ready.
- **Exploratory analyses** (`scripts/revision_analyses.py`, `results/revision_exploratory.json`, `.txt`; labelled exploratory in the paper) and
  `scripts/revision_recheck.py` (`results/revision_recheck.json`).
- **New scripts:** `ft_eval.py`, `ft_unstructured.py`, `ft_stats.py`, `revision_numbers.py` (adds the `ft_*` and revision keys to
  `paper_numbers.py`), `pdf_tools.py` (pypdf fallback when poppler is absent), `smoke_compare.py`.
- **Order after the original scripts:** `revision_analyses.py`, `ft_eval.py` (training: a job), `ft_stats.py`, `revision_recheck.py`, then
  `check_bib.py`, `paper_numbers.py`, `build_paper.py`, `check_claims.py`, `submission_checks.py`. Final pipeline run three times: the same two PDFs each time.
- `submission/AIR2027_supplement.zip`: optional anonymous supplement for reviewers (built locally, never uploaded).
- `revision/`: plan, log, notes, review and report (`revision/REPORT.md`).

## Step 1 (added)

Three CPU-only studies and one engineering task, run under `STEP1_PLAN.md` (log, notes and report in `step1/`;
`step1/REPORT.md` is the summary). Registration: `prereg/PREREG_S1.md` and `scripts/s1_stats.py`, committed before any
trained simulator was driven by a policy (hash in `results/prereg_hash_s1.json`).

- **Study CL** (registered): the label-free checks with the authors' released actor supplying the joint commands.
  Verdicts: CL1 WORSE WITH POLICY COMMANDS, CL2 NOT SETTLED.
- **Study AR** (registered): the same check comparisons on 38 MLP, RSSM and Transformer simulators, 12 of them newly
  trained. Verdicts: AR1 LABEL-FREE BETTER, AR2 NOT SETTLED.
- **Study TB** (descriptive, not registered): does training length (2,500 against 10,000 iterations) change the
  conclusions? Stored values only.
- **Task LP:** the authors' lite policy-training code run on this CPU (`scripts/lite/`; nothing in the paper).

**New scripts:** `s1_policy.py` (actor as a command source), `s1_rollout.py` (checks under recorded, policy, closed-loop
and noisy commands), `s1_arch.py` (MLP, RSSM, Transformer adapters), `s1_train_baseline.py` (copy of the reproduction's
trainer, changed lines marked `S1:`), `s1_eval.py` (the job), `s1_stats.py` (registered analysis), `s1_extra.py`
(exploratory), `s1_budget.py` (study TB), `s1_numbers.py` (paper keys, called by `paper_numbers.py`), `test_s1.py`
(tests i–xi), `s1_probe.py`, `s2_probe.py` (timing probes), `build_supplement.py`, and `lite/` (`run_lite.py`,
`adapter.py`, `test_lite.py`, `stubs/`).

**New result files:** `results/s1_models.json`, `results/raw_S1.npz`, `results/s1/` (job records),
`results/s1_stats.json`, `results/s1_report.txt`, `results/s1_extra.json`, `results/s1_budget.json` and `.txt`,
`results/s1_probe.json`, `results/prereg_hash_s1.json`. New weights go to `runs/s1_baselines/`, lite runs to
`runs/lite/` (neither committed).

**Run order** (always through `./py`):

```
./py scripts/s1_probe.py; ./py scripts/s2_probe.py   # timing probes (random-weight models)
./py scripts/test_s1.py                              # tests i-xi
zsh step1/jobs/S3.zsh                                # job: s1_eval.py, trains 12 runs and scores every model (hours)
./py scripts/s1_stats.py                             # registered verdicts -> s1_stats.json, s1_report.txt
./py scripts/s1_extra.py                             # exploratory additions
./py scripts/s1_budget.py                            # study TB (descriptive)
./py scripts/paper_numbers.py; ./py scripts/build_paper.py
./py scripts/check_claims.py; ./py scripts/check_bib.py; ./py scripts/submission_checks.py
./py scripts/build_supplement.py                     # optional reviewer supplement
./py scripts/lite/test_lite.py                       # lite code checks (Task LP; separate from the paper)
```

## Rewrite round (W0-W5, added)

The main text (`paper/body.template.tex`) was rewritten around all the evidence (original study, revision round 1,
Step 1) so that it fits six two-column pages **with a blank Step 2 slot** (0.30 of the text height, one column wide, at
the end of Sec. 4). Plan: `REWRITE_PLAN.md`; everything else is in `rewrite/`:

- `rewrite/REPORT.md`: the summary for the author (start here); `rewrite/review/section_changes.md`: what changed, section by section.
- `rewrite/OUTLINE.md` (the plan for the text), `LOG.md`, `CHECKPOINT.md`, `DECISIONS.md`, `notes/` (one note per session), `review/` (the W4 reviews).
- New scripts: `scripts/slack.py` (fit check: two-column and reserve builds at most six pages, abstract at most 200 words) and
  `scripts/rw_numbers.py` (22 `rw_t_*` re-prints from `results/s1_stats.json`, called by `paper_numbers.py`).
- Changed: `body.template.tex`, `appendix.template.tex` (A1, A5, A6 gained material, no renumbering), `check_claims.py` (107 -> 127 assertions), the Step 2 slot in `main_*.tex` and `build_paper.py`.
- No registration, analysis script or result file was touched, and nothing was rerun.

Run order after the rewrite (always through `./py`): `paper_numbers.py`, `build_paper.py`, `slack.py`, `check_claims.py`,
`check_numbers.py`, `check_bib.py`, `submission_checks.py`, then optionally `build_supplement.py`.

## Improvement round (I0-I5, added)

An expert review of the rewritten paper was applied without new experiments. Plan: `IMPROVE_PLAN.md`; everything else is in `improve/`:

- `improve/REPORT.md`: the summary for the author (start here); `improve/review/section_changes.md`: what changed, section by section
  (`latexdiff` is not installed); `improve/review/referee.md`, `stats.md`: the I4 reviews.
- `improve/LOG.md`, `CHECKPOINT.md`, `DECISIONS.md` (choices made and points declined), `notes/` (one note per session).
- New scripts: `scripts/im_facts.py` (writes `results/external_facts.json`: facts from the RWM, RWM-U and iKCE papers, with source, version
  and quoted line), `scripts/im_within.py` (`results/im_within.json`, `.txt`: rankings among simulators trained the same way; exploratory),
  `scripts/im_numbers.py` (number keys, called by `paper_numbers.py`).
- Changed: `body.template.tex`, `appendix.template.tex`, `make_figures.py` (Fig. 1 horizon line), `check_claims.py` (127 -> 159 assertions).
- No registration, analysis script or earlier result file was touched, and nothing was rerun. Git tags: `air2027-improve-start`, `air2027-improve`.

## Step 2 (M0-M6, added)

The robot-level test: do the paper's checks predict how well policies trained inside learned simulators walk in physics?
The physics is a MuJoCo model of ANYmal D that this round built by hand from the manufacturer's published description
(BSD 3-Clause). It is not the data's own simulator and not hardware. Plan: `STEP2_PLAN.md`; log, checkpoint, decisions
and notes in `step2/` (`LOG.md`, `CHECKPOINT.md`, `DECISIONS.md`, `notes/`). Registration: `prereg/PREREG_S2.md` and
`scripts/s2_stats.py`, committed after the model's two validation gates and before any registered policy was trained
(hash in `results/prereg_hash_s2.json`).

- **Result (registered):** R0, R1 and R2 are all NOT SETTLED over nine simulators (the released model and its eight
  compressed copies, two policies each). The fallback pool replaced the planned main-pool one because Gate 2 failed:
  policies trained in rebuilt single-network simulators fell in every pilot episode. Secondary and exploratory
  analyses are NOT SETTLED too; the rollout-trained subset is NOT APPLICABLE. Details: Sec. 4.2, Table III, appendices A11 and A12.
- **Motor model:** the ANYdrive actuator network could not be obtained (`step2/assets/manifest.json`: FAILED), so a PD
  controller with a fixed delay, fitted to the recorded torques, was used. The network file is not in any deliverable.
- **Code:** `scripts/s2/` (model build, motor fit, calibration, fidelity, environment, policy training, evaluation, tests),
  `scripts/s2_main.py` (the 18-policy job), `scripts/s2_stats.py` (registered analysis), `scripts/s2_numbers.py` (283 `s2_*` keys,
  called by `paper_numbers.py`), `scripts/s2_probe.py` (timing probe).
- **Results:** `results/s2_stats.json`/`.txt` (registered analysis), `s2_main.json`, `s2_pilot.json`/`.txt` (Gate 2),
  `s2_fidelity.json`/`.txt` (calibration, held-out replay, Gate 1), `prereg_hash_s2.json`, and `results/s2/` (calibration grid,
  checks, per-policy training and evaluation records). Weights go to `runs/` (not committed).
- **Model files:** `step2/model/` (`anymal_d.xml`, `anymal_d_cal.xml`, `calibration.json`). `step2/assets/` (downloaded
  description, MuJoCo install) is never committed.
- **Jobs** (run by the author's launchers, never by a session): `step2/jobs/fetch.zsh`, `calib.zsh`, `pilot.zsh`, `main.zsh` (about 20 h).
- **Changed in M5-M6:** `body.template.tex`, `appendix.template.tex` (A11, A12), `paper_numbers.py` (calls `s2_numbers.py`),
  `build_paper.py`, `check_claims.py` (159 -> 181 assertions), `build_supplement.py` (Step 2 files). Git tags: `air2027-step2-start`, `air2027-step2`.

Run order after the main job (always through `./py`; MuJoCo work through `./py-mj`):

```
./py scripts/s2/test_s2.py                           # tests of the registered code
./py scripts/s2_stats.py                             # registered analysis (run once, unedited)
./py scripts/paper_numbers.py; ./py scripts/build_paper.py
./py scripts/slack.py; ./py scripts/check_claims.py; ./py scripts/check_numbers.py
./py scripts/check_bib.py; ./py scripts/submission_checks.py
./py scripts/build_supplement.py                     # optional reviewer supplement
```

## Options C and D (simulators of a known physics model, added)

Does a simulator that scores well on the paper's checks produce policies that walk? Option C (from-scratch simulators) failed; Option D
(the released model fine-tuned in a searched ladder of runs) is the registered test. The world is the hand-built MuJoCo ANYmal D of Step 2
(fitted PD motor; not Isaac Lab, not hardware). Plans: `OPTIONC_PLAN.md`, `OPTIOND_PLAN.md`; log, checkpoint, decisions and notes in `optc/`
and `optd/`. Registration: `prereg/PREREG_D.md` and `scripts/d_stats.py`, committed after the record of existing results and before any
policy was trained in a fine-tuned simulator (hash in `results/prereg_hash_d.json`).

- **Result (registered, descriptive and secondary labelled in the paper):** R0, R1 and R2 are all NOT SETTLED over 19 simulators (the
  released model and 18 tested saves). 13 of the 18 saves failed (two or more of their 40 policies below 0.15). Every save had lower one-step
  and 8-step error than the released model, and those two checks ranked walking in reverse (rho about -0.4). The walk test's tracking ranked
  walking well (secondary). Option C's from-scratch simulators failed the walk test (two design mistakes: state scaling, and about 0.1 of one pass
  over the data); policies from rebuilt single-network simulators fell in every pilot episode. Details: Sec. 4.2, Table "ladder", appendices A12 and A13.
- **Code:** `scripts/c/` (Option C), `scripts/d/` (record, fine-tuning, checks, walk test, policy wrapper with the stop rule, search, tests),
  `scripts/d_stats.py` (registered analysis), `scripts/d_numbers.py` (`d_*` and `c_*` keys, called last by `paper_numbers.py`).
- **Results:** `results/d_record.*`, `results/d_ladder.*`, `results/d_stats.*` (the registered analysis, never overwritten), `results/d/`,
  `results/c_*.json`, `results/c/`. Data, simulators and policies are in `optc/data/` and `runs/` (not committed; `optc/data_manifest.json`
  lists the generator, seeds and hashes: `scripts/c/gen_data.py` and `scripts/d/finetune.py` recreate them).
- **Jobs** (run by the launcher, never by a session): `optd/jobs/record.zsh`, `optd/jobs/ladder.zsh` (53 h).
- **Changed in D2-D3:** `body.template.tex`, `appendix.template.tex` (A12 root cause, new A13), `paper_numbers.py`, `check_claims.py`
  (181 -> 210 assertions; the Option D block is the last), `build_supplement.py` (Option C and D files, the Option C data manifest, path
  neutralisation in the staged copy). Git tags: `air2027-optd-start`, `air2027-optd`.

Run order after the ladder job (always through `./py`):

```
./py scripts/d_stats.py                              # registered analysis (already run; never overwritten)
./py scripts/paper_numbers.py; ./py scripts/build_paper.py
./py scripts/slack.py; ./py scripts/check_claims.py; ./py scripts/check_numbers.py
./py scripts/check_bib.py; ./py scripts/submission_checks.py
./py scripts/build_supplement.py                     # optional reviewer supplement (full 73 MB; --slim 4 MB)
```

## Published copy (added 17 Nov 2026)

This folder was written as a submission to Advances in Robotics (AIR 2027). That venue has since been dropped; the
text above is kept as it was. Licence: Apache 2.0 (`LICENSE`); attributions and upstream licences: `NOTICE`.

**Weights and policies.** `runs/` is not in git. The simulators, policies and training records behind the results
above are on Hugging Face at https://huggingface.co/Joyjeetsingh/rwm-robot-level, at the same `runs/...` paths, with each file's size and
SHA-256 in `MANIFEST.json`. To put them back into a clone of this repository:

```
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('Joyjeetsingh/rwm-robot-level', local_dir='.', allow_patterns='runs/*')"
```

Test runs, resumable training state, diagnostic arrays and the policy trainer's intermediate checkpoints were not
uploaded. The scripts expect the PDM folder layout (`../rwm_repro`, `../robotic_world_model_lite`); the reproduction
is at https://github.com/joyjeet-singh/rwm.

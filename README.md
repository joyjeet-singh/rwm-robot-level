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
| `results/paper_numbers.json` | every number in the paper, with the file and field it came from |
| `results/bib_check.json`, `results/submission_checks.json` | references traced; anonymity and page checks |
| `figures/` | the two figures |
| `scripts/` | everything above (order below) |
| `SUBMIT.md` | what only the author can do before submitting |
| `ANOMALIES.md` | problems found after the pre-registration |
| `history.bundle` | the git history of this folder (`git clone history.bundle air-history`) |

## Reproducing

From `scripts/`, with `PDM_ROOT` pointing at the PDM folder (default: the cloud workspace's copy; on the Mac
`PDM_ROOT=~/Downloads/PDM`), in this order:

```
python checks.py            # pre-registered checks C1-C5 (1 minute)
python compute.py F18 X5 REL   # every model's rollouts, errors, exponents, residuals (about 45 minutes, 2 cores)
python stats.py             # every registered question and descriptive result
python check_numbers.py     # independent re-computation of the registered statistics
python make_figures.py
python check_bib.py         # every reference traced to an earlier verified record
python paper_numbers.py     # every number the paper prints, with its source
python build_paper.py       # fills the template, builds both PDFs, refuses on a missing number
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

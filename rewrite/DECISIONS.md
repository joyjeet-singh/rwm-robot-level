# Rewrite decisions

## W0
- Step 2 slot is drawn inline (a framed `\parbox`, one column wide, 0.30 textheight, with \medskip around), not as a float: conservative for the fit check, since an inline box cannot move to fill a column gap.
- `scripts/slack.py` writes `results/slack.json` and updates `results/build.json` (the reserve entry) via build_paper.py; no registered result file touched.
- HEAD at start was c699388, tagged air2027-step1 (as expected). `figures/fig_horizon.*` left unstaged.

# Cloud test scripts behind `optd/REVIEW.md` (reference only)

These scripts ran on a cloud copy of the project, not on the Mac. They use cloud paths (`/mnt/user-data/uploads/PDM`) and will not run here unchanged. They are kept so that each number in `optd/REVIEW.md` can be traced to the code that produced it. Option D's own code is written fresh in `scripts/d/`.

| Script | What it did | REVIEW section |
|---|---|---|
| `cl.py` | closed-loop helpers: load a simulator, build the lite imagination, run a policy in it, summarise | 1–3 |
| `prof.py` | time profiles of the walk test; when and how robots are predicted to fall | 2, 3 |
| `ol.py` | open-loop errors on held-out MuJoCo data, common units | 6 |
| `chk.py` | the paper's long-horizon checks, common units | 7, 11 |
| `mjref.py` | a policy in MuJoCo under the imagination's commands, pushes and noise | 1, 9 |
| `relmj.py` | the released model started from MuJoCo states | 1 |
| `ft.py` | fine-tuning the released model, with the walk test and the fall test at intervals | 8, 10 |
| `showft.py` | prints `ft.py`'s log as a table | 8, 10 |
| `ppo.py` | policy training with the authors' lite code inside a simulator | 9 |
| `mjeval.py` | the walking score of an actor file, with the project's protocol | 9 |
| `tri.py` | one policy judged in its own simulator, in the released model and in MuJoCo | 9 |
| `exploit.py` | records a policy in MuJoCo and scores each simulator's open-loop predictions on that recording | 10 |
| `score_s2.py` | the walking score of each of Step 2's 22 saved policies, with `mjeval.py` | 13 |
| `design.py` | the simulated ladder: the plan's search played many times under each rule for policies per simulator | 13 |
| `design_cal.py` | `design.py` again, with the walking policies' spread measured in Step 2 (0.12) | 13 |
| `design_*.log` | the printed results of the two scripts above | 13 |

`ppo.py` builds the imagination from the lite classes with the lite config's values. It is not Step 2's wrapper; Option D's policies use Step 2's wrapper. Section 13's two extra `FT160` policies are `ppo.py` with `--seed 1` and `--seed 2`, and otherwise section 9's control settings.

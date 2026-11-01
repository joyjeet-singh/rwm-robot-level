"""
EXPLORATORY analyses, written after the registered results were seen (stats.json) and labelled exploratory in the
paper. They ask where each check works: across training choices or across compressed copies of one model, and
whether the kinematic residual, registered on F18 only, also ranks the five-split pool.
  X1  X5: within-split Spearman of r_J with E100, against E1 and E8 (models resampled within splits)
  X2  F18 subsets: training choices (pools R and T, 57) and compressed copies (pool C, 40)
  X3  X5 subsets: uncompressed models (11 per split) and compressed copies (40 per split), within splits
-> results/exploratory.json
"""
import json
import os

import numpy as np

import common as C
from stats import boot_diff, boot_diff_within, boot_rho, spearman

T = json.load(open(os.path.join(C.RESULTS, "model_table.json")))
f18, x5 = T["F18"], T["X5"]
folds = [C.fid(f) for f in C.FOLDS]
out = {"label": "exploratory"}

# X1
groups = [[r for r in x5 if r["arena"] == f] for f in folds]
out["X1_rJ_vs_E1"] = boot_diff_within(groups, "rJ", "E1", "E100")
out["X1_rJ_vs_E8"] = boot_diff_within(groups, "rJ", "E8", "E100")

# X2
col = lambda rows, k: [r[k] for r in rows]                                      # noqa: E731
for name, rows in (("training_choices", [r for r in f18 if r["pool"] in ("R", "T")]),
                   ("compressed", [r for r in f18 if r["pool"] == "C"])):
    out[f"X2_{name}"] = {c: boot_rho(col(rows, c), col(rows, "E100")) for c in ("E1", "E8", "lam1", "rJ")}
    out[f"X2_{name}"]["n"] = len(rows)
    out[f"X2_{name}"]["lam_minus_E1"] = boot_diff(col(rows, "lam1"), col(rows, "E1"), col(rows, "E100"),
                                                  levels=(95.0,))

# X3
for name, pred in (("uncompressed", lambda r: r["variant"] is None), ("compressed", lambda r: r["variant"] is not None)):
    gs = [[r for r in x5 if r["arena"] == f and pred(r)] for f in folds]
    res = {}
    for c in ("E1", "E8", "lam1", "rJ"):
        res[c] = float(np.mean([spearman(col(g, c), col(g, "E100")) for g in gs]))
    res["n_per_split"] = [len(g) for g in gs]
    res["lam_minus_E1"] = boot_diff_within(gs, "lam1", "E1", "E100")
    out[f"X3_{name}"] = res

json.dump(out, open(os.path.join(C.RESULTS, "exploratory.json"), "w"), indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: (vv if not isinstance(vv, dict) else
                  {a: b for a, b in vv.items() if a in ("rho", "ci95", "delta", "rho_new_mean", "rho_base_mean")})
                  for kk, vv in v.items()}) for k, v in out.items()}, indent=1, default=float)[:4000])

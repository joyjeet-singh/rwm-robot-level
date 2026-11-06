"""
EXPLORATORY additions to studies CL and AR (not in prereg/PREREG_S1.md), written in session S6 after the registered
analysis (scripts/s1_stats.py) had run. Point estimates only, no intervals, no labels.

- X5: the within-split mean Spearman correlation with E100 (as s1_stats' CL4 computes it) of the growth rate and the
  kinematic residual under noisy recorded joint commands, and of the closed-loop growth rate lambda1_cl. CL4 itself
  covers only the policy's joint commands; the recorded and policy values are recomputed here as a cross-check.
- AR: which baseline runs diverged under recorded joint commands, and on how many of their trajectories.

Reads results/s1_models.json and results/model_table.json; writes results/s1_extra.json.
"""
import json
import os

import numpy as np

import common as C
import stats as ST
from s1_stats import cl_rows, col


def main():
    db = json.load(open(os.path.join(C.RESULTS, "s1_models.json")))
    table = json.load(open(os.path.join(C.RESULTS, "model_table.json")))
    rep = json.load(open(os.path.join(C.RESULTS, "s1_stats.json")))
    x5, miss = cl_rows(db, table, "X5")
    assert not miss
    groups = [[r for r in x5 if r["arena"] == C.fid(f)] for f in C.FOLDS]
    assert sum(map(len, groups)) == len(x5)
    out = {"label": "exploratory", "X5_within_split_mean_rho_with_E100": {}}
    for k in ("lam1_rec", "rJ_rec", "lam1_pol", "rJ_pol", "lam1_noise", "rJ_noise", "lam1_cl"):
        per = [ST.spearman(col(g, k), col(g, "E100")) for g in groups]
        out["X5_within_split_mean_rho_with_E100"][k] = {"mean": float(np.mean(per)), "per_split": per}
    m = out["X5_within_split_mean_rho_with_E100"]
    # cross-check against the registered CL4 point estimates
    for q, new, base in (("CL1", "rJ_pol", "rJ_rec"), ("CL2", "lam1_pol", "lam1_rec")):
        assert abs(m[new]["mean"] - rep["CL"]["CL4"][q]["rho_new_mean"]) < 1e-12, q
        assert abs(m[base]["mean"] - rep["CL"]["CL4"][q]["rho_base_mean"]) < 1e-12, q
    div = {}
    for k, s in sorted(db["models"]["AR"].items()):
        if s["n_div_rec"]:
            div[k] = {"arch": s["arch"], "regime": s["regime"], "spec": s["spec"], "seed": s["seed"],
                      "new_run": s["new_run"], "trajectories_diverged": s["n_div_rec"]}
    out["AR_diverged_recorded"] = div
    out["AR_n_trajectories_per_run"] = 4
    json.dump(out, open(os.path.join(C.RESULTS, "s1_extra.json"), "w"), indent=1)
    for k, v in m.items():
        print(f"X5 {k}: {v['mean']:+.3f}")
    print("AR diverged:", {k: v["trajectories_diverged"] for k, v in div.items()})


if __name__ == "__main__":
    main()

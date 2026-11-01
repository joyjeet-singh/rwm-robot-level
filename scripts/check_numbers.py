"""
Independent re-computation of the registered statistics from the raw arrays, with different code from stats.py:
scipy's Spearman and Kendall, per-model values rebuilt from raw_*.npz without model_table.json, and a bootstrap
with its own generator (seed 12345). Point estimates must agree to 1e-12; verdicts must agree; interval ends
within 0.03. -> results/check_numbers.json
"""
import json
import os

import numpy as np
from scipy.stats import kendalltau, spearmanr

import common as C


def per_model(tag):
    raw = np.load(os.path.join(C.RESULTS, f"raw_{tag}.npz"))
    meta = json.load(open(os.path.join(C.RESULTS, "models.json")))[tag]
    out = {}
    for m in meta:
        k = m["key"]
        out[k] = {"pool": m["pool"], "group": m["group"], "arena": m["arena"], "variant": m["variant"],
                  "E1": raw[f"E1_{k}"].mean(), "E8": raw[f"E8_{k}"].mean(), "E100": raw[f"E100_{k}"].mean(),
                  "lam": raw[f"lam_{k}"][:, 0].mean(), "rJ": raw[f"rJ_{k}"].mean()}
    return out


def sp(a, b):
    a, b = np.nan_to_num(np.asarray(a, float), nan=np.inf), np.nan_to_num(np.asarray(b, float), nan=np.inf)
    return spearmanr(a, b).statistic


def boot(fn, n, rng, B=20000):
    vals = []
    for _ in range(B):
        v = fn(rng.integers(0, n, n))
        if np.isfinite(v):
            vals.append(v)
    return np.percentile(vals, [2.5, 97.5]), np.percentile(vals, [1.25, 98.75])


def lab(ci, pos, neg):
    return pos if ci[0] > 0 else (neg if ci[1] < 0 else "NOT SETTLED")


S = json.load(open(os.path.join(C.RESULTS, "stats.json")))
f18 = per_model("F18")
keys = list(f18)
V = {c: np.array([f18[k][c] for k in keys]) for c in ("E1", "E8", "E100", "lam", "rJ")}
rng = np.random.default_rng(12345)
out, ok = {}, True
for q, base, pos, neg in (("Q1", "E1", "LABEL-FREE BETTER", "ONE-STEP BETTER"),
                          ("Q2", "E8", "LABEL-FREE BETTER", "TRAINING-HORIZON BETTER")):
    est = sp(V["lam"], V["E100"]) - sp(V[base], V["E100"])
    ci95, ci975 = boot(lambda i: sp(V["lam"][i], V["E100"][i]) - sp(V[base][i], V["E100"][i]), len(keys), rng)
    verdict = lab(ci975, pos, neg)
    agree = (abs(est - S[q]["delta"]) < 1e-12 and verdict == S[q]["verdict"]
             and max(abs(ci975[0] - S[q]["ci97.5"][0]), abs(ci975[1] - S[q]["ci97.5"][1])) < 0.03)
    out[q] = {"delta": float(est), "ci95": ci95.tolist(), "ci97.5": ci975.tolist(), "verdict": verdict,
              "tau_lam": float(kendalltau(V["lam"], V["E100"]).statistic),
              "tau_base": float(kendalltau(V[base], V["E100"]).statistic), "agrees_with_stats": bool(agree)}
    ok &= agree

# Q3: within-fold mean, models resampled within folds
x5 = per_model("X5")
folds = [C.fid(f) for f in C.FOLDS]
G = [{c: np.array([v[c] for v in x5.values() if v["arena"] == f]) for c in ("E1", "E8", "E100", "lam")}
     for f in folds]
for base in ("E1", "E8"):
    est = np.mean([sp(g["lam"], g["E100"]) - sp(g[base], g["E100"]) for g in G])
    vals = []
    for _ in range(20000):
        d = []
        for g in G:
            i = rng.integers(0, len(g["E100"]), len(g["E100"]))
            d.append(sp(g["lam"][i], g["E100"][i]) - sp(g[base][i], g["E100"][i]))
        if np.all(np.isfinite(d)):
            vals.append(np.mean(d))
    ci = np.percentile(vals, [2.5, 97.5])
    ref = S[f"Q3_vs_{base}"]
    verdict = lab(ci, "LABEL-FREE BETTER", "ONE-STEP BETTER" if base == "E1" else "TRAINING-HORIZON BETTER")
    agree = abs(est - ref["delta"]) < 1e-12 and verdict == ref["verdict"] and \
        max(abs(ci[0] - ref["ci95"][0]), abs(ci[1] - ref["ci95"][1])) < 0.03
    out[f"Q3_vs_{base}"] = {"delta": float(est), "ci95": ci.tolist(), "verdict": verdict, "agrees_with_stats": bool(agree)}
    ok &= agree

# Q6: lambda_1 of the 30%-pruned variant minus its parent's, 25 parents
par = [k for k, v in x5.items() if v["group"] == "A"]
d = np.array([x5[f"{p}__u30"]["lam"] - x5[p]["lam"] for p in par])
vals = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(20000)]
ci = np.percentile(vals, [2.5, 97.5])
verdict = lab(ci, "MORE EXPANSIVE", "MORE CONTRACTIVE")
agree = abs(d.mean() - S["Q6"]["mean"]) < 1e-12 and verdict == S["Q6"]["verdict"]
out["Q6"] = {"mean": float(d.mean()), "ci95": ci.tolist(), "verdict": verdict, "agrees_with_stats": bool(agree)}
ok &= agree
out["all_agree"] = bool(ok)
json.dump(out, open(os.path.join(C.RESULTS, "check_numbers.json"), "w"), indent=1)
print(json.dumps(out, indent=1))

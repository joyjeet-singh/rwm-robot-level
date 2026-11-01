"""R2: independent recheck of R1 point estimates (A1, A3, A2b rho/deltas). numpy, json, scipy.stats.spearmanr only.
Does not import revision_analyses.py or stats.py. Spec: revision/notes/R1_spec.md."""
import json
import numpy as np
from scipy.stats import spearmanr

DT = 0.02
RES = "results/"
MT = json.load(open(RES + "model_table.json"))
EX = json.load(open(RES + "revision_exploratory.json"))
RAW = {a: np.load(RES + f"raw_{a}.npz") for a in ("F18", "X5")}


def rho(a, b):
    return float(spearmanr(a, b).statistic)


def traj(a, key):
    z = RAW[a]
    L, R = z["log1_" + key], z["rJ_" + key]
    return {"E1": z["E1_" + key], "E8": z["E8_" + key], "E24": z["E24_" + key], "E100": z["E100_" + key],
            "E368": z["err_" + key][:, -1] if False else None, "log1": L, "rJ": R, "err": z["err_" + key]}


def lam(L, k):
    k = min(k, 367)
    return L[:, :k].sum(axis=1) / (k * DT)       # per trajectory


def feats(a, key, sel=None):
    """quantities per model; sel = trajectory indices (None = all) averaged over trajectories."""
    t = traj(a, key)
    s = slice(None) if sel is None else sel
    f = {"E1": t["E1"][s].mean(), "E8": t["E8"][s].mean(), "E24": t["E24"][s].mean(), "E100": t["E100"][s].mean(),
         "E368": t["err"][s].mean()}   # E368 = mean over all 368 steps
    for k in (8, 24, 100, 368):
        f[f"lam1_{k}"] = lam(t["log1"][s], k).mean()
        f[f"rJ_{k}"] = t["rJ"][s][:, :k].mean()
    return f


def pools(a):
    rows = [r for r in MT[a]]
    if a == "F18":
        return {"all": rows, "trained": [r for r in rows if r["variant"] is None],
                "compressed": [r for r in rows if r["variant"] is not None]}
    return {"X5": rows}


def split_rho(a, rows, check, target, sel_c=None, sel_t=None, folds=None):
    """F18: plain rho; X5: rho within each fold then mean."""
    if a == "F18":
        F = [(feats(a, r["key"], sel_c), feats(a, r["key"], sel_t)) for r in rows]
        return rho([c[check] for c, _ in F], [t[target] for _, t in F])
    out = []
    for fo in folds:
        rr = [r for r in rows if r["extra"]["fold"] == fo]
        F = [(feats(a, r["key"], sel_c), feats(a, r["key"], sel_t)) for r in rr]
        out.append(rho([c[check] for c, _ in F], [t[target] for _, t in F]))
    return float(np.mean(out))


FOLDS = ["18", "02", "34", "56", "79"]
CH = [f"lam1_{k}" for k in (8, 24, 100, 368)] + [f"rJ_{k}" for k in (8, 24, 100, 368)] + ["E1", "E8"]
res, cmp = {}, []


def check(name, mine, theirs):
    cmp.append((name, mine, theirs, abs(mine - theirs)))


# A1
A1 = EX["A1_horizon_matched"]
for pn, rows in pools("F18").items():
    for tg in ("E24", "E100", "E368"):
        for c in CH:
            check(f"A1/F18/{pn}/{tg}/{c}", split_rho("F18", rows, c, tg), A1["F18"][pn][tg][c]["rho"])
rows = pools("X5")["X5"]
for tg in ("E24", "E100", "E368"):
    for c in CH:
        check(f"A1/X5/{tg}/{c}", split_rho("X5", rows, c, tg, folds=FOLDS), A1["X5"][tg][c]["rho"])
D = A1["diffs_vs_E100"]
for nm, c in (("rJ_8_minus_E8", "rJ_8"), ("rJ_100_minus_E8", "rJ_100")):
    for pn, rows in pools("F18").items():
        d = split_rho("F18", rows, c, "E100") - split_rho("F18", rows, "E8", "E100")
        check(f"A1/diff/{nm}/F18_{pn}", d, D[nm][f"F18_{pn}"]["delta"])
    d = split_rho("X5", pools("X5")["X5"], c, "E100", folds=FOLDS) - split_rho("X5", pools("X5")["X5"], "E8", "E100", folds=FOLDS)
    check(f"A1/diff/{nm}/X5", d, D[nm]["X5"]["delta"])

# A3 split-half: trajectory order within an arena is [ep a, ep a, ep b, ep b]
A3 = EX["A3_split_half"]
for a_, eps in (("F18", EX["A3_split_half"]["episodes"]["18"]),):
    assert eps == [1, 1, 8, 8]
first, second = [0, 2], [1, 3]
for c in ("E1", "E8", "lam1", "rJ"):
    ck = {"E1": "E1", "E8": "E8", "lam1": "lam1_368", "rJ": "rJ_368"}[c]
    for pn, rows in pools("F18").items():
        v = np.mean([split_rho("F18", rows, ck, "E100", first, second), split_rho("F18", rows, ck, "E100", second, first)])
        check(f"A3/F18/{pn}/{c}", float(v), A3["F18"][pn][c]["rho"])
    rows = pools("X5")["X5"]
    v = np.mean([split_rho("X5", rows, ck, "E100", first, second, FOLDS),
                 split_rho("X5", rows, ck, "E100", second, first, FOLDS)])
    check(f"A3/X5/{c}", float(v), A3["X5"][c]["rho"])

# A2(b): X5 without split {1,8}
B = EX["A2b_X5_without_split18"]
f4 = ["02", "34", "56", "79"]
assert B["splits"] == f4
rows = pools("X5")["X5"]
T2 = {"E1": "E1", "E8": "E8", "lam1": "lam1_368", "rJ": "rJ_368"}
r4 = {k: split_rho("X5", rows, v, "E100", folds=f4) for k, v in T2.items()}
for k in T2:
    check(f"A2b/table2/{k}", r4[k], B["table2_X5"][k]["rho"])
check("A2b/Q3_vs_E1", r4["lam1"] - r4["E1"], B["diffs"]["Q3_vs_E1"]["delta"])
check("A2b/Q3_vs_E8", r4["lam1"] - r4["E8"], B["diffs"]["Q3_vs_E8"]["delta"])
check("A2b/rJ_vs_E8", r4["rJ"] - r4["E8"], B["diffs"]["rJ_vs_E8"]["delta"])

mx = max(c[3] for c in cmp)
bad = [c for c in cmp if c[3] > 1e-9]
out = {"all_agree": not bad, "n_compared": len(cmp), "max_abs_diff": mx,
       "disagreements": [{"name": n, "recheck": m, "r1": t} for n, m, t, _ in bad],
       "scope": "point estimates only; intervals not rechecked",
       "implementation": "numpy+json+scipy.stats.spearmanr; no import of revision_analyses.py or stats.py"}
json.dump(out, open(RES + "revision_recheck.json", "w"), indent=1)
print(out["all_agree"], len(cmp), mx)
for n, m, t, dd in bad[:20]:
    print(n, m, t)

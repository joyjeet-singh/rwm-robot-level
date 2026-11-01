"""
EXPLORATORY re-analyses for the revision (REVISION_PLAN.md, session R1). Nothing here is registered.
Reads the stored results only (results/model_table.json, stats.json, exploratory.json, raw_F18.npz, raw_X5.npz).
  A1  horizon-matched label-free checks: lambda_1(k) and mean r_J over steps 1..k, against E24, E100, E368
  A2  wider intervals: (a) two-level bootstrap (clusters of a parent and its compressed copies, and trajectories);
      (b) X5 without split {1, 8}; (c) Holm correction over the secondary registered tests
  A3  split-half: checks from one trajectory of each held-out episode, target E100 from the other
-> results/revision_exploratory.json, results/revision_exploratory.txt
Registered numbers recomputed here are asserted equal to results/stats.json (and exploratory.json) before
anything new is computed. Model-resampled intervals for new statistics use the registered draws
(default_rng(0), same shapes and order as stats.py) through a vectorised weighted Spearman, checked against
stats.py's own loops on the registered statistics.
"""
import json
import os
import platform
import time

import numpy as np

import common as C
from stats import boot_diff, boot_diff_within, spearman

T0 = time.time()
B = C.N_BOOT
DT = C.DT
NSTEP = 367                                     # log-stretch entries (forecast steps 1 -> 368)
KS = (8, 24, 100, 368)
TARGETS = ("E24", "E100", "E368")
FOLDS = [C.fid(f) for f in C.FOLDS]
OUT = {"label": "exploratory", "session": "R1",
       "versions": {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine(),
                    "system": platform.system()}}
TBL = json.load(open(os.path.join(C.RESULTS, "model_table.json")))
ST = json.load(open(os.path.join(C.RESULTS, "stats.json")))
EX = json.load(open(os.path.join(C.RESULTS, "exploratory.json")))
f18, x5 = TBL["F18"], TBL["X5"]
col = lambda rows, k: [r[k] for r in rows]                                        # noqa: E731


def say(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# ------------------------------------------------------------------ per-trajectory arrays, built once
def traj_arrays(tag, rows):
    raw = np.load(os.path.join(C.RESULTS, f"raw_{tag}.npz"))
    V = {}
    for h in (1, 8, 24, 100):
        V[f"E{h}"] = np.stack([raw[f"E{h}_{r['key']}"] for r in rows]).astype(np.float64)
    V["E368"] = np.stack([raw[f"err_{r['key']}"].astype(np.float64).mean(1) for r in rows])
    log1 = np.stack([raw[f"log1_{r['key']}"] for r in rows])                       # (n, traj, 367)
    rJ = np.stack([raw[f"rJ_{r['key']}"] for r in rows])                           # (n, traj, 368)
    for k in KS:
        m = min(k, NSTEP)
        V[f"lam1_{k}"] = log1[:, :, :m].sum(-1) / (m * DT)
        V[f"rJ_{k}"] = rJ[:, :, :k].mean(-1)
    # model-level values: E_h exactly as stats.model_table; lambda_1(k) and r_J(k) as defined in the plan
    M = {f"E{h}": np.array(col(rows, f"E{h}")) for h in (1, 8, 24, 100, 368)}
    for k in KS:
        M[f"lam1_{k}"] = V[f"lam1_{k}"].mean(1)
        M[f"rJ_{k}"] = np.array([float(np.mean(raw[f"rJ_{r['key']}"][:, :k])) for r in rows])
    # E_h means of the stored arrays reproduce model_table exactly
    for h in (1, 8, 24, 100):
        assert all(float(np.mean(raw[f"E{h}_{r['key']}"])) == r[f"E{h}"] for r in rows), (tag, h)
    return V, M


V18, M18 = traj_arrays("F18", f18)
VX5, MX5 = traj_arrays("X5", x5)

# lambda_1 from log1 against the registered lam1 (model_table, computed in torch in the cloud workspace)
lam_dev = {tag: float(np.max(np.abs(M["lam1_368"] - np.array(col(rows, "lam1")))))
           for tag, M, rows in (("F18", M18, f18), ("X5", MX5, x5))}
lam_exact = {tag: int(np.sum(M["lam1_368"] == np.array(col(rows, "lam1"))))
             for tag, M, rows in (("F18", M18, f18), ("X5", MX5, x5))}
assert max(lam_dev.values()) < 1e-12, lam_dev
assert all(np.array_equal(M["rJ_368"], np.array(col(rows, "rJ"))) for M, rows in ((M18, f18), (MX5, x5)))
OUT["lam1_from_log1"] = {"max_abs_dev_from_model_table": lam_dev, "n_bit_identical": lam_exact,
                         "n_models": {"F18": len(f18), "X5": len(x5)},
                         "note": "registered lam1 = mean of torch sum(log1)/(367*0.02); numpy summation differs "
                                 "in the last bit; rank-based results are identical (asserted below)"}

# pools (as stats.py and exploratory.py)
POOL = np.array(col(f18, "pool"))
SUB18 = {"all": np.ones(len(f18), bool), "trained": np.isin(POOL, ("R", "T")), "compressed": POOL == "C",
         "RC": np.isin(POOL, ("R", "C"))}
assert (SUB18["all"].sum(), SUB18["trained"].sum(), SUB18["compressed"].sum()) == (97, 57, 40)
GIDX = [np.array([i for i, r in enumerate(x5) if r["arena"] == f]) for f in FOLDS]   # X5 rows of each split
assert [len(g) for g in GIDX] == [51] * 5

# ------------------------------------------------------------------ 0. registered numbers, asserted exactly
say("asserting registered numbers")
P, RC = f18, [r for r in f18 if r["pool"] in ("R", "C")]
groups = [[r for r in x5 if r["arena"] == f] for f in FOLDS]
_ci = C.ci
captured = []


def _ci_capture(v, level=95.0):
    captured.append((level, np.asarray(v, dtype=float)))
    return _ci(v, level)


REG = {}
C.ci = _ci_capture                    # records the draws stats.py passes to C.ci; values returned unchanged
try:
    for name, args, kw in (("Q1", ("lam1", "E1", "E100", P), {}), ("Q2", ("lam1", "E8", "E100", P), {}),
                           ("Q4_vs_E1", ("rJ", "E1", "E100", P), {"levels": (95.0,)}),
                           ("Q4_vs_E8", ("rJ", "E8", "E100", P), {"levels": (95.0,)}),
                           ("Q5_vs_E1", ("lam1", "E1", "E100", RC), {"levels": (95.0,)}),
                           ("Q5_vs_E8", ("lam1", "E8", "E100", RC), {"levels": (95.0,)})):
        a, b_, y, rows = args
        captured.clear()
        REG[name] = boot_diff(col(rows, a), col(rows, b_), col(rows, y), **kw)
        REG[name]["draws"] = captured[0][1]
    for name, a, b_ in (("Q3_vs_E1", "lam1", "E1"), ("Q3_vs_E8", "lam1", "E8")):
        captured.clear()
        REG[name] = boot_diff_within(groups, a, b_, "E100")
        REG[name]["draws"] = captured[0][1]
finally:
    C.ci = _ci
for name, r in REG.items():
    for k, v in r.items():
        if k != "draws":
            assert v == ST[name][k], (name, k, v, ST[name][k])
# Q6, with the draws of stats.py (fresh default_rng(0), 25 parents, u30)
byk = {r["key"]: r for r in x5}
parents = [r["key"] for r in x5 if r["group"] == "A"]
pidx = np.random.default_rng(0).integers(0, len(parents), size=(B, len(parents)))
d6 = np.array([byk[f"{p}__u30"]["lam1"] - byk[p]["lam1"] for p in parents])
q6_draws = np.nanmean(np.where(np.isfinite(d6), d6, np.nan)[pidx], 1)
assert float(np.nanmean(d6)) == ST["Q6"]["mean"] and C.ci(q6_draws) == ST["Q6"]["ci95"]
REG["Q6"] = {"mean": ST["Q6"]["mean"], "ci95": ST["Q6"]["ci95"], "draws": q6_draws}
# Table 2 correlations: stats.json (registered) and exploratory.json (exploratory columns)
T2 = {}
for c in ("E1", "E8", "lam1", "rJ"):
    T2[f"F18_all_{c}"] = spearman(col(P, c), col(P, "E100"))
    T2[f"F18_trained_{c}"] = spearman(col([r for r in P if r["pool"] in ("R", "T")], c),
                                      col([r for r in P if r["pool"] in ("R", "T")], "E100"))
    T2[f"F18_compressed_{c}"] = spearman(col([r for r in P if r["pool"] == "C"], c),
                                         col([r for r in P if r["pool"] == "C"], "E100"))
    T2[f"X5_{c}"] = float(np.mean([spearman(col(g, c), col(g, "E100")) for g in groups]))
assert T2["F18_all_E1"] == ST["Q1"]["rho_base"] and T2["F18_all_E8"] == ST["Q2"]["rho_base"]
assert T2["F18_all_lam1"] == ST["Q1"]["rho_new"] and T2["F18_all_rJ"] == ST["Q4_vs_E1"]["rho_new"]
assert T2["X5_E1"] == ST["Q3_vs_E1"]["rho_base_mean"] and T2["X5_E8"] == ST["Q3_vs_E8"]["rho_base_mean"]
assert T2["X5_lam1"] == ST["Q3_vs_E1"]["rho_new_mean"] and T2["X5_rJ"] == EX["X1_rJ_vs_E1"]["rho_new_mean"]
for c in ("E1", "E8", "lam1", "rJ"):
    assert T2[f"F18_trained_{c}"] == EX["X2_training_choices"][c]["rho"]
    assert T2[f"F18_compressed_{c}"] == EX["X2_compressed"][c]["rho"]
# the same correlations with lambda_1 and r_J rebuilt from log1 and rJ (ranks unaffected by the last bit)
for sub in ("all", "trained", "compressed"):
    m = SUB18[sub]
    assert spearman(M18["lam1_368"][m], M18["E100"][m]) == T2[f"F18_{sub}_lam1"]
assert float(np.mean([spearman(MX5["lam1_368"][g], MX5["E100"][g]) for g in GIDX])) == T2["X5_lam1"]
OUT["registered_reproduced"] = {"stats_json": ["Q1", "Q2", "Q3_vs_E1", "Q3_vs_E8", "Q4_vs_E1", "Q4_vs_E8",
                                               "Q5_vs_E1", "Q5_vs_E8", "Q6", "Table 2 correlations"],
                                "exploratory_json": ["X1 rJ", "X2 training choices", "X2 compressed"],
                                "exact": True}
say("registered numbers reproduce exactly")


# ------------------------------------------------------------------ vectorised weighted Spearman
def counts(idx, n):
    """(D, m) indices in 0..n-1 -> (D, n) multiplicities."""
    D = idx.shape[0]
    return np.bincount((idx + n * np.arange(D)[:, None]).ravel(), minlength=D * n).reshape(D, n).astype(float)


def wrank(v, w):
    """Average ranks of each item in the multiset where item j appears w[d, j] times (stats._clean: non-finite
    values rank last, tied)."""
    v = np.where(np.isfinite(v), v, np.inf)
    v = np.broadcast_to(v, w.shape)
    order = np.argsort(v, axis=1, kind="stable")
    vs, ws = np.take_along_axis(v, order, 1), np.take_along_axis(w, order, 1)
    cw = np.cumsum(ws, 1)
    new = np.ones(vs.shape, bool)
    new[:, 1:] = vs[:, 1:] != vs[:, :-1]
    last = np.ones(vs.shape, bool)
    last[:, :-1] = new[:, 1:]
    below = np.maximum.accumulate(np.where(new, cw - ws, 0.0), 1)                  # weight strictly below group
    upto = np.minimum.accumulate(np.where(last, cw, np.inf)[:, ::-1], 1)[:, ::-1]  # weight up to group end
    rs = below + (upto - below + 1.0) / 2.0
    out = np.empty_like(rs)
    np.put_along_axis(out, order, rs, 1)
    return out


def wspear(x, y, w):
    """Spearman of x and y over the multiset given by weights w (D, n); equals stats.spearman on x[idx], y[idx]."""
    rx, ry = wrank(x, w), wrank(y, w)
    W = w.sum(1, keepdims=True)
    dx = rx - (w * rx).sum(1, keepdims=True) / W
    dy = ry - (w * ry).sum(1, keepdims=True) / W
    vx, vy = (w * dx * dx).sum(1), (w * dy * dy).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (w * dx * dy).sum(1) / np.sqrt(vx * vy)
    r[(vx <= 1e-9) | (vy <= 1e-9)] = np.nan
    return r


def model_draws(n):
    """The draws of stats.boot_rho and stats.boot_diff."""
    return counts(np.random.default_rng(0).integers(0, n, size=(B, n)), n)


def within_draws(sizes):
    """The draws of stats.boot_diff_within: per draw, one call per group, in group order."""
    rng = np.random.default_rng(0)
    sel = [np.empty((B, n), int) for n in sizes]
    for b in range(B):
        for s, n in zip(sel, sizes):
            s[b] = rng.integers(0, n, n)
    return [counts(s, n) for s, n in zip(sel, sizes)]


W18 = {sub: model_draws(int(m.sum())) for sub, m in SUB18.items()}
WX5 = within_draws([len(g) for g in GIDX])
WX5_4 = within_draws([len(g) for g in GIDX[1:]])                                 # without split {1, 8}
assert FOLDS[0] == "18"


def rho_f18(M, c, t, sub):
    m = SUB18[sub]
    return wspear(M[c][m], M[t][m], W18[sub])


def rho_x5(M, c, t, gi=GIDX, Ws=WX5):
    return np.mean([wspear(M[c][g], M[t][g], w) for g, w in zip(gi, Ws)], 0)


def summ(full, draws, levels=(95.0,)):
    out = {"rho": float(full)}
    for lv in levels:
        out[f"ci{lv:g}"] = C.ci(draws, lv)
    out["dropped_draws"] = int(np.sum(~np.isfinite(draws)))
    return out


def full_x5(M, c, t, gi=GIDX):
    return float(np.mean([spearman(M[c][g], M[t][g]) for g in gi]))


def close(a, b, tol=1e-12):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


# the vectorised draws reproduce stats.py's intervals (to rounding)
chk = {
    "D1 lam1~E100": (C.ci(rho_f18(M18, "lam1_368", "E100", "all")), ST["D1_rho_F18"]["lam1~E100"]["ci95"]),
    "D1 E1~E100": (C.ci(rho_f18(M18, "E1", "E100", "all")), ST["D1_rho_F18"]["E1~E100"]["ci95"]),
    "X2 trained rJ": (C.ci(rho_f18(M18, "rJ_368", "E100", "trained")), EX["X2_training_choices"]["rJ"]["ci95"]),
    "X2 compressed lam1": (C.ci(rho_f18(M18, "lam1_368", "E100", "compressed")),
                           EX["X2_compressed"]["lam1"]["ci95"]),
    "Q4_vs_E8": (C.ci(rho_f18(M18, "rJ_368", "E100", "all") - rho_f18(M18, "E8", "E100", "all")),
                 ST["Q4_vs_E8"]["ci95"]),
    "Q5_vs_E1": (C.ci(rho_f18(M18, "lam1_368", "E100", "RC") - rho_f18(M18, "E1", "E100", "RC")),
                 ST["Q5_vs_E1"]["ci95"]),
    "Q3_vs_E1": (C.ci(rho_x5(MX5, "lam1_368", "E100") - rho_x5(MX5, "E1", "E100")), ST["Q3_vs_E1"]["ci95"]),
    "X1_rJ_vs_E8": (C.ci(rho_x5(MX5, "rJ_368", "E100") - rho_x5(MX5, "E8", "E100")), EX["X1_rJ_vs_E8"]["ci95"]),
}
for k, (a, b_) in chk.items():
    assert close(a, b_), (k, a, b_)
OUT["vectorised_check"] = {k: {"vectorised": a, "stored": b_} for k, (a, b_) in chk.items()}
say("vectorised bootstrap agrees with stats.py")

# ------------------------------------------------------------------ A1 horizon-matched checks
say("A1")
CHECKS = [f"lam1_{k}" for k in KS] + [f"rJ_{k}" for k in KS] + ["E1", "E8"]
A1 = {"definition": "lam1_k: mean over trajectories of sum(log1[:min(k,367)]) / (min(k,367)*0.02 s); rJ_k: mean "
                    "over trajectories of mean r_J over forecast steps 1..k. lam1_368 and rJ_368 are the registered "
                    "lambda_1 and r_J. Intervals: models resampled (within splits on X5), 95%.",
      "F18": {}, "X5": {}}
for sub in ("all", "trained", "compressed"):
    A1["F18"][sub] = {"n": int(SUB18[sub].sum())}
    for t in TARGETS:
        m = SUB18[sub]
        A1["F18"][sub][t] = {c: summ(spearman(M18[c][m], M18[t][m]), rho_f18(M18, c, t, sub)) for c in CHECKS}
for t in TARGETS:
    A1["X5"][t] = {c: summ(full_x5(MX5, c, t), rho_x5(MX5, c, t)) for c in CHECKS}
A1["diffs_vs_E100"] = {}
for c in ("rJ_8", "rJ_100"):
    key = f"{c}_minus_E8"
    res = {}
    for sub in ("all", "trained", "compressed"):
        m = SUB18[sub]
        full = spearman(M18[c][m], M18["E100"][m]) - spearman(M18["E8"][m], M18["E100"][m])
        d = rho_f18(M18, c, "E100", sub) - rho_f18(M18, "E8", "E100", sub)
        res[f"F18_{sub}"] = {"delta": float(full), "ci95": C.ci(d), "dropped_draws": int(np.sum(~np.isfinite(d)))}
    full = full_x5(MX5, c, "E100") - full_x5(MX5, "E8", "E100")
    d = rho_x5(MX5, c, "E100") - rho_x5(MX5, "E8", "E100")
    res["X5"] = {"delta": float(full), "ci95": C.ci(d), "dropped_draws": int(np.sum(~np.isfinite(d)))}
    A1["diffs_vs_E100"][key] = res
OUT["A1_horizon_matched"] = A1

# ------------------------------------------------------------------ A2b X5 without split {1, 8}
say("A2b")
G4 = GIDX[1:]
A2b = {"splits": FOLDS[1:], "resampling": "models within splits (stats.boot_diff_within draws for 4 groups)",
       "table2_X5": {}, "diffs": {}}
for c, cc in (("E1", "E1"), ("E8", "E8"), ("lam1", "lam1_368"), ("rJ", "rJ_368")):
    A2b["table2_X5"][c] = summ(full_x5(MX5, cc, "E100", G4), rho_x5(MX5, cc, "E100", G4, WX5_4))
    A2b["table2_X5"][c]["five_split_rho"] = T2[f"X5_{c}"]
for name, a, b_, ref in (("Q3_vs_E1", "lam1_368", "E1", ST["Q3_vs_E1"]), ("Q3_vs_E8", "lam1_368", "E8",
                                                                         ST["Q3_vs_E8"]),
                         ("rJ_vs_E8", "rJ_368", "E8", EX["X1_rJ_vs_E8"])):
    full = full_x5(MX5, a, "E100", G4) - full_x5(MX5, b_, "E100", G4)
    d = rho_x5(MX5, a, "E100", G4, WX5_4) - rho_x5(MX5, b_, "E100", G4, WX5_4)
    A2b["diffs"][name] = {"delta": float(full), "ci95": C.ci(d), "dropped_draws": int(np.sum(~np.isfinite(d))),
                          "five_split_delta": ref["delta"], "five_split_ci95": ref["ci95"]}
OUT["A2b_X5_without_split18"] = A2b

# ------------------------------------------------------------------ A2c Holm over the secondary registered tests
say("A2c")
FAM = ["Q3_vs_E1", "Q3_vs_E8", "Q4_vs_E1", "Q4_vs_E8", "Q5_vs_E1", "Q5_vs_E8", "Q6"]
pv = {}
for q in FAM:
    d = REG[q]["draws"]
    d = d[np.isfinite(d)]
    assert C.ci(d, 95.0) == ST[q]["ci95"], q
    pv[q] = float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))
    qq = min(np.mean(d <= 0), np.mean(d >= 0))
    pse = float(2 * np.sqrt(qq * (1 - qq) / len(d)))                               # Monte Carlo SE of p
    pv[q] = (pv[q], pse)
order = sorted(FAM, key=lambda q: pv[q][0])
holm, stop, adj_prev = {}, False, 0.0
for i, q in enumerate(order):
    m = len(FAM) - i
    p_, pse = pv[q]
    adj = max(adj_prev, min(1.0, m * p_))
    adj_prev = adj
    rej = (not stop) and p_ <= 0.05 / m
    stop = stop or not rej
    holm[q] = {"p": p_, "p_mc_se": pse, "threshold": 0.05 / m, "p_holm": adj, "rejects_zero": bool(rej),
               "within_2se_of_threshold": bool(abs(p_ - 0.05 / m) <= 2 * pse),
               "registered_verdict": ST[q]["verdict"]}
OUT["A2c_holm"] = {"family": FAM, "alpha_familywise": 0.05, "n_draws": B,
                   "p_definition": "2*min(P(d<=0), P(d>=0)) over the registered bootstrap draws; p = 0 means no "
                                   f"draw on the other side (p < {2 / B:g})",
                   "order": order, "tests": holm,
                   "survive": [q for q in order if holm[q]["rejects_zero"]]}

# ------------------------------------------------------------------ A3 split-half
say("A3")
def arena_of(fold):
    """common.arena(fold) without building the state tensors (not needed here, so torch is not used)."""
    keep = C.tensors
    C.tensors = lambda starts: (None, None)
    try:
        return C.arena(fold)
    finally:
        C.tensors = keep


ARENA = {C.fid(f): arena_of(f) for f in C.FOLDS}
EPIS = {f: a["episode"] for f, a in ARENA.items()}
MODELS = json.load(open(os.path.join(C.RESULTS, "models.json")))
assert all(m["starts"] == ARENA["18"]["starts"] for m in MODELS["F18"])           # F18's arena is split {1, 8}
assert all(m["starts"] == ARENA[m["arena"]]["starts"] for m in MODELS["X5"])


def halves(ep):
    first, second = [], []
    for e in sorted(set(ep)):
        ii = [i for i, x in enumerate(ep) if x == e]
        assert len(ii) == 2, ep
        first.append(ii[0])
        second.append(ii[1])
    return np.array(first), np.array(second)


for f, ep in EPIS.items():
    assert len(ep) == 4 and len(set(ep)) == 2
H18 = halves(EPIS["18"])


A3 = {"episodes": EPIS, "assignment_1": "checks from the first trajectory of each episode, E100 from the second",
      "assignment_2": "swapped", "checks": ["E1", "E8", "lam1", "rJ"], "target": "E100",
      "same_half": "baseline with checks and E100 from the same trajectories (first, then second), averaged; it "
                   "uses as many trajectories per quantity as the split-half, so split minus same-half isolates "
                   "the separation of checks from target",
      "all_trajectories_rho": "Table 2 value (all 4 trajectories per quantity)",
      "intervals": "models resampled (within splits on X5), 95%; all assignments share each draw", "F18": {},
      "X5": {}}
CH3 = (("E1", "E1"), ("E8", "E8"), ("lam1", "lam1_368"), ("rJ", "rJ_368"))
HX = [halves(EPIS[f]) for f in FOLDS]


def a3_stat(cc, pairs, sub=None):
    """Mean over (check half, target half) pairs of Spearman(check, E100); F18 subset `sub`, else X5 within splits.
    Returns the full-data value and the draws."""
    fulls, dr = [], []
    for pi in pairs:
        if sub is not None:
            m, hh = SUB18[sub], H18
            x, y = V18[cc][m][:, hh[pi[0]]].mean(1), V18["E100"][m][:, hh[pi[1]]].mean(1)
            fulls.append(spearman(x, y))
            dr.append(wspear(x, y, W18[sub]))
        else:
            per, perd = [], []
            for g, hh, w in zip(GIDX, HX, WX5):
                x, y = VX5[cc][g][:, hh[pi[0]]].mean(1), VX5["E100"][g][:, hh[pi[1]]].mean(1)
                per.append(spearman(x, y))
                perd.append(wspear(x, y, w))
            fulls.append(float(np.mean(per)))
            dr.append(np.mean(perd, 0))
    return fulls, np.mean(dr, 0)


SPLIT, SAME = ((0, 1), (1, 0)), ((0, 0), (1, 1))
for where, subs in (("F18", ("all", "trained", "compressed")), ("X5", (None,))):
    for sub in subs:
        dst = A3["F18"].setdefault(sub, {}) if where == "F18" else A3["X5"]
        for c, cc in CH3:
            f_sp, d_sp = a3_stat(cc, SPLIT, sub)
            f_sm, d_sm = a3_stat(cc, SAME, sub)
            dd = d_sp - d_sm
            dst[c] = dict(summ(np.mean(f_sp), d_sp), assignment_rhos=f_sp,
                          same_half=summ(np.mean(f_sm), d_sm),
                          split_minus_same_half={"delta": float(np.mean(f_sp) - np.mean(f_sm)), "ci95": C.ci(dd),
                                                 "dropped_draws": int(np.sum(~np.isfinite(dd)))},
                          all_trajectories_rho=T2[f"F18_{sub}_{c}" if sub else f"X5_{c}"])
OUT["A3_split_half"] = A3

# ------------------------------------------------------------------ A2a two-level bootstrap
say("A2a")


def clusters(rows):
    par = [r["key"].split("__")[0] for r in rows]
    uniq = list(dict.fromkeys(par))
    return np.array([uniq.index(p) for p in par]), len(uniq)


CL18, NC18 = clusters(f18)
# each F18 subset resamples its own clusters, so the number of clusters in a subset is fixed in every draw
CLS18 = {sub: clusters([r for r, keep in zip(f18, SUB18[sub]) if keep]) for sub in ("all", "trained", "compressed",
                                                                                        "RC")}
assert {s: n for s, (_, n) in CLS18.items()} == {"all": 57, "trained": 57, "compressed": 5, "RC": 48}
CLX = [clusters([x5[i] for i in g]) for g in GIDX]
assert NC18 == 57 and all(n == 11 for _, n in CLX)
VARS = ("E1", "E8", "E24", "E100", "E368", "lam1_368", "rJ_368")
assert all(np.isfinite(V[v]).all() for V in (V18, VX5) for v in VARS)       # 0 * inf would poison tc @ V.T
assert V18["E1"].shape[1] == 4 and all(VX5["E1"][g].shape[1] == 4 for g in GIDX)
PAR_U30 = [[(list(g).index(j), list(g).index(next(i for i in g if x5[i]["key"] == f"{x5[j]['key']}__u30")))
            for j in g if x5[j]["group"] == "A"] for g in GIDX]
assert sum(len(p) for p in PAR_U30) == 25


def draw_vals(V, rows_idx, tc):
    """Model means over the drawn trajectories: V[var][rows] (n, ntraj), tc (D, ntraj) -> (D, n)."""
    return {v: tc @ V[v][rows_idx].T / tc.sum(1, keepdims=True) for v in VARS}


STAT_NAMES = (["F18_all_" + c for c in ("E1", "E8", "lam1", "rJ")] +
              ["F18_trained_" + c for c in ("E1", "E8", "lam1", "rJ")] +
              ["F18_compressed_" + c for c in ("E1", "E8", "lam1", "rJ")] +
              ["X5_" + c for c in ("E1", "E8", "lam1", "rJ")] +
              ["Q1", "Q2", "Q1_E24", "Q2_E24", "Q1_E368", "Q2_E368", "Q3_vs_E1", "Q3_vs_E8", "Q4_vs_E1", "Q4_vs_E8",
               "Q5_vs_E1", "Q5_vs_E8", "Q6"])
CN = {"E1": "E1", "E8": "E8", "lam1": "lam1_368", "rJ": "rJ_368"}


def two_level(n_draws, rng):
    t18 = counts(rng.integers(0, 4, size=(n_draws, 4)), 4)
    c18 = {sub: counts(rng.integers(0, nc, size=(n_draws, nc)), nc) for sub, (_, nc) in CLS18.items()}
    tx, cx = [], []
    for (cl, nc) in CLX:
        tx.append(counts(rng.integers(0, 4, size=(n_draws, 4)), 4))
        cx.append(counts(rng.integers(0, nc, size=(n_draws, nc)), nc))
    res = {k: np.empty(n_draws) for k in STAT_NAMES}
    for lo in range(0, n_draws, 2500):
        sl = slice(lo, min(lo + 2500, n_draws))
        Mv = draw_vals(V18, np.arange(len(f18)), t18[sl])
        r = {}
        for sub in ("all", "trained", "compressed", "RC"):
            m = SUB18[sub]
            for c in ("E1", "E8", "lam1", "rJ"):
                for t in TARGETS:
                    if sub == "all" or t == "E100":
                        r[(sub, c, t)] = wspear(Mv[CN[c]][:, m], Mv[t][:, m], c18[sub][sl][:, CLS18[sub][0]])
        for sub in ("all", "trained", "compressed"):
            for c in ("E1", "E8", "lam1", "rJ"):
                res[f"F18_{sub}_{c}"][sl] = r[(sub, c, "E100")]
        res["Q1"][sl] = r[("all", "lam1", "E100")] - r[("all", "E1", "E100")]
        res["Q2"][sl] = r[("all", "lam1", "E100")] - r[("all", "E8", "E100")]
        for t in ("E24", "E368"):
            res[f"Q1_{t}"][sl] = r[("all", "lam1", t)] - r[("all", "E1", t)]
            res[f"Q2_{t}"][sl] = r[("all", "lam1", t)] - r[("all", "E8", t)]
        res["Q4_vs_E1"][sl] = r[("all", "rJ", "E100")] - r[("all", "E1", "E100")]
        res["Q4_vs_E8"][sl] = r[("all", "rJ", "E100")] - r[("all", "E8", "E100")]
        res["Q5_vs_E1"][sl] = r[("RC", "lam1", "E100")] - r[("RC", "E1", "E100")]
        res["Q5_vs_E8"][sl] = r[("RC", "lam1", "E100")] - r[("RC", "E8", "E100")]
        xr = {c: [] for c in ("E1", "E8", "lam1", "rJ")}
        q6n, q6w = 0.0, 0.0
        for gi, (g, (cl, nc)) in enumerate(zip(GIDX, CLX)):
            Mg = draw_vals(VX5, g, tx[gi][sl])
            wg = cx[gi][sl][:, cl]
            for c in xr:
                xr[c].append(wspear(Mg[CN[c]], Mg["E100"], wg))
            for jp, ju in PAR_U30[gi]:
                q6n = q6n + wg[:, jp] * (Mg["lam1_368"][:, ju] - Mg["lam1_368"][:, jp])
                q6w = q6w + wg[:, jp]
        for c in xr:
            res[f"X5_{c}"][sl] = np.mean(xr[c], 0)
        res["Q3_vs_E1"][sl] = res["X5_lam1"][sl] - res["X5_E1"][sl]
        res["Q3_vs_E8"][sl] = res["X5_lam1"][sl] - res["X5_E8"][sl]
        with np.errstate(invalid="ignore", divide="ignore"):
            res["Q6"][sl] = q6n / q6w
    return res


t_pilot = time.time()
two_level(1000, np.random.default_rng(0))
t_pilot = time.time() - t_pilot
N2 = 20000 if t_pilot * 20 < 480 else 5000
t2 = time.time()
TL = two_level(N2, np.random.default_rng(0))
t2 = time.time() - t2
say(f"A2a {N2} draws in {t2:.0f}s")

# registered (model-resampled) intervals for the same statistics
reg_ci = {}
for sub, src in (("all", None), ("trained", EX["X2_training_choices"]), ("compressed", EX["X2_compressed"])):
    for c in ("E1", "E8", "lam1", "rJ"):
        reg_ci[f"F18_{sub}_{c}"] = (ST["D1_rho_F18"][f"{c}~E100"]["ci95"] if src is None else src[c]["ci95"],
                                    "stats.json D1_rho_F18" if src is None else "exploratory.json X2")
for c in ("E1", "E8", "lam1", "rJ"):
    reg_ci[f"X5_{c}"] = (A1["X5"]["E100"][CN[c]]["ci95"], "this script, A1 (models resampled within splits)")
A2a = {"n_draws": N2, "pilot_1000_draws_s": round(t_pilot, 1), "run_s": round(t2, 1),
       "rng": "numpy.random.default_rng(0); order: F18 trajectories (D,4), F18 clusters of each subset (all D,57; "
              "trained D,57; compressed D,5; RC D,48), then per split "
              "{18,02,34,56,79}: trajectories (D,4), clusters (D,11)",
       "clusters": "a parent with its eight compressed copies; every other model its own cluster (F18: 57, "
                   "X5: 11 per split); each F18 subset resamples its own clusters (compressed: only 5, too few "
                   "for a reliable percentile interval). Q6 resamples parents within splits (registered Q6 pools "
                   "the 25 parents), so its width ratio mixes stratification with trajectory resampling. The 4 "
                   "trajectories of an arena come from 2 episodes; the unit resampled is the trajectory, as "
                   "specified, which may understate width if the episode is the independent unit.",
       "descriptive_only": True, "table2": {}, "registered_tests": {}}
for k in STAT_NAMES[:16]:
    d = TL[k]
    ci_reg, src = reg_ci[k]
    ci2 = C.ci(d)
    A2a["table2"][k] = {"rho": T2[k], "ci95_registered_scheme": ci_reg, "registered_scheme_source": src,
                        "ci95_two_level": ci2, "width_ratio": (ci2[1] - ci2[0]) / (ci_reg[1] - ci_reg[0]),
                        "dropped_draws": int(np.sum(~np.isfinite(d)))}
for q in STAT_NAMES[16:]:
    d = TL[q]
    lv = 97.5 if q in ("Q1", "Q2") else 95.0
    ci_reg = ST[q][f"ci{lv:g}"]
    ci2 = C.ci(d, lv)
    A2a["registered_tests"][q] = {
        "estimate": ST[q].get("delta", ST[q].get("mean")), "level": lv, "ci_registered": ci_reg,
        "ci_two_level": ci2, "width_ratio": (ci2[1] - ci2[0]) / (ci_reg[1] - ci_reg[0]),
        "registered_verdict": ST[q]["verdict"], "two_level_excludes_zero": bool(ci2[0] > 0 or ci2[1] < 0),
        "dropped_draws": int(np.sum(~np.isfinite(d)))}
    if lv == 97.5:
        A2a["registered_tests"][q]["ci95_two_level"] = C.ci(d, 95.0)
OUT["A2a_two_level"] = A2a
OUT["run_s"] = round(time.time() - T0, 1)


# ------------------------------------------------------------------ write
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


json.dump(clean(OUT), open(os.path.join(C.RESULTS, "revision_exploratory.json"), "w"), indent=1)

N_DROP = sum(v["dropped_draws"] for part in ("table2", "registered_tests") for v in A2a[part].values())
f2 = lambda v: f"{v:+.2f}"                                                         # noqa: E731
cs = lambda c: f"[{c[0]:+.2f}, {c[1]:+.2f}]"                                       # noqa: E731
c3 = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]{'*' if c[0] > 0 or c[1] < 0 else ''}"   # noqa: E731
L = ["REVISION EXPLORATORY ANALYSES (R1) -- nothing here is registered; registered verdicts stand.",
     f"Registered Q1-Q6, Table 2 rhos reproduce exactly (lam1 from log1 to 2e-16). Run {OUT['run_s']:.0f}s; "
     f"numpy {np.__version__}. * = 95% CI excludes 0.",
     "A1 horizon-matched checks, Spearman with E100 [95% CI, models resampled]",
     "  check        F18 all            F18 trained        F18 compressed     X5 (within splits)"]
for c in CHECKS:
    a = [A1["F18"][s]["E100"][c] for s in ("all", "trained", "compressed")] + [A1["X5"]["E100"][c]]
    L.append(f"  {c:<10} " + " ".join(f"{f2(x['rho'])} {cs(x['ci95'])}" for x in a))
L.append("  same checks with E24 / E368 (rho on F18 all; X5):")
for c in ("lam1_8", "lam1_368", "rJ_8", "rJ_368", "E1", "E8"):
    L.append(f"  {c:<10} E24 {f2(A1['F18']['all']['E24'][c]['rho'])}; {f2(A1['X5']['E24'][c]['rho'])}   "
             f"E368 {f2(A1['F18']['all']['E368'][c]['rho'])}; {f2(A1['X5']['E368'][c]['rho'])}")
for k, v in A1["diffs_vs_E100"].items():
    L.append(f"  {k} (vs E100): " + "; ".join(f"{s} {x['delta']:+.3f} {c3(x['ci95'])}" for s, x in v.items()))
L.append(f"A2a two-level bootstrap ({N2} draws; clusters x trajectories; compressed has only 5 clusters), "
         f"descriptive; 95% CI, width ratio to registered; dropped draws {N_DROP}")
for pre in ("F18_all", "F18_trained", "F18_compressed", "X5"):
    L.append(f"  {pre:<14} " + "; ".join(f"{c} {f2(A2a['table2'][f'{pre}_{c}']['rho'])} "
                                         f"{cs(A2a['table2'][f'{pre}_{c}']['ci95_two_level'])} "
                                         f"x{A2a['table2'][f'{pre}_{c}']['width_ratio']:.1f}"
                                         for c in ("E1", "E8", "lam1", "rJ")))
for q, v in A2a["registered_tests"].items():
    L.append(f"  {q:<9} {f2(v['estimate'])} {v['level']:g}%: reg {cs(v['ci_registered'])} two-level "
             f"{cs(v['ci_two_level'])} x{v['width_ratio']:.1f}  {v['registered_verdict']}; excludes 0: "
             f"{'yes' if v['two_level_excludes_zero'] else 'NO'}")
L += ["A2b X5 without split {1,8} (4 splits; five-split value in brackets)"]
L.append("  " + "; ".join(f"{c} {f2(v['rho'])} {cs(v['ci95'])} ({f2(v['five_split_rho'])})"
                          for c, v in A2b["table2_X5"].items() if c in ("E1", "E8")))
L.append("  " + "; ".join(f"{c} {f2(v['rho'])} {cs(v['ci95'])} ({f2(v['five_split_rho'])})"
                          for c, v in A2b["table2_X5"].items() if c in ("lam1", "rJ")))
for q, v in A2b["diffs"].items():
    L.append(f"  {q}: {v['delta']:+.3f} {c3(v['ci95'])} (five splits {v['five_split_delta']:+.3f} "
             f"{c3(v['five_split_ci95'])})")
L += ["A2c Holm over Q3-Q6 (family-wise 5%)"]
for q in order:
    v = holm[q]
    L.append(f"  {q:<9} p={v['p']:.4f} (MC SE {v['p_mc_se']:.4f}) threshold {v['threshold']:.4f} "
             f"p_holm={v['p_holm']:.4f} {'survives' if v['rejects_zero'] else 'does not survive'}"
             f"{' (within 2 MC SE)' if v['within_2se_of_threshold'] else ''} ({v['registered_verdict']})")
L += ["A3 split-half (checks from one trajectory per episode, E100 from the other; mean of both): rho [CI] | "
      "same-half rho | split minus same-half [CI]; columns F18 all, trained, compressed, X5"]
for c in ("E1", "E8", "lam1", "rJ"):
    a = [A3["F18"][s][c] for s in ("all", "trained", "compressed")] + [A3["X5"][c]]
    L.append(f"  {c:<5} " + " || ".join(f"{f2(x['rho'])} {cs(x['ci95'])} | {f2(x['same_half']['rho'])} | "
                                        f"{x['split_minus_same_half']['delta']:+.3f} "
                                        f"{c3(x['split_minus_same_half']['ci95'])}" for x in a))
assert len(L) <= 60, len(L)
open(os.path.join(C.RESULTS, "revision_exploratory.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

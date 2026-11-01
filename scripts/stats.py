"""
Every registered question and descriptive result of PREREG.md §5, from results/raw_*.npz and models.json.
Writes results/stats.json, results/model_table.json and results/stats_report.txt.
"""
import json
import os

import numpy as np

import common as C
from ranks import rankdata

B = C.N_BOOT
HS = (1, 8, 24, 100)
OUT = {}


# ------------------------------------------------------------------ per-model table
def model_table(tag):
    raw = np.load(os.path.join(C.RESULTS, f"raw_{tag}.npz"))
    meta = json.load(open(os.path.join(C.RESULTS, "models.json")))[tag]
    rows = []
    for m in meta:
        k = m["key"]
        r = {"key": k, "pool": m["pool"], "group": m["group"], "variant": m["variant"], "arena": m["arena"],
             "extra": m["extra"], "finite": m["finite"]}
        for h in HS:
            r[f"E{h}"] = float(np.mean(raw[f"E{h}_{k}"]))
        r["E368"] = float(np.mean(raw[f"err_{k}"]))
        r["lam1"] = float(np.mean(raw[f"lam_{k}"][:, 0]))
        r["rJ"] = float(np.mean(raw[f"rJ_{k}"]))
        rows.append(r)
    return rows, raw


def floors(raw, fold):
    f = {f"E{h}": float(np.mean(raw[f"floor_E{h}_{fold}"])) for h in HS}
    f["E368"] = float(np.mean(raw[f"floor_{fold}"]))
    return f


# ------------------------------------------------------------------ correlations
def _clean(x):
    x = np.asarray(x, dtype=float).copy()
    x[~np.isfinite(x)] = np.inf
    return x


def spearman(a, b):
    ra, rb = rankdata(_clean(a)), rankdata(_clean(b))
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def kendall_tau_b(a, b):
    a, b = _clean(a), _clean(b)
    n = len(a)
    da = np.sign(a[:, None] - a[None, :])
    db = np.sign(b[:, None] - b[None, :])
    iu = np.triu_indices(n, 1)
    s = (da * db)[iu].sum()
    n0 = n * (n - 1) / 2
    t1 = (da[iu] == 0).sum()
    t2 = (db[iu] == 0).sum()
    return float(s / np.sqrt((n0 - t1) * (n0 - t2)))


def boot_diff(x_new, x_base, y, levels=(95.0, 97.5)):
    """rho(x_new, y) - rho(x_base, y) over models; paired model bootstrap."""
    x_new, x_base, y = map(np.asarray, (x_new, x_base, y))
    n = len(y)
    rng = np.random.default_rng(0)
    draws = rng.integers(0, n, size=(B, n))
    d, dropped = [], 0
    for idx in draws:
        a, b_ = spearman(x_new[idx], y[idx]), spearman(x_base[idx], y[idx])
        if not (np.isfinite(a) and np.isfinite(b_)):
            dropped += 1
            continue
        d.append(a - b_)
    d = np.array(d)
    out = {"rho_new": spearman(x_new, y), "rho_base": spearman(x_base, y),
           "tau_new": kendall_tau_b(x_new, y), "tau_base": kendall_tau_b(x_base, y), "n_models": int(n),
           "dropped_draws": int(dropped)}
    out["delta"] = out["rho_new"] - out["rho_base"]
    for lv in levels:
        out[f"ci{lv:g}"] = C.ci(d, lv)
    return out


def boot_rho(x, y, level=95.0):
    x, y = np.asarray(x), np.asarray(y)
    n = len(y)
    rng = np.random.default_rng(0)
    vals = []
    for idx in rng.integers(0, n, size=(B, n)):
        r = spearman(x[idx], y[idx])
        if np.isfinite(r):
            vals.append(r)
    return {"rho": spearman(x, y), "ci95": C.ci(vals, level), "tau": kendall_tau_b(x, y)}


def boot_diff_within(groups, new, base, y):
    """Mean over groups of within-group rho(new, y) - rho(base, y); models resampled within each group."""
    rng = np.random.default_rng(0)
    ga = [{k: np.asarray([r[k] for r in g]) for k in (new, base, y)} for g in groups]

    def stat(sel):
        a = [spearman(g[new][i], g[y][i]) for g, i in zip(ga, sel)]
        b_ = [spearman(g[base][i], g[y][i]) for g, i in zip(ga, sel)]
        return float(np.mean(a)), float(np.mean(b_))

    full = stat([np.arange(len(g[y])) for g in ga])
    d, dropped = [], 0
    for _ in range(B):
        sel = [rng.integers(0, len(g[y]), len(g[y])) for g in ga]
        a, b_ = stat(sel)
        if not (np.isfinite(a) and np.isfinite(b_)):
            dropped += 1
            continue
        d.append(a - b_)
    return {"rho_new_mean": full[0], "rho_base_mean": full[1], "delta": full[0] - full[1],
            "ci95": C.ci(d, 95.0), "dropped_draws": int(dropped), "n_groups": len(groups),
            "n_per_group": [len(g) for g in groups]}


def lab(ci_, pos, neg):
    return C.label3(ci_, pos, neg)


# ------------------------------------------------------------------ main
def main():
    f18, raw18 = model_table("F18")
    x5, rawx5 = model_table("X5")
    rel, rawrel = model_table("REL")
    json.dump({"F18": f18, "X5": x5, "REL": rel}, open(os.path.join(C.RESULTS, "model_table.json"), "w"), indent=1)
    OUT["n_models"] = {"F18": len(f18), "X5": len(x5)}
    OUT["n_nonfinite"] = {"F18": sum(not r["finite"] for r in f18), "X5": sum(not r["finite"] for r in x5)}
    fl18 = floors(raw18, "18")
    OUT["floor_F18"] = fl18

    P = f18
    col = lambda rows, k: [r[k] for r in rows]                                    # noqa: E731

    # Q1, Q2 (primary, 97.5% for the verdict)
    q1 = boot_diff(col(P, "lam1"), col(P, "E1"), col(P, "E100"))
    q1["verdict"] = lab(q1["ci97.5"], "LABEL-FREE BETTER", "ONE-STEP BETTER")
    q2 = boot_diff(col(P, "lam1"), col(P, "E8"), col(P, "E100"))
    q2["verdict"] = lab(q2["ci97.5"], "LABEL-FREE BETTER", "TRAINING-HORIZON BETTER")
    OUT["Q1"], OUT["Q2"] = q1, q2
    # secondary targets
    for tgt in ("E24", "E368"):
        a = boot_diff(col(P, "lam1"), col(P, "E1"), col(P, tgt), levels=(95.0,))
        a["verdict"] = lab(a["ci95"], "LABEL-FREE BETTER", "ONE-STEP BETTER")
        b_ = boot_diff(col(P, "lam1"), col(P, "E8"), col(P, tgt), levels=(95.0,))
        b_["verdict"] = lab(b_["ci95"], "LABEL-FREE BETTER", "TRAINING-HORIZON BETTER")
        OUT[f"Q1_{tgt}"], OUT[f"Q2_{tgt}"] = a, b_

    # Q3: X5 within folds
    groups = [[r for r in x5 if r["arena"] == C.fid(f)] for f in C.FOLDS]
    q3a = boot_diff_within(groups, "lam1", "E1", "E100")
    q3a["verdict"] = lab(q3a["ci95"], "LABEL-FREE BETTER", "ONE-STEP BETTER")
    q3b = boot_diff_within(groups, "lam1", "E8", "E100")
    q3b["verdict"] = lab(q3b["ci95"], "LABEL-FREE BETTER", "TRAINING-HORIZON BETTER")
    OUT["Q3_vs_E1"], OUT["Q3_vs_E8"] = q3a, q3b

    # Q4: physics residual on P
    q4a = boot_diff(col(P, "rJ"), col(P, "E1"), col(P, "E100"), levels=(95.0,))
    q4a["verdict"] = lab(q4a["ci95"], "LABEL-FREE BETTER", "ONE-STEP BETTER")
    q4b = boot_diff(col(P, "rJ"), col(P, "E8"), col(P, "E100"), levels=(95.0,))
    q4b["verdict"] = lab(q4b["ci95"], "LABEL-FREE BETTER", "TRAINING-HORIZON BETTER")
    OUT["Q4_vs_E1"], OUT["Q4_vs_E8"] = q4a, q4b

    # Q5: R and C only
    RC = [r for r in P if r["pool"] in ("R", "C")]
    q5a = boot_diff(col(RC, "lam1"), col(RC, "E1"), col(RC, "E100"), levels=(95.0,))
    q5a["verdict"] = lab(q5a["ci95"], "LABEL-FREE BETTER", "ONE-STEP BETTER")
    q5b = boot_diff(col(RC, "lam1"), col(RC, "E8"), col(RC, "E100"), levels=(95.0,))
    q5b["verdict"] = lab(q5b["ci95"], "LABEL-FREE BETTER", "TRAINING-HORIZON BETTER")
    OUT["Q5_vs_E1"], OUT["Q5_vs_E8"] = q5a, q5b

    # Q6 and per-variant lambda shifts on X5 (paired over the 25 parents)
    byk = {r["key"]: r for r in x5}
    parents = [r["key"] for r in x5 if r["group"] == "A"]
    rng = np.random.default_rng(0)
    pidx = rng.integers(0, len(parents), size=(B, len(parents)))
    shifts = {}
    for v in ("fp16", "int8w", "u10", "u20", "u30", "u50", "s192", "s128"):
        d = np.array([byk[f"{p}__{v}"]["lam1"] - byk[p]["lam1"] for p in parents])
        dd = np.where(np.isfinite(d), d, np.nan)
        draws = np.nanmean(dd[pidx], 1)
        shifts[v] = {"mean": float(np.nanmean(dd)), "ci95": C.ci(draws), "n_parents": len(parents),
                     "n_positive_lam1": int(sum(byk[f"{p}__{v}"]["lam1"] > 0 for p in parents)),
                     "mean_lam1": float(np.nanmean([byk[f"{p}__{v}"]["lam1"] for p in parents]))}
    OUT["lam_shift_X5"] = shifts
    OUT["Q6"] = dict(shifts["u30"], verdict=lab(shifts["u30"]["ci95"], "MORE EXPANSIVE", "MORE CONTRACTIVE"))
    OUT["parent_mean_lam1_X5"] = float(np.mean([byk[p]["lam1"] for p in parents]))

    # D1: group means and all correlations on P and X (with tau)
    def gmeans(rows):
        out = {}
        for g in sorted({r["group"] for r in rows}):
            rr = [r for r in rows if r["group"] == g]
            out[g] = {k: float(np.mean([r[k] for r in rr])) for k in ("E1", "E8", "E24", "E100", "E368",
                                                                       "lam1", "rJ")}
            out[g]["n"] = len(rr)
        return out
    OUT["D1_groups_F18"], OUT["D1_groups_X5"] = gmeans(f18), gmeans(x5)
    OUT["D1_rho_F18"] = {f"{c}~{t}": boot_rho(col(P, c), col(P, t)) for c in ("E1", "E8", "E24", "lam1", "rJ")
                         for t in ("E24", "E100", "E368") if c != t}
    OUT["D1_rho_RC"] = {f"{c}~E100": boot_rho(col(RC, c), col(RC, "E100")) for c in ("E1", "E8", "lam1", "rJ")}

    # D2: skill retained by compressed variants on X5 (trajectory bootstrap, seeds averaged per trajectory)
    def traj_vals(key, h):
        return rawx5[f"err_{key}"].mean(1) if h == 368 else rawx5[f"E{h}_{key}"]
    folds = [C.fid(f) for f in C.FOLDS]
    rngt = np.random.default_rng(0)
    tdraw = rngt.integers(0, 20, size=(B, 20))
    D2 = {}
    for v in ("fp16", "int8w", "u10", "u20", "u30", "u50", "s192", "s128"):
        D2[v] = {}
        for h in HS + (368,):
            par, var, flo = [], [], []
            for f in folds:
                ps = [k for k in parents if k.split("_")[1] == f]
                par.append(np.mean([traj_vals(k, h) for k in ps], 0))
                var.append(np.mean([traj_vals(f"{k}__{v}", h) for k in ps], 0))
                flo.append(rawx5[f"floor_{f}"].mean(1) if h == 368 else rawx5[f"floor_E{h}_{f}"])
            par, var, flo = map(np.concatenate, (par, var, flo))
            dlt, skl = var - par, flo - par
            Kfull = 1 - dlt.mean() / skl.mean()
            Kd = 1 - dlt[tdraw].mean(1) / skl[tdraw].mean(1)
            D2[v][f"h{h}"] = {"K": float(Kfull), "ci95": C.ci(Kd)}
    OUT["D2_skill_retained_X5"] = D2

    # D3: Arm A against Arm B (and the ensemble) on X5, normalised and physical errors
    def arm_traj(kind, getter):
        vals = []
        for f in folds:
            ks = [r["key"] for r in x5 if r["group"] == kind and r["arena"] == f]
            vals.append(np.mean([getter(k) for k in ks], 0))
        return np.concatenate(vals)
    D3 = {}
    for h in HS + (368,):
        A = arm_traj("A", lambda k: traj_vals(k, h))
        Bv = arm_traj("B", lambda k: traj_vals(k, h))
        A5 = arm_traj("A5", lambda k: traj_vals(k, h))
        fl = np.concatenate([rawx5[f"floor_{f}"].mean(1) if h == 368 else rawx5[f"floor_E{h}_{f}"] for f in folds])
        dd = (Bv - A)
        D3[f"h{h}"] = {"A": float(A.mean()), "B": float(Bv.mean()), "A5": float(A5.mean()), "floor": float(fl.mean()),
                       "B_minus_A": float(dd.mean()), "ci95": C.ci(dd[tdraw].mean(1)),
                       "B_over_A": float(Bv.mean() / A.mean()), "n_traj_B_worse": int((dd > 0).sum())}
    phys = {}
    for name in ("jdeg", "vel", "pos"):
        for h in (1, 24, 100, 368):
            if name == "pos":
                get = lambda k, n=name: rawx5[f"{n}_{k}"][:, h - 1]                      # noqa: E731
                flo = np.concatenate([rawx5[f"floor_{name}_{f}"][:, h - 1] for f in folds])
            else:
                get = lambda k, n=name: rawx5[f"{n}_{k}"][:, :h].mean(1)                 # noqa: E731
                flo = np.concatenate([rawx5[f"floor_{name}_{f}"][:, :h].mean(1) for f in folds])
            A, Bv, A5 = arm_traj("A", get), arm_traj("B", get), arm_traj("A5", get)
            dd = Bv - A
            phys[f"{name}_h{h}"] = {"A": float(A.mean()), "B": float(Bv.mean()), "A5": float(A5.mean()),
                                    "floor": float(flo.mean()), "B_minus_A_ci95": C.ci(dd[tdraw].mean(1)),
                                    "n_traj_B_worse": int((dd > 0).sum())}
    dist = np.concatenate([rawx5[f"dist_{f}"] for f in folds])
    phys["dist_recorded"] = {f"h{h}": float(dist[:, h - 1].mean()) for h in (24, 100, 368)}
    D3["physical"] = phys
    D3["lam1"] = {k: float(np.mean([r["lam1"] for r in x5 if r["group"] == k])) for k in ("A", "A5", "B")}
    D3["lam1_positive_share"] = {k: float(np.mean([r["lam1"] > 0 for r in x5 if r["group"] == k]))
                                 for k in ("A", "A5", "B")}
    OUT["D3_arms_X5"] = D3

    # D4: worse than the floor at E100, by the sign of lambda_1
    def d4(rows, flo_of):
        out = {"lam_pos": [0, 0], "lam_nonpos": [0, 0]}                                # [worse than floor, n]
        for r in rows:
            k = "lam_pos" if r["lam1"] > 0 else "lam_nonpos"
            out[k][1] += 1
            out[k][0] += int(r["E100"] > flo_of(r))
        return out
    OUT["D4_F18"] = d4(f18, lambda r: fl18["E100"])
    flx = {f: floors(rawx5, f) for f in folds}
    OUT["D4_X5"] = d4(x5, lambda r: flx[r["arena"]]["E100"])

    # D5: the released checkpoint (in-sample arena: all 20 trajectories)
    r = rel[0]
    flr = floors(rawrel, "all")
    D5 = {k: r[k] for k in ("E1", "E8", "E24", "E100", "E368", "lam1", "rJ")}
    D5["floor"] = flr
    D5["lam1_positive_share_traj"] = float(np.mean(rawrel["lam_REL"][:, 0] > 0))
    D5["pos_h100"] = float(rawrel["pos_REL"][:, 99].mean())
    D5["pos_h368"] = float(rawrel["pos_REL"][:, 367].mean())
    D5["jdeg_h368"] = float(rawrel["jdeg_REL"].mean())
    OUT["D5_released"] = D5

    # D6: choosing a training configuration on F18
    D6 = {"picks": {}, "regret": {}}
    by18 = {x["key"]: x for x in f18}
    for c in ("E1", "E8", "lam1", "rJ"):
        regs = []
        for s in range(3):
            keys = [f"mn_M{M}_N{N}_s{s}" for M, N in ((1, 8), (2, 8), (8, 8), (16, 8), (32, 2), (32, 16), (32, 32),
                                                      (32, 1))] + [f"A_s{s}"]
            pick = min(keys, key=lambda k: by18[k][c])
            best = min(by18[k]["E100"] for k in keys)
            regs.append(by18[pick]["E100"] - best)
            D6["picks"][f"{c}_s{s}"] = pick
        D6["regret"][c] = {"per_seed": regs, "mean": float(np.mean(regs))}
    D6["best_E100_mean"] = float(np.mean([min(by18[k]["E100"] for k in
                                              [f"mn_M{M}_N{N}_s{s}" for M, N in ((1, 8), (2, 8), (8, 8), (16, 8),
                                                                                  (32, 2), (32, 16), (32, 32), (32, 1))]
                                              + [f"A_s{s}"]) for s in range(3)]))
    OUT["D6_config_choice_F18"] = D6

    json.dump(OUT, open(os.path.join(C.RESULTS, "stats.json"), "w"), indent=1)
    with open(os.path.join(C.RESULTS, "stats_report.txt"), "w") as f:
        for k in ("Q1", "Q2", "Q1_E24", "Q2_E24", "Q1_E368", "Q2_E368", "Q3_vs_E1", "Q3_vs_E8", "Q4_vs_E1",
                  "Q4_vs_E8", "Q5_vs_E1", "Q5_vs_E8", "Q6"):
            v = OUT[k]
            f.write(f"{k}: {v.get('verdict')}  delta/mean={v.get('delta', v.get('mean')):+.4f}  "
                    f"ci95={v.get('ci95')}  ci97.5={v.get('ci97.5', '-')}  "
                    f"rho_new={v.get('rho_new', v.get('rho_new_mean', '-'))}  "
                    f"rho_base={v.get('rho_base', v.get('rho_base_mean', '-'))}\n")
    print(open(os.path.join(C.RESULTS, "stats_report.txt")).read())


if __name__ == "__main__":
    main()

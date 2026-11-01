"""Study FT's registered analysis (prereg/PREREG_FT.md, items 5-7), from results/raw_FT.npz and results/ft_models.json.

  ./py scripts/ft_stats.py     -> results/ft_stats.json and results/ft_report.txt (at most 60 lines)

Unit: the trajectory (20: four per X5 fold, folds in common.FOLDS order). Per arm and trajectory the error is first
averaged over the three seeds. K(x, ref, h) = 1 - mean_t(e_x - e_ref) / mean_t(e_floor - e_ref), the estimator of
stats.py's D2 (ref = P gives K_orig, ref = CTRL gives K_ctrl). Intervals: percentile intervals over 20,000 draws of
the 20 trajectories with replacement, numpy.random.default_rng(0).integers(0, 20, size=(20000, 20)); one set of
draws for every quantity. A draw on which the reference's skill over the floor is <= 0 leaves K undefined: it counts
as -inf (PerconAI PREREG_R2.md 2.1), and in a difference of two K it widens both ends; such draws are counted. Refuses to run if any planned model's score or any check result is missing.
"""
import json
import os
import sys

import numpy as np

import common as C

B = 20000
NT = 20
SEEDS = (0, 1, 2)
HS = (1, 8, 24, 100, 368)
FOLDS = [C.fid(f) for f in C.FOLDS]
CHECKS = ("C-FT1", "C-FT2", "C-FT3")          # an arm failing one of these is excluded from its questions
FT_ARMS = ("U30F", "U50F")


def draws():
    return np.random.default_rng(0).integers(0, NT, size=(B, NT))


# ------------------------------------------------------------------ per-trajectory values
def traj_vals(raw, key, h):
    return raw[f"err_{key}"].mean(1) if h == 368 else raw[f"E{h}_{key}"]


def floor_vals(raw, f, h):
    return raw[f"floor_{f}"].mean(1) if h == 368 else raw[f"floor_E{h}_{f}"]


def arm_traj(raw, keys_by_fold, h, get=traj_vals):
    """Concatenate, over folds, the per-trajectory value averaged over the fold's models (seeds)."""
    return np.concatenate([np.mean([get(raw, k, h) for k in keys_by_fold[f]], 0) for f in FOLDS])


def floor_traj(raw, h):
    return np.concatenate([floor_vals(raw, f, h) for f in FOLDS])


# ------------------------------------------------------------------ the estimator (stats.py D2, unchanged)
def k_skill(x, ref, flo, tdraw):
    """(K on the full sample, K on every draw, mean skill of ref over the floor on the full sample).

    D2's lines, with the arithmetic held at the inputs' dtype (float32 scores): numpy 2 (the cloud, NEP 50) keeps
    `1 - float32` in float32 while numpy 1.26 (this Mac) promotes it to float64; holding the dtype gives the same
    digits on both, and reproduces stats.json's D2 exactly (scripts/test_ft.py, test v).
    A draw on which the reference's skill over the floor is <= 0 leaves K undefined; as in PerconAI's PREREG_R2.md
    section 2.1 it counts as K = -inf, sorts below every finite draw, and an endpoint that falls on it is -inf.
    """
    dlt, skl = x - ref, flo - ref
    one = np.asarray(1, dtype=dlt.dtype)
    Kfull = one - (dlt.mean() / skl.mean()).astype(dlt.dtype)
    sd = skl[tdraw].mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Kd = (one - dlt[tdraw].mean(1) / sd).astype(dlt.dtype)
    Kd[sd <= 0] = -np.inf
    return Kfull, Kd, skl.mean()


def undefined(Kd):
    return int(np.isneginf(Kd).sum())


def percentile(v, q):
    """numpy >= 2's percentile(v, q, method="linear") for 1-d v, with the interpolation held at v's dtype."""
    v = np.sort(np.asarray(v))
    n = v.size
    qq = np.true_divide(q, 100)
    virt = n * qq + (1 + qq * (1 - 1 - 1)) - 1                  # numpy's _compute_virtual_index, alpha = beta = 1
    prev = np.floor(virt)
    gamma = np.asarray(virt - prev, dtype=v.dtype)
    i = int(np.clip(prev, 0, n - 1))
    j = int(np.clip(prev + 1, 0, n - 1))
    a, b = v[i], v[j]
    if gamma == 0:
        return float(a)
    if np.isinf(a) or np.isinf(b):                              # an endpoint that falls on an infinite draw
        return float(a if np.isinf(a) else b)
    d = b - a
    return float(b - d * (np.asarray(1, dtype=v.dtype) - gamma) if gamma >= 0.5 else a + d * gamma)


def ci(v, level=95.0):
    """common.ci's percentile interval, except that no draw is dropped: an infinite draw (an undefined K, k_skill)
    is kept, so that an endpoint falling on it is infinite; a NaN draw cannot arise and is refused."""
    v = np.asarray(v)
    assert not np.isnan(v).any(), "NaN draw"
    a = (100.0 - level) / 2.0
    return [percentile(v, a), percentile(v, 100.0 - a)]


def d_draws(k1, k0):
    """FT1's paired difference on every draw. A draw on which either K is undefined counts as -inf for the lower
    endpoint and +inf for the upper, so it can only widen the interval. Returns (lower draws, upper draws, count)."""
    bad = np.isneginf(k1) | np.isneginf(k0)
    with np.errstate(invalid="ignore"):
        d = k1 - k0
    lo, hi = d.copy(), d.copy()
    lo[bad], hi[bad] = -np.inf, np.inf
    return lo, hi, int(bad.sum())


def ci_d(lo, hi, level=95.0):
    return [ci(lo, level)[0], ci(hi, level)[1]]


def mean_diff(x, ref, tdraw):
    d = x - ref
    return d.mean(), d[tdraw].mean(1)


# ------------------------------------------------------------------ rules (item 6)
def rule_ft1(ci_):
    return C.label3(ci_, "FINE-TUNING RESTORES SKILL", "FINE-TUNING REMOVES SKILL")


def rule_ft2(ci_, skill_mean):
    """PerconAI's Q3 (PREREG_R2.md section 3, lines 68-80; section 5, lines 95-96): in this order; ties earn
    nothing."""
    if skill_mean <= 0:
        return "NO SKILL TO LOSE"
    lo, hi = ci_
    if lo > 0.90:
        return "NEARLY AS SKILLED"
    if lo > 0.50:
        return "KEEPS MOST"
    if hi < 0.50:
        return "LOSES MOST"
    return "NOT SETTLED"


def rule_onestep(ci_):
    return C.label3(ci_, "ONE-STEP READING OVERSTATES", "ONE-STEP READING UNDERSTATES")


def rule_change(ci_):
    return C.label3(ci_, "INCREASES", "DECREASES")


# ------------------------------------------------------------------ inputs and checks
def load():
    meta = json.load(open(os.path.join(C.RESULTS, "ft_models.json")))
    raw = dict(np.load(os.path.join(C.RESULTS, "raw_FT.npz")))
    probe = json.load(open(os.path.join(C.RESULTS, "ft_probe.json")))
    return meta, raw, probe


def validate(meta, raw, probe):
    """Every planned model scored and every check result present; else refuse. Returns excluded arms."""
    arms = list(probe["arms"])
    if meta.get("arms") != arms:
        sys.exit(f"REFUSED: scored arms {meta.get('arms')} differ from the time rule's {arms}")
    fields = ["err", "lam", "rJ"] + [f"E{h}" for h in HS if h != 368]
    miss, excluded = [], {}
    for arm in arms:                                # checks first: a failed check excludes the arm (section 7)
        if arm in FT_ARMS:
            for f in FOLDS:
                for s in SEEDS:
                    key = f"{arm}_{f}_s{s}"
                    miss += [f"{c} {key}" for c in CHECKS if key not in meta["checks"].get(c, {})]
            if not miss:
                failed = sorted({c for c in CHECKS for f in FOLDS for s in SEEDS
                                 if not meta["checks"][c][f"{arm}_{f}_s{s}"]["passed"]})
                if failed:
                    excluded[arm] = failed
    for arm in arms:
        if arm in excluded:                         # an excluded arm's scores are not needed (a run that failed
            continue                                # C-FT3 was never trained or scored)
        for f in FOLDS:
            for s in SEEDS:
                key = f"{arm}_{f}_s{s}"
                if key not in meta["models"]:
                    miss.append(f"score {key}")
                miss += [f"{x}_{key}" for x in fields if f"{x}_{key}" not in raw]
                if arm in ("P", "U30") and key not in meta["checks"].get("C-FT4", {}):
                    miss.append(f"C-FT4 {key}")
    for f in FOLDS:
        miss += [k for k in [f"floor_{f}"] + [f"floor_E{h}_{f}" for h in HS if h != 368] if k not in raw]
    if miss:
        sys.exit(f"REFUSED: {len(miss)} missing: {miss[:10]}")
    ft4_fail = sorted(k for k, v in meta["checks"]["C-FT4"].items() if not v["passed"])
    return arms, excluded, ft4_fail


# ------------------------------------------------------------------ analysis
def main():
    meta, raw, probe = load()
    arms, excluded, ft4_fail = validate(meta, raw, probe)
    tdraw = draws()
    keys = {a: {f: [f"{a}_{f}_s{s}" for s in SEEDS] for f in FOLDS} for a in arms}
    E = {a: {h: arm_traj(raw, keys[a], h) for h in HS} for a in arms if a not in excluded}
    FL = {h: floor_traj(raw, h) for h in HS}

    def blocked(*xs):
        """None if the arms may be used; else the reason (item 7 exclusion, or a non-finite score)."""
        for a in xs:
            if a not in arms:
                return "NOT RUN"
            if a in excluded:
                return f"EXCLUDED ({', '.join(excluded[a])} failed)"
        return None

    def finite(*arrs):
        return all(np.isfinite(a).all() for a in arrs)

    def kq(x, ref, h):
        Kf, Kd, sk = k_skill(E[x][h], E[ref][h], FL[h], tdraw)
        return {"K": float(Kf), "ci95": ci(Kd), "ci97.5": ci(Kd, 97.5), "skill_mean": float(sk),
                "undefined_draws": undefined(Kd)}, Kd

    def ft1(lvl, h, level):
        a0, a1 = f"U{lvl}", f"U{lvl}F"
        why = blocked("P", a0, a1)
        if why:
            return {"verdict": why}
        if not finite(E["P"][h], E[a0][h], E[a1][h], FL[h]):
            return {"verdict": "NOT COMPUTABLE (non-finite score)"}
        k1, d1 = kq(a1, "P", h)
        k0, d0 = kq(a0, "P", h)
        if k1["skill_mean"] <= 0:
            return {"verdict": "NOT COMPUTABLE (P has no skill over the floor)", "skill_mean_P": k1["skill_mean"]}
        lo, hi, nu = d_draws(d1, d0)
        out = {"d": k1["K"] - k0["K"], "ci95": ci_d(lo, hi), "ci97.5": ci_d(lo, hi, 97.5), "K_orig_ft": k1["K"],
               "K_orig_noft": k0["K"], "skill_mean_P": k1["skill_mean"], "undefined_draws": nu, "level": level}
        out["verdict"] = rule_ft1(out[f"ci{level}"])
        return out

    def ft2(lvl, h, level):
        a1 = f"U{lvl}F"
        why = blocked("CTRL", a1)
        if why:
            return {"verdict": why}
        if not finite(E["CTRL"][h], E[a1][h], FL[h]):
            return {"verdict": "NOT COMPUTABLE (non-finite score)"}
        k, _ = kq(a1, "CTRL", h)
        out = {"K_ctrl": k["K"], "ci95": k["ci95"], "ci97.5": k["ci97.5"], "skill_mean_ctrl": k["skill_mean"],
               "undefined_draws": k["undefined_draws"], "level": level}
        out["verdict"] = rule_ft2(out[f"ci{level}"], k["skill_mean"])
        if out["verdict"] == "NO SKILL TO LOSE":
            dm, dd = mean_diff(E[a1][h], E["CTRL"][h], tdraw)
            out["delta_ctrl"] = {"mean": float(dm), "ci95": ci(dd), "ci97.5": ci(dd, 97.5)}
        return out

    OUT = {"arms": arms, "excluded_arms": excluded, "C-FT4_failures": ft4_fail, "n_trajectories": NT,
           "n_draws": B, "seeds": list(SEEDS),
           "primary": {"FT1": ft1(30, 100, "97.5"), "FT2": ft2(30, 100, "97.5")},
           "secondary": {}}
    S = OUT["secondary"]
    for h in (24, 368):
        S[f"FT1_h{h}"], S[f"FT2_h{h}"] = ft1(30, h, "95"), ft2(30, h, "95")
    S["FT1_u50_h100"], S["FT2_u50_h100"] = ft1(50, 100, "95"), ft2(50, 100, "95")
    why = blocked("P", "U30F")
    if why or not finite(E["P"][1], E["U30F"][1], E["P"][100], E["U30F"][100], FL[1], FL[100]):
        S["onestep_U30F"] = {"verdict": why or "NOT COMPUTABLE (non-finite score)"}
    else:
        k1, d1 = kq("U30F", "P", 1)
        k100, d100 = kq("U30F", "P", 100)
        if min(k1["skill_mean"], k100["skill_mean"]) <= 0:
            S["onestep_U30F"] = {"verdict": "NOT COMPUTABLE (P has no skill over the floor)"}
        else:
            lo, hi, nu = d_draws(d1, d100)
            S["onestep_U30F"] = {"K_orig_h1": k1["K"], "K_orig_h1_ci95": k1["ci95"], "K_orig_h100": k100["K"],
                                 "K_orig_h100_ci95": k100["ci95"], "d": k1["K"] - k100["K"],
                                 "ci95": ci_d(lo, hi), "undefined_draws": nu}
            S["onestep_U30F"]["verdict"] = rule_onestep(S["onestep_U30F"]["ci95"])
    lam = lambda r, k, h: r[f"lam_{k}"][:, 0]                                   # noqa: E731
    rbj = lambda r, k, h: r[f"rJ_{k}"].mean(1)                                  # noqa: E731
    for name, get in (("lam1", lam), ("rJ", rbj)):
        for a in ("U30F", "U30"):
            why = blocked("P", a)
            if why:
                S[f"change_{name}_{a}_minus_P"] = {"verdict": why}
                continue
            x, p = arm_traj(raw, keys[a], None, get), arm_traj(raw, keys["P"], None, get)
            if not finite(x, p):
                S[f"change_{name}_{a}_minus_P"] = {"verdict": "NOT COMPUTABLE (non-finite score)"}
                continue
            m, dd = mean_diff(x, p, tdraw)
            S[f"change_{name}_{a}_minus_P"] = {"mean": float(m), "ci95": ci(dd), "P_mean": float(p.mean()),
                                               "arm_mean": float(x.mean())}
            S[f"change_{name}_{a}_minus_P"]["verdict"] = rule_change(S[f"change_{name}_{a}_minus_P"]["ci95"])

    # descriptive: K of every arm against P and against CTRL, every horizon (no verdicts)
    D = {}
    for a in arms:
        for ref in ("P", "CTRL"):
            if a == ref or blocked(a, ref):
                continue
            for h in HS:
                if finite(E[a][h], E[ref][h], FL[h]):
                    k, _ = kq(a, ref, h)
                    D[f"K_{a}_vs_{ref}_h{h}"] = {"K": k["K"], "ci95": k["ci95"]}
    OUT["descriptive"] = D
    OUT["mean_error"] = {a: {f"h{h}": float(E[a][h].mean()) for h in HS} for a in E}
    OUT["mean_error"]["floor"] = {f"h{h}": float(FL[h].mean()) for h in HS}
    with open(os.path.join(C.RESULTS, "ft_stats.json"), "w") as f:
        json.dump(OUT, f, indent=1, default=float)
    report(OUT)


def fmt(ci_):
    return f"[{ci_[0]:.3f}, {ci_[1]:.3f}]"


def report(O):
    L = ["Study FT (prereg/PREREG_FT.md): fine-tuning magnitude-pruned copies, 20 X5 trajectories, seeds 0-2.",
         f"arms {O['arms']}; excluded {O['excluded_arms'] or 'none'}; C-FT4 failures {O['C-FT4_failures'] or 'none'}",
         "", "PRIMARY (97.5% intervals)"]
    p = O["primary"]
    if "d" in p["FT1"]:
        L.append(f"FT1 d = K_orig(U30F,100) - K_orig(U30,100) = {p['FT1']['d']:.3f} {fmt(p['FT1']['ci97.5'])}"
                 f"  (U30F {p['FT1']['K_orig_ft']:.3f}, U30 {p['FT1']['K_orig_noft']:.3f}) -> {p['FT1']['verdict']}")
    else:
        L.append(f"FT1 -> {p['FT1']['verdict']}")
    if p["FT2"].get("verdict") == "NO SKILL TO LOSE":
        dc = p["FT2"]["delta_ctrl"]
        L.append(f"FT2 -> NO SKILL TO LOSE; mean(e_U30F - e_CTRL) at 100 = {dc['mean']:.4f} {fmt(dc['ci97.5'])}")
    elif "K_ctrl" in p["FT2"]:
        L.append(f"FT2 K_ctrl(U30F,100) = {p['FT2']['K_ctrl']:.3f} {fmt(p['FT2']['ci97.5'])} -> {p['FT2']['verdict']}")
    else:
        L.append(f"FT2 -> {p['FT2']['verdict']}")
    L += ["", "SECONDARY (95%, not corrected for multiplicity)"]
    for k, v in O["secondary"].items():
        if "d" in v and "K_orig_ft" in v:
            L.append(f"{k}: d = {v['d']:.3f} {fmt(v['ci95'])} -> {v['verdict']}")
        elif v.get("verdict") == "NO SKILL TO LOSE":
            L.append(f"{k}: NO SKILL TO LOSE; mean(e_x - e_CTRL) = {v['delta_ctrl']['mean']:.4f}"
                     f" {fmt(v['delta_ctrl']['ci95'])}")
        elif "K_ctrl" in v:
            L.append(f"{k}: K_ctrl = {v['K_ctrl']:.3f} {fmt(v['ci95'])} -> {v['verdict']}")
        elif "K_orig_h1" in v:
            L.append(f"{k}: K_orig h1 {v['K_orig_h1']:.3f}, h100 {v['K_orig_h100']:.3f}; d = {v['d']:.3f}"
                     f" {fmt(v['ci95'])} -> {v['verdict']}")
        elif "mean" in v:
            L.append(f"{k}: {v['mean']:.4f} {fmt(v['ci95'])} -> {v['verdict']}")
        else:
            L.append(f"{k}: {v['verdict']}")
    L += ["", "DESCRIPTIVE K (95%)"]
    for k, v in list(O["descriptive"].items())[:24]:
        L.append(f"{k}: {v['K']:.3f} {fmt(v['ci95'])}")
    with open(os.path.join(C.RESULTS, "ft_report.txt"), "w") as f:
        f.write("\n".join(L[:60]) + "\n")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()

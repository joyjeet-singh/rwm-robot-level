"""
Every registered question of studies CL and AR (prereg/PREREG_S1.md items 5-8), from results/s1_models.json (job S3,
scripts/s1_eval.py), the stored registered values (results/model_table.json) and, for the exploratory two-level
intervals only, the per-trajectory arrays (results/raw_S1.npz, results/raw_F18.npz).
Writes results/s1_stats.json and results/s1_report.txt (at most 60 lines). Refuses to run if a check is missing.

Joint commands are the 12 joint position targets given at every step; the velocity command is the walking speed and
turning rate the policy is asked to follow. "pol" values come from rollouts whose joint commands the authors' policy
supplied; "rec" values from rollouts under the recorded joint commands (the registered measures).

Statistics are the registered ones, imported from scripts/stats.py, never copied: spearman (average ranks;
non-finite values rank worst, tied), boot_diff (paired model bootstrap, 20,000 draws, default_rng(0); both
correlations from the same draw), boot_rho, and boot_diff_within (models resampled within their own group; the
statistic is the mean over groups of the within-group difference). boot_diff_within reports only a 95% interval; its
97.5% interval is taken from the same 20,000 draws, which `within` captures as they reach common.ci.
"""
import json
import os
import sys

import numpy as np

import common as C
import stats as ST

MARGIN = 0.10
CHECKS = ("S1-C1", "S1-C2", "S1-C3", "S1-C4", "S1-C5")
ARCHES = ("mlp", "rssm", "transformer")
CL_GROUPS = ("rollout", "one-step", "compressed")
Q1_LAB = ("LABEL-FREE BETTER", "ONE-STEP BETTER")
Q2_LAB = ("LABEL-FREE BETTER", "TRAINING-HORIZON BETTER")


# ------------------------------------------------------------------ labels
def cl_label(ci, source="POLICY"):
    """The registered rule of CL1 and CL2 (PREREG_S1 item 6), applied in order, on the interval [lo, hi]."""
    lo, hi = ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "NOT SETTLED"
    if lo >= -MARGIN and hi <= MARGIN:
        lab = "UNCHANGED BY THE COMMAND SOURCE"
        if hi < 0:
            lab += " (small decrease)"
        elif lo > 0:
            lab += " (small increase)"
        return lab
    if hi < 0:
        return f"WORSE WITH {source} COMMANDS"
    if lo > 0:
        return f"BETTER WITH {source} COMMANDS"
    return "NOT SETTLED"


def gate(label, family, gates):
    return label if gates[family]["verdict"] else f"NO VERDICT ({gates[family]['reason']})"


def labelled(r, family, gates, rule_label, key="label"):
    """Writes the label (gated) and the rule's label; a gated family keeps neither (PREREG_S1 item 7)."""
    r["rule_label"] = rule_label if gates[family]["verdict"] else None
    r[key] = gate(rule_label, family, gates)
    return r


# ------------------------------------------------------------------ statistics (imported)
def within(groups, new, base, y, levels=(95.0, 97.5)):
    """stats.boot_diff_within, with the intervals at `levels` from its own draws."""
    captured, orig = [], C.ci

    def cap(v, level=95.0):
        captured.append(np.asarray(v, dtype=float).copy())
        return orig(v, level)

    C.ci = cap
    try:
        out = ST.boot_diff_within(groups, new, base, y)
    finally:
        C.ci = orig
    assert len(captured) == 1
    assert out["ci95"] == orig(captured[0], 95.0)
    for lv in levels:
        out[f"ci{lv:g}"] = orig(captured[0], lv)
    out["rho_new_per_group"] = [ST.spearman([r[new] for r in g], [r[y] for r in g]) for g in groups]
    out["rho_base_per_group"] = [ST.spearman([r[base] for r in g], [r[y] for r in g]) for g in groups]
    return out


def col(rows, k):
    return [r[k] for r in rows]


def diff(rows, new, base, y="E100", levels=(95.0,)):
    return ST.boot_diff(col(rows, new), col(rows, base), col(rows, y), levels=levels)


# ------------------------------------------------------------------ inputs
def require_checks(db):
    miss = [c for c in CHECKS if c not in db.get("checks", {}) or "pass" not in db["checks"][c]]
    if miss:
        raise SystemExit(f"s1_stats: refusing to run, check result(s) missing: {miss}")


def gates_of(db):
    """Which families may carry a verdict (PREREG_S1 item 7)."""
    c = {k: bool(db["checks"][k]["pass"]) for k in CHECKS}
    cl_bad = [k for k in ("S1-C1", "S1-C2") if not c[k]]
    ar_bad = [k for k in ("S1-C1", "S1-C3") if not c[k]]
    why = "the new code does not reproduce the registered measures: {} failed"
    return {"CL": {"verdict": not cl_bad, "reason": why.format(", ".join(cl_bad)) if cl_bad else ""},
            "AR": {"verdict": not ar_bad, "reason": why.format(", ".join(ar_bad)) if ar_bad else ""},
            "checks": c}


def cl_rows(db, table, tag):
    """Stored registered values (recorded commands) joined with the job's policy values, in table order."""
    rows, missing = [], []
    for t in table[tag]:
        s = db["models"][tag].get(t["key"])
        if s is None:
            missing.append(t["key"])
            continue
        r = {"key": t["key"], "arena": t["arena"], "variant": t["variant"], "group": t["group"],
             "cl_group": s["cl_group"], "lam1_rec": t["lam1"], "rJ_rec": t["rJ"], "E1": t["E1"], "E8": t["E8"],
             "E100": t["E100"]}
        for k in ("lam1_pol", "rJ_pol", "lam1_cl", "lam1_noise", "rJ_noise", "ret", "ret_unc", "n_div_pol",
                  "n_div_cl", "n_div_noise", "n_traj"):
            r[k] = s[k]
        rows.append(r)
    return rows, missing


def ar_rows(db):
    rows = []
    for k, s in sorted(db["models"]["AR"].items()):
        rows.append(dict(s, key=k))
    return rows


# ------------------------------------------------------------------ study CL
def study_cl(db, table, gates, traj=None):
    out = {}
    f18, miss18 = cl_rows(db, table, "F18")
    x5, missx5 = cl_rows(db, table, "X5")
    out["n"] = {"F18": len(f18), "X5": len(x5), "F18_missing": miss18, "X5_missing": missx5}
    assert len(f18) + len(miss18) == len(table["F18"]) and len(x5) + len(missx5) == len(table["X5"])
    # primary: CL1, CL2 (97.5%)
    for q, new, base in (("CL1", "rJ_pol", "rJ_rec"), ("CL2", "lam1_pol", "lam1_rec")):
        r = diff(f18, new, base, levels=(95.0, 97.5))
        out[q] = labelled(r, "CL", gates, cl_label(r["ci97.5"]), key="verdict")
    # CL3: rank agreement of each check under policy and recorded commands (95%)
    out["CL3"] = {"lam1": ST.boot_rho(col(f18, "lam1_pol"), col(f18, "lam1_rec")),
                  "rJ": ST.boot_rho(col(f18, "rJ_pol"), col(f18, "rJ_rec"))}
    # CL4: X5, within the five splits (95%)
    groups = [[r for r in x5 if r["arena"] == C.fid(f)] for f in C.FOLDS]
    assert sum(map(len, groups)) == len(x5), "an X5 model is in no split"
    if not missx5:
        assert all(len(g) == len(table["X5"]) // len(C.FOLDS) for g in groups), [len(g) for g in groups]
    out["CL4"] = {}
    for q, new, base in (("CL1", "rJ_pol", "rJ_rec"), ("CL2", "lam1_pol", "lam1_rec")):
        r = within(groups, new, base, "E100", levels=(95.0,))
        out["CL4"][q] = labelled(r, "CL", gates, cl_label(r["ci95"]))
    # CL5: closed-loop exponent (95%)
    r = diff(f18, "lam1_cl", "lam1_rec")
    out["CL5"] = labelled(r, "CL", gates, cl_label(r["ci95"]))
    # CL6: imagined return vs E100, models whose policy rollouts never diverged (95%)
    ok = [r for r in f18 if r["n_div_pol"] == 0]
    rho6 = ST.boot_rho(col(ok, "ret"), col(ok, "E100")) if len(ok) >= 3 else \
        {"rho": None, "ci95": None, "tau": None, "not_computable": "fewer than 3 models without a diverged rollout"}
    out["CL6"] = dict(rho6, n_models=len(ok), n_excluded_diverged=len(f18) - len(ok),
                      reading="a positive value means that less accurate simulators flatter the policy")
    # CL7: divergence counts by group
    cnt = {}
    for tag, rows in (("F18", f18), ("X5", x5)):
        cnt[tag] = {}
        for g in CL_GROUPS:
            gr = [r for r in rows if r["cl_group"] == g]
            cnt[tag][g] = {"models": len(gr), "trajectories": int(sum(r["n_traj"] for r in gr))}
            for c in ("pol", "cl", "noise"):
                cnt[tag][g][f"models_diverged_{c}"] = int(sum(r[f"n_div_{c}"] > 0 for r in gr))
                cnt[tag][g][f"trajectories_diverged_{c}"] = int(sum(r[f"n_div_{c}"] for r in gr))
    out["CL7"] = cnt
    # CL8: noisy recorded joint commands in place of the policy's (95%)
    out["CL8"] = {}
    for q, new, base in (("CL1", "rJ_noise", "rJ_rec"), ("CL2", "lam1_noise", "lam1_rec")):
        r = diff(f18, new, base)
        out["CL8"][q] = labelled(r, "CL", gates, cl_label(r["ci95"], source="NOISY"))
    # CL9: within trained (57) and compressed (40) simulators separately (95%)
    out["CL9"] = {}
    subs = (("trained", [r for r in f18 if r["variant"] is None]),
            ("compressed", [r for r in f18 if r["variant"] is not None]))
    if not miss18:
        n_tr = sum(t["variant"] is None for t in table["F18"])
        assert (len(subs[0][1]), len(subs[1][1])) == (n_tr, len(table["F18"]) - n_tr)
    for sub, rows in subs:
        out["CL9"][sub] = {}
        for q, new, base in (("CL1", "rJ_pol", "rJ_rec"), ("CL2", "lam1_pol", "lam1_rec")):
            r = diff(rows, new, base)
            out["CL9"][sub][q] = labelled(r, "CL", gates, cl_label(r["ci95"]))
    # exploratory: two-level intervals (clusters of a parent and its compressed copies, and trajectories)
    if traj is not None:
        out["exploratory_two_level"] = two_level(f18, traj)
        if not miss18:
            assert out["exploratory_two_level"]["n_clusters"] == sum(t["variant"] is None for t in table["F18"])
    # reference only: the released model
    rel = db["models"].get("REL", {}).get("REL")
    if rel:
        out["REL_reference"] = {k: rel[k] for k in ("lam1_pol", "rJ_pol", "lam1_cl", "lam1_noise", "rJ_noise",
                                                    "lam1_rec", "rJ_rec", "ret", "ret_unc", "n_div_pol",
                                                    "n_div_cl", "n_div_noise", "n_traj") if k in rel}
    return out


def two_level(f18, traj, n_draws=None, seed=0):
    """EXPLORATORY. Per draw: the 4 trajectories resampled, then the clusters (a parent and its compressed copies;
    every other model its own cluster); model values are means over the drawn trajectories."""
    n_draws = ST.B if n_draws is None else n_draws
    par = [r["key"].split("__")[0] for r in f18]
    uniq = list(dict.fromkeys(par))
    members = [np.array([i for i, p in enumerate(par) if p == u]) for u in uniq]
    V = {k: np.asarray([traj[k][r["key"]] for r in f18], dtype=float)
         for k in ("rJ_pol", "rJ_rec", "lam1_pol", "lam1_rec", "E100")}
    ntr = V["E100"].shape[1]
    for k, v in V.items():
        assert v.shape == (len(f18), 4), (k, v.shape)
    rng = np.random.default_rng(seed)
    d = {"CL1": [], "CL2": []}
    for _ in range(n_draws):
        t = rng.integers(0, ntr, ntr)
        c = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[j] for j in c])
        M = {k: np.where(np.isfinite(v[:, t]).all(1), v[:, t].mean(1), np.inf)[idx] for k, v in V.items()}
        d["CL1"].append(ST.spearman(M["rJ_pol"], M["E100"]) - ST.spearman(M["rJ_rec"], M["E100"]))
        d["CL2"].append(ST.spearman(M["lam1_pol"], M["E100"]) - ST.spearman(M["lam1_rec"], M["E100"]))
    out = {"label": "exploratory", "n_draws": n_draws, "n_clusters": len(uniq), "n_trajectories": ntr}
    for q, v in d.items():
        v = np.asarray(v, dtype=float)
        out[q] = {"ci95": C.ci(v, 95.0), "ci97.5": C.ci(v, 97.5), "dropped_draws": int((~np.isfinite(v)).sum())}
    return out


# ------------------------------------------------------------------ study AR
def study_ar(db, table, gates):
    out = {}
    rows = ar_rows(db)
    groups = [[r for r in rows if r["arch"] == a] for a in ARCHES]
    assert sum(map(len, groups)) == len(rows), "a baseline run has an unknown architecture"
    out["n"] = {a: len(g) for a, g in zip(ARCHES, groups)}
    out["n_new_runs"] = sum(r["new_run"] for r in rows)
    out["divergence_recorded_commands"] = {a: {"runs_diverged": int(sum(r["n_div_rec"] > 0 for r in g)),
                                               "runs_lam_nonfinite": int(sum(r["n_lam_div"] > 0 for r in g))}
                                           for a, g in zip(ARCHES, groups)}
    if not db["checks"]["S1-C5"]["pass"]:
        out["note"] = "S1-C5 failed: no new run was trained; AR uses the existing runs only"
    for q, new, base, labs in (("AR1", "lam1_rec", "E1", Q1_LAB), ("AR2", "rJ_rec", "E8", Q2_LAB)):
        r = within(groups, new, base, "E100")
        out[q] = labelled(r, "AR", gates, C.label3(r["ci97.5"], *labs), key="verdict")
    # AR3: pooled across architectures, and the extended pool (F18's stored values plus the baselines), 95%
    ext = [{"lam1_rec": t["lam1"], "rJ_rec": t["rJ"], "E1": t["E1"], "E8": t["E8"], "E100": t["E100"]}
           for t in table["F18"]] + rows
    out["AR3"] = {}
    for pool, P in (("pooled", rows), ("extended", ext)):
        out["AR3"][pool] = {"n_models": len(P)}
        for q, new, base, labs in (("AR1", "lam1_rec", "E1", Q1_LAB), ("AR2", "rJ_rec", "E8", Q2_LAB)):
            r = diff(P, new, base)
            out["AR3"][pool][q] = labelled(r, "AR", gates, C.label3(r["ci95"], *labs))
    # AR4: descriptive, paired by seed: teacher-forced better one step ahead and worse at 100 steps?
    out["AR4"] = {"label": "descriptive"}
    for a in ARCHES:
        pairs = []
        for s in range(5):
            tf = [r for r in rows if r["arch"] == a and r["spec"] == "s7" and r["regime"] == "tf" and r["seed"] == s]
            ar = [r for r in rows if r["arch"] == a and r["spec"] == "s7" and r["regime"] == "ar" and r["seed"] == s]
            if tf and ar:
                tf, ar = tf[0], ar[0]
                pairs.append({"seed": s, "E1_tf": tf["E1"], "E1_ar": ar["E1"], "E100_tf": tf["E100"],
                              "E100_ar": ar["E100"], "tf_better_E1": bool(tf["E1"] < ar["E1"]),
                              "tf_worse_E100": bool(tf["E100"] > ar["E100"])})
        out["AR4"][a] = {"n_pairs": len(pairs), "tf_better_E1": sum(p["tf_better_E1"] for p in pairs),
                         "tf_worse_E100": sum(p["tf_worse_E100"] for p in pairs),
                         "both": sum(p["tf_better_E1"] and p["tf_worse_E100"] for p in pairs), "pairs": pairs}
    return out


# ------------------------------------------------------------------ report
def f(x):
    return "-" if x is None else (f"{x:+.3f}" if np.isfinite(x) else str(x))


def cs(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def report(o):
    L = ["Step 1 studies CL and AR (prereg/PREREG_S1.md). Registered unless labelled descriptive or exploratory.",
         "checks: " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in o["gates"]["checks"].items())]
    for fam in ("CL", "AR"):
        if not o["gates"][fam]["verdict"]:
            L.append(f"  family {fam}: NO VERDICT, {o['gates'][fam]['reason']}")
    cl, ar = o["CL"], o["AR"]
    L.append(f"CL (F18 n={cl['n']['F18']}, X5 n={cl['n']['X5']}; missing F18 {len(cl['n']['F18_missing'])}, "
             f"X5 {len(cl['n']['X5_missing'])})")
    for q in ("CL1", "CL2"):
        r = cl[q]
        L.append(f"  {q} primary: {r['verdict']}  d={f(r['delta'])} ci97.5={cs(r['ci97.5'])} ci95={cs(r['ci95'])} "
                 f"rho_pol={f(r['rho_new'])} rho_rec={f(r['rho_base'])} dropped={r['dropped_draws']}")
    for c, r in cl["CL3"].items():
        L.append(f"  CL3 {c}: rho(pol, rec)={f(r['rho'])} ci95={cs(r['ci95'])}")
    for q, r in cl["CL4"].items():
        L.append(f"  CL4 X5 {q}: {r['label']}  d={f(r['delta'])} ci95={cs(r['ci95'])}")
    r = cl["CL5"]
    L.append(f"  CL5 lam1_cl: {r['label']}  d={f(r['delta'])} ci95={cs(r['ci95'])} rho_cl={f(r['rho_new'])}")
    r = cl["CL6"]
    L.append(f"  CL6 rho(return, E100)={f(r['rho'])} ci95={cs(r['ci95']) if r['ci95'] else '-'} n={r['n_models']} "
             f"(excluded, diverged: {r['n_excluded_diverged']})")
    for tag, g in cl["CL7"].items():
        L.append(f"  CL7 {tag} models diverged pol/cl/noise: " + "; ".join(
            f"{k} {v['models_diverged_pol']}/{v['models_diverged_cl']}/{v['models_diverged_noise']} of {v['models']}"
            for k, v in g.items()))
    for q, r in cl["CL8"].items():
        L.append(f"  CL8 noise {q}: {r['label']}  d={f(r['delta'])} ci95={cs(r['ci95'])}")
    for sub, d in cl["CL9"].items():
        for q, r in d.items():
            L.append(f"  CL9 {sub} {q}: {r['label']}  d={f(r['delta'])} ci95={cs(r['ci95'])} n={r['n_models']}")
    if "exploratory_two_level" in cl:
        e = cl["exploratory_two_level"]
        L.append(f"  exploratory two-level ({e['n_clusters']} clusters x {e['n_trajectories']} trajectories): "
                 f"CL1 ci97.5={cs(e['CL1']['ci97.5'])}  CL2 ci97.5={cs(e['CL2']['ci97.5'])}")
    if "REL_reference" in cl:
        r = cl["REL_reference"]
        L.append(f"  released model (reference only): lam1 pol {f(r['lam1_pol'])} rec {f(r.get('lam1_rec'))}; "
                 f"rJ pol {f(r['rJ_pol'])} rec {f(r.get('rJ_rec'))}; diverged pol {r['n_div_pol']} of {r['n_traj']}")
    L.append(f"AR (runs per architecture {ar['n']}; new runs {ar['n_new_runs']}) {ar.get('note', '')}")
    L.append(f"  diverged under recorded commands: {ar['divergence_recorded_commands']}")
    for q in ("AR1", "AR2"):
        r = ar[q]
        L.append(f"  {q} primary: {r['verdict']}  d={f(r['delta'])} ci97.5={cs(r['ci97.5'])} ci95={cs(r['ci95'])} "
                 f"dropped={r['dropped_draws']}")
        L.append(f"     per architecture rho_check={[round(x, 3) for x in r['rho_new_per_group']]} "
                 f"rho_base={[round(x, 3) for x in r['rho_base_per_group']]}")
    for pool, d in ar["AR3"].items():
        for q in ("AR1", "AR2"):
            r = d[q]
            L.append(f"  AR3 {pool} (n={d['n_models']}) {q}: {r['label']}  d={f(r['delta'])} ci95={cs(r['ci95'])}")
    for a in ARCHES:
        r = ar["AR4"][a]
        L.append(f"  AR4 (descriptive) {a}: tf better at E1 in {r['tf_better_E1']}/{r['n_pairs']}, worse at E100 "
                 f"in {r['tf_worse_E100']}/{r['n_pairs']}, both {r['both']}/{r['n_pairs']}")
    assert len(L) <= 60, len(L)
    return "\n".join(L)


# ------------------------------------------------------------------ main
def load_traj(db, table):
    """Per-trajectory values for the exploratory two-level intervals (F18)."""
    raw = np.load(os.path.join(C.RESULTS, "raw_F18.npz"))
    s1 = np.load(os.path.join(C.RESULTS, "raw_S1.npz"))
    tr = {k: {} for k in ("rJ_pol", "rJ_rec", "lam1_pol", "lam1_rec", "E100")}
    for t in table["F18"]:
        k = t["key"]
        if k not in db["models"]["F18"]:
            continue
        tr["lam1_rec"][k] = raw[f"lam_{k}"][:, 0]
        tr["rJ_rec"][k] = raw[f"rJ_{k}"].mean(-1)
        tr["E100"][k] = raw[f"E100_{k}"]
        tr["lam1_pol"][k] = s1[f"F18__{k}__lam1_pol"]
        tr["rJ_pol"][k] = s1[f"F18__{k}__rJ_pol"]
    return tr


def analyse(db, table, traj=None):
    require_checks(db)
    gates = gates_of(db)
    return {"gates": gates, "margin": MARGIN, "dry_run": bool(db.get("dry_run", False)),
            "CL": study_cl(db, table, gates, traj), "AR": study_ar(db, table, gates)}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def main():
    db = json.load(open(os.path.join(C.RESULTS, "s1_models.json")))
    require_checks(db)
    if db.get("dry_run"):
        raise SystemExit("s1_stats: refusing to run on a dry run's output")
    table = json.load(open(os.path.join(C.RESULTS, "model_table.json")))
    out = analyse(db, table, load_traj(db, table))
    json.dump(clean(out), open(os.path.join(C.RESULTS, "s1_stats.json"), "w"), indent=1)
    txt = report(out)
    open(os.path.join(C.RESULTS, "s1_report.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    sys.exit(main())

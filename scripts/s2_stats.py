"""
Step 2's registered analysis (prereg/PREREG_S2.md): do the checks that rank learned simulators by long-horizon error
also rank them by how well policies trained inside them walk in the hand-built MuJoCo ANYmal D?

    ./py scripts/s2_stats.py          # -> results/s2_stats.json, results/s2_stats.txt (never overwritten)

Inputs (all written by scripts; nothing typed):
  results/s2/checks.json        the nine simulators' checks (scripts/s2/fallback_checks.py; hash fixed below)
  results/s2/train/<p>.json     each policy's imagined return (scripts/s2/train_policy.py)
  results/s2/eval/<p>.json      each policy's MuJoCo measures (scripts/s2/evaluate.py)

The simulator is the unit: its score is the mean over its measured policies (two seeds). Every check is turned
"higher is better" through its rank, so a non-finite error ranks worst: errors, lambda_1 and r_J are negated
(-rank), the imagined return is not. Spearman correlation, paired bootstrap over simulators: stats.boot_rho and
stats.boot_diff, imported, never copied (20,000 draws, numpy default_rng(0), the same draws for every analysis).
Labels: common.label3 (an interval wholly above 0 or wholly below 0; otherwise NOT SETTLED).
"""
import json
import os
import sys

import numpy as np

import common as C
import stats as ST
from ranks import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
AIR = os.path.dirname(HERE)
RES = os.path.join(AIR, "results")
POOL = ["REL", "REL__fp16", "REL__int8w", "REL__u10", "REL__u20", "REL__u30", "REL__u50", "REL__s192",
        "REL__s128"]
SEEDS = (1, 2)
CHECKS_SHA256 = "23e9daaf219febe57b25b01924f3fa6eba29a312f48d31d7832591e550a351c2"
B = C.N_BOOT
PRIMARY, SECONDARY = 97.5, 95.0
R0_LAB = ("SUPPORTED", "CONTRADICTED")
R1_LAB = ("LABEL-FREE BETTER", "ONE-STEP BETTER")
R2_LAB = ("LABEL-FREE BETTER", "TRAINING-HORIZON BETTER")
RHO_LAB = ("POSITIVE", "NEGATIVE")
LOWER_IS_BETTER = ("E1", "E8", "E100", "lam1", "rJ", "lam1_pol", "rJ_pol")
SEL_CHECKS = ("E1", "E8", "E100", "lam1", "rJ", "lam1_pol", "rJ_pol", "imag")
TOP = 3
NA = "NOT COMPUTABLE"
MAX_DROPPED = 0.10                      # more undefined draws than this share: NOT COMPUTABLE


# ------------------------------------------------------------------ inputs
def policy(sim, seed):
    return f"{sim}__p{seed}"


def load(res=RES, check_hash=True):
    """
    Per-simulator rows; a simulator's measures are means over its policies.
      * measured policy: results/s2/eval/<p>.json;
      * policy failed by its simulator (results/s2/failed/<p>.json, cause "simulator", written by s2_main when a
        step fails twice for a reason other than the machine): fall rate 1 and the lowest physics return of all
        measured policies; tracking errors not available;
      * anything else is missing; a simulator with no policy left is dropped and listed.
    Imagined return: from results/s2/train/<p>.json. A policy without a training record or value is missing (the
    simulator leaves the imagined-return analyses if none of its policies has one); a non-finite value is kept and
    ranks worst.
    """
    cpath = os.path.join(res, "s2", "checks.json")
    if check_hash and C.sha256(cpath) != CHECKS_SHA256:
        raise SystemExit(f"{cpath} differs from the registered file (sha256)")
    checks = json.load(open(cpath))["models"]
    ev_all, fail_all = {}, {}
    for sim in POOL:
        for s in SEEDS:
            p = policy(sim, s)
            ep = os.path.join(res, "s2", "eval", p + ".json")
            fp = os.path.join(res, "s2", "failed", p + ".json")
            if os.path.exists(ep):
                ev_all[p] = json.load(open(ep))["summary"]
            elif os.path.exists(fp) and json.load(open(fp)).get("cause") == "simulator":
                fail_all[p] = json.load(open(fp))
    worst = min((e["ret"] for e in ev_all.values()), default=np.nan)
    rows, missing = [], []
    for sim in POOL:
        per, imag, track = [], [], []
        for s in SEEDS:
            p = policy(sim, s)
            if p in ev_all:
                e = ev_all[p]
                per.append({"policy": p, "ret": float(e["ret"]), "fall": float(e["fall_rate"]), "failed": False})
                track.append((e["speed_err"], e["turn_err"]))
            elif p in fail_all:
                per.append({"policy": p, "ret": float(worst), "fall": 1.0, "failed": True})
            else:
                missing.append(p)
                continue
            tp = os.path.join(res, "s2", "train", p + ".json")
            ir = json.load(open(tp)).get("imagined_return") if os.path.exists(tp) else None
            if ir is not None:
                imag.append(float(ir))
        if not per:
            continue
        r = {"sim": sim, "n_policies": len(per), "n_failed": sum(q["failed"] for q in per), "per_policy": per,
             "ret": float(np.mean([q["ret"] for q in per])), "fall": float(np.mean([q["fall"] for q in per])),
             "speed_err": float(np.mean([t[0] for t in track])) if track else np.nan,
             "turn_err": float(np.mean([t[1] for t in track])) if track else np.nan,
             "imag_missing": not imag,
             "imag": (float(np.mean(imag)) if np.all(np.isfinite(imag)) else np.nan) if imag else np.nan}
        for k in LOWER_IS_BETTER:
            r[k] = float(checks[sim][k])
        rows.append(r)
    return rows, missing


def better(x, lower_is_better=True):
    """Higher is better, through ranks; a non-finite value ranks worst."""
    x = np.asarray(x, dtype=float)
    return -rankdata(ST._clean(x if lower_is_better else -x))


def col(rows, k):
    if k == "imag":
        return better([r[k] for r in rows], lower_is_better=False)
    if k in LOWER_IS_BETTER:
        return better([r[k] for r in rows])
    return np.asarray([r[k] for r in rows], dtype=float)


def computable(y):
    """An outcome is usable with at least 3 simulators, all finite, taking at least 3 distinct values."""
    y = np.asarray(y, dtype=float)
    return len(y) >= 3 and np.all(np.isfinite(y)) and len(np.unique(y)) >= 3


def spread(x):
    return len(np.unique(np.asarray(x, dtype=float))) >= 2


def n_dropped(xs, y):
    """Draws (the same default_rng(0) draws as stats.boot_rho/boot_diff) in which any rho(x, y) is undefined."""
    n = len(y)
    rng = np.random.default_rng(0)
    bad = 0
    for idx in rng.integers(0, n, size=(B, n)):
        if not all(np.isfinite(ST.spearman(x[idx], y[idx])) for x in xs):
            bad += 1
    return bad


def na(why, n):
    return {"label": NA, "why": why, "n_models": int(n)}


# ------------------------------------------------------------------ registered quantities
def rho_q(x, y, lab):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if not computable(y):
        return na("outcome: fewer than 3 simulators or 3 distinct values", len(y))
    if not spread(x) or not np.isfinite(ST.spearman(x, y)):
        return na("check: all values tied", len(y))
    dropped = n_dropped([x], y)
    if dropped > MAX_DROPPED * B:
        return dict(na(f"{dropped} of {B} draws undefined (more than 10%)", len(y)), dropped_draws=dropped)
    out = ST.boot_rho(x, y, SECONDARY)
    out["label"] = C.label3(out["ci95"], *lab)
    out["n_models"] = int(len(y))
    out["dropped_draws"] = int(dropped)
    return out


def diff_q(x_new, x_base, y, lab, level):
    x_new, x_base, y = (np.asarray(v, dtype=float) for v in (x_new, x_base, y))
    if not computable(y):
        return na("outcome: fewer than 3 simulators or 3 distinct values", len(y))
    if not (spread(x_new) and spread(x_base)):
        return na("check: all values tied", len(y))
    out = ST.boot_diff(x_new, x_base, y, levels=(SECONDARY, PRIMARY))
    out["level"] = level
    if not (np.isfinite(out["rho_new"]) and np.isfinite(out["rho_base"])):
        return dict(na("rho undefined on the full sample", len(y)), **out, label=NA)
    if out["dropped_draws"] > MAX_DROPPED * B:
        return dict(out, label=NA, why=f"{out['dropped_draws']} of {B} draws undefined (more than 10%)")
    out["label"] = C.label3(out[f"ci{level:g}"], *lab)
    return out


def family(rows, y, lam1="lam1", rJ="rJ", level=PRIMARY, with_r0=True):
    out = {}
    if with_r0:
        out["R0"] = rho_q(col(rows, "E100"), y, R0_LAB)
    out["R1"] = diff_q(col(rows, lam1), col(rows, "E1"), y, R1_LAB, level)
    out["R2"] = diff_q(col(rows, rJ), col(rows, "E8"), y, R2_LAB, level)
    return out


def top_mean(score, y, pos, k=TOP):
    """
    Mean of y over the k entries with the highest score. Entries are the drawn simulators; a simulator drawn twice
    is two entries. Ties in score: the lower pool position `pos` first.
    """
    order = np.lexsort((np.asarray(pos), -np.asarray(score, dtype=float)))
    return float(np.mean(np.asarray(y)[order[:k]]))


def selection(rows, y):
    """Shortfall of each check's top three against the best three by physics return; paired bootstrap."""
    n = len(y)
    if not computable(y) or n < 2 * TOP:
        return na("outcome not computable, or fewer than 6 simulators", n)
    y = np.asarray(y, dtype=float)
    checks = [k for k in SEL_CHECKS if not (k == "imag" and any(r["imag_missing"] for r in rows))]
    sc = {k: col(rows, k) for k in checks}

    def shortfalls(idx):
        best = top_mean(y[idx], y[idx], idx)
        return {k: best - top_mean(sc[k][idx], y[idx], idx) for k in checks}

    full = shortfalls(np.arange(n))
    rng = np.random.default_rng(0)
    draws = [shortfalls(idx) for idx in rng.integers(0, n, size=(B, n))]
    out = {"best_three_mean": top_mean(y, y, np.arange(n)), "n_models": int(n), "checks": {},
           "left_out": [k for k in SEL_CHECKS if k not in checks]}
    for k in checks:
        order = np.lexsort((np.arange(n), -sc[k]))[:TOP]
        out["checks"][k] = {"chosen": [rows[i]["sim"] for i in order], "shortfall": full[k],
                            "ci95": C.ci([d[k] for d in draws], SECONDARY)}
    for q, new, base, lab in (("SEL1", "lam1", "E1", R1_LAB), ("SEL2", "rJ", "E8", R2_LAB)):
        d = [x[base] - x[new] for x in draws]                    # > 0: the label-free check chose better
        ci_ = C.ci(d, SECONDARY)
        out[q] = {"new": new, "base": base, "delta": full[base] - full[new], "ci95": ci_,
                  "label": C.label3(ci_, *lab)}
    return out


def two_level(rows, lam1="lam1", rJ="rJ"):
    """Exploratory: simulators resampled, then each drawn simulator's policies resampled within it."""
    if not computable([r["ret"] for r in rows]):
        return {"label": NA}
    n = len(rows)
    per = [np.asarray([p["ret"] for p in r["per_policy"]]) for r in rows]
    x = {k: col(rows, k) for k in ("E100", "E1", "E8", lam1, rJ)}
    rng = np.random.default_rng(0)
    d = {"R0": [], "R1": [], "R2": []}
    dropped = 0
    for _ in range(B):
        idx = rng.integers(0, n, n)
        y = np.asarray([per[i][rng.integers(0, len(per[i]), len(per[i]))].mean() for i in idx])
        r0 = ST.spearman(x["E100"][idx], y)
        r1 = ST.spearman(x[lam1][idx], y) - ST.spearman(x["E1"][idx], y)
        r2 = ST.spearman(x[rJ][idx], y) - ST.spearman(x["E8"][idx], y)
        if not all(np.isfinite(v) for v in (r0, r1, r2)):
            dropped += 1
            continue
        d["R0"].append(r0)
        d["R1"].append(r1)
        d["R2"].append(r2)
    if len(d["R0"]) == 0:
        return {"label": NA, "dropped_draws": int(dropped)}
    return {"R0": {"ci95": C.ci(d["R0"], SECONDARY)},
            "R1": {"ci97.5": C.ci(d["R1"], PRIMARY), "ci95": C.ci(d["R1"], SECONDARY)},
            "R2": {"ci97.5": C.ci(d["R2"], PRIMARY), "ci95": C.ci(d["R2"], SECONDARY)},
            "dropped_draws": int(dropped), "label": "exploratory"}


def rollout_subset(rows):
    """The rollout-trained simulators alone: uncompressed members (here only the released model)."""
    sub = [r for r in rows if "__" not in r["sim"]]
    y = [r["ret"] for r in sub]
    if not computable(y):
        return {"label": "NOT APPLICABLE", "n_models": len(sub),
                "why": "the fallback pool has one rollout-trained simulator; the other eight are compressed copies"}
    return family(sub, np.asarray(y), level=SECONDARY)


def analyse(rows, missing):
    y = col(rows, "ret")
    yf = -col(rows, "fall")
    im = [r for r in rows if not r["imag_missing"]]
    out = {
        "pool": [r["sim"] for r in rows], "n_simulators": len(rows), "missing_policies": missing,
        "failed_policies": [q["policy"] for r in rows for q in r["per_policy"] if q["failed"]],
        "imagined_return_left_out": [r["sim"] for r in rows if r["imag_missing"]],
        "premise_and_primary": family(rows, y),
        "secondary": {
            "fall_rate": family(rows, yf, level=SECONDARY),
            "policy_commands": family(rows, y, "lam1_pol", "rJ_pol", level=SECONDARY, with_r0=False),
            "imagined_return": rho_q(col(im, "imag"), col(im, "ret"), RHO_LAB),
            "selection": selection(rows, y),
            "rollout_subset": rollout_subset(rows),
        },
        "exploratory": {"two_level": two_level(rows)},
        "descriptive": {"rows": rows},
    }
    return out


# ------------------------------------------------------------------ report
def f(x, d=3):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def ci_s(c):
    return f"[{f(c[0])}, {f(c[1])}]"


def report(o):
    L = ["Step 2 registered analysis (prereg/PREREG_S2.md); hand-built MuJoCo ANYmal D, not the data's simulator,",
         "not hardware. Unit: simulator (policy seeds averaged). Spearman rho, checks turned higher-is-better.",
         f"simulators: {o['n_simulators']}; missing policies: {', '.join(o['missing_policies']) or 'none'}; "
         f"failed by their simulator (falls 1, lowest return): {', '.join(o['failed_policies']) or 'none'}",
         f"left out of the imagined-return analyses: {', '.join(o['imagined_return_left_out']) or 'none'}", ""]
    P = o["premise_and_primary"]
    r0 = P["R0"]
    if r0["label"] == NA:
        L.append(f"R0 premise  {NA} ({r0.get('why')})")
    else:
        L.append(f"R0 premise  rho(-E100, return) = {f(r0['rho'])}, 95% {ci_s(r0['ci95'])}  -> {r0['label']}"
                 f"  (dropped draws {r0['dropped_draws']})")
    for q, nm in (("R1", "rho(-lam1) - rho(-E1)"), ("R2", "rho(-rJ) - rho(-E8)")):
        r = P[q]
        if r["label"] == NA:
            L.append(f"{q} primary {NA} ({r.get('why')})")
            continue
        L.append(f"{q} primary {nm} = {f(r['delta'])} (new {f(r['rho_new'])}, base {f(r['rho_base'])}), "
                 f"97.5% {ci_s(r['ci97.5'])}  -> {r['label']}  (dropped draws {r['dropped_draws']})")
    L += ["", "secondary (95%):"]
    S = o["secondary"]
    for name, fam in (("fall rate (outcome -falls)", S["fall_rate"]), ("policy commands", S["policy_commands"])):
        for q, r in fam.items():
            if r["label"] == NA:
                L.append(f"  {name} {q}: {NA} ({r.get('why')})")
            elif q == "R0":
                L.append(f"  {name} {q}: rho {f(r['rho'])} {ci_s(r['ci95'])} -> {r['label']} "
                         f"(dropped {r['dropped_draws']})")
            else:
                L.append(f"  {name} {q}: delta {f(r['delta'])} {ci_s(r['ci95'])} -> {r['label']} "
                         f"(dropped {r['dropped_draws']})")
    im = S["imagined_return"]
    L.append(f"  rho(imagined return, physics return): "
             + (f"{NA} ({im.get('why')})" if im["label"] == NA
                else f"{f(im['rho'])} {ci_s(im['ci95'])} -> {im['label']} (dropped {im['dropped_draws']})"))
    sel = S["selection"]
    if sel.get("label") == NA:
        L.append(f"  selection: {NA}")
    else:
        L.append(f"  selection: best three by physics return, mean {f(sel['best_three_mean'], 2)}; shortfall of each "
                 "check's top three:")
        for k, v in sel["checks"].items():
            L.append(f"    {k:9s} {f(v['shortfall'], 2):>7s} {ci_s(v['ci95'])}  chosen {', '.join(v['chosen'])}")
        for q in ("SEL1", "SEL2"):
            v = sel[q]
            L.append(f"    {q} shortfall({v['base']}) - shortfall({v['new']}) = {f(v['delta'], 2)} "
                     f"{ci_s(v['ci95'])} -> {v['label']}")
    rs = S["rollout_subset"]
    L.append(f"  rollout-trained subset: {rs.get('label', 'see json')} (n = {rs.get('n_models', '?')})")
    tl = o["exploratory"]["two_level"]
    if tl.get("label") != NA:
        L += ["", "exploratory, two-level intervals (policy seeds resampled within simulators):",
              f"  R0 95% {ci_s(tl['R0']['ci95'])}; R1 97.5% {ci_s(tl['R1']['ci97.5'])}; "
              f"R2 97.5% {ci_s(tl['R2']['ci97.5'])}"]
    L += ["", "descriptive, per simulator:",
          f"  {'simulator':12s} {'n':>2s} {'return':>8s} {'falls':>6s} {'speed':>6s} {'turn':>6s} {'imag':>6s} "
          f"{'E1':>6s} {'E8':>6s} {'E100':>6s} {'lam1':>6s} {'rJ':>8s} {'lam1p':>6s} {'rJp':>8s}"]
    for r in o["descriptive"]["rows"]:
        L.append(f"  {r['sim']:12s} {r['n_policies']:2d} {f(r['ret'], 2):>8s} {f(r['fall']):>6s} "
                 f"{f(r['speed_err']):>6s} {f(r['turn_err']):>6s} {f(r['imag'], 2):>6s} {f(r['E1']):>6s} "
                 f"{f(r['E8']):>6s} {f(r['E100']):>6s} {f(r['lam1']):>6s} {f(r['rJ'], 2):>8s} "
                 f"{f(r['lam1_pol']):>6s} {f(r['rJ_pol'], 2):>8s}")
    return "\n".join(L) + "\n"


def main(res=RES, out_json=None, out_txt=None, check_hash=True):
    out_json = out_json or os.path.join(res, "s2_stats.json")
    out_txt = out_txt or os.path.join(res, "s2_stats.txt")
    for p in (out_json, out_txt):
        if os.path.exists(p):
            raise SystemExit(f"{p} exists; result files are never overwritten")
    rows, missing = load(res, check_hash)
    o = analyse(rows, missing)
    o["what"] = ("Step 2 registered analysis (prereg/PREREG_S2.md); MuJoCo is a hand-built ANYmal D model, not the "
                 "data's simulator and not hardware")
    o["bootstrap"] = {"draws": B, "rng": "numpy default_rng(0)", "unit": "simulator"}
    json.dump(o, open(out_json, "w"), indent=1, default=float)
    txt = report(o)
    open(out_txt, "w").write(txt)
    print(txt)
    return o


if __name__ == "__main__":
    main()

"""
Option D's registered analysis (prereg/PREREG_D.md; OPTIOND_PLAN.md, "The registered analysis"): do the paper's checks
rank simulators adapted from the released model by how well policies trained inside them walk in the calibrated,
hand-built MuJoCo ANYmal D (with a fitted PD motor; not Isaac Lab, not hardware)?

    ./py scripts/d_stats.py           # -> results/d_stats.json, results/d_stats.txt (never overwritten)

Inputs (all written by scripts; nothing typed):
  results/d_ladder.json            the ladder's tested simulators and their policies (scripts/d/ladder.py final)
  results/d/eval/<policy>.json     walking score, fall rate, physics return of REL's and Step 2's copies' policies
  results/d/checks/<sim>.json      checks in common units, expert (primary) and mixed (secondary) test sets
  results/d/walk/<sim>.json        the walk test (secondary)
  results/d_record.json            the descriptive simulators (Option C's four, A5_s0, Step 2's three rebuilds)

The unit is the simulator; its outcome is the mean walking score over its two or three policies. Every check is made
"higher is better" through its rank (Step 2's `better`: errors, lambda1, r_J, walk-test falls and disagreement are
negated, -rank, so a non-finite value ranks worst; walk-test tracking is used as it is). Spearman correlation and the
paired bootstrap over simulators (20,000 draws, numpy default_rng(0), the same draws for every analysis):
stats.spearman, stats.boot_rho and stats.boot_diff, imported through Step 2's registered helpers (scripts/s2_stats.py:
rho_q, diff_q, better, computable, top_mean), never copied. Labels and the NOT COMPUTABLE rules are Step 2's
(common.label3; an interval wholly above 0 gives the first label, wholly below the second, else NOT SETTLED).
A pool of fewer than 4 simulators makes every analysis NOT COMPUTABLE; so does any reason Step 2's rules give, and the
reason is printed.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import stats as ST  # noqa: E402
import s2_stats as S2  # noqa: E402

RES = os.path.join(AIR, "results")
B = C.N_BOOT
PRIMARY, SECONDARY = 97.5, 95.0
FAIL_SCORE = 0.15
MIN_POOL = 4
NA = S2.NA
R0_LAB, R1_LAB, R2_LAB, RHO_LAB = S2.R0_LAB, S2.R1_LAB, S2.R2_LAB, S2.RHO_LAB
LONG_LAB = ("LONG-HORIZON BETTER", "TRAINING-HORIZON BETTER")
WALK_LAB = ("WALK TEST BETTER", "TRAINING-HORIZON BETTER")
RA_LAB = ("ATTITUDE RESIDUAL BETTER", "TRAINING-HORIZON BETTER")
AUC_LAB = ("ABOVE CHANCE", "BELOW CHANCE")
STEP2_COPIES = tuple(S2.POOL[1:])                     # REL's eight compressed copies (Step 2)
DESCRIPTIVE = ("A_s0", "B_s0", "A_s0__u50", "A5_s0", "sim_6M_N8_s0", "sim_6M_N8_s0__2S", "sim_10k_N8_s0",
               "sim_10k_N8_s0__2S")
SEED0_RUNS = ("W_s0", "M_s0", "Wtf_s0", "Wg_s0", "Mg_s0")
CHECKS = ("E1", "E8", "E24", "E100", "E368", "lambda1", "rJ")
SEL = ("E1", "E8", "E100", "E368", "lambda1", "rJ")
WALK_KEYS = {"tracking": ("tracking_per_step", False), "falls": ("falls", True), "u": ("u_bar", True)}
TOP = 3


# ------------------------------------------------------------------ inputs
def _json(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def make_row(sim, pool, policies, checks=None, walk=None, run=None, steps=None):
    """One simulator. policies: [{seed, score, fall_rate, physics_return, failed}] (failed-by-simulator policies:
    score 0, fall rate 1, physics return None until filled). checks: {"expert": {...}, "mixed": {...}}."""
    ps = sorted(policies, key=lambda p: p["seed"])
    f = [bool(p["failed"]) for p in ps]
    row = {"sim": sim, "pool": pool, "run": run, "steps": steps, "policies": ps, "n_policies": len(ps),
           "score": float(np.mean([p["score"] for p in ps])),
           "falls": float(np.mean([p["fall_rate"] for p in ps])),
           "score_p1": next((float(p["score"]) for p in ps if p["seed"] == 1), np.nan),
           "failed": sum(f) >= 2, "checks": checks or {}, "walk": walk or {}}
    return row


def fill_returns(rows):
    """Step 2's rule: a policy that failed by its simulator gets the lowest physics return among the measured
    policies of the rows given (the pool analysed; finite values only). Then each row's mean physics return. A
    measured policy without a physics return is missing data: its row's return is NaN, and the row leaves the
    physics-return analyses (listed)."""
    measured = [p["physics_return"] for r in rows for p in r["policies"]
                if p.get("status") != "failed_by_simulator" and p.get("physics_return") is not None]
    measured = [v for v in measured if np.isfinite(v)]
    low = float(np.min(measured)) if measured else np.nan
    for r in rows:
        v = []
        for p in r["policies"]:
            if p.get("status") == "failed_by_simulator":
                v.append(low)
            else:
                v.append(np.nan if p.get("physics_return") is None else float(p["physics_return"]))
        r["ret"] = float(np.mean(v)) if v else np.nan
        r["ret_filled"] = sum(p.get("status") == "failed_by_simulator" for p in r["policies"])
    return rows


def _checks(sim, res):
    c = _json(os.path.join(res, "d", "checks", f"{sim}.json"))
    if c is None:
        return None
    # a value stored as null (or absent) is missing: the key is left out, so the simulator leaves the analyses that
    # use it (listed); a computed value that is not finite (NaN, inf) stays and ranks worst
    return {s: {k: float(v[k]) for k in CHECKS + ("rA",) if v.get(k) is not None} for s, v in c["sets"].items()}


def _walk(sim, res):
    w = _json(os.path.join(res, "d", "walk", f"{sim}.json"))
    return None if w is None else {k: float(w[src]) for k, (src, _) in WALK_KEYS.items() if w.get(src) is not None}


def _eval_policy(p, seed, res):
    e = _json(os.path.join(res, "d", "eval", f"{p}.json"))
    if e is None:
        return None
    return {"policy": p, "seed": seed, "status": "finished", "score": e["score"], "fall_rate": e["fall_rate"],
            "physics_return": e.get("physics_return"), "failed": e["score"] < FAIL_SCORE}


def load(res=RES):
    """(rows, missing): REL, the ladder's tested simulators and Step 2's eight copies."""
    rows, missing = [], []
    for sim, pool in [("REL", "REL")] + [(s, "step2") for s in STEP2_COPIES]:
        ps = [_eval_policy(f"{sim}__p{s}", s, res) for s in (1, 2)]
        if any(p is None for p in ps):
            missing.append(f"{sim}: a registered policy's walking score is missing")
            ps = [p for p in ps if p is not None]
        if ps:
            rows.append(make_row(sim, pool, ps, _checks(sim, res), _walk(sim, res)))
    lad = _json(os.path.join(res, "d_ladder.json"))
    if lad is None:
        missing.append("results/d_ladder.json is missing: no ladder simulator")
    else:
        for t in lad["tested"]:
            if t["verdict"] is None:
                missing.append(f"{t['sim']}: its test did not finish; left out")
                continue
            ps = [{"policy": p["policy"], "seed": p["seed"], "status": p["status"], "score": p["score"],
                   "fall_rate": p["fall_rate"], "physics_return": p.get("physics_return"), "failed": p["failed"],
                   "stopped_at": p.get("stopped_at")}
                  for p in t["policies"]]
            row = make_row(t["sim"], "ladder", ps, _checks(t["sim"], res), _walk(t["sim"], res), t["run"], t["steps"])
            if row["failed"] != (t["verdict"] == "failed"):
                raise SystemExit(f"{t['sim']}: the ladder's verdict ({t['verdict']}) disagrees with its policies")
            rows.append(row)
    return fill_returns(rows), missing


# ------------------------------------------------------------------ helpers
def have(rows, kind, key, set_="expert"):
    """Rows that have the check (missing checks leave the analysis, listed; non-finite values stay and rank worst)."""
    if kind == "walk":
        return [r for r in rows if key in r["walk"]]
    return [r for r in rows if set_ in r["checks"] and key in r["checks"][set_]]


def col(rows, key, set_="expert"):
    if key in WALK_KEYS:
        return S2.better([r["walk"][key] for r in rows], lower_is_better=WALK_KEYS[key][1])
    return S2.better([r["checks"][set_][key] for r in rows])


def small(rows):
    return len(rows) < MIN_POOL


def too_small(rows):
    return S2.na(f"pool of {len(rows)} simulators (fewer than {MIN_POOL})", len(rows))


def rho(rows, key, y, lab, set_="expert"):
    use = have(rows, "walk" if key in WALK_KEYS else "checks", key, set_)
    use = _with_y(use, y)
    if small(use):
        return too_small(use)
    out = S2.rho_q(col(use, key, set_), _y(use, y), lab)
    out["left_out"] = _left(rows, use)
    return out


def diff(rows, new, base, y, lab, level, set_="expert"):
    ok = {id(r) for r in have(rows, "checks", base, set_)}
    use = [r for r in have(rows, "walk" if new in WALK_KEYS else "checks", new, set_) if id(r) in ok]
    use = _with_y(use, y)
    if small(use):
        return too_small(use)
    out = S2.diff_q(col(use, new, set_), col(use, base, set_), _y(use, y), lab, level)
    out["left_out"] = _left(rows, use)
    return out


def _with_y(rows, y):
    """Rows whose outcome exists (a missing seed-1 policy or a missing physics return leaves the row out, listed)."""
    v = _y(rows, y)
    return [r for r, x in zip(rows, v) if not np.isnan(x)]


def _left(rows, use):
    """Simulators left out of an analysis because a check is missing (listed)."""
    keep = {id(r) for r in use}
    return [r["sim"] for r in rows if id(r) not in keep]


def _y(rows, y):
    """The outcome: 'score' (mean walking score), 'score_p1', 'falls' (used as -fall rate), 'ret'."""
    if y == "falls":
        return -np.asarray([r["falls"] for r in rows], dtype=float)
    return np.asarray([r[y] for r in rows], dtype=float)


def family(rows, y="score", set_="expert", level=PRIMARY, with_r0=True):
    out = {}
    if with_r0:
        out["R0"] = rho(rows, "E100", y, R0_LAB, set_)
    out["R1"] = diff(rows, "lambda1", "E1", y, R1_LAB, level, set_)
    out["R2"] = diff(rows, "rJ", "E8", y, R2_LAB, level, set_)
    return out


def auc_one(x, failed):
    """P(a simulator that did not fail has a better check value than one that failed); ties count one half."""
    x, failed = np.asarray(x, dtype=float), np.asarray(failed, dtype=bool)
    a, b = x[~failed], x[failed]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    d = a[:, None] - b[None, :]
    return float(np.mean((d > 0) + 0.5 * (d == 0)))


def auc(rows, key, set_="expert"):
    use = have(rows, "checks", key, set_)
    if small(use):
        return too_small(use)
    f = np.asarray([r["failed"] for r in use])
    if f.sum() < 2 or (~f).sum() < 2:
        return S2.na(f"{int(f.sum())} failed and {int((~f).sum())} other simulators (at least 2 of each needed)",
                     len(use))
    x = col(use, key, set_)
    n = len(use)
    rng = np.random.default_rng(0)
    vals, dropped = [], 0
    for idx in rng.integers(0, n, size=(B, n)):
        v = auc_one(x[idx], f[idx])
        if np.isfinite(v):
            vals.append(v)
        else:
            dropped += 1
    if dropped > S2.MAX_DROPPED * B:
        return S2.na(f"{dropped} of {B} draws had only one kind of simulator (more than 10%)", n)
    ci = C.ci(vals, SECONDARY)
    return {"auc": auc_one(x, f), "ci95": ci, "label": C.label3(ci, *AUC_LAB, ref=0.5), "n_models": n,
            "n_failed": int(f.sum()), "dropped_draws": dropped}


def best_three(rows, y="score"):
    """Each check's three best simulators, and its shortfall: the mean score of the best three by score minus the
    mean score of the check's three (Step 2's selection: paired bootstrap, a simulator drawn twice is two entries,
    ties to the lower pool position)."""
    ok = [{id(r) for r in have(rows, "checks", k)} for k in SEL]
    use = [r for r in rows if all(id(r) in s for s in ok)]
    n = len(use)
    yy = _y(use, y)
    if n < 2 * TOP or not S2.computable(yy):
        return S2.na("fewer than 6 simulators with every check, or an outcome with fewer than 3 distinct values", n)
    sc = {k: col(use, k) for k in SEL}

    def shortfalls(idx):
        best = S2.top_mean(yy[idx], yy[idx], idx)
        return {k: best - S2.top_mean(sc[k][idx], yy[idx], idx) for k in SEL}

    full = shortfalls(np.arange(n))
    rng = np.random.default_rng(0)
    draws = [shortfalls(idx) for idx in rng.integers(0, n, size=(B, n))]
    out = {"best_three": [use[i]["sim"] for i in np.lexsort((np.arange(n), -yy))[:TOP]],
           "best_three_mean": S2.top_mean(yy, yy, np.arange(n)), "n_models": n, "checks": {}}
    for k in SEL:
        order = np.lexsort((np.arange(n), -sc[k]))[:TOP]
        out["checks"][k] = {"chosen": [use[i]["sim"] for i in order], "shortfall": full[k],
                            "ci95": C.ci([d[k] for d in draws], SECONDARY)}
    return out


# ------------------------------------------------------------------ the analysis
def analyse(rows, missing=(), ladder=None, record=None):
    prim = [r for r in rows if r["pool"] in ("REL", "ladder")]
    pooled = prim + [r for r in rows if r["pool"] == "step2"]
    seed0 = [r for r in prim if r["pool"] == "REL" or r["run"] in SEED0_RUNS]
    o = {"what": "Option D registered analysis (prereg/PREREG_D.md)", "n_boot": B, "seed": 0,
         "primary_level": PRIMARY, "secondary_level": SECONDARY, "fail_score": FAIL_SCORE, "min_pool": MIN_POOL,
         "missing": list(missing),
         "pools": {"primary": [r["sim"] for r in prim], "pooled": [r["sim"] for r in pooled],
                   "seed0": [r["sim"] for r in seed0]}}
    if small(prim):
        o["note"] = (f"The primary pool has {len(prim)} simulators, fewer than {MIN_POOL}: every primary analysis is "
                     f"NOT COMPUTABLE.")
    o["premise"] = {"R0": rho(prim, "E100", "score", R0_LAB)}
    o["primary"] = family(prim, "score", "expert", PRIMARY, with_r0=False)
    s = {}
    s["pooled"] = family(pooled, "score", "expert", SECONDARY)
    s["mixed"] = family(prim, "score", "mixed", SECONDARY)
    s["E368_vs_E8"] = diff(prim, "E368", "E8", "score", LONG_LAB, SECONDARY)
    w = {}
    for k in WALK_KEYS:
        w[k] = {"rho": rho(prim, k, "score", RHO_LAB), "minus_E8": diff(prim, k, "E8", "score", WALK_LAB, SECONDARY)}
    s["walk_test"] = w
    s["seed0_runs"] = family(seed0, "score", "expert", SECONDARY, with_r0=False)
    s["seed1_policy"] = family(prim, "score_p1", "expert", SECONDARY)
    s["failed_auc"] = {k: auc(prim, k) for k in CHECKS}
    s["fall_rate"] = family(prim, "falls", "expert", SECONDARY)
    fill_returns(prim)                                 # the fill uses the pool analysed (the primary pool)
    s["physics_return"] = family(prim, "ret", "expert", SECONDARY)
    s["best_three"] = best_three(prim, "score")
    o["secondary"] = s
    o["exploratory"] = {"rA_vs_E8": diff(prim, "rA", "E8", "score", RA_LAB, SECONDARY)}
    o["descriptive"] = describe(rows, ladder, record)
    return o


def describe(rows, ladder, record):
    d = {"simulators": [{k: r[k] for k in ("sim", "pool", "run", "steps", "n_policies", "score", "falls", "ret",
                                           "score_p1", "failed")} | {"policies": r["policies"]} for r in rows]}
    tested = [r for r in rows if r["pool"] == "ladder"]
    d["tests"] = len(tested)
    d["tests_with_third_policy"] = sum(r["n_policies"] == 3 for r in tested)
    d["checks"] = {r["sim"]: {"checks": r["checks"], "walk": r["walk"]} for r in rows}
    d["stops"] = [{"sim": r["sim"], "policy": p.get("policy"), "stopped_at": p.get("stopped_at")}
                  for r in rows for p in r["policies"] if p.get("status") == "stopped"]
    if ladder is not None:
        d["runs"] = {k: {"status": v.get("status"), "reason": v.get("reason") or (v.get("ended") or {}).get("why")}
                     for k, v in ladder["runs"].items()}
        d["job_hours"] = ladder.get("job", {}).get("elapsed_hours")
    if record is not None:
        d["descriptive_simulators"] = {s: {"how": record["simulators"][s]["how"],
                                           "checks_expert": (record["simulators"][s].get("checks") or {}),
                                           "policies": [{k: p.get(k) for k in ("policy", "score", "fall_rate",
                                                                               "role")}
                                                        for p in record["simulators"][s]["policies"]]}
                                       for s in DESCRIPTIVE if s in record["simulators"]}
    return d


# ------------------------------------------------------------------ report
def f(x, d=3):
    return "nan" if x is None or not np.isfinite(x) else f"{x:+.{d}f}"


def ci_s(c):
    return f"[{f(c[0])}, {f(c[1])}]"


def line(name, q, level=SECONDARY):
    if q.get("label") == NA and "rho_new" not in q and "rho" not in q:
        return f"{name}: NOT COMPUTABLE ({q['why']}; n = {q['n_models']})"
    if "auc" in q:
        return f"{name}: AUC {q['auc']:.3f}, 95% {ci_s(q['ci95'])} -> {q['label']} (n = {q['n_models']})"
    if "delta" in q:
        lv = q.get("level", level)
        s = (f"{name}: rho {f(q['rho_new'])} - {f(q['rho_base'])} = {f(q['delta'])}, {lv:g}% "
             f"{ci_s(q[f'ci{lv:g}'])} -> {q['label']} (n = {q['n_models']}, dropped draws {q['dropped_draws']})")
        return s + (f"; {q['why']}" if q.get("why") else "")
    return (f"{name}: rho {f(q['rho'])}, 95% {ci_s(q['ci95'])} -> {q['label']} (n = {q['n_models']}, dropped draws "
            f"{q['dropped_draws']})")


def report(o):
    L = ["Option D: registered analysis (prereg/PREREG_D.md). World: the calibrated hand-built MuJoCo ANYmal D with a",
         "fitted PD motor (not Isaac Lab, not hardware). Outcome: mean walking score of a simulator's 2-3 policies.",
         f"Primary pool ({len(o['pools']['primary'])}): {', '.join(o['pools']['primary'])}",
         f"Pooled ({len(o['pools']['pooled'])}); seed-0 runs with REL ({len(o['pools']['seed0'])})."]
    if o.get("note"):
        L.append(o["note"])
    for m in o["missing"]:
        L.append(f"missing: {m}")
    L.append("")
    L.append("Premise (95%)")
    L.append("  " + line("R0 rho(-E100, score)", o["premise"]["R0"]))
    L.append("Primary family (97.5% each)")
    L.append("  " + line("R1 rho(-lambda1) - rho(-E1)", o["primary"]["R1"], PRIMARY))
    L.append("  " + line("R2 rho(-rJ) - rho(-E8)", o["primary"]["R2"], PRIMARY))
    s = o["secondary"]
    L.append("Secondary (95%)")
    for key, title in (("pooled", "pooled pool"), ("mixed", "mixed test set"), ("seed0_runs", "seed-0 runs and REL"),
                       ("seed1_policy", "seed-1 policy alone"), ("fall_rate", "outcome -fall rate"),
                       ("physics_return", "outcome physics return")):
        for q, v in s[key].items():
            L.append("  " + line(f"{q} [{title}]", v))
    L.append("  " + line("rho(-E368) - rho(-E8)", s["E368_vs_E8"]))
    for k, v in s["walk_test"].items():
        L.append("  " + line(f"walk test {k}: rho", v["rho"]))
        L.append("  " + line(f"walk test {k}: minus rho(-E8)", v["minus_E8"]))
    for k, v in s["failed_auc"].items():
        L.append("  " + line(f"'failed' separated by {k}", v))
    bt = s["best_three"]
    if bt.get("label") == NA:
        L.append(f"  best three: NOT COMPUTABLE ({bt['why']})")
    else:
        L.append(f"  best three by score: {', '.join(bt['best_three'])} (mean {bt['best_three_mean']:.3f})")
        for k, v in bt["checks"].items():
            L.append(f"    {k:8s} chose {', '.join(v['chosen'])}; shortfall {f(v['shortfall'])}, 95% {ci_s(v['ci95'])}")
    L.append("Exploratory")
    L.append("  " + line("rho(-rA) - rho(-E8)", o["exploratory"]["rA_vs_E8"]))
    d = o["descriptive"]
    L.append(f"Descriptive: {d['tests']} ladder tests; {d['tests_with_third_policy']} needed a third policy.")
    return "\n".join(L) + "\n"


def write_once(path, text):
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; not overwriting")
    with open(path, "w") as fh:
        fh.write(text)


def main(res=RES, out_json=None, out_txt=None):
    rows, missing = load(res)
    o = analyse(rows, missing, _json(os.path.join(res, "d_ladder.json")), _json(os.path.join(res, "d_record.json")))
    txt = report(o)
    write_once(out_json or os.path.join(res, "d_stats.json"), json.dumps(o, indent=1, default=float))
    write_once(out_txt or os.path.join(res, "d_stats.txt"), txt)
    print(txt)
    return o


if __name__ == "__main__":
    main()

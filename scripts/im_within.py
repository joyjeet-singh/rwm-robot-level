"""
I1 (improvement round): how well each check ranks simulators that were trained the same way.  EXPLORATORY.

Added after the registered results were seen.  Nothing is trained, rolled out or measured: every value is a stored
per-model value (results/model_table.json, results/s1_models.json), ranked against the stored 100-step error E100
under recorded joint commands.  Writes results/im_within.json and results/im_within.txt (at most 40 lines).

Groups (fixed by IMPROVE_PLAN.md before anything was computed):
  a  F18's 48 rollout-trained simulators (pool R), whose training settings differ;
  b  X5's rollout-trained simulators: the 5 seeds and 3 ensembles of each of the 5 splits (40 in all); rho is taken
     within each split (8 models) and averaged; models are resampled within splits;
  c  F18's 40 compressed copies (pool C);
  d  all 97 of F18, for reference: these must reproduce the stored values of Table 2 exactly.
Checks: E1, E8, lambda_1 and rbar_J under recorded joint commands; on F18 also lambda_1 and rbar_J under the policy's
joint commands and the negative of the imagined return (higher return is better, so the sign is flipped to read like
the error-type checks).  Intervals: 95% percentile intervals from 20,000 bootstrap draws of models,
numpy.random.default_rng(0); the two differences use the same draws as the intervals.
Spearman, the non-finite rule (non-finite ranks worst, tied) and the percentile helper are imported, never copied.
"""
import json
import os

import numpy as np

import common as C
import stats as ST

B = C.N_BOOT
assert B == 20000
SIZES = {"a": 48, "b": 40, "c": 40, "d": 97}
REC = ("E1", "E8", "lam1", "rJ")
POLICY = ("lam1_pol", "rJ_pol", "negret")
NAMES = {"E1": "E1", "E8": "E8", "lam1": "lambda1", "rJ": "rJ", "lam1_pol": "lambda1 (policy cmds)",
         "rJ_pol": "rJ (policy cmds)", "negret": "-imagined return"}
OUT = os.path.join(C.RESULTS, "im_within.json")
TXT = os.path.join(C.RESULTS, "im_within.txt")


def _ld(path):
    with open(path) as fh:
        return json.load(fh)


def groups_of_models():
    table = _ld(os.path.join(C.RESULTS, "model_table.json"))
    s1 = _ld(os.path.join(C.RESULTS, "s1_models.json"))["models"]["F18"]
    f18, x5 = table["F18"], table["X5"]
    rows = []
    for r in f18:
        s = s1[r["key"]]                                   # stored per-model values of study CL
        assert s["n_div_pol"] == 0, r["key"]               # no policy rollout diverged (as in CL6's full model set)
        rows.append(dict(r, lam1_pol=s["lam1_pol"], rJ_pol=s["rJ_pol"], ret=s["ret"], negret=-s["ret"]))
    G = {"a": [[r for r in rows if r["pool"] == "R"]],
         "c": [[r for r in rows if r["pool"] == "C"]],
         "d": [rows]}
    x5r = [r for r in x5 if r["group"] in ("A", "A5")]
    G["b"] = [[r for r in x5r if r["arena"] == C.fid(f)] for f in C.FOLDS]
    sizes = {k: sum(len(s) for s in v) for k, v in G.items()}
    if sizes != SIZES or any(len(s) != 8 for s in G["b"]) or len(G["b"]) != 5:
        raise SystemExit(f"im_within: refusing to run, group sizes {sizes} differ from the definitions {SIZES}")
    return G, rows


def boot(splits, checks, rng):
    """rho(check, E100) per check: point value and B draws (models resampled within each split, mean over splits).
    Draws are made as stats.py makes them: one (B, n) block for one split (boot_rho); one draw of every split per
    iteration for several (boot_diff_within).  A draw in which any check is non-finite is dropped for all checks."""
    arr = [{c: np.array([r[c] for r in s], float) for c in checks + ("E100",)} for s in splits]
    block = rng.integers(0, len(splits[0]), size=(B, len(splits[0]))) if len(splits) == 1 else None

    def stat(sel):
        return {c: float(np.mean([ST.spearman(a[c][i], a["E100"][i]) for a, i in zip(arr, sel)])) for c in checks}

    full = stat([np.arange(len(s)) for s in splits])
    V = {c: np.empty(B) for c in checks}
    for b in range(B):
        sel = [block[b]] if block is not None else [rng.integers(0, len(s), len(s)) for s in splits]
        v = stat(sel)
        for c in checks:
            V[c][b] = v[c]
    keep = np.all([np.isfinite(V[c]) for c in checks], axis=0)
    return full, {c: V[c][keep] for c in checks}, int((~keep).sum()), \
        [{c: ST.spearman(a[c], a["E100"]) for c in checks} for a in arr]


def diff_block(full, V, new, base):
    d = V[new] - V[base]
    return {"delta": full[new] - full[base], "ci95": C.ci(d)}


def main():
    G, rows = groups_of_models()
    res = {}
    for g in ("a", "b", "c", "d"):
        checks = REC + (POLICY if g != "b" else ())
        full, V, dropped, per = boot(G[g], checks, np.random.default_rng(0))
        res[g] = {"n_models": SIZES[g], "n_splits": len(G[g]), "dropped_draws": dropped,
                  "checks": {c: {"rho": full[c], "ci95": C.ci(V[c])} for c in checks},
                  "diffs": {"lam1_minus_E8": diff_block(full, V, "lam1", "E8"),
                            "rJ_minus_E8": diff_block(full, V, "rJ", "E8")}}
        if g == "b":
            res[g]["per_split_rho"] = {c: [p[c] for p in per] for c in checks}
        # the draws are those of stats.py, so the intervals and differences must equal its own helpers' (a, c, d: boot_rho
        # per check; b: boot_diff_within), unless a draw was dropped
        if g == "b":
            for new in ("lam1", "rJ"):
                want = ST.boot_diff_within(G["b"], new, "E8", "E100")
                got = res[g]["diffs"][f"{new}_minus_E8"]
                if dropped == 0 and (got["ci95"] != want["ci95"] or got["delta"] != want["delta"]):
                    raise SystemExit(f"im_within: group b {new} - E8 differs from stats.boot_diff_within")
        else:
            rows_g = G[g][0]
            for c in REC:
                want = ST.boot_rho([r[c] for r in rows_g], [r["E100"] for r in rows_g])
                if dropped == 0 and (res[g]["checks"][c]["ci95"] != want["ci95"] or full[c] != want["rho"]):
                    raise SystemExit(f"im_within: group {g} {c} differs from stats.boot_rho")

    # ---- group d reproduces the stored values of Table 2 exactly (rho only; the intervals are this script's own)
    st, s1s = _ld(os.path.join(C.RESULTS, "stats.json")), _ld(os.path.join(C.RESULTS, "s1_stats.json"))["CL"]
    d = res["d"]["checks"]
    want = {"E1": st["Q1"]["rho_base"], "E8": st["Q2"]["rho_base"], "lam1": st["Q1"]["rho_new"],
            "rJ": st["Q4_vs_E1"]["rho_new"], "lam1_pol": s1s["CL2"]["rho_new"], "rJ_pol": s1s["CL1"]["rho_new"]}
    def need(ok, what):
        if not ok:
            raise SystemExit(f"im_within: group d does not reproduce the stored values: {what}")

    for k, v in want.items():
        need(d[k]["rho"] == v, f"{k}: {d[k]['rho']} vs {v}")
    need(s1s["CL6"]["n_models"] == 97, "CL6 n_models")
    ret_rho = ST.spearman([r["ret"] for r in G["d"][0]], [r["E100"] for r in G["d"][0]])
    need(ret_rho == s1s["CL6"]["rho"], f"return: {ret_rho} vs {s1s['CL6']['rho']}")
    need(abs(d["negret"]["rho"] + ret_rho) < 1e-12, "negated return")
    need(abs(d["negret"]["ci95"][0] + s1s["CL6"]["ci95"][1]) < 1e-9
         and abs(d["negret"]["ci95"][1] + s1s["CL6"]["ci95"][0]) < 1e-9, "negated return interval")
    # the intervals of E1, E8, lambda1 and rJ were already checked against stats.boot_rho above

    repro = {k: {"here": d[k]["rho"], "stored": v, "equal": True} for k, v in want.items()}
    repro["negret"] = {"here": d["negret"]["rho"], "stored": -s1s["CL6"]["rho"], "equal": True,
                       "note": "stored CL6 rho is for the return itself; the check here is its negative"}
    res["reproduction_of_stored_values"] = repro

    # ---- already known: the growth rate against the 368-step error, earlier studies (appendix A2)
    ng = _ld(os.path.join(C.PDM, "NGCAPS", "results", "main.json"))["Q1"]
    ic = _ld(os.path.join(C.PDM, "ICMAI", "results", "lyap_main.json"))["H3"]
    res["known_earlier"] = {
        "a": {"rho": ng["rho"], "ci": ng["ci"], "source": "NGCAPS/results/main.json Q1", "target": "E368"},
        "b": {"rho": ic["rho"], "ci": ic["ci"], "source": "ICMAI/results/lyap_main.json H3", "target": "E368"}}
    res["status"] = "EXPLORATORY: added after the registered results were seen; stored values only"
    res["definition"] = {"target": "E100 under recorded commands", "draws": B, "seed": 0, "level": 95,
                         "groups": SIZES, "b": "rho within each of 5 splits (8 models), averaged; resampled within splits"}
    json.dump(res, open(OUT, "w"), indent=1)
    write_text(res)
    print(open(TXT).read())


def write_text(res):
    f = lambda v: f"{v:+.2f}"                                                          # noqa: E731
    L = ["I1 within-method rankings, EXPLORATORY (added after the registered results; stored values only).",
         "Spearman rho with E100 (recorded commands); 95% interval, 20,000 draws of models, default_rng(0); the return",
         "enters negated. a = 48 rollout-trained (F18); b = 40 rollout-trained (X5: rho per split of 8, averaged, resampled",
         "within splits; the 3 ensembles per split may share seeds, so b intervals are likely too narrow);",
         "c = 40 compressed copies (F18); d = all 97 of F18 (its rho values reproduce Table 2 exactly, asserted)."]
    for g in ("a", "b", "c", "d"):
        L.append(f"-- group {g} (n = {res[g]['n_models']})")
        for c, v in res[g]["checks"].items():
            L.append(f"   {NAMES[c]:24s} {f(v['rho'])}  [{f(v['ci95'][0])}, {f(v['ci95'][1])}]")
        dl, dj = res[g]["diffs"]["lam1_minus_E8"], res[g]["diffs"]["rJ_minus_E8"]
        L.append(f"   minus rho(E8): lambda1 {f(dl['delta'])} [{f(dl['ci95'][0])}, {f(dl['ci95'][1])}]; "
                 f"rJ {f(dj['delta'])} [{f(dj['ci95'][0])}, {f(dj['ci95'][1])}]")
    ka = res["known_earlier"]
    L.append("Already known (appendix A2, an earlier study, lambda1 against the 368-step error): "
             f"a {f(ka['a']['rho'])}, b {f(ka['b']['rho'])}.")
    if len(L) > 40:
        raise SystemExit(f"im_within: text has {len(L)} lines, at most 40 allowed")
    open(TXT, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

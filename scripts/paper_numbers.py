"""
Every number the paper prints, read from a result file (never typed): this study's results/ and, read-only, the
earlier studies of the same models in ../rwm_repro, ../NGCAPS, ../ICPAIAS and ../PerconAI.
Writes results/paper_numbers.json: {key: {"value": LaTeX string, "source": file, "path": where in it}}.
"""
import json
import math
import os

import common as C

N = {}
PDM = C.PDM
SRC = {
    "tmlr": os.path.join(PDM, "rwm_repro", "results", "paper_numbers.json"),
    "ngcaps": os.path.join(PDM, "NGCAPS", "results", "main.json"),
    "icp_detect": os.path.join(PDM, "ICPAIAS", "results", "r2_detect.json"),
    "icp_verdicts": os.path.join(PDM, "ICPAIAS", "results", "r2_verdicts.json"),
    "icp_repair": os.path.join(PDM, "ICPAIAS", "results", "r2_repair.json"),
    "perc": os.path.join(PDM, "PerconAI", "results", "paper_numbers.json"),
    "stats": os.path.join(C.RESULTS, "stats.json"),
    "checks": os.path.join(C.RESULTS, "checks.json"),
    "models": os.path.join(C.RESULTS, "models.json"),
    "figfacts": os.path.join(C.RESULTS, "figure_facts.json"),
    "explo": os.path.join(C.RESULTS, "exploratory.json"),
}
J = {}


def load(name):
    if name not in J:
        J[name] = json.load(open(SRC[name]))
    return J[name]


def rel(name):
    return os.path.relpath(SRC[name], os.path.dirname(C.AIR))


# ------------------------------------------------------------------ formatting
MINUS = "$-$"


def sig(x, n=2):
    """x to n significant figures, trailing zeros kept, LaTeX minus."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        raise ValueError(f"non-finite value {x}")
    if x == 0:
        return "0"
    d = n - 1 - int(math.floor(math.log10(abs(x))))
    d = max(d, 0)
    s = f"{abs(round(x, d)):.{d}f}"
    if abs(round(x, d)) >= 10 ** n and d == 0:                      # e.g. 1827 with n=2 stays 1827, not 1800
        s = f"{abs(x):.0f}"
    return (MINUS if x < 0 else "") + s


def dp(x, d=2, signed=False):
    s = f"{abs(x):.{d}f}"
    if x < 0 and float(s) != 0:
        return MINUS + s
    return ("+" if signed else "") + s


def sgn(x, n=2):
    return ("+" if x >= 0 else "") + sig(x, n)


def pct(x, n=2):
    """fraction -> percent string with n significant figures."""
    return sig(100 * x, n) + r"\%"


def cib(lo, hi, f):
    return "[" + f(lo) + ", " + f(hi) + "]"


def put(key, value, source, path):
    assert key not in N, key
    N[key] = {"value": str(value), "source": source, "path": path}


def tm(key):
    """A TMLR paper number (string as stored) and its own source."""
    v = load("tmlr")[key]
    return v["value"], v.get("source", "")


def tm_float(key):
    return float(str(tm(key)[0]).replace(",", "").replace("+", "").replace("%", ""))


def tm_ci(key):
    s = tm(key)[0].strip("[]").split(",")
    return float(s[0].replace("+", "")), float(s[1].replace("+", ""))


# ------------------------------------------------------------------ earlier studies
def earlier():
    t = rel("tmlr")
    # the rebuild and the data
    put("rows", tm("rows")[0].replace(",", "{,}"), t, "rows")
    put("budget_pct", sig(tm_float("c2_pct"), 2) + r"\%", t, "c2_pct")
    put("diff_terms", tm("diff_terms")[0], t, "diff_terms")
    put("diff_tensors", tm("diff_n_params")[0], t, "diff_n_params")
    assert tm("diff_grad_max")[0] == "0.000e+00"
    put("diff_max", "0", t, "diff_grad_max")
    put("win_naive", tm("win_naive")[0].replace(",", "{,}"), t, "win_naive")
    put("win_cross", tm("win_cross")[0], t, "win_cross")
    # the stale action
    put("stale_h1", sig(tm_float("adh20_rel_h1"), 2) + r"\%", t, "adh20_rel_h1")
    lo, hi = tm_ci("adh20_rel_ci_h1")
    put("stale_h1_ci", cib(lo, hi, lambda v: sig(v, 2) + r"\%"), t, "adh20_rel_ci_h1")
    put("stale_h100", sig(tm_float("adh20_rel_h100"), 2) + r"\%", t, "adh20_rel_h100")
    lo, hi = tm_ci("adh20_rel_ci_h100")
    put("stale_h100_ci", cib(lo, hi, lambda v: sig(v, 2) + r"\%"), t, "adh20_rel_ci_h100")
    # teacher forcing against rollout training, five folds (pre-registered rule X3)
    put("x3_gap", sgn(tm_float("x3_gap")), t, "x3_gap")
    lo, hi = tm_ci("x3_ci")
    put("x3_ci", cib(lo, hi, sgn), t, "x3_ci")
    put("x3_ratio", tm("x3_ratio_h368")[0], t, "x3_ratio_h368")
    put("x3_eps_pos", tm("x3_eps_pos")[0], t, "x3_eps_pos")
    put("x3_n_eps", tm("x3_n_eps")[0], t, "x3_n_eps")
    put("x3_p", tm("n7_x3_ep_p_h368")[0], t, "n7_x3_ep_p_h368")
    put("x3_reading", tm("x3_reading")[0].lower(), t, "x3_reading")
    put("x3_h1_gap", sgn(tm_float("x3_h1_gap")), t, "x3_h1_gap")
    lo, hi = tm_ci("x3_h1_ci")
    put("x3_h1_ci", cib(lo, hi, sgn), t, "x3_h1_ci")
    put("x3_h1_n", tm("x3_h1_n")[0], t, "x3_h1_n")
    put("x3_h1_unit", tm("x3_h1_unit")[0], t, "x3_h1_unit")
    # uncertainty of the released ensemble
    for h in (1, 100, 368):
        put(f"k_h{h}", tm(f"n8_k_rel_epi_h{h}")[0], t, f"n8_k_rel_epi_h{h}")
    put("cov1_h100", sig(tm_float("x9_rel_mean_cov1_h100"), 2) + r"\%", t, "x9_rel_mean_cov1_h100")
    put("cov1_h100_per", tm("x9_rel_per_cov1_h100")[0] + r"\%", t, "x9_rel_per_cov1_h100")
    put("cov1_h1", sig(tm_float("d1n_epi_cov1_h1"), 2) + r"\%", t, "d1n_epi_cov1_h1")
    put("cov_nom1", tm("v3_cov_nominal1")[0] + r"\%", t, "v3_cov_nominal1")
    put("cov_nom2", tm("v3_cov_nominal2")[0] + r"\%", t, "v3_cov_nominal2")
    put("dis_spear_h100", sgn(tm_float("n11_d2_sp_epi_h100")), t, "n11_d2_sp_epi_h100")
    put("shared_pct", sig(tm_float("v1_shared_pct"), 2) + r"\%", t, "v1_shared_pct")
    put("synth_span", sig(tm_float("e5s_span"), 2), t, "e5s_span")
    put("synth_under", tm("e5s_mse_under")[0], t, "e5s_mse_under")
    put("synth_nll", sig(tm_float("e5s_nll_ratio"), 2), t, "e5s_nll_ratio")
    put("x7_summary", tm("x7_summary")[0].lower(), t, "x7_summary")
    put("cov2_perh_h100", tm("n9_epi_perh_cov2_h100")[0] + r"\%", t, "n9_epi_perh_cov2_h100")

    # NGCAPS: training horizon and stability (pre-registered Q4), run-time growth signal (Q3)
    g, ng = load("ngcaps"), rel("ngcaps")
    q4 = g["Q4"]
    assert q4["verdict"] == "LONGER IS MORE STABLE"
    put("ng_q4_rho", sgn(q4["rho"], 2), ng, "Q4.rho")
    put("ng_q4_ci", cib(*q4["ci"], f=lambda v: sgn(v, 2)), ng, "Q4.ci")
    put("ng_q4_n", str(q4["n"]), ng, "Q4.n")
    q3 = g["Q3"]
    assert q3["verdict"] == "WORSE THAN COUNTER"
    put("ng_q3_auc", sig(q3["mean_auc_g"], 2), ng, "Q3.mean_auc_g")
    put("ng_q3_counter", sig(q3["mean_auc_counter"], 2), ng, "Q3.mean_auc_counter")
    put("ng_q3_delta", dp(q3["delta_auc_g_minus_counter"], 2, signed=True), ng, "Q3.delta_auc_g_minus_counter")
    put("ng_q3_ci", cib(*q3["ci"], f=lambda v: dp(v, 2, signed=True)), ng, "Q3.ci")
    tp = g["descriptive"]["training_paths"]
    put("ng_pathA_lam_500", sgn(tp["A_500"]["lam1"]), ng, "descriptive.training_paths.A_500.lam1")
    put("ng_pathA_lam_10000", sgn(tp["A_10000"]["lam1"]), ng, "descriptive.training_paths.A_10000.lam1")
    put("ng_pathB_lam_500", sgn(tp["B_500"]["lam1"]), ng, "descriptive.training_paths.B_500.lam1")
    put("ng_pathB_lam_10000", sgn(tp["B_10000"]["lam1"]), ng, "descriptive.training_paths.B_10000.lam1")
    put("ng_pathB_E_2500", sig(tp["B_2500"]["E368"]), ng, "descriptive.training_paths.B_2500.E368")
    put("ng_pathB_E_10000", sig(tp["B_10000"]["E368"]), ng, "descriptive.training_paths.B_10000.E368")

    # ICPAIAS: physics score as an alarm, repairs, re-synchronisation (registered round 2)
    d, nd = load("icp_detect")["component_auroc"]["A-ens"], rel("icp_detect")
    put("icp_auc_phys", f"{d['P_on']['auroc']:.2f}", nd, "component_auroc.A-ens.P_on.auroc")
    put("icp_auc_phys_ci", cib(*d["P_on"]["ci"], f=lambda v: f"{v:.2f}"), nd, "component_auroc.A-ens.P_on.ci")
    put("icp_auc_dis", f"{d['F(D)']['auroc']:.2f}", nd, "component_auroc.A-ens.F(D).auroc")
    put("icp_auc_counter", f"{d['C']['auroc']:.2f}", nd, "component_auroc.A-ens.C.auroc")
    v, nv = load("icp_verdicts")["families"], rel("icp_verdicts")
    h7 = v["use"]["tests"]["H7"]
    assert h7["verdict"] == "SCHEDULE BETTER"
    put("icp_h7_excess", f"{100 * (math.exp(h7['estimate']) - 1):.1f}" + r"\%", nv, "families.use.tests.H7.estimate")
    put("icp_h7_ci", cib(*h7["ci"], f=lambda x: f"{100 * (math.exp(x) - 1):.1f}" + r"\%"), nv,
        "families.use.tests.H7.ci")
    r, nr = load("icp_repair")["A5"]["ens5"], rel("icp_repair")
    put("icp_G", sig(100 * (r["err|G|368"]["ratio"] - 1), 2) + r"\%", nr, "A5.ens5.err|G|368.ratio")
    put("icp_G_ci", cib(*r["err|G|368"]["ci"], f=lambda x: sig(100 * (x - 1), 2) + r"\%"), nr,
        "A5.ens5.err|G|368.ci")
    put("icp_J_x", sig(r["err|J|368"]["ratio"], 2), nr, "A5.ens5.err|J|368.ratio")

    # PerconAI: 8-bit dynamic quantisation (pre-registered Q1) and CPU time per step
    p, npc = load("perc")["keys"], rel("perc")
    assert p["e2.Q1_int8_dynamic.label"]["value"] == "KEEPS"
    put("perc_int8_K", f"{100 * p['e2.Q1_int8_dynamic.K.est']['value']:.1f}" + r"\%", npc, "keys.e2.Q1_int8_dynamic.K.est")
    put("perc_int8_K_ci", cib(p["e2.Q1_int8_dynamic.K.lo"]["value"], p["e2.Q1_int8_dynamic.K.hi"]["value"],
                              lambda x: f"{100 * x:.1f}" + r"\%"), npc, "keys.e2.Q1_int8_dynamic.K.lo/hi")
    us = lambda k: p[f"e1c.{k}.step_b1_t1.us.median"]["value"]                          # noqa: E731
    put("perc_us_fp32", sig(us("a25_fp32"), 3), npc, "keys.e1c.a25_fp32.step_b1_t1.us.median")
    put("perc_us_jit_int8", sig(us("a25_jit_int8_dynamic"), 3), npc,
        "keys.e1c.a25_jit_int8_dynamic.step_b1_t1.us.median")
    put("perc_ratio_jit", sig(us("a25_jit_int8_dynamic") / us("a25_jit_fp32"), 2), npc,
        "keys.e1c.a25_jit_int8_dynamic / a25_jit_fp32 (step_b1_t1.us.median)")
    put("perc_ratio_eager", sig(us("a25_int8_dynamic") / us("a25_fp32"), 2), npc,
        "keys.e1c.a25_int8_dynamic / a25_fp32 (step_b1_t1.us.median)")


if __name__ == "__main__":
    earlier()
    import air_numbers
    air_numbers.add(put, sig, sgn, pct, cib, dp, load, rel)
    import revision_numbers
    revision_numbers.add(put, sig, sgn, pct, cib, dp, load, rel)
    import s1_numbers
    s1_numbers.add(put, sig, sgn, pct, cib, dp, load, rel)
    json.dump(N, open(os.path.join(C.RESULTS, "paper_numbers.json"), "w"), indent=1, sort_keys=True)
    with open(os.path.join(C.RESULTS, "paper_numbers.md"), "w") as f:
        f.write("# Every number the paper can print, and where it comes from\n\n"
                "Generated by scripts/paper_numbers.py; paths are relative to the PDM folder.\n\n"
                "| key | value | source | field |\n|---|---|---|---|\n")
        for k in sorted(N):
            v = N[k]
            f.write(f"| `{k}` | {v['value'].replace('|', '/')} | `{v['source']}` | `{v['path'].replace('|', '/')}` |\n")
    print(len(N), "numbers")

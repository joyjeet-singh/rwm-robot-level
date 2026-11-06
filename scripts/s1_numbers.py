"""Numbers for Step 1 (called by paper_numbers.py after revision_numbers). Every value is read from a result file.

Sources: results/s1_stats.json (registered analysis of studies CL and AR, scripts/s1_stats.py), results/s1_models.json
(the job's output, scripts/s1_eval.py), results/s1_extra.json (EXPLORATORY additions, scripts/s1_extra.py),
results/s1_budget.json (study TB, DESCRIPTIVE, scripts/s1_budget.py), scripts/s1_rollout.py (the noise scale) and git
(the registration commit). Secondary results must be printed as secondary, exploratory ones as exploratory, and every
study TB value as descriptive.
"""
import datetime as dt
import json
import math
import os
import re
import subprocess

import common as C

STATS = os.path.join(C.RESULTS, "s1_stats.json")
MODELS = os.path.join(C.RESULTS, "s1_models.json")
EXTRA = os.path.join(C.RESULTS, "s1_extra.json")
BUDGET = os.path.join(C.RESULTS, "s1_budget.json")
ROLLOUT = os.path.join(C.AIR, "scripts", "s1_rollout.py")
ARCHES = ("mlp", "rssm", "transformer")


def _rel(path):
    return os.path.relpath(path, os.path.dirname(C.AIR))


def _ld(path):
    with open(path) as fh:
        return json.load(fh)


def add(put, sig, sgn, pct, cib, dp, load, rel):
    S, s = _ld(STATS), _rel(STATS)
    M, m = _ld(MODELS), _rel(MODELS)
    X, x = _ld(EXTRA), _rel(EXTRA)
    B, b = _ld(BUDGET), _rel(BUDGET)
    assert X["label"] == "exploratory"
    assert all(S["gates"]["checks"].values()) and S["gates"]["CL"]["verdict"] and S["gates"]["AR"]["verdict"]
    r2 = lambda v: dp(v, 2)                                                              # noqa: E731
    r3 = lambda v: dp(v, 3, signed=True)                                                 # noqa: E731
    low = lambda t: t.lower()                                                            # noqa: E731
    CL, AR = S["CL"], S["AR"]

    def diffkeys(key, blk, path, level="ci95", lab="label"):
        put(key + "_delta", r3(blk["delta"]), s, path + ".delta")
        put(key + "_ci", cib(*blk[level], f=r3), s, f"{path}.{level}")
        put(key + "_label", low(blk[lab]), s, f"{path}.{lab}")

    # ---- registration (git) and the job
    out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h %ct", "--", "prereg/PREREG_S1.md"],
                         cwd=C.AIR, capture_output=True, text=True, check=True).stdout.split()
    assert len(out) == 2, out
    t_reg = dt.datetime.fromtimestamp(int(out[1]), dt.timezone.utc)
    put("s1_reg_commit", out[0], "git", "log --diff-filter=A -- prereg/PREREG_S1.md (%h)")
    put("s1_reg_time", t_reg.strftime("%Y-%m-%d %H:%M") + " UTC", "git",
        "log --diff-filter=A -- prereg/PREREG_S1.md (%ct, printed in UTC)")
    t_job = dt.datetime.fromisoformat(M["created"]).astimezone(dt.timezone.utc)      # job's local time
    assert t_job > t_reg, (t_job, t_reg)
    put("s1_job_hours", sig(sum(p["seconds"] for p in M["phases"].values()) / 3600, 2), m, "phases.*.seconds / 3600")

    # ---- design
    txt = open(ROLLOUT).read()
    nf = float(re.search(r"^NOISE_FRACTION = ([0-9.]+)$", txt, re.M).group(1))
    put("s1_noise_frac", sig(nf, 1), _rel(ROLLOUT), "NOISE_FRACTION")
    put("s1_margin", dp(S["margin"], 2), s, "margin")
    put("s1_nF18", str(CL["n"]["F18"]), s, "CL.n.F18")
    put("s1_nX5", str(CL["n"]["X5"]), s, "CL.n.X5")
    assert not CL["n"]["F18_missing"] and not CL["n"]["X5_missing"]
    c1 = M["checks"]["S1-C1"]["per_pool"]
    rj = c1["F18"]["rJ"]
    put("s1_c1_rJ_nout", str(len(rj["outside"])), m, "checks.S1-C1.per_pool.F18.rJ.outside (count)")
    put("s1_c1_rJ_maxrel", sig(100 * rj["max_difference"], 1) + r"\%", m, "checks.S1-C1.per_pool.F18.rJ.max_difference")
    rmin = min(c1[p][c]["spearman_recomputed_vs_stored"] for p in c1 for c in ("lam1", "rJ"))
    put("s1_c1_rho_min", f"{math.floor(rmin * 1e4) / 1e4:.4f}",
        m, "checks.S1-C1.per_pool.*.*.spearman_recomputed_vs_stored (min)")
    assert all(rj_o.startswith("B10k_") for rj_o in rj["outside"])

    # ---- A8 table: rho with E100 (F18 registered values; X5 within-split means)
    put("s1_t_F18_lam1_rec", r2(CL["CL2"]["rho_base"]), s, "CL.CL2.rho_base")
    put("s1_t_F18_lam1_pol", r2(CL["CL2"]["rho_new"]), s, "CL.CL2.rho_new")
    put("s1_t_F18_lam1_noise", r2(CL["CL8"]["CL2"]["rho_new"]), s, "CL.CL8.CL2.rho_new")
    put("s1_t_F18_lam1_cl", r2(CL["CL5"]["rho_new"]), s, "CL.CL5.rho_new")
    put("s1_t_F18_rJ_rec", r2(CL["CL1"]["rho_base"]), s, "CL.CL1.rho_base")
    put("s1_t_F18_rJ_pol", r2(CL["CL1"]["rho_new"]), s, "CL.CL1.rho_new")
    put("s1_t_F18_rJ_noise", r2(CL["CL8"]["CL1"]["rho_new"]), s, "CL.CL8.CL1.rho_new")
    xm = X["X5_within_split_mean_rho_with_E100"]
    for k in ("lam1_rec", "lam1_pol", "lam1_noise", "lam1_cl", "rJ_rec", "rJ_pol", "rJ_noise"):
        put(f"s1_t_X5_{k}", r2(xm[k]["mean"]), x, f"X5_within_split_mean_rho_with_E100.{k}.mean")

    # ---- primary CL (97.5%)
    for q in ("CL1", "CL2"):
        diffkeys(f"s1_{q.lower()}", CL[q], f"CL.{q}", level="ci97.5", lab="verdict")
    # ---- secondary CL (95%)
    for c in ("lam1", "rJ"):
        put(f"s1_cl3_{c}_rho", r2(CL["CL3"][c]["rho"]), s, f"CL.CL3.{c}.rho")
        put(f"s1_cl3_{c}_ci", cib(*CL["CL3"][c]["ci95"], f=r2), s, f"CL.CL3.{c}.ci95")
    for q in ("CL1", "CL2"):
        diffkeys(f"s1_cl4_{q.lower()}", CL["CL4"][q], f"CL.CL4.{q}")
        diffkeys(f"s1_cl8_{q.lower()}", CL["CL8"][q], f"CL.CL8.{q}")
        for g in ("trained", "compressed"):
            diffkeys(f"s1_cl9_{g}_{q.lower()}", CL["CL9"][g][q], f"CL.CL9.{g}.{q}")
    for g in ("trained", "compressed"):
        put(f"s1_cl9_{g}_n", str(CL["CL9"][g]["CL1"]["n_models"]), s, f"CL.CL9.{g}.CL1.n_models")
    diffkeys("s1_cl5", CL["CL5"], "CL.CL5")
    put("s1_cl6_rho", sig(CL["CL6"]["rho"], 2), s, "CL.CL6.rho")
    put("s1_cl6_ci", cib(*CL["CL6"]["ci95"], f=r2), s, "CL.CL6.ci95")
    put("s1_cl6_nexcl", str(CL["CL6"]["n_excluded_diverged"]), s, "CL.CL6.n_excluded_diverged")
    c7 = CL["CL7"]
    nmod = sum(c7[p][g]["models"] for p in ("F18", "X5") for g in c7[p])
    ntraj = sum(c7[p][g]["trajectories"] for p in ("F18", "X5") for g in c7[p])
    ndiv = sum(c7[p][g][f"{u}_diverged_{c}"] for p in ("F18", "X5") for g in c7[p]
               for u in ("models", "trajectories") for c in ("pol", "cl", "noise"))
    assert nmod == CL["n"]["F18"] + CL["n"]["X5"]
    put("s1_cl7_nmodels", str(nmod), s, "CL.CL7.*.*.models (sum)")
    put("s1_cl7_ntraj", f"{ntraj:,}".replace(",", "{,}"), s, "CL.CL7.*.*.trajectories (sum)")
    put("s1_cl7_ndiv", str(ndiv), s, "CL.CL7.*.*.{models,trajectories}_diverged_{pol,cl,noise} (sum)")
    TL = CL["exploratory_two_level"]
    assert TL["label"] == "exploratory"
    for q in ("CL1", "CL2"):
        put(f"s1_tl_{q.lower()}_ci", cib(*TL[q]["ci97.5"], f=r3), s, f"CL.exploratory_two_level.{q}.ci97.5")
    put("s1_tl_nclusters", str(TL["n_clusters"]), s, "CL.exploratory_two_level.n_clusters")
    R = CL["REL_reference"]
    for k in ("lam1_pol", "lam1_rec", "rJ_pol", "rJ_rec"):
        put(f"s1_rel_{k}", r2(R[k]), s, f"CL.REL_reference.{k}")
    put("s1_rel_ntraj", str(R["n_traj"]), s, "CL.REL_reference.n_traj")

    # ---- A9: study AR
    put("s1_ar_n", str(sum(AR["n"].values())), s, "AR.n (sum)")
    for a in ARCHES:
        put(f"s1_ar_n_{a}", str(AR["n"][a]), s, f"AR.n.{a}")
    put("s1_ar_nnew", str(AR["n_new_runs"]), s, "AR.n_new_runs")
    put("s1_ar_nold", str(sum(AR["n"].values()) - AR["n_new_runs"]), s, "AR.n (sum) - AR.n_new_runs")
    T = M["training"]
    assert len(T) == AR["n_new_runs"]
    assert all(v["weights_finite"] and not v["loss_ever_nonfinite"] and v["threads"] == 2 for v in T.values())
    its = {v["iterations"] for v in T.values()}
    assert len(its) == 1
    put("s1_ar_iters", f"{its.pop():,}".replace(",", "{,}"), m, "training.*.iterations")
    put("s1_ar_train_h", sig(sum(v["wall_clock_s"] for v in T.values()) / 3600, 2), m,
        "training.*.wall_clock_s (sum) / 3600")
    for q, base, check in (("AR1", "E1", "lam1"), ("AR2", "E8", "rJ")):
        A = AR[q]
        for i, a in enumerate(ARCHES):
            put(f"s1_art_{a}_{check}", r2(A["rho_new_per_group"][i]), s, f"AR.{q}.rho_new_per_group[{i}]")
            put(f"s1_art_{a}_{base}", r2(A["rho_base_per_group"][i]), s, f"AR.{q}.rho_base_per_group[{i}]")
        put(f"s1_art_mean_{check}", r2(A["rho_new_mean"]), s, f"AR.{q}.rho_new_mean")
        put(f"s1_art_mean_{base}", r2(A["rho_base_mean"]), s, f"AR.{q}.rho_base_mean")
        for p in ("pooled", "extended"):
            put(f"s1_art_{p}_{check}", r2(AR["AR3"][p][q]["rho_new"]), s, f"AR.AR3.{p}.{q}.rho_new")
            put(f"s1_art_{p}_{base}", r2(AR["AR3"][p][q]["rho_base"]), s, f"AR.AR3.{p}.{q}.rho_base")
            diffkeys(f"s1_ar3_{p}_{q.lower()}", AR["AR3"][p][q], f"AR.AR3.{p}.{q}")
        diffkeys(f"s1_{q.lower()}", A, f"AR.{q}", level="ci97.5", lab="verdict")
    put("s1_ar3_ext_n", str(AR["AR3"]["extended"]["n_models"]), s, "AR.AR3.extended.n_models")
    A4 = AR["AR4"]
    assert A4["label"] == "descriptive"
    for a in ARCHES:
        for k in ("n_pairs", "tf_better_E1", "tf_worse_E100", "both"):
            put(f"s1_ar4_{a}_{k}", str(A4[a][k]), s, f"AR.AR4.{a}.{k}")
    dv = AR["divergence_recorded_commands"]
    put("s1_ar_ndiv", str(sum(dv[a]["runs_diverged"] for a in ARCHES)), s,
        "AR.divergence_recorded_commands.*.runs_diverged (sum)")
    assert sum(dv[a]["runs_lam_nonfinite"] for a in ARCHES) == 0
    D = X["AR_diverged_recorded"]
    assert len(D) == sum(dv[a]["runs_diverged"] for a in ARCHES)
    assert {(v["arch"], v["regime"]) for v in D.values()} == {("mlp", "tf")}
    nt = {v["trajectories_diverged"] for v in D.values()}
    assert len(nt) == 1
    put("s1_ar_div_traj", str(nt.pop()), x, "AR_diverged_recorded.*.trajectories_diverged")
    put("s1_ar_div_seeds", ", ".join(str(v["seed"]) for v in sorted(D.values(), key=lambda v: v["seed"])), x,
        "AR_diverged_recorded.*.seed")
    put("s1_ar_ntraj", str(X["AR_n_trajectories_per_run"]), x, "AR_n_trajectories_per_run")

    # ---- A10: study TB (descriptive)
    assert B["label"].startswith("DESCRIPTIVE")
    a = B["a_still_improving"]
    for n in ("2500", "10000"):
        put(f"s1_tb_fall_{n}", str(a[f"arm_{n}"]["n_falling"]), b, f"a_still_improving.arm_{n}.n_falling")
        put(f"s1_tb_n_{n}", str(a[f"arm_{n}"]["n"]), b, f"a_still_improving.arm_{n}.n")
    for arm, tag in (("rollout_trained", "ro"), ("teacher_forced", "tf")):
        for n in ("2500", "10000"):
            for h in ("E1", "E100", "E368"):
                put(f"s1_tb_{tag}_{n}_{h}", dp(B["b_scores"][arm][n]["mean"][h], 3), b,
                    f"b_scores.{arm}.{n}.mean.{h}")
    cc = B["c_conclusions"]
    allseeds = all(cc[n][sd][k] for n in ("2500", "10000") for sd in ("0", "1", "2")
                   for k in ("tf_better_E1", "tf_worse_E100", "lam1_tf_positive", "lam1_rollout_negative"))
    assert allseeds
    put("s1_tb_nseeds", str(len([sd for sd in cc["10000"] if sd != "mean"])), b, "c_conclusions.{2500,10000}.{0,1,2} (all four statements true in each)")
    K = [v["K"] for v in B["d_ft_control"]["K_P_vs_CTRL"].values()]
    put("s1_tb_K_min", pct(min(K), 3), b, "d_ft_control.K_P_vs_CTRL.*.K (min)")
    put("s1_tb_K_max", pct(max(K), 3), b, "d_ft_control.K_P_vs_CTRL.*.K (max)")

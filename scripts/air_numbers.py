"""This study's own numbers for the paper (called by paper_numbers.py). Every value is read from results/."""
import json
import os

import common as C


def add(put, sig, sgn, pct, cib, dp, load, rel):
    S, s = load("stats"), rel("stats")
    M, m = load("models"), rel("models")
    F, f = load("figfacts"), rel("figfacts")
    ck, c = load("checks"), rel("checks")
    assert ck["all_pass"], "a pre-registered check failed"
    put("checks_all_pass", "all", c, "all_pass")

    # ---- counts
    f18, x5 = M["F18"], M["X5"]
    trained = {r["path"] for r in f18 + x5 if r["variant"] is None}
    put("n_trained", str(len(trained)), m, "distinct trained weight files in F18 and X5")
    nc = sum(r["variant"] is not None for r in x5)
    put("n_compressed", str(nc), m, "X5 entries with a variant (the F18 variants are among them)")
    assert {(r["path"], r["variant"]) for r in f18 if r["variant"]} <= {(r["path"], r["variant"]) for r in x5}
    put("n_sims", str(len(trained) + nc + 1), m, "trained + compressed + released")
    pools = {p: sum(r["pool"] == p for r in f18) for p in ("R", "T", "C")}
    put("n_P", str(len(f18)), m, "F18")
    put("n_R", str(pools["R"]), m, "F18 pool R")
    put("n_T", str(pools["T"]), m, "F18 pool T")
    put("n_C", str(pools["C"]), m, "F18 pool C")
    put("n_RC", str(pools["R"] + pools["C"]), m, "F18 pools R and C")
    put("n_X", str(len(x5)), m, "X5")
    put("n_X_fold", str(len(x5) // 5), m, "X5 per fold")
    put("n_nonfinite_F18", str(S["n_nonfinite"]["F18"]), s, "n_nonfinite.F18")
    put("n_nonfinite_X5", str(S["n_nonfinite"]["X5"]), s, "n_nonfinite.X5")

    r2 = lambda v: dp(v, 2, signed=True)                                                 # noqa: E731

    # ---- Q1, Q2 (primary)
    for q, base in (("Q1", "e1"), ("Q2", "e8")):
        v = S[q]
        put(f"{q.lower()}_rho_lam", r2(v["rho_new"]), s, f"{q}.rho_new")
        put(f"{q.lower()}_rho_{base}", r2(v["rho_base"]), s, f"{q}.rho_base")
        put(f"{q.lower()}_delta", r2(v["delta"]), s, f"{q}.delta")
        put(f"{q.lower()}_ci975", cib(*v["ci97.5"], f=r2), s, f"{q}.ci97.5")
        put(f"{q.lower()}_ci95", cib(*v["ci95"], f=r2), s, f"{q}.ci95")
        put(f"{q.lower()}_verdict", v["verdict"].lower(), s, f"{q}.verdict")
        put(f"{q.lower()}_tau_lam", r2(v["tau_new"]), s, f"{q}.tau_new")
        put(f"{q.lower()}_tau_{base}", r2(v["tau_base"]), s, f"{q}.tau_base")
        for tgt in ("E24", "E368"):
            w = S[f"{q}_{tgt}"]
            k = f"{q.lower()}_{tgt.lower()}"
            put(f"{k}_rho_lam", r2(w["rho_new"]), s, f"{q}_{tgt}.rho_new")
            put(f"{k}_rho_base", r2(w["rho_base"]), s, f"{q}_{tgt}.rho_base")
            put(f"{k}_delta", r2(w["delta"]), s, f"{q}_{tgt}.delta")
            put(f"{k}_ci", cib(*w["ci95"], f=r2), s, f"{q}_{tgt}.ci95")
            put(f"{k}_verdict", w["verdict"].lower(), s, f"{q}_{tgt}.verdict")

    # ---- Q3 (X5 within folds), Q4 (kinematic residual), Q5 (R and C only)
    for key, q in (("q3_1", "Q3_vs_E1"), ("q3_8", "Q3_vs_E8")):
        v = S[q]
        put(f"{key}_rho_new", r2(v["rho_new_mean"]), s, f"{q}.rho_new_mean")
        put(f"{key}_rho_base", r2(v["rho_base_mean"]), s, f"{q}.rho_base_mean")
        put(f"{key}_delta", r2(v["delta"]), s, f"{q}.delta")
        put(f"{key}_ci", cib(*v["ci95"], f=r2), s, f"{q}.ci95")
        put(f"{key}_verdict", v["verdict"].lower(), s, f"{q}.verdict")
    for key, q in (("q4_1", "Q4_vs_E1"), ("q4_8", "Q4_vs_E8"), ("q5_1", "Q5_vs_E1"), ("q5_8", "Q5_vs_E8")):
        v = S[q]
        put(f"{key}_rho_new", r2(v["rho_new"]), s, f"{q}.rho_new")
        put(f"{key}_rho_base", r2(v["rho_base"]), s, f"{q}.rho_base")
        put(f"{key}_delta", r2(v["delta"]), s, f"{q}.delta")
        put(f"{key}_ci", cib(*v["ci95"], f=r2), s, f"{q}.ci95")
        put(f"{key}_verdict", v["verdict"].lower(), s, f"{q}.verdict")

    # ---- Q6 and per-variant stability shifts (X5, 25 parents)
    v = S["Q6"]
    put("q6_mean", r2(v["mean"]), s, "Q6.mean")
    put("q6_ci", cib(*v["ci95"], f=r2), s, "Q6.ci95")
    put("q6_verdict", v["verdict"].lower(), s, "Q6.verdict")
    put("q6_npos", str(v["n_positive_lam1"]), s, "Q6.n_positive_lam1")
    put("q6_nparents", str(v["n_parents"]), s, "Q6.n_parents")
    put("parent_lam", r2(S["parent_mean_lam1_X5"]), s, "parent_mean_lam1_X5")
    for var, w in S["lam_shift_X5"].items():
        put(f"lamshift_{var}", r2(w["mean"]), s, f"lam_shift_X5.{var}.mean")
        put(f"lamshift_{var}_ci", cib(*w["ci95"], f=r2), s, f"lam_shift_X5.{var}.ci95")
        put(f"lam_{var}", r2(w["mean_lam1"]), s, f"lam_shift_X5.{var}.mean_lam1")
        put(f"lampos_{var}", str(w["n_positive_lam1"]), s, f"lam_shift_X5.{var}.n_positive_lam1")

    # ---- D1: training horizon (F18 groups) and training methods
    g = S["D1_groups_F18"]
    names = {1: "sweep M32 N1", 2: "sweep M32 N2", 8: "A (M32,N8)", 16: "sweep M32 N16", 32: "sweep M32 N32"}
    for n, gname in names.items():
        for k in ("E1", "E8", "E24", "E100", "E368"):
            put(f"N{n}_{k}", sig(g[gname][k], 2), s, f"D1_groups_F18.{gname}.{k}")
        put(f"N{n}_lam", r2(g[gname]["lam1"]), s, f"D1_groups_F18.{gname}.lam1")
    for k in ("E1", "E8", "E24", "E100", "E368"):
        put(f"N1overN8_{k}", sig(g[names[1]][k] / g[names[8]][k], 2), s,
            f"D1_groups_F18.{names[1]}.{k} / D1_groups_F18.{names[8]}.{k}")
    for gname, k in (("B (teacher forced)", "B"), ("A (M32,N8)", "A")):
        for q in ("E1", "E100", "E368"):
            put(f"g{k}_{q}", sig(g[gname][q], 2), s, f"D1_groups_F18.{gname}.{q}")
        put(f"g{k}_lam", r2(g[gname]["lam1"]), s, f"D1_groups_F18.{gname}.lam1")
    put("floor_F18_E100", sig(S["floor_F18"]["E100"], 2), s, "floor_F18.E100")
    put("floor_F18_E1", sig(S["floor_F18"]["E1"], 2), s, "floor_F18.E1")
    for kk, v in S["D1_rho_F18"].items():
        a, b = kk.split("~")
        put(f"rho_{a}_{b}", r2(v["rho"]), s, f"D1_rho_F18.{kk}.rho")
        put(f"rho_{a}_{b}_ci", cib(*v["ci95"], f=r2), s, f"D1_rho_F18.{kk}.ci95")
    for kk, v in S["D1_rho_RC"].items():
        a = kk.split("~")[0]
        put(f"rhoRC_{a}", r2(v["rho"]), s, f"D1_rho_RC.{kk}.rho")

    # ---- D2: skill retained by the compressed variants (X5)
    def fk(x):
        """skill retained: one decimal from 95% up (where whole per cents hide the loss), else whole per cents"""
        v = 100 * x
        if abs(v) >= 99.95:
            return "100\\%"
        return (f"{v:.1f}" if abs(v) >= 95 else f"{v:.0f}").replace("-", "$-$") + r"\%"
    for var, d in S["D2_skill_retained_X5"].items():
        for h, w in d.items():
            put(f"K_{var}_{h}", fk(w["K"]), s, f"D2_skill_retained_X5.{var}.{h}.K")
            put(f"K_{var}_{h}_ci", cib(*w["ci95"], f=fk), s, f"D2_skill_retained_X5.{var}.{h}.ci95")

    # ---- D3: rollout training against teacher forcing on X5, normalised and physical units
    d3 = S["D3_arms_X5"]
    for h in ("h1", "h8", "h24", "h100", "h368"):
        w = d3[h]
        put(f"ab_{h}_A", sig(w["A"], 2), s, f"D3_arms_X5.{h}.A")
        put(f"ab_{h}_B", sig(w["B"], 2), s, f"D3_arms_X5.{h}.B")
        put(f"ab_{h}_floor", sig(w["floor"], 2), s, f"D3_arms_X5.{h}.floor")
        put(f"ab_{h}_diff", sgn(w["B_minus_A"], 2), s, f"D3_arms_X5.{h}.B_minus_A")
        put(f"ab_{h}_ci", cib(*w["ci95"], f=lambda x: sgn(x, 2)), s, f"D3_arms_X5.{h}.ci95")
        put(f"ab_{h}_ratio", sig(w["B_over_A"], 2), s, f"D3_arms_X5.{h}.B_over_A")
        put(f"ab_{h}_nworse", str(w["n_traj_B_worse"]), s, f"D3_arms_X5.{h}.n_traj_B_worse")
    ph = d3["physical"]
    for k, w in ph.items():
        if k == "dist_recorded":
            for h, v in w.items():
                put(f"dist_{h}", sig(v, 2), s, f"D3_arms_X5.physical.dist_recorded.{h}")
            continue
        for arm in ("A", "B", "A5", "floor"):
            put(f"ph_{k}_{arm}", sig(w[arm], 2), s, f"D3_arms_X5.physical.{k}.{arm}")
        put(f"ph_{k}_ci", cib(*w["B_minus_A_ci95"], f=lambda x: sgn(x, 2)), s, f"D3_arms_X5.physical.{k}.B_minus_A_ci95")
        put(f"ph_{k}_nworse", str(w["n_traj_B_worse"]), s, f"D3_arms_X5.physical.{k}.n_traj_B_worse")
    for k in ("A", "A5", "B"):
        put(f"x5_lam_{k}", r2(d3["lam1"][k]), s, f"D3_arms_X5.lam1.{k}")
        put(f"x5_lampos_{k}", sig(100 * d3["lam1_positive_share"][k], 2) + r"\%", s,
            f"D3_arms_X5.lam1_positive_share.{k}")

    # ---- D4: worse than the floor at 100 steps, by the sign of lambda_1
    for arena in ("F18", "X5"):
        w = S[f"D4_{arena}"]
        put(f"d4_{arena}_pos_worse", str(w["lam_pos"][0]), s, f"D4_{arena}.lam_pos[0]")
        put(f"d4_{arena}_pos_n", str(w["lam_pos"][1]), s, f"D4_{arena}.lam_pos[1]")
        put(f"d4_{arena}_non_worse", str(w["lam_nonpos"][0]), s, f"D4_{arena}.lam_nonpos[0]")
        put(f"d4_{arena}_non_n", str(w["lam_nonpos"][1]), s, f"D4_{arena}.lam_nonpos[1]")

    # ---- D5: the released checkpoint (its 20 trajectories; training rows unrecorded)
    d5 = S["D5_released"]
    for k in ("E1", "E100", "E368", "rJ"):
        put(f"rel_{k}", sig(d5[k], 2), s, f"D5_released.{k}")
    put("rel_lam", r2(d5["lam1"]), s, "D5_released.lam1")
    put("rel_floor_E100", sig(d5["floor"]["E100"], 2), s, "D5_released.floor.E100")
    put("rel_pos_h368", sig(d5["pos_h368"], 2), s, "D5_released.pos_h368")

    # ---- D6: choosing a configuration
    d6 = S["D6_config_choice_F18"]
    for k, w in d6["regret"].items():
        put(f"regret_{k}", sig(w["mean"], 2) if w["mean"] != 0 else "0", s, f"D6_config_choice_F18.regret.{k}.mean")
    put("best_E100_cfg", sig(d6["best_E100_mean"], 2), s, "D6_config_choice_F18.best_E100_mean")
    picks = d6["picks"]
    put("n_pick_E1_N1", str(sum(picks[f"E1_s{i}"].startswith("mn_M32_N1_") for i in range(3))), s,
        "D6_config_choice_F18.picks.E1_s*")

    # ---- derived values used in prose (each from the same fields)
    put("ab_h1_pct_better", sig(100 * (1 - d3["h1"]["B_over_A"]), 2) + r"\%", s, "100*(1 - D3_arms_X5.h1.B_over_A)")
    w = S["D4_F18"]
    put("d4_F18_worse_total", str(w["lam_pos"][0] + w["lam_nonpos"][0]), s, "D4_F18 lam_pos[0] + lam_nonpos[0]")
    best = d6["best_E100_mean"]
    put("regret_E1_ratio", sig((best + d6["regret"]["E1"]["mean"]) / best, 2), s,
        "(D6.best_E100_mean + D6.regret.E1.mean) / D6.best_E100_mean")
    put("n_pick_E1_onestep", str(sum(picks[f"E1_s{i}"].startswith(("mn_M32_N1_", "mn_M32_N2_")) for i in range(3))), s,
        "D6_config_choice_F18.picks.E1_s* that are N=1 or N=2 models")
    put("n_pick_lam_best", str(sum(v == 0 for v in d6["regret"]["lam1"]["per_seed"])), s,
        "D6_config_choice_F18.regret.lam1.per_seed == 0")
    put("released_floor_E368", sig(d5["floor"]["E368"], 2), s, "D5_released.floor.E368")
    put("released_floor_E1", sig(d5["floor"]["E1"], 2), s, "D5_released.floor.E1")
    put("released_lam_pos_traj", sig(100 * d5["lam1_positive_share_traj"], 2) + r"\%", s,
        "D5_released.lam1_positive_share_traj")

    # ---- exploratory (results/exploratory.json; labelled exploratory in the paper)
    X, x = load("explo"), rel("explo")
    for k in ("E1", "E8", "lam1", "rJ"):
        put(f"x2_tc_{k}", r2(X["X2_training_choices"][k]["rho"]), x, f"X2_training_choices.{k}.rho")
        put(f"x2_tc_{k}_ci", cib(*X["X2_training_choices"][k]["ci95"], f=r2), x, f"X2_training_choices.{k}.ci95")
        put(f"x2_c_{k}", r2(X["X2_compressed"][k]["rho"]), x, f"X2_compressed.{k}.rho")
        put(f"x3_unc_{k}", r2(X["X3_uncompressed"][k]), x, f"X3_uncompressed.{k}")
        put(f"x3_c_{k}", r2(X["X3_compressed"][k]), x, f"X3_compressed.{k}")
    put("x2_tc_n", str(X["X2_training_choices"]["n"]), x, "X2_training_choices.n")
    put("x2_c_n", str(X["X2_compressed"]["n"]), x, "X2_compressed.n")
    put("x1_rJ", r2(X["X1_rJ_vs_E1"]["rho_new_mean"]), x, "X1_rJ_vs_E1.rho_new_mean")
    put("x1_rJ_minus_E1", r2(X["X1_rJ_vs_E1"]["delta"]), x, "X1_rJ_vs_E1.delta")
    put("x1_rJ_minus_E1_ci", cib(*X["X1_rJ_vs_E1"]["ci95"], f=r2), x, "X1_rJ_vs_E1.ci95")
    put("x1_rJ_minus_E8", r2(X["X1_rJ_vs_E8"]["delta"]), x, "X1_rJ_vs_E8.delta")
    put("x1_rJ_minus_E8_ci", cib(*X["X1_rJ_vs_E8"]["ci95"], f=r2), x, "X1_rJ_vs_E8.ci95")

    # ---- figure facts
    put("fig_B_cross", str(F["B_cross_floor_step"]), f, "B_cross_floor_step")
    put("fig_U_cross", str(F["U_cross_floor_step"]), f, "U_cross_floor_step")

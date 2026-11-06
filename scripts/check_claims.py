"""
The paper's words, checked against the numbers. Every qualitative statement the text makes about a result (an
order, a sign, "worse than predicting no change", "grows with the horizon") is an assertion here, read from the same
files the numbers come from. A failing assertion means the sentence is wrong, whatever the number beside it says.
-> results/claims_check.json; exits non-zero on a failure.
"""
import datetime
import json
import math
import os
import subprocess
import sys

import common as C

S = json.load(open(os.path.join(C.RESULTS, "stats.json")))
F = json.load(open(os.path.join(C.RESULTS, "figure_facts.json")))
TM = json.load(open(os.path.join(C.PDM, "rwm_repro", "results", "paper_numbers.json")))
NG = json.load(open(os.path.join(C.PDM, "NGCAPS", "results", "main.json")))
D1, D2, D3 = S["D1_groups_F18"], S["D2_skill_retained_X5"], S["D3_arms_X5"]
K = lambda v, h: D2[v][f"h{h}"]["K"]                                                       # noqa: E731
tmf = lambda k: float(str(TM[k]["value"]).replace(",", "").replace("+", ""))               # noqa: E731
N = {1: "sweep M32 N1", 2: "sweep M32 N2", 8: "A (M32,N8)", 16: "sweep M32 N16", 32: "sweep M32 N32"}

CLAIMS = {
    # Training objective (Sec. 4.1, Fig. 1)
    "TF wins one step ahead on short windows (TMLR, interval below 0)":
        tmf("x3_h1_gap") < 0 and float(TM["x3_h1_ci"]["value"].strip("[]").split(",")[1]) < 0,
    "TF error ratio grows with the horizon from 8 steps on":
        D3["h8"]["B_over_A"] < D3["h24"]["B_over_A"] < D3["h100"]["B_over_A"] < D3["h368"]["B_over_A"],
    "TF is worse than rollout training from 24 steps on":
        all(D3[h]["B_over_A"] > 1 for h in ("h24", "h100", "h368")),
    "TF crosses the floor in Fig. 1": F["B_cross_floor_step"] is not None,
    "TF position error exceeds rollout training's at 2 s and 7.4 s":
        D3["physical"]["pos_h100"]["B"] > D3["physical"]["pos_h100"]["A"]
        and D3["physical"]["pos_h368"]["B"] > D3["physical"]["pos_h368"]["A"],
    "TF joint-angle error exceeds rollout training's over 2 s":
        D3["physical"]["jdeg_h100"]["B"] > D3["physical"]["jdeg_h100"]["A"],
    # Training horizon
    "N=1 worse than N=8 at 100 steps": D1[N[1]]["E100"] > D1[N[8]]["E100"],
    "lambda_1 falls monotonically with N":
        all(D1[N[a]]["lam1"] > D1[N[b]]["lam1"] for a, b in ((1, 2), (2, 8), (8, 16), (16, 32))),
    "lambda_1 sign changes between N=2 and N=8": D1[N[2]]["lam1"] > 0 > D1[N[8]]["lam1"],
    "lambda_1 positive at N=1": D1[N[1]]["lam1"] > 0,
    "more teacher-forced than rollout-trained models expand (X5)":
        D3["lam1_positive_share"]["B"] > D3["lam1_positive_share"]["A"],
    "TF training path: lambda rises, error falls (NGCAPS)":
        NG["descriptive"]["training_paths"]["B_10000"]["lam1"] > NG["descriptive"]["training_paths"]["B_500"]["lam1"]
        and NG["descriptive"]["training_paths"]["B_10000"]["E368"] < NG["descriptive"]["training_paths"]["B_2500"]["E368"],
    "rollout training path: lambda falls (NGCAPS)":
        NG["descriptive"]["training_paths"]["A_10000"]["lam1"] < NG["descriptive"]["training_paths"]["A_500"]["lam1"],
    # Compression
    "30% pruning: K at 368 negative (worse than no change)": K("u30", 368) < 0,
    "30% pruning: K falls with the horizon": K("u30", 1) > K("u30", 100) > K("u30", 368),
    "width 192 without fine-tuning: K below one half at every horizon": all(K("s192", h) < 0.5 for h in (1, 8, 24, 100, 368)),
    # Uncertainty
    "spread factor grows with the horizon": tmf("n8_k_rel_epi_h1") < tmf("n8_k_rel_epi_h100") < tmf("n8_k_rel_epi_h368"),
    "stale command: one-step increase resolved, 100-step not":
        float(TM["adh20_rel_ci_h1"]["value"].strip("[]").split(",")[0]) > 0
        and float(TM["adh20_rel_ci_h100"]["value"].strip("[]").split(",")[0]) < 0
        < float(TM["adh20_rel_ci_h100"]["value"].strip("[]").split(",")[1]),
}

T = json.load(open(os.path.join(C.RESULTS, "model_table.json")))
f18 = {r["key"]: r for r in T["F18"]}
X = json.load(open(os.path.join(C.RESULTS, "exploratory.json")))
ICP_D = json.load(open(os.path.join(C.PDM, "ICPAIAS", "results", "r2_detect.json")))["component_auroc"]["A-ens"]
ICP_R = json.load(open(os.path.join(C.PDM, "ICPAIAS", "results", "r2_repair.json")))["A5"]["ens5"]
cfg = lambda s_: [f"mn_M{M}_N{N_}_s{s_}" for M, N_ in ((1, 8), (2, 8), (8, 8), (16, 8), (32, 2), (32, 16), (32, 32),
                                                       (32, 1))] + [f"A_s{s_}"]          # noqa: E731
onestep = [k for k, r in f18.items() if r["pool"] == "T" or k.startswith("mn_M32_N2_")]
rollout_best = [k for k, r in f18.items() if r["pool"] == "R" and not k.startswith("mn_M32_N2_")]
CLAIMS.update({
    "TF 10% more accurate one step ahead, interval below 0 (X5)": D3["h1"]["B_over_A"] < 1 and D3["h1"]["ci95"][1] < 0,
    "by 8 steps TF is worse on most trajectories": D3["h8"]["B_over_A"] > 1 and D3["h8"]["n_traj_B_worse"] > 10,
    "position over 2 s: neither beats constant velocity":
        D3["physical"]["pos_h100"]["A"] > D3["physical"]["pos_h100"]["floor"]
        and D3["physical"]["pos_h100"]["B"] > D3["physical"]["pos_h100"]["floor"],
    "position at 7.4 s: rollout beats constant velocity, TF does not":
        D3["physical"]["pos_h368"]["A"] < D3["physical"]["pos_h368"]["floor"] < D3["physical"]["pos_h368"]["B"],
    "every TF model expands, no rollout-trained model does (X5)":
        D3["lam1_positive_share"]["B"] == 1.0 and D3["lam1_positive_share"]["A"] == 0.0
        and D3["lam1_positive_share"]["A5"] == 0.0,
    "N=1 more accurate than N=8 one step ahead": D1[N[1]]["E1"] < D1[N[8]]["E1"],
    "N=32 highest one-step error and lowest 100-step error of the nine configurations (seed means)":
        all(D1[N[32]]["E1"] >= D1[g]["E1"] for g in D1 if g.startswith("sweep") or g == N[8])
        and all(D1[N[32]]["E100"] <= D1[g]["E100"] for g in D1 if g.startswith("sweep") or g == N[8]),
    "none of the 30%-pruned copies expands (X5)": S["lam_shift_X5"]["u30"]["n_positive_lam1"] == 0,
    "Q6 not more expansive": S["Q6"]["verdict"] != "MORE EXPANSIVE",
    "the same held at 368 steps": S["Q1_E368"]["verdict"] == S["Q1"]["verdict"] and S["Q2_E368"]["verdict"] == S["Q2"]["verdict"],
    "at 24 steps the 8-step error ranked ahead of the growth rate": S["Q2_E24"]["verdict"] == "TRAINING-HORIZON BETTER",
    "without one-step-trained models lambda still beats E1": S["Q5_vs_E1"]["verdict"] == "LABEL-FREE BETTER",
    "r_J best of all on F18 and ahead of E8":
        S["Q4_vs_E8"]["verdict"] == "LABEL-FREE BETTER"
        and S["Q4_vs_E1"]["rho_new"] > max(S["Q1"]["rho_new"], S["Q1"]["rho_base"], S["Q2"]["rho_base"]),
    "one-step error ranks training choices backwards (interval below 0; X5 negative too)":
        X["X2_training_choices"]["E1"]["ci95"][1] < 0 and X["X3_uncompressed"]["E1"] < 0,
    "every check ranks compressed copies above 0.8 (F18)":
        all(X["X2_compressed"][c]["rho"] > 0.8 for c in ("E1", "E8", "lam1", "rJ")),
    "replication did not hold for lambda (X5)": S["Q3_vs_E1"]["verdict"] != "LABEL-FREE BETTER",
    "on X5 the 8-step error ranked better still than E1": S["Q3_vs_E8"]["rho_base_mean"] > S["Q3_vs_E1"]["rho_base_mean"],
    "released: low one-step error, positive growth rate, worse than floor at 368":
        S["D5_released"]["E1"] < S["D5_released"]["floor"]["E1"] and S["D5_released"]["lam1"] > 0
        and S["D5_released"]["E368"] > S["D5_released"]["floor"]["E368"],
    "r_J level with E8 and best on X5 (exploratory)":
        X["X1_rJ_vs_E8"]["ci95"][0] < 0 < X["X1_rJ_vs_E8"]["ci95"][1]
        and X["X1_rJ_vs_E1"]["rho_new_mean"] >= max(S["Q3_vs_E8"]["rho_base_mean"], S["Q3_vs_E1"]["rho_new_mean"]),
    "run-time: no label-free AUROC above 0.65 on A-ens; growth signal worse than counter":
        max(ICP_D["P_on"]["auroc"], ICP_D["F(D)"]["auroc"]) < 0.65 and NG["Q3"]["verdict"] == "WORSE THAN COUNTER",
    "exact gravity repair helps (interval below 1); approximate joint repair hurts":
        ICP_R["err|G|368"]["ci"][1] < 1 and ICP_R["err|J|368"]["ratio"] > 1,
})

# Added in revision round R8: sentences of the revised text (Sec. 3-5, appendix). Exploratory results are read from
# results/revision_exploratory.json (labelled exploratory in the paper).
RX = json.load(open(os.path.join(C.RESULTS, "revision_exploratory.json")))
PC = json.load(open(os.path.join(C.PDM, "PerconAI", "results", "paper_numbers.json")))["keys"]
pc = lambda k: PC[k]["value"]                                                              # noqa: E731
ST = json.load(open(os.path.join(C.PDM, "rwm_repro", "results", "n5_stale_armA_ci.json")))
ho = ST["results"]["released"]["held_out_n4"]["by_horizon"]
LS = S["lam_shift_X5"]
ICP_V = json.load(open(os.path.join(C.PDM, "ICPAIAS", "results", "r2_verdicts.json")))
A1, A1d, A2a = RX["A1_horizon_matched"], RX["A1_horizon_matched"]["diffs_vs_E100"], RX["A2a_two_level"]["registered_tests"]
holds = {k: v["two_level_excludes_zero"] == (v["registered_verdict"] != "NOT SETTLED") for k, v in A2a.items()}
mid = ("u10", "u20", "u30")
CLAIMS.update({
    "30%-pruned copy crosses the floor in Fig. 1; starts within 10% of its parent":
        F["U_cross_floor_step"] is not None and F["U_k1"] < 1.1 * F["A_k1"],
    "abstract: TF and N=1 both worse than N=8 at 2 s": D3["h100"]["B_over_A"] > 1 and D1[N[1]]["E100"] > D1[N[8]]["E100"],
    "stale command: one-step effect larger than 100-step; held-out pair resolves it at 100 steps":
        tmf("adh20_rel_h1") > tmf("adh20_rel_h100") and ho["100"]["ci95_pct"][0] > 0,
    "rollout training better on all held-out episodes at 368 steps (sign test)":
        tmf("x3_eps_pos") == tmf("x3_n_eps") and float(TM["n7_x3_ep_p_h368"]["value"]) < 0.01,
    "growth rate falls with N: registered rank correlation negative, interval below 0":
        NG["Q4"]["rho"] < 0 and NG["Q4"]["ci"][1] < 0,
    "every F18 model trained on own rollouts of 8+ steps contracts; no teacher-forced one does":
        all(f18[k]["lam1"] < 0 for k in rollout_best) and all(r["lam1"] > 0 for r in f18.values() if r["pool"] == "T"),
    "no copy pruned by 10-30% has lambda_1 > 0": all(LS[u]["n_positive_lam1"] == 0 for u in mid),
    "30% pruning: shift unresolved; 10 and 20%: resolved, smaller":
        LS["u30"]["ci95"][0] < 0 < LS["u30"]["ci95"][1] and LS["u10"]["ci95"][0] > 0 and LS["u20"]["ci95"][0] > 0
        and max(LS["u10"]["mean"], LS["u20"]["mean"]) < LS["u30"]["mean"],
    "moderate pruning adds error (K < 1 at 100 steps) while barely moving lambda_1 (< a quarter of the 50% shift)":
        all(K(u, 100) < 1 for u in mid) and max(LS[u]["mean"] for u in mid) < 0.25 * LS["u50"]["mean"],
    "heavier pruning and unit removal raise lambda_1 (interval above 0); some width-192 and 128 copies unstable":
        all(LS[v]["ci95"][0] > 0 for v in ("u50", "s192", "s128")) and LS["s192"]["n_positive_lam1"] > 0
        and LS["s128"]["n_positive_lam1"] > 0,
    "30% pruning has lost most skill by 2 s (K < 0.5 at 100 steps)": K("u30", 100) < 0.5,
    "fine-tuned: width 192 keeps most at 368 (interval above 0.5); width 128 not settled (interval spans 0.5)":
        pc("r3.pool.q3.w192.h368.K.lo") > 0.5 and pc("r3.pool.q3.w128.h368.K.lo") < 0.5 < pc("r3.pool.q3.w128.h368.K.hi"),
    "scratch better at width 128, not separated at 192 (pooled)":
        pc("r3.pool.q4.w128.h368.label") == "SCRATCH BETTER" and pc("r3.pool.q4.w192.h368.label") == "NOT SETTLED",
    "8-bit dynamic quantisation keeps most of the 368-step skill": pc("e2.Q1_int8_dynamic.K.lo") > 0.5,
    "synthetic data: released head far below the noise; likelihood loss recovers it":
        tmf("e5s_mse_under") > 5 and abs(tmf("e5s_nll_ratio") - 1) < 0.1,
    "per-horizon multiplier misses the +-2 sigma tails at 100 steps": tmf("n9_epi_perh_cov2_h100") < tmf("v3_cov_nominal2"),
    "primary 1: growth rate beats one-step error; primary 2: not separated from the 8-step error":
        S["Q1"]["verdict"] == "LABEL-FREE BETTER" and S["Q2"]["verdict"] == "NOT SETTLED"
        and S["Q1"]["rho_base"] < S["Q1"]["rho_new"],
    "5-split pool: one-step error ranks better than the growth rate (interval below 0)":
        S["Q3_vs_E1"]["verdict"] == "ONE-STEP BETTER" and S["Q3_vs_E1"]["ci95"][1] < 0,
    "r_J over E8 on F18: favoured by the registered rule, not after Holm":
        S["Q4_vs_E8"]["verdict"] == "LABEL-FREE BETTER" and not RX["A2c_holm"]["tests"]["Q4_vs_E8"]["rejects_zero"],
    "Holm: exactly Q3_vs_E8 (8-step over growth rate, X5) and Q4_vs_E1 (r_J over E1, F18) survive":
        set(RX["A2c_holm"]["survive"]) == {"Q3_vs_E8", "Q4_vs_E1"} and len(RX["A2c_holm"]["family"]) == 7,
    "wider intervals: exactly Q1_E24, Q5_vs_E1 and Q3_vs_E1 lose their reading (3 of 13)":
        {k for k, h in holds.items() if not h} == {"Q1_E24", "Q5_vs_E1", "Q3_vs_E1"} and len(holds) == 13,
    "r_J over 100 steps alone still ranks well (rho > 0.9, F18 and X5; exploratory)":
        A1["F18"]["all"]["E100"]["rJ_100"]["rho"] > 0.9 and A1["X5"]["E100"]["rJ_100"]["rho"] > 0.9,
    "r_J over 8 steps: level with E8 on F18, below it on X5 (exploratory)":
        A1d["rJ_8_minus_E8"]["F18_all"]["ci95"][0] < 0 < A1d["rJ_8_minus_E8"]["F18_all"]["ci95"][1]
        and A1d["rJ_8_minus_E8"]["X5"]["ci95"][1] < 0,
    "lambda_1 over 8 steps ranks far below its 100-step reading (F18 and X5; exploratory)":
        A1["F18"]["all"]["E100"]["lam1_8"]["rho"] < A1["F18"]["all"]["E100"]["lam1_100"]["rho"] - 0.3
        and A1["X5"]["E100"]["lam1_8"]["rho"] < A1["X5"]["E100"]["lam1_100"]["rho"] - 0.3,
    "X5 without split {1,8}: growth rate still does not beat one-step error (exploratory)":
        RX["A2b_X5_without_split18"]["diffs"]["Q3_vs_E1"]["ci95"][0] <= 0,
    "X5 without split {1,8}: r_J not behind E8, at most slightly ahead (exploratory)":
        RX["A2b_X5_without_split18"]["diffs"]["rJ_vs_E8"]["ci95"][0] >= 0
        and RX["A2b_X5_without_split18"]["diffs"]["rJ_vs_E8"]["ci95"][1] < 0.1,
    "most F18 simulators worse than holding the last state have lambda_1 <= 0":
        S["D4_F18"]["lam_nonpos"][0] > S["D4_F18"]["lam_pos"][0] > 0,
    "re-synchronising on a score adds error versus a fixed schedule (interval above 0)":
        ICP_V["families"]["use"]["tests"]["H7"]["ci"][0] > 0,
})

# Study FT (registered, prereg/PREREG_FT.md): Sec. 4.1, Table 1, abstract, Discussion (2), Limitations, Conclusion, A7
FT = json.load(open(os.path.join(C.RESULTS, "ft_stats.json")))
FTM = json.load(open(os.path.join(C.RESULTS, "ft_models.json")))
FTP = json.load(open(os.path.join(C.RESULTS, "ft_probe.json")))
FTp, FTs, FTd = FT["primary"], FT["secondary"], FT["descriptive"]
CLAIMS.update({
    "FT1: fine-tuning restored the 100-step skill (registered; 97.5% interval above 0)":
        FTp["FT1"]["verdict"] == "FINE-TUNING RESTORES SKILL" and FTp["FT1"]["ci97.5"][0] > 0
        and FTp["FT1"]["K_orig_ft"] > FTp["FT1"]["K_orig_noft"],
    "FT2: fine-tuned 30%-pruned copies nearly as skilled as the equally fine-tuned control (lower end above 0.90)":
        FTp["FT2"]["verdict"] == "NEARLY AS SKILLED" and FTp["FT2"]["ci97.5"][0] > 0.90,
    "Table 1: fine-tuned row is above 100% of the parent at every horizon shown":
        all(FTd[f"K_U30F_vs_P_h{h}"]["K"] > 1 for h in (1, 24, 100, 368)),
    "A7: secondary readings at 24 and 368 steps and at 50% agree with the primary verdicts":
        all(FTs[q]["verdict"] == "FINE-TUNING RESTORES SKILL" for q in ("FT1_h24", "FT1_h368", "FT1_u50_h100"))
        and all(FTs[q]["verdict"] == "NEARLY AS SKILLED" for q in ("FT2_h24", "FT2_h368", "FT2_u50_h100")),
    "A7: K above 100% mostly reflects fine-tuning: the control alone is above 100% and level with U30F at 100 steps":
        FTd["K_CTRL_vs_P_h100"]["ci95"][0] > 1 and FTd["K_CTRL_vs_P_h100"]["K"] >= FTd["K_U30F_vs_P_h100"]["K"]
        and FTd["K_U30F_vs_CTRL_h100"]["ci95"][0] < 1 < FTd["K_U30F_vs_CTRL_h100"]["ci95"][1],
    "A7: one-step reading of the fine-tuned copy not separated from the 100-step one (secondary)":
        FTs["onestep_U30F"]["verdict"] == "NOT SETTLED",
    "A7: unretrained pruning raised r_J, growth rate not resolved; after fine-tuning both below the parent's":
        FTs["change_rJ_U30_minus_P"]["verdict"] == "INCREASES" and FTs["change_lam1_U30_minus_P"]["verdict"] == "NOT SETTLED"
        and FTs["change_rJ_U30F_minus_P"]["verdict"] == "DECREASES"
        and FTs["change_lam1_U30F_minus_P"]["verdict"] == "DECREASES",
    "A7: all four checks passed for every run; no arm excluded; no undefined draw":
        all(all((x["passed"] if isinstance(x, dict) else x) is True for x in FTM["checks"][c].values())
            for c in ("C-FT1", "C-FT2", "C-FT3", "C-FT4"))
        and not FT["excluded_arms"]
        and all(v.get("undefined_draws", 0) == 0 for b in (FTp, FTs) for v in b.values()),
    "A7: probe projected the study within the cap, so the 50% arm ran":
        FTP["projected_hours"]["total_with_u50"] <= FTP["cap_hours"] and FTP["run_u50f"] is True
        and "U50F" in FT["arms"],
    "Limitations: only the 30% and 50% pruned copies were fine-tuned in this study":
        {round(json.load(open(f))["level"], 2) for f in
         [os.path.join(C.RESULTS, "raw", "ft", n) for n in os.listdir(os.path.join(C.RESULTS, "raw", "ft"))]}
        == {0.3, 0.5},
    "Sec. 4.1: unretrained 30% copies of the FT parents had lost most 100-step skill (below one half)":
        FTp["FT1"]["K_orig_noft"] < 0.5,
})

# R10: sentences added or reworded after the fresh-eyes review.
_short = [k for k in f18 if k.startswith(("mn_M32_N1_", "mn_M32_N2_")) or f18[k]["pool"] == "T"]
CLAIMS.update({
    "bullet 2: lambda_1 > 0 for every model trained one or two steps ahead; < 0 for every other rollout-trained model (F18)":
        all(f18[k]["lam1"] > 0 for k in _short)
        and all(r["lam1"] < 0 for k, r in f18.items() if r["pool"] == "R" and k not in _short),
    "A5: every copy with lambda_1 > 0 is 50%-pruned or width-reduced":
        sum(LS[v]["n_positive_lam1"] for v in ("u10", "u20", "u30", "u50", "s192", "s128"))
        == sum(LS[v]["n_positive_lam1"] for v in ("u50", "s192", "s128")) > 0,
    "Sec. 4.1: 10% and 20% shifts similar to 30%, intervals exclude zero (exploratory)":
        all(LS[u]["ci95"][0] > 0 and abs(LS[u]["mean"] - LS["u30"]["mean"]) < 0.02 for u in ("u10", "u20")),
    "Sec. 4.2/Discussion: released model positive lambda but 100-step error below the floor":
        S["D5_released"]["lam1"] > 0 and S["D5_released"]["E100"] < S["D5_released"]["floor"]["E100"],
})

# S6 (Step 1): studies CL and AR (registered, prereg/PREREG_S1.md) and TB (descriptive): body and appendix A8-A10.
S1 = json.load(open(os.path.join(C.RESULTS, "s1_stats.json")))
S1M = json.load(open(os.path.join(C.RESULTS, "s1_models.json")))
S1X = json.load(open(os.path.join(C.RESULTS, "s1_extra.json")))
S1B = json.load(open(os.path.join(C.RESULTS, "s1_budget.json")))
_cl, _ar, _mg = S1["CL"], S1["AR"], S1["margin"]
_lab = lambda r: r.get("verdict", r.get("label"))                                          # noqa: E731
_cl7 = [g for p in ("F18", "X5") for g in _cl["CL7"][p].values()]
_arm = S1M["models"]["AR"]
_old = [v for v in _arm.values() if not v["new_run"]]
_new = [v for v in _arm.values() if v["new_run"]]
_c1 = S1M["checks"]["S1-C1"]["per_pool"]
_reg = subprocess.run(["git", "log", "--diff-filter=A", "--format=%ct", "--", "prereg/PREREG_S1.md"], cwd=C.AIR,
                      capture_output=True, text=True, check=True).stdout.split()
_tb = S1B["c_conclusions"]
CLAIMS.update({
    "S1: every registered check S1-C1..C5 passed, so both families give verdicts":
        all(S1["gates"]["checks"].values()) and S1["gates"]["CL"]["verdict"] and S1["gates"]["AR"]["verdict"],
    "A8: the job that produced every value started after the registration commit":
        len(_reg) == 1 and datetime.datetime.fromisoformat(S1M["created"]).timestamp() > int(_reg[0]),
    "A8: recomputed recorded-command checks agree except 3 F18 residuals of the 10k teacher-forced models, < 1%":
        all(_c1[p][c]["pass"] for p in _c1 for c in ("lam1", "rJ"))
        and all(not _c1[p][c]["outside"] for p in _c1 for c in ("lam1", "rJ") if (p, c) != ("F18", "rJ"))
        and all(f18[k]["group"] == "B 10k iterations" for k in _c1["F18"]["rJ"]["outside"])
        and _c1["F18"]["rJ"]["max_difference"] < 0.01,
    "Discussion/Conclusion/A8: CL1 worse with policy commands (97.5% interval below 0, beyond the margin)":
        _lab(_cl["CL1"]) == "WORSE WITH POLICY COMMANDS" and _cl["CL1"]["ci97.5"][1] < 0
        and _cl["CL1"]["ci97.5"][0] < -_mg and _cl["CL1"]["rho_new"] < _cl["CL1"]["rho_base"],
    "Discussion/A8: CL2 not settled (interval includes 0 and reaches beyond the margin)":
        _lab(_cl["CL2"]) == "NOT SETTLED" and _cl["CL2"]["ci97.5"][0] < -_mg < 0 < _cl["CL2"]["ci97.5"][1],
    "A8: CL3 rank agreement closer for lambda_1 than for r_J":
        _cl["CL3"]["lam1"]["rho"] > _cl["CL3"]["rJ"]["rho"],
    "A8: CL4 on X5: r_J worse with policy commands, lambda_1 unchanged":
        _cl["CL4"]["CL1"]["label"] == "WORSE WITH POLICY COMMANDS"
        and _cl["CL4"]["CL2"]["label"] == "UNCHANGED BY THE COMMAND SOURCE",
    "A8: CL8 noisy commands leave both correlations unchanged":
        all(_cl["CL8"][q]["label"] == "UNCHANGED BY THE COMMAND SOURCE" for q in ("CL1", "CL2")),
    "A8: CL9 drop in trained, unchanged in compressed; lambda_1 not settled in both":
        _cl["CL9"]["trained"]["CL1"]["label"] == "WORSE WITH POLICY COMMANDS"
        and _cl["CL9"]["compressed"]["CL1"]["label"] == "UNCHANGED BY THE COMMAND SOURCE"
        and all(_cl["CL9"][g]["CL2"]["label"] == "NOT SETTLED" for g in ("trained", "compressed")),
    "A8: closed-loop growth rate did not rank the simulators (|rho| < 0.1 on F18); CL5 worse":
        abs(_cl["CL5"]["rho_new"]) < 0.1 and _cl["CL5"]["label"] == "WORSE WITH POLICY COMMANDS",
    "A8: CL7 no rollout diverged under policy, closed-loop or noisy commands, F18 and X5":
        all(g[f"{u}_diverged_{c}"] == 0 for g in _cl7 for u in ("models", "trajectories") for c in ("pol", "cl", "noise")),
    "A8: CL6 imagined return fell as E100 rose (interval below 0), no model left out":
        _cl["CL6"]["rho"] < 0 and _cl["CL6"]["ci95"][1] < 0 and _cl["CL6"]["n_excluded_diverged"] == 0,
    "A8: exploratory two-level intervals wider; CL1's still wholly below zero":
        _cl["exploratory_two_level"]["CL1"]["ci97.5"][1] < 0
        and all(_cl["exploratory_two_level"][q]["ci97.5"][1] - _cl["exploratory_two_level"][q]["ci97.5"][0]
                > _cl[q]["ci97.5"][1] - _cl[q]["ci97.5"][0] for q in ("CL1", "CL2")),
    "A8: X5 table: recorded and policy columns equal the registered CL4 point estimates":
        abs(S1X["X5_within_split_mean_rho_with_E100"]["rJ_pol"]["mean"] - _cl["CL4"]["CL1"]["rho_new_mean"]) < 1e-12
        and abs(S1X["X5_within_split_mean_rho_with_E100"]["lam1_rec"]["mean"] - _cl["CL4"]["CL2"]["rho_base_mean"]) < 1e-12,
    "A9: pool 38 = 26 earlier + 12 new; new = seeds 3, 4 of tf_s7 and ar_s7 per architecture; five seeds each":
        len(_old) == 26 and len(_new) == 12
        and sorted((v["arch"], v["regime"], v["spec"], v["seed"]) for v in _new)
        == sorted((a, r, "s7", s) for a in ("mlp", "rssm", "transformer") for r in ("tf", "ar") for s in (3, 4))
        and all(sorted(v["seed"] for v in _arm.values() if (v["arch"], v["regime"], v["spec"]) == (a, r, "s7"))
                == [0, 1, 2, 3, 4] for a in ("mlp", "rssm", "transformer") for r in ("tf", "ar")),
    "A9: rest of the pool: 3 matched rollout-trained MLP and Transformer each, 2 teacher-forced RSSM variants":
        sorted((v["arch"], v["regime"]) for v in _old if v["spec"] == "matched")
        == [("mlp", "ar")] * 3 + [("transformer", "ar")] * 3
        and sorted((v["arch"], v["regime"], v["spec"]) for v in _old if v["spec"] in ("x1v1", "x1v2"))
        == [("rssm", "tf", "x1v1"), ("rssm", "tf", "x1v2")],
    "A9: new runs trained 2,500 iterations at 2 threads on F18's split, all finite; S1-C5 passed":
        all(v["iterations"] == 2500 and v["threads"] == 2 and v["holdout_episodes"] == [1, 8] and v["weights_finite"]
            and not v["loss_ever_nonfinite"] for v in S1M["training"].values())
        and S1M["checks"]["S1-C5"]["pass"],
    "A9: AR1 label-free better; rests on MLP and Transformer (E1 rho < -0.5); RSSM: E1 ranks well, lambda_1 poorly":
        _lab(_ar["AR1"]) == "LABEL-FREE BETTER" and _ar["AR1"]["ci97.5"][0] > 0
        and _ar["AR1"]["rho_base_per_group"][0] < -0.5 and _ar["AR1"]["rho_base_per_group"][2] < -0.5
        and _ar["AR1"]["rho_base_per_group"][1] > 0.8 and _ar["AR1"]["rho_new_per_group"][1] < 0.3,
    "A9: AR2 not settled; r_J level with E8 in the MLP, above it in RSSM and Transformer":
        _lab(_ar["AR2"]) == "NOT SETTLED"
        and abs(_ar["AR2"]["rho_new_per_group"][0] - _ar["AR2"]["rho_base_per_group"][0]) < 1e-9
        and all(_ar["AR2"]["rho_new_per_group"][i] > _ar["AR2"]["rho_base_per_group"][i] for i in (1, 2)),
    "A9: AR3 pooled both not settled, lambda_1 barely ranks (|rho| < 0.2); extended AR1 label-free, AR2 not settled":
        all(_ar["AR3"]["pooled"][q]["label"] == "NOT SETTLED" for q in ("AR1", "AR2"))
        and abs(_ar["AR3"]["pooled"]["AR1"]["rho_new"]) < 0.2
        and _ar["AR3"]["extended"]["AR1"]["label"] == "LABEL-FREE BETTER"
        and _ar["AR3"]["extended"]["AR2"]["label"] == "NOT SETTLED",
    "A9: AR4 reversal in 5/5 for MLP and Transformer; RSSM better at 1 step 5/5, worse at 100 in 0/5":
        all(_ar["AR4"][a]["both"] == _ar["AR4"][a]["n_pairs"] == 5 for a in ("mlp", "transformer"))
        and _ar["AR4"]["rssm"]["tf_better_E1"] == 5 and _ar["AR4"]["rssm"]["tf_worse_E100"] == 0,
    "A9: the runs that diverged under recorded commands are teacher-forced MLPs with finite values":
        len(S1X["AR_diverged_recorded"]) == 3
        and all(v["arch"] == "mlp" and v["regime"] == "tf" for v in S1X["AR_diverged_recorded"].values())
        and all(math.isfinite(_arm[k][m]) for k in S1X["AR_diverged_recorded"] for m in ("E100", "lam1_rec", "rJ_rec")),
    "A10: longer training lowered E100 for both arms, E1 moved by < 0.01; all four statements in every seed":
        all(S1B["b_scores"][a]["10000"]["mean"]["E100"] < S1B["b_scores"][a]["2500"]["mean"]["E100"]
            and abs(S1B["b_scores"][a]["10000"]["mean"]["E1"] - S1B["b_scores"][a]["2500"]["mean"]["E1"]) < 0.01
            for a in ("rollout_trained", "teacher_forced"))
        and all(_tb[n][s][k] for n in ("2500", "10000") for s in ("0", "1", "2")
                for k in ("tf_better_E1", "tf_worse_E100", "lam1_tf_positive", "lam1_rollout_negative")),
    "A10: slopes at 10k much smaller than at 2,500 (largest magnitude at 10k below smallest at 2,500)":
        max(abs(S1B["a_still_improving"]["arm_10000"]["slope_min"]), abs(S1B["a_still_improving"]["arm_10000"]["slope_max"]))
        < min(abs(S1B["a_still_improving"]["arm_2500"]["slope_min"]), abs(S1B["a_still_improving"]["arm_2500"]["slope_max"])),
    "A10: parent kept less than all of the control's skill at every horizon (more training added little)":
        all(0.9 < v["K"] < 1 for v in S1B["d_ft_control"]["K_P_vs_CTRL"].values()),
})

if __name__ == "__main__":
    out = {k: bool(v) for k, v in CLAIMS.items()}
    failed = [k for k, v in out.items() if not v]
    json.dump({"claims": out, "failed": failed}, open(os.path.join(C.RESULTS, "claims_check.json"), "w"), indent=1)
    print(json.dumps({"n": len(out), "failed": failed}, indent=1))
    sys.exit(1 if failed else 0)

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
    "A9: AR3 pooled both not settled, lambda_1 does not rank the pool (|rho| < 0.2); extended AR1 label-free, AR2 not settled":
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
    "A10: E100 lower after longer training for both arms, E1 moved by < 0.01; all four statements in every seed":
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

# W3 (rewrite): assertions for sentences new or changed in the rewritten main text (Sec. 4, Tables 2-3, abstract).
_ft = FT["primary"]
_x5 = T["X5"]
_ar1, _ar2 = _ar["AR1"], _ar["AR2"]
_holm = RX["A2c_holm"]["tests"]
CLAIMS.update({
    "Table 3: Q1-Q4 verdicts as registered (Q1 label-free better, Q2 not settled, Q3 one-step better than lambda_1 vs E1 "
    "and 8-step better vs E8, Q4 label-free better against both)":
        (S["Q1"]["verdict"], S["Q2"]["verdict"], S["Q3_vs_E1"]["verdict"], S["Q3_vs_E8"]["verdict"],
         S["Q4_vs_E1"]["verdict"], S["Q4_vs_E8"]["verdict"])
        == ("LABEL-FREE BETTER", "NOT SETTLED", "ONE-STEP BETTER", "TRAINING-HORIZON BETTER",
            "LABEL-FREE BETTER", "LABEL-FREE BETTER"),
    "Table 3: AR1 label-free better, AR2 not settled, CL1 worse with policy commands, CL2 not settled, FT1 restores, "
    "FT2 nearly as skilled":
        (_lab(_ar["AR1"]), _lab(_ar["AR2"]), _lab(_cl["CL1"]), _lab(_cl["CL2"]), _ft["FT1"]["verdict"], _ft["FT2"]["verdict"])
        == ("LABEL-FREE BETTER", "NOT SETTLED", "WORSE WITH POLICY COMMANDS", "NOT SETTLED",
            "FINE-TUNING RESTORES SKILL", "NEARLY AS SKILLED"),
    "Table 3: the 97.5% intervals of Q1, AR1 (above 0) and CL1 (below 0) agree in sign with their verdicts; Q2 and CL2 span 0":
        S["Q1"]["ci97.5"][0] > 0 and _ar1["ci97.5"][0] > 0 and _cl["CL1"]["ci97.5"][1] < 0
        and S["Q2"]["ci97.5"][0] < 0 < S["Q2"]["ci97.5"][1] and _cl["CL2"]["ci97.5"][0] < 0 < _cl["CL2"]["ci97.5"][1],
    "Table 3 daggers: Q3 vs E1 and Q4 vs E8 fail Holm's correction; Q3 vs E8 and Q4 vs E1 survive":
        not _holm["Q3_vs_E1"]["rejects_zero"] and not _holm["Q4_vs_E8"]["rejects_zero"]
        and _holm["Q3_vs_E8"]["rejects_zero"] and _holm["Q4_vs_E1"]["rejects_zero"],
    "Sec. 4.1: Q6 (30% pruning shift in lambda_1) not settled": S["Q6"]["verdict"] == "NOT SETTLED",
    "Sec. 4.1: 8-bit weights keep most of the 100-step skill (K above 0.9)": K("int8w", 100) > 0.9,
    "Sec. 4.1: longer training: teacher forcing better at 1 step and worse at 100 in 3 of 3 seeds; loss still falling "
    "in most of the 10,000-iteration runs":
        len([k for k in _tb["10000"] if k != "mean"]) == 3
        and all(_tb["10000"][s_]["tf_better_E1"] and _tb["10000"][s_]["tf_worse_E100"] for s_ in ("0", "1", "2"))
        and S1B["a_still_improving"]["arm_10000"]["n_falling"] > S1B["a_still_improving"]["arm_10000"]["n"] / 2,
    "Sec. 4.1/abstract: reversal in the MLP and the Transformer (5/5), none in the RSSM at 100 steps (0/5)":
        _ar["AR4"]["mlp"]["both"] == _ar["AR4"]["transformer"]["both"] == 5 and _ar["AR4"]["rssm"]["tf_worse_E100"] == 0
        and _ar["AR4"]["rssm"]["tf_better_E1"] > 0,
    "Sec. 4.2 / Table 2: in the RSSM one-step error ranks better than lambda_1; in the MLP and the Transformer lambda_1 does":
        _ar1["rho_base_per_group"][1] > _ar1["rho_new_per_group"][1]
        and _ar1["rho_new_per_group"][0] > _ar1["rho_base_per_group"][0]
        and _ar1["rho_new_per_group"][2] > _ar1["rho_base_per_group"][2],
    "Sec. 4.2: five-split pool: most simulators per split are compressed copies":
        sum(r["variant"] is not None for r in _x5) / 5 > len(_x5) / 5 / 2,
    "Sec. 4.2: F18 rho: lambda_1 ranks better than E1, E8 not separated (the Q1/Q2 correlations in Table 2)":
        S["Q1"]["rho_new"] > S["Q1"]["rho_base"] and S["Q2"]["rho_new"] <= S["Q2"]["rho_base"] + 0.1,
    "Sec. 4.1 uncertainty: released ensemble's spread ranks error well (Spearman above 0.5) yet its factor grows with horizon":
        tmf("n11_d2_sp_epi_h100") > 0.5 and tmf("n8_k_rel_epi_h1") > 1,
    "Sec. 4.2: the policy-command drop is in the residual, and the closed-loop growth rate does not rank (|rho| < 0.1)":
        _cl["CL1"]["rho_new"] < _cl["CL1"]["rho_base"] and abs(_cl["CL5"]["rho_new"]) < 0.1,
    "Sec. 4.2: imagined return and error: higher-error simulators gave lower return (rho < 0, secondary)":
        _cl["CL6"]["rho"] < 0,
    "Discussion: simulators from one-step training amplify errors (lambda_1 > 0 for N=1 and teacher forcing, X5 all B)":
        D1[N[1]]["lam1"] > 0 and D3["lam1_positive_share"]["B"] == 1.0,
})

# W4 (rewrite): sentences changed after the fresh-eyes review (Q5 in Sec. 4.2, lost verdicts named, labels).
_a2a = RX["A2a_two_level"]["registered_tests"]
CLAIMS.update({
    "Sec. 4.2: Q5 (no one-step-trained models): lambda_1 ranks better than E1 (label-free better, secondary), yet "
    "fails Holm's correction and the two-level intervals":
        S["Q5_vs_E1"]["verdict"] == "LABEL-FREE BETTER" and S["Q5_vs_E1"]["rho_new"] > S["Q5_vs_E1"]["rho_base"]
        and not _holm["Q5_vs_E1"]["rejects_zero"] and not _a2a["Q5_vs_E1"]["two_level_excludes_zero"],
    "Sec. 4.2: Q3 vs E1 fails Holm's correction and the two-level intervals":
        not _holm["Q3_vs_E1"]["rejects_zero"] and not _a2a["Q3_vs_E1"]["two_level_excludes_zero"],
    "Limitations: the verdicts lost under two-level intervals are exactly Q1 at 24 steps, Q3 vs E1 and Q5 vs E1":
        sorted(k for k, v in _a2a.items() if v["registered_verdict"] != "NOT SETTLED"
               and not v["two_level_excludes_zero"]) == ["Q1_E24", "Q3_vs_E1", "Q5_vs_E1"],
    "Checklist: the released (rollout-trained) model's growth rate is positive too":
        S["D5_released"]["lam1"] > 0,
    "Sec. 4.1: FT1's skill before fine-tuning (38%) is on seeds 0-2":
        sorted(int(x) for x in FT["seeds"]) == [0, 1, 2],
})

# I3 (improvement round): every result-bearing sentence added or changed in I2 (improve/notes/I2_claims.md).
EF = json.load(open(os.path.join(C.RESULTS, "external_facts.json")))
IW = json.load(open(os.path.join(C.RESULTS, "im_within.json")))
PN = json.load(open(os.path.join(C.RESULTS, "paper_numbers.json")))
PN = PN.get("values", PN)
_iw = lambda g, c: IW[g]["checks"][c]["rho"]                                             # noqa: E731
_iwd = lambda g, d: IW[g]["diffs"][d]                                                    # noqa: E731
_pn = lambda k: PN[k]["value"]                                                           # noqa: E731
_pni = lambda k: int(str(_pn(k)).replace(",", "").replace("{", "").replace("}", ""))     # noqa: E731
_v = lambda x: str(x).lower()                                                            # noqa: E731
_pol = _cl["CL1"]["rho_new"], _cl["CL2"]["rho_new"], _cl["CL5"]["rho_new"]
CLAIMS.update({
    # Abstract, Introduction, Sec. 4.1: the horizon RWM and RWM-U imagine (external facts)
    "I2 Abstract/Intro/Fig. 1: RWM and RWM-U both imagine 100 steps":
        EF["rwm_imag_steps"]["value"] == 100 and EF["rwmu_imag_steps"]["value"] == 100
        and _pn("ext_rwm_imag_steps") == _pn("ext_rwmu_imag_steps") == "100",
    "I2 Fig. 1: the one vertical line is the imagined horizon, 100 steps, which lies among the plotted steps":
        EF["rwm_imag_steps"]["value"] == 100 and F["B_cross_floor_step"] is not None,
    "I2 Abstract: teacher-forced and N=1 are worse than the rollout-trained / N=8 model at 100 steps (ratios above 1)":
        D3["h100"]["B_over_A"] > 1 and D1[N[1]]["E100"] > D1[N[8]]["E100"],
    "I2 Abstract: pruning 30% keeps skill one step ahead (K > 0.85) but little at 100 steps (K < 0.5)":
        K("u30", 1) > 0.85 and K("u30", 100) < 0.5,
    "I2 Abstract/Sec. 4.1: fine-tuning restores skill (FT1, registered, primary)":
        _v(FT["primary"]["FT1"]["verdict"]) == "fine-tuning restores skill",
    "I2 Sec. 4.1: the fine-tuned copies were nearly as skilled as their parents fine-tuned the same way (FT2, registered)":
        _v(FT["primary"]["FT2"]["verdict"]) == "nearly as skilled",
    "I2 Abstract: the growth rate ranked better than one-step error (Q1, registered primary)":
        _v(S["Q1"]["verdict"]) == "label-free better",
    "I2 Abstract/Sec. 4.2: the growth rate was not separated from the 8-step error (Q2 not settled, primary)":
        _v(S["Q2"]["verdict"]) == "not settled",
    "I2 Sec. 4.2: the 8-step error ranked ahead of the growth rate at 24 steps (Q2 at 24, secondary)":
        _v(S["Q2_E24"]["verdict"]) == "training-horizon better",
    "I2 Abstract/Sec. 4.2: on the five-split pool one-step error ranked better than the growth rate, and the 8-step error better still (Q3, secondary)":
        _v(S["Q3_vs_E1"]["verdict"]) == "one-step better" and _v(S["Q3_vs_E8"]["verdict"]) == "training-horizon better",
    "I2 Abstract/Sec. 4.2: the kinematic residual ranked better than one-step error (Q4, secondary)":
        _v(S["Q4_vs_E1"]["verdict"]) == "label-free better",
    "I2 Abstract/Sec. 4.2: with policy commands the residual ranked worse (CL1, registered primary), the growth rate's change not settled (CL2)":
        _v(_cl["CL1"]["verdict"]) == "worse with policy commands" and _v(_cl["CL2"]["verdict"]) == "not settled",
    "I2 Sec. 4.2: AR1 label-free better, AR2 not settled (registered primary)":
        _v(_ar["AR1"]["verdict"]) == "label-free better" and _v(_ar["AR2"]["verdict"]) == "not settled",
    "I2 Sec. 4.1: the growth-rate shift at 30% pruning was not settled (Q6, secondary)":
        _v(S["Q6"]["verdict"]) == "not settled",
    "I2 Discussion (1): no label-free check beat the 8-step error in a primary test (Q2 and AR2 not settled)":
        _v(S["Q2"]["verdict"]) == "not settled" and _v(_ar["AR2"]["verdict"]) == "not settled",
    "I2 Conclusion: the growth rate beat one-step error on the main pool but failed its replication (Q1, Q3 vs E1); "
    "the residual's rank correlation with recorded commands is the highest of Table 2's F18 column (descriptive)":
        _v(S["Q1"]["verdict"]) == "label-free better" and _v(S["Q3_vs_E1"]["verdict"]) == "one-step better"
        and S["Q4_vs_E1"]["rho_new"] > max(S["Q1"]["rho_new"], S["Q4_vs_E1"]["rho_base"]),
    # Related work, Sec. 3 (external facts, counts)
    "I2 Related work: iKCE diagnosed one model (DreamerV3 on walker-walk) and does not rank models":
        EF["ikce_n_models"]["value"] == 1 and "walker" in _v(EF["ikce_task"]["value"])
        and EF["ikce_ranks_models"]["value"] is False,
    "I2 Sec. 3: 7,991 transitions are 0.13% of RWM's 6 million pretraining transitions":
        _pni("ext_ours_transitions") == 7991 and EF["rwm_pretrain_transitions"]["value"] == 6000000
        and round(100 * 7991 / 6000000, 2) == 0.13 and _pn("ext_budget_fraction") == "0.13\\%",
    "I2 Sec. 4.1: RWM reports about 750 times our transitions (6 million / 7,991, to the nearest 10)":
        round(6000000 / 7991, -1) == 750 and _pn("scale_ratio") == "about 750",
    "I2 Sec. 3: 101 trained and 200 compressed simulators; main pool 97 = 57 + 40; five-split pool 255 = 55 + 200; "
    "11 trained models in both; 352 pool entries":
        _pni("n_trained") == _pni("im_f18_trained") + _pni("im_x5_trained") - _pni("im_both_trained") == 101
        and _pni("n_P") == _pni("im_f18_trained") + _pni("n_C") == 97
        and _pni("n_X") == _pni("im_x5_trained") + _pni("n_compressed") == 255
        and _pni("im_pool_entries") == _pni("n_P") + _pni("n_X") == 352 and _pni("n_compressed") == 200,
    # Sec. 4.2 and appendix A4: rankings within one training method (EXPLORATORY, added after the registered results)
    "I2 Sec. 4.2 (exploratory): every check ranked the 40 compressed copies well, rho from +0.82 to +0.96":
        IW["c"]["n_models"] == 40 and min(_iw("c", c) for c in ("E1", "E8", "lam1", "rJ")) > 0.815
        and max(_iw("c", c) for c in ("E1", "E8", "lam1", "rJ")) < 0.965
        and round(_iw("c", "lam1"), 2) == 0.82 and round(_iw("c", "E8"), 2) == 0.96,
    "I2 Sec. 4.2 (exploratory): among the 48 rollout-trained main-pool simulators the residual ranked better than "
    "8-step error (interval of the difference above 0)":
        IW["a"]["n_models"] == 48 and _iw("a", "rJ") > _iw("a", "E8") and _iwd("a", "rJ_minus_E8")["ci95"][0] > 0
        and round(_iw("a", "rJ"), 2) == 0.85 and round(_iw("a", "E8"), 2) == 0.38,
    "I2 Sec. 4.2 (exploratory): on the five-split pool's 40 rollout-trained simulators the growth rate failed "
    "(rho < 0, interval includes 0) and the residual's lead over 8-step error was unclear (interval includes 0)":
        IW["b"]["n_models"] == 40 and _iw("b", "lam1") < 0 and IW["b"]["checks"]["lam1"]["ci95"][1] > 0
        and _iwd("b", "rJ_minus_E8")["ci95"][0] < 0 < _iwd("b", "rJ_minus_E8")["ci95"][1],
    "I2 Disc. (3) (exploratory): within one method the residual ranked better than 8-step error on the main pool, not "
    "clearly on the five-split pool, where the growth rate failed":
        _iwd("a", "rJ_minus_E8")["ci95"][0] > 0 and _iw("b", "lam1") < 0 < IW["b"]["checks"]["lam1"]["ci95"][1]
        and _iwd("b", "rJ_minus_E8")["ci95"][0] < 0 < _iwd("b", "rJ_minus_E8")["ci95"][1],
    "I2 Sec. 4.2 (exploratory): every recorded-command check ranked the 40 compressed copies at 0.82 or better":
        all(_iw("c", c) > 0.815 for c in ("E1", "E8", "lam1", "rJ")),
    # Sec. 4.2: imagined return (CL6 secondary; comparison exploratory, I4)
    "I2/I4 Sec. 4.2: imagined return is nominally the strongest check under policy commands (|rho| above the residual's "
    "and the growth rate's; CL6 secondary, comparison exploratory, not a registered check)":
        abs(_cl["CL6"]["rho"]) > max(_pol[0], _pol[1], abs(_pol[2]))
        and round(abs(_cl["CL6"]["rho"]), 2) == round(_iw("d", "negret"), 2) == 0.86
        and _cl["CL6"]["rho"] < 0,
    "I4 Sec. 4.2: closed loop, the growth rate ranked the simulators worse (CL5 WORSE WITH POLICY COMMANDS, secondary), "
    "rho near zero":
        abs(_cl["CL5"]["rho_new"]) < 0.05 and _cl["CL5"]["label"] == "WORSE WITH POLICY COMMANDS",
    "I4 Sec. 4.2 and Disc.: across architectures together the growth rate's lead over one-step error was not settled "
    "(AR3 pooled AR1 NOT SETTLED, secondary); rho descriptive, below 0.2":
        _ar["AR3"]["pooled"]["AR1"]["label"] == "NOT SETTLED" and _ar["AR3"]["pooled"]["AR1"]["rho_new"] < 0.2,
    "I4 Abstract: the residual ranked the main pool better than one-step error (Q4 vs E1, secondary, LABEL-FREE BETTER)":
        S["Q4_vs_E1"]["verdict"] == "LABEL-FREE BETTER" and S["Q4_vs_E1"]["ci95"][0] > 0,
    "I4 Sec. 4.1: longer training and two of three other networks gave the same reversal (MLP, Transformer all "
    "pairs; RSSM none)":
        _pn("s1_ar4_mlp_both") == _pn("s1_ar4_mlp_n_pairs")
        and _pn("s1_ar4_transformer_both") == _pn("s1_ar4_transformer_n_pairs")
        and _pn("s1_ar4_rssm_both") == "0",
    "I4 Sec. 4.2: on X5 the residual was not separated from the 8-step error (X1 interval spans 0; exploratory)":
        X["X1_rJ_vs_E8"]["ci95"][0] < 0 < X["X1_rJ_vs_E8"]["ci95"][1],
    "I2 Sec. 4.1 (descriptive): other networks - MLP better one step but worse at 100 steps in every pair; "
    "RSSM better one step but worse at 100 steps in none":
        _pn("s1_ar4_rssm_tf_better_E1") == _pn("s1_ar4_rssm_n_pairs")
        and _pn("s1_ar4_rssm_tf_worse_E100") == "0"
        and _pn("s1_ar4_mlp_tf_worse_E100") == _pn("s1_ar4_mlp_n_pairs"),
})

# M6 (Step 2): every result-bearing sentence added in M5 (Sec. 4.2 run-in paragraph, Table III's physics block, abstract,
# Limitations, Conclusion, appendices A11-A12; step2/notes/M5.md). Registered verdicts stand as registered.
S2 = json.load(open(os.path.join(C.RESULTS, "s2_stats.json")))
S2F = json.load(open(os.path.join(C.RESULTS, "s2_fidelity.json")))
S2P = json.load(open(os.path.join(C.RESULTS, "s2_pilot.json")))
S2M = json.load(open(os.path.join(C.RESULTS, "s2_main.json")))
_s2p = S2["premise_and_primary"]
_s2s = S2["secondary"]
_s2d = {r["sim"]: r for r in S2["descriptive"]["rows"]}
_s2rank = sorted(_s2d, key=lambda x: -_s2d[x]["ret"])
_s2still = [k for k, r in _s2d.items() if r["fall"] == 0]
_s2pil = S2P["policies"]
_s2g = S2F["gait_authors_policy"]["calibrated"]["summary"]
_s2rec = S2F["gait_recorded"]["summary"]
_s2rep = S2F["replay_held_out"]["calibrated"]["horizons"]
_s2top = [q["fall"] for sim in ("REL", "REL__fp16") for q in _s2d[sim]["per_policy"]]
_s2iv = [(v["ret_ci95"][0], v["ret_ci95"][1]) for v in _s2pil.values()]
CLAIMS.update({
    "M6 Abstract/Sec. 4.2/Table III: no check was shown to predict physics return nor shown not to "
    "(R0, R1, R2 all NOT SETTLED, registered)":
        _s2p["R0"]["label"] == _s2p["R1"]["label"] == _s2p["R2"]["label"] == "NOT SETTLED"
        and _s2p["R0"]["ci95"][0] < 0 < _s2p["R0"]["ci95"][1] and _s2p["R1"]["ci97.5"][0] < 0 and _s2p["R2"]["ci97.5"][0] < 0,
    "M6 Abstract/Sec. 4.2: nine simulators, eighteen policies, none failed; the pool is the released model and its eight copies":
        S2["n_simulators"] == 9 and len(S2M["policies"]) == 18 and not S2["failed_policies"] and not S2["missing_policies"]
        and sorted(_s2d) == sorted(["REL"] + [f"REL__{v}" for v in
                                              ("fp16", "int8w", "u10", "u20", "u30", "u50", "s192", "s128")]),
    "M6 Sec. 4.2: Policies from three rebuilt simulators fell in every pilot episode, the released model's did not (Gate 2: P1, P3 passed, P2 failed)":
        all(_s2pil[k]["fall_rate"] == 1.0 for k in ("A_s0__p0", "B_s0__p0", "A_s0__u50__p0"))
        and _s2pil["REL__p0"]["fall_rate"] < 0.25 and not all(lo <= v["ret"] <= hi for lo, hi in _s2iv for v in _s2pil.values()),
    "M6 Sec. 4.2 (descriptive): the best policies (released model, 16-bit copy) fell in 31% to 45% of episodes; "
    "the top two simulators by physics return are those two":
        _s2rank[:2] == ["REL", "REL__fp16"] and round(100 * min(_s2top)) == 31 and round(100 * max(_s2top)) == 45,
    "M6 A11: the authors' policy walks in the calibrated model (Gate 1): falls < 10%, two-feet gap < 10 points, "
    "speed-error ratio <= 1.5":
        _s2g["fall_rate"] < 0.10 and abs(_s2g["two_feet"] - _s2rec["two_feet"]) < 0.10
        and _s2g["speed_err"] / _s2rec["speed_err"] <= 1.5,
    "M6 A11 (descriptive): the calibrated model's replay error is below the floor up to 24 steps and above it at 100; "
    "no window unstable":
        all(_s2rep[h]["mean"] < _s2rep[h]["floor"] for h in ("1", "8", "24")) and _s2rep["100"]["mean"] > _s2rep["100"]["floor"]
        and sum(S2F["replay_held_out"]["calibrated"]["unstable_windows"]) == 0,
    "M6 A11: the policy turned more accurately in the model than the recording did":
        _s2g["turn_err"] < _s2rec["turn_err"],
    "M6 A12: all rebuilt-simulator pilot policies had higher imagined returns than the released-model policy":
        all(_s2pil[k]["imagined_return"] > _s2pil["REL__p0"]["imagined_return"]
            for k in ("A_s0__p0", "B_s0__p0", "A_s0__u50__p0")),
    "M6 A12: R1 points towards one-step error (lambda_1 ranks in reverse, E_1 with the right sign) but does not exclude zero":
        _s2p["R1"]["rho_new"] < 0 < _s2p["R1"]["rho_base"] and _s2p["R1"]["delta"] < 0 and _s2p["R1"]["ci97.5"][1] >= 0,
    "M6 A12: R0 ranks physics return weakly (rho between 0 and 0.5)": 0 < _s2p["R0"]["rho"] < 0.5,
    "M6 A12: R2 the residual and 8-step error rank alike (both rho within 0.1)":
        abs(_s2p["R2"]["rho_new"] - _s2p["R2"]["rho_base"]) < 0.1,
    "M6 A12 (exploratory): the two-level intervals give the same verdicts (all include 0 / end at 0)":
        S2["exploratory"]["two_level"]["R0"]["ci95"][0] < 0 < S2["exploratory"]["two_level"]["R0"]["ci95"][1]
        and S2["exploratory"]["two_level"]["R1"]["ci97.5"][1] >= 0 and S2["exploratory"]["two_level"]["R2"]["ci97.5"][1] >= 0,
    "M6 A12: the rollout-trained subset is not applicable (one rollout-trained simulator)":
        _s2s["rollout_subset"]["label"] == "NOT APPLICABLE" and _s2s["rollout_subset"]["n_models"] == 1,
    "M6 A12 (secondary): every secondary verdict NOT SETTLED":
        _s2s["fall_rate"]["R0"]["label"] == "NOT SETTLED" and _s2s["imagined_return"]["label"] == "NOT SETTLED"
        and all(_s2s["fall_rate"][k]["label"] == "NOT SETTLED" for k in ("R1", "R2"))
        and all(_s2s["policy_commands"][k]["label"] == "NOT SETTLED" for k in ("R1", "R2"))
        and all(_s2s["selection"][k]["label"] == "NOT SETTLED" for k in ("SEL1", "SEL2")),
    "M6 A12 (secondary): with fall rate as outcome the 100-step error ranked in reverse (rho < 0, interval includes 0)":
        _s2s["fall_rate"]["R0"]["rho"] < 0 < _s2s["fall_rate"]["R0"]["ci95"][1],
    "M6 A12 (secondary): imagined return ranked physics return weakly (0 < rho < 0.5)":
        0 < _s2s["imagined_return"]["rho"] < 0.5,
    "M6 A12 (secondary): selection by error checks fell short of the best three by less than the growth rate's did":
        _s2s["selection"]["checks"]["E1"]["shortfall"] < _s2s["selection"]["checks"]["lam1"]["shortfall"]
        and _s2s["selection"]["checks"]["E1"]["chosen"] == _s2s["selection"]["checks"]["E8"]["chosen"]
        == _s2s["selection"]["checks"]["E100"]["chosen"] == _s2s["selection"]["checks"]["rJ"]["chosen"],
    "M6 A12 (descriptive): the standing-still policies (50% pruned, width 128) never fell, rank 3rd and 4th by physics return, "
    "and their simulators have the two largest E100":
        sorted(_s2still) == ["REL__s128", "REL__u50"] and sorted(_s2rank[2:4]) == sorted(_s2still)
        and sorted(_s2d, key=lambda x: -_s2d[x]["E100"])[:2] == sorted(_s2still, key=lambda x: -_s2d[x]["E100"]),
    "M6 A12 (descriptive): the two policy seeds of the width-192 copy differ widely":
        abs(_s2d["REL__s192"]["per_policy"][0]["ret"] - _s2d["REL__s192"]["per_policy"][1]["ret"]) > 10,
    "M6 Limitations/Conclusion: the registered physics test was committed before the main job (registration commit precedes the results)":
        subprocess.run(["git", "merge-base", "--is-ancestor", json.load(open(os.path.join(C.RESULTS, "prereg_hash_s2.json")))
                        .get("commit", "HEAD"), "HEAD"], cwd=C.AIR).returncode == 0,
})

if __name__ == "__main__":
    out = {k: bool(v) for k, v in CLAIMS.items()}
    failed = [k for k, v in out.items() if not v]
    json.dump({"claims": out, "failed": failed}, open(os.path.join(C.RESULTS, "claims_check.json"), "w"), indent=1)
    print(json.dumps({"n": len(out), "failed": failed}, indent=1))
    sys.exit(1 if failed else 0)

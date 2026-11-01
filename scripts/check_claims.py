"""
The paper's words, checked against the numbers. Every qualitative statement the text makes about a result (an
order, a sign, "worse than predicting no change", "grows with the horizon") is an assertion here, read from the same
files the numbers come from. A failing assertion means the sentence is wrong, whatever the number beside it says.
-> results/claims_check.json; exits non-zero on a failure.
"""
import json
import os
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
    "30%-pruned copy crosses the floor in Fig. 1, after tens of steps":
        F["U_cross_floor_step"] is not None and F["U_cross_floor_step"] >= 20,
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
    "fp16 keeps essentially all skill at 368 (K > 0.99)": K("fp16", 368) > 0.99,
    # Uncertainty
    "spread factor grows with the horizon": tmf("n8_k_rel_epi_h1") < tmf("n8_k_rel_epi_h100") < tmf("n8_k_rel_epi_h368"),
    "coverage at 100 steps below nominal": tmf("x9_rel_mean_cov1_h100") < tmf("v3_cov_nominal1"),
    "per-member rollout raises coverage but stays below nominal":
        tmf("x9_rel_mean_cov1_h100") < tmf("x9_rel_per_cov1_h100") < tmf("v3_cov_nominal1"),
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
    "one- and two-step-trained models: lower one-step error, several times the best 100-step error":
        sum(f18[k]["E1"] for k in onestep) / len(onestep) < sum(f18[k]["E1"] for k in rollout_best) / len(rollout_best)
        and min(f18[k]["E100"] for k in onestep) > 2 * min(f18[k]["E100"] for k in rollout_best),
    "one-step error ranks training choices backwards (interval below 0; X5 negative too)":
        X["X2_training_choices"]["E1"]["ci95"][1] < 0 and X["X3_uncompressed"]["E1"] < 0,
    "every check ranks compressed copies above 0.8 (F18)":
        all(X["X2_compressed"][c]["rho"] > 0.8 for c in ("E1", "E8", "lam1", "rJ")),
    "choosing by E1 picked an N=1 or N=2 model for every seed":
        all(S["D6_config_choice_F18"]["picks"][f"E1_s{i}"].startswith(("mn_M32_N1_", "mn_M32_N2_")) for i in range(3)),
    "replication did not hold for lambda (X5)": S["Q3_vs_E1"]["verdict"] != "LABEL-FREE BETTER",
    "on X5 the 8-step error ranked better still than E1": S["Q3_vs_E8"]["rho_base_mean"] > S["Q3_vs_E1"]["rho_base_mean"],
    "released: low one-step error, positive growth rate, worse than floor at 368":
        S["D5_released"]["E1"] < S["D5_released"]["floor"]["E1"] and S["D5_released"]["lam1"] > 0
        and S["D5_released"]["E368"] > S["D5_released"]["floor"]["E368"],
    "r_J level with E8 and best on X5 (exploratory)":
        X["X1_rJ_vs_E8"]["ci95"][0] < 0 < X["X1_rJ_vs_E8"]["ci95"][1]
        and X["X1_rJ_vs_E1"]["rho_new_mean"] >= max(S["Q3_vs_E8"]["rho_base_mean"], S["Q3_vs_E1"]["rho_new_mean"]),
    "abstract: TF about twice as wrong over 2 s": 1.9 <= D3["h100"]["B_over_A"] < 2.1,
    "run-time: no label-free AUROC above 0.65 on A-ens; growth signal worse than counter":
        max(ICP_D["P_on"]["auroc"], ICP_D["F(D)"]["auroc"]) < 0.65 and NG["Q3"]["verdict"] == "WORSE THAN COUNTER",
    "exact gravity repair helps (interval below 1); approximate joint repair hurts":
        ICP_R["err|G|368"]["ci"][1] < 1 and ICP_R["err|J|368"]["ratio"] > 1,
})

if __name__ == "__main__":
    out = {k: bool(v) for k, v in CLAIMS.items()}
    failed = [k for k, v in out.items() if not v]
    json.dump({"claims": out, "failed": failed}, open(os.path.join(C.RESULTS, "claims_check.json"), "w"), indent=1)
    print(json.dumps({"n": len(out), "failed": failed}, indent=1))
    sys.exit(1 if failed else 0)

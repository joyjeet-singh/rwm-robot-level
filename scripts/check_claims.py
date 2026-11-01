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

if __name__ == "__main__":
    out = {k: bool(v) for k, v in CLAIMS.items()}
    failed = [k for k, v in out.items() if not v]
    json.dump({"claims": out, "failed": failed}, open(os.path.join(C.RESULTS, "claims_check.json"), "w"), indent=1)
    print(json.dumps({"n": len(out), "failed": failed}, indent=1))
    sys.exit(1 if failed else 0)

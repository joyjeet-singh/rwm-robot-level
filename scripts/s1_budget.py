"""Study TB (descriptive, not registered): does the training length change the conclusions?
Reads stored files only; writes results/s1_budget.json and results/s1_budget.txt.
Existing before this plan: training_tail_slopes.json, model_table.json (F18 rows A/B and A10k/B10k), ft_stats.json.
Nothing is rescored here; the 10,000-iteration models already have stored values in model_table.json."""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDM = os.environ.get("PDM_ROOT", os.path.dirname(ROOT))
M = ["E1", "E8", "E24", "E100", "E368", "lam1", "rJ"]
SEEDS = [0, 1, 2]

slopes = json.load(open(os.path.join(PDM, "rwm_repro/results/training_tail_slopes.json")))
table = {r["key"]: r for r in json.load(open(os.path.join(ROOT, "results/model_table.json")))["F18"]}
ft = json.load(open(os.path.join(ROOT, "results/ft_stats.json")))

out = {"label": "DESCRIPTIVE (study TB, not registered); three seeds, no intervals",
       "existed_before_plan": ["training_tail_slopes.json", "model_table.json", "ft_stats.json"],
       "newly_scored": "none (the 10,000-iteration models already have stored values)",
       "scope": "one robot, one trotting gait, simulated data; the 10k arms are 3 seeds each"}

# (a) still improving: loss slope over the last 250 iterations < 0, as stored
a = {}
for g in ["arm_2500", "arm_10000"]:
    s = slopes["summary"][g]
    a[g] = {"n": s["n"], "n_falling": s["n_falling"], "slope_min": s["min"], "slope_max": s["max"]}
a["runs_10k"] = {k: {"slope": v["slope_per_iter"], "falling": v["still_falling"], "final_state_loss": v["final_state_loss"]}
                 for k, v in slopes["runs"].items() if v["group"] == "arm_10000"}
a["runs_2500"] = {k: {"slope": v["slope_per_iter"], "falling": v["still_falling"], "final_state_loss": v["final_state_loss"]}
                  for k, v in slopes["runs"].items() if v["group"] == "arm_2500"}
out["a_still_improving"] = a

# (b) per method and seed
meth = {"rollout_trained": ("A", "A10k"), "teacher_forced": ("B", "B10k")}
b = {}
for name, (k25, k10) in meth.items():
    b[name] = {}
    for it, k in [("2500", k25), ("10000", k10)]:
        b[name][it] = {str(s): {m: table[f"{k}_s{s}"][m] for m in M} for s in SEEDS}
        b[name][it]["mean"] = {m: float(np.mean([table[f"{k}_s{s}"][m] for s in SEEDS])) for m in M}
out["b_scores"] = b

# (c) at 10,000 (and at 2,500 for comparison): TF vs rollout, per seed and mean
c = {}
for it in ["2500", "10000"]:
    r = {}
    for s in SEEDS + ["mean"]:
        A, B = b["rollout_trained"][it][str(s)], b["teacher_forced"][it][str(s)]
        r[str(s)] = {"tf_better_E1": B["E1"] < A["E1"], "tf_worse_E100": B["E100"] > A["E100"],
                     "lam1_tf_positive": B["lam1"] > 0, "lam1_rollout_negative": A["lam1"] < 0,
                     "E1_tf_minus_rollout": B["E1"] - A["E1"], "E100_tf_minus_rollout": B["E100"] - A["E100"]}
    c[it] = r
out["c_conclusions"] = c

# (d) fine-tuning control: P vs CTRL (P + 500 iterations), 20 X5 trajectories, from ft_stats.json
H = ["h1", "h8", "h24", "h100", "h368"]
me = ft["mean_error"]
d = {"note": "CTRL = P fine-tuned 500 more iterations (PREREG_FT.md); different arena (20 X5 trajectories); "
             "K_P_vs_CTRL = fraction of CTRL's skill over the floor that P already has",
     "mean_error": {h: {"P": me["P"][h], "CTRL": me["CTRL"][h], "floor": me["floor"][h],
                        "P_minus_CTRL": me["P"][h] - me["CTRL"][h]} for h in H},
     "K_P_vs_CTRL": {h: ft["descriptive"][f"K_P_vs_CTRL_{h}"] for h in H}}
out["d_ft_control"] = d

json.dump(out, open(os.path.join(ROOT, "results/s1_budget.json"), "w"), indent=1)

L = ["Study TB: training length (DESCRIPTIVE, not registered; 3 seeds, no intervals)",
     "Stored before this plan: tail slopes, model_table, ft_stats. Nothing rescored."]
L.append(f"(a) still falling (loss slope, last 250 its <0): 2500 its {a['arm_2500']['n_falling']}/{a['arm_2500']['n']}; "
         f"10000 its {a['arm_10000']['n_falling']}/{a['arm_10000']['n']}")
L.append(f"    slope range 2500: [{a['arm_2500']['slope_min']:.1e},{a['arm_2500']['slope_max']:.1e}]; "
         f"10000: [{a['arm_10000']['slope_min']:.1e},{a['arm_10000']['slope_max']:.1e}]")
L.append("(b) seed-mean scores     " + " ".join(f"{m:>6}" for m in M))
for name in meth:
    for it in ["2500", "10000"]:
        L.append(f"  {name[:8]:8s} {it:>5s}   " + " ".join(f"{b[name][it]['mean'][m]:6.3f}" for m in M))
L.append("(c) TF vs rollout, seeds 0/1/2/mean (1=yes)   [at 10000 its]")
for key, lab in [("tf_better_E1", "TF better at 1 step"), ("tf_worse_E100", "TF worse at 100 steps"),
                 ("lam1_tf_positive", "lam1>0 for TF"), ("lam1_rollout_negative", "lam1<0 for rollout")]:
    L.append(f"  {lab:24s} " + " ".join(str(int(c['10000'][s][key])) for s in ["0", "1", "2", "mean"])
             + "   [2500: " + " ".join(str(int(c['2500'][s][key])) for s in ["0", "1", "2", "mean"]) + "]")
L.append("(d) FT control P vs CTRL (+500 its), mean error h1/h8/h24/h100/h368")
L.append("  P-CTRL: " + " ".join(f"{d['mean_error'][h]['P_minus_CTRL']:.4f}" for h in H))
L.append("  K_P_vs_CTRL: " + " ".join(f"{d['K_P_vs_CTRL'][h]['K']:.3f}" for h in H))
open(os.path.join(ROOT, "results/s1_budget.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

"""Print a compact human-readable summary of results/stats.json (for reading the results; nothing is written)."""
import json
import os

import common as C

S = json.load(open(os.path.join(C.RESULTS, "stats.json")))
print(open(os.path.join(C.RESULTS, "stats_report.txt")).read())
print("n_nonfinite", S["n_nonfinite"], "floor_F18", {k: round(v, 3) for k, v in S["floor_F18"].items()})
print("\nD1 groups F18 (E1 E8 E24 E100 E368 lam rJ):")
for g, v in S["D1_groups_F18"].items():
    print(f"  {g:24s} n={v['n']:2d} " + " ".join(f"{v[k]:.4f}" for k in ("E1", "E8", "E24", "E100", "E368", "lam1", "rJ")))
print("\nD1 groups X5:")
for g, v in S["D1_groups_X5"].items():
    print(f"  {g:24s} n={v['n']:3d} " + " ".join(f"{v[k]:.4f}" for k in ("E1", "E8", "E24", "E100", "E368", "lam1", "rJ")))
print("\nD1 rho F18:", {k: round(v["rho"], 3) for k, v in S["D1_rho_F18"].items()})
print("D1 rho RC:", {k: round(v["rho"], 3) for k, v in S["D1_rho_RC"].items()})
print("\nD2 skill retained (X5):")
for v, d in S["D2_skill_retained_X5"].items():
    print(f"  {v:6s} " + "  ".join(f"{h}:{w['K']*100:7.1f} [{w['ci95'][0]*100:6.1f},{w['ci95'][1]*100:6.1f}]" for h, w in d.items()))
print("\nlambda shifts (X5):", {k: (round(v["mean"], 3), [round(x, 3) for x in v["ci95"]], v["n_positive_lam1"])
                                for k, v in S["lam_shift_X5"].items()})
print("\nD3 arms:")
for h in ("h1", "h8", "h24", "h100", "h368"):
    w = S["D3_arms_X5"][h]
    print(f"  {h}: A {w['A']:.4f} B {w['B']:.4f} A5 {w['A5']:.4f} floor {w['floor']:.4f}  B-A {w['B_minus_A']:+.4f} "
          f"{[round(x, 4) for x in w['ci95']]}  B/A {w['B_over_A']:.2f}  nworse {w['n_traj_B_worse']}")
print("  physical:", json.dumps(S["D3_arms_X5"]["physical"], indent=0)[:1500])
print("  lam:", S["D3_arms_X5"]["lam1"], S["D3_arms_X5"]["lam1_positive_share"])
print("\nD4:", S["D4_F18"], S["D4_X5"])
print("D5 released:", json.dumps(S["D5_released"])[:600])
print("D6:", json.dumps(S["D6_config_choice_F18"])[:900])

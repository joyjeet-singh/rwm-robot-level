"""Numbers added in the rewrite round (W2; called by paper_numbers.py after s1_numbers). Every value is read from a
result file; nothing is computed beyond formatting.

Source: results/s1_stats.json (registered analysis of studies CL and AR, scripts/s1_stats.py). The keys below re-print
values that s1_numbers.py already prints unsigned, as signed two-decimal correlations, so that the main text's Table 2
reads alike in every column. Status as in s1_numbers.py: CL1/CL2 and AR1/AR2 primary, CL5 secondary.
"""
import json
import os

import common as C

STATS = os.path.join(C.RESULTS, "s1_stats.json")
ARCHES = ("mlp", "rssm", "transformer")


def _rel(path):
    return os.path.relpath(path, os.path.dirname(C.AIR))


def _ld(path):
    with open(path) as fh:
        return json.load(fh)


def add(put, sig, sgn, pct, cib, dp, load, rel):
    S, s = _ld(STATS), _rel(STATS)
    r2 = lambda v: dp(v, 2, signed=True)                                                 # noqa: E731
    CL, AR = S["CL"], S["AR"]

    # ---- Table 2, main pool (F18) under the policy's joint commands (CL1, CL2 primary; CL5 secondary)
    put("rw_t_F18pol_rJ", r2(CL["CL1"]["rho_new"]), s, "CL.CL1.rho_new")
    put("rw_t_F18pol_lam1", r2(CL["CL2"]["rho_new"]), s, "CL.CL2.rho_new")
    put("rw_t_F18pol_lam1cl", r2(CL["CL5"]["rho_new"]), s, "CL.CL5.rho_new")

    # ---- Table 2, other architectures (study AR): correlation with E100 within each architecture
    for i, a in enumerate(ARCHES):
        assert AR["AR1"]["n_per_group"][i] == AR["AR2"]["n_per_group"][i]
        put(f"rw_t_{a}_E1", r2(AR["AR1"]["rho_base_per_group"][i]), s, f"AR.AR1.rho_base_per_group[{i}]")
        put(f"rw_t_{a}_lam1", r2(AR["AR1"]["rho_new_per_group"][i]), s, f"AR.AR1.rho_new_per_group[{i}]")
        put(f"rw_t_{a}_E8", r2(AR["AR2"]["rho_base_per_group"][i]), s, f"AR.AR2.rho_base_per_group[{i}]")
        put(f"rw_t_{a}_rJ", r2(AR["AR2"]["rho_new_per_group"][i]), s, f"AR.AR2.rho_new_per_group[{i}]")
        put(f"rw_t_{a}_n", str(AR["AR1"]["n_per_group"][i]), s, f"AR.AR1.n_per_group[{i}]")
    for k, blk, fld in (("E1", "AR1", "rho_base_mean"), ("lam1", "AR1", "rho_new_mean"),
                        ("E8", "AR2", "rho_base_mean"), ("rJ", "AR2", "rho_new_mean")):
        put(f"rw_t_armean_{k}", r2(AR[blk][fld]), s, f"AR.{blk}.{fld}")

"""Numbers added in the improvement round (I1; called by paper_numbers.py after rw_numbers). Every value is read from
a result file; nothing is computed beyond formatting.

Sources: results/im_within.json (scripts/im_within.py; EXPLORATORY, added after the registered results were seen) and
results/external_facts.json (scripts/im_facts.py; facts from other papers, each with URL, version and quote).
Groups of im_within: a = F18's 48 rollout-trained simulators, b = X5's 40 rollout-trained ones (rho within each split,
averaged), c = F18's 40 compressed copies, d = all 97 of F18.
"""
import json
import os

import common as C

WITHIN = os.path.join(C.RESULTS, "im_within.json")
FACTS = os.path.join(C.RESULTS, "external_facts.json")
BUDGET = os.path.join(C.PDM, "rwm_repro", "results", "task_c2_data_budget.json")
KEYS = {"E1": "E1", "E8": "E8", "lam1": "lam1", "rJ": "rJ", "lam1_pol": "lam1pol", "rJ_pol": "rJpol",
        "negret": "negret"}


def _rel(path):
    return os.path.relpath(path, os.path.dirname(C.AIR))


def _ld(path):
    with open(path) as fh:
        return json.load(fh)


def add(put, sig, sgn, pct, cib, dp, load, rel):
    W, w = _ld(WITHIN), _rel(WITHIN)
    X, x = _ld(FACTS), _rel(FACTS)
    r2 = lambda v: dp(v, 2, signed=True)                                                 # noqa: E731

    # ---- I1: how each check ranks simulators trained the same way (exploratory)
    for g in ("a", "b", "c", "d"):
        G = W[g]
        put(f"im_{g}_n", str(G["n_models"]), w, f"{g}.n_models")
        for c, k in KEYS.items():
            if c not in G["checks"]:
                continue
            v = G["checks"][c]
            put(f"im_{g}_{k}", r2(v["rho"]), w, f"{g}.checks.{c}.rho")
            put(f"im_{g}_{k}_ci", cib(*v["ci95"], f=r2), w, f"{g}.checks.{c}.ci95")
        for d, k in (("lam1_minus_E8", "dlam1"), ("rJ_minus_E8", "drJ")):
            v = G["diffs"][d]
            put(f"im_{g}_{k}", r2(v["delta"]), w, f"{g}.diffs.{d}.delta")
            put(f"im_{g}_{k}_ci", cib(*v["ci95"], f=r2), w, f"{g}.diffs.{d}.ci95")

    # ---- I1: facts from other papers (value, URL, version and quote are in external_facts.json)
    def fact(key, name, value=None):
        f = X[name]
        put(key, str(f["value"]) if value is None else value, x, f"{name}.value ({f['paper']}, {f['section']}, {f['version']})")
    fact("ext_rwm_imag_steps", "rwm_imag_steps")
    fact("ext_rwm_imag_envs", "rwm_imag_envs")
    fact("ext_rwm_history", "rwm_history")
    fact("ext_rwm_forecast", "rwm_forecast")
    fact("ext_rwmu_imag_steps", "rwmu_imag_steps")
    fact("ext_rwm_pretrain", "rwm_pretrain_transitions", "%d million" % (X["rwm_pretrain_transitions"]["value"] // 10 ** 6))
    fact("ext_ikce_n_models", "ikce_n_models")
    fact("ext_ikce_task", "ikce_task")
    assert X["ikce_ranks_models"]["value"] is False
    # our data as a fraction of RWM's pretraining data: 7991 transitions of ours over the 6 million of RWM's Table 1
    ours = _ld(BUDGET)["ours"]["distinct_transitions"]
    ref = X["rwm_pretrain_transitions"]["value"]
    assert _ld(BUDGET)["reference"]["value"] == ref
    assert abs(ours / ref - _ld(BUDGET)["ours_as_fraction_of_reference"]) < 1e-5     # agrees with the stored fraction
    put("ext_ours_transitions", f"{ours:,}".replace(",", "{,}"), _rel(BUDGET), "ours.distinct_transitions")
    frac = pct(ours / ref, 2)
    put("ext_budget_fraction", frac, _rel(BUDGET) + " and " + x, "ours.distinct_transitions / rwm_pretrain_transitions.value")

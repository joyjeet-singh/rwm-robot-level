"""The checks of the Step 2 fallback pool, computed once, before registration and before any pool policy exists.

    ./py scripts/s2/fallback_checks.py        # -> results/s2/checks.json, results/s2/checks_raw.npz (never overwritten)

Why this exists. Gate 2's P2 failed, so the plan's fallback pool is registered: the released five-network model and
its eight compressed copies (scripts/compress.py VARIANTS). The released model's checks are stored
(results/model_table.json, results/s1_models.json), but its compressed copies were never built in earlier rounds,
so none of their checks is stored. This script computes all nine with the registered functions only, imported,
never copied:
  * recorded commands: compute.build and compute.per_model (E1, E8, E24, E100, lambda_1, r_J), as compute.py did
    for the released model;
  * policy commands (the authors' actor in closed loop): s1_rollout.score_gru(items=("pol", "lam_pol"), p=1),
    as s1_eval.phase_cl did for the released model; a model's value is s1_eval.model_mean (one non-finite
    trajectory makes it +inf);
  * the arena is the released model's own: all 20 recorded trajectories ("all"), as for its stored values. The
    released model was trained on all ten episodes, so no held-out episode exists for this pool.
The released model is recomputed too (one code path for all nine), and its recomputed values are compared with
the stored ones (validation; written into the result). No physics number is read or used here.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402
import compress as K  # noqa: E402
import compute  # noqa: E402
import s1_policy as SP  # noqa: E402
import s1_rollout as S  # noqa: E402
from s1_eval import model_mean  # noqa: E402

torch.set_num_threads(2)
OUT = os.path.join(C.RESULTS, "s2", "checks.json")
RAW = os.path.join(C.RESULTS, "s2", "checks_raw.npz")
HW = compute.HW
SIMS = ["REL"] + [f"REL__{v}" for v in K.VARIANTS]


def variant_of(sim):
    return None if sim == "REL" else sim.split("__", 1)[1]


def main(out=OUT, raw_path=RAW, sims=SIMS):
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; result files are never overwritten")
    t_all = time.time()
    ar, sc = compute.setup_arena("all")
    actor = SP.load_actor()
    raw, models = {}, {}
    for sim in sims:
        t0 = time.time()
        v = variant_of(sim)
        m, info = compute.build(C.W_RELEASED, 5, v)
        rec = compute.per_model(m, ar, sc)                                  # recorded commands
        m2, _ = compute.build(C.W_RELEASED, 5, v)                           # fresh copy: lyapunov changes dtype
        pol = S.score_gru(m2, ar, sc, items=("pol", "lam_pol"), p=1, actor=actor)
        pol.pop("seconds")
        e = {"key": sim, "variant": v, "variant_info": info, "ensemble": 5, "arena": "all",
             "n_traj": int(len(ar["starts"])), "finite_recorded": rec["finite"]}
        for h in HW:
            e[f"E{h}"] = float(np.mean(rec[f"E{h}"]))
        e["E368"] = float(np.mean(rec["err"]))
        e["lam1"] = float(np.mean(rec["lam"][:, 0]))
        e["rJ"] = float(np.mean(rec["rJ"]))
        e["lam1_pol"] = model_mean(pol["lam1_pol"])
        e["rJ_pol"] = model_mean(pol["rJ_pol"])
        e["n_div_pol"] = int(np.sum(np.asarray(pol["div_pol"]) > 0))
        e["ret_authors_actor"] = float(np.mean(pol["ret"]))
        e["seconds"] = time.time() - t0
        models[sim] = e
        for k in ("lam", "rJ", "err") + tuple(f"E{h}" for h in HW):
            raw[f"{k}_{sim}"] = rec[k]
        for k in ("lam1_pol", "rJ_pol", "div_pol", "ret"):
            raw[f"{k}_{sim}"] = np.asarray(pol[k])
        print(f"{sim}: E1 {e['E1']:.4f} E8 {e['E8']:.4f} E100 {e['E100']:.4f} lam1 {e['lam1']:.4f} "
              f"rJ {e['rJ']:.4f} lam1_pol {e['lam1_pol']:.4f} rJ_pol {e['rJ_pol']:.4f} "
              f"div {e['n_div_pol']} ({e['seconds']:.0f}s)", flush=True)

    # validation: the released model's recomputed values against the stored ones
    val = {}
    if "REL" in models:
        st = json.load(open(os.path.join(C.RESULTS, "model_table.json")))["REL"][0]
        s1 = json.load(open(os.path.join(C.RESULTS, "s1_models.json")))["models"]["REL"]["REL"]
        pairs = [(k, st[k]) for k in ("E1", "E8", "E24", "E100", "E368", "lam1", "rJ")]
        pairs += [("lam1_pol", s1["lam1_pol"]), ("rJ_pol", s1["rJ_pol"])]
        for k, stored in pairs:
            new = models["REL"][k]
            val[k] = {"stored": stored, "recomputed": new,
                      "rel_diff": abs(new - stored) / max(abs(stored), 1e-12)}
    res = {
        "what": "Step 2 fallback pool: checks of the released model and its eight compressed copies, computed "
                "by the registered functions before registration; no physics number used",
        "weights": os.path.relpath(C.W_RELEASED, C.PDM), "weights_sha256": C.sha256(C.W_RELEASED),
        "arena": "all (20 recorded trajectories; the released model was trained on all ten episodes)",
        "policy_commands": "the authors' actor in closed loop (s1_rollout.score_gru items pol, lam_pol, p=1)",
        "sims": list(sims), "models": models, "validation_REL_vs_stored": val,
        "torch": torch.__version__, "threads": torch.get_num_threads(),
        "seconds_total": time.time() - t_all, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(raw_path, **raw)
    tmp = out + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1, default=float)
    os.replace(tmp, out)
    print("validation (rel. diff):", {k: f"{v['rel_diff']:.1e}" for k, v in val.items()})
    return res


if __name__ == "__main__":
    main()

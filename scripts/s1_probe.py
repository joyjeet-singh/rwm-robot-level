"""
Step 1, S1 probe: seconds per model for each item of s1_rollout.score_gru, on random-weight GRUs of the release's
architecture (width 256; one head, as most pool models, and five heads, as the release and the A5 models), on F18's
arena (4 trajectories; every X5 arena also has 4). Two threads. Writes results/s1_probe.json.
No trained simulator is used.
"""
import json
import os
import platform
import time

import numpy as np
import torch

import common as C
import compute
import s1_policy as SP
import s1_rollout as S

torch.set_num_threads(2)
REPS = 2


def main():
    ar, sc = compute.setup_arena((1, 8))
    t0 = time.time()
    actor = SP.load_actor()
    t_actor = time.time() - t0
    per = {}
    for ens in (1, 5):
        torch.manual_seed(ens)
        m = C.build_width(256, ensemble_size=ens).eval()
        runs = []
        for _ in range(REPS):
            t0 = time.time()
            out = S.score_gru(m, ar, sc, actor=actor)
            runs.append(dict(out["seconds"], total=time.time() - t0))
        per[f"ensemble_{ens}"] = {k: float(min(r[k] for r in runs)) for k in runs[0]}
        per[f"ensemble_{ens}"]["diverged_trajectories"] = {k: int((out[k] > 0).sum())
                                                            for k in ("div_pol", "div_cl", "div_noise")}
    meta = json.load(open(os.path.join(C.RESULTS, "models.json")))
    proj = {}
    for tag in ("F18", "X5"):
        ens = [m["ensemble"] for m in meta[tag]]
        n5 = sum(e == 5 for e in ens)
        n1 = len(ens) - n5
        sec = n1 * per["ensemble_1"]["total"] + n5 * per["ensemble_5"]["total"]
        proj[tag] = {"models": len(ens), "one_head": n1, "five_heads": n5, "hours": sec / 3600.0}
    out = {"purpose": "S1 timing probe of s1_rollout.score_gru (STEP1_PLAN.md S1); random-weight models only",
           "arena": "F18 (held-out episodes 1, 8), 4 trajectories x 368 forecast steps",
           "threads": torch.get_num_threads(), "lyapunov_exponents_p": 1, "repetitions_min_of": REPS,
           "seconds_actor_load": t_actor, "seconds_per_model": per, "projected": proj,
           "note": ("Projection assumes every model costs what a width-256 GRU costs; width-124 and structurally "
                    "pruned variants are cheaper, so it is an upper bound for those. Trained models may diverge "
                    "less or more than random ones, which changes nothing in the cost."),
           "machine": {"platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__}}
    json.dump(out, open(os.path.join(C.RESULTS, "s1_probe.json"), "w"), indent=1)
    print(json.dumps({"per": per, "proj": proj}, indent=1))


if __name__ == "__main__":
    main()

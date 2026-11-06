"""
Step 1, S2 probe (STEP1_PLAN.md S2): adds key "S2" to results/s1_probe.json.
  - seconds of s1_arch.score_arch per model, on random weights of every (class, spec) in the pool, F18's arena, 2 threads;
    seconds to load one existing run's final weights per class (loading only: nothing is scored);
  - seconds per training iteration of s1_train_baseline (20 iterations, 2 threads) for tf_s7 and ar_s7 of each class,
    and of two RSSM tf_s7 runs side by side at 1 thread each (two processes);
  - projections: minutes to score all runs (26 existing, 12 new) and hours to train the 12 new runs, one at a time and
    two at a time.
Training here writes only to .cache/s2_probe/. No trained baseline is scored.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter

import torch

import common as C
import compute
import s1_arch as A
import s1_train_baseline as TB

torch.set_num_threads(2)
CACHE = os.path.join(C.AIR, ".cache", "s2_probe")
ITERS = 20
NEW = [(a, r, "s7", s) for a in A.ARCHES for r in ("tf", "ar") for s in (3, 4)]          # the 12 new runs
SIDE = r'''
import json, sys, torch
import s1_train_baseline as TB
rec = TB.train("rssm", "tf", "s7", 0, iters={iters}, out_dir=sys.argv[1], threads=1)
print("S_PER_ITER", rec["s_per_iter"])
'''


def existing():
    out = []
    for d in sorted(os.listdir(os.path.join(C.RWM, "runs"))):
        if d.startswith("baseline_") and "_seed" in d:
            _, arch, regime, spec, _ = d.split("_")
            out.append((arch, regime, spec, os.path.join(C.RWM, "runs", d)))
    return out


def main():
    shutil.rmtree(CACHE, ignore_errors=True)
    ar, sc = compute.setup_arena((1, 8))
    runs = existing()
    assert len(runs) == 26, len(runs)
    kinds = sorted({(a, s) for a, _, s, _ in runs} | {(a, s) for a, _, s, _ in NEW})
    score = {}
    for arch, spec in kinds:
        torch.manual_seed(0)
        m = A.build(arch, spec).eval()
        t0 = time.time()
        out = A.score_arch(None, ar, sc, model=m)
        score[f"{arch}_{spec}"] = dict(out["seconds"], total=time.time() - t0)
        print("score", arch, spec, f"{time.time() - t0:.1f}s", flush=True)
    load = {}
    for arch in A.ARCHES:
        rd = next(d for a, r, s, d in runs if a == arch and r == "tf" and s == "s7")
        t0 = time.time()
        A.load(rd)
        load[arch] = time.time() - t0
    count = Counter(f"{a}_{s}" for a, _, s, _ in runs) + Counter(f"{a}_{s}" for a, _, s, _ in NEW)
    score_min = sum(n * (score[k]["total"] + load[k.split("_")[0]]) for k, n in count.items()) / 60.0

    train = {}
    for arch in A.ARCHES:
        for regime in ("tf", "ar"):
            rec = TB.train(arch, regime, "s7", 0, iters=ITERS, out_dir=os.path.join(CACHE, "alone"), threads=2)
            train[f"{arch}_{regime}_s7"] = rec["s_per_iter"]
    torch.set_num_threads(2)
    env = dict(os.environ)
    procs = [subprocess.Popen([sys.executable, "-c", SIDE.format(iters=ITERS), os.path.join(CACHE, f"side{i}")],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
             for i in range(2)]
    side = []
    for p in procs:
        o, _ = p.communicate()
        side.append(float([ln for ln in o.splitlines() if ln.startswith("S_PER_ITER")][0].split()[1]))
    slow = max(side) / train["rssm_tf_s7"]                  # side-by-side seconds per iteration over alone
    alone_h = sum(2500 * train[f"{a}_{r}_s7"] for a, r, _, _ in NEW) / 3600.0
    two_h = alone_h * slow / 2.0
    rec_times = {}
    for a, r, s, _ in runs:
        if s == "s7":
            j = json.load(open(os.path.join(C.RWM, "results", f"baseline_run_{a}_{r}_s7_seed0.json")))
            rec_times[f"{a}_{r}_s7"] = j["s_per_iter"]
    res = {"purpose": "S2 timing probe of s1_arch.score_arch and s1_train_baseline (STEP1_PLAN.md S2); "
                      "random weights for scoring; training written to .cache only",
           "threads": 2, "arena": "F18 (held-out episodes 1, 8), 4 trajectories",
           "score_seconds_per_model_random_weights": score,
           "load_seconds_per_run": load,
           "runs_to_score": dict(count), "projected_minutes_score_all_38": score_min,
           "train_seconds_per_iter_2_threads": train,
           "train_seconds_per_iter_rssm_tf_two_side_by_side_1_thread_each": side,
           "side_by_side_slowdown": slow,
           "recorded_seconds_per_iter_reproduction_seed0": rec_times,
           "projected_hours_train_12_new": {"one_at_a_time_2_threads": alone_h,
                                            "two_at_a_time_1_thread_each": two_h},
           "note": ("Two-at-a-time hours assume every class slows down as the RSSM did side by side (the RSSM is "
                    "about two thirds of the total). Scoring minutes include loading the weights and use random "
                    "weights; a trained model costs the same per step.")}
    p = os.path.join(C.RESULTS, "s1_probe.json")
    d = json.load(open(p))
    assert "S2" not in d
    d["S2"] = res
    json.dump(d, open(p, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("projected_minutes_score_all_38", "train_seconds_per_iter_2_threads",
                                          "train_seconds_per_iter_rssm_tf_two_side_by_side_1_thread_each",
                                          "projected_hours_train_12_new")}, indent=1))


if __name__ == "__main__":
    main()

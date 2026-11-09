"""Option C (C1): probes for the job design, written once to results/c/probe.json.

    ./py scripts/c/probe_c1.py

  1. transitions per second per worker, two generator workers at once, on the real shard plan (into .cache/ only);
  2. seconds per training step for five networks, N = 8 and N = 1, two trainings at once, on a 1M-row stand-in rung:
     the generated training shards tiled to 1M rows (the full 1M rung does not exist yet; the step's arithmetic and the
     memory-mapped reading do not depend on the data's values). Held-out sets are not used;
  3. peak memory, and S by the design's rule: S = max(2500, most steps of the slowest setting (N = 8) that fit in 8 hours
     with two simulators training at once). If two trainings would exceed 6 GB, the jobs train one at a time on 4 threads.
"""
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AIR = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(AIR, "results", "c", "probe.json")
PROBE = os.path.join(AIR, ".cache", "probe_data")
GEN_SECONDS = 75
STEPS_N8, STEPS_N1 = 12, 15
HOURS, S_MIN, MEM_GB = 8, 2500, 6.0
RUNG_6M_GB = 6_000_000 * 68 * 4 / (1 << 30)


def spawn(cmd):
    return subprocess.Popen(cmd, cwd=AIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def finish(procs):
    out = []
    for p in procs:
        text, _ = p.communicate()
        if p.returncode != 0:
            raise SystemExit("probe process failed:\n" + text[-2000:])
        out.append(json.loads(text.strip().splitlines()[-1]))
    return out


def gen():
    os.makedirs(PROBE, exist_ok=True)
    base = ["./py-mj", "scripts/c/gen_data.py", "--nworkers", "2", "--data-dir", PROBE, "--no-progress",
            "--time-limit", str(GEN_SECONDS)]
    res = finish([spawn(base + ["--worker", str(k)]) for k in (0, 1)])
    return [r["rows"] / r["seconds"] for r in res], res


def build_standin():
    """Tile the probe's training shards to at least 1M rows, each copy of an episode a new episode."""
    files = sorted(glob.glob(os.path.join(PROBE, "train_s*.npy")))
    a = np.concatenate([np.load(f) for f in files])
    copies = int(np.ceil(1_000_000 / len(a)))
    path = os.path.join(PROBE, "rung_standin.npy")
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(a) * copies, 68))
    ep = np.zeros(len(a) * copies, dtype=np.int64)
    src_ep = a[:, 66].astype(np.int64)
    for k in range(copies):
        out[k * len(a):(k + 1) * len(a)] = a
        ep[k * len(a):(k + 1) * len(a)] = src_ep + k * 100_000_000
    out.flush()
    np.save(os.path.join(PROBE, "rung_standin.ep.npy"), ep)
    np.save(os.path.join(PROBE, "rung_standin.fall.npy"), np.asarray(out[:, 65]) > 0.5)
    stats = {"rows": int(len(ep)), "episodes": int(len(np.unique(ep))),
             "state_mean": a[:, :45].mean(0).astype(float).tolist(), "state_std": a[:, :45].std(0).astype(float).tolist()}
    json.dump(stats, open(os.path.join(PROBE, "rung_standin.stats.json"), "w"))
    return {"rows": int(len(ep)), "source_rows": int(len(a)), "copies": copies, "source_shards": len(files)}


def train(N, steps):
    base = ["./py", "scripts/c/train_sim.py", "--rung", "standin", "--N", str(N), "--seed", "0", "--threads", "2",
            "--data-dir", PROBE, "--probe", str(steps), "--no-progress"]
    return finish([spawn(base), spawn(base)])


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists; result files are never overwritten")
    t0 = time.time()
    rates, gen_res = gen()
    standin = build_standin()
    n8 = train(8, STEPS_N8)
    n1 = train(1, STEPS_N1)
    s_n8, s_n1 = float(np.mean([r["seconds_per_step"] for r in n8])), float(np.mean([r["seconds_per_step"] for r in n1]))
    peak_each = max(r["peak_rss_gb"] for r in n8 + n1)
    two_on_1m = float(sum(max(a["peak_rss_gb"], b["peak_rss_gb"]) for a, b in zip(n8, n1)))
    # Memory-mapped pages of a larger rung are also counted in the process's resident size, up to the file's size.
    bound_6m = two_on_1m + 2 * max(0.0, RUNG_6M_GB - 1_000_000 * 68 * 4 / (1 << 30))
    parallel = bound_6m <= MEM_GB
    steps_8h = int(HOURS * 3600 / s_n8)
    S = max(S_MIN, steps_8h)
    rec = {
        "what": "Option C job-design probes (hand-built calibrated MuJoCo ANYmal D, fitted PD motor; not Isaac Lab, not "
                "hardware). Stand-in rung: generated training shards tiled to 1M rows.",
        "generation": {"transitions_per_second_per_worker": rates, "workers": gen_res,
                       "hours_for_6M_two_workers": 6_000_000 / sum(rates) / 3600},
        "training": {"standin": standin, "batch": 256, "threads_each": 2, "two_at_once": True,
                     "N8": n8, "N1": n1, "seconds_per_step_N8": s_n8, "seconds_per_step_N1": s_n1},
        "memory": {"peak_rss_gb_one_training": peak_each, "peak_rss_gb_two_trainings_1M_rung": two_on_1m,
                   "conservative_bound_two_trainings_6M_rung_gb": bound_6m, "limit_gb": MEM_GB},
        "S_rule": f"S = max({S_MIN}, int({HOURS} h / seconds per step of N = 8 with two trainings at once))",
        "steps_in_8h_N8": steps_8h, "S": S, "parallel_two_trainings": parallel,
        "job_layout": "two trainings at once, 2 threads each" if parallel else "one training at a time, 4 threads",
        "probe_seconds": time.time() - t0,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(tmp, OUT)
    print(json.dumps({k: rec[k] for k in ("S", "steps_in_8h_N8", "parallel_two_trainings", "probe_seconds")}))
    print(json.dumps({"rates": rates, "N8": s_n8, "N1": s_n1, "mem_bound": bound_6m, "peak": peak_each}))


if __name__ == "__main__":
    main()

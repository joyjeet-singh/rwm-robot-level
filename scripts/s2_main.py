"""
Step 2 main job (prereg/PREREG_S2.md): train, export and measure every registered policy. Run by step2/jobs/main.zsh.

    ./py scripts/s2_main.py --workers 2 --threads 2      # resumable; skips finished work
    ./py scripts/s2_main.py --plan                       # print the 18 policies and their state; do nothing

The pool and seeds are s2_stats.POOL x s2_stats.SEEDS (nine simulators x seeds 1 and 2 = 18 policies), seed 1 of
every simulator first. Each policy, in turn, by the first free worker:
  1. scripts/s2/train_policy.py --sim <sim> --seed <s> --route rebuilt   (./py; 500 iterations, 4,096 robots;
     every simulator, the released model too, through common.load and the lite adapter; no holdout, as the
     released model was trained on all ten episodes); skipped by train_policy itself if done;
  2. scripts/s2/evaluate.py --policy <sim>__p<s>   (./py-mj; the fixed protocol, 10 schedules x 8 starts in the
     calibrated MuJoCo model); skipped if results/s2/eval/<policy>.json exists.
A failed step is retried once (training restarts from iteration 0). A step that fails twice with non-finite
values in its log (nan, inf, "finite") was failed by its simulator: results/s2/failed/<policy>.json (cause
"simulator"), which s2_stats scores as falls 1 and the lowest measured return (prereg/PREREG_S2.md section 7).
Any other double failure (killed, out of memory, a code error) is not a result: it is written to
.cache/s2_failed/<policy>.json, the job ends with exit 1, and a re-run retries it. Progress: results/s2/progress.json, from
train_policy (every 25 iterations), evaluate (every 5 episodes) and here (each policy's start and end). State:
.cache/s2_main_status.json. When all 18 are measured or failed by their simulator: results/s2_main.json (once,
never overwritten).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import s2_stats as SS  # noqa: E402

PY = os.path.join(AIR, "py")
PYMJ = os.path.join(AIR, "py-mj")
TRAIN = os.path.join(AIR, "results", "s2", "train")
EVAL = os.path.join(AIR, "results", "s2", "eval")
PROGRESS = os.path.join(AIR, "results", "s2", "progress.json")
STATUS = os.path.join(AIR, ".cache", "s2_main_status.json")
LOGS = os.path.join(AIR, ".cache", "logs")
OUT = os.path.join(AIR, "results", "s2_main.json")
FAILED = os.path.join(AIR, "results", "s2", "failed")
FAILED_OTHER = os.path.join(AIR, ".cache", "s2_failed")
NONFINITE = re.compile(r"\bnan\b|\binf\b|non-?finite|not finite|isfinite", re.IGNORECASE)
DONE = ("measured", "failed: simulator")
_lock = threading.Lock()


def plan():
    """[(sim, seed, policy)], seed-major so that every simulator has one policy before any has two."""
    return [(sim, s, SS.policy(sim, s)) for s in SS.SEEDS for sim in SS.POOL]


def state(pol):
    if os.path.exists(os.path.join(EVAL, pol + ".json")):
        return "measured"
    if os.path.exists(os.path.join(FAILED, pol + ".json")):
        return "failed: simulator"
    if os.path.exists(os.path.join(TRAIN, pol + ".json")):
        return "trained"
    return "todo"


def progress(step, done, total):
    rec = {"job": "main", "step": step, "done": int(done), "total": int(total),
           "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with _lock:
        os.makedirs(os.path.dirname(PROGRESS), exist_ok=True)
        tmp = f"{PROGRESS}.{os.getpid()}.main.tmp"
        json.dump(rec, open(tmp, "w"))
        os.replace(tmp, PROGRESS)


def env_threads(t):
    e = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "S2_THREADS"):
        e[k] = str(t)
    return e


def run_step(cmd, log, threads):
    with open(log, "a") as fh:
        fh.write(f"\n[{time.strftime('%F %T')}] {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.run(cmd, cwd=AIR, stdout=fh, stderr=subprocess.STDOUT, env=env_threads(threads)).returncode


def do_policy(sim, seed, pol, threads, runner=run_step):
    """Train (if needed) and measure (if needed) one policy; each step tried at most twice. Returns a status."""
    os.makedirs(LOGS, exist_ok=True)
    if state(pol) in DONE:
        return state(pol)
    steps = [("train", [PY, os.path.join(HERE, "s2", "train_policy.py"), "--sim", sim, "--seed", str(seed),
                        "--threads", str(threads), "--route", "rebuilt", "--job", "main"]),
             ("evaluate", [PYMJ, os.path.join(HERE, "s2", "evaluate.py"), "--policy", pol, "--job", "main"])]
    for name, cmd in steps:
        if name == "train" and state(pol) != "todo":
            continue
        log = os.path.join(LOGS, f"s2_main_{name}_{pol}.log")
        for attempt in (1, 2):
            rc = runner(cmd, log, threads)
            if rc == 0:
                break
        else:
            return record_failure(pol, name, rc, log)
    return "measured" if state(pol) == "measured" else "failed: no result file"


def record_failure(pol, step, rc, log):
    """A step failed twice. Non-finite values in the log's tail and an ordinary exit: failed by the simulator."""
    tail = open(log).read()[-4000:] if os.path.exists(log) else ""
    tb = tail[tail.rfind("Traceback"):] if "Traceback" in tail else tail[-1000:]     # the error itself
    cause = "simulator" if (rc is not None and rc > 0 and NONFINITE.search(tb)) else "other"
    d = FAILED if cause == "simulator" else FAILED_OTHER
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, pol + ".json")
    if not (cause == "simulator" and os.path.exists(path)):
        json.dump({"policy": pol, "step": step, "returncode": rc, "cause": cause, "log_tail": tail[-1500:],
                   "time": time.strftime("%FT%T")}, open(path, "w"), indent=1)
    return "failed: simulator" if cause == "simulator" else f"failed at {step}"


def run_all(workers, threads, runner=run_step, items=None):
    items = items if items is not None else plan()
    queue = [it for it in items if state(it[2]) not in DONE]
    status = {it[2]: state(it[2]) for it in items}
    total = len(items)
    t0 = time.time()

    def save():
        with _lock:
            os.makedirs(os.path.dirname(STATUS), exist_ok=True)
            json.dump({"status": status, "seconds": time.time() - t0, "time": time.strftime("%FT%T")},
                      open(STATUS, "w"), indent=1)

    def worker():
        while True:
            with _lock:
                if not queue:
                    return
                sim, seed, pol = queue.pop(0)
            progress(f"start {pol}", sum(v == "measured" for v in status.values()), total)
            st = do_policy(sim, seed, pol, threads, runner)
            with _lock:
                status[pol] = st
            save()
            progress(f"{st} {pol}", sum(v == "measured" for v in status.values()), total)

    save()
    ts = [threading.Thread(target=worker) for _ in range(max(1, workers))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return status


def summary(status, workers, threads, seconds):
    import common as C
    pols = []
    for sim, seed, pol in plan():
        tp, ep = os.path.join(TRAIN, pol + ".json"), os.path.join(EVAL, pol + ".json")
        tr = json.load(open(tp)) if os.path.exists(tp) else {}
        ev = json.load(open(ep))["summary"] if os.path.exists(ep) else {}
        pols.append({"policy": pol, "sim": sim, "seed": seed, "status": status.get(pol), "route": tr.get("route"),
                     "train_sha256": C.sha256(tp) if tr else None, "eval_sha256": C.sha256(ep) if ev else None,
                     "actor_sha256": tr.get("actor_sha256"), "train_wall_seconds": tr.get("wall_seconds"),
                     "losses_all_finite": tr.get("losses_all_finite"), "ret": ev.get("ret"),
                     "fall_rate": ev.get("fall_rate")})
    return {"what": "Step 2 main job: every registered policy trained and measured (prereg/PREREG_S2.md); the "
                    "registered analysis is scripts/s2_stats.py, run by M5",
            "policies": pols, "workers": workers, "threads_per_worker": threads, "seconds_this_run": seconds,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args(argv)
    if a.plan:
        for sim, seed, pol in plan():
            print(f"{pol:22s} {state(pol)}")
        return 0
    cpath = os.path.join(AIR, "results", "s2", "checks.json")
    if not os.path.exists(cpath) or SS.C.sha256(cpath) != SS.CHECKS_SHA256:
        raise SystemExit("results/s2/checks.json is missing or differs from the registered file")
    t0 = time.time()
    status = run_all(a.workers, a.threads)
    bad = {p: s for p, s in status.items() if s not in DONE}
    print(json.dumps(status, indent=1))
    if bad:
        print(f"[s2_main] {len(bad)} policies not measured: {bad}")
        return 1
    if not os.path.exists(OUT):
        tmp = OUT + ".tmp"
        json.dump(summary(status, a.workers, a.threads, time.time() - t0), open(tmp, "w"), indent=1)
        os.replace(tmp, OUT)
    progress("done", len(status), len(status))
    nf = sum(s == "failed: simulator" for s in status.values())
    print(f"[s2_main] all {len(status)} policies done ({nf} failed by their simulator); {os.path.relpath(OUT, AIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

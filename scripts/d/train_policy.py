"""Option D (D1): train one policy in a ladder simulator, or in one of Step 2's nine, by Step 2's recipe.  ./py

    ./py scripts/d/train_policy.py --sim W_s0_d80 --seed 1      # -> runs/d/policies/W_s0_d80__p1/actor.pt,
                                                                 #    results/d/train/W_s0_d80__p1.json
    ./py scripts/d/train_policy.py --sim REL --seed 1 --name _test_d1 --iterations 3 --num_envs 256 --threads 1
    ./py scripts/d/train_policy.py --sim W_s0_d80 --seed 2 --mark-failed     # after two equal crashes and a passed
                                                                              # crash guard (scripts/d/ladder.py)

OPTIOND_PLAN.md, Stage 2, "Policies: Step 2's recipe, with one stop rule":
  * Step 2's recipe exactly as scripts/s2/train_policy.py runs it with `--route rebuilt` for a simulator with no
    hold-out: scripts/lite/run_lite.py unchanged, 500 iterations, 4,096 robots, the torch seed, 2 threads, the lite
    config's uncertainty weight, start states from the authors' recorded data with no episode held out; no noise
    floor, no fall cost. The one swap: the simulator comes from scripts/d/sims.load (ladder simulators, `REL` and its
    eight compressed copies), wrapped by the lite adapter. It is built before run_lite sets its seed, as Step 2's
    wrapper builds it, so the random numbers are the same (scripts/d/test_d1.py checks the logs to 1e-6);
  * the stop rule (collapse in imagination), through the lite log hook: from iteration 50 on, training stops once
    the mean imagined episode length (`Train/mean_episode_length`) has stayed under 128 steps (half the limit) for 50
    iterations in a row; iterations up to 50 do not count, so the earliest stop is at iteration 100. An iteration
    with no logged length (no imagined episode ended in it) breaks the run of short iterations (optd/DECISIONS.md
    D0-3). The stopped policy's actor weights at that iteration are exported; it is measured and counts as failed;
  * a training run that fails by its simulator: a loss or step reward that is not finite (status
    `failed_by_simulator`; walking score 0 and fall rate 1, not measured), or the same crash inside the lite code
    twice, after the crash guard ran clean on `REL` (scripts/d/ladder.py drives the attempts and the guard; a crash
    here writes runs/d/policies/<policy>/crash_<attempt>.json and exits with code 4; `--mark-failed` then writes the
    record);
  * outputs: the actor runs/d/policies/<policy>/actor.pt (the layout s1_policy.load_actor reads), the lite run
    runs/lite/d_<policy>/, the record results/d/train/<policy>.json (written once; a finished policy is skipped).
    Policy names are `<simulator>__p<seed>`. An interrupted run restarts from iteration 0.
  * Before prereg/PREREG_D.md is committed, only `_test*` names with at most 3 iterations are accepted. `_test*` names
    may use other sizes and write their record under .cache/d_train/.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
AIR = os.path.dirname(SCRIPTS)
sys.dont_write_bytecode = True
for _q in (SCRIPTS, HERE):
    if _q not in sys.path:
        sys.path.append(_q)

ITERATIONS = 500
NUM_ENVS = 4096
SEEDS = (1, 2, 3)
STOP = {"after": 50, "run": 50, "threshold": 128.0}     # OPTIOND_PLAN.md: the stop rule
LAST = 25                                               # Step 2's imagined-return window
CRASH_EXIT = 4
POLICIES = os.path.join(AIR, "runs", "d", "policies")
RES = os.path.join(AIR, "results", "d", "train")
TEST_RES = os.path.join(AIR, ".cache", "d_train")


# ------------------------------------------------------------------ pure parts (imported by tests and the driver)
def policy_name(sim, seed):
    return f"{sim}__p{seed}"


def stop_at(lengths, after=STOP["after"], run=STOP["run"], threshold=STOP["threshold"]):
    """The iteration (1-based) at which the stop rule fires, or None.

    lengths[i] is iteration i + 1's mean imagined episode length (None where the lite code logged none). Iterations
    1..after do not count; the rule fires at the first n whose last `run` iterations all have a length under
    `threshold`. A missing value breaks the run. Earliest stop: after + run = 100."""
    count = 0
    for i, v in enumerate(lengths):
        n = i + 1
        if n <= after:
            continue
        count = count + 1 if (v is not None and v < threshold) else 0
        if count >= run:
            return n
    return None


def lengths_from_metrics(path):
    """Per-iteration Train/mean_episode_length from a lite metrics.jsonl (line 0 is the data log, line n is
    iteration n), None where none was logged."""
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        lines = [json.loads(x) for x in fh if x.strip()]
    return [d.get("Train/mean_episode_length") for d in lines[1:]
            if "Train/mean_reward" in d or "Train/mean_episode_length" in d or "Imagination/uncertainty" in d]


def registered():
    """True once prereg/PREREG_D.md is in a commit on this branch."""
    r = subprocess.run(["git", "log", "-1", "--format=%H", "--", "prereg/PREREG_D.md"], cwd=AIR,
                       capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def paths(name):
    test = name.startswith("_test")
    out_dir = os.path.join(POLICIES, name)
    return {"dir": out_dir, "actor": os.path.join(out_dir, "actor.pt"),
            "res": os.path.join(TEST_RES if test else RES, name + ".json"),
            "lite": os.path.join(AIR, "runs", "lite", "d_" + name), "lite_name": "d_" + name}


def done_record(name):
    """The policy's record if it is finished and consistent, else None."""
    p = paths(name)
    if not os.path.exists(p["res"]):
        return None
    with open(p["res"]) as fh:
        rec = json.load(fh)
    if rec.get("actor_sha256"):
        import common as C
        if not os.path.exists(p["actor"]) or C.sha256(p["actor"]) != rec["actor_sha256"]:
            raise SystemExit(f"{p['actor']} does not match {p['res']}")
    return rec


def _write_once(path, rec, test):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and not test:
        raise SystemExit(f"{path} exists; not overwriting a result file")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    os.replace(tmp, path)


def _progress(job, step, done=None, total=None):
    sys.path.insert(0, HERE)
    import dprogress
    dprogress.write(job, step, done, total)


class Stopped(Exception):
    """Raised from the lite log hook when the stop rule fires."""

    def __init__(self, n):
        super().__init__(f"stopped by the stop rule at iteration {n}")
        self.n = n


# ------------------------------------------------------------------ one training attempt
def train(sim, seed, name, iterations, num_envs, threads, job, attempt, stop=None):
    """One uninterrupted, seeded training run (Step 2's recipe plus the stop rule). Returns the record."""
    sys.path.insert(0, os.path.join(SCRIPTS, "lite"))
    import run_lite as RL               # sets its own sys.path, torch.load and the wandb stub (as Step 2's wrapper)
    import torch
    for q in (SCRIPTS, HERE):
        if q not in sys.path:
            sys.path.append(q)
    import importlib.util               # Step 2's wrapper (export_actor), by path: this file has the same name
    spec = importlib.util.spec_from_file_location("s2_train_policy", os.path.join(SCRIPTS, "s2", "train_policy.py"))
    TP = importlib.util.module_from_spec(spec)
    sys.modules["s2_train_policy"] = TP
    spec.loader.exec_module(TP)
    import common as C
    import sims
    import adapter
    import policy_training as PT

    test = name.startswith("_test")
    p = paths(name)
    # the simulator, built before run_lite sets its seed (as scripts/s2/train_policy.py builds it)
    kind = sims.kind_of(sim)
    if kind not in ("released", "compressed", "ladder"):
        raise SystemExit(f"{sim}: Option D trains policies only in REL, its eight copies and ladder simulators")
    m, _, _, srec = sims.load(sim)
    if srec["holdout"]:
        raise SystemExit(f"{sim}: has a hold-out; Option D's simulators have none")
    model = adapter.LiteDynamicsAdapter(m)
    handed = []

    def load_rebuild(path, ens):                          # run_lite calls it after pick_model
        handed.append(model)
        return model
    adapter.load_rebuild = load_rebuild
    RL.pick_model = lambda spec: ("rebuilt", None, None)

    # the log hook: progress, the per-iteration episode length, the stop rule
    lengths, last_write, state = [], [0.0], {}
    orig_log = PT.PolicyTraining.log

    def log(self, locs, *args, **kw):
        out = orig_log(self, locs, *args, **kw)
        n = int(locs["it"]) + 1                           # the log comes after this iteration's update
        L = self.log_dict.get("Train/mean_episode_length") if len(locs["lenbuffer"]) > 0 else None
        lengths.append(float(L) if L is not None else None)
        if len(lengths) != n:
            raise RuntimeError(f"log hook: {len(lengths)} lengths after iteration {n}")
        if n % 25 == 0 or n == iterations or time.time() - last_write[0] > 300:
            _progress(job, f"train {name}", n, iterations)
            last_write[0] = time.time()
        hit = stop_at(lengths, **(stop or {}))
        if hit is not None:
            sd = {k: v.clone() for k, v in self.alg.policy.state_dict().items() if k.startswith("actor.")}
            state["stopped_sd"] = sd
            raise Stopped(hit)
        return out
    PT.PolicyTraining.log = log

    os.makedirs(p["dir"], exist_ok=True)
    _progress(job, f"train {name}", 0, iterations)
    t0 = time.time()
    argv = ["--name", p["lite_name"], "--model", "rebuilt:" + sim, "--num_envs", str(num_envs),
            "--max_iterations", str(iterations), "--seed", str(seed), "--threads", str(threads)]
    status, summ, stopped_at = "finished", None, None
    try:
        summ = RL.main(argv)
    except Stopped as err:
        status, stopped_at = "stopped", err.n
    except Exception as err:                              # a crash inside the lite code
        crash = {"policy": name, "sim": sim, "seed": seed, "attempt": attempt, "type": type(err).__name__,
                 "message": str(err)[:500], "traceback": traceback.format_exc()[-3000:],
                 "iterations_logged": len(lengths), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(os.path.join(p["dir"], f"crash_{attempt}.json"), "w") as fh:
            json.dump(crash, fh, indent=1)
        print(f"[train_policy] {name}: CRASH ({crash['type']}: {crash['message'][:200]}) on attempt {attempt}")
        sys.exit(CRASH_EXIT)
    wall = time.time() - t0
    if len(handed) != 1:
        raise SystemExit("the lite trainer did not take the simulator built here")

    os.makedirs(p["lite"], exist_ok=True)
    if status == "stopped":                               # run_lite did not reach its own metrics write
        with open(os.path.join(p["lite"], "metrics.jsonl"), "w") as fh:
            for d in RL.LOGGED:
                fh.write(json.dumps(d, default=float) + "\n")
        tmp = p["actor"] + ".tmp"
        torch.save({"model_state_dict": state["stopped_sd"], "iter": stopped_at - 1, "source": "stop rule"}, tmp)
        os.replace(tmp, p["actor"])
        src = f"stop rule at iteration {stopped_at}"
        losses = [v for d in RL.LOGGED for k, v in d.items() if k.startswith("Loss/") and isinstance(v, float)]
        losses_finite = all(v == v and abs(v) != float("inf") for v in losses)
        step_r, reward_finite = None, None
        if not losses_finite:                             # a loss that is not finite takes precedence
            status = "failed_by_simulator"
    else:
        src = TP.export_actor(p["lite"], p["actor"], iterations)
        losses_finite, reward_finite = summ["losses_all_finite"], summ["step_reward_all_finite"]
        step_r = summ["step_reward_mean_per_iter"]
        if not (losses_finite and reward_finite):
            status = "failed_by_simulator"

    last = RL.LOGGED[-LAST:]
    mr = [d["Train/mean_reward"] for d in last if isinstance(d.get("Train/mean_reward"), float)]
    rec = {
        "policy": name, "sim": sim, "seed": seed, "status": status,
        "failed": status != "finished", "measure": status in ("finished", "stopped"),
        "failed_by_simulator": None if status != "failed_by_simulator" else {
            "why": "a loss or step reward that is not finite", "losses_all_finite": losses_finite,
            "step_reward_all_finite": reward_finite, "score": 0.0, "fall_rate": 1.0},
        "stopped_at": stopped_at, "iterations_planned": iterations, "iterations_run": len(lengths),
        "stop_rule": dict(stop or STOP, what="from iteration 50 on, mean imagined episode length under 128 steps for 50 "
                                     "iterations in a row; a missing length breaks the run; earliest stop 100"),
        "mean_episode_length_per_iter": lengths,
        "sim_record": {k: srec.get(k) for k in ("name", "kind", "how", "sha256", "variant", "steps", "run", "data",
                                                "lr", "N", "seed")},
        "route": "rebuilt (Step 2's recipe; simulator from scripts/d/sims.load)", "holdout": None,
        "num_envs": num_envs, "threads": threads, "attempt": attempt,
        "imagined_return": (sum(mr) / len(mr)) if mr else None, "imagined_return_iterations": len(mr),
        "imagined_step_reward": (sum(step_r[-LAST:]) / len(step_r[-LAST:])) if step_r else None,
        "mean_reward_per_iter": [d.get("Train/mean_reward") for d in RL.LOGGED],
        "step_reward_mean_per_iter": step_r, "losses_all_finite": losses_finite,
        "step_reward_all_finite": reward_finite,
        "seconds_per_iter_median": summ["seconds_per_iter_median"] if summ else None, "wall_seconds": wall,
        "checkpoint": src, "actor_sha256": C.sha256(p["actor"]) if os.path.exists(p["actor"]) else None,
        "lite_run_dir": os.path.relpath(p["lite"], AIR), "torch": torch.__version__,
        "what": "Option D policy: Step 2's recipe (lite code, unchanged) in a learned simulator, with the stop rule",
    }
    _write_once(p["res"], rec, test)
    _progress(job, f"trained {name} ({status})", len(lengths), iterations)
    print(json.dumps({k: rec[k] for k in ("policy", "status", "stopped_at", "iterations_run", "imagined_return",
                                          "losses_all_finite", "step_reward_all_finite", "wall_seconds")}))
    return rec


def mark_failed(sim, seed, name, guard):
    """The record of a policy whose training crashed the same way twice inside the lite code, after the crash guard
    ran clean on REL: failed by its simulator (walking score 0, fall rate 1, not measured)."""
    p = paths(name)
    crashes = []
    for a in (1, 2):
        f = os.path.join(p["dir"], f"crash_{a}.json")
        if not os.path.exists(f):
            raise SystemExit(f"{name}: --mark-failed needs crash_1.json and crash_2.json")
        with open(f) as fh:
            crashes.append(json.load(fh))
    if crashes[0]["type"] != crashes[1]["type"]:
        raise SystemExit(f"{name}: the two crashes differ ({crashes[0]['type']}, {crashes[1]['type']})")
    rec = {"policy": name, "sim": sim, "seed": seed, "status": "failed_by_simulator", "failed": True,
           "measure": False, "failed_by_simulator": {"why": "the same crash inside the lite code twice; the crash "
                                                            "guard ran clean on REL", "score": 0.0, "fall_rate": 1.0,
                                                     "crashes": crashes, "guard": guard},
           "actor_sha256": None, "what": "Option D policy whose training failed by its simulator"}
    _write_once(p["res"], rec, name.startswith("_test"))
    print(f"[train_policy] {name}: failed by its simulator (the same crash twice)")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="a ladder simulator (<run>_d<steps>), REL or REL__<variant>")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--name", default=None, help="policy name (default <sim>__p<seed>); `_test*` for tests")
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    ap.add_argument("--num_envs", type=int, default=NUM_ENVS)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--job", default="ladder")
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--mark-failed", action="store_true", help="after two equal crashes and a clean crash guard")
    ap.add_argument("--guard", default=None, help="the crash guard's record (JSON), with --mark-failed")
    ap.add_argument("--test-stop", default=None, help="after,run,threshold of the stop rule (`_test*` names only)")
    a = ap.parse_args(argv)
    name = a.name or policy_name(a.sim, a.seed)
    test = name.startswith("_test")
    if not test:
        if name != policy_name(a.sim, a.seed):
            raise SystemExit("policy names are <simulator>__p<seed>")
        if a.seed not in SEEDS:
            raise SystemExit(f"the design's seeds are {SEEDS}")
        if (a.iterations, a.num_envs) != (ITERATIONS, NUM_ENVS):
            raise SystemExit("the design fixes 500 iterations and 4,096 robots; other sizes are for `_test*` names")
    if not registered() and not (test and a.iterations <= 3):
        raise SystemExit("prereg/PREREG_D.md is not committed: only `_test*` names with at most 3 iterations")
    rec = done_record(name)
    if rec is not None and not test:
        print(f"[train_policy] {name}: done already ({rec['status']}), skipped")
        return rec
    if a.mark_failed:
        return mark_failed(a.sim, a.seed, name, json.loads(a.guard) if a.guard else None)
    stop = None
    if a.test_stop:
        if not test:
            raise SystemExit("--test-stop is for `_test*` names only")
        x = a.test_stop.split(",")
        stop = {"after": int(x[0]), "run": int(x[1]), "threshold": float(x[2])}
    return train(a.sim, a.seed, name, a.iterations, a.num_envs, a.threads, a.job, a.attempt, stop)


if __name__ == "__main__":
    main()

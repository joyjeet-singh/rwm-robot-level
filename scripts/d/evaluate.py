"""Option D (D0): the walking score of one policy in the calibrated MuJoCo ANYmal D.  ./py-mj

    ./py-mj scripts/d/evaluate.py --policy authors        # RWM's released policy -> results/d/eval/authors.json
    ./py-mj scripts/d/evaluate.py --policy REL__p1        # a Step 2 policy (runs/s2/REL__p1/actor.pt)
    ./py-mj scripts/d/evaluate.py --policy <name> --actor <file>    # any other actor file (after registration)

Option C's measure and protocol, with its functions (scripts/c/evaluate.py: walk_schedule, walk_start, run_episode,
standing_reference, boot, summary): 24 velocity-command schedules x 4 seeded starts = 96 episodes of 1,000 steps,
calibrated physics, no pushes, no observation noise. Recorded as well: the fall rate on those 96 episodes, and Step 2's
physics return (Step 2's protocol: the ten recorded schedules x 8 starts; Step 2's reward), with its fall rate.
The world is a hand-built MuJoCo model of ANYmal D with a fitted PD motor; not Isaac Lab, not hardware.

Guard: until prereg/PREREG_D.md is committed, only RWM's released policy (`authors`), Step 2's 22 policies and `_test*`
names may be measured. A Step 2 actor must match the sha256 Step 2 recorded (results/s2_main.json, or its training
record for the pilots). `_test*` names write under .cache/ and may use fewer schedules.
Resumable: finished episodes are kept in runs/d/eval/<policy>.jsonl; the result file is written once.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
AIR = os.path.dirname(SCRIPTS)
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("c_evaluate", os.path.join(SCRIPTS, "c", "evaluate.py"))
CE = importlib.util.module_from_spec(_spec)        # Option C's evaluator (by path: this file has the same name)
sys.modules["c_evaluate"] = CE
_spec.loader.exec_module(CE)
sys.path.insert(0, HERE)
import dprogress  # noqa: E402

F, C = CE.F, CE.C
EVAL_DIR = os.path.join(AIR, "results", "d", "eval")
PARTS = os.path.join(AIR, "runs", "d", "eval")
S2_SIMS = ("REL", "REL__fp16", "REL__u10", "REL__u20", "REL__u30", "REL__u50", "REL__int8w", "REL__s192", "REL__s128")
S2_REGISTERED = tuple(f"{s}__p{k}" for s in S2_SIMS for k in (1, 2))
S2_PILOT = ("REL__p0", "A_s0__p0", "B_s0__p0", "A_s0__u50__p0")
S2_POLICIES = S2_REGISTERED + S2_PILOT                      # 22


def registered():
    """True once prereg/PREREG_D.md is in a commit on this branch."""
    r = subprocess.run(["git", "log", "-1", "--format=%H", "--", "prereg/PREREG_D.md"], cwd=AIR,
                       capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def check_allowed(policy):
    if policy == "authors" or policy.startswith("_test") or policy in S2_POLICIES or registered():
        return
    raise SystemExit(f"{policy}: prereg/PREREG_D.md is not committed; before registration only RWM's released policy, "
                     "Step 2's 22 policies and `_test*` names may be measured (OPTIOND_PLAN.md D0)")


def step2_sha(policy):
    """The actor sha256 Step 2 recorded for one of its policies."""
    main = json.load(open(os.path.join(AIR, "results", "s2_main.json")))["policies"]
    for p in main:
        if p["policy"] == policy:
            return p["actor_sha256"]
    for path in (os.path.join(AIR, "results", "s2", "train", f"{policy}.json"),
                 os.path.join(AIR, "runs", "s2", policy, "train.json")):
        if os.path.exists(path):
            return json.load(open(path))["actor_sha256"]
    raise SystemExit(f"{policy}: no actor sha256 recorded by Step 2")


def actor_of(policy, actor_arg):
    """(function obs -> joint command, sha256, path relative to AIR, iterations trained or None)."""
    import torch
    import s1_policy as P
    if policy == "authors":
        return F.torch_actor(P.load_actor(None)), C.sha256(C.W_RELEASED), os.path.relpath(C.W_RELEASED, AIR), None
    if policy in S2_POLICIES and not actor_arg:
        path = os.path.join(AIR, "runs", "s2", policy, "actor.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{policy}: {path} is missing")
        sha = C.sha256(path)
        if sha != step2_sha(policy):
            raise SystemExit(f"{policy}: actor sha256 differs from Step 2's record")
    else:
        if not actor_arg:
            raise SystemExit(f"{policy}: --actor is required")
        path = actor_arg
        sha = C.sha256(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    it = int(ck["iter"]) + 1 if isinstance(ck, dict) and "iter" in ck else None
    return F.torch_actor(P.load_actor(path)), sha, os.path.relpath(path, AIR), it


def evaluate(policy, actor_arg=None, schedules=None, starts=CE.N_STARTS, recorded=True, out=None, job="record"):
    check_allowed(policy)
    test = policy.startswith("_test")
    schedules = tuple(range(CE.N_SCHEDULES)) if schedules is None else tuple(schedules)
    full = schedules == tuple(range(CE.N_SCHEDULES)) and starts == CE.N_STARTS and recorded
    if not test and not full:
        raise SystemExit("the protocol is fixed (24 schedules x 4 starts, and the recorded schedules); other sizes are "
                         "for `_test*` names only")
    if out is None:
        out = os.path.join(AIR, ".cache", "d_eval", f"{policy}.json") if test else os.path.join(EVAL_DIR, f"{policy}.json")
    if os.path.exists(out):
        print(f"[evaluate] {policy}: {os.path.relpath(out, AIR)} exists, skipped")
        return json.load(open(out))

    import torch
    torch.set_num_threads(int(os.environ.get("S2_THREADS", "1")))
    actor, sha, source, iters = actor_of(policy, actor_arg)
    cal = F.frozen_calibration()
    env = F.make_env("cal", reward=True)
    part_dir = os.path.join(AIR, ".cache", "d_eval") if test else PARTS
    os.makedirs(part_dir, exist_ok=True)
    part = os.path.join(part_dir, f"{policy}.jsonl")
    done = {}
    if os.path.exists(part):
        for line in open(part):
            r = json.loads(line)
            if r.get("actor_sha256") == sha:
                done[(r["kind"], r["schedule"], r["start"])] = r
    plan = [("walk", e, j) for e in schedules for j in range(starts)]
    if recorded:
        plan += [("recorded", e, j) for e in F.EVAL_SCHEDULES for j in range(F.EVAL_STARTS)]
    t0, last = time.time(), 0.0
    with open(part, "a") as fh:
        for kind, e, j in plan:
            if (kind, e, j) in done:
                continue
            if kind == "walk":
                off, yaw = CE.walk_start(e, j)
                r = CE.run_episode(env, actor, CE.walk_schedule(e), off, yaw, e, j)
            else:
                off, yaw = F.eval_start(e, j)
                r = CE.run_episode(env, actor, F.schedule(e, CE.STEPS + 1), off, yaw, e, j)
            r.update(kind=kind, actor_sha256=sha)
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            done[(kind, e, j)] = r
            if time.time() - last > 60 or len(done) == len(plan):
                dprogress.write(job, f"evaluate {policy}", len(done), len(plan))
                last = time.time()

    strip = lambda r: {k: v for k, v in r.items() if k not in ("actor_sha256", "kind")}   # noqa: E731
    walk = [done[k] for k in plan if k[0] == "walk"]
    s = CE.summary(walk)
    rec = {
        "what": "Option D: one policy in the calibrated hand-built MuJoCo ANYmal D with a fitted PD motor (not Isaac Lab, "
                "not hardware); Option C's walking score on 24 schedules x 4 starts, its fall rate, and Step 2's physics "
                "return on the recorded schedules",
        "policy": policy, "iterations": iters, "actor_sha256": sha, "actor": source,
        "cal_xml_sha256": cal["cal_xml_sha256"], "calibration": cal["chosen"],
        "protocol": {"schedules": len(schedules), "starts": starts, "steps": CE.STEPS, "seed": CE.WALK_SEED},
        "interval": f"95% percentile bootstrap of the mean, {CE.BOOT} draws, seed {CE.BOOT_SEED}",
        "score": float(np.mean([r["score"] for r in walk])), "score_ci95": CE.boot(walk, "score", False),
        "fall_rate": s["fall_rate"], "standing_reference": CE.standing_reference(schedules),
        "n_episodes": len(walk), "summary": s, "episodes": [strip(r) for r in walk],
        "seconds_this_run": time.time() - t0,
    }
    if recorded:
        eps = [done[k] for k in plan if k[0] == "recorded"]
        sr = CE.summary(eps)
        rec["physics_return"] = sr["ret"]
        rec["physics_fall_rate"] = sr["fall_rate"]
        rec["recorded"] = {"what": "Step 2's protocol: ten recorded schedules x 8 starts", "summary": sr,
                           "episodes": [strip(r) for r in eps]}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    F.write_new(out, rec)
    print(json.dumps({"policy": policy, "score": rec["score"], "fall_rate": rec["fall_rate"],
                      "physics_return": rec.get("physics_return"), "standing_reference": rec["standing_reference"]}))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="authors, a Step 2 policy, or (after registration) any name")
    ap.add_argument("--actor", default=None, help="actor file (required for names other than authors and Step 2's)")
    ap.add_argument("--job", default="record")
    ap.add_argument("--schedules", default=None, help="comma list (`_test*` only)")
    ap.add_argument("--starts", type=int, default=CE.N_STARTS, help="starts per schedule (`_test*` only)")
    ap.add_argument("--no-recorded", action="store_true", help="skip the recorded schedules (`_test*` only)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    sch = tuple(int(x) for x in a.schedules.split(",")) if a.schedules else None
    return evaluate(a.policy, a.actor, sch, a.starts, not a.no_recorded, a.out, a.job)


if __name__ == "__main__":
    main()

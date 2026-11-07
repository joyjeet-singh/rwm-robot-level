"""Measure one exported Step 2 actor in the calibrated MuJoCo ANYmal D, with the fixed protocol.

    ./py-mj scripts/s2/evaluate.py --policy A_s0__p0          # runs/s2/A_s0__p0/actor.pt -> results/s2/eval/A_s0__p0.json

Protocol (STEP2_PLAN.md, "Measuring a policy in MuJoCo"; the code is M2's, reused unchanged):
  * the frozen calibrated model step2/model/anymal_d_cal.xml (fidelity.make_env("cal", reward=True));
  * the ten recorded episodes' velocity-command schedules, 20 s (1,000 policy steps) each, x eight seeded starts
    (fidelity.eval_start: joint angles +-0.05 rad, random heading), 80 episodes; an episode ends at a fall;
  * per episode (fidelity.run_policy_episode): physics return (the lite reward terms and weights from MuJoCo states
    and foot contacts), fall, speed-tracking error (m/s), turn-tracking error (rad/s), gait shares;
  * per policy: means over the 80 episodes, and 95% intervals of the mean return and fall rate by a bootstrap that
    resamples the eight starts within each schedule (10,000 draws, seed 2027), since every policy meets the same
    ten schedules.
Resumable: finished episodes are kept in runs/s2/<policy>/eval_partial.jsonl; the result file is written once,
never overwritten. Registration guard: until prereg/PREREG_S2.md is committed, only the four pilot policies (and
`_test*` names, which write to .cache/) may be measured.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fidelity as F  # noqa: E402

C = F.C
AIR = F.AIR
RUNS = os.path.join(AIR, "runs", "s2")
EVAL_DIR = os.path.join(AIR, "results", "s2", "eval")
PILOT = ("REL__p0", "A_s0__p0", "B_s0__p0", "A_s0__u50__p0")
BOOT = 10000
BOOT_SEED = 2027


def registered():
    """True once prereg/PREREG_S2.md exists and is tracked by git."""
    p = os.path.join(AIR, "prereg", "PREREG_S2.md")
    if not os.path.exists(p):
        return False
    r = subprocess.run(["git", "ls-files", "--error-unmatch", "prereg/PREREG_S2.md"], cwd=AIR,
                       capture_output=True)
    return r.returncode == 0


def boot_interval(eps, key, n=BOOT, seed=BOOT_SEED):
    """95% percentile interval of the mean of `key`, resampling starts within each schedule."""
    rng = np.random.default_rng(seed)
    by = {}
    for r in eps:
        by.setdefault(r["schedule"], []).append(float(r[key]))
    groups = [np.array(v) for _, v in sorted(by.items())]
    total = sum(len(g) for g in groups)
    means = np.zeros(n)
    for g in groups:
        idx = rng.integers(0, len(g), size=(n, len(g)))
        means += g[idx].sum(1)
    means /= total
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def evaluate(policy, schedules=F.EVAL_SCHEDULES, starts=F.EVAL_STARTS, out=None, job="eval"):
    test = policy.startswith("_test")
    if not test and policy not in PILOT and not registered():
        raise SystemExit(f"{policy}: prereg/PREREG_S2.md is not committed; only the pilot policies {PILOT} "
                         "may be measured before registration (STEP2_PLAN.md C4)")
    full = tuple(schedules) == tuple(F.EVAL_SCHEDULES) and starts == F.EVAL_STARTS
    if not test and not full:
        raise SystemExit("the protocol is fixed (10 schedules x 8 starts); other sizes are for `_test*` names only")
    out = out or (os.path.join(AIR, ".cache", "s2_eval_" + policy + ".json") if test
                  else os.path.join(EVAL_DIR, policy + ".json"))
    if os.path.exists(out):
        print(f"[evaluate] {policy}: {os.path.relpath(out, AIR)} exists, skipped")
        return json.load(open(out))

    import s1_policy as P
    import torch
    torch.set_num_threads(int(os.environ.get("S2_THREADS", "2")))
    actor_path = os.path.join(RUNS, policy, "actor.pt")
    actor_sha = C.sha256(actor_path)
    actor = F.torch_actor(P.load_actor(actor_path))
    cal = F.frozen_calibration()
    env = F.make_env("cal", reward=True)

    part = os.path.join(RUNS, policy, "eval_partial.jsonl")
    done = {}
    if os.path.exists(part):
        for line in open(part):
            r = json.loads(line)
            if r.get("actor_sha256") == actor_sha:
                done[(r["schedule"], r["start"])] = r
    plan = [(e, j) for e in schedules for j in range(starts)]
    t0 = time.time()
    with open(part, "a") as fh:
        for i, (e, j) in enumerate(plan):
            if (e, j) in done:
                continue
            r = F.run_policy_episode(env, actor, e, j)
            r["actor_sha256"] = actor_sha
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            done[(e, j)] = r
            if len(done) % 5 == 0 or len(done) == len(plan):
                F.progress(job, f"evaluate {policy}", len(done), len(plan))
    eps = [done[k] for k in plan]
    summ = F.summarise(eps)
    summ["ret_ci95"] = boot_interval(eps, "ret")
    summ["fall_rate_ci95"] = boot_interval(eps, "fell")
    rec = {
        "what": "Step 2: one policy in the calibrated hand-built MuJoCo ANYmal D (not the data's simulator, not "
                "hardware); 10 schedules x 8 starts",
        "policy": policy, "actor_sha256": actor_sha, "cal_xml_sha256": cal["cal_xml_sha256"],
        "calibration": cal["chosen"],
        "interval": f"95% percentile bootstrap of the mean, starts resampled within schedules, {BOOT} draws, "
                    f"seed {BOOT_SEED}",
        "seconds_this_run": time.time() - t0, "summary": summ,
        "episodes": [{k: v for k, v in r.items() if k != "actor_sha256"} for r in eps],
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    F.write_new(out, rec)
    print(json.dumps({"policy": policy, "fall_rate": summ["fall_rate"], "ret": summ["ret"],
                      "ret_ci95": summ["ret_ci95"], "speed_err": summ["speed_err"], "turn_err": summ["turn_err"]}))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--job", default="eval")
    ap.add_argument("--schedules", default=None, help="comma list (tests only)")
    ap.add_argument("--starts", type=int, default=F.EVAL_STARTS, help="starts per schedule (tests only)")
    ap.add_argument("--out", default=None, help="output path (tests only)")
    a = ap.parse_args(argv)
    sch = tuple(int(x) for x in a.schedules.split(",")) if a.schedules else F.EVAL_SCHEDULES
    return evaluate(a.policy, sch, a.starts, a.out, a.job)


if __name__ == "__main__":
    main()

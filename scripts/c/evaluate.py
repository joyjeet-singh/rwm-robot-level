"""Option C (C2): measure one policy's walking in the calibrated MuJoCo ANYmal D, with the fixed protocol.

    ./py-mj scripts/c/evaluate.py --policy authors                        # -> results/c_authors.json
    ./py-mj scripts/c/evaluate.py --policy sim_6M_N8_s0__p0 --iters 500   # -> results/c/eval/sim_6M_N8_s0__p0__i500.json

Built from Step 2's evaluator (scripts/s2/evaluate.py, fidelity.run_policy_episode); the world is the same frozen model.
The world here is a hand-built MuJoCo model of ANYmal D with a fitted PD motor. It is not Isaac Lab and not hardware.

The walking score (OPTIONC_PLAN.md, "Walking score", the primary outcome):
  * each step scores exp(-(|v_xy - c_xy|^2 + (w_z - c_w)^2) / 0.25), with v and w the base-frame speed and turn rate
    after the step and c the velocity command the step was taken under; after a fall every remaining step scores 0;
  * an episode scores the mean over its 1,000 steps; a policy the mean over its episodes;
  * protocol: calibrated physics, no pushes, no observation noise; 24 fixed command schedules (uniform in -1 to 1 on
    each axis, a new command every 10 s) x 4 seeded starts = 96 episodes;
  * the standing-still reference is computed exactly from the same schedules: the score of a robot with zero velocity.

Secondary measures: fall rate, speed and turn errors, physics return (Step 2's definitions), and Step 2's protocol on the
ten recorded schedules x 8 starts. Intervals are 95% percentile bootstraps of the mean (10,000 draws, seed 2027):
`*_ci95` resamples the 96 episodes, `*_ci95_within_schedules` resamples starts within schedules as Step 2 did.

A state that is not finite, or a MuJoCo divergence warning (which would otherwise silently restart the robot), ends the
episode and counts as a fall; the number of such episodes is reported.

Resumable: finished episodes are kept in runs/c/<policy>/eval_i<iters>.jsonl; the result file is written once.
Registration guard: until prereg/PREREG_C.md is committed, only the authors' policy, the pilot policies and `_test*` names
may be measured. `_test*` names write under .cache/ and may use other sizes.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(1, os.path.join(AIR, "scripts", "s2"))
import fidelity as F  # noqa: E402

C, E = F.C, F.E
import mujoco  # noqa: E402

RUNS = os.path.join(AIR, "runs", "c")
EVAL_DIR = os.path.join(AIR, "results", "c", "eval")
AUTHORS_OUT = os.path.join(AIR, "results", "c_authors.json")
PROGRESS = os.path.join(AIR, "results", "c", "progress.json")
F.RES_DIR, F.PROGRESS = os.path.dirname(PROGRESS), PROGRESS

WALK_SEED = 20275
N_SCHEDULES = 24
N_STARTS = 4
STEPS = 1000
CMD_STEPS = 500                  # a new velocity command every 10 s
SCALE = 0.25
BOOT = 10000
BOOT_SEED = 2027
SAVES = (500, 1000, 1500, 2000)
PILOT_SIMS = ("sim_6M_N8_s0", "sim_10k_N8_s0")
RETRY_WEIGHTS = (2.0, 4.0, 0.5, 0.25)
BAD = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC)


def policy_name(sim, seed, weight=1.0):
    return f"{sim}__p{seed}" if weight == 1.0 else f"{sim}__w{weight:g}__p{seed}"


PILOT = tuple(policy_name(s, 0) for s in PILOT_SIMS) + tuple(policy_name(PILOT_SIMS[0], 0, w) for w in RETRY_WEIGHTS)
# Amendment 1 (C2R): the second pilot's arms and its Gate C' policy, in the pilot simulators or their 2 x S retrainings
PILOT2_SIMS = PILOT_SIMS + tuple(s + "__2S" for s in PILOT_SIMS)
PILOT2_TAGS = ("cal", "cal_fall4", "w0", "w0_fall4")
PILOT += tuple(f"{s}__{t}__p0" for s in PILOT2_SIMS for t in PILOT2_TAGS)


def registered():
    """True once prereg/PREREG_C.md is in a commit on this branch."""
    r = subprocess.run(["git", "log", "-1", "--format=%H", "--", "prereg/PREREG_C.md"], cwd=AIR,
                       capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def check_allowed(policy):
    if policy == "authors" or policy.startswith("_test") or policy in PILOT or registered():
        return
    raise SystemExit(f"{policy}: prereg/PREREG_C.md is not committed; before registration only the authors' policy and "
                     f"the pilot policies {PILOT} may be measured (OPTIONC_PLAN.md C4)")


def walk_schedule(e, steps=STEPS):
    """(steps + 1, 3) velocity commands of schedule e: two commands, uniform in -1 to 1, 10 s each."""
    rng = np.random.default_rng([WALK_SEED, 0, e])
    c0, c1 = rng.uniform(-1.0, 1.0, 3), rng.uniform(-1.0, 1.0, 3)
    cmd = np.empty((steps + 1, 3))
    cmd[:CMD_STEPS], cmd[CMD_STEPS:] = c0, c1
    return cmd


def walk_start(e, j):
    """Start j of schedule e: joint offsets (12,) and heading, drawn as fidelity.eval_start draws them."""
    rng = np.random.default_rng([WALK_SEED, 1, e, j])
    return rng.uniform(-F.JOINT_JITTER, F.JOINT_JITTER, 12), float(rng.uniform(-np.pi, np.pi))


def step_score(s, c):
    """One step's score from a state (base-frame v at 0:2, turn rate at 5) and the command (speed x, y, turn)."""
    return float(np.exp(-(np.sum((s[0:2] - c[0:2]) ** 2) + (s[5] - c[2]) ** 2) / SCALE))


def run_episode(env, actor, cmd_all, off, yaw, e, j, steps=STEPS):
    """One episode: Step 2's loop (fidelity.run_policy_episode) with the step scores added and divergence caught."""
    env.reset(joint_offset=off, yaw=yaw, cmd=cmd_all[0])
    V, W, CM, FT = [], [], [], []
    ret, total = 0.0, 0.0
    fell, broken = False, False
    for k in range(steps):
        env.cmd = np.asarray(cmd_all[k])
        obs = E.observation(env.s, env.cmd, env.last_action)
        a = actor(obs)
        _, info = env.step(a)
        s = info["state"]
        if not np.all(np.isfinite(s)) or any(env.d.warning[w].number > 0 for w in BAD):
            fell, broken = True, True
            break
        V.append(s[0:2]), W.append(s[5]), CM.append(cmd_all[k + 1]), FT.append(info["contacts"][4:8] > 0.5)
        ret += info.get("reward", 0.0)
        total += step_score(s, env.cmd)
        if info["fell"]:
            fell = True
            break
    if V:
        st = F.gait_stats(np.array(V), np.array(W), np.array(CM), np.array(FT, dtype=float))
    else:                         # diverged on its first step: the robot had not moved
        st = {"rows": 0, "two_feet": 0.0, "duty": 0.0, "feet_down_hist": [1.0, 0.0, 0.0, 0.0, 0.0],
              "speed_err": float(np.linalg.norm(cmd_all[0, :2])), "turn_err": float(abs(cmd_all[0, 2]))}
    return {"schedule": e, "start": j, "fell": fell, "diverged": broken, "steps": len(V), "ret": ret,
            "score": total / steps, **st}


def standing_reference(schedules, steps=STEPS):
    """The score of a robot with zero velocity on these schedules (no fall, exact)."""
    vals = [np.mean(np.exp(-np.sum(walk_schedule(e, steps)[:steps] ** 2, axis=1) / SCALE)) for e in schedules]
    return float(np.mean(vals))


def boot(eps, key, within_schedules, n=BOOT, seed=BOOT_SEED):
    """95% percentile interval of the mean of `key`: over all episodes, or over starts within each schedule."""
    rng = np.random.default_rng(seed)
    if not within_schedules:
        v = np.array([float(r[key]) for r in eps])
        means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    else:
        by = {}
        for r in eps:
            by.setdefault(r["schedule"], []).append(float(r[key]))
        groups = [np.array(v) for _, v in sorted(by.items())]
        total, means = sum(len(g) for g in groups), np.zeros(n)
        for g in groups:
            means += g[rng.integers(0, len(g), size=(n, len(g)))].sum(1)
        means /= total
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def summary(eps):
    s = F.summarise(eps)
    s.pop("per_schedule")
    for key in ("ret", "fell"):
        name = "fall_rate" if key == "fell" else key
        s[f"{name}_ci95"], s[f"{name}_ci95_within_schedules"] = boot(eps, key, False), boot(eps, key, True)
    s["diverged"] = int(sum(r["diverged"] for r in eps))
    return s


def get_actor(policy, iters, actor_arg):
    """(function obs -> joint command, sha256 of the weights, path or a label)."""
    import s1_policy as P
    if policy.startswith("_test_stand"):
        return (lambda obs: np.zeros(12)), "standing", "joint command zero: holds the standing pose"
    if policy == "authors":
        return F.torch_actor(P.load_actor(None)), C.sha256(C.W_RELEASED), os.path.relpath(C.W_RELEASED, AIR)
    path = actor_arg or os.path.join(RUNS, policy, f"actor_i{iters}.pt")
    return F.torch_actor(P.load_actor(path)), C.sha256(path), os.path.relpath(path, AIR)


def evaluate(policy, iters=None, schedules=None, starts=N_STARTS, recorded=True, out=None, job="pilot", actor_arg=None):
    check_allowed(policy)
    test = policy.startswith("_test")
    schedules = tuple(range(N_SCHEDULES)) if schedules is None else tuple(schedules)
    full = schedules == tuple(range(N_SCHEDULES)) and starts == N_STARTS and recorded
    if not test and not full:
        raise SystemExit("the protocol is fixed (24 schedules x 4 starts, and the recorded schedules); other sizes are "
                         "for `_test*` names only")
    if policy != "authors" and not test and iters not in SAVES:
        raise SystemExit(f"--iters must be one of {SAVES}")
    tag = "" if policy == "authors" else f"__i{iters}"
    if out is None:
        if test:
            out = os.path.join(AIR, ".cache", f"c_eval_{policy}{tag}.json")
        elif policy == "authors":
            out = AUTHORS_OUT
        else:
            out = os.path.join(EVAL_DIR, f"{policy}{tag}.json")
    if os.path.exists(out):
        print(f"[evaluate] {policy}{tag}: {os.path.relpath(out, AIR)} exists, skipped")
        return json.load(open(out))

    import torch
    torch.set_num_threads(int(os.environ.get("S2_THREADS", "1")))
    actor, actor_sha, source = get_actor(policy, iters, actor_arg)
    cal = F.frozen_calibration()
    env = F.make_env("cal", reward=True)

    part_dir = os.path.join(AIR, ".cache") if test else os.path.join(RUNS, policy)
    os.makedirs(part_dir, exist_ok=True)
    part = os.path.join(part_dir, f"eval_i{iters}.jsonl" if not test else f"c_eval_{policy}{tag}.jsonl")
    done = {}
    if os.path.exists(part):
        for line in open(part):
            r = json.loads(line)
            if r.get("actor_sha256") == actor_sha:
                done[(r["kind"], r["schedule"], r["start"])] = r
    plan = [("walk", e, j) for e in schedules for j in range(starts)]
    if recorded:
        plan += [("recorded", e, j) for e in F.EVAL_SCHEDULES for j in range(F.EVAL_STARTS)]
    t0 = time.time()
    with open(part, "a") as fh:
        for i, (kind, e, j) in enumerate(plan):
            if (kind, e, j) in done:
                continue
            if kind == "walk":
                off, yaw = walk_start(e, j)
                r = run_episode(env, actor, walk_schedule(e), off, yaw, e, j)
            else:
                off, yaw = F.eval_start(e, j)
                r = run_episode(env, actor, F.schedule(e, STEPS + 1), off, yaw, e, j)
            r.update(kind=kind, actor_sha256=actor_sha)
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            done[(kind, e, j)] = r
            if len(done) % 5 == 0 or len(done) == len(plan):
                F.progress(job, f"evaluate {policy}{tag}", len(done), len(plan))

    strip = lambda r: {k: v for k, v in r.items() if k not in ("actor_sha256", "kind")}   # noqa: E731
    walk = [done[k] for k in plan if k[0] == "walk"]
    score = float(np.mean([r["score"] for r in walk]))
    rec = {
        "what": "Option C: one policy in the calibrated hand-built MuJoCo ANYmal D with a fitted PD motor (not Isaac Lab, "
                "not hardware); walking score on 24 schedules x 4 starts",
        "policy": policy, "iterations": iters, "actor_sha256": actor_sha, "actor": source,
        "cal_xml_sha256": cal["cal_xml_sha256"], "calibration": cal["chosen"],
        "protocol": {"schedules": len(schedules), "starts": starts, "steps": STEPS, "seed": WALK_SEED},
        "interval": f"95% percentile bootstrap of the mean, {BOOT} draws, seed {BOOT_SEED}",
        "score": score, "score_ci95": boot(walk, "score", False),
        "score_ci95_within_schedules": boot(walk, "score", True),
        "standing_reference": standing_reference(schedules),
        "n_episodes": len(walk), "summary": summary(walk), "episodes": [strip(r) for r in walk],
        "seconds_this_run": time.time() - t0,
    }
    if recorded:
        eps = [done[k] for k in plan if k[0] == "recorded"]
        rec["recorded"] = {"what": "Step 2's protocol: ten recorded schedules x 8 starts", "summary": summary(eps),
                           "episodes": [strip(r) for r in eps]}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    F.write_new(out, rec)
    s = rec["summary"]
    print(json.dumps({"policy": policy, "iterations": iters, "score": score, "score_ci95": rec["score_ci95"],
                      "standing_reference": rec["standing_reference"], "fall_rate": s["fall_rate"], "ret": s["ret"],
                      "speed_err": s["speed_err"], "turn_err": s["turn_err"], "diverged": s["diverged"]}))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="authors, a pilot policy name, or `_test*`")
    ap.add_argument("--iters", type=int, default=None, help="which save to measure (500, 1000, 1500 or 2000)")
    ap.add_argument("--job", default="pilot")
    ap.add_argument("--schedules", default=None, help="comma list (tests only)")
    ap.add_argument("--starts", type=int, default=N_STARTS, help="starts per schedule (tests only)")
    ap.add_argument("--no-recorded", action="store_true", help="skip the recorded schedules (tests only)")
    ap.add_argument("--actor", default=None, help="actor file (tests only)")
    a = ap.parse_args(argv)
    if a.actor and not a.policy.startswith("_test"):
        ap.error("--actor is for `_test*` names only")
    sch = tuple(int(x) for x in a.schedules.split(",")) if a.schedules else None
    return evaluate(a.policy, a.iters, sch, a.starts, not a.no_recorded, job=a.job, actor_arg=a.actor)


if __name__ == "__main__":
    main()

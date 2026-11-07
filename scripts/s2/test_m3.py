"""M3 tests: a 2-iteration training run exports an actor that evaluate.py measures (one schedule, one start).

    ./py-mj scripts/s2/test_m3.py        # about 1 minute; writes results/s2/m3_tests.json (a new file only)

  a  compressed simulator (A_s0__u50, 256 robots, 2 iterations): run finite, holdout 1,8, actor exported in the
     layout s1_policy.load_actor reads, its weights equal the last checkpoint's actor;
  b  the variant really is applied: the simulator handed to the lite env has the pruned count of zero weights;
  c  the released model (REL), same sizes: no holdout (trained on all episodes), actor exported;
  d  evaluate.py measures the exported actor on schedule 0, start 0: finite return and errors; a rerun is skipped;
     and the episode equals fidelity.run_policy_episode on the same actor;
  e  guards: plan sizes are enforced for real names (training and evaluation), and a non-pilot policy is refused
     before registration.
Test outputs go to runs/s2/_test_*, runs/lite/s2__test_* and .cache/ (all git-ignored).
"""
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AIR = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(AIR, "results", "s2", "m3_tests.json")
PY, PYMJ = os.path.join(AIR, "py"), os.path.join(AIR, "py-mj")
ENV = dict(os.environ, OMP_NUM_THREADS="2", S2_THREADS="2")


def run(cmd):
    r = subprocess.run(cmd, cwd=AIR, capture_output=True, text=True, env=ENV)
    return r.returncode, r.stdout + r.stderr


def last_json(text):
    for line in reversed(text.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return None


def main():
    t0 = time.time()
    for n in ("_test_u50", "_test_rel"):
        shutil.rmtree(os.path.join(AIR, "runs", "s2", n), ignore_errors=True)
        shutil.rmtree(os.path.join(AIR, "runs", "lite", "s2_" + n), ignore_errors=True)
        p = os.path.join(AIR, ".cache", f"s2_eval_{n}.json")
        if os.path.exists(p):
            os.remove(p)
    res = {}

    # a
    rc, out = run([PY, "scripts/s2/train_policy.py", "--sim", "A_s0__u50", "--seed", "0", "--name", "_test_u50",
                   "--iterations", "2", "--num_envs", "256", "--job", "test"])
    tr = json.load(open(os.path.join(AIR, "runs", "s2", "_test_u50", "train.json"))) if rc == 0 else {}
    import torch
    sys.path.insert(0, os.path.dirname(HERE))
    import s1_policy as P
    ok_a = rc == 0 and tr.get("holdout") == "1,8" and tr["step_reward_all_finite"] and tr["losses_all_finite"]
    if ok_a:
        net = P.load_actor(os.path.join(AIR, "runs", "s2", "_test_u50", "actor.pt"))
        ck = torch.load(os.path.join(AIR, "runs", "lite", "s2__test_u50", "policy_1.pt"), map_location="cpu",
                        weights_only=False)["model_state_dict"]
        same = all(torch.equal(net[i].weight, ck[f"actor.{i}.weight"]) and torch.equal(net[i].bias, ck[f"actor.{i}.bias"])
                   for i in (0, 2, 4, 6))
        ok_a = ok_a and same and ck is not None
    res["a_train_export"] = {"pass": bool(ok_a), "rc": rc, "tail": out.strip().splitlines()[-1:] if rc else None}

    # b
    code = ("import sys, json, os; sys.argv=['x']; sys.path.insert(0, 'scripts/s2'); import train_policy as T; "
            "row=T.sim_row('A_s0__u50'); sim=T.build_sim(row); import torch; "
            "z=sum(int((p==0).sum()) for n,p in sim.net.named_parameters() if p.dim()>=2); "
            "print(json.dumps({'zeros': z, 'pruned': row['variant_info']['pruned_entries']}))")
    rc, out = run([PY, "-c", code])
    b = last_json(out) or {}
    res["b_variant_applied"] = {"pass": rc == 0 and b.get("zeros", 0) >= b.get("pruned", 1), **b,
                                "tail": None if rc == 0 else out.strip().splitlines()[-2:]}

    # c
    rc, out = run([PY, "scripts/s2/train_policy.py", "--sim", "REL", "--seed", "0", "--name", "_test_rel",
                   "--iterations", "2", "--num_envs", "256", "--job", "test"])
    trr = json.load(open(os.path.join(AIR, "runs", "s2", "_test_rel", "train.json"))) if rc == 0 else {}
    res["c_released"] = {"pass": rc == 0 and trr.get("holdout") is None and trr.get("sim_ensemble") == 5
                         and os.path.exists(os.path.join(AIR, "runs", "s2", "_test_rel", "actor.pt")),
                         "rc": rc, "tail": None if rc == 0 else out.strip().splitlines()[-2:]}

    # d
    rc, out = run([PYMJ, "scripts/s2/evaluate.py", "--policy", "_test_u50", "--schedules", "0", "--starts", "1",
                   "--job", "test"])
    ev = json.load(open(os.path.join(AIR, ".cache", "s2_eval__test_u50.json"))) if rc == 0 else {}
    import math
    s = ev.get("summary", {})
    fin = rc == 0 and all(math.isfinite(s[k]) for k in ("ret", "speed_err", "turn_err", "fall_rate"))
    rc2, out2 = run([PYMJ, "scripts/s2/evaluate.py", "--policy", "_test_u50", "--schedules", "0", "--starts", "1"])
    code = ("import sys, json; sys.path.insert(0, 'scripts/s2'); import fidelity as F, s1_policy as P; "
            "env=F.make_env('cal', reward=True); a=F.torch_actor(P.load_actor('runs/s2/_test_u50/actor.pt')); "
            "r=F.run_policy_episode(env, a, 0, 0); print(json.dumps({'ret': r['ret'], 'steps': r['steps']}))")
    rc3, out3 = run([PYMJ, "-c", code])
    ref = last_json(out3) or {}
    ep = (ev.get("episodes") or [{}])[0]
    res["d_evaluate"] = {"pass": bool(fin and rc2 == 0 and "skipped" in out2 and rc3 == 0
                                      and ref.get("ret") == ep.get("ret") and ref.get("steps") == ep.get("steps")),
                         "ret": s.get("ret"), "steps": ep.get("steps"), "fell": ep.get("fell"),
                         "speed_err": s.get("speed_err"), "turn_err": s.get("turn_err"),
                         "tail": None if rc == 0 else out.strip().splitlines()[-2:]}

    # e
    g1, o1 = run([PY, "scripts/s2/train_policy.py", "--sim", "A_s0", "--seed", "0", "--iterations", "2"])
    g2, o2 = run([PYMJ, "scripts/s2/evaluate.py", "--policy", "A_s0__p0", "--schedules", "0", "--starts", "1"])
    g3, o3 = run([PYMJ, "scripts/s2/evaluate.py", "--policy", "A_s1__p0"])
    res["e_guards"] = {"pass": g1 != 0 and "500 iterations" in o1 and g2 != 0 and "protocol is fixed" in o2
                       and g3 != 0 and "PREREG_S2" in o3}

    allp = all(v["pass"] for v in res.values())
    rec = {"what": "M3 tests (scripts/s2/test_m3.py)", "seconds": time.time() - t0, "all_pass": allp, **res}
    print(json.dumps(rec, indent=1, default=str))
    dest = OUT if allp else os.path.join(AIR, ".cache", "s2_m3_tests_failed.json")
    if allp and os.path.exists(OUT):
        dest = os.path.join(AIR, ".cache", "s2_m3_tests_rerun.json")
        print(f"{OUT} exists; not overwritten; this run -> {dest}")
    json.dump(rec, open(dest, "w"), indent=1, default=str)
    print("ALL PASS" if allp else "FAILED")
    sys.exit(0 if allp else 1)


if __name__ == "__main__":
    main()

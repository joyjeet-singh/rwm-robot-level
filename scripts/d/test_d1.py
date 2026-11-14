"""Option D (D1): tests of the policy wrapper, the rules, the driver and the statistics.  ./py scripts/d/test_d1.py

OPTIOND_PLAN.md D1 step 5 (at most 15 minutes). Each group runs in its own process; output under .cache/d_test1/.
  1. the recipe is Step 2's: one thread, 256 robots, 3 iterations, seed 1: scripts/d/train_policy.py on REL and
     scripts/s2/train_policy.py --sim REL --route rebuilt log the same losses and rewards (to 1e-6); and the stop
     hook really stops and exports (REL__s128, a `_test`-only stop rule that fires at iteration 2). `_test_d*` names;
     every folder they make under runs/ is deleted;
  2. the stop rule on made-up histories of episode lengths;
  3. the verdict rule;  4. the search rule (every row of the table);
  5. the driver in toy mode: the four search paths and both start conditions; all seven runs when both hold; a run
     whose loss is not finite; the sanity gate; the 56-hour limit with started runs finishing their tests; seeds 1
     and 2 for every tested simulator and seed 3 only after a split (a stopped policy fails, a score of exactly 0.15
     does not); the crash guard both ways; a stop and a resume between two policies of one test;
  6. d_stats.py on synthetic pools: a clear case of each label, NOT COMPUTABLE, and a pool of four.
"""
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
AIR = os.path.dirname(SCRIPTS)
TMP = os.path.join(AIR, ".cache", "d_test1")
OUT = os.path.join(TMP, "results.json")
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", D_PROGRESS=os.path.join(TMP, "progress.json"))


def ok(name, cond, detail=""):
    print(json.dumps({"test": name, "passed": bool(cond), "detail": detail}, default=str), flush=True)
    return bool(cond)


# ------------------------------------------------------------------ 1. the recipe
def _metrics(path):
    with open(path) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def t1():
    names = {"s2": "_test_d1_s2", "d": "_test_d1_d", "stop": "_test_d1_stop"}
    made = [os.path.join(AIR, "runs", "s2", names["s2"]), os.path.join(AIR, "runs", "lite", "s2_" + names["s2"])]
    for k in ("d", "stop"):
        made += [os.path.join(AIR, "runs", "d", "policies", names[k]), os.path.join(AIR, "runs", "lite", "d_" + names[k])]
    for p in made:
        if os.path.exists(p):
            raise SystemExit(f"{p} exists before the test")
    os.makedirs(TMP, exist_ok=True)
    try:
        code = ("import sys; sys.path.insert(0, 'scripts/s2'); import train_policy as TP; "
                f"TP.PROGRESS = {os.path.join(TMP, 's2_progress.json')!r}; "
                f"TP.main(['--sim', 'REL', '--route', 'rebuilt', '--seed', '1', '--name', {names['s2']!r}, "
                "'--iterations', '3', '--num_envs', '256', '--threads', '1'])")
        r1 = subprocess.run(["./py", "-c", code], cwd=AIR, env=ENV, capture_output=True, text=True)
        r2 = subprocess.run(["./py", "scripts/d/train_policy.py", "--sim", "REL", "--seed", "1", "--name", names["d"],
                             "--iterations", "3", "--num_envs", "256", "--threads", "1", "--job", "test"],
                            cwd=AIR, env=ENV, capture_output=True, text=True)
        if r1.returncode or r2.returncode:
            return ok("1 recipe: both wrappers ran", False, (r1.stderr[-800:], r2.stderr[-800:]))
        a = _metrics(os.path.join(AIR, "runs", "lite", "s2_" + names["s2"], "metrics.jsonl"))
        b = _metrics(os.path.join(AIR, "runs", "lite", "d_" + names["d"], "metrics.jsonl"))
        diff, n = 0.0, 0
        same_keys = len(a) == len(b) == 4
        for x, y in zip(a, b):
            kx = {k for k in x if not k.startswith("Perf/")}
            same_keys &= kx == {k for k in y if not k.startswith("Perf/")}
            for k in kx:
                if isinstance(x[k], (int, float)) and isinstance(y.get(k), (int, float)):
                    diff, n = max(diff, abs(x[k] - y[k])), n + 1
        sa = json.load(open(os.path.join(AIR, "runs", "lite", "s2_" + names["s2"], "summary.json")))
        sb = json.load(open(os.path.join(AIR, "runs", "lite", "d_" + names["d"], "summary.json")))
        ra, rb = sa["step_reward_mean_per_iter"], sb["step_reward_mean_per_iter"]
        rdiff = max(abs(x - y) for x, y in zip(ra, rb)) if len(ra) == len(rb) == 3 else float("inf")
        good = ok("1a recipe: Step 2's and Option D's wrappers log the same losses and rewards (1e-6)",
                  same_keys and n > 40 and diff <= 1e-6 and rdiff <= 1e-6,
                  {"values_compared": n, "max_abs_diff": diff, "step_reward_max_abs_diff": rdiff})
        rec = json.load(open(os.path.join(AIR, ".cache", "d_train", names["d"] + ".json")))
        good &= ok("1b record: finished, 3 iterations, actor exported", rec["status"] == "finished" and
                   rec["iterations_run"] == 3 and os.path.exists(os.path.join(AIR, "runs", "d", "policies", names["d"],
                                                                              "actor.pt")), rec["status"])
        r3 = subprocess.run(["./py", "scripts/d/train_policy.py", "--sim", "REL__s128", "--seed", "1", "--name",
                             names["stop"], "--iterations", "3", "--num_envs", "64", "--threads", "1", "--job", "test",
                             "--test-stop", "0,2,1e9"], cwd=AIR, env=ENV, capture_output=True, text=True)
        rs = json.load(open(os.path.join(AIR, ".cache", "d_train", names["stop"] + ".json"))) if r3.returncode == 0 \
            else {}
        import torch
        ck = torch.load(os.path.join(AIR, "runs", "d", "policies", names["stop"], "actor.pt"), weights_only=False) \
            if r3.returncode == 0 else {}
        good &= ok("1c stop hook: stops at the rule's iteration, exports that actor, counts as failed and is measured",
                   rs.get("status") == "stopped" and rs.get("stopped_at") == 2 and ck.get("iter") == 1 and
                   rs.get("failed") is True and rs.get("measure") is True,
                   {k: rs.get(k) for k in ("status", "stopped_at", "failed", "measure")})
        r4 = subprocess.run(["./py", "scripts/d/train_policy.py", "--sim", "REL", "--seed", "1", "--name",
                             "_test_d1_long", "--iterations", "4", "--num_envs", "64"], cwd=AIR, env=ENV,
                            capture_output=True, text=True)
        r5 = subprocess.run(["./py", "scripts/d/train_policy.py", "--sim", "REL", "--seed", "1"], cwd=AIR, env=ENV,
                            capture_output=True, text=True)
        import train_policy as TPD
        if TPD.registered():
            good &= ok("1d (registered) a non-test name needs the design's sizes", r4.returncode != 0, "")
        else:
            good &= ok("1d before registration: only `_test*` names with at most 3 iterations",
                       r4.returncode != 0 and r5.returncode != 0 and "not committed" in r4.stderr + r4.stdout
                       and "not committed" in r5.stderr + r5.stdout, (r4.returncode, r5.returncode))
        return good
    finally:
        for p in made + [os.path.join(AIR, "runs", "d", "policies", "_test_d1_long"),
                         os.path.join(AIR, "runs", "lite", "d__test_d1_long")]:
            shutil.rmtree(p, ignore_errors=True)
        for k in ("d", "stop"):
            f = os.path.join(AIR, ".cache", "d_train", names[k] + ".json")
            if os.path.exists(f):
                os.remove(f)


# ------------------------------------------------------------------ 2-4. the rules
def t2():
    sys.path.insert(0, HERE)
    import train_policy as T
    S = 10.0
    L = 200.0
    cases = [
        ("short from iteration 1: earliest stop at 100", [S] * 300, 100),
        ("short 1..99 only: 51..99 is 49 in a row, no stop", [S] * 99 + [L] * 201, None),
        ("short from 51: stop at 100", [L] * 50 + [S] * 250, 100),
        ("short 50..99, long at 100: iteration 50 does not count, no stop", [L] * 49 + [S] * 50 + [L] * 201, None),
        ("short from 60: stop at 109", [L] * 59 + [S] * 241, 109),
        ("a missing length at 75 breaks the run: stop at 125", [S] * 74 + [None] + [S] * 225, 125),
        ("exactly 128 is not under 128", [128.0] * 300, None),
        ("127.9 is under 128", [127.9] * 300, 100),
        ("needs 50 in a row: 49 short, 1 long, repeated", ([S] * 49 + [L]) * 10, None),
        ("fewer than 100 iterations never stop", [S] * 99, None),
    ]
    good = True
    for name, h, want in cases:
        got = T.stop_at(h)
        good &= ok(f"2 stop rule: {name}", got == want, {"got": got, "want": want})
    return good


def t3():
    sys.path.insert(0, HERE)
    import ladder as LD
    V, F, N = LD.verdict_rule, LD.FAILED, LD.NOT_FAILED
    pf = LD.policy_failed
    cases = [
        ("no policy yet: seed 1", V([]), ("train", 1)),
        ("after one: seed 2", V([True]), ("train", 2)),
        ("two failed", V([True, True]), ("verdict", F)),
        ("neither failed", V([False, False]), ("verdict", N)),
        ("a split: seed 3", V([True, False]), ("train", 3)),
        ("a split: seed 3 (other order)", V([False, True]), ("train", 3)),
        ("split settled by the third: failed", V([True, False, True]), ("verdict", F)),
        ("split settled by the third: did not fail", V([False, True, False]), ("verdict", N)),
        ("a stopped policy fails (whatever its score)", pf({"status": "stopped", "score": 0.9}), True),
        ("failed by its simulator fails", pf({"status": "failed_by_simulator", "score": 0.0}), True),
        ("a score of exactly 0.15 does not fail", pf({"status": "finished", "score": 0.15}), False),
        ("a score just under 0.15 fails", pf({"status": "finished", "score": 0.1499999}), True),
        ("stopped + walking policy is a split", V([pf({"status": "stopped", "score": 0.9}),
                                                    pf({"status": "finished", "score": 0.5})]), ("train", 3)),
    ]
    good = all(ok(f"3 verdict rule: {n}", got == want, {"got": got}) for n, got, want in cases)
    bad = 0
    for x in ([True, True, False], [True, False, False, True]):
        try:
            V(x)
        except ValueError:
            bad += 1
    return good & ok("3 verdict rule: no third policy without a split; at most three", bad == 2, bad)


def t4():
    sys.path.insert(0, HERE)
    import ladder as LD
    S = LD.search_rule
    cases = [
        ("first test at 80", S([]), 80),
        ("80 failed: 20", S([(80, True)]), 20),
        ("80 did not fail: 320", S([(80, False)]), 320),
        ("80 failed, 20 failed: 5", S([(80, True), (20, True)]), 5),
        ("80 failed, 20 did not fail: 40", S([(80, True), (20, False)]), 40),
        ("80 did not fail, 320 failed: 160", S([(80, False), (320, True)]), 160),
        ("80 did not fail, 320 did not fail: 1280", S([(80, False), (320, False)]), 1280),
        ("after three tests: done", S([(80, False), (320, False), (1280, True)]), "done"),
    ]
    good = all(ok(f"4 search rule: {n}", got == want, {"got": got}) for n, got, want in cases)
    bad = 0
    for x in ([(20, True)], [(80, True), (320, True)]):
        try:
            S(x)
        except ValueError:
            bad += 1
    return good & ok("4 search rule: an impossible history is refused", bad == 2, bad)


# ------------------------------------------------------------------ 5. the driver, toy mode
def _toy(name, spec, runs=None, stop_after=None, fresh=True):
    d = os.path.join(TMP, "toy", name)
    if fresh:
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    sp = os.path.join(d, "spec.json")
    with open(sp, "w") as fh:
        json.dump(dict({"hour_seconds": 0.01, "hours": {"policy": 0.5}}, **spec), fh)
    args = ["./py", "scripts/d/ladder.py", "run", "--toy", sp, "--state-dir", d]
    if runs:
        args += ["--runs", ",".join(runs)]
    if stop_after:
        args += ["--stop-after", str(stop_after)]
    r = subprocess.run(args, cwd=AIR, env=ENV, capture_output=True, text=True)
    st = {}
    for f in os.listdir(d):
        if f.endswith(".json") and f not in ("spec.json", "toy_world.json", "job.json"):
            st[f[:-5]] = json.load(open(os.path.join(d, f)))
    return r.returncode, st, r


def _path(s):
    return [(t["steps"], t["verdict"]) for t in s["tests"]]


def _seeds_ok(states):
    """Seeds 1 and 2 for every tested simulator, 3 only after a split, in order."""
    for s in states.values():
        for t in s.get("tests", []):
            seeds = [p["seed"] for p in t["policies"]]
            f = [p["failed"] for p in t["policies"]]
            if t["verdict"] is None:
                continue
            if seeds[:2] != [1, 2]:
                return False
            if len(seeds) == 3 and not (seeds[2] == 3 and f[0] != f[1]):
                return False
            if len(seeds) == 2 and f[0] != f[1]:
                return False
    return True


def t5():
    F, N = "failed", "did not fail"
    good = True
    # a: four search paths; W_s0 finds no boundary (W_s1 skipped), M_s0 does (M_s1 runs)
    rc, st, r = _toy("a", {"fail_from": {"W_s0": 5, "M_s0": 40, "Wtf_s0": 160, "Wg_s0": None, "M_s1": 20,
                                         "Mg_s0": 1280}})
    good &= ok("5a toy job exits 0", rc == 0, r.stderr[-500:])
    good &= ok("5a path 80 F, 20 F -> 5", _path(st["W_s0"]) == [(80, F), (20, F), (5, F)], _path(st["W_s0"]))
    good &= ok("5a path 80 F, 20 N -> 40", _path(st["M_s0"]) == [(80, F), (20, N), (40, F)], _path(st["M_s0"]))
    good &= ok("5a path 80 N, 320 F -> 160", _path(st["Wtf_s0"]) == [(80, N), (320, F), (160, F)], _path(st["Wtf_s0"]))
    good &= ok("5a path 80 N, 320 N -> 1280", _path(st["Wg_s0"]) == [(80, N), (320, N), (1280, N)], _path(st["Wg_s0"]))
    good &= ok("5a start condition false: W_s1 skipped (W_s0 found no boundary)",
               st["W_s1"]["status"] == "skipped" and "no boundary" in st["W_s1"]["reason"], st["W_s1"].get("reason"))
    good &= ok("5a start condition true: M_s1 ran (M_s0 found a boundary)", st["M_s1"]["status"] == "done", "")
    good &= ok("5a seeds 1 and 2 for every tested simulator; no seed 3 without a split", _seeds_ok(st), "")
    world = json.load(open(os.path.join(TMP, "toy", "a", "toy_world.json")))
    ft = [c for c in world["calls"] if c[0] == "finetune"]
    good &= ok("5a fine-tuned only as far as the next test needs (W_s0 never past 80)",
               max(c[2] for c in ft if c[1] == "W_s0") == 80 and world["step"]["W_s0"] == 80, world["step"])
    lines = [x for x in open(os.path.join(TMP, "toy", "a", "d_ladder.txt")) if "__p" in x]
    npol = sum(len(t["policies"]) for s in st.values() for t in s.get("tests", []))
    good &= ok("5a one text line per measured policy", len(lines) == npol, (len(lines), npol))
    # b: all seven runs when both start conditions hold
    rc, st, r = _toy("b", {"fail_from": {"W_s0": 40, "M_s0": 160, "Wtf_s0": 5, "W_s1": 80, "M_s1": None,
                                         "Wg_s0": 320, "Mg_s0": 20}})
    good &= ok("5b both start conditions hold: all seven runs done",
               rc == 0 and all(st[k]["status"] == "done" for k in ("W_s0", "M_s0", "Wtf_s0", "W_s1", "M_s1", "Wg_s0",
                                                                   "Mg_s0")),
               {k: v["status"] for k, v in st.items()})
    # c: a run whose loss is not finite
    rc, st, r = _toy("c", {"nonfinite": {"M_s0": 100}, "fail_from": {"W_s0": 40}}, runs=["W_s0", "M_s0"])
    m = st["M_s0"]
    good &= ok("5c a loss that is not finite ends the run; its tested simulator stays",
               rc == 0 and m["status"] == "ended" and _path(m) == [(80, N)] and "not finite" in m["ended"]["why"],
               {"status": m["status"], "path": _path(m), "ended": m.get("ended")})
    # d: the sanity gate
    rc, st, r = _toy("d", {"nonfinite": {"W_s0": 50, "M_s0": 70}}, runs=["W_s0", "M_s0", "Wtf_s0"])
    good &= ok("5d sanity gate: neither W_s0 nor M_s0 reaches 80 with finite weights -> exit 3",
               rc == 3 and st["W_s0"]["status"] == "ended" and st["M_s0"]["status"] == "ended", (rc, r.stdout[-300:]))
    rc, st, r = _toy("d2", {"nonfinite": {"W_s0": 50}}, runs=["W_s0", "M_s0"])
    good &= ok("5d only one of them fails before 80: no gate", rc == 0 and st["M_s0"]["status"] == "done", rc)
    # e: the 56-hour limit; started runs finish their tests
    rc, st, r = _toy("e", {"hours": {"policy": 10.0}})
    later = ("Wtf_s0", "W_s1", "M_s1", "Wg_s0", "Mg_s0")
    good &= ok("5e no run starts after 56 hours; the two started runs finished all three tests past 56 h",
               rc == 0 and all(st[k]["status"] == "done" and len(st[k]["tests"]) == 3 and st[k]["finished_hours"] > 56
                               for k in ("W_s0", "M_s0"))
               and all(st[k]["status"] == "skipped" and ("56 hours" in st[k]["reason"] or "did not run" in
                                                         st[k]["reason"] or "no boundary" in st[k]["reason"])
                       for k in later) and any("56 hours" in st[k]["reason"] for k in later),
               {k: (st[k]["status"], st[k].get("reason"), round(st[k].get("finished_hours", 0), 1)) for k in st})
    # f: splits, a stopped policy, a score of exactly 0.15
    rc, st, r = _toy("f", {"scores": {"W_s0_d80__p1": 0.6, "W_s0_d80__p2": 0.1, "W_s0_d80__p3": 0.05,
                                      "M_s0_d80__p1": 0.05, "M_s0_d80__p2": 0.5, "M_s0_d80__p3": 0.5,
                                      "Wtf_s0_d80__p1": 0.9, "Wtf_s0_d80__p2": 0.15, "Wtf_s0_d80__p3": 0.4},
                           "stopped": ["Wtf_s0_d80__p1"]}, runs=["W_s0", "M_s0", "Wtf_s0"])
    t = {k: st[k]["tests"][0] for k in ("W_s0", "M_s0", "Wtf_s0")}
    good &= ok("5f a split settled by the third policy: failed", t["W_s0"]["verdict"] == F and
               len(t["W_s0"]["policies"]) == 3, t["W_s0"]["verdict"])
    good &= ok("5f a split settled by the third policy: did not fail", t["M_s0"]["verdict"] == N and
               len(t["M_s0"]["policies"]) == 3, t["M_s0"]["verdict"])
    pw = t["Wtf_s0"]["policies"]
    good &= ok("5f a stopped policy (score 0.9) fails; a score of exactly 0.15 does not; so a third policy",
               pw[0]["status"] == "stopped" and pw[0]["failed"] and pw[1]["score"] == 0.15 and not pw[1]["failed"]
               and len(pw) == 3 and t["Wtf_s0"]["verdict"] == N, [(p["status"], p["score"], p["failed"]) for p in pw])
    good &= ok("5f seed 3 only after a split, everywhere", _seeds_ok(st) and rc == 0, "")
    # g: the crash guard, both ways
    rc, st, r = _toy("g", {"crashes": {"W_s0_d80__p1": 2, "W_s0_d80__p2": 1}}, runs=["W_s0"])
    p = st["W_s0"]["tests"][0]["policies"]
    good &= ok("5g crash twice, guard clean: failed by its simulator (score 0, falls 1); one crash then fine: trained",
               rc == 0 and p[0]["status"] == "failed_by_simulator" and p[0]["score"] == 0.0 and p[0]["fall_rate"] == 1.0
               and p[0]["failed"] and p[0]["crashes"] == 2 and p[1]["status"] == "finished" and p[1]["crashes"] == 1,
               [(q["status"], q["score"], q["crashes"]) for q in p])
    rc, st, r = _toy("g2", {"crashes": {"W_s0_d80__p1": 2}, "guard_crash": True}, runs=["W_s0"])
    p = st["W_s0"]["tests"][0]["policies"]
    world = json.load(open(os.path.join(TMP, "toy", "g2", "toy_world.json")))
    good &= ok("5g guard crashes on REL too: job error (exit 1), nothing recorded as failed",
               rc == 1 and p == [] and "W_s0_d80__p1" not in world["records"], (rc, p))
    # h: stop and resume between two policies of one test
    spec = {"fail_from": {"W_s0": 40, "M_s0": 160}}
    rc0, ref, _ = _toy("h_ref", spec, runs=["W_s0", "M_s0", "Wtf_s0"])
    rc1, mid, r1 = _toy("h", spec, runs=["W_s0", "M_s0", "Wtf_s0"], stop_after=1)
    between = any(len(t["policies"]) == 1 and t["verdict"] is None for s in mid.values() for t in s.get("tests", []))
    rc2, fin, r2 = _toy("h", spec, runs=["W_s0", "M_s0", "Wtf_s0"], fresh=False)

    def strip(states):
        return {k: [(t["steps"], t["verdict"], [(q["policy"], q["score"], q["failed"]) for q in t["policies"]])
                    for t in v.get("tests", [])] for k, v in states.items()}
    lines = [x for x in open(os.path.join(TMP, "toy", "h", "d_ladder.txt")) if "__p" in x]
    pols = [x.split()[4] for x in lines]
    job = json.load(open(os.path.join(TMP, "toy", "h", "job.json")))
    good &= ok("5h stopped between two policies of one test, resumed: same result as without the stop",
               rc1 == 9 and between and rc2 == 0 and strip(fin) == strip(ref) and len(pols) == len(set(pols))
               and job["invocations"] == 2, {"rc": (rc0, rc1, rc2), "between": between, "lines": len(pols)})
    return good


# ------------------------------------------------------------------ 6. d_stats on synthetic pools
def _pool(n, chk, pools=None):
    sys.path.insert(0, SCRIPTS)
    import numpy as np
    import d_stats as D
    q = np.linspace(0, 1, n)
    rows = []
    for i in range(n):
        sc = 0.05 + 0.7 * q[i]
        ps = [{"seed": s, "score": sc + 0.001 * s, "fall_rate": 1 - q[i], "physics_return": 20 * q[i],
               "failed": sc + 0.001 * s < 0.15} for s in (1, 2)]
        c = {k: float(v[i]) for k, v in chk.items()}
        base = {"E1": 1.0, "E8": 1.0, "E24": 1.0, "E100": 1.0, "E368": 1.0, "lambda1": 1.0, "rJ": 1.0, "rA": 1.0}
        base.update(c)
        rows.append(D.make_row(f"x{i}", (pools or ["REL"] + ["ladder"] * (n - 1))[i], ps,
                               {"expert": base, "mixed": dict(base)}, {"tracking": q[i], "falls": 1 - q[i], "u": 1 - q[i]},
                               run="W_s0", steps=80))
    return D.fill_returns(rows)


def t6():
    sys.path.insert(0, SCRIPTS)
    import numpy as np
    import d_stats as D
    n = 16
    q = np.linspace(0, 1, n)
    bad = 1 - q                                    # lower is better: tracks the outcome
    noise = np.random.default_rng(7).permutation(n).astype(float)
    good = True
    cases = [
        ("R0 SUPPORTED", lambda: D.rho(_pool(n, {"E100": bad}), "E100", "score", D.R0_LAB), "SUPPORTED"),
        ("R0 CONTRADICTED", lambda: D.rho(_pool(n, {"E100": q}), "E100", "score", D.R0_LAB), "CONTRADICTED"),
        ("R0 NOT SETTLED", lambda: D.rho(_pool(n, {"E100": noise}), "E100", "score", D.R0_LAB), "NOT SETTLED"),
        ("R1 LABEL-FREE BETTER", lambda: D.diff(_pool(n, {"lambda1": bad, "E1": noise}), "lambda1", "E1", "score",
                                                D.R1_LAB, D.PRIMARY), "LABEL-FREE BETTER"),
        ("R1 ONE-STEP BETTER", lambda: D.diff(_pool(n, {"lambda1": noise, "E1": bad}), "lambda1", "E1", "score",
                                              D.R1_LAB, D.PRIMARY), "ONE-STEP BETTER"),
        ("R2 LABEL-FREE BETTER", lambda: D.diff(_pool(n, {"rJ": bad, "E8": noise}), "rJ", "E8", "score", D.R2_LAB,
                                                D.PRIMARY), "LABEL-FREE BETTER"),
        ("R2 TRAINING-HORIZON BETTER", lambda: D.diff(_pool(n, {"rJ": noise, "E8": bad}), "rJ", "E8", "score",
                                                      D.R2_LAB, D.PRIMARY), "TRAINING-HORIZON BETTER"),
        ("NOT COMPUTABLE: a check with all values tied", lambda: D.diff(_pool(n, {"rJ": np.ones(n), "E8": bad}), "rJ",
                                                                        "E8", "score", D.R2_LAB, D.PRIMARY),
         "NOT COMPUTABLE"),
        ("NOT COMPUTABLE: a pool of three", lambda: D.rho(_pool(3, {"E100": bad[:3]}), "E100", "score", D.R0_LAB),
         "NOT COMPUTABLE"),
    ]
    for name, fn, want in cases:
        o = fn()
        good &= ok(f"6 d_stats: {name}", o["label"] == want, {k: o.get(k) for k in ("label", "why", "rho", "delta",
                                                                                    "ci95", "ci97.5")})
    rows = _pool(n, {"E100": bad, "E1": bad})
    for r in rows[:5]:
        r["failed"] = True
    a = D.auc(rows, "E100")
    good &= ok("6 d_stats: 'failed' separated by a check that tracks the score: AUC 1, ABOVE CHANCE",
               a["label"] == "ABOVE CHANCE" and a["auc"] == 1.0, a)
    # a pool of four, the whole analysis
    four = _pool(4, {"E100": [1, 3, 2, 4], "E1": [1, 2, 3, 4], "lambda1": [0.1, 0.3, 0.2, 0.4], "E8": [4, 3, 2, 1],
                     "rJ": [2, 1, 4, 3], "E368": [1, 2, float("inf"), 4]},
                 pools=["REL", "ladder", "ladder", "ladder"])
    four[1]["policies"][1].update(physics_return=None, status="failed_by_simulator", score=0.0, fall_rate=1.0)
    four[2]["policies"][0]["physics_return"] = None          # measured, but its physics return is missing
    D.fill_returns(four)
    t0 = time.time()
    o = D.analyse(four, ["synthetic"])
    txt = D.report(o)
    good &= ok("6 d_stats: a pool of four runs end to end and says plainly what cannot be computed",
               "Premise" in txt and "best three: NOT COMPUTABLE" in txt and o["premise"]["R0"]["label"] and
               all(v["label"] for v in o["primary"].values()) and "NOT COMPUTABLE (" in txt,
               {"seconds": round(time.time() - t0), "R0": o["premise"]["R0"]["label"],
                "R1": o["primary"]["R1"]["label"], "R2": o["primary"]["R2"]["label"]})
    good &= ok("6 d_stats: a failed-by-simulator policy gets the lowest measured physics return; a missing one "
               "leaves its simulator out of the physics-return analyses",
               abs(four[1]["ret"] - np.mean([four[1]["policies"][0]["physics_return"], 0.0])) < 1e-12
               and np.isnan(four[2]["ret"]) and "x2" in o["secondary"]["physics_return"]["R0"].get("left_out", ["x2"]),
               (four[1]["ret"], four[2]["ret"], o["secondary"]["physics_return"]["R0"].get("why")))
    return good


GROUPS = {"1": t1, "2": t2, "3": t3, "4": t4, "5": t5, "6": t6}


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--group":
        sys.exit(0 if GROUPS[sys.argv[2]]() else 1)
    os.makedirs(TMP, exist_ok=True)
    t0 = time.time()
    res = {}
    for g in GROUPS:
        t = time.time()
        r = subprocess.run(["./py", "scripts/d/test_d1.py", "--group", g], cwd=AIR, env=ENV, capture_output=True,
                           text=True)
        lines = [json.loads(x) for x in r.stdout.splitlines() if x.startswith("{\"test\"")]
        res[g] = {"passed": r.returncode == 0 and bool(lines) and all(x["passed"] for x in lines),
                  "seconds": round(time.time() - t, 1), "tests": lines,
                  "stderr_tail": r.stderr[-600:] if r.returncode else ""}
        for x in lines:
            print(("PASS " if x["passed"] else "FAIL ") + x["test"] + ("" if x["passed"] else f"  {x['detail']}"))
        if r.returncode and not lines:
            print(f"group {g} crashed: {r.stderr[-600:]}")
    out = {"passed": all(v["passed"] for v in res.values()), "seconds": round(time.time() - t0, 1), "groups": res,
           "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps({"passed": out["passed"], "seconds": out["seconds"],
                      "groups": {g: (v["passed"], v["seconds"]) for g, v in res.items()}}))
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()

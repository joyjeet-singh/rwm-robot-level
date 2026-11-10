"""Option C (C2R): the second pilot's gates, applied mechanically (OPTIONC_PLAN.md, Amendment 1, changes 2 and 6-9).

    ./py scripts/c/pilot2_rule.py a2                      # which simulators: "<6M> <10k>", "retrain", or BLOCKED (exit 3)
    ./py scripts/c/pilot2_rule.py saves <policy>          # the saves a trained policy kept (a collapsed arm: those so far)
    ./py scripts/c/pilot2_rule.py choose --sims "A B"     # Gate B' and the iteration rule': "<penalty> <fall> <iters> <10k
                                                          #   policy>", or BLOCKED (exit 3); results/c/pilot2_decision.json
    ./py scripts/c/pilot2_rule.py report --sims "A B"     # Gate C' and results/c_pilot.json and .txt
    ./py scripts/c/pilot2_rule.py selftest                # every branch on fixtures

  * Gate A2 (results/c/calibration.json, then results/c/calibration_2S.json after the retraining with 2 x S steps):
    passed -> the pilot uses those simulators; failed once -> "retrain"; failed twice -> BLOCKED.
  * Arms P1-P4 in the 6M simulator, seed 0, saves at 500, 1,000 and 1,500: P1 penalty lambda_6M, P2 lambda_6M and fall
    cost 4.0, P3 penalty 0, P4 penalty 0 and fall cost 4.0. An arm stopped early ("collapsed") keeps its saves so far.
  * Gate B': check P1, then P2, P3, P4. An arm passes if its best save (the highest walking score; on a tie, the earlier
    save) falls in under 20% of episodes and scores at least half the authors' policy. The first to pass sets the
    recipe; if none passes: BLOCKED.
  * Iteration rule': the smallest saved count at which the chosen arm scores at least 90% of its best save.
  * Gate C' (not a stop): the 10k policy, trained with the chosen recipe and iteration count, scores below the chosen
    arm's best save, with their 95% intervals over episodes apart. If not: add the 2k rung, recorded before registering.

Every value comes from result files (scripts/c/evaluate.py, scripts/c/train_policy.py, scripts/c/calibrate_penalty.py).
The world is the hand-built calibrated MuJoCo ANYmal D with a fitted PD motor, not Isaac Lab and not hardware. Results
are written once; a re-run must reach the same answer.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

AIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_SIMS = ("sim_6M_N8_s0", "sim_10k_N8_s0")
RETRAINED = ("sim_6M_N8_s0__2S", "sim_10k_N8_s0__2S")
ARMS = (("P1", "cal", 0.0), ("P2", "cal", 4.0), ("P3", "0", 0.0), ("P4", "0", 4.0))
SAVES = (500, 1000, 1500)
FALL_LIMIT = 0.20
SCORE_SHARE = 0.5
ITER_SHARE = 0.9


def recipe_name(sim, seed, penalty, fall_cost):           # = scripts/c/train_policy.recipe_name
    tag = "cal" if penalty == "cal" else f"w{float(penalty):g}"
    if fall_cost:
        tag += f"_fall{fall_cost:g}"
    return f"{sim}__{tag}__p{seed}"


class Paths:
    def __init__(self, root):
        self.root = root
        r = lambda *a: os.path.join(root, *a)   # noqa: E731
        self.cal, self.cal2 = r("results", "c", "calibration.json"), r("results", "c", "calibration_2S.json")
        self.eval_dir, self.train_dir = r("results", "c", "eval"), r("results", "c", "train")
        self.authors = r("results", "c_authors.json")
        self.decision = r("results", "c", "pilot2_decision.json")
        self.out_json, self.out_txt = r("results", "c_pilot.json"), r("results", "c_pilot.txt")
        self.decisions, self.log = r("optc", "DECISIONS.md"), r("optc", "LOG.md")


def write_once(path, obj):
    if os.path.exists(path):
        if json.load(open(path)) != json.loads(json.dumps(obj)):
            raise SystemExit(f"{path} exists and differs from this run's answer; result files are never overwritten")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(path + ".tmp", path)


def write_text_once(path, text):
    if os.path.exists(path):
        if open(path).read() != text:
            raise SystemExit(f"{path} exists and differs from this run's answer")
        return
    with open(path, "w") as fh:
        fh.write(text)


def mark_blocked(p, mark, what, options, recommendation):
    old = open(p.decisions).read() if os.path.exists(p.decisions) else ""
    if mark in old:
        return
    stamp = time.strftime("%Y-%m-%d %H:%M")
    os.makedirs(os.path.dirname(p.decisions), exist_ok=True)
    with open(p.decisions, "a") as fh:
        fh.write(f"\n{mark} ({stamp})\n\n**What is blocked:** {what}\n\n**Options:** {options}\n\n"
                 f"**Recommendation:** {recommendation}\n")
    with open(p.log, "a") as fh:
        fh.write(f"\n## Pilot job status: BLOCKED ({stamp})\n{what} See `optc/DECISIONS.md` and `results/c_pilot.txt`.\n")


# ------------------------------------------------------------------ Gate A2
def a2_state(p):
    """('use', sims) | ('retrain', None) | ('blocked', info)."""
    c1 = json.load(open(p.cal))
    if c1["gate_a2"]["passed"]:
        return "use", BASE_SIMS, c1["gate_a2"], None
    if not os.path.exists(p.cal2):
        return "retrain", None, c1["gate_a2"], None
    c2 = json.load(open(p.cal2))
    if c2["gate_a2"]["passed"]:
        return "use", RETRAINED, c1["gate_a2"], c2["gate_a2"]
    return "blocked", None, c1["gate_a2"], c2["gate_a2"]


def gate_a2(p):
    state, sims, g1, g2 = a2_state(p)
    if state == "use":
        print(" ".join(sims))
        return 0
    if state == "retrain":
        print("retrain")
        return 0
    fmt = lambda g: f"{g['simulator']}: fell {g['share_fell']:.1%}, tracking {g['tracking_ratio']:.0%} of MuJoCo's"  # noqa
    what = (f"Gate A2 failed twice ({fmt(g1)}; after retraining with 2 x S steps, {fmt(g2)}; limits: at most 10% falls "
            "and at least 80% of MuJoCo's tracking). The simulator cannot hold the authors' walk in imagination; the "
            "plan says to stop.")
    res = {"what": "Option C second pilot: BLOCKED at Gate A2 (hand-built calibrated MuJoCo ANYmal D, fitted PD motor; "
                   "not Isaac Lab, not hardware)", "status": "BLOCKED", "blocked_at": "Gate A2",
           "gate_a2": {"first": g1, "after_2S": g2}}
    write_once(p.out_json, res)
    write_text_once(p.out_txt, "Option C second pilot: BLOCKED at Gate A2\n" + what + "\n")
    mark_blocked(p, "## Pilot job: BLOCKED at Gate A2", what,
                 "(a) stop Option C and report Gate A2 as the finding (the simulators, even with twice the training, "
                 "do not hold the authors' walk); (b) a further change to the simulator's training (more steps, a "
                 "different loss weighting or data mix), which is a change to the fixed design and needs the author.",
                 "(a), unless the author chooses a further simulator change.")
    print("BLOCKED")
    return 3


# ------------------------------------------------------------------ arms
def train_record(p, name):
    """The trainer's record: (status, saves) or (None, []) if the policy was not trained."""
    rec = os.path.join(p.train_dir, name + ".recipe.json")
    res = os.path.join(p.train_dir, name + ".json")
    if os.path.exists(res):
        r = json.load(open(res))
        if r.get("status") == "collapsed":
            return "collapsed", list(r["saves"]), r.get("collapsed")
        if os.path.exists(rec):
            return "finished", list(json.load(open(rec))["saves"]), None
    return None, [], None


def load_eval(p, name, n):
    path = os.path.join(p.eval_dir, f"{name}__i{n}.json")
    if not os.path.exists(path):
        raise SystemExit(f"missing result: {path}")
    r = json.load(open(path))
    s = r["summary"]
    return {"score": r["score"], "score_ci95": r["score_ci95"], "fall_rate": s["fall_rate"], "ret": s["ret"],
            "speed_err": s["speed_err"], "turn_err": s["turn_err"], "diverged": s["diverged"]}


def arm_table(p, sim6):
    out = {}
    for arm, pen, fall in ARMS:
        name = recipe_name(sim6, 0, pen, fall)
        status, saves, coll = train_record(p, name)
        if status is None:
            raise SystemExit(f"arm {arm} ({name}) has no training record")
        out[arm] = {"policy": name, "penalty": pen, "fall_cost": fall, "status": status, "collapsed": coll,
                    "saves": {str(n): load_eval(p, name, n) for n in saves}}
    return out


def gate_b_prime(arm, authors):
    if not arm["saves"]:
        return {"best_save": None, "passed": False, "why": "no save (stopped before iteration 500)"}
    best_n = max(arm["saves"], key=lambda n: (arm["saves"][n]["score"], -int(n)))
    b = arm["saves"][best_n]
    fall_ok, score_ok = b["fall_rate"] < FALL_LIMIT, b["score"] >= SCORE_SHARE * authors
    return {"best_save": int(best_n), "score": b["score"], "fall_rate": b["fall_rate"], "fall_below_limit": fall_ok,
            "score_at_least_half": score_ok, "passed": fall_ok and score_ok}


def choose(p, sims):
    sim6, sim10 = sims
    authors = json.load(open(p.authors))["score"]
    arms = arm_table(p, sim6)
    gates = {a: gate_b_prime(arms[a], authors) for a in arms}
    chosen = next((a for a, _, _ in ARMS if gates[a]["passed"]), None)
    dec = {"what": "Gate B' and the iteration rule' (Amendment 1)", "simulators": list(sims), "authors_score": authors,
           "gate_b_prime": gates, "order": [a for a, _, _ in ARMS], "chosen_arm": chosen}
    if chosen is not None:
        arm = arms[chosen]
        best = gates[chosen]["score"]
        iters = min(int(n) for n, r in arm["saves"].items() if r["score"] >= ITER_SHARE * best)
        dec.update(penalty=arm["penalty"], fall_cost=arm["fall_cost"], iterations=iters,
                   iteration_rule={"best_score": best, "threshold": ITER_SHARE * best},
                   policy_10k=recipe_name(sim10, 0, arm["penalty"], arm["fall_cost"]))
    write_once(p.decision, dec)
    if chosen is None:
        res = assemble(p, sims, arms, dec, None)
        finish(p, res)
        mark_blocked(p, "## Pilot job: BLOCKED at Gate B'", res["blocked_because"],
                     "(a) stop Option C and report both pilots as they stand; (b) a further change to the policy "
                     "recipe or the simulators, which is a change to the fixed design and needs the author.",
                     "(a), reporting `results/c_pilot.txt`, unless the author chooses otherwise.")
        print("BLOCKED")
        return 3
    print(f"{dec['penalty']} {dec['fall_cost']:g} {dec['iterations']} {dec['policy_10k']}")
    return 0


def assemble(p, sims, arms, dec, gate_c):
    authors_rec = json.load(open(p.authors))
    state, _, g1, g2 = a2_state(p)
    res = {"what": "Option C second pilot (Amendment 1): Gate A2, Gate B', iteration rule', Gate C' (hand-built "
                   "calibrated MuJoCo ANYmal D, fitted PD motor; not Isaac Lab, not hardware)",
           "gate_a2": {"first": g1, "after_2S": g2}, "simulators": list(sims),
           "authors": {"score": authors_rec["score"], "score_ci95": authors_rec["score_ci95"],
                       "fall_rate": authors_rec["summary"]["fall_rate"]},
           "standing_reference": authors_rec["standing_reference"], "arms": arms, "decision": dec}
    if dec["chosen_arm"] is None:
        res.update(status="BLOCKED", blocked_at="Gate B'",
                   blocked_because="no arm of the second pilot passes Gate B' (best save under 20% falls and at "
                                   "least half the authors' score); the plan says to stop.")
    else:
        res.update(status="PASSED", gate_c_prime=gate_c)
    return res


def report(p, sims):
    dec = json.load(open(p.decision))
    arms = arm_table(p, sims[0])
    if dec["chosen_arm"] is None:
        finish(p, assemble(p, sims, arms, dec, None))
        return 3
    a = arms[dec["chosen_arm"]]
    best = a["saves"][str(dec["gate_b_prime"][dec["chosen_arm"]]["best_save"])]
    r10 = load_eval(p, dec["policy_10k"], dec["iterations"])
    st10 = train_record(p, dec["policy_10k"])
    apart = r10["score_ci95"][1] < best["score_ci95"][0]
    gc = {"policy_10k": dec["policy_10k"], "iterations": dec["iterations"], "status_10k": st10[0],
          "score_10k": r10["score"], "ci95_10k": r10["score_ci95"], "fall_rate_10k": r10["fall_rate"],
          "score_6M_best": best["score"], "ci95_6M_best": best["score_ci95"],
          "below": r10["score"] < best["score"], "intervals_apart": apart,
          "passed": r10["score"] < best["score"] and apart}
    gc["action"] = ("none" if gc["passed"] else
                    "add a 2k rung (rollout-trained seeds 0 and 1, teacher-forced seed 0) and record it before registering")
    finish(p, assemble(p, sims, arms, dec, gc))
    return 0


def finish(p, res):
    write_once(p.out_json, res)
    text = render(res)
    write_text_once(p.out_txt, text)
    print(text)


def render(res):
    a = res["authors"]
    L = ["Option C second pilot (calibrated hand-built MuJoCo ANYmal D, fitted PD motor; not Isaac Lab, not hardware)"]
    for k, g in res["gate_a2"].items():
        if g:
            L.append(f"Gate A2 ({k}, {g['simulator']}): {'passed' if g['passed'] else 'FAILED'}; authors' policy in "
                     f"imagination fell {g['share_fell']:.1%}, tracking {g['tracking_ratio']:.0%} of MuJoCo's")
    L.append(f"Authors' policy: score {a['score']:.3f} ({a['score_ci95'][0]:.3f} to {a['score_ci95'][1]:.3f}), falls "
             f"{a['fall_rate']:.0%}; standing still {res['standing_reference']:.3f}; half the authors' {SCORE_SHARE * a['score']:.3f}")
    L.append(f"{'arm':4s} {'policy':36s} {'iter':>5s} {'score':>6s} {'95% interval':>15s} {'falls':>6s}")
    for arm, d in res["arms"].items():
        if d["status"] == "collapsed":
            L.append(f"{arm:4s} {d['policy']:36s} stopped early (collapsed) at iteration {d['collapsed']['collapsed_at']}")
        for n, r in d["saves"].items():
            ci = f"{r['score_ci95'][0]:.3f}-{r['score_ci95'][1]:.3f}"
            L.append(f"{arm:4s} {d['policy']:36s} {n:>5s} {r['score']:6.3f} {ci:>15s} {r['fall_rate']:6.0%}")
    dec = res["decision"]
    for arm, g in dec["gate_b_prime"].items():
        if g["best_save"] is None:
            L.append(f"Gate B' {arm}: failed ({g['why']})")
        else:
            L.append(f"Gate B' {arm}: {'passed' if g['passed'] else 'failed'} (best save {g['best_save']}: score "
                     f"{g['score']:.3f}, falls {g['fall_rate']:.0%})")
    L.append(f"Status: {res['status']}")
    if res["status"] == "BLOCKED":
        L.append(res["blocked_because"])
    else:
        L.append(f"Chosen: {dec['chosen_arm']} (penalty {'lambda_s' if dec['penalty'] == 'cal' else '0'}, fall cost "
                 f"{dec['fall_cost']:g}); main study uses {dec['iterations']} iterations")
        c = res["gate_c_prime"]
        L.append(f"Gate C': {'passed' if c['passed'] else 'NOT passed'} (10k {c['score_10k']:.3f}, 6M best "
                 f"{c['score_6M_best']:.3f}, intervals {'apart' if c['intervals_apart'] else 'overlap'}); action: {c['action']}")
    assert len(L) <= 40, len(L)
    return "\n".join(L) + "\n"


def saves(p, name):
    status, s, _ = train_record(p, name)
    if status is None:
        raise SystemExit(f"{name}: no training record")
    print(" ".join(str(n) for n in s))
    return 0


# ------------------------------------------------------------------ fixtures
def self_test():
    def rec(score, fall, ci=0.02):
        return {"score": score, "score_ci95": [score - ci, score + ci], "standing_reference": 0.07,
                "summary": {"fall_rate": fall, "ret": 1.0, "speed_err": 0.1, "turn_err": 0.1, "diverged": 0}}

    def a2rec(sim, passed):
        return {"gate_a2": {"simulator": sim, "share_fell": 0.05 if passed else 0.2, "tracking_ratio": 0.9 if passed else 0.3,
                            "passed": passed}}

    def setup(root, a2=(True,), arms=None, ten=None, authors=0.9):
        for d in ("results/c/eval", "results/c/train", "optc"):
            os.makedirs(os.path.join(root, d), exist_ok=True)
        json.dump(rec(authors, 0.0), open(os.path.join(root, "results/c_authors.json"), "w"))
        json.dump(a2rec("sim_6M_N8_s0", a2[0]), open(os.path.join(root, "results/c/calibration.json"), "w"))
        if len(a2) > 1:
            json.dump(a2rec("sim_6M_N8_s0__2S", a2[1]), open(os.path.join(root, "results/c/calibration_2S.json"), "w"))
        sims = BASE_SIMS if a2[0] else RETRAINED
        for arm, pen, fall in ARMS:
            spec = (arms or {}).get(arm, {"saves": {500: (0.1, 0.9), 1000: (0.1, 0.9), 1500: (0.1, 0.9)}})
            put_policy(root, recipe_name(sims[0], 0, pen, fall), spec)
        if ten is not None:
            put_policy(root, ten[0], {"saves": {ten[1]: ten[2]}})
        return sims

    def put_policy(root, name, spec):
        sv = sorted(spec["saves"])
        if spec.get("collapsed"):
            json.dump({"status": "collapsed", "saves": sv, "collapsed": {"collapsed_at": spec["collapsed"]}},
                      open(os.path.join(root, "results/c/train", name + ".json"), "w"))
        else:
            json.dump({"policy": name}, open(os.path.join(root, "results/c/train", name + ".json"), "w"))
            json.dump({"saves": sv}, open(os.path.join(root, "results/c/train", name + ".recipe.json"), "w"))
        for n, (score, fall) in spec["saves"].items():
            json.dump(rec(score, fall), open(os.path.join(root, "results/c/eval", f"{name}__i{n}.json"), "w"))

    def quiet(fn, *a):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = fn(*a)
        return rc, buf.getvalue().strip().splitlines()[-1] if buf.getvalue().strip() else ""

    tmp = tempfile.mkdtemp(prefix="c2r_rule_")
    try:
        def case(tag):
            root = os.path.join(tmp, tag)
            os.makedirs(root)
            return root, Paths(root)

        # Gate A2: passed; failed once -> retrain; 2S passed; failed twice -> BLOCKED
        r, p = case("a2_pass"); setup(r, (True,))                                  # noqa: E702
        assert quiet(gate_a2, p) == (0, "sim_6M_N8_s0 sim_10k_N8_s0")
        r, p = case("a2_retrain"); setup(r, (False,))                              # noqa: E702
        assert quiet(gate_a2, p) == (0, "retrain")
        r, p = case("a2_2S"); setup(r, (False, True))                              # noqa: E702
        assert quiet(gate_a2, p) == (0, "sim_6M_N8_s0__2S sim_10k_N8_s0__2S")
        r, p = case("a2_block"); setup(r, (False, False))                          # noqa: E702
        assert quiet(gate_a2, p) == (3, "BLOCKED")
        assert json.load(open(p.out_json))["blocked_at"] == "Gate A2" and "Gate A2" in open(p.decisions).read()
        assert "BLOCKED" in open(p.log).read()
        print("ok: Gate A2 branches: pass, retrain, pass after 2 x S, blocked (records written)")

        # Gate B': P1 passes; iteration rule picks the smallest save at >= 90% of the best
        r, p = case("p1"); sims = setup(r, arms={"P1": {"saves": {500: (0.43, 0.1), 1000: (0.6, 0.05), 1500: (0.58, 0.0)}}})  # noqa: E702
        assert quiet(choose, p, sims) == (0, "cal 0 1000 sim_10k_N8_s0__cal__p0")
        put_policy(r, "sim_10k_N8_s0__cal__p0", {"saves": {500: (0.2, 0.5), 1000: (0.3, 0.4)}})
        assert quiet(report, p, sims)[0] == 0
        res = json.load(open(p.out_json))
        assert res["status"] == "PASSED" and res["gate_c_prime"]["passed"] and res["gate_c_prime"]["action"] == "none"
        print("ok: P1 passes; iteration rule' takes 1,000 (0.60 best, 0.54 needed); Gate C' passed (intervals apart)")

        r, p = case("p1_iter500"); sims = setup(r, arms={"P1": {"saves": {500: (0.55, 0.1), 1000: (0.6, 0.05), 1500: (0.59, 0.0)}}})  # noqa: E702
        assert quiet(choose, p, sims)[1].split()[2] == "500"
        print("ok: iteration rule' takes 500 when the first save is within 90% of the best")

        # P1 fails on falls at its best save (a lower save would pass: the best save decides), P2 passes
        r, p = case("p2"); sims = setup(r, arms={  # noqa: E702
            "P1": {"saves": {500: (0.5, 0.1), 1000: (0.7, 0.25), 1500: (0.6, 0.3)}},
            "P2": {"saves": {500: (0.46, 0.15), 1000: (0.46, 0.1), 1500: (0.2, 0.5)}}})
        assert quiet(choose, p, sims) == (0, "cal 4 500 sim_10k_N8_s0__cal_fall4__p0")
        d = json.load(open(p.decision))
        assert d["gate_b_prime"]["P1"]["best_save"] == 1000 and not d["gate_b_prime"]["P1"]["fall_below_limit"]
        assert d["gate_b_prime"]["P2"]["best_save"] == 500          # a tie goes to the earlier save
        print("ok: P1 fails at its best save on falls; P2 passes (tie between saves goes to the earlier one)")

        # only P4 passes (P1 collapsed without saves, P2 collapsed with one, P3 scores too low)
        r, p = case("p4"); sims = setup(r, (False, True), arms={  # noqa: E702
            "P1": {"saves": {}, "collapsed": 400},
            "P2": {"saves": {500: (0.1, 0.9)}, "collapsed": 650},
            "P3": {"saves": {500: (0.3, 0.0), 1000: (0.44, 0.0), 1500: (0.449, 0.0)}},
            "P4": {"saves": {500: (0.3, 0.1), 1000: (0.5, 0.1), 1500: (0.52, 0.12)}}})
        assert quiet(choose, p, sims) == (0, "0 4 1000 sim_10k_N8_s0__2S__w0_fall4__p0")
        put_policy(r, "sim_10k_N8_s0__2S__w0_fall4__p0", {"saves": {500: (0.5, 0.1), 1000: (0.49, 0.1)}})
        assert quiet(report, p, sims)[0] == 0
        res = json.load(open(p.out_json))
        gc = res["gate_c_prime"]
        assert gc["below"] and not gc["intervals_apart"] and not gc["passed"] and gc["action"].startswith("add a 2k")
        assert res["arms"]["P1"]["status"] == "collapsed" and res["decision"]["gate_b_prime"]["P1"]["best_save"] is None
        print("ok: collapsed arms (with and without saves) fail; P4 chosen in the 2 x S simulators; Gate C' not passed "
              "when the intervals overlap -> add the 2k rung")

        r, p = case("c_higher"); sims = setup(r, arms={"P1": {"saves": {500: (0.6, 0.0), 1000: (0.6, 0.0), 1500: (0.6, 0.0)}}})  # noqa: E702
        quiet(choose, p, sims)
        put_policy(r, "sim_10k_N8_s0__cal__p0", {"saves": {500: (0.8, 0.0)}})
        quiet(report, p, sims)
        gc = json.load(open(p.out_json))["gate_c_prime"]
        assert not gc["below"] and not gc["passed"]
        print("ok: Gate C' not passed when the 10k policy scores higher")

        # none passes -> BLOCKED, recorded once; a re-run reaches the same answer
        r, p = case("none"); sims = setup(r)                                       # noqa: E702
        assert quiet(choose, p, sims) == (3, "BLOCKED")
        res = json.load(open(p.out_json))
        assert res["status"] == "BLOCKED" and res["blocked_at"] == "Gate B'"
        assert open(p.decisions).read().count("BLOCKED at Gate B'") == 1
        assert quiet(choose, p, sims) == (3, "BLOCKED")
        assert open(p.decisions).read().count("BLOCKED at Gate B'") == 1 and quiet(report, p, sims)[0] == 3
        print("ok: no arm passes -> BLOCKED (exit 3), recorded once in DECISIONS and LOG; re-runs agree")

        # saves helper and a changed result refuses to overwrite
        r, p = case("saves"); sims = setup(r, arms={"P2": {"saves": {500: (0.1, 0.9)}, "collapsed": 700}})  # noqa: E702
        assert quiet(saves, p, recipe_name(sims[0], 0, "cal", 4.0)) == (0, "500")
        assert quiet(saves, p, recipe_name(sims[0], 0, "0", 0.0)) == (0, "500 1000 1500")
        r, p = case("differs"); sims = setup(r, arms={"P1": {"saves": {500: (0.6, 0.0), 1000: (0.6, 0.0), 1500: (0.6, 0.0)}}})  # noqa: E702
        quiet(choose, p, sims)
        json.dump(rec(0.1, 0.9), open(os.path.join(r, "results/c/eval", f"{recipe_name(sims[0], 0, 'cal', 0)}__i500.json"), "w"))
        try:
            quiet(choose, p, sims)
            raise AssertionError("an overwrite was allowed")
        except SystemExit:
            pass
        print("ok: saves helper (finished and collapsed arms); a changed answer is refused, not overwritten")
    finally:
        shutil.rmtree(tmp)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("a2", "saves", "choose", "report", "selftest"))
    ap.add_argument("policy", nargs="?")
    ap.add_argument("--sims", default=None, help='"<6M simulator> <10k simulator>"')
    ap.add_argument("--root", default=AIR, help="fixture folder (tests only)")
    a = ap.parse_args(argv)
    p = Paths(a.root)
    if a.stage == "selftest":
        return self_test()
    if a.stage == "a2":
        return gate_a2(p)
    if a.stage == "saves":
        return saves(p, a.policy)
    sims = tuple(a.sims.split())
    if len(sims) != 2:
        raise SystemExit("--sims needs two simulators")
    return choose(p, sims) if a.stage == "choose" else report(p, sims)


if __name__ == "__main__":
    sys.exit(main())

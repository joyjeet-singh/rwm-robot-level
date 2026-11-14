"""Option D (D0): the record (Stage 1) and the record job's two sanity gates.  ./py

    ./py scripts/d/record.py table [--strict]   # -> results/d_record.json and results/d_record.txt (written once)
    ./py scripts/d/record.py gate1              # RWM's released policy scores what results/c_authors.json says (1e-6)
    ./py scripts/d/record.py gate2              # REL's two registered policies score at least 0.3 on average

The table has a row per simulator of the record (scripts/d/sims.py EXISTING): how it was made; its checks on the two
MuJoCo test sets (results/d/checks/<sim>.json); its walk test (results/d/walk/<sim>.json, secondary); and how each policy
trained in it did. Step 2's 22 policies come from results/d/eval/<policy>.json (walking score, fall rate, Step 2's
physics return), beside what Step 2 itself stored (results/s2/eval/<policy>.json). Option C's pilot policies enter from
its own result files (results/c_pilot1.json, results/c/d3_eval/), measured by Option C's recipes, not Step 2's.

Verdicts (OPTIOND_PLAN.md, Terms): a policy failed when its walking score is below 0.15; a simulator failed when at
least two of its policies failed. Given here for REL and its eight copies from their two registered policies (seeds 1,
2); descriptive. The rebuilt pilots have one policy each, and Option C's simulators had policies trained by other
recipes, so they get no verdict.
Also descriptive: whether Option D's stop rule (collapse in imagination) would have fired in each Step 2 run, from its
logged mean imagined episode length (runs/lite/s2_<policy>/metrics.jsonl).
The world is the hand-built calibrated MuJoCo ANYmal D with a fitted PD motor; not Isaac Lab, not hardware.
With --rehearsal a gate only reports. A gate that fails appends its reason to optd/DECISIONS.md and optd/LOG.md and exits with code 3.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
AIR = os.path.dirname(SCRIPTS)
sys.path.insert(0, HERE)
import sims  # noqa: E402

RES = os.path.join(AIR, "results")
FAIL_SCORE = 0.15
GATE2_MIN = 0.3
GATE1_TOL = 1e-6
STOP = {"after": 50, "run": 50, "threshold": 128.0}      # OPTIOND_PLAN.md, Policies, the stop rule
S2_SIMS = ("REL",) + sims.REL_COPIES
S2_PILOT = {"REL": "REL__p0", "A_s0": "A_s0__p0", "B_s0": "B_s0__p0", "A_s0__u50": "A_s0__u50__p0"}
CHECK_KEYS = ("E1", "E8", "E24", "E100", "E368", "floor_E1", "floor_E8", "floor_E100", "floor_E368", "rJ", "rA",
              "rJ_true", "rA_true", "lambda1", "finite", "windows_used", "episodes_too_short")


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def stop_rule_at(lengths, after=STOP["after"], run=STOP["run"], threshold=STOP["threshold"]):
    """First iteration (1-based) at which the stop rule fires, or None. lengths[i] is iteration i + 1's mean imagined
    episode length, None where the lite code logged none; a missing value breaks a run of short iterations."""
    count = 0
    for i, v in enumerate(lengths):
        n = i + 1
        if n <= after:
            continue
        count = count + 1 if (v is not None and v < threshold) else 0
        if count >= run:
            return n
    return None


def lite_lengths(policy):
    """Per-iteration Train/mean_episode_length of a Step 2 run (metrics.jsonl line n is iteration n)."""
    p = os.path.join(AIR, "runs", "lite", f"s2_{policy}", "metrics.jsonl")
    if not os.path.exists(p):
        return None
    out = []
    with open(p) as fh:
        lines = [json.loads(x) for x in fh if x.strip()]
    for d in lines[1:]:
        if "Train/mean_reward" in d or "Train/mean_episode_length" in d or "Imagination/uncertainty" in d:
            out.append(d.get("Train/mean_episode_length"))
    return out


def step2_policy(policy, sim, role, eval_dir):
    ev = _load(os.path.join(eval_dir, f"{policy}.json"))
    s2 = _load(os.path.join(RES, "s2", "eval", f"{policy}.json"))
    tr = _load(os.path.join(RES, "s2", "train", f"{policy}.json")) or _load(os.path.join(AIR, "runs", "s2", policy,
                                                                                        "train.json"))
    lengths = lite_lengths(policy)
    rec = {"policy": policy, "sim": sim, "role": role, "recipe": "Step 2 (lite code, 500 iterations, 4,096 robots)",
           "route": (tr or {}).get("route", "released" if policy == "REL__p0" else "rebuilt"),
           "measured": ev is not None}
    if ev is not None:
        rec.update(score=ev["score"], score_ci95=ev["score_ci95"], fall_rate=ev["fall_rate"],
                   physics_return=ev.get("physics_return"), physics_fall_rate=ev.get("physics_fall_rate"),
                   failed=bool(ev["score"] < FAIL_SCORE), actor_sha256=ev["actor_sha256"])
    if s2 is not None:
        s = s2.get("summary", s2)
        rec.update(step2_return=s.get("ret"), step2_fall_rate=s.get("fall_rate"))
    if tr is not None:
        rec.update(imagined_return=tr.get("imagined_return"), losses_all_finite=tr.get("losses_all_finite"))
    if lengths is not None:
        vals = [v for v in lengths if v is not None]
        rec.update(stop_rule_would_fire_at=stop_rule_at(lengths), iterations_logged=len(lengths),
                   iterations_without_length=sum(v is None for v in lengths),
                   min_mean_episode_length_after_50=min((v for v in lengths[STOP["after"]:] if v is not None),
                                                        default=None),
                   final_mean_episode_length=vals[-1] if vals else None)
    return rec


def optc_policies(sim):
    """Option C's policies trained in one of its simulators or in A5_s0, from its own result files."""
    p1 = _load(os.path.join(RES, "c_pilot1.json")) or {}
    out = []
    for name, info in (p1.get("policies") or {}).items():
        if not name.startswith(sim + "__"):
            continue
        rec = {"policy": name, "sim": sim, "role": "Option C", "label": info.get("label"),
               "iterations": info.get("iterations"), "source": "results/c_pilot1.json"}
        meas = (p1.get("measured_weight_1") or {}).get(name)
        if meas:
            rec["walking_score_by_save"] = {k: {x: v[x] for x in ("score", "fall_rate", "ret")} for k, v in meas.items()}
            last = meas[max(meas, key=int)]
            rec.update(score=last["score"], fall_rate=last["fall_rate"], physics_return=last["ret"],
                       failed=bool(last["score"] < FAIL_SCORE), score_at=int(max(meas, key=int)))
        d3 = _load(os.path.join(RES, "c", "d3_eval", f"{name}.json"))
        if d3 is not None:
            s = d3["summary"]
            rec.update(step2_return=s.get("ret"), step2_fall_rate=s.get("fall_rate"),
                       source="results/c/d3_eval (Step 2's protocol; no walking score measured)")
        out.append(rec)
    return out


def checks_row(c):
    if c is None:
        return None
    return {s: {k: v.get(k) for k in CHECK_KEYS} for s, v in c["sets"].items()}


def walk_row(w):
    if w is None:
        return None
    return {k: w[k] for k in ("falls", "tracking_per_step", "tracking_ratio", "u_bar")}


def build(eval_dir, checks_dir, walk_dir, names=None):
    names = list(names or sims.EXISTING)
    rows, missing = {}, []
    for n in names:
        c = _load(os.path.join(checks_dir, f"{n}.json"))
        w = _load(os.path.join(walk_dir, f"{n}.json"))
        if c is None:
            missing.append(f"checks {n}")
        if w is None:
            missing.append(f"walk {n}")
        row = {"how": sims.HOW.get(n), "kind": sims.kind_of(n),
               "scales": (c or {}).get("simulator", {}).get("scales", "own" if n in sims.OPTC else "authors"),
               "checks": checks_row(c), "walk": walk_row(w), "policies": [], "verdict": None}
        if n in S2_SIMS:
            for k in (1, 2):
                row["policies"].append(step2_policy(f"{n}__p{k}", n, "Step 2 registered", eval_dir))
        if n in S2_PILOT:
            row["policies"].append(step2_policy(S2_PILOT[n], n, "Step 2 pilot", eval_dir))
        if n in sims.OPTC or n in sims.A5:
            row["policies"] += optc_policies(n)
        for p in row["policies"]:
            if p["role"].startswith("Step 2") and not p["measured"]:
                missing.append(f"eval {p['policy']}")
        if n in S2_SIMS:
            reg = [p for p in row["policies"] if p["role"] == "Step 2 registered"]
            if all(p.get("score") is not None for p in reg):
                nf = sum(p["failed"] for p in reg)
                row["verdict"] = "failed" if nf >= 2 else "did not fail"
                row["mean_score"] = float(np.mean([p["score"] for p in reg]))
                row["policies_failed"] = nf
        rows[n] = row
    authors = _load(os.path.join(eval_dir, "authors.json"))
    return rows, authors, missing


def text(rows, authors):
    L = ["Option D record (Stage 1): what exists, measured on this Mac. Descriptive. World: hand-built calibrated MuJoCo",
         "ANYmal D with a fitted PD motor (not Isaac Lab, not hardware). Errors in the authors' scales (common units),",
         "expert test set; walk test = RWM's released policy in the simulator's imagination (secondary).", ""]
    if authors:
        L.append(f"RWM's released policy: walking score {authors['score']:.4f}, falls {authors['fall_rate']:.3f}, "
                 f"physics return {authors.get('physics_return', float('nan')):.2f}; standing still "
                 f"{authors['standing_reference']:.4f}")
        L.append("")
    L.append(f"{'simulator':20s} {'E1':>6s} {'E8':>6s} {'E100':>6s} {'E368':>6s} {'rJ':>7s} {'lam1':>6s} | "
             f"{'falls':>5s} {'track':>5s} {'u':>5s} | verdict, policies (score / falls)")
    f = lambda v, d=3: "  -   " if v is None else f"{v:6.{d}f}"   # noqa: E731
    for n, r in rows.items():
        c = (r["checks"] or {}).get("expert") or {}
        w = r["walk"] or {}
        pol = "; ".join(f"{p['policy'].split('__')[-1]} {p['score']:.3f}/{p['fall_rate']:.2f}"
                        if p.get("score") is not None else f"{p['policy'].split('__')[-1]} not scored"
                        for p in r["policies"])
        L.append(f"{n:20s} {f(c.get('E1'))} {f(c.get('E8'))} {f(c.get('E100'))} {f(c.get('E368'))} "
                 f"{f(c.get('rJ'), 2):>7s} {f(c.get('lambda1'))} | {f(w.get('falls'), 2)[-5:]} "
                 f"{f(w.get('tracking_ratio'), 2)[-5:]} {f(w.get('u_bar'), 2)[-5:]} | {r['verdict'] or '-'}; {pol}")
    L += ["", "score = walking score (0.15 or more: walks); a simulator failed when at least two of its policies failed",
          "(given for REL and its copies only). track = walk-test tracking as a share of MuJoCo's; u = disagreement.",
          "Option C's policies (sim_*, A5_s0) were trained by Option C's recipes; A5_s0's have no walking score."]
    return "\n".join(L) + "\n"


def write_once(path, content):
    if os.path.exists(path):
        print(f"[record] {os.path.relpath(path, AIR)} exists, kept")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        fh.write(content)
    os.replace(path + ".tmp", path)


def table(a):
    rows, authors, missing = build(a.eval_dir, a.checks_dir, a.walk_dir, a.sims.split(",") if a.sims else None)
    if a.strict:
        primary = [m for m in missing if not m.startswith("walk ")]
        if primary or authors is None:
            raise SystemExit(f"[record] missing primary inputs: {primary or ['eval authors']}")
    rec = {"what": "Option D record (Stage 1): every simulator that exists, its checks in common units on the MuJoCo test "
                   "sets, its walk test (secondary), and how each policy trained in it did. Descriptive. World: "
                   "hand-built calibrated MuJoCo ANYmal D with a fitted PD motor; not Isaac Lab, not hardware",
           "authors": None if authors is None else {k: authors.get(k) for k in ("score", "score_ci95", "fall_rate",
                                                                               "physics_return", "standing_reference")},
           "fail_score": FAIL_SCORE, "stop_rule": STOP, "missing": missing, "simulators": rows,
           "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_once(a.out + ".json", json.dumps(rec, indent=1, default=float))
    write_once(a.out + ".txt", text(rows, authors))
    print(text(rows, authors)[-3000:])
    if missing:
        print(f"[record] missing: {missing}")


def blocked(gate, why, a):
    if a.rehearsal:
        print(f"[record] rehearsal: {gate} would fail here ({why}); not stopping")
        return
    stamp = time.strftime("%Y-%m-%d %H:%M")
    with open(os.path.join(a.optd, "DECISIONS.md"), "a") as fh:
        fh.write(f"\n## Record job: BLOCKED at {gate} ({stamp})\n{why}\nOptions: check the evaluator against "
                 f"scripts/c/evaluate.py, or the actors' files; the plan says to stop here. Recommendation: the "
                 f"author reads results/d/eval/ before any re-run.\n")
    with open(os.path.join(a.optd, "LOG.md"), "a") as fh:
        fh.write(f"\n## Record job status: BLOCKED ({stamp})\nSanity {gate} failed: {why}\n")
    print(f"[record] {gate} FAILED: {why}")
    sys.exit(3)


def gate1(a):
    mine = _load(os.path.join(a.eval_dir, "authors.json"))
    ref = _load(os.path.join(RES, "c_authors.json"))
    if mine is None or ref is None:
        raise SystemExit("[record] gate1: results missing")
    d = abs(mine["score"] - ref["score"])
    if d > GATE1_TOL:
        blocked("gate 1", f"RWM's released policy scores {mine['score']:.8f}; results/c_authors.json says "
                          f"{ref['score']:.8f} (difference {d:.2e}, limit {GATE1_TOL:g}).", a)
        return
    print(f"[record] gate 1 passed: {mine['score']:.8f} against {ref['score']:.8f} (difference {d:.1e})")


def gate2(a):
    sc = []
    for p in ("REL__p1", "REL__p2"):
        r = _load(os.path.join(a.eval_dir, f"{p}.json"))
        if r is None:
            raise SystemExit(f"[record] gate2: {p} not measured")
        sc.append(r["score"])
    m = float(np.mean(sc))
    if m < GATE2_MIN:
        blocked("gate 2", f"REL's two registered policies score {sc[0]:.4f} and {sc[1]:.4f}, mean {m:.4f}; the gate "
                          f"needs at least {GATE2_MIN}.", a)
        return
    print(f"[record] gate 2 passed: REL__p1 {sc[0]:.4f}, REL__p2 {sc[1]:.4f}, mean {m:.4f} (needs {GATE2_MIN})")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("table", "gate1", "gate2"))
    ap.add_argument("--eval-dir", default=os.path.join(RES, "d", "eval"))
    ap.add_argument("--checks-dir", default=os.path.join(RES, "d", "checks"))
    ap.add_argument("--walk-dir", default=os.path.join(RES, "d", "walk"))
    ap.add_argument("--out", default=os.path.join(RES, "d_record"), help="prefix: <out>.json and <out>.txt")
    ap.add_argument("--sims", default=None, help="rehearsal only: a comma list")
    ap.add_argument("--strict", action="store_true", help="fail if a primary input is missing")
    ap.add_argument("--optd", default=os.path.join(AIR, "optd"), help="where a failed gate writes (tests only)")
    ap.add_argument("--rehearsal", action="store_true", help="gates report but never stop (toy sizes)")
    a = ap.parse_args(argv)
    {"table": table, "gate1": gate1, "gate2": gate2}[a.what](a)


if __name__ == "__main__":
    main()

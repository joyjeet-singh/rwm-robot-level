"""Option C (C2): the pilot's gates and retry rule, applied mechanically (OPTIONC_PLAN.md, "Gates and the pilot").

    ./py scripts/c/pilot_rule.py decide   # after the first two policies: prints the retry weights, or "none"
    ./py scripts/c/pilot_rule.py report   # after any retry: results/c_pilot.json and .txt; exit 3 if BLOCKED

  * Gate B: at 2,000 iterations the 6M policy falls in under 20% of episodes and scores at least half the authors' policy.
  * Retry: if Gate B fails, two more 6M policies: uncertainty weights 2.0 and 4.0 if it fell in 20% of episodes or more,
    else 0.5 and 0.25. The kept weight is the highest-scoring one among those passing Gate B. If none passes: BLOCKED.
  * Iteration rule: the smallest saved count at which the kept 6M policy scores at least 90% of its own 2,000-iteration
    score.
  * Gate C (not a stop): at 2,000 iterations the 10k policy (weight 1.0, the pilot trains only one) scores below the kept
    6M policy, with their 95% intervals over episodes apart. If not, the main pool adds a 2k rung.

Every value comes from the result files of scripts/c/evaluate.py. The world is the hand-built calibrated MuJoCo ANYmal D
with a fitted PD motor, not Isaac Lab and not hardware. Results are written once; a re-run must reach the same answer.
"""
import argparse
import json
import os
import sys
import time

AIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVES = (500, 1000, 1500, 2000)
SIM_6M, SIM_10K = "sim_6M_N8_s0", "sim_10k_N8_s0"
FALL_LIMIT = 0.20
SCORE_SHARE = 0.5
ITER_SHARE = 0.9
UP, DOWN = (2.0, 4.0), (0.5, 0.25)


def policy_name(sim, seed, weight=1.0):
    return f"{sim}__p{seed}" if weight == 1.0 else f"{sim}__w{weight:g}__p{seed}"


class Paths:
    def __init__(self, root):
        self.root = root
        self.eval_dir = os.path.join(root, "results", "c", "eval")
        self.authors = os.path.join(root, "results", "c_authors.json")
        self.decision = os.path.join(root, "results", "c", "pilot_decision.json")
        self.gate_a = os.path.join(root, "results", "c", "gate_a.json")
        self.out_json = os.path.join(root, "results", "c_pilot.json")
        self.out_txt = os.path.join(root, "results", "c_pilot.txt")
        self.decisions = os.path.join(root, "optc", "DECISIONS.md")
        self.log = os.path.join(root, "optc", "LOG.md")


def load_eval(p, policy, n):
    path = os.path.join(p.eval_dir, f"{policy}__i{n}.json")
    if not os.path.exists(path):
        raise SystemExit(f"missing result: {path}")
    return json.load(open(path))


def gate_b(r, authors_score):
    fall, score = r["summary"]["fall_rate"], r["score"]
    return {"fall_rate": fall, "score": score, "authors_score": authors_score, "fall_below_limit": fall < FALL_LIMIT,
            "score_at_least_half": score >= SCORE_SHARE * authors_score,
            "passed": fall < FALL_LIMIT and score >= SCORE_SHARE * authors_score}


def write_once(path, obj):
    if os.path.exists(path):
        if json.load(open(path)) != obj:
            raise SystemExit(f"{path} exists and differs from this run's answer; result files are never overwritten")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(path + ".tmp", path)


def decide(p):
    authors = json.load(open(p.authors))["score"]
    base = gate_b(load_eval(p, policy_name(SIM_6M, 0), 2000), authors)
    if base["passed"]:
        weights, why = [], "Gate B passed at weight 1.0"
    elif not base["fall_below_limit"]:
        weights, why = list(UP), f"fell in {FALL_LIMIT:.0%} of episodes or more"
    else:
        weights, why = list(DOWN), "fell less, but scored below half the authors' policy"
    write_once(p.decision, {"gate_b_at_weight_1": base, "retry_weights": weights, "why": why})
    return weights


def row(r):
    s = r["summary"]
    return {"score": r["score"], "score_ci95": r["score_ci95"], "fall_rate": s["fall_rate"], "ret": s["ret"],
            "speed_err": s["speed_err"], "turn_err": s["turn_err"], "diverged": s["diverged"],
            "standing_reference": r["standing_reference"]}


def report(p):
    authors_rec = json.load(open(p.authors))
    authors = authors_rec["score"]
    dec = json.load(open(p.decision))
    retry = dec["retry_weights"]
    base_6m = policy_name(SIM_6M, 0)
    runs = {1.0: base_6m}
    runs.update({w: policy_name(SIM_6M, 0, w) for w in retry})
    table = {name: {str(n): row(load_eval(p, name, n)) for n in SAVES}
             for name in [*runs.values(), policy_name(SIM_10K, 0)]}
    passes = {w: gate_b(load_eval(p, name, 2000), authors) for w, name in runs.items()}

    if passes[1.0]["passed"]:
        kept = 1.0
    else:
        good = [w for w in retry if passes[w]["passed"]]
        kept = max(good, key=lambda w: passes[w]["score"]) if good else None

    res = {"what": "Option C pilot: Gate A, Gate B, retry rule, iteration rule and Gate C (hand-built calibrated MuJoCo "
                   "ANYmal D, fitted PD motor; not Isaac Lab, not hardware)",
           "gate_a": json.load(open(p.gate_a)) if os.path.exists(p.gate_a) else None,
           "authors": {"score": authors, "score_ci95": authors_rec["score_ci95"],
                       "fall_rate": authors_rec["summary"]["fall_rate"]},
           "standing_reference": authors_rec["standing_reference"], "retry": dec,
           "gate_b": {str(w): v for w, v in passes.items()}, "policies": table}

    if kept is None:
        res.update(status="BLOCKED", kept_weight=None, iterations=None, gate_c=None,
                   blocked_because="no uncertainty weight's 6M policy passes Gate B")
    else:
        name = runs[kept]
        ref = table[name]["2000"]["score"]
        chosen = next(n for n in SAVES if table[name][str(n)]["score"] >= ITER_SHARE * ref)
        kept_row, row_10k = table[name]["2000"], table[policy_name(SIM_10K, 0)]["2000"]
        apart = row_10k["score_ci95"][1] < kept_row["score_ci95"][0]
        gc = {"score_10k": row_10k["score"], "score_6M": kept_row["score"], "ci95_10k": row_10k["score_ci95"],
              "ci95_6M": kept_row["score_ci95"], "below": row_10k["score"] < kept_row["score"],
              "intervals_apart": apart, "passed": row_10k["score"] < kept_row["score"] and apart}
        gc["action"] = ("none" if gc["passed"] else
                        "add a 2k rung (rollout-trained seeds 0 and 1, teacher-forced seed 0) and record it before registering")
        res.update(status="PASSED", kept_weight=kept, kept_policy=name, iterations=chosen,
                   iteration_rule={"reference_score_at_2000": ref, "threshold": ITER_SHARE * ref}, gate_c=gc)

    write_once(p.out_json, res)
    text = render(res)
    if os.path.exists(p.out_txt):
        if open(p.out_txt).read() != text:
            raise SystemExit(f"{p.out_txt} exists and differs from this run's answer")
    else:
        with open(p.out_txt, "w") as fh:
            fh.write(text)
    if res["status"] == "BLOCKED":
        mark_blocked(p, res)
    print(text)
    return res


def render(res):
    a = res["authors"]
    lines = ["Option C pilot (calibrated hand-built MuJoCo ANYmal D, fitted PD motor; not Isaac Lab, not hardware)"]
    ga = res["gate_a"]
    if ga is not None:
        lines.append(f"Gate A: {'passed' if ga['passed'] else 'FAILED'} (expert set, E8 and E24 against holding the last state)")
    lines.append(f"Authors' policy: score {a['score']:.3f}, 95% interval {a['score_ci95'][0]:.3f} to {a['score_ci95'][1]:.3f}, "
                 f"falls {a['fall_rate']:.0%}; standing still scores {res['standing_reference']:.3f}")
    lines.append(f"Retry: {res['retry']['why']}; weights {res['retry']['retry_weights'] or 'none'}")
    lines.append(f"{'policy':34s} {'iter':>5s} {'score':>6s} {'95% interval':>17s} {'falls':>6s} {'return':>8s}")
    for name, by in res["policies"].items():
        for n, r in by.items():
            ci = f"{r['score_ci95'][0]:.3f} to {r['score_ci95'][1]:.3f}"
            lines.append(f"{name:34s} {n:>5s} {r['score']:6.3f} {ci:>17s} {r['fall_rate']:6.0%} {r['ret']:8.2f}")
    for w, g in res["gate_b"].items():
        lines.append(f"Gate B at weight {w}: {'passed' if g['passed'] else 'failed'} (falls {g['fall_rate']:.0%}, "
                     f"score {g['score']:.3f}, half the authors' {SCORE_SHARE * g['authors_score']:.3f})")
    lines.append(f"Status: {res['status']}")
    if res["status"] == "BLOCKED":
        lines.append(res["blocked_because"])
    else:
        c = res["gate_c"]
        lines.append(f"Kept weight {res['kept_weight']}, policy {res['kept_policy']}; main study uses {res['iterations']} iterations")
        lines.append(f"Gate C: {'passed' if c['passed'] else 'NOT passed'} (10k {c['score_10k']:.3f}, 6M {c['score_6M']:.3f}, "
                     f"intervals {'apart' if c['intervals_apart'] else 'overlap'}); action: {c['action']}")
    assert len(lines) <= 40, len(lines)
    return "\n".join(lines) + "\n"


def mark_blocked(p, res):
    mark = "## Pilot job: BLOCKED"
    old = open(p.decisions).read() if os.path.exists(p.decisions) else ""
    if mark in old:
        return
    stamp = time.strftime("%Y-%m-%d %H:%M")
    gates = "; ".join(f"weight {w}: falls {g['fall_rate']:.0%}, score {g['score']:.3f}" for w, g in res["gate_b"].items())
    with open(p.decisions, "a") as fh:
        fh.write(f"\n{mark} ({stamp})\n\n"
                 f"**What is blocked:** no uncertainty weight's 6M policy passes Gate B ({gates}; half the authors' score is "
                 f"{SCORE_SHARE * res['authors']['score']:.3f}). The plan says to stop.\n\n"
                 "**Options:** (a) stop Option C and report the pilot as it stands; (b) change the design (other weights, more "
                 "iterations, a different start-state rule), which needs a new decision and is a change to the fixed design.\n\n"
                 "**Recommendation:** (a), reporting the pilot in `results/c_pilot.txt`, unless the author chooses otherwise.\n")
    with open(p.log, "a") as fh:
        fh.write(f"\n## Pilot job status: BLOCKED ({stamp})\nNo uncertainty weight's 6M policy passes Gate B; see `optc/DECISIONS.md` "
                 "and `results/c_pilot.txt`.\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("decide", "report"))
    ap.add_argument("--root", default=AIR, help="fixture folder (tests only)")
    a = ap.parse_args(argv)
    p = Paths(a.root)
    if a.stage == "decide":
        weights = decide(p)
        print(" ".join(f"{w:g}" for w in weights) if weights else "none")
        return 0
    return 3 if report(p)["status"] == "BLOCKED" else 0


if __name__ == "__main__":
    sys.exit(main())

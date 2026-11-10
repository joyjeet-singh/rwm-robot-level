"""Option C (C2R, step 2): the record of the first pilot (OPTIONC_PLAN.md, Amendment 1).

    ./py scripts/c/pilot1_record.py               # session: results/c_pilot1.json and .txt, measurements pending
    ./py scripts/c/pilot1_record.py --measured    # pilot job: adds the first pilot's surviving saves, measured

  * Gate B at uncertainty weight 1.0, copied from results/c/pilot_decision.json (never rewritten here), with the
    measured saves of the two weight-1.0 policies (results/c/eval/);
  * per policy, from the lite metrics (runs/.../metrics.jsonl, one row per iteration): mean imagined episode length
    (Train/mean_episode_length, the last 100 finished episodes), mean ensemble disagreement per step, and every logged
    reward term, after iterations 1 ("0"), 100, 500, 1,000 and the last. The lite log's term values are episode sums
    divided by the 256-step limit's duration (5.12 s; results/c/c2r_facts.json), so they shrink with episode length; the
    approximate total reward per 0.02 s step is (sum of terms) x 5.12 / length, also given;
  * the same for Step 2's REL__p0 (released model) and D3's two policies in A5_s0, for comparison;
  * the weight-2.0 run crashed at about iteration 1,500, before the lite code writes its metrics, so it has none;
  * with --measured (the pilot job): the walking score of the surviving saves (weight 2.0 at 500, 1,000 and 1,500;
    weight 4.0 at 500 to 2,000), measured with the full protocol by scripts/c/evaluate.py.

The file is written once by the session; the job adds only the measured saves (OPTIONC_PLAN.md C2R step 2), after
checking that every other value is unchanged. The world is the hand-built calibrated MuJoCo ANYmal D with a fitted PD
motor, not Isaac Lab and not hardware.
"""
import argparse
import json
import os
import sys

AIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_JSON = os.path.join(AIR, "results", "c_pilot1.json")
OUT_TXT = os.path.join(AIR, "results", "c_pilot1.txt")
LIMIT_S = 256 * 0.02
RUNS = {
    "sim_6M_N8_s0__p0": ("first pilot, 6M, weight 1.0", "runs/c/lite/runs/lite/s2_sim_6M_N8_s0__p0"),
    "sim_10k_N8_s0__p0": ("first pilot, 10k, weight 1.0", "runs/c/lite/runs/lite/s2_sim_10k_N8_s0__p0"),
    "sim_6M_N8_s0__w2__p0": ("first pilot retry, 6M, weight 2.0 (crashed)", None),
    "sim_6M_N8_s0__w4__p0": ("first pilot retry, 6M, weight 4.0", "runs/c/lite/runs/lite/s2_sim_6M_N8_s0__w4__p0"),
    "REL__p0": ("Step 2 pilot, released model", "runs/lite/s2_REL__p0"),
    "A5_s0__p0": ("D3, A5_s0 seed 0", "runs/c/d3_lite/runs/lite/s2_A5_s0__p0"),
    "A5_s0__p1": ("D3, A5_s0 seed 1", "runs/c/d3_lite/runs/lite/s2_A5_s0__p1"),
}
AT = (1, 100, 500, 1000)
SURVIVING = {"sim_6M_N8_s0__w2__p0": (500, 1000, 1500), "sim_6M_N8_s0__w4__p0": (500, 1000, 1500, 2000)}
TERMS_SHORT = {"track_lin_vel_xy_exp": "lin", "track_ang_vel_z_exp": "ang", "lin_vel_z_l2": "vz",
               "ang_vel_xy_l2": "wxy", "flat_orientation_l2": "tilt", "uncertainty": "unc"}


def metrics(path):
    rows = [json.loads(x) for x in open(os.path.join(AIR, path, "metrics.jsonl"))]
    return [r for r in rows if "Imagination/epistemic_uncertainty" in r]


def at(rows, n):
    r = rows[n - 1]
    terms = {k.split("/", 1)[1]: v for k, v in r.items() if k.startswith("Imagination/") and k not in
             ("Imagination/epistemic_uncertainty", "Imagination/num_valid_imagination_envs")}
    L = r.get("Train/mean_episode_length")
    tot = sum(terms.values())
    return {"iteration": n, "mean_length": L, "disagreement": r["Imagination/epistemic_uncertainty"],
            "noise_std": r.get("Policy/mean_noise_std"), "mean_return": r.get("Train/mean_reward"),
            "terms_logged": terms, "sum_terms_logged": tot,
            "approx_total_per_step": (tot * LIMIT_S / L) if L else None}


def eval_row(name, n):
    p = os.path.join(AIR, "results", "c", "eval", f"{name}__i{n}.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    return {"score": r["score"], "score_ci95": r["score_ci95"], "fall_rate": r["summary"]["fall_rate"],
            "ret": r["summary"]["ret"]}


def build():
    dec = json.load(open(os.path.join(AIR, "results", "c", "pilot_decision.json")))
    authors = json.load(open(os.path.join(AIR, "results", "c_authors.json")))
    pol = {}
    for name, (label, path) in RUNS.items():
        if path is None:
            pol[name] = {"label": label, "iterations": None,
                         "lite_metrics": "none: the run crashed (exploration noise below zero inside an update) "
                                         "before the lite code writes its metrics"}
            continue
        rows = metrics(path)
        pts = {str(n if n > 1 else 0): at(rows, n) for n in AT if n <= len(rows)}
        pts["end"] = at(rows, len(rows))
        pol[name] = {"label": label, "iterations": len(rows), "at": pts}
    measured_w1 = {name: {str(n): eval_row(name, n) for n in (500, 1000, 1500, 2000)}
                   for name in ("sim_6M_N8_s0__p0", "sim_10k_N8_s0__p0")}
    return {"what": "Option C first pilot (failed; Amendment 1). Lite metrics in imagination; walking scores in the "
                    "calibrated hand-built MuJoCo ANYmal D, fitted PD motor (not Isaac Lab, not hardware)",
            "gate_b_at_weight_1": dec["gate_b_at_weight_1"], "retry_rule_then": {k: dec[k] for k in ("retry_weights", "why")},
            "authors": {"score": authors["score"], "score_ci95": authors["score_ci95"],
                        "fall_rate": authors["summary"]["fall_rate"]},
            "measured_weight_1": measured_w1, "policies": pol,
            "units": "terms_logged: episode sum / 5.12 s (the lite log); approx_total_per_step: reward per 0.02 s step",
            "surviving_saves_measured": None}


def render(res):
    a = res["authors"]
    g = res["gate_b_at_weight_1"]
    L = ["Option C first pilot (failed). Imagination: lite metrics; walking: calibrated hand-built MuJoCo ANYmal D, "
         "fitted PD motor; not Isaac Lab, not hardware",
         f"Authors' policy: score {a['score']:.3f}, falls {a['fall_rate']:.0%}. Gate B at weight 1.0: FAILED "
         f"(6M at 2,000: falls {g['fall_rate']:.0%}, score {g['score']:.4f})",
         "Measured (weight 1.0): policy, save: score, falls"]
    for name, by in res["measured_weight_1"].items():
        L.append("  " + name + ": " + "; ".join(f"{n} {r['score']:.3f}, {r['fall_rate']:.0%}" for n, r in by.items()
                                                 if r is not None))
    L.append("Lite metrics: iter: length, disagreement, total reward/step (approx.); logged terms lin ang vz tilt unc")
    for name, p in res["policies"].items():
        if "at" not in p:
            L.append(f"  {name}: no lite metrics (crashed)")
            continue
        L.append(f"  {name} ({p['iterations']} iterations)")
        for k, r in p["at"].items():
            if k == "end" and p["iterations"] in AT:
                continue                                  # the same row as the count before it
            t = r["terms_logged"]
            ln = f"{r['mean_length']:6.1f}" if r["mean_length"] is not None else "     -"
            ps = f"{r['approx_total_per_step']:+.4f}" if r["approx_total_per_step"] is not None else "      -"
            L.append(f"    {k:>4s}: {ln} {r['disagreement']:.2f} {ps} | " +
                     " ".join(f"{t.get(x, 0.0):+.3f}" for x in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp",
                                                               "lin_vel_z_l2", "flat_orientation_l2", "uncertainty")))
    m = res["surviving_saves_measured"]
    if m is None:
        L.append("Surviving saves (weight 2.0 at 500-1,500; 4.0 at 500-2,000): measured by the pilot job")
    else:
        for name, by in m.items():
            L.append(f"  {name}: " + "; ".join(f"{n} {r['score']:.3f}, {r['fall_rate']:.0%}" for n, r in by.items()))
    return L


def write(res, overwrite_measured=False):
    text = "\n".join(render(res)) + "\n"
    assert text.count("\n") <= 40, text.count("\n")
    for path, body in ((OUT_JSON, json.dumps(res, indent=1)), (OUT_TXT, text)):
        if os.path.exists(path) and not overwrite_measured:
            if open(path).read() != body:
                raise SystemExit(f"{path} exists and differs; result files are never overwritten")
            continue
        with open(path + ".tmp", "w") as fh:
            fh.write(body)
        os.replace(path + ".tmp", path)
    print(text)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured", action="store_true", help="the pilot job: add the surviving saves' measurements")
    a = ap.parse_args(argv)
    res = build()
    if not a.measured:
        write(res)
        return 0
    if not os.path.exists(OUT_JSON):
        raise SystemExit(f"{OUT_JSON} is missing; the session writes it first")
    old = json.load(open(OUT_JSON))
    if old.get("surviving_saves_measured") is not None:
        print("surviving saves already added")
        return 0
    if json.loads(json.dumps(res)) != old:
        raise SystemExit(f"{OUT_JSON}: the session's values changed; not adding the measurements")
    meas = {name: {str(n): eval_row(name, n) for n in ns} for name, ns in SURVIVING.items()}
    missing = [f"{k}@{n}" for k, by in meas.items() for n, r in by.items() if r is None]
    if missing:
        raise SystemExit(f"not measured yet: {missing}")
    res["surviving_saves_measured"] = meas
    write(res, overwrite_measured=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

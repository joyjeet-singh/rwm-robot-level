"""The Step 2 pilot report: results/s2_pilot.json and results/s2_pilot.txt (new files only).

    ./py scripts/s2/pilot_report.py

Reads results/s2/train/<policy>.json and results/s2/eval/<policy>.json of the four pilot policies (one seed each:
the released model, seed 0, one teacher-forced model, the 50%-pruned copy of seed 0) and, for reference, the
authors' policy in the same calibrated model (results/s2_fidelity.json). It prints the numbers Gate 2 reads; M4
applies P1-P3. Validation, not results (STEP2_PLAN.md C4).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
AIR = os.path.dirname(os.path.dirname(HERE))
PILOT = [("REL__p0", "released model (5 networks)"), ("A_s0__p0", "seed 0 (A_s0)"),
         ("B_s0__p0", "teacher-forced (B_s0)"), ("A_s0__u50__p0", "50%-pruned seed 0 (A_s0__u50)")]
OUT_JSON = os.path.join(AIR, "results", "s2_pilot.json")
OUT_TXT = os.path.join(AIR, "results", "s2_pilot.txt")


def main():
    for p in (OUT_JSON, OUT_TXT):
        if os.path.exists(p):
            raise SystemExit(f"{p} exists; result files are never overwritten")
    rows = {}
    for pol, label in PILOT:
        tr = json.load(open(os.path.join(AIR, "results", "s2", "train", pol + ".json")))
        ev = json.load(open(os.path.join(AIR, "results", "s2", "eval", pol + ".json")))
        s = ev["summary"]
        rows[pol] = {"label": label, "sim": tr["sim"], "seed": tr["seed"], "holdout": tr["holdout"],
                     "imagined_return": tr["imagined_return"], "imagined_step_reward": tr["imagined_step_reward"],
                     "train_hours": tr["wall_seconds"] / 3600, "seconds_per_iter_median": tr["seconds_per_iter_median"],
                     "actor_sha256": tr["actor_sha256"],
                     **{k: s[k] for k in ("n", "fall_rate", "fall_rate_ci95", "ret", "ret_ci95", "speed_err",
                                          "turn_err", "two_feet", "duty", "per_schedule")}}
    fid = json.load(open(os.path.join(AIR, "results", "s2_fidelity.json")))
    ref = fid["gait_authors_policy"]["calibrated"]["summary"]
    rec = {"what": "Step 2 pilot: four policies, one seed each, trained in learned simulators (lite code, 500 "
                   "iterations, 4,096 robots) and measured in the calibrated hand-built MuJoCo ANYmal D "
                   "(10 schedules x 8 starts). Validation for Gate 2, not results.",
           "gate2": {"P1": "released-model policy falls in fewer than 25% of episodes",
                     "P2": "at least one of the three rebuilt-simulator policies falls in fewer than 50%",
                     "P3": "the four physics returns are not all inside each other's 95% intervals"},
           "interval": "95% percentile bootstrap of the mean, starts resampled within schedules (evaluate.py)",
           "policies": rows,
           "authors_policy_reference": {k: ref[k] for k in ("fall_rate", "ret", "speed_err", "turn_err", "two_feet")}}
    tmp = OUT_JSON + ".tmp"
    json.dump(rec, open(tmp, "w"), indent=1)
    os.replace(tmp, OUT_JSON)

    L = ["Step 2 pilot (validation for Gate 2, not results; hand-built MuJoCo ANYmal D, calibrated)",
         "one seed each; 500 iterations, 4,096 robots; 80 episodes (10 schedules x 8 starts)", "",
         f"{'policy':15s} {'falls':>6s} {'falls 95%':>15s} {'return':>8s} {'return 95%':>17s} "
         f"{'speed':>6s} {'turn':>6s} {'2 feet':>6s} {'imag.':>8s} {'hours':>6s}"]
    for pol, r in rows.items():
        im = f"{r['imagined_return']:8.2f}" if r["imagined_return"] is not None else f"{'nan':>8s}"
        L.append(f"{pol:15s} {r['fall_rate']:6.3f} [{r['fall_rate_ci95'][0]:5.3f}, {r['fall_rate_ci95'][1]:5.3f}] "
                 f"{r['ret']:8.2f} [{r['ret_ci95'][0]:6.2f}, {r['ret_ci95'][1]:6.2f}] "
                 f"{r['speed_err']:6.3f} {r['turn_err']:6.3f} {r['two_feet']:6.3f} {im} {r['train_hours']:6.2f}")
    L.append(f"{'authors policy':15s} {ref['fall_rate']:6.3f} {'':15s} {ref['ret']:8.2f} {'':17s} "
             f"{ref['speed_err']:6.3f} {ref['turn_err']:6.3f} {ref['two_feet']:6.3f}")
    L += ["", "falls = share of episodes ending in a fall; return = physics return (lite reward terms in MuJoCo);",
          "speed in m/s, turn in rad/s; imag. = imagined return, mean of the last 25 training iterations",
          "Gate 2 is applied by M4: P1 released < 25% falls; P2 one rebuilt policy < 50% falls;",
          "P3 the four returns not all inside each other's 95% intervals"]
    open(OUT_TXT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

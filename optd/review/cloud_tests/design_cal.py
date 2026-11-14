"""Design calculation, recalibrated: walkers' seed-to-seed spread as measured in Step 2 (SD about 0.12,
against 0.09 in design.py), for the plan's rule (2+1) against its neighbours, with six, seven and nine runs.
Usage: python3 design_cal.py <n_studies> <family share> <walkers' SD>"""
import sys, json
import numpy as np
import design as D

n_st, fam, sdw = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
conds = [("2nf", "first", 6), ("2", "first", 6), ("2+1", "maj", 6), ("3", "maj", 6), ("5", "maj", 6),
         ("2+1", "maj", 7), ("2+1", "maj", 9)]
out = {}
for (ra, rb), ename in (((0.7, -0.4), "large effect"), ((0.75, 0.2), "moderate effect")):
    for w, wname in ((0.4, "sharp boundary"), (1.5, "gradual boundary")):
        print(f"\n== {ename}; {wname}; families share {fam:.0%} of the checks' noise; walkers' SD {sdw}")
        print(f"{'rule':5s} {'runs':>4s} {'policies':>8s} {'hours':>6s} {'track':>6s} {'label err':>9s} {'P(A>0)':>7s} {'P(A>B)':>7s} {'splits':>7s}")
        for rule, decide, nr in conds:
            rng = np.random.default_rng(2027)
            R = [D.study(rng, w, rule, decide, n_runs=nr, sd_walk=sdw, rho_a=ra, rho_b=rb, fam=fam) for _ in range(n_st)]
            g = lambda key: float(np.nanmean([r[key] for r in R]))
            row = dict(policies=g("policies"), hours=g("policies") * D.HOURS_PER_POLICY / 2, track=g("track"),
                       label_err=g("label_err"), t0=g("t0"), t1=g("t1"), o0=g("o0"), o1=g("o1"), disagree=g("disagree"))
            out[f"{ename}|{wname}|{rule}|{nr}"] = row
            print(f"{rule:5s} {nr:4d} {row['policies']:8.1f} {row['hours']:6.1f} {row['track']:6.3f} {row['label_err']:9.3f} "
                  f"{row['t0']:7.2f} {row['t1']:7.2f} {row['disagree']:7.2f}", flush=True)
json.dump(out, open(f"design_cal_fam{int(fam * 100)}_sd{int(round(sdw * 100))}.json", "w"), indent=1)

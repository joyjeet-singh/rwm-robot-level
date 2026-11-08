"""Numbers for Step 2 (called by paper_numbers.py after im_numbers). Every value is read from a result file.

Step 2 trains walking policies inside learned simulators and measures them in a hand-built MuJoCo ANYmal D model
(not the data's simulator, not hardware). Sources:
  results/s2_stats.json        registered analysis (scripts/s2_stats.py, prereg/PREREG_S2.md): R0-R2, secondary,
                               exploratory (two-level intervals) and descriptive (per simulator) blocks
  results/s2_fidelity.json     the model's calibration, held-out replay and Gate 1 (validation, not results)
  results/s2_pilot.json        Gate 2: the four pilot policies (validation, not results)
  results/s2_main.json         the main job: the 18 registered policies
  results/s2/pd_fit.json       the PD motor model fitted to recorded torques
  results/s2/m1_tests.json     the model's mass check
  results/s2/eval/<p>.json     each policy's MuJoCo measures (duty factor, used descriptively)
  results/s2/train/<p>.json    each policy's training record (hours)
  step2/model/calibration.json the frozen calibration grid
  git                          the registration commit of prereg/PREREG_S2.md
Secondary results must be printed as secondary, exploratory ones as exploratory, descriptive ones as descriptive.
"""
import datetime as dt
import json
import os
import subprocess

import common as C

STATS = os.path.join(C.RESULTS, "s2_stats.json")
FID = os.path.join(C.RESULTS, "s2_fidelity.json")
PILOT = os.path.join(C.RESULTS, "s2_pilot.json")
MAIN = os.path.join(C.RESULTS, "s2_main.json")
PD = os.path.join(C.RESULTS, "s2", "pd_fit.json")
M1T = os.path.join(C.RESULTS, "s2", "m1_tests.json")
EVAL = os.path.join(C.RESULTS, "s2", "eval")
TRAIN = os.path.join(C.RESULTS, "s2", "train")
CALIB = os.path.join(C.AIR, "step2", "model", "calibration.json")
# simulator id -> key suffix (LaTeX-safe) used in the per-simulator keys
SIMKEY = {"REL": "rel", "REL__fp16": "fp16", "REL__int8w": "int8w", "REL__u10": "u10", "REL__u20": "u20",
          "REL__u30": "u30", "REL__u50": "u50", "REL__s192": "s192", "REL__s128": "s128"}
PILOTKEY = {"REL__p0": "rel", "A_s0__p0": "a", "B_s0__p0": "b", "A_s0__u50__p0": "au50"}


def _rel(path):
    return os.path.relpath(path, os.path.dirname(C.AIR))


def _ld(path):
    with open(path) as fh:
        return json.load(fh)


def add(put, sig, sgn, pct, cib, dp, load, rel):
    S, s = _ld(STATS), _rel(STATS)
    F, f = _ld(FID), _rel(FID)
    P, p = _ld(PILOT), _rel(PILOT)
    M, m = _ld(MAIN), _rel(MAIN)
    PDF, pdr = _ld(PD), _rel(PD)
    T1, t1 = _ld(M1T), _rel(M1T)
    CAL, cal = _ld(CALIB), _rel(CALIB)
    r2 = lambda v: dp(v, 2)                                                              # noqa: E731
    s2 = lambda v: "0.00" if round(v, 2) == 0 else dp(v, 2, signed=True)                 # noqa: E731
    low = lambda t: t.lower()                                                            # noqa: E731
    pc0 = lambda x: f"{100 * x:.0f}" + r"\%"                                              # noqa: E731
    pc2 = lambda x: f"{100 * x:.2f}" + r"\%"                                              # noqa: E731

    assert S["n_simulators"] == 9 and not S["missing_policies"] and not S["failed_policies"]
    assert T1["all_pass"] and CAL["frozen"]

    # ---- registration and the job
    out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h %ct", "--", "prereg/PREREG_S2.md"],
                         cwd=C.AIR, capture_output=True, text=True, check=True).stdout.split()
    assert len(out) == 2, out
    put("s2_reg_commit", out[0], "git", "log --diff-filter=A -- prereg/PREREG_S2.md (%h)")
    put("s2_reg_time", dt.datetime.fromtimestamp(int(out[1]), dt.timezone.utc).strftime("%Y-%m-%d %H:%M") + " UTC",
        "git", "log --diff-filter=A -- prereg/PREREG_S2.md (%ct, printed in UTC)")
    pols = M["policies"]
    assert all(q["status"] == "measured" and q["losses_all_finite"] for q in pols)
    put("s2_n_sims", str(S["n_simulators"]), s, "n_simulators")
    put("s2_n_policies", str(len(pols)), m, "policies (count, all status measured)")
    put("s2_job_hours", sig(M["seconds_this_run"] / 3600, 2), m, "seconds_this_run / 3600")
    put("s2_workers", str(M["workers"]), m, "workers")
    th = [_ld(os.path.join(TRAIN, q["policy"] + ".json")) for q in pols]
    assert all(x["iterations"] == 500 and x["num_envs"] == 4096 for x in th)
    hrs = sorted(x["wall_seconds"] / 3600 for x in th)
    put("s2_train_h_min", sig(hrs[0], 2), _rel(TRAIN), "<policy>.json wall_seconds / 3600 (min over 18)")
    put("s2_train_h_max", sig(hrs[-1], 2), _rel(TRAIN), "<policy>.json wall_seconds / 3600 (max over 18)")
    put("s2_iters", str(th[0]["iterations"]), _rel(TRAIN), "<policy>.json iterations (all 18 equal)")
    put("s2_envs", f"{th[0]['num_envs']:,}".replace(",", "{,}"), _rel(TRAIN), "<policy>.json num_envs (all 18 equal)")
    ne = {_ld(os.path.join(EVAL, q["policy"] + ".json"))["summary"]["n"] for q in pols}
    assert len(ne) == 1, ne
    put("s2_n_eps", str(ne.pop()), _rel(EVAL), "<policy>.json summary.n (all 18 equal)")

    # ---- the model (appendix A11; validation)
    put("s2_mass", f"{T1['a']['total_mass_kg']:.2f}", t1, "a.total_mass_kg")
    put("s2_kp", sig(PDF["kp"], 3), pdr, "kp")
    put("s2_kd", sig(PDF["kd"], 3), pdr, "kd")
    put("s2_delay_ms", f"{1000 * PDF['delay_s']:.0f}", pdr, "delay_s * 1000")
    put("s2_delay_steps", str(PDF["delay_physics_steps"]), pdr, "delay_physics_steps")
    put("s2_pd_r2", r2(PDF["r2"]), pdr, "r2")
    put("s2_pd_rmse", sig(PDF["rmse_Nm"], 2), pdr, "rmse_Nm")
    put("s2_clip_sat", f"{PDF['clip']['saturation_effort']:.0f}", pdr, "clip.saturation_effort")
    put("s2_clip_lim", f"{PDF['clip']['effort_limit']:.0f}", pdr, "clip.effort_limit")
    put("s2_clip_vel", f"{PDF['clip']['velocity_limit']:.1f}", pdr, "clip.velocity_limit")
    put("s2_pd_n_eps", str(len(PDF["episodes"])), pdr, "episodes (count)")
    g = CAL["grid"]
    ng = len(g["friction"]) * len(g["timeconst"]) * len(g["armature"]) * len(g["damping"])
    assert ng == len(os.listdir(os.path.join(C.RESULTS, "s2", "calib")))
    put("s2_grid_n", str(ng), cal, "grid (product of the four lists' lengths)")
    ch, un = F["calibration"]["chosen"], CAL["uncalibrated_point"]
    assert ch == CAL["chosen"]
    for k, fmt in (("friction", "{:.1f}"), ("timeconst", "{:.2f}"), ("armature", "{:.2f}"), ("damping", "{:.1f}")):
        put(f"s2_cal_{k}", fmt.format(ch[k]), f, f"calibration.chosen.{k}")
        put(f"s2_uncal_{k}", fmt.format(un[k]), cal, f"uncalibrated_point.{k}")
    put("s2_cal_obj_E8", r2(F["calibration"]["objective"]["E8"]), f, "calibration.objective.E8")
    put("s2_cal_obj_E1", r2(F["calibration"]["objective"]["E1"]), f, "calibration.objective.E1")
    put("s2_cal_worst_E8", r2(CAL["worst"]["E8"]), cal, "worst.E8")
    # held-out replay of recorded joint commands (descriptive)
    for lab, blk in (("cal", "calibrated"), ("uncal", "uncalibrated")):
        R = F["replay_held_out"][blk]
        assert sum(R["unstable_windows"]) == 0
        for h in ("1", "8", "24", "100"):
            hb = R["horizons"][h]
            put(f"s2_rep_{lab}_E{h}", r2(hb["mean"]), f, f"replay_held_out.{blk}.horizons.{h}.mean")
            put(f"s2_rep_{lab}_Nm{h}", sig(hb["torque_Nm"], 2) if not isinstance(hb["torque_Nm"], dict)
                else sig(hb["torque_Nm"]["mean"], 2), f, f"replay_held_out.{blk}.horizons.{h}.torque_Nm")
            if lab == "cal":
                put(f"s2_rep_floor_E{h}", r2(hb["floor"]), f, f"replay_held_out.{blk}.horizons.{h}.floor")
    put("s2_rep_ntraj", str(len(F["replay_held_out"]["calibrated"]["starts"])), f,
        "replay_held_out.calibrated.starts (count)")
    # Gate 1: the authors' policy in the model against the recorded data
    for lab, blk in (("cal", "calibrated"), ("uncal", "uncalibrated")):
        G = F["gait_authors_policy"][blk]["summary"]
        put(f"s2_g1_{lab}_falls", pc2(G["fall_rate"]), f, f"gait_authors_policy.{blk}.summary.fall_rate")
        put(f"s2_g1_{lab}_nfalls", f"{G['fall_rate'] * G['n']:.0f}", f, f"gait_authors_policy.{blk}.summary.fall_rate * n")
        put(f"s2_g1_{lab}_twofeet", r2(G["two_feet"]), f, f"gait_authors_policy.{blk}.summary.two_feet")
        put(f"s2_g1_{lab}_speed", f"{G['speed_err']:.3f}", f, f"gait_authors_policy.{blk}.summary.speed_err")
        put(f"s2_g1_{lab}_turn", f"{G['turn_err']:.3f}", f, f"gait_authors_policy.{blk}.summary.turn_err")
        put(f"s2_g1_{lab}_ret", sig(G["ret"], 3), f, f"gait_authors_policy.{blk}.summary.ret")
    Rc = F["gait_recorded"]["summary"]
    put("s2_g1_rec_twofeet", r2(Rc["two_feet"]), f, "gait_recorded.summary.two_feet")
    put("s2_g1_rec_speed", f"{Rc['speed_err']:.3f}", f, "gait_recorded.summary.speed_err")
    put("s2_g1_rec_turn", f"{Rc['turn_err']:.3f}", f, "gait_recorded.summary.turn_err")
    Gc = F["gait_authors_policy"]["calibrated"]["summary"]
    put("s2_g1_gap_pts", f"{100 * abs(Gc['two_feet'] - Rc['two_feet']):.1f}", f,
        "|gait_authors_policy.calibrated.summary.two_feet - gait_recorded.summary.two_feet| * 100")
    put("s2_g1_speed_ratio", r2(Gc["speed_err"] / Rc["speed_err"]), f,
        "gait_authors_policy.calibrated.summary.speed_err / gait_recorded.summary.speed_err")
    put("s2_g1_n", str(Gc["n"]), f, "gait_authors_policy.calibrated.summary.n")

    # ---- Gate 2: the pilot (validation, not results)
    for pid, k in PILOTKEY.items():
        q = P["policies"][pid]
        put(f"s2_pilot_{k}_falls", pc0(q["fall_rate"]), p, f"policies.{pid}.fall_rate")
        put(f"s2_pilot_{k}_falls_ci", cib(*q["fall_rate_ci95"], f=pc0), p, f"policies.{pid}.fall_rate_ci95")
        put(f"s2_pilot_{k}_ret", r2(q["ret"]), p, f"policies.{pid}.ret")
        put(f"s2_pilot_{k}_ret_ci", cib(*q["ret_ci95"], f=r2), p, f"policies.{pid}.ret_ci95")
        put(f"s2_pilot_{k}_imag", r2(q["imagined_return"]), p, f"policies.{pid}.imagined_return")

    # ---- registered: premise and primary
    B = S["premise_and_primary"]
    put("s2_r0_rho", s2(B["R0"]["rho"]), s, "premise_and_primary.R0.rho")
    put("s2_r0_ci", cib(*B["R0"]["ci95"], f=s2), s, "premise_and_primary.R0.ci95")
    put("s2_r0_label", low(B["R0"]["label"]), s, "premise_and_primary.R0.label")
    for r in ("R1", "R2"):
        k, b = r.lower(), B[r]
        assert b["level"] == 97.5
        put(f"s2_{k}_delta", s2(b["delta"]), s, f"premise_and_primary.{r}.delta")
        put(f"s2_{k}_ci", cib(*b["ci97.5"], f=s2), s, f"premise_and_primary.{r}.ci97.5")
        put(f"s2_{k}_label", low(b["label"]), s, f"premise_and_primary.{r}.label")
        put(f"s2_{k}_new", s2(b["rho_new"]), s, f"premise_and_primary.{r}.rho_new")
        put(f"s2_{k}_base", s2(b["rho_base"]), s, f"premise_and_primary.{r}.rho_base")
    labs = [B["R0"]["label"], B["R1"]["label"], B["R2"]["label"]]
    put("s2_n_settled", str(sum(x != "NOT SETTLED" for x in labs)), s, "premise_and_primary.R0-R2.label (count settled)")

    # ---- secondary (95%)
    X = S["secondary"]
    fr = X["fall_rate"]
    put("s2_f_r0_rho", s2(fr["R0"]["rho"]), s, "secondary.fall_rate.R0.rho")
    put("s2_f_r0_ci", cib(*fr["R0"]["ci95"], f=s2), s, "secondary.fall_rate.R0.ci95")
    put("s2_f_r0_label", low(fr["R0"]["label"]), s, "secondary.fall_rate.R0.label")
    for blk, pre in ((fr, "s2_f"), (X["policy_commands"], "s2_pc")):
        path = "secondary.fall_rate" if pre == "s2_f" else "secondary.policy_commands"
        for r in ("R1", "R2"):
            k, b = r.lower(), blk[r]
            put(f"{pre}_{k}_delta", s2(b["delta"]), s, f"{path}.{r}.delta")
            put(f"{pre}_{k}_ci", cib(*b["ci95"], f=s2), s, f"{path}.{r}.ci95")
            put(f"{pre}_{k}_label", low(b["label"]), s, f"{path}.{r}.label")
            put(f"{pre}_{k}_new", s2(b["rho_new"]), s, f"{path}.{r}.rho_new")
            put(f"{pre}_{k}_base", s2(b["rho_base"]), s, f"{path}.{r}.rho_base")
    ir = X["imagined_return"]
    put("s2_imag_rho", s2(ir["rho"]), s, "secondary.imagined_return.rho")
    put("s2_imag_ci", cib(*ir["ci95"], f=s2), s, "secondary.imagined_return.ci95")
    put("s2_imag_label", low(ir["label"]), s, "secondary.imagined_return.label")
    sel = X["selection"]
    put("s2_sel_best3", r2(sel["best_three_mean"]), s, "secondary.selection.best_three_mean")
    for c, b in sel["checks"].items():
        put(f"s2_sel_{c}", r2(b["shortfall"]), s, f"secondary.selection.checks.{c}.shortfall")
        put(f"s2_sel_{c}_ci", cib(*b["ci95"], f=r2), s, f"secondary.selection.checks.{c}.ci95")
    for k in ("SEL1", "SEL2"):
        b = sel[k]
        put(f"s2_{k.lower()}_delta", s2(b["delta"]), s, f"secondary.selection.{k}.delta")
        put(f"s2_{k.lower()}_ci", cib(*b["ci95"], f=s2), s, f"secondary.selection.{k}.ci95")
        put(f"s2_{k.lower()}_label", low(b["label"]), s, f"secondary.selection.{k}.label")
    assert sel["checks"]["E1"]["chosen"] == sel["checks"]["E8"]["chosen"] == sel["checks"]["E100"]["chosen"] \
        == sel["checks"]["rJ"]["chosen"]
    put("s2_sel_err_chosen", ", ".join(r"\texttt{" + x.replace("_", r"\_") + "}" for x in sel["checks"]["E1"]["chosen"]),
        s, "secondary.selection.checks.E1.chosen (= E8, E100, rJ)")
    rs = X["rollout_subset"]
    assert rs["label"] == "NOT APPLICABLE"
    put("s2_rollout_label", low(rs["label"]), s, "secondary.rollout_subset.label")

    # ---- exploratory: two-level intervals
    E = S["exploratory"]["two_level"]
    assert S["exploratory"]["two_level"]["label"] == "exploratory"
    put("s2_x_r0_ci", cib(*E["R0"]["ci95"], f=s2), s, "exploratory.two_level.R0.ci95")
    put("s2_x_r1_ci", cib(*E["R1"]["ci97.5"], f=s2), s, "exploratory.two_level.R1.ci97.5")
    put("s2_x_r2_ci", cib(*E["R2"]["ci97.5"], f=s2), s, "exploratory.two_level.R2.ci97.5")

    # ---- descriptive: per simulator (table in A12) and the facts quoted in the body
    rows = {r["sim"]: r for r in S["descriptive"]["rows"]}
    assert set(rows) == set(SIMKEY)
    for sim, k in SIMKEY.items():
        r = rows[sim]
        base = f"descriptive.rows[{sim}]"
        put(f"s2_d_{k}_ret", r2(r["ret"]), s, base + ".ret")
        put(f"s2_d_{k}_falls", pc0(r["fall"]), s, base + ".fall")
        put(f"s2_d_{k}_speed", r2(r["speed_err"]), s, base + ".speed_err")
        put(f"s2_d_{k}_imag", sig(r["imag"], 2), s, base + ".imag")
        for c in ("E1", "E8", "E100"):
            put(f"s2_d_{k}_{c}", r2(r[c]), s, f"{base}.{c}")
        put(f"s2_d_{k}_lam1", r2(r["lam1"]), s, base + ".lam1")
        put(f"s2_d_{k}_rJ", sig(r["rJ"], 2), s, base + ".rJ")
        lp = r["lam1_pol"]
        put(f"s2_d_{k}_lam1pol", r2(lp) if lp is not None and lp == lp and abs(lp) != float("inf") else "div.",
            s, base + ".lam1_pol (non-finite: diverged, ranked worst)")
        pr = [q["ret"] for q in r["per_policy"]]
        put(f"s2_d_{k}_ret_p1", r2(pr[0]), s, base + ".per_policy[0].ret")
        put(f"s2_d_{k}_ret_p2", r2(pr[1]), s, base + ".per_policy[1].ret")
        duty = [_ld(os.path.join(EVAL, q["policy"] + ".json"))["summary"]["duty"] for q in r["per_policy"]]
        put(f"s2_d_{k}_duty", r2(sum(duty) / len(duty)), _rel(EVAL), f"{sim}__p1/p2.json summary.duty (mean)")
    # policies that stood still: never fell, four feet down nearly all the time
    still = [sim for sim, r in rows.items() if r["fall"] == 0]
    assert still == ["REL__u50", "REL__s128"], still
    dmin = min(_ld(os.path.join(EVAL, q["policy"] + ".json"))["summary"]["duty"]
               for sim in still for q in rows[sim]["per_policy"])
    put("s2_still_duty_min", f"{dmin:.3f}", _rel(EVAL), "REL__u50/REL__s128 __p1/p2.json summary.duty (min)")
    put("s2_still_n", str(len(still)), s, "descriptive.rows with fall == 0 (count)")
    ranked = sorted(rows, key=lambda x: -rows[x]["ret"])
    assert ranked[:2] == ["REL", "REL__fp16"], ranked
    assert sorted(ranked[2:4]) == sorted(still), ranked          # A12: the still policies rank just below the top two
    assert sorted(still, key=lambda x: -rows[x]["E100"]) == sorted(rows, key=lambda x: -rows[x]["E100"])[:2]  # worst E100
    fl = [q["fall"] for sim in ("REL", "REL__fp16") for q in rows[sim]["per_policy"]]
    put("s2_top_falls_min", pc0(min(fl)), s, "descriptive.rows[REL, REL__fp16].per_policy.fall (min)")
    put("s2_top_falls_max", pc0(max(fl)), s, "descriptive.rows[REL, REL__fp16].per_policy.fall (max)")
    put("s2_ret_min_sim", r2(rows[ranked[-1]]["ret"]), s, f"descriptive.rows[{ranked[-1]}].ret (lowest)")

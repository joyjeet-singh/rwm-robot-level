"""Numbers for Option D, with Option C's records (called by paper_numbers.py after s2_numbers). Every value is read
from a result file; nothing is typed.

The world is the calibrated hand-built MuJoCo ANYmal D with a fitted PD motor (not Isaac Lab, not hardware).
Sources:
  results/d_stats.json        the registered analysis (scripts/d_stats.py, prereg/PREREG_D.md): premise R0, primary
                              R1-R2 (97.5%), secondary (95%), exploratory, descriptive (per simulator, per run)
  results/d_ladder.json       the ladder job: every run, test and policy
  results/d_record.json       Stage 1, the record: every simulator that exists, its checks, walk test and policies
  results/d/walk/<sim>.json   each simulator's walk test (tracking as a share of MuJoCo's)
  results/d/stage1.json       the stop rule applied to stored training logs
  results/c_pilot.json        Option C's Gate A2 (from-scratch simulators)
  results/c/pilot_sims.json, pilot_sims_2S.json   Option C's simulators (training steps)
  results/c/probe.json        Option C's job-design probe (training batch; the 8-hour step budget)
  results/c_diag.json         Option C's diagnosis of why rebuilt-simulator policies fell (D0-D4)
  optc/data/rung_6M.stats.json, rung_1M.stats.json  row counts of Option C's MuJoCo data
  git                         the registration commit of prereg/PREREG_D.md
Secondary results must be printed as secondary, exploratory ones as exploratory, descriptive ones as descriptive.
"""
import datetime as dt
import json
import os
import subprocess

import common as C

RES = C.RESULTS
STATS = os.path.join(RES, "d_stats.json")
LADDER = os.path.join(RES, "d_ladder.json")
RECORD = os.path.join(RES, "d_record.json")
WALK = os.path.join(RES, "d", "walk")
STAGE1 = os.path.join(RES, "d", "stage1.json")
CPILOT = os.path.join(RES, "c_pilot.json")
CSIMS = os.path.join(RES, "c", "pilot_sims.json")
CSIMS2 = os.path.join(RES, "c", "pilot_sims_2S.json")
CDIAG = os.path.join(RES, "c_diag.json")
PROBE = os.path.join(RES, "c", "probe.json")
R6M = os.path.join(C.AIR, "optc", "data", "rung_6M.stats.json")
R1M = os.path.join(C.AIR, "optc", "data", "rung_1M.stats.json")

# simulator id -> LaTeX-safe key part
RUNS = ["W_s0", "W_s1", "Wtf_s0", "M_s0", "Wg_s0", "Mg_s0"]
RUNKEY = {"W_s0": "ws0", "W_s1": "ws1", "Wtf_s0": "wtf", "M_s0": "ms0", "M_s1": "ms1", "Wg_s0": "wg", "Mg_s0": "mg"}
STEPS = [5, 20, 80, 160, 320]
RECKEY = {"REL": "rel", "REL__fp16": "fp16", "REL__int8w": "int8w", "REL__u10": "u10", "REL__u20": "u20",
          "REL__u30": "u30", "REL__u50": "u50", "REL__s192": "s192", "REL__s128": "s128", "A_s0": "a", "B_s0": "b",
          "A_s0__u50": "au50", "A5_s0": "afive", "sim_6M_N8_s0": "csixm", "sim_10k_N8_s0": "ctenk",
          "sim_6M_N8_s0__2S": "csixmtwo", "sim_10k_N8_s0__2S": "ctenktwo"}
CHK = {"E1": "Eone", "E8": "Eeight", "E24": "Etwfour", "E100": "Ehund", "E368": "Elong", "lambda1": "lam", "rJ": "rJ"}


def _rel(path):
    return os.path.relpath(path, os.path.dirname(C.AIR))


def _ld(path):
    with open(path) as fh:
        return json.load(fh)


def simkey(sim):
    """W_s0_d80 -> ws0d80 (LaTeX-safe, no underscores inside the simulator part)."""
    run, steps = sim.rsplit("_d", 1)
    return RUNKEY[run] + "d" + steps


def add(put, sig, sgn, pct, cib, dp, load, rel):
    S, s = _ld(STATS), _rel(STATS)
    L, lr = _ld(LADDER), _rel(LADDER)
    R, rr = _ld(RECORD), _rel(RECORD)
    s2 = lambda v: "0.00" if round(v, 2) == 0 else dp(v, 2, signed=True)                 # noqa: E731
    d2 = lambda v: dp(v, 2)                                                              # noqa: E731
    d3 = lambda v: dp(v, 3)                                                              # noqa: E731
    low = lambda t: t.lower()                                                            # noqa: E731
    pc0 = lambda x: f"{100 * x:.0f}" + r"\%"                                              # noqa: E731
    cnt = lambda n: f"{n:,}".replace(",", "{,}")                                          # noqa: E731

    D = S["descriptive"]
    sims = {x["sim"]: x for x in D["simulators"]}
    checks = D["checks"]
    assert not S["missing"] and not D["stops"]
    assert L["n_tested"] == D["tests"] and L["n_third_policy"] == D["tests_with_third_policy"]

    # ---- registration
    out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h %ct", "--", "prereg/PREREG_D.md"],
                         cwd=C.AIR, capture_output=True, text=True, check=True).stdout.split()
    assert len(out) == 2, out
    put("d_reg_commit", out[0], "git", "log --diff-filter=A -- prereg/PREREG_D.md (%h)")
    put("d_reg_time", dt.datetime.fromtimestamp(int(out[1]), dt.timezone.utc).strftime("%Y-%m-%d %H:%M") + " UTC",
        "git", "log --diff-filter=A -- prereg/PREREG_D.md (%ct, printed in UTC)")

    # ---- the walking score and its references (record)
    A = R["authors"]
    put("d_auth_score", d3(A["score"]), rr, "authors.score")
    put("d_still_score", d3(A["standing_reference"]), rr, "authors.standing_reference")
    put("d_fail_score", d2(S["fail_score"]), s, "fail_score")
    rs = R["stop_rule"]
    put("d_stop_after", str(rs["after"]), rr, "stop_rule.after")
    put("d_stop_run", str(rs["run"]), rr, "stop_rule.run")
    put("d_stop_len", f"{rs['threshold']:.0f}", rr, "stop_rule.threshold")

    # ---- the record (Stage 1): every simulator with a policy, and Step 2's verdicts by this design's rule
    RS = R["simulators"]
    s2sims = [k for k, v in RS.items() if v["kind"] in ("released", "compressed")]
    nfail = sum(RS[k]["verdict"] == "failed" for k in s2sims)
    put("d_s2_n", str(len(s2sims)), rr, "simulators with kind released/compressed (count)")
    put("d_s2_n_failed", str(nfail), rr, "simulators[released/compressed].verdict == failed (count)")
    nsplit = sum(len({p["failed"] for p in RS[k]["policies"] if p.get("role") == "Step 2 registered"}) > 1
                 for k in s2sims)
    put("d_s2_n_split", str(nsplit), rr, "Step 2 registered policy pairs that disagree (count)")
    reg = [p for k in s2sims for p in RS[k]["policies"] if p.get("role") == "Step 2 registered"]
    stops = [p for p in reg if p.get("stop_rule_would_fire_at") is not None]
    put("d_s2_n_pol", str(len(reg)), rr, "Step 2 registered policies (count)")
    put("d_s2_n_stop", str(len(stops)), rr, "registered policies with stop_rule_would_fire_at set (count)")
    assert all(p["failed"] for p in stops)
    for sim, k in RECKEY.items():
        v = RS[sim]
        ex = v["checks"]["expert"]
        for c in ("E1", "E8", "E100"):
            put(f"d_rec_{k}_{CHK[c]}", d3(ex[c]), rr, f"simulators.{sim}.checks.expert.{c}")
        w = v["walk"]
        put(f"d_rec_{k}_wfalls", pc0(w["falls"]), rr, f"simulators.{sim}.walk.falls")
        put(f"d_rec_{k}_wtrack", pc0(w["tracking_ratio"]), rr, f"simulators.{sim}.walk.tracking_ratio")
        pols = [p for p in v["policies"] if p.get("score") is not None]
        if pols:
            txt = ", ".join(d2(p["score"]) if p["score"] >= 0.005 else d3(p["score"]) for p in pols)
            put(f"d_rec_{k}_scores", txt, rr, f"simulators.{sim}.policies[].score (in order)")
            put(f"d_rec_{k}_falls", ", ".join(pc0(p["fall_rate"]) for p in pols), rr,
                f"simulators.{sim}.policies[].fall_rate (in order)")
        elif v["policies"]:
            put(f"d_rec_{k}_pfalls", ", ".join(pc0(p["step2_fall_rate"]) for p in v["policies"]), rr,
                f"simulators.{sim}.policies[].step2_fall_rate (Step 2's protocol; no walking score)")
        if v.get("verdict"):
            put(f"d_rec_{k}_verdict", v["verdict"], rr, f"simulators.{sim}.verdict")
    rp0 = [p for p in RS["REL"]["policies"] if p["policy"] == "REL__p0"][0]
    put("d_rel_p0_score", d2(rp0["score"]), rr, "simulators.REL.policies[REL__p0].score")

    # ---- Option C: the from-scratch simulators and Gate A2
    CP, cp = _ld(CPILOT), _rel(CPILOT)
    g1, g2 = CP["gate_a2"]["first"], CP["gate_a2"]["after_2S"]
    put("c_ga_fell", pc0(g1["share_fell"]), cp, "gate_a2.first.share_fell")
    put("c_ga_track", pc0(g1["tracking_ratio"]), cp, "gate_a2.first.tracking_ratio")
    put("c_gb_fell", pc0(g2["share_fell"]), cp, "gate_a2.after_2S.share_fell")
    put("c_gb_track", pc0(g2["tracking_ratio"]), cp, "gate_a2.after_2S.tracking_ratio")
    put("c_ga_fell_lim", pc0(g1["fall_limit"]), cp, "gate_a2.first.fall_limit")
    put("c_ga_track_lim", pc0(g1["tracking_needed"] / g1["tracking_mujoco_per_step"]), cp,
        "gate_a2.first.tracking_needed / tracking_mujoco_per_step")
    CS, cs = _ld(CSIMS), _rel(CSIMS)
    CS2, cs2 = _ld(CSIMS2), _rel(CSIMS2)
    st1, st2 = CS["sims"]["sim_6M_N8_s0"]["steps"], CS2["sims"]["sim_6M_N8_s0__2S"]["steps"]
    put("c_steps", cnt(st1), cs, "sims.sim_6M_N8_s0.steps")
    put("c_steps_two", cnt(st2), cs2, "sims.sim_6M_N8_s0__2S.steps")
    rows6 = _ld(R6M)["rows"]
    put("c_rows_sixm", sig(rows6 / 1e6, 2), _rel(R6M), "rows / 1e6")
    PR, prr = _ld(PROBE), _rel(PROBE)
    batch = PR["training"]["batch"]
    assert PR["S"] == st1 and st2 == 2 * st1
    put("c_batch", str(batch), prr, "training.batch")
    put("c_pass", sig(st1 * batch / rows6, 1), f"{cs}; {prr}; {_rel(R6M)}",
        "sims.sim_6M_N8_s0.steps * training.batch / rung_6M rows (share of one pass)")
    put("c_pass_two", sig(st2 * batch / rows6, 1), f"{cs2}; {prr}; {_rel(R6M)}",
        "sims.sim_6M_N8_s0__2S.steps * training.batch / rung_6M rows")
    # the diagnosis (root cause of the rebuilt simulators' falls)
    CD, cd = _ld(CDIAG), _rel(CDIAG)
    put("c_d0_rows", cnt(CD["D0"]["rows"]), cd, "D0.rows")
    put("c_d0_falls", str(CD["D0"]["fall_rows"]), cd, "D0.fall_rows")
    single = ["A_s0__p0", "B_s0__p0", "A_s0__u50__p0"]
    mx = max(CD["D1"][p]["imagination"]["sampled"]["share_fell"] for p in single)
    put("c_d1_imfell_max", sig(100 * mx, 1) + r"\%", cd, "max over single-network D1.<p>.imagination.sampled.share_fell")
    put("c_d1_realfell", pc0(min(CD["D1"][p]["physics"]["fall_rate"] for p in single)), cd,
        "min over single-network D1.<p>.physics.fall_rate")
    rat = [CD["D2"][p]["ratio_of_means"] for p in single]
    put("c_d2_min", sig(min(rat), 2), cd, "min over single-network D2.<p>.ratio_of_means")
    put("c_d2_max", sig(max(rat), 2), cd, "max over single-network D2.<p>.ratio_of_means")
    put("c_d2_rel", sig(CD["D2"]["REL__p0"]["ratio_of_means"], 2), cd, "D2.REL__p0.ratio_of_means")
    a0, a1 = CD["D3"]["A5_s0__p0"], CD["D3"]["A5_s0__p1"]
    put("c_d3_p0_falls", pc0(a0["fall_rate"]), cd, "D3.A5_s0__p0.fall_rate")
    put("c_d3_p0_speed", d2(a0["speed_err"]), cd, "D3.A5_s0__p0.speed_err")
    put("c_d3_p1_falls", pc0(a1["fall_rate"]), cd, "D3.A5_s0__p1.fall_rate")
    put("c_d4_share", pc0(CD["D4"]["all_ten_episodes"]["share_exact"]), cd, "D4.all_ten_episodes.share_exact")

    # ---- the ladder
    rows1 = _ld(R1M)["rows"]
    put("d_rows_onem", dp(rows1 / 1e6, 1), _rel(R1M), "rows / 1e6 (one decimal)")
    put("d_rows_onem_n", cnt(rows1), _rel(R1M), "rows")
    runs = D["runs"]
    done = [r for r, v in runs.items() if v["status"] == "done"]
    skipped = [r for r, v in runs.items() if v["status"] != "done"]
    assert skipped == ["M_s1"] and runs["M_s1"]["reason"] == "M_s0 found no boundary"
    put("d_n_runs_all", str(len(runs)), s, "descriptive.runs (count)")
    put("d_n_runs", str(len(done)), s, "descriptive.runs[status == done] (count)")
    put("d_job_h", sig(D["job_hours"], 2), s, "descriptive.job_hours")
    lad = [x for x in D["simulators"] if x["pool"] == "ladder"]
    assert len(lad) == D["tests"]
    put("d_n_tested", str(len(lad)), s, "descriptive.tests")
    put("d_n_failed", str(sum(x["failed"] for x in lad)), s, "descriptive.simulators[pool == ladder].failed (count)")
    put("d_n_walk", str(sum(not x["failed"] for x in lad)), s, "descriptive.simulators[ladder, not failed] (count)")
    put("d_n_third", str(D["tests_with_third_policy"]), s, "descriptive.tests_with_third_policy")
    lp = [p for x in lad for p in x["policies"]]
    put("d_n_pol", str(len(lp)), s, "descriptive.simulators[ladder].policies (count)")
    put("d_n_pol_failed", str(sum(p["failed"] for p in lp)), s, "ladder policies with failed (count)")
    fbs = [p for p in lp if p["status"] == "failed_by_simulator"]
    put("d_n_fbs", str(len(fbs)), s, "ladder policies with status failed_by_simulator (count)")
    assert all(p["policy"].startswith("M") for p in fbs)
    put("d_n_stops", str(len(D["stops"])), s, "descriptive.stops (count)")
    put("d_n_primary", str(len(S["pools"]["primary"])), s, "pools.primary (count)")
    put("d_n_pooled", str(len(S["pools"]["pooled"])), s, "pools.pooled (count)")
    put("d_n_seedzero", str(len(S["pools"]["seed0"])), s, "pools.seed0 (count)")
    rel = sims["REL"]
    put("d_rel_score", d2(rel["score"]), s, "descriptive.simulators[REL].score")
    put("d_rel_p1", d2(rel["policies"][0]["score"]), s, "descriptive.simulators[REL].policies[0].score")
    put("d_rel_p2", d2(rel["policies"][1]["score"]), s, "descriptive.simulators[REL].policies[1].score")
    rx = checks["REL"]["checks"]["expert"]
    for c in ("E1", "E8", "E100", "E368"):
        put(f"d_rel_{CHK[c]}", d3(rx[c]), s, f"descriptive.checks.REL.checks.expert.{c}")
        vals = [checks[x["sim"]]["checks"]["expert"][c] for x in lad]
        put(f"d_min_{CHK[c]}", d3(min(vals)), s, f"min over ladder descriptive.checks.<sim>.checks.expert.{c}")
        put(f"d_max_{CHK[c]}", d3(max(vals)), s, f"max over ladder descriptive.checks.<sim>.checks.expert.{c}")
        put(f"d_n_lower_{CHK[c]}", str(sum(v < rx[c] for v in vals)), s,
            f"ladder simulators with expert {c} below REL's (count)")
    walking = [x for x in lad if not x["failed"]]
    put("d_walk_min", d2(min(x["score"] for x in walking)), s, "min score over ladder simulators that did not fail")
    put("d_walk_max", d2(max(x["score"] for x in walking)), s, "max score over ladder simulators that did not fail")
    put("d_fail_max", d2(max(x["score"] for x in lad if x["failed"])), s, "max score over failed ladder simulators")
    put("d_steps_min", str(min(x["steps"] for x in lad)), s, "min steps over ladder simulators")
    put("d_steps_max", str(max(x["steps"] for x in lad)), s, "max steps over ladder simulators")

    # ladder table cells: mean score, marked when the simulator failed
    for x in lad:
        k = simkey(x["sim"])
        mark = r"$^\times$" if x["failed"] else ""
        put(f"d_l_{k}", d2(x["score"]) + mark, s, f"descriptive.simulators[{x['sim']}].score (+ failed mark)")
    # full ladder table (appendix): checks, walk test, policies
    for x in [rel] + lad:
        k = "rel" if x["sim"] == "REL" else simkey(x["sim"])
        ck = checks[x["sim"]]
        ex = ck["checks"]["expert"]
        for c in ("E1", "E8", "E100", "E368", "lambda1", "rJ"):
            f = (lambda v: sig(v, 3)) if c == "rJ" else d3
            put(f"d_t_{k}_{CHK[c]}", f(ex[c]), s, f"descriptive.checks.{x['sim']}.checks.expert.{c}")
        wf = _ld(os.path.join(WALK, x["sim"] + ".json"))
        put(f"d_t_{k}_wtrack", f"{100 * wf['tracking_ratio']:.1f}" + r"\%", _rel(WALK),
            f"{x['sim']}.json tracking_ratio (one decimal)")
        assert abs(wf["tracking_per_step"] - ck["walk"]["tracking"]) < 1e-12
        put(f"d_t_{k}_wfalls", pc0(ck["walk"]["falls"]), s, f"descriptive.checks.{x['sim']}.walk.falls")
        put(f"d_t_{k}_u", d2(ck["walk"]["u"]), s, f"descriptive.checks.{x['sim']}.walk.u")
        ps = []
        for p in x["policies"]:
            t = d2(p["score"])
            if p["status"] == "failed_by_simulator":
                t += r"$^\ddagger$"
            ps.append(t)
        put(f"d_t_{k}_pols", ", ".join(ps), s, f"descriptive.simulators[{x['sim']}].policies[].score")
        put(f"d_t_{k}_falls", ", ".join(pc0(p["falls"] if "falls" in p else p["fall_rate"]) for p in x["policies"]),
            s, f"descriptive.simulators[{x['sim']}].policies[].fall_rate")
        put(f"d_t_{k}_score", d2(x["score"]), s, f"descriptive.simulators[{x['sim']}].score")
        put(f"d_t_{k}_verdict", "failed" if x["failed"] else "walks", s,
            f"descriptive.simulators[{x['sim']}].failed")

    # ---- registered results
    P, Pr = S["premise"]["R0"], S["primary"]
    put("d_r0_rho", s2(P["rho"]), s, "premise.R0.rho")
    put("d_r0_ci", cib(*P["ci95"], f=s2), s, "premise.R0.ci95")
    put("d_r0_label", low(P["label"]), s, "premise.R0.label")
    for r in ("R1", "R2"):
        b, k = Pr[r], r.lower()
        assert b["level"] == 97.5
        put(f"d_{k}_new", s2(b["rho_new"]), s, f"primary.{r}.rho_new")
        put(f"d_{k}_base", s2(b["rho_base"]), s, f"primary.{r}.rho_base")
        put(f"d_{k}_delta", s2(b["delta"]), s, f"primary.{r}.delta")
        put(f"d_{k}_ci", cib(*b["ci97.5"], f=s2), s, f"primary.{r}.ci97.5")
        put(f"d_{k}_label", low(b["label"]), s, f"primary.{r}.label")
    labs = [P["label"], Pr["R1"]["label"], Pr["R2"]["label"]]
    put("d_n_settled", str(sum(x != "NOT SETTLED" for x in labs)), s, "premise/primary labels (count settled)")

    # secondary (95%)
    Sec = S["secondary"]

    def rset(pre, blk, path, rs=("R0", "R1", "R2")):
        for r in rs:
            b, k = blk[r], r.lower()
            if r == "R0":
                put(f"{pre}_{k}_rho", s2(b["rho"]), s, f"{path}.{r}.rho")
            else:
                assert b["level"] == 95.0
                put(f"{pre}_{k}_new", s2(b["rho_new"]), s, f"{path}.{r}.rho_new")
                put(f"{pre}_{k}_base", s2(b["rho_base"]), s, f"{path}.{r}.rho_base")
                put(f"{pre}_{k}_rho", s2(b["delta"]), s, f"{path}.{r}.delta")
            put(f"{pre}_{k}_ci", cib(*b["ci95"], f=s2), s, f"{path}.{r}.ci95")
            put(f"{pre}_{k}_label", low(b["label"]), s, f"{path}.{r}.label")

    rset("d_pool", Sec["pooled"], "secondary.pooled")
    rset("d_mix", Sec["mixed"], "secondary.mixed")
    rset("d_sz", Sec["seed0_runs"], "secondary.seed0_runs", rs=("R1", "R2"))
    rset("d_sone", Sec["seed1_policy"], "secondary.seed1_policy")
    rset("d_fr", Sec["fall_rate"], "secondary.fall_rate")
    rset("d_ret", Sec["physics_return"], "secondary.physics_return")
    b = Sec["E368_vs_E8"]
    put("d_long_new", s2(b["rho_new"]), s, "secondary.E368_vs_E8.rho_new")
    put("d_long_rho", s2(b["delta"]), s, "secondary.E368_vs_E8.delta")
    put("d_long_ci", cib(*b["ci95"], f=s2), s, "secondary.E368_vs_E8.ci95")
    put("d_long_label", low(b["label"]), s, "secondary.E368_vs_E8.label")
    for w, wk in (("tracking", "track"), ("falls", "falls"), ("u", "u")):
        b = Sec["walk_test"][w]
        put(f"d_wt_{wk}_rho", s2(b["rho"]["rho"]), s, f"secondary.walk_test.{w}.rho.rho")
        put(f"d_wt_{wk}_ci", cib(*b["rho"]["ci95"], f=s2), s, f"secondary.walk_test.{w}.rho.ci95")
        put(f"d_wt_{wk}_label", low(b["rho"]["label"]), s, f"secondary.walk_test.{w}.rho.label")
        put(f"d_wt_{wk}_d", s2(b["minus_E8"]["delta"]), s, f"secondary.walk_test.{w}.minus_E8.delta")
        put(f"d_wt_{wk}_dci", cib(*b["minus_E8"]["ci95"], f=s2), s, f"secondary.walk_test.{w}.minus_E8.ci95")
        put(f"d_wt_{wk}_dlabel", low(b["minus_E8"]["label"]), s, f"secondary.walk_test.{w}.minus_E8.label")
    au = Sec["failed_auc"]
    put("d_auc_nfail", str(au["E1"]["n_failed"]), s, "secondary.failed_auc.E1.n_failed")
    for c, b in au.items():
        put(f"d_auc_{CHK[c]}", d2(b["auc"]), s, f"secondary.failed_auc.{c}.auc")
        put(f"d_auc_{CHK[c]}_ci", cib(*b["ci95"], f=d2), s, f"secondary.failed_auc.{c}.ci95")
        put(f"d_auc_{CHK[c]}_label", low(b["label"]), s, f"secondary.failed_auc.{c}.label")
    put("d_auc_n_settled", str(sum(b["label"] != "NOT SETTLED" for b in au.values())), s,
        "secondary.failed_auc.*.label (count settled)")
    B3 = Sec["best_three"]
    put("d_bt_mean", d2(B3["best_three_mean"]), s, "secondary.best_three.best_three_mean")
    nice = lambda ids: ", ".join(i.replace("_", r"\_") for i in ids)                     # noqa: E731
    put("d_bt_best", nice(B3["best_three"]), s, "secondary.best_three.best_three")
    for c, b in B3["checks"].items():
        put(f"d_bt_{CHK[c]}", d2(b["shortfall"]), s, f"secondary.best_three.checks.{c}.shortfall")
        put(f"d_bt_{CHK[c]}_ci", cib(*b["ci95"], f=d2), s, f"secondary.best_three.checks.{c}.ci95")
        put(f"d_bt_{CHK[c]}_chosen", nice(b["chosen"]), s, f"secondary.best_three.checks.{c}.chosen")
    b = S["exploratory"]["rA_vs_E8"]
    put("d_ra_new", s2(b["rho_new"]), s, "exploratory.rA_vs_E8.rho_new")
    put("d_ra_rho", s2(b["delta"]), s, "exploratory.rA_vs_E8.delta")
    put("d_ra_ci", cib(*b["ci95"], f=s2), s, "exploratory.rA_vs_E8.ci95")
    put("d_ra_label", low(b["label"]), s, "exploratory.rA_vs_E8.label")

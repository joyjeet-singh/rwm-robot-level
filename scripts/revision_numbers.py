"""Numbers for the revision (called by paper_numbers.py after air_numbers). Every value is read from a result file.
Exploratory results (A1, A2, A3 from results/revision_exploratory.json) must be labelled "(exploratory)" where printed.
"""
import json
import os
import statistics

import common as C

REVISION = os.path.join(C.RESULTS, "revision_exploratory.json")
RWM_RESULTS = os.path.join(C.PDM, "rwm_repro", "results")
CHECKS = ("E1", "E8", "lam1", "rJ")


def _rel(path):
    return os.path.relpath(path, os.path.dirname(C.AIR))


def _ld(path):
    with open(path) as fh:
        return json.load(fh)


def add(put, sig, sgn, pct, cib, dp, load, rel):
    R, r = _ld(REVISION), _rel(REVISION)
    r2 = lambda v: dp(v, 2, signed=True)                                                 # noqa: E731
    r3 = lambda v: dp(v, 3, signed=True)                                                 # noqa: E731
    yn = lambda b: "yes" if b else "no"                                                  # noqa: E731

    # ---- a. A1: horizon-matched checks against E100 (exploratory)
    A1, a1 = R["A1_horizon_matched"], "A1_horizon_matched"
    sets = {"F18": (A1["F18"]["all"]["E100"], "F18.all.E100"), "X5": (A1["X5"]["E100"], "X5.E100")}
    for name, (blk, path) in sets.items():
        for k in ("lam1_100", "lam1_8", "rJ_100", "rJ_8", "E8"):
            key = f"a1_{name}_{k}"
            put(key + "_rho", r2(blk[k]["rho"]), r, f"{a1}.{path}.{k}.rho")
            put(key + "_ci", cib(*blk[k]["ci95"], f=r2), r, f"{a1}.{path}.{k}.ci95")
    for d, lab in (("rJ_8_minus_E8", "rj8_e8"), ("rJ_100_minus_E8", "rj100_e8")):
        for name, sub in (("F18", "F18_all"), ("X5", "X5")):
            v = A1["diffs_vs_E100"][d][sub]
            put(f"a1_{name}_{lab}_delta", r3(v["delta"]), r, f"{a1}.diffs_vs_E100.{d}.{sub}.delta")
            put(f"a1_{name}_{lab}_ci", cib(*v["ci95"], f=r3), r, f"{a1}.diffs_vs_E100.{d}.{sub}.ci95")

    # horizon-matched table: every check window against E100 for each F18 pool and X5 (appendix A3, exploratory)
    for name, blk, path in (("F18all", A1["F18"]["all"]["E100"], "F18.all.E100"),
                            ("F18tr", A1["F18"]["trained"]["E100"], "F18.trained.E100"),
                            ("F18cp", A1["F18"]["compressed"]["E100"], "F18.compressed.E100"),
                            ("X5", A1["X5"]["E100"], "X5.E100")):
        for k in ("lam1_8", "lam1_24", "lam1_100", "lam1_368", "rJ_8", "rJ_24", "rJ_100", "rJ_368", "E1", "E8"):
            put(f"a1t_{name}_{k}", r2(blk[k]["rho"]), r, f"{a1}.{path}.{k}.rho")

    # ---- b. A2: two-level intervals (exploratory)
    A2, a2 = R["A2a_two_level"], "A2a_two_level"
    put("a2_ndraws", f"{A2['n_draws']:,}".replace(",", "{,}"), r, f"{a2}.n_draws")
    ratios = []
    for k, v in A2["table2"].items():
        low = k.lower()
        put(f"a2_t2_{low}_ci", cib(*v["ci95_two_level"], f=r2), r, f"{a2}.table2.{k}.ci95_two_level")
        put(f"a2_t2_{low}_ratio", sig(v["width_ratio"], 2), r, f"{a2}.table2.{k}.width_ratio")
        ratios.append(v["width_ratio"])
    t2_ratios = list(ratios)
    tests = A2["registered_tests"]
    held = 0
    for k, v in tests.items():
        low = k.lower()
        lo, hi = v["ci_two_level"]
        excl = lo > 0 or hi < 0
        assert excl == v["two_level_excludes_zero"], k
        if excl:
            assert (lo > 0) == (v["estimate"] > 0), k
        holds = excl == (v["registered_verdict"] != "NOT SETTLED")
        held += holds
        put(f"a2_{low}_ci", cib(lo, hi, f=r2), r, f"{a2}.registered_tests.{k}.ci_two_level (level {v['level']}%)")
        put(f"a2_{low}_ratio", sig(v["width_ratio"], 2), r, f"{a2}.registered_tests.{k}.width_ratio")
        put(f"a2_{low}_holds", yn(holds), r,
            f"{a2}.registered_tests.{k}: two-level interval gives the registered verdict {v['registered_verdict']}")
        ratios.append(v["width_ratio"])
    for k in ("Q1", "Q2"):
        put(f"a2_{k.lower()}_ci95", cib(*tests[k]["ci95_two_level"], f=r2), r,
            f"{a2}.registered_tests.{k}.ci95_two_level")
    put("a2_n_tests", str(len(tests)), r, f"{a2}.registered_tests (count)")
    put("a2_n_hold", str(held), r, "count of registered_tests whose two-level interval gives the registered verdict")
    put("a2_n_lost", str(len(tests) - held), r, "count of registered_tests that do not")
    put("a2_median_ratio_t2", sig(statistics.median(t2_ratios), 2), r, f"{a2}.table2.*.width_ratio (median of 16)")
    put("a2_median_ratio_tests", sig(statistics.median([v["width_ratio"] for v in tests.values()]), 2), r,
        f"{a2}.registered_tests.*.width_ratio (median of 13)")
    put("a2_median_ratio", sig(statistics.median(ratios), 2), r, "median of all 29 width_ratio values")
    put("a2_ratio_min", sig(min(ratios), 2), r, "minimum of all 29 width_ratio values")
    put("a2_ratio_max", sig(max(ratios), 2), r, "maximum of all 29 width_ratio values")

    # X5 without the split {1, 8} (exploratory)
    B, b = R["A2b_X5_without_split18"], "A2b_X5_without_split18"
    for k in CHECKS:
        put(f"a2b_X5_{k}_rho", r2(B["table2_X5"][k]["rho"]), r, f"{b}.table2_X5.{k}.rho")
        put(f"a2b_X5_{k}_ci", cib(*B["table2_X5"][k]["ci95"], f=r2), r, f"{b}.table2_X5.{k}.ci95")
    for k, v in B["diffs"].items():
        low = k.lower()
        put(f"a2b_{low}_delta", r2(v["delta"]), r, f"{b}.diffs.{k}.delta")
        put(f"a2b_{low}_ci", cib(*v["ci95"], f=r2), r, f"{b}.diffs.{k}.ci95")

    # Holm over the seven secondary verdicts (exploratory)
    H, h = R["A2c_holm"], "A2c_holm"
    put("a2c_alpha", dp(H["alpha_familywise"], 2), r, f"{h}.alpha_familywise")
    for k, v in H["tests"].items():
        low = k.lower()
        p = "$<$0.0001" if v["p"] == 0 else dp(v["p"], 4)
        put(f"a2c_{low}_p", p, r, f"{h}.tests.{k}.p")
        put(f"a2c_{low}_pholm", "$<$0.001" if v["p_holm"] == 0 else dp(v["p_holm"], 3), r, f"{h}.tests.{k}.p_holm")
        put(f"a2c_{low}_survives", yn(k in H["survive"]), r, f"{h}.survive")
        assert (k in H["survive"]) == v["rejects_zero"], k
        put(f"a2c_{low}_near", yn(v["within_2se_of_threshold"]), r, f"{h}.tests.{k}.within_2se_of_threshold")
    put("a2c_n_survive", str(len(H["survive"])), r, f"{h}.survive (count)")
    put("a2c_n_family", str(len(H["family"])), r, f"{h}.family (count)")

    # ---- c. A3: split-half (exploratory)
    A3, a3 = R["A3_split_half"], "A3_split_half"
    drops = []
    for name, blk, path in (("F18", A3["F18"]["all"], "F18.all"), ("X5", A3["X5"], "X5")):
        for k in CHECKS:
            v = blk[k]
            put(f"a3_{name}_{k}_rho", r2(v["rho"]), r, f"{a3}.{path}.{k}.rho")
            put(f"a3_{name}_{k}_ci", cib(*v["ci95"], f=r2), r, f"{a3}.{path}.{k}.ci95")
            put(f"a3_{name}_{k}_same", r2(v["same_half"]["rho"]), r, f"{a3}.{path}.{k}.same_half.rho")
            drops.append(-v["split_minus_same_half"]["delta"])
    put("a3_max_drop", dp(max(drops), 2), r, "max over F18 all and X5 of (same-half rho minus split-half rho)")

    # ---- d. growth-rate shifts: lamshift_*, lam_*, lampos_* exist (air_numbers); totals here
    S, s = load("stats"), rel("stats")
    sh = S["lam_shift_X5"]
    removed = ("u10", "u20", "u30", "u50", "s192", "s128")
    wide = ("u50", "s192", "s128")
    assert all(sh[v]["n_parents"] == 25 for v in removed)
    put("n_unstable_removed", str(sum(sh[v]["n_positive_lam1"] for v in removed)), s,
        "sum of lam_shift_X5.<v>.n_positive_lam1, v in u10,u20,u30,u50,s192,s128")
    put("n_removed_copies", str(sum(sh[v]["n_parents"] for v in removed)), s,
        "sum of lam_shift_X5.<v>.n_parents over the same six variants")
    put("n_unstable_wide", str(sum(sh[v]["n_positive_lam1"] for v in wide)), s,
        "sum of n_positive_lam1 over u50, s192, s128")
    put("n_wide_copies", str(sum(sh[v]["n_parents"] for v in wide)), s, "sum of n_parents over u50, s192, s128")

    # ---- e. scale of the released model's data: 1 / (the fraction in budget_pct)
    t = os.path.join(RWM_RESULTS, "paper_numbers.json")
    c2 = float(_ld(t)["c2_pct"]["value"])
    put("scale_ratio", "about " + str(int(round(100.0 / c2, -1))), _rel(t), "c2_pct: 100 / c2_pct, to the nearest 10")

    earlier_studies(put, sig, cib, dp)
    study_ft(put, sig, sgn, cib, dp)


def _exp10(x, d=0):
    """1e-05 -> $1\\times10^{-5}$; d: decimals of the mantissa"""
    m, e = f"{x:.{d}e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def earlier_studies(put, sig, cib, dp):
    r2 = lambda v: dp(v, 2, signed=True)                                                 # noqa: E731

    # ---- f. stale command on the held-out pair at h = 100, released checkpoint (earlier study; comparison only)
    f = os.path.join(RWM_RESULTS, "n5_stale_armA_ci.json")
    d = _ld(f)
    assert d["horizons"] == [1, 8, 32, 100, 128, 368]
    v = d["results"]["released"]["held_out_n4"]["by_horizon"]["100"]
    path = "results.released.held_out_n4.by_horizon[100]"
    put("stale_ho_h100", sig(v["E_pct"], 2) + r"\%", _rel(f), path + ".E_pct")
    put("stale_ho_h100_ci", cib(*v["ci95_pct"], f=lambda x: sig(x, 2) + r"\%"), _rel(f), path + ".ci95_pct")

    # ---- g. training settings of the rollout-trained parents (Arm A, five folds x five seeds)
    M = _ld(os.path.join(C.RESULTS, "models.json"))["X5"]
    parents = sorted({r["key"].split("__")[0] for r in M if r["variant"]})
    assert len(parents) == 25 and all(k.startswith("A_") for k in parents), parents
    recs = {}
    for k in parents:
        _, fold, sd = k.split("_")
        name = f"step5_armA_seed{sd[1:]}.json" if fold == "18" else f"x3_run_armA_ho{fold}_seed{sd[1:]}.json"
        recs[k] = _ld(os.path.join(RWM_RESULTS, name))
    ref_file = os.path.join(RWM_RESULTS, "step5_armA_seed0.json")
    ref = _ld(ref_file)["hyperparameters"]
    same = ("batch", "iterations", "learning_rate", "weight_decay", "loss_weights", "action_offset", "gradient_clipping")
    for k, rec in recs.items():
        assert rec["arm"] == "A" and not rec["teacher_forcing"], k
        assert all(rec["hyperparameters"][h] == ref[h] for h in same), k
    others = [rec["hyperparameters"] for k, rec in recs.items() if not k.startswith("A_18")]
    assert len(others) == 20 and len({(h["rnn_hidden_size"], h["rnn_num_layers"]) for h in others}) == 1
    src = _rel(ref_file)
    optim_src = os.path.join(C.RWM, "src", "rwm_train.py")
    assert "torch.optim.Adam(" in open(optim_src).read()
    put("hp_optimiser", "Adam", _rel(optim_src), "make_optimizer: torch.optim.Adam")
    put("hp_lr", _exp10(ref["learning_rate"]), src, "hyperparameters.learning_rate (equal in all 25 parents' records)")
    put("hp_wd", _exp10(ref["weight_decay"]), src, "hyperparameters.weight_decay")
    put("hp_batch", str(ref["batch"]), src, "hyperparameters.batch")
    put("hp_iters", f"{ref['iterations']:,}".replace(",", "{,}"), src, "hyperparameters.iterations")
    nz = {k: w for k, w in ref["loss_weights"].items() if w != 0}
    put("hp_loss_weights", ", ".join(f"{k} {w:g}" for k, w in nz.items()), src, "hyperparameters.loss_weights (non-zero)")
    put("hp_n_loss_terms", str(len(nz)), src, "number of non-zero loss weights")
    assert ref["gradient_clipping"] is None
    put("hp_clip", "none", src, "hyperparameters.gradient_clipping (null)")
    put("hp_n_records", str(len(recs)), src, "parents whose records were compared (step5_armA_seed0-4, x3_run_armA_ho*_seed0-4)")

    # how the other F18 models' records differ from the rollout-trained parent (appendix A1)
    def hp_of(name):
        return _ld(os.path.join(RWM_RESULTS, name))
    iters = lambda n: f"{n:,}".replace(",", "{,}")                                       # noqa: E731
    tfs = [hp_of(f"step5_armB_seed{i}.json") for i in range(3)]
    assert all(t["teacher_forcing"] and all(t["hyperparameters"][h] == ref[h] for h in same) for t in tfs)
    put("hp_tf_n_records", str(len(tfs)), _rel(os.path.join(RWM_RESULTS, "step5_armB_seed0.json")),
        "step5_armB_seed0-2: teacher_forcing true, batch/iterations/lr/wd/loss weights/clipping equal to the parent's")
    n1s = [hp_of(f"mn_sweep_run_M32_N1_seed{i}.json") for i in range(3)]
    n8 = hp_of("mn_sweep_run_M16_N8_seed0.json")["hyperparameters"]
    n1_src = _rel(os.path.join(RWM_RESULTS, "mn_sweep_run_M32_N1_seed0.json"))
    for t in n1s:
        h = t["hyperparameters"]
        assert not t["teacher_forcing"] and h["forecast_horizon"] == 1 and h["history_horizon"] == 32
        assert all(h[k] == ref[k] for k in same) and h["rnn_hidden_size"] == 256 and h["rnn_num_layers"] == 2
    put("hp_n1_N", str(n1s[0]["hyperparameters"]["forecast_horizon"]), n1_src, "hyperparameters.forecast_horizon")
    put("hp_M", str(n1s[0]["hyperparameters"]["history_horizon"]), n1_src, "hyperparameters.history_horizon (M=32)")
    put("hp_hidden", str(n1s[0]["hyperparameters"]["rnn_hidden_size"]), n1_src, "hyperparameters.rnn_hidden_size")
    parentA = [r for r in _ld(os.path.join(C.RESULTS, "models.json"))["F18"] if r["key"] == "A_s0"][0]
    put("hp_parent_N", str(parentA["extra"]["N"]), _rel(os.path.join(C.RESULTS, "models.json")), "F18 A_s0 extra.N")
    put("hp_n1_n_records", str(len(n1s)), n1_src, "mn_sweep_run_M32_N1_seed0-2 compared with the parent")
    t10 = hp_of("step5_armA_seed0_10k.json")["hyperparameters"]
    assert all(t10[k] == ref[k] for k in same if k != "iterations")
    put("hp_10k_iters", iters(t10["iterations"]), _rel(os.path.join(RWM_RESULTS, "step5_armA_seed0_10k.json")),
        "hyperparameters.iterations (other settings equal)")
    w124 = hp_of("step5_armA_seed0_m49h124.json")["hyperparameters"]
    put("hp_w124_hidden", str(w124["rnn_hidden_size"]), _rel(os.path.join(RWM_RESULTS, "step5_armA_seed0_m49h124.json")),
        "hyperparameters.rnn_hidden_size")
    put("hp_ens_n", str(hp_of("step5_armA_seed0_ens5.json")["hyperparameters"]["ensemble"]),
        _rel(os.path.join(RWM_RESULTS, "step5_armA_seed0_ens5.json")), "hyperparameters.ensemble")

    # ---- h. what was known before the registration (PREREG.md section 2)
    def rho(file, path, rho_key, ci_key, name, n=None):
        d = _ld(file)
        for p in path:
            d = d[p]
        put(f"pre_{name}_rho", r2(d[rho_key]), _rel(file), ".".join(path) + "." + rho_key)
        put(f"pre_{name}_ci", cib(*d[ci_key], f=r2), _rel(file), ".".join(path) + "." + ci_key)
        if n:
            put(f"pre_{name}_n", str(d[n]), _rel(file), ".".join(path) + "." + n)
        return d

    ng = os.path.join(C.PDM, "NGCAPS", "results", "main.json")
    rho(ng, ["Q1"], "rho", "ci", "ng_R", "n")
    rho(ng, ["Q1_secondary_RT"], "rho", "ci", "ng_RT", "n")
    ic = os.path.join(C.PDM, "ICMAI", "results", "lyap_main.json")
    rho(ic, ["H2"], "rho", "ci", "icmai_all", "n_models")
    h3 = rho(ic, ["H3"], "rho", "ci", "icmai_roll", "n_models")
    assert h3["verdict"] == "NOT SETTLED"
    ip = os.path.join(C.PDM, "ICPAIAS", "results", "physics_monitor.json")
    q3 = _ld(ip)["Q3"]["mean_rJ"]
    put("pre_icp_rho", r2(q3["spearman"]), _rel(ip), "Q3.mean_rJ.spearman")
    put("pre_icp_ci", cib(*q3["ci"], f=r2), _rel(ip), "Q3.mean_rJ.ci")
    put("pre_icp_n", str(_ld(ip)["Q3"]["n_models"]), _rel(ip), "Q3.n_models")

    # ---- i. PerconAI unit removal: fine-tuned narrow copies against the equally fine-tuned control
    pf = os.path.join(C.PDM, "PerconAI", "results", "paper_numbers.json")
    K = _ld(pf)["keys"]
    pr = _rel(pf)
    kp = lambda x: sig(100 * x, 2) + r"\%"                                              # noqa: E731
    reads = {"first": "r2", "rep": "r3.rep", "pool": "r3.pool"}
    for tag, pre in reads.items():
        for w in (192, 128):
            for h in (100, 368):
                b = f"{pre}.q3.w{w}.h{h}"
                put(f"perc_K_{tag}_w{w}_h{h}", kp(K[b + ".K.est"]["value"]), pr, f"keys.{b}.K.est")
                put(f"perc_K_{tag}_w{w}_h{h}_ci", cib(K[b + ".K.lo"]["value"], K[b + ".K.hi"]["value"], f=kp), pr,
                    f"keys.{b}.K.lo/hi")
                put(f"perc_K_{tag}_w{w}_h{h}_label", K[b + ".label"]["value"].lower(), pr, f"keys.{b}.label")
            q4 = f"{pre}.q4.w{w}." + ("it3000_h368" if tag == "first" else "h368")
            assert K[q4 + ".iterations"]["value"] == 3000
            put(f"perc_Q4_{tag}_w{w}", K[q4 + ".label"]["value"].lower(), pr, f"keys.{q4}.label")
            put(f"perc_Q4_{tag}_w{w}_delta", r2(K[q4 + ".delta.est"]["value"]), pr, f"keys.{q4}.delta.est")
            put(f"perc_Q4_{tag}_w{w}_ci", cib(K[q4 + ".delta.lo"]["value"], K[q4 + ".delta.hi"]["value"], f=r2), pr,
                f"keys.{q4}.delta.lo/hi")
    ko = {"first": "r3.korig.s0", "rep": "r3.korig.rep", "pool": "r3.korig.pool"}
    for tag, pre in ko.items():
        for h in (100, 368):
            put(f"perc_ctrl_{tag}_h{h}", kp(K[f"{pre}.h{h}.K.est"]["value"]), pr, f"keys.{pre}.h{h}.K.est")
            put(f"perc_ctrl_{tag}_h{h}_ci", cib(K[f"{pre}.h{h}.K.lo"]["value"], K[f"{pre}.h{h}.K.hi"]["value"], f=kp),
                pr, f"keys.{pre}.h{h}.K.lo/hi")
    assert abs(K["r2.korig.control_w256_500it.h368.K.est"]["value"] - K["r3.korig.s0.h368.K.est"]["value"]) < 1e-4


def study_ft(put, sig, sgn, cib, dp):
    """Study FT (prereg/PREREG_FT.md): registered; results/ft_stats.json is written by the registered ft_stats.py."""
    fs, fm = os.path.join(C.RESULTS, "ft_stats.json"), os.path.join(C.RESULTS, "ft_models.json")
    S, M, rs, rm = _ld(fs), _ld(fm), _rel(fs), _rel(fm)

    def fk(x):
        """skill retained in per cent: one decimal from 95% up, else whole per cents; values above 100% are kept"""
        v = 100 * x
        return (f"{v:.1f}" if v >= 95 else f"{v:.0f}").replace("-", "$-$") + r"\%"
    pts = lambda x: dp(100 * x, 1, signed=True)                                          # noqa: E731
    pts_ci = lambda x: dp(100 * x, 1)                                                    # noqa: E731
    lam = lambda x: dp(x, 3, signed=True)                                                # noqa: E731
    lam_ci = lambda x: dp(x, 3)                                                          # noqa: E731
    rj = lambda x: sgn(x, 2)                                                             # noqa: E731
    rj_ci = lambda x: sig(x, 2)                                                          # noqa: E731

    assert not S["excluded_arms"] and not S["C-FT4_failures"]
    put("ft_arms_excluded", "none", rs, "excluded_arms")
    put("ft_ntraj", str(S["n_trajectories"]), rs, "n_trajectories")
    put("ft_ndraws", f"{S['n_draws']:,}".replace(",", "{,}"), rs, "n_draws")
    put("ft_seeds", "--".join(str(x) for x in (min(S["seeds"]), max(S["seeds"]))), rs, "seeds")

    # ---- verdicts and differences (primary: 97.5% intervals; secondary: 95%)
    blocks = [("primary", q, "ci97.5") for q in S["primary"]] + [("secondary", q, "ci95") for q in S["secondary"]]
    undefined = 0
    for part, q, lev in blocks:
        v, path, key = S[part][q], f"{part}.{q}", "ft_" + q.replace("_minus_", "_")
        put(key + "_verdict", v["verdict"].lower(), rs, path + ".verdict")
        undefined += v.get("undefined_draws", 0)
        if q.startswith("FT1"):
            put(key + "_d", pts(v["d"]), rs, path + ".d (x100, points)")
            put(key + "_ci", cib(*v[lev], f=pts_ci), rs, f"{path}.{lev} (x100, points)")
            put(key + "_Kft", fk(v["K_orig_ft"]), rs, path + ".K_orig_ft")
            put(key + "_Knoft", fk(v["K_orig_noft"]), rs, path + ".K_orig_noft")
        elif q.startswith("FT2"):
            put(key + "_K", fk(v["K_ctrl"]), rs, path + ".K_ctrl")
            put(key + "_ci", cib(*v[lev], f=fk), rs, f"{path}.{lev}")
        elif q == "onestep_U30F":
            put(key + "_d", pts(v["d"]), rs, path + ".d (x100, points)")
            put(key + "_ci", cib(*v["ci95"], f=pts_ci), rs, path + ".ci95 (x100, points)")
        else:
            f, fc = (lam, lam_ci) if "lam1" in q else (rj, rj_ci)
            put(key + "_mean", f(v["mean"]), rs, path + ".mean")
            put(key + "_ci", cib(*v["ci95"], f=fc), rs, path + ".ci95")
            put(key + "_P", dp(v["P_mean"], 2) if "lam1" in q else sig(v["P_mean"], 2), rs, path + ".P_mean")
    assert undefined == 0
    put("ft_undefined_draws", str(undefined), rs, "primary/secondary.*.undefined_draws (sum)")

    # ---- descriptive K (95%) of every arm against the parent P and against the control CTRL
    for k, v in S["descriptive"].items():
        put("ft_" + k, fk(v["K"]), rs, f"descriptive.{k}.K")
        put("ft_" + k + "_ci", cib(*v["ci95"], f=fk), rs, f"descriptive.{k}.ci95")

    # ---- checks C-FT1 to C-FT4 (results/ft_models.json)
    ch = M["checks"]
    for c in ("C-FT1", "C-FT2", "C-FT3", "C-FT4"):
        ok = sum(1 for x in ch[c].values() if (x["passed"] if isinstance(x, dict) else x) is True)
        assert ok == len(ch[c])
        put("ft_" + c.replace("-", "").lower() + "_pass", f"{ok} of {len(ch[c])}", rm, f"checks.{c}.*.passed")
    put("ft_cft4_maxrel", _exp10(max(x["max_rel_per_trajectory"] for x in ch["C-FT4"].values()), 1), rm,
        "checks.C-FT4.*.max_rel_per_trajectory (max)")
    put("ft_n_scored", str(len(M["models"])), rm, "models (count)")

    # ---- runs: count, wall-clock time and pruned entries (results/raw/ft/*.json)
    import glob
    runs = [_ld(f) for f in sorted(glob.glob(os.path.join(C.RESULTS, "raw", "ft", "*.json")))]
    rr = _rel(os.path.join(C.RESULTS, "raw", "ft"))
    assert all(r["trained"] and r["mask_check"]["passed"] and not r["non_finite_traces"] for r in runs)
    put("ft_nruns", str(len(runs)), rr, "*.json (count)")
    par = {r["source_run"] for r in runs if abs(r["level"] - 0.30) < 1e-9}
    assert par == {r["source_run"] for r in runs if abs(r["level"] - 0.50) < 1e-9}
    put("ft_nparents", str(len(par)), rr, "*.json source_run (distinct, level 0.3)")
    wall = [r["wall_clock_s"] for r in runs]
    put("ft_wall_mean_min", sig(statistics.mean(wall) / 60, 2), rr, "*.json wall_clock_s (mean / 60)")
    put("ft_wall_min_min", sig(min(wall) / 60, 2), rr, "*.json wall_clock_s (min / 60)")
    put("ft_wall_max_min", sig(max(wall) / 60, 2), rr, "*.json wall_clock_s (max / 60)")
    put("ft_wall_total_h", sig(sum(wall) / 3600, 2), rr, "*.json wall_clock_s (sum / 3600)")
    put("ft_s_per_iter", sig(statistics.mean(r["s_per_iter"] for r in runs), 2), rr, "*.json s_per_iter (mean)")
    iters = {len(r["curves"]["total"]) for r in runs}
    assert len(iters) == 1
    put("ft_iters", str(iters.pop()), rr, "*.json curves.total (length)")
    for lv in (0.30, 0.50):
        n = {r["pruned_entries"] for r in runs if abs(r["level"] - lv) < 1e-9}
        assert len(n) == 1
        put(f"ft_pruned_u{int(round(100 * lv))}", f"{n.pop():,}".replace(",", "{,}"), rr,
            f"*.json pruned_entries (level {lv})")

    # ---- the time rule (results/ft_probe.json, written before any fine-tuned model existed) and scoring time
    fp = os.path.join(C.RESULTS, "ft_probe.json")
    P, rp = _ld(fp), _rel(fp)
    assert P["run_u50f"] is True
    put("ft_probe_spi", sig(P["s_per_iter_mean_after_first"], 2), rp, "s_per_iter_mean_after_first")
    put("ft_probe_total_h", sig(P["projected_hours"]["total_with_u50"], 2), rp, "projected_hours.total_with_u50")
    put("ft_cap_h", sig(P["cap_hours"], 1), rp, "cap_hours")
    fg = os.path.join(C.RESULTS, "ft", "progress.json")
    G = _ld(fg)
    assert G["done"] == G["of"] and not G["missing_runs"]
    put("ft_score_min", sig(G["elapsed_s"] / 60, 2), _rel(fg), "elapsed_s / 60")

    # ---- the registration: the commit that added the pre-registration (git)
    import subprocess
    out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h %ci", "--", "prereg/PREREG_FT.md"],
                         cwd=C.AIR, capture_output=True, text=True, check=True).stdout.split()
    assert len(out) == 4, out
    put("ft_reg_commit", out[0], "git", "log --diff-filter=A -- prereg/PREREG_FT.md (%h)")
    put("ft_reg_time", f"{out[1]} {out[2][:5]}", "git", "log --diff-filter=A -- prereg/PREREG_FT.md (%ci, UTC" +
        out[3] + ")")

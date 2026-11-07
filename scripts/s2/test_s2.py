"""M4 tests: the registered analysis on synthetic results, and the main job's plan, resume and failure handling.

    ./py-mj scripts/s2/test_s2.py        # -> prints ALL PASS; results to .cache/s2_m4_tests.json

  a  labels from constructed intervals (common.label3), every label of R0, R1, R2 and the boundaries;
  b  synthetic results built so that each label comes out of the full bootstrap: R0 SUPPORTED / CONTRADICTED /
     NOT SETTLED, R1 LABEL-FREE BETTER / ONE-STEP BETTER / NOT SETTLED, R2 LABEL-FREE BETTER /
     TRAINING-HORIZON BETTER / NOT SETTLED; and rho(-x, y) = -rho(x, y) when everything is finite;
  c  a non-finite check ranks worst (errors and the imagined return); a constant outcome is NOT COMPUTABLE;
  d  the whole analysis on a synthetic results folder: missing policies (one seed missing: the other is used; both
     missing: the simulator is dropped and listed), selection (a perfect check has shortfall 0), the rollout
     subset NOT APPLICABLE, the report written, and no result file overwritten;
  e  the registered inputs: s2_stats.POOL equals fallback_checks.SIMS, checks.json matches the registered hash and
     holds every simulator, and the released model's recomputed checks equal the stored ones;
  f  s2_main: 18 policies, seed-major; finished work skipped; a failing step tried twice, then reported; two
     workers finish a fake queue; train_policy maps REL__<variant> to the released weights with no holdout and
     refuses unknown variants;
  g  until prereg/PREREG_S2.md is committed, evaluate.py refuses a main-pool policy;
  h  review fixes: NOT COMPUTABLE when a check is all tied, when the outcome has fewer than 3 distinct values, or
     when more than 10% of draws are undefined; selection ties go to the lower pool position, duplicates count as
     entries; a policy failed by its simulator scores falls 1 and the lowest measured return; a simulator with no
     imagined return leaves those analyses; s2_main writes a simulator failure only for non-finite errors.
"""
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
AIR = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import common as C  # noqa: E402
import s2_main as SM  # noqa: E402
import s2_stats as SS  # noqa: E402
import stats as ST  # noqa: E402

RES = {}
FAIL = []


def check(name, ok, detail=""):
    RES[name] = {"pass": bool(ok), "detail": str(detail)}
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAIL.append(name)


def rows_from(ret, **checks):
    n = len(ret)
    rows = []
    for i in range(n):
        r = {"sim": f"S{i}", "ret": float(ret[i]), "fall": 0.5, "imag": float(i)}
        for k in SS.LOWER_IS_BETTER:
            r[k] = float(checks.get(k, np.arange(n))[i])
        rows.append(r)
    return rows


# ------------------------------------------------------------------ a
def test_a():
    L = C.label3
    ok = (L([0.1, 0.5], *SS.R0_LAB) == "SUPPORTED" and L([-0.5, -0.1], *SS.R0_LAB) == "CONTRADICTED"
          and L([-0.1, 0.2], *SS.R0_LAB) == "NOT SETTLED" and L([0.0, 0.3], *SS.R0_LAB) == "NOT SETTLED"
          and L([0.2, 0.9], *SS.R1_LAB) == "LABEL-FREE BETTER" and L([-0.9, -0.2], *SS.R1_LAB) == "ONE-STEP BETTER"
          and L([-0.3, 0.0], *SS.R1_LAB) == "NOT SETTLED"
          and L([0.01, 0.02], *SS.R2_LAB) == "LABEL-FREE BETTER"
          and L([-0.02, -0.01], *SS.R2_LAB) == "TRAINING-HORIZON BETTER" and L([-1, 1], *SS.R2_LAB) == "NOT SETTLED")
    check("a labels from constructed intervals (all labels, boundaries NOT SETTLED)", ok)


# ------------------------------------------------------------------ b
def test_b():
    n = 9
    ret = np.arange(n, dtype=float) * 2.0                     # higher index walks better
    up, down = np.arange(n, dtype=float), np.arange(n, dtype=float)[::-1].copy()
    noise = np.array([4, 1, 7, 2, 8, 0, 6, 3, 5], dtype=float)   # unrelated order
    y = SS.col(rows_from(ret), "ret")
    got = {}
    for lab, e100 in (("SUPPORTED", down), ("CONTRADICTED", up), ("NOT SETTLED", noise)):
        got[lab] = SS.rho_q(SS.better(e100), y, SS.R0_LAB)["label"]
    check("b R0 labels: SUPPORTED, CONTRADICTED, NOT SETTLED", all(k == v for k, v in got.items()), got)
    got = {}
    for lab, lam, e1 in (("LABEL-FREE BETTER", down, up), ("ONE-STEP BETTER", up, down), ("NOT SETTLED", down, down)):
        rows = rows_from(ret, lam1=lam, E1=e1, rJ=lam, E8=e1)
        fam = SS.family(rows, y)
        got[lab] = (fam["R1"]["label"], fam["R2"]["label"], fam["R1"]["ci97.5"])
    want = {"LABEL-FREE BETTER": ("LABEL-FREE BETTER", "LABEL-FREE BETTER"),
            "ONE-STEP BETTER": ("ONE-STEP BETTER", "TRAINING-HORIZON BETTER"),
            "NOT SETTLED": ("NOT SETTLED", "NOT SETTLED")}
    check("b R1/R2 labels from the bootstrap: LABEL-FREE BETTER, ONE-STEP/TRAINING-HORIZON BETTER, NOT SETTLED",
          all(got[k][:2] == want[k] for k in want), {k: v[:2] for k, v in got.items()})
    x = np.array([3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0, 5.5])
    check("b rho(-x, y) through ranks equals -rho(x, y) for finite x",
          abs(ST.spearman(SS.better(x), ret) + ST.spearman(x, ret)) < 1e-12)
    r = SS.diff_q(SS.better(down), SS.better(up), y, SS.R1_LAB, SS.PRIMARY)
    check("b boot_diff: 20,000 draws, both levels kept",
          np.allclose(r["ci95"], [2.0, 2.0]) and np.allclose(r["ci97.5"], [2.0, 2.0]) and ST.B == 20000,
          r["ci97.5"])


# ------------------------------------------------------------------ c
def test_c():
    b = SS.better([1.0, np.inf, 2.0, np.nan])
    bi = SS.better([1.0, np.nan, 2.0], lower_is_better=False)
    check("c non-finite errors rank worst; non-finite imagined return ranks worst",
          b[1] == b.min() == b[3] and b[0] > b[2] and bi[1] < bi[0] < bi[2], (b.tolist(), bi.tolist()))
    r = SS.rho_q(SS.better(np.arange(9.0)), np.ones(9), SS.R0_LAB)
    d = SS.diff_q(SS.better(np.arange(9.0)), SS.better(np.arange(9.0)), np.ones(9), SS.R1_LAB, SS.PRIMARY)
    check("c constant outcome -> NOT COMPUTABLE", r["label"] == d["label"] == SS.NA)


# ------------------------------------------------------------------ d
def fake_results(root, drop=(), failed=(), no_imag=()):
    os.makedirs(os.path.join(root, "s2", "train"))
    os.makedirs(os.path.join(root, "s2", "eval"))
    os.makedirs(os.path.join(root, "s2", "failed"))
    n = len(SS.POOL)
    models = {}
    for i, sim in enumerate(SS.POOL):
        models[sim] = {"E1": 0.1 + 0.01 * ((i * 4) % n), "E8": 0.2 + 0.01 * ((i * 4) % n), "E100": 1.0 + 0.1 * i,
                       "lam1": 0.5 + 0.01 * i, "rJ": 3.0 + 0.1 * ((i * 2) % n),
                       "lam1_pol": 1.0 + 0.01 * i, "rJ_pol": np.inf if i == 5 else 1.0 + 0.1 * i}
    json.dump({"models": models}, open(os.path.join(root, "s2", "checks.json"), "w"))
    rng = np.random.default_rng(5)
    for i, sim in enumerate(SS.POOL):
        for s in SS.SEEDS:
            p = SS.policy(sim, s)
            if p in drop:
                continue
            if p in failed:
                json.dump({"policy": p, "cause": "simulator"}, open(os.path.join(root, "s2", "failed", p + ".json"), "w"))
                continue
            ret = 30.0 - 3.0 * i + rng.normal(0, 0.5)
            json.dump({"summary": {"ret": ret, "fall_rate": min(1.0, 0.1 * i), "speed_err": 0.1, "turn_err": 0.1}},
                      open(os.path.join(root, "s2", "eval", p + ".json"), "w"))
            if sim not in no_imag:
                json.dump({"imagined_return": 5.0 - 0.3 * i},
                          open(os.path.join(root, "s2", "train", p + ".json"), "w"))


def test_d():
    root = tempfile.mkdtemp(prefix="s2_m4_")
    try:
        drop = (SS.policy("REL__u10", 2), SS.policy("REL__s128", 1), SS.policy("REL__s128", 2))
        fake_results(root, drop, failed=(SS.policy("REL__u50", 1),), no_imag=("REL__u20",))
        t0 = time.time()
        o = SS.main(res=root, check_hash=False)
        sec = time.time() - t0
        rows = {r["sim"]: r for r in o["descriptive"]["rows"]}
        check("d missing policies: one seed -> the other used; both -> simulator dropped and listed",
              rows["REL__u10"]["n_policies"] == 1 and "REL__s128" not in rows and o["n_simulators"] == 8
              and sorted(o["missing_policies"]) == sorted(drop), o["missing_policies"])
        P = o["premise_and_primary"]
        check("d synthetic pool: return falls with E100 and lam1 -> R0 SUPPORTED; E1 scrambled -> R1 not ONE-STEP",
              P["R0"]["label"] == "SUPPORTED" and P["R1"]["label"] != "ONE-STEP BETTER" and P["R1"]["delta"] > 0,
              (P["R0"]["label"], P["R1"]["label"], P["R2"]["label"]))
        sel = o["secondary"]["selection"]
        check("d selection: a perfect check (lam1, E100) has shortfall 0; chosen are the three best",
              sel["checks"]["lam1"]["shortfall"] == 0.0 and sel["checks"]["E100"]["shortfall"] == 0.0
              and sel["checks"]["lam1"]["chosen"] == ["REL", "REL__fp16", "REL__int8w"]
              and sel["checks"]["E1"]["shortfall"] > 0, sel["checks"]["lam1"])
        check("d rollout-trained subset NOT APPLICABLE (one member)",
              o["secondary"]["rollout_subset"]["label"] == "NOT APPLICABLE")
        check("d infinite rJ_pol handled (policy-command family computed)",
              o["secondary"]["policy_commands"]["R2"]["label"] != SS.NA)
        check("d two-level exploratory intervals computed",
              len(o["exploratory"]["two_level"]["R1"]["ci97.5"]) == 2, o["exploratory"]["two_level"]["dropped_draws"])
        u50 = rows["REL__u50"]["per_policy"]
        lowest = min(q["ret"] for r in rows.values() for q in r["per_policy"] if not q["failed"])
        check("d failed by its simulator: falls 1 and the lowest measured return; listed",
              u50[0]["failed"] and u50[0]["fall"] == 1.0 and u50[0]["ret"] == lowest
              and o["failed_policies"] == [SS.policy("REL__u50", 1)], u50)
        check("d no imagined return: simulator left out of the imagined-return analyses, listed",
              o["imagined_return_left_out"] == ["REL__u20"] and o["secondary"]["imagined_return"]["n_models"] == 7
              and "imag" in o["secondary"]["selection"]["left_out"], o["imagined_return_left_out"])
        check("d report and json written", os.path.exists(os.path.join(root, "s2_stats.txt"))
              and json.load(open(os.path.join(root, "s2_stats.json")))["n_simulators"] == 8, f"{sec:.0f}s")
        try:
            SS.main(res=root, check_hash=False)
            refused = False
        except SystemExit:
            refused = True
        check("d result files never overwritten", refused)
        try:
            SS.load(root, check_hash=True)
            hashed = False
        except SystemExit:
            hashed = True
        check("d a checks file other than the registered one is refused", hashed)
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------------------------ e
def test_e():
    import fallback_checks as FC
    path = os.path.join(C.RESULTS, "s2", "checks.json")
    d = json.load(open(path))
    check("e pool = fallback_checks.SIMS (the released model and its eight compressed copies)",
          SS.POOL == FC.SIMS and len(SS.POOL) == 9 and set(SS.POOL) <= set(d["models"]))
    check("e checks.json matches the registered hash", C.sha256(path) == SS.CHECKS_SHA256)
    v = d["validation_REL_vs_stored"]
    tight = all(v[k]["rel_diff"] < 1e-5 for k in ("E1", "E8", "E24", "E100", "lam1", "lam1_pol", "rJ_pol"))
    check("e released model: recomputed E1/E8/E24/E100/lam1 and policy-command checks equal the stored values "
          "(rel 1e-5); r_J within 1%", tight and v["rJ"]["rel_diff"] < 0.01,
          {k: f"{x['rel_diff']:.1e}" for k, x in v.items()})


# ------------------------------------------------------------------ f
def test_f():
    p = SM.plan()
    check("f 18 policies, seeds 1 and 2, seed-major", len(p) == 18 and len({x[2] for x in p}) == 18
          and [x[1] for x in p[:9]] == [1] * 9 and p[0][2] == "REL__p1", p[:2])
    tmp = tempfile.mkdtemp(prefix="s2_m4_main_")
    saved = (SM.TRAIN, SM.EVAL, SM.PROGRESS, SM.STATUS, SM.LOGS, SM.FAILED, SM.FAILED_OTHER)
    try:
        SM.TRAIN, SM.EVAL = os.path.join(tmp, "train"), os.path.join(tmp, "eval")
        SM.PROGRESS, SM.STATUS, SM.LOGS = (os.path.join(tmp, "progress.json"), os.path.join(tmp, "status.json"),
                                           os.path.join(tmp, "logs"))
        SM.FAILED, SM.FAILED_OTHER = os.path.join(tmp, "failed"), os.path.join(tmp, "failed_other")
        os.makedirs(SM.TRAIN)
        os.makedirs(SM.EVAL)
        calls = []

        def ok_runner(cmd, log, threads):
            calls.append(cmd)
            pol = (f"{cmd[cmd.index('--sim') + 1]}__p{cmd[cmd.index('--seed') + 1]}" if "--sim" in cmd
                   else cmd[cmd.index("--policy") + 1])
            d = SM.TRAIN if "--sim" in cmd else SM.EVAL
            json.dump({}, open(os.path.join(d, pol + ".json"), "w"))
            return 0
        items = p[:4]
        json.dump({}, open(os.path.join(SM.EVAL, items[0][2] + ".json"), "w"))         # already measured
        json.dump({}, open(os.path.join(SM.TRAIN, items[1][2] + ".json"), "w"))        # trained, not measured
        st = SM.run_all(2, 1, ok_runner, items)
        n_train = sum("--sim" in c for c in calls)
        n_eval = sum("--policy" in c for c in calls)
        check("f resume: measured skipped, trained only measured; two workers finish the queue",
              all(v == "measured" for v in st.values()) and n_train == 2 and n_eval == 3
              and all("--route" in c and c[c.index("--route") + 1] == "rebuilt" for c in calls if "--sim" in c),
              (n_train, n_eval))
        fails = []

        def bad_runner(cmd, log, threads):
            fails.append(cmd)
            return 1
        st = SM.run_all(1, 1, bad_runner, p[4:5])
        check("f a failing step is tried twice, then reported (no result)",
              len(fails) == 2 and st[p[4][2]] == "failed at train"
              and not os.path.exists(os.path.join(SM.FAILED, p[4][2] + ".json"))
              and os.path.exists(os.path.join(SM.FAILED_OTHER, p[4][2] + ".json")), st)

        def nan_runner(cmd, log, threads):
            with open(log, "a") as fh:
                fh.write("Mean reward: nan\nTraceback (most recent call last):\nValueError: actions are not finite\n")
            return 1
        st = SM.run_all(1, 1, nan_runner, p[5:6])
        rec = json.load(open(os.path.join(SM.FAILED, p[5][2] + ".json")))
        again = SM.run_all(1, 1, bad_runner, p[5:6])
        check("f non-finite error twice -> failed by its simulator, recorded, not retried",
              st[p[5][2]] == "failed: simulator" and rec["cause"] == "simulator"
              and again[p[5][2]] == "failed: simulator" and len(fails) == 2, st)

        def code_runner(cmd, log, threads):
            with open(log, "a") as fh:
                fh.write("Mean reward: nan\nTraceback (most recent call last):\nKeyError: 'ret'\n")
            return 1
        st = SM.run_all(1, 1, code_runner, p[6:7])
        check("f a code error is not a simulator failure, even with nan earlier in the log",
              st[p[6][2]] == "failed at train", st)
    finally:
        SM.TRAIN, SM.EVAL, SM.PROGRESS, SM.STATUS, SM.LOGS, SM.FAILED, SM.FAILED_OTHER = saved
        shutil.rmtree(tmp, ignore_errors=True)
    sys.path.insert(0, os.path.join(SCRIPTS, "lite"))
    import train_policy as TP
    row = TP.sim_row("REL__u50")
    ok = (row["variant"] == "u50" and row["arena"] == "all" and TP.HOLDOUT_OF_ARENA[str(row["arena"])] == ""
          and row["path"].endswith("pretrain_rnn_ens.pt") and int(row["ensemble"]) == 5
          and row["variant_info"]["kind"] == "unstructured")
    try:
        TP.sim_row("REL__u99")
        refused = False
    except SystemExit:
        refused = True
    check("f train_policy: REL__u50 = released weights, five networks, no holdout; unknown variant refused",
          ok and refused)


# ------------------------------------------------------------------ g
def test_g():
    import evaluate as EV
    if EV.registered():
        check("g evaluate guard (skipped: PREREG_S2.md is already committed)", True, "skipped")
        return
    try:
        EV.evaluate("REL__p1")
        refused = False
    except SystemExit as e:
        refused = "PREREG_S2" in str(e)
    check("g before registration, evaluate.py refuses a main-pool policy", refused)


def test_h():
    n = 9
    ret = np.arange(n, dtype=float)
    r = SS.rho_q(SS.better(np.ones(n)), ret, SS.R0_LAB)
    d = SS.diff_q(SS.better(np.ones(n)), SS.better(ret), ret, SS.R1_LAB, SS.PRIMARY)
    check("h a check with all values tied -> NOT COMPUTABLE", r["label"] == d["label"] == SS.NA, (r, d["label"]))
    two = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=float)
    check("h an outcome with 2 distinct values -> NOT COMPUTABLE",
          SS.rho_q(SS.better(ret), two, SS.R0_LAB)["label"] == SS.NA)
    eight = np.array([0, 0, 0, 0, 0, 0, 0, 1, 2], dtype=float)       # 3 distinct; many draws tied in y
    r = SS.rho_q(SS.better(-ret), eight, SS.R0_LAB)
    check("h more than 10% undefined draws -> NOT COMPUTABLE (dropped draws counted)",
          r["label"] == SS.NA and r["dropped_draws"] > 0.1 * SS.B, r.get("dropped_draws"))
    r = SS.rho_q(SS.better(-ret), ret, SS.R0_LAB)
    check("h dropped draws reported for a computable rho", r["dropped_draws"] >= 0 and r["label"] == "SUPPORTED",
          r["dropped_draws"])
    score = np.array([1.0, 5.0, 5.0, 5.0, 5.0])
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    m1 = SS.top_mean(score, y, np.array([0, 4, 3, 2, 1]))            # ties: lower pool position first
    m2 = SS.top_mean(np.array([5.0, 5.0, 1.0]), np.array([7.0, 7.0, 0.0]), np.array([2, 2, 0]), k=2)
    check("h selection: ties to the lower pool position; a simulator drawn twice is two entries",
          m1 == np.mean([4.0, 3.0, 2.0]) and m2 == 7.0, (m1, m2))


def main():
    t0 = time.time()
    for t in (test_a, test_b, test_c, test_d, test_e, test_f, test_g, test_h):
        t()
    out = {"tests": RES, "seconds": time.time() - t0, "all_pass": not FAIL, "time": time.strftime("%FT%T")}
    os.makedirs(os.path.join(AIR, ".cache"), exist_ok=True)
    json.dump(out, open(os.path.join(AIR, ".cache", "s2_m4_tests.json"), "w"), indent=1)
    print("ALL PASS" if not FAIL else f"FAILED: {FAIL}", f"({time.time() - t0:.0f}s)")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

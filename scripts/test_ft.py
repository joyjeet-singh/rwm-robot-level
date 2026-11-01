"""Tests for study FT's code (REVISION_PLAN.md R3 step 4). Saves nothing outside .cache/.

  ./py scripts/test_ft.py

(i)   three iterations on fold 1-8, seed 0, level 0.30 keep every pruned entry at zero (and C-FT1's check on
      the weights saved to .cache and reloaded agrees)
(ii)  two three-iteration runs give identical weights
(iii) the settings assertion passes for all 15 parents (no training)
(iv)  ft_eval's scoring of one parent and its U30 reproduces raw_X5.npz's E100 within 1e-3 relative (C-FT4's code)
(v)   ft_stats's K estimator on the stored X5 values of U30, seeds 0-4, reproduces stats.json's D2 u30 to 1e-12
(vi)  ft_stats's rules give the right labels on constructed intervals above, below and straddling each threshold
(vii) ft_stats end to end on synthetic scores written to .cache/test_ft: verdicts, refusal, exclusion, C-FT4
      failures kept, the U50 arms, NO SKILL TO LOSE, NOT COMPUTABLE
"""
import json
import os
import time

import numpy as np
import torch

import common as C
import ft_eval as FE
import ft_stats as FS
import ft_unstructured as F

T0 = time.time()
RES = {}


def ok(name, cond, detail=""):
    RES[name] = bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name} {detail} ({time.time() - T0:.0f} s)", flush=True)


def three(fold=(1, 8), seed=0, level=0.30):
    rid, cfg, ds, model, pinfo, mk, opt, gen, _, _ = F.setup(fold, seed, level)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    hist, _ = F.train(model, mk, opt, ds, gen, cfg, 3, verbose=False)
    return model, mk, pinfo, before, hist


def test_i_ii():
    m1, mk, pinfo, before, hist = three()
    chk = F.mask_check(m1, mk, pinfo)
    moved = any(not torch.equal(before[k], v) for k, v in m1.state_dict().items())
    wp = os.path.join(C.AIR, ".cache", "test_ft", "weights_3.pt")         # C-FT1's check on reloaded weights
    os.makedirs(os.path.dirname(wp), exist_ok=True)
    torch.save({"model_state_dict": m1.state_dict()}, wp)
    reload = FE.check_ft1(0.30, (1, 8), 0, C.load(wp, 1))
    ok("(i) mask held after 3 iterations", chk["passed"] and reload["passed"] and moved
       and all(np.isfinite(hist["total"])), f"zeros {chk['mask_zeros']} = pruned {chk['pruned_entries']}")
    m2, _, _, _, _ = three()
    s1, s2 = m1.state_dict(), m2.state_dict()
    ok("(ii) two runs identical", s1.keys() == s2.keys() and all(torch.equal(s1[k], s2[k]) for k in s1))


def test_iii():
    bad = []
    for f in C.FOLDS:
        for s in F.SEEDS:
            try:
                F.setup(f, s, 0.30)
            except AssertionError as e:
                bad.append((F.run_id(f, s), str(e)[:200]))
    ok("(iii) settings equal the source record, 15 parents", not bad, str(bad))


def test_iv():
    raw = {}
    for key, arm, f, s, build in FE.entries(["P", "U30"]):
        if f == (1, 8) and s == 0:
            m, _, _ = build()
            raw[f"E100_{key}"] = FE.score_model(m, f)["E100"]
    res = FE.ft4(raw, np.load(os.path.join(C.RESULTS, "raw_X5.npz")))
    ok("(iv) E100 of P and U30 (1-8, seed 0) vs raw_X5", len(res) == 2 and all(v["passed"] for v in res.values()),
       json.dumps({k: v["max_rel_per_trajectory"] for k, v in res.items()}))


def test_v():
    raw = dict(np.load(os.path.join(C.RESULTS, "raw_X5.npz")))
    D2 = json.load(open(os.path.join(C.RESULTS, "stats.json")))["D2_skill_retained_X5"]["u30"]
    par = {f: [f"A_{f}_s{s}" for s in range(5)] for f in FS.FOLDS}
    var = {f: [f"A_{f}_s{s}__u30" for s in range(5)] for f in FS.FOLDS}
    td = FS.draws()
    worst = 0.0
    for h in FS.HS:
        Kf, Kd, _ = FS.k_skill(FS.arm_traj(raw, var, h), FS.arm_traj(raw, par, h), FS.floor_traj(raw, h), td)
        ref = D2[f"h{h}"]
        worst = max(worst, abs(float(Kf) - ref["K"]), *(abs(a - b) for a, b in zip(FS.ci(Kd), ref["ci95"])))
    ok("(v) K estimator reproduces D2 u30 (K and ci95, 5 horizons)", worst <= 1e-12, f"max |diff| {worst:.2e}")


def test_vi():
    c = []
    r1 = {(0.01, 0.2): "FINE-TUNING RESTORES SKILL", (-0.2, -0.01): "FINE-TUNING REMOVES SKILL",
          (-0.1, 0.1): "NOT SETTLED", (0.0, 0.2): "NOT SETTLED", (-0.2, 0.0): "NOT SETTLED"}
    c += [FS.rule_ft1(k) == v for k, v in r1.items()]
    r2 = {(0.91, 1.1): "NEARLY AS SKILLED", (0.90, 1.1): "KEEPS MOST", (0.89, 0.95): "KEEPS MOST",
          (0.51, 0.8): "KEEPS MOST", (0.50, 0.8): "NOT SETTLED", (0.3, 0.6): "NOT SETTLED",
          (0.2, 0.49): "LOSES MOST", (0.2, 0.50): "NOT SETTLED", (-1.0, -0.5): "LOSES MOST"}
    c += [FS.rule_ft2(k, 0.1) == v for k, v in r2.items()]
    c += [FS.rule_ft2((0.95, 1.0), 0.0) == "NO SKILL TO LOSE", FS.rule_ft2((0.95, 1.0), -0.01) == "NO SKILL TO LOSE"]
    c += [FS.rule_onestep((0.1, 0.3)) == "ONE-STEP READING OVERSTATES",
          FS.rule_onestep((-0.3, -0.1)) == "ONE-STEP READING UNDERSTATES",
          FS.rule_onestep((-0.1, 0.1)) == "NOT SETTLED",
          FS.rule_change((0.1, 0.3)) == "INCREASES", FS.rule_change((-0.3, -0.1)) == "DECREASES",
          FS.rule_change((-0.1, 0.1)) == "NOT SETTLED"]
    rng = np.random.default_rng(1)
    ref, flo = rng.random(20), rng.random(20) + 2.0
    td = FS.draws()
    c += [abs(FS.k_skill(ref, ref, flo, td)[0] - 1) < 1e-12, abs(FS.k_skill(flo, ref, flo, td)[0]) < 1e-12,
          td.shape == (20000, 20) and td.min() == 0 and td.max() == 19]
    for n in (7, 20, 20000):                       # float64: percentile() equals numpy's own to 1e-12
        v = rng.normal(size=n)
        c += [all(abs(FS.percentile(v, q) - float(np.percentile(v, q))) <= 1e-12 for q in (0, 1.25, 2.5, 50, 97.5, 98.75, 100))]
    w = np.array([1.0, 2.0, 3.0, 5.0])
    c += [np.allclose(FS.ci(w, 95.0), C.ci(w), rtol=0, atol=1e-12)]
    try:                                             # a NaN draw is refused, never dropped
        FS.ci(np.array([1.0, np.nan, 2.0]))
        c.append(False)
    except AssertionError:
        c.append(True)
    # undefined draws (PerconAI PREREG_R2.md 2.1): a reference without skill on a draw gives K = -inf there
    x, r = np.zeros(20), np.zeros(20)
    fl = np.full(20, 0.1)
    fl[0] = -100.0                                   # the draws that take trajectory 0 often have skill <= 0
    Kf, Kd, sk = FS.k_skill(x, r, fl, td)
    nu = FS.undefined(Kd)
    c += [sk < 0 or nu > 0, 0 < nu < 20000, np.isneginf(Kd[(fl - r)[td].mean(1) <= 0]).all()]
    many = np.where(np.arange(20000) < 1000, -np.inf, 1.0)          # 5% undefined: lower 2.5% endpoint is -inf
    c += [FS.ci(many)[0] == -np.inf, FS.ci(many)[1] == 1.0]
    lo, hi, n = FS.d_draws(np.where(np.arange(20000) < 300, -np.inf, 0.5), np.zeros(20000))
    c += [n == 300, FS.ci_d(lo, hi) == [0.5, 0.5], FS.ci_d(lo, hi, 99.0)[0] == -np.inf,
          FS.ci_d(lo, hi, 99.0)[1] == np.inf, FS.rule_ft1(FS.ci_d(lo, hi, 99.0)) == "NOT SETTLED"]
    ok("(vi) labels on constructed inputs", all(c), f"{sum(c)}/{len(c)}")


def synth(arms, offs, rng):
    raw, models = {}, {}
    chk = {c: {} for c in ("C-FT1", "C-FT2", "C-FT3", "C-FT4")}
    base = {f: rng.random((4, 368)).astype(np.float32) + 0.2 for f in FS.FOLDS}
    for f in FS.FOLDS:
        raw[f"floor_{f}"] = base[f] + 1.0
        for h in (1, 8, 24, 100):
            raw[f"floor_E{h}_{f}"] = raw[f"floor_{f}"][:, :h].mean(1)
        for a, off in zip(arms, offs):
            for s in FS.SEEDS:
                k = f"{a}_{f}_s{s}"
                e = base[f] + off + 0.01 * rng.random((4, 368)).astype(np.float32)
                raw[f"err_{k}"] = e
                for h in (1, 8, 24, 100):
                    raw[f"E{h}_{k}"] = e[:, :h].mean(1)
                raw[f"lam_{k}"] = rng.normal(size=(4, 4))
                raw[f"rJ_{k}"] = rng.random((4, 368))
                models[k] = {"arm": a}
                if a in FS.FT_ARMS:
                    for c in ("C-FT1", "C-FT2", "C-FT3"):
                        chk[c][k] = {"passed": True}
                if a in ("P", "U30"):
                    chk["C-FT4"][k] = {"passed": True}
    return raw, {"arms": list(arms), "models": models, "checks": chk}


def test_vii():
    """End to end on synthetic scores in .cache/test_ft: verdicts, refusal on a missing check or score, exclusion
    of an arm whose check failed (with or without its scores), C-FT4 failures kept, the U50 arms, NO SKILL TO LOSE,
    and FT1's NOT COMPUTABLE."""
    d = os.path.join(C.AIR, ".cache", "test_ft")
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(2)

    def put(raw, meta):
        np.savez(os.path.join(d, "raw_FT.npz"), **raw)
        json.dump({"arms": meta["arms"]}, open(os.path.join(d, "ft_probe.json"), "w"))
        json.dump(meta, open(os.path.join(d, "ft_models.json"), "w"))

    def go():
        FS.main()
        return json.load(open(os.path.join(d, "ft_stats.json")))

    def refused():
        try:
            FS.main()
            return False
        except SystemExit as e:
            return str(e).startswith("REFUSED")

    old = C.RESULTS
    C.RESULTS = d
    c = []
    try:
        arms = ["P", "U30", "U30F", "CTRL", "U50", "U50F"]
        raw, meta = synth(arms, (0.0, 0.5, 0.1, -0.02, 0.9, 0.6), rng)
        put(raw, meta)
        o = go()
        c += [o["primary"]["FT1"]["verdict"] == "FINE-TUNING RESTORES SKILL",
              o["primary"]["FT2"]["verdict"] in ("KEEPS MOST", "NEARLY AS SKILLED"),
              o["secondary"]["FT1_u50_h100"]["verdict"] == "FINE-TUNING RESTORES SKILL",
              o["secondary"]["FT2_u50_h100"]["verdict"] == "LOSES MOST",
              o["primary"]["FT1"]["undefined_draws"] == 0]
        del meta["checks"]["C-FT2"]["U30F_18_s1"]                        # a missing check: refuse
        put(raw, meta)
        c.append(refused())
        meta["checks"]["C-FT2"]["U30F_18_s1"] = {"passed": True}
        r2 = dict(raw)
        del r2["E24_U30_34_s2"]                                         # a missing score: refuse
        put(r2, meta)
        c.append(refused())
        meta["checks"]["C-FT1"]["U30F_02_s2"] = {"passed": False}       # a failed check: exclude U30F
        meta["checks"]["C-FT4"]["P_56_s0"] = {"passed": False}          # C-FT4 failure: reported, scores kept
        put(raw, meta)
        o = go()
        c += [o["primary"]["FT1"]["verdict"].startswith("EXCLUDED"), o["primary"]["FT2"]["verdict"].startswith("EXCLUDED"),
              o["secondary"]["onestep_U30F"]["verdict"].startswith("EXCLUDED"),
              "mean" in o["secondary"]["change_lam1_U30_minus_P"], o["C-FT4_failures"] == ["P_56_s0"],
              o["secondary"]["FT1_u50_h100"]["verdict"] == "FINE-TUNING RESTORES SKILL"]
        meta["checks"]["C-FT1"]["U30F_02_s2"] = {"passed": True}
        meta["checks"]["C-FT3"]["U50F_79_s0"] = {"passed": False}       # C-FT3 failed, never scored: excluded
        del meta["models"]["U50F_79_s0"]
        r3 = {k: v for k, v in raw.items() if not k.endswith("_U50F_79_s0")}
        put(r3, meta)
        o = go()
        c += [o["secondary"]["FT1_u50_h100"]["verdict"].startswith("EXCLUDED"),
              o["primary"]["FT1"]["verdict"] == "FINE-TUNING RESTORES SKILL"]
        # the control has no skill over the floor: FT2 says NO SKILL TO LOSE and reports the difference
        raw, meta = synth(arms[:4], (0.0, 0.5, 0.1, 1.5), rng)
        put(raw, meta)
        o = go()
        c += [o["primary"]["FT2"]["verdict"] == "NO SKILL TO LOSE", "delta_ctrl" in o["primary"]["FT2"],
              o["secondary"]["FT1_u50_h100"]["verdict"] == "NOT RUN"]
        # P has no skill over the floor: FT1 and the one-step reading are NOT COMPUTABLE
        raw, meta = synth(arms[:4], (1.5, 2.0, 1.6, 0.0), rng)
        put(raw, meta)
        o = go()
        c += [o["primary"]["FT1"]["verdict"].startswith("NOT COMPUTABLE"),
              o["secondary"]["onestep_U30F"]["verdict"].startswith("NOT COMPUTABLE")]
    finally:
        C.RESULTS = old
    ok("(vii) ft_stats end to end on synthetic scores (.cache)", all(c), f"{sum(c)}/{len(c)} {c}")


if __name__ == "__main__":
    torch.set_num_threads(2)
    test_vi()
    test_vii()
    test_v()
    test_i_ii()
    test_iii()
    test_iv()
    print(f"{sum(RES.values())}/{len(RES)} passed in {time.time() - T0:.0f} s")
    raise SystemExit(0 if all(RES.values()) else 1)

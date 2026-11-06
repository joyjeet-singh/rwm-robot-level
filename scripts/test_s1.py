"""
Tests of the Step 1 code (STEP1_PLAN.md, S1, S2 and S3). Saves nothing outside .cache/. Uses only random-weight models,
constructed systems and the stored registered values; no trained simulator is driven by the policy or by noise.
(vi) compares the trained baselines' one-step predictions with the reproduction's own code (predictions only, no
error); (ix) trains each class for 50 iterations into .cache/ and compares training losses only. (x) and (xi)
test the registered labels and s1_stats on constructed values only.

  ./py scripts/test_s1.py            all tests
  ./py scripts/test_s1.py ii iv      some
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import common as C
import compute
import lyap
import s1_policy as SP
import s1_rollout as S

torch.set_num_threads(2)
RESULTS = {}


def check(name, ok, detail):
    RESULTS[name] = (bool(ok), detail)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


# ------------------------------------------------------------------ (i) the actor and observation vs task1c
def r2(pred, true):
    ss_res = ((true - pred) ** 2).sum(0)
    ss_tot = ((true - true.mean(0)) ** 2).sum(0)
    return float((1.0 - ss_res / ss_tot).mean())


def test_i():
    ref = json.load(open(SP.TASK1C_JSON))
    cfg, dat, ep = C.data()
    actor = SP.load_actor()
    cmd = SP.command_per_row()
    n = len(dat)
    t = np.arange(n)
    ok = np.isin(ep, ref["test_episodes"]) & (ep >= 0) & (t - 1 >= 0)
    ok &= ep[np.clip(t - 1, 0, n - 1)] == ep
    t = t[ok]
    true_a = dat[t][:, C.R.ACTION_COLS]
    for k in (-1, 0):
        rows = t + k
        s = torch.as_tensor(C.R.normalise_state(dat[rows][:, C.R.STATE_COLS], cfg["state_data_mean"],
                                                cfg["state_data_std"]), dtype=torch.float32)
        a_prev = SP.norm_action(torch.as_tensor(dat[t - 1][:, C.R.ACTION_COLS], dtype=torch.float32))
        c = torch.as_tensor(cmd[rows], dtype=torch.float32)
        with torch.no_grad():
            pol = SP.Policy(actor, c[:, None])                 # cmd indexed [:, r]; one "row" per sample
            pred = SP.denorm_action(pol(s, a_prev, 0)).numpy()
        v, want = r2(pred, true_a), ref["by_offset"][str(k)]["mean_r2"]
        check(f"i  mean R2 at offset {k:+d}", abs(v - want) < 1e-4, f"{v:.6f} vs task1c {want:.6f}")
    # our differentiable observation equals the authors' get_imagination_observation
    env = SP.stand_in_env(len(t))
    o1 = SP.observation(s, c, a_prev)
    o2 = SP.authors_observation(env, s, c, a_prev)
    d = float((o1 - o2).abs().max())
    check("i  observation == authors' get_imagination_observation", d == 0.0, f"max |diff| {d:.2e}")


# ------------------------------------------------------------------ constructed linear systems
class LinRNN(nn.Module):
    """A GRU stand-in whose step is linear: z = (h, s) -> A z + U a. Output carries (z', s) for the head."""

    def __init__(self, A, U, H, ds):
        super().__init__()
        self.A, self.U = nn.Parameter(torch.as_tensor(A)), nn.Parameter(torch.as_tensor(U))
        self.H, self.ds = H, ds

    def forward(self, x, h=None):
        B, T, _ = x.shape
        if h is None:
            h = torch.zeros(1, B, self.H, dtype=x.dtype)
        outs = []
        for t in range(T):
            s, a = x[:, t, :self.ds], x[:, t, self.ds:]
            z = torch.cat([h[0], s], -1) @ self.A.T + a @ self.U.T
            h = z[None, :, :self.H]
            outs.append(torch.cat([z, s], -1))
        return torch.stack(outs, 1), h


class LinHead(nn.Module):
    def __init__(self, H, ds):
        super().__init__()
        self.H, self.ds = H, ds
        self.state_mean_layers = lambda x: x[..., self.H:self.H + self.ds] - x[..., self.H + self.ds:]


def lin_model(A, U, H, ds):
    m = nn.Module()
    m.state_base = nn.Module()
    m.state_base.memory = nn.Module()
    m.state_base.memory.rnn = LinRNN(A, U, H, ds)
    m.state_heads = nn.ModuleList([LinHead(H, ds)])
    return m.double()


def spectrum(n, rho1, rho2, rng):
    """M = V (rho1 * O  (+)  rho2) V^T: every eigenvalue on S1 has modulus rho1 and S1 is orthogonal to S2."""
    O, _ = np.linalg.qr(rng.standard_normal((n - 1, n - 1)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    D = np.zeros((n, n))
    D[:n - 1, :n - 1] = rho1 * O
    D[n - 1, n - 1] = rho2
    return V @ D @ V.T


def test_ii():
    H, ds, da, B = 200, 6, 2, 2
    n = H + ds
    rng = np.random.default_rng(123)
    rho1, rho2 = np.exp(0.1), 0.5
    M = spectrum(n, rho1, rho2, rng)
    want = np.log(np.abs(np.linalg.eigvals(M)).max()) / C.DT
    st = torch.as_tensor(rng.standard_normal((B, C.LEN, ds)) * 0.01)
    ac = torch.as_tensor(rng.standard_normal((B, C.LEN, da)) * 0.01)
    U = rng.standard_normal((n, da))
    # open loop: lyap.lyapunov on z -> M z + U a
    m = lin_model(M, U, H, ds)
    lam = lyap.lyapunov(m, st, ac, p=1)[0][:, 0]
    check("ii lyap.lyapunov on a linear map", np.abs(lam - want).max() < 1e-3,
          f"{lam} vs log|eig|/dt {want:.6f}")
    # closed loop: u = K s, A = M - U K P_s, so A + BK = M
    K = rng.standard_normal((da, ds)) * 0.1
    P = np.zeros((ds, n))
    P[:, H:] = np.eye(ds)
    A = M - U @ K @ P
    m = lin_model(A, U, H, ds)
    Kt = torch.as_tensor(K)
    lam_cl = S.lyapunov_cl(m, st, ac, lambda s, a_prev, r: s @ Kt.T, p=1)[0][:, 0]
    m = lin_model(A, U, H, ds)
    lam_open = lyap.lyapunov(m, st, ac, p=1)[0][:, 0]
    want_open = np.log(np.abs(np.linalg.eigvals(A)).max()) / C.DT
    check("ii lyapunov_cl with u = K x gives A + BK's exponent", np.abs(lam_cl - want).max() < 1e-3,
          f"{lam_cl} vs {want:.6f} (open loop of A alone: {lam_open[0]:.4f}, log|eig A|/dt {want_open:.4f})")
    # exact finite-time value of the composed map on (h, s, a_prev), from lyap.lyapunov's start direction
    J = np.zeros((n + da, n + da))
    J[:n, :n] = M
    J[n:, :n] = K @ P
    q = np.linalg.qr(np.random.default_rng(0).standard_normal((n, 4)))[0][:, 0]
    q = np.concatenate([q, np.zeros(da)])
    tot = 0.0
    for _ in range(C.LEN - C.START - 1):
        q = J @ q
        tot += np.log(np.linalg.norm(q))
        q /= np.linalg.norm(q)
    exact = tot / ((C.LEN - C.START - 1) * C.DT)
    e = float(np.abs(lam_cl - exact).max())
    check("ii lyapunov_cl equals the exact finite-time value", e < 1e-9, f"{lam_cl[0]:.9f} vs {exact:.9f}")


# ------------------------------------------------------------------ (iii) stored registered values
def test_iii():
    raw = np.load(os.path.join(C.RESULTS, "raw_F18.npz"))
    meta = {m["key"]: m for m in json.load(open(os.path.join(C.RESULTS, "models.json")))["F18"]}
    ar, sc = compute.setup_arena((1, 8))
    for key in ("A_s0", "B_s0"):
        mm = meta[key]
        m = C.load(os.path.join(C.PDM, mm["path"]), mm["ensemble"])
        out = S.score_gru(m, ar, sc, items=("rec",))         # recorded commands only
        lam_s, rJ_s = raw[f"lam_{key}"][:, 0], raw[f"rJ_{key}"].mean(-1)
        e1 = np.abs(out["lam1_rec"] / lam_s - 1).max()
        e2 = np.abs(out["rJ_rec"] / rJ_s - 1).max()
        check(f"iii {key} lambda_1 and r_J equal the stored values", max(e1, e2) < 1e-4,
              f"max rel diff lambda_1 {e1:.1e}, r_J {e2:.1e} ({out['seconds']})")


# ------------------------------------------------------------------ (iv) a random GRU, a policy that replays the recording
def test_iv():
    torch.manual_seed(0)
    ar, sc = compute.setup_arena((1, 8))
    m = C.build_width(256, ensemble_size=5).eval().double()
    st, ac = ar["st"].double(), ar["ac"].double()
    seen = {"bad": 0}

    def replay(s, a_prev, r):
        # task1c offset -1: fed state row r and the command of row r, the actor stands for row r + 1
        seen["bad"] += int(not torch.equal(a_prev, ac[:, r]))
        return ac[:, r + 1]

    pred, A = S.closed_loop(m, st, ac, replay)
    ref = lyap.rollout_map(m, st, ac)
    d = float((pred - ref).abs().max())
    check("iv closed loop with a replaying policy == open-loop rollout (float64)",
          d < 1e-10 and torch.equal(A, ac) and seen["bad"] == 0,
          f"max |state diff| {d:.1e}; previous-command mismatches {seen['bad']}")
    _, _, _, sp = S.aux_outputs(m, pred, A)
    d2 = float((sp[:, C.START:] - pred[:, C.START:]).abs().max())
    check("iv one-pass auxiliary path is aligned with the step-wise rollout", d2 < 1e-10, f"max |diff| {d2:.1e}")
    lam_cl = S.lyapunov_cl(m, st, ac, lambda s, a_prev, r: ac[:, r + 1], p=1)[0][:, 0]
    lam_rec = lyap.lyapunov(m, st, ac, p=1)[0][:, 0]
    e = float(np.abs(lam_cl / lam_rec - 1).max())
    check("iv lambda1_cl with a state-independent policy == lambda_1 (same start directions)", e < 1e-8,
          f"max rel diff {e:.1e}")


# ------------------------------------------------------------------ (v) divergence
def test_v():
    x = torch.zeros(3, C.LEN, 45)
    x[0, C.START + 10, 5] = 1e3            # at the threshold: not a divergence
    x[1, C.START + 20, 7] = -1e3 - 1e-3    # step 21
    x[1, C.START + 30, 7] = float("nan")
    x[2, C.START + 4, 0] = float("inf")    # step 5
    x[2, :C.START, 0] = 1e9                # history rows are not checked
    d = S.divergence_step(x)
    check("v  divergence rule on constructed arrays", list(d) == [0, 21, 5], f"{list(d)}")
    # end to end: a random GRU whose 'policy' replays the recording, then blows up at a known row
    torch.manual_seed(0)
    ar, sc = compute.setup_arena((1, 8))
    m = C.build_width(256, ensemble_size=1).eval()
    ac = ar["ac"]
    r0 = C.START + 50                     # the command applied to row r0 -> r0 + 1 is huge for trajectory 1

    def blow(s, a_prev, r):
        a = ac[:, r + 1].clone()
        if r >= r0:
            a[1] = float("inf")
        return a

    out = S.score_gru(m, ar, sc, items=("pol", "lam_pol", "cl"), policy=blow)
    k = r0 + 1 - C.START + 1
    ok = (out["div_pol"][1] == k and out["div_pol"][[0, 2, 3]].sum() == 0 and np.isinf(out["rJ_pol"][1])
          and np.isinf(out["lam1_pol"][1]) and np.isinf(out["lam1_cl"][1]) and out["div_cl"][1] == k
          and np.isfinite(out["rJ_pol"][[0, 2, 3]]).all() and np.isfinite(out["lam1_pol"][[0, 2, 3]]).all()
          and np.isfinite(out["ret"]).all() and out["ret_diverged"][1] and not out["ret_diverged"][0])
    check("v  divergence end to end (closed loop)", ok,
          f"div_pol {out['div_pol']} (want {k} for traj 1), div_cl {out['div_cl']}, rJ_pol {out['rJ_pol']}, "
          f"lam1_pol {out['lam1_pol']}, lam1_cl {out['lam1_cl']}, ret {out['ret']}")


# ------------------------------------------------------------------ Study AR adapters (S2)
REPRO = os.path.join(C.PDM, "rwm_repro")


def existing_runs():
    """The reproduction's 26 baseline runs (S0 item d): runs/baseline_<arch>_<regime>_<spec>_seed<s>/."""
    import glob
    return sorted(d for d in glob.glob(os.path.join(REPRO, "runs", "baseline_*_seed*")) if os.path.isdir(d))


def test_vi():
    """One-step predictions on 100 recorded windows equal the reproduction's own baseline code (predictions only)."""
    import s1_arch as A
    from baselines.common import action_rows_for
    ar, _ = compute.setup_arena((1, 8))
    ks = [C.START + 14 * j for j in range(25)]                                  # 25 positions x 4 trajectories
    idx = torch.as_tensor([[k - C.START + j for j in range(C.START + 1)] for k in ks])
    sw = ar["st"][:, idx].reshape(-1, C.START + 1, 45)
    aw = ar["ac"][:, idx].reshape(-1, C.START + 1, 12)
    runs = existing_runs()
    worst, per, split_ok = 0.0, {}, True
    for rd in runs:
        name = os.path.basename(rd)
        rec = json.load(open(os.path.join(REPRO, "results", f"baseline_run_{name[len('baseline_'):]}.json")))
        split_ok &= rec["hyperparameters"]["holdout_episodes"] == [1, 8]
        m, info = A.load(rd)
        with torch.no_grad():
            if info["arch"] == "rssm":
                ref = m.rollout(sw, aw, C.START)[:, C.START]
            else:
                lo, hi = action_rows_for(C.START, m.history)
                ref = m.predict(sw[:, :C.START], aw[:, lo:hi])[0]
        mine = A.predict_one(m, sw, aw)
        r = float(((mine - ref).norm(dim=-1) / ref.norm(dim=-1)).max())
        per[info["arch"]] = max(per.get(info["arch"], 0.0), r)
        worst = max(worst, r)
    check("vi one-step predictions = reproduction's code (26 runs, 100 windows each)",
          len(runs) == 26 and split_ok and worst <= 1e-6 and len(sw) == 100,
          f"{len(runs)} runs, all held out [1, 8]: {split_ok}; max relative difference by class {per}")


def test_vii():
    """The step maps' JVPs match central finite differences; the generic code reproduces the registered GRU functions."""
    import evalh
    import s1_arch as A
    ar, _ = compute.setup_arena((1, 8))
    st, ac = ar["st"], ar["ac"]
    res = {}
    for arch, spec in (("mlp", "s7"), ("mlp", "matched"), ("transformer", "s7"), ("transformer", "matched"),
                       ("rssm", "s7"), ("rssm", "x1v1"), ("rssm", "x1v2")):
        torch.manual_seed(0)
        m = A.build(arch, spec).eval()
        res[f"{arch}_{spec}"] = A.fd_check(m, st, ac)
    check("vii JVP vs finite differences (random weights)", max(res.values()) <= 1e-4,
          ", ".join(f"{k} {v:.1e}" for k, v in res.items()))
    torch.manual_seed(0)
    g = C.build_width(256, ensemble_size=1).eval()
    d_roll = float((A.rollout(g, st, ac) - lyap.rollout_map(g, st, ac)).abs().max())
    d_win = max(float(np.max(np.abs(A.window_error(g, st, ac, h) - evalh.window_error(g, st, ac, h)[0])
                             / evalh.window_error(g, st, ac, h)[0])) for h in (1, 8, 100))
    lam_ref = lyap.lyapunov(g, st, ac, p=1)[0][:, 0]
    g.float()
    lam = A.lyapunov(g, st, ac, p=1)[0][:, 0]
    d_lam = float(np.max(np.abs(lam - lam_ref) / np.abs(lam_ref)))
    check("vii generic code on a GRU = lyap.rollout_map, evalh, lyap.lyapunov",
          d_roll == 0.0 and d_win <= 1e-6 and d_lam <= 1e-9,
          f"rollout max |diff| {d_roll:.1e}; E1/E8/E100 rel {d_win:.1e}; lam1 rel {d_lam:.1e} ({lam_ref})")


def test_viii():
    """score_arch on random weights: finite values of the expected shapes; rollouts equal the reproduction's."""
    import s1_arch as A
    ar, sc = compute.setup_arena((1, 8))
    st, ac = ar["st"], ar["ac"]
    shapes = {"E1": (4,), "E8": (4,), "E24": (4,), "E100": (4,), "E368": (4,), "lam1_rec": (4,), "rJ_rec": (4,),
              "err_steps": (4, 368), "rJ_rec_steps": (4, 368), "log1": (4, 367), "div_rec": (4,), "lam_div": (4,)}
    for arch in A.ARCHES:
        torch.manual_seed(0)
        m = A.build(arch, "s7").eval()
        with torch.no_grad():
            ref = m.rollout(st, ac, C.START)
        d = float((A.rollout(m, st, ac) - ref).abs().max() / ref.abs().max())
        out = A.score_arch(None, ar, sc, model=m)
        bad = [k for k, shp in shapes.items() if np.shape(out[k]) != shp]
        fin = all(np.isfinite(out[k]).all() for k in shapes if k not in ("div_rec", "lam_div"))
        check(f"viii score_arch {arch} (random weights)", not bad and fin and d <= 1e-6 and not out["lam_div"].any(),
              f"shapes wrong {bad}; finite {fin}; rollout vs reproduction rel {d:.1e}; "
              f"E100 {np.round(out['E100'], 3)}, lam1 {np.round(out['lam1_rec'], 3)}, div {out['div_rec']}; "
              f"{sum(out['seconds'].values()):.1f}s")


def test_ix():
    """The training copy reproduces the stored loss curves of tf_s7 seed 0 for 50 iterations (training data only)."""
    import shutil
    import s1_arch as A
    import s1_train_baseline as TB
    out_dir = os.path.join(C.AIR, ".cache", "s1_test_ix")
    shutil.rmtree(out_dir, ignore_errors=True)
    for arch in A.ARCHES:
        ref = np.array(json.load(open(os.path.join(REPRO, "results", f"baseline_run_{arch}_tf_s7_seed0.json")))
                       ["curves"]["total"][:50])
        calls = []
        t0 = time.time()
        rec = TB.train(arch, "tf", "s7", 0, iters=50, out_dir=out_dir, threads=2,
                       progress=lambda *a: calls.append(a[1]))
        c = np.array(rec["curves"]["total"])
        r = float(np.max(np.abs(c - ref) / np.abs(ref)))
        summ = json.load(open(os.path.join(out_dir, "baseline_training.json")))[f"s1_baseline_{arch}_tf_s7_seed0"]
        rd = os.path.join(out_dir, f"s1_baseline_{arch}_tf_s7_seed0")
        m = A.build(arch, "s7")
        m.load_state_dict(torch.load(A.final_weights(rd)[0], weights_only=False)["model_state_dict"], strict=True)
        check(f"ix training copy {arch} tf_s7 seed 0, 50 iterations",
              r <= 1e-4 and calls == [50] and summ["weights_finite"] and summ["threads"] == 2,
              f"max relative loss difference {r:.1e} (bitwise equal at {int((c == ref).sum())}/50 iterations); "
              f"progress calls {calls}; {time.time() - t0:.0f}s")
    torch.set_num_threads(2)


# ------------------------------------------------------------------ (x) registered labels on constructed intervals
def test_x():
    """CL1/CL2's rule (PREREG_S1 item 6) and Q1/Q2's labels on constructed intervals."""
    import s1_stats as SS
    cases = [([-0.05, 0.05], "UNCHANGED BY THE COMMAND SOURCE"),
             ([-0.10, 0.10], "UNCHANGED BY THE COMMAND SOURCE"),
             ([0.0, 0.05], "UNCHANGED BY THE COMMAND SOURCE"),
             ([0.01, 0.09], "UNCHANGED BY THE COMMAND SOURCE (small increase)"),
             ([-0.09, -0.01], "UNCHANGED BY THE COMMAND SOURCE (small decrease)"),
             ([-0.30, -0.05], "WORSE WITH POLICY COMMANDS"),
             ([-0.11, -0.01], "WORSE WITH POLICY COMMANDS"),
             ([0.05, 0.30], "BETTER WITH POLICY COMMANDS"),
             ([0.02, 0.101], "BETTER WITH POLICY COMMANDS"),
             ([-0.20, 0.20], "NOT SETTLED"),
             ([-0.05, 0.15], "NOT SETTLED"),
             ([float("nan"), 0.1], "NOT SETTLED")]
    got = [(ci, SS.cl_label(ci), want) for ci, want in cases]
    bad = [(ci, g, w) for ci, g, w in got if g != w]
    noisy = SS.cl_label([-0.4, -0.2], source="NOISY") == "WORSE WITH NOISY COMMANDS"
    q = (C.label3([0.1, 0.3], *SS.Q1_LAB) == "LABEL-FREE BETTER" and
         C.label3([-0.3, -0.1], *SS.Q2_LAB) == "TRAINING-HORIZON BETTER" and
         C.label3([-0.1, 0.1], *SS.Q1_LAB) == "NOT SETTLED")
    check("x rule labels on constructed intervals (inside, inside and excluding 0, below, above, straddling)",
          not bad and noisy and q, f"{len(cases)} CL cases, wrong: {bad}; NOISY wording {noisy}; Q labels {q}")


# ------------------------------------------------------------------ (xi) s1_stats on a synthetic s1_models.json
def synthetic(seed=0):
    """A constructed table and job output: in F18, rJ_pol equals rJ_rec, lam1_pol equals E100 (a perfect ranking)
    and lam1_rec is E100 plus noise; in AR, lam1_rec equals E100 within each architecture and E1 is unrelated."""
    rng = np.random.default_rng(seed)
    table = {"F18": [], "X5": []}
    db = {"dry_run": False, "checks": {k: {"pass": True} for k in ("S1-C1", "S1-C2", "S1-C3", "S1-C4", "S1-C5")},
          "models": {"F18": {}, "X5": {}, "AR": {}, "REL": {}}}

    def add(tag, key, arena, variant, group, e100):
        lam_rec = e100 + rng.normal(0, 0.3)
        rj = e100 * 2 + rng.normal(0, 0.2)
        table[tag].append({"key": key, "arena": arena, "variant": variant, "group": group, "lam1": lam_rec,
                           "rJ": rj, "E1": rng.normal(), "E8": e100 + rng.normal(0, 0.5), "E100": e100})
        div = int(rng.random() < 0.1)
        db["models"][tag][key] = {
            "cl_group": "compressed" if variant else ("one-step" if group.startswith("B") else "rollout"),
            "lam1_pol": e100, "rJ_pol": rj, "lam1_cl": e100 + rng.normal(0, 0.1), "lam1_noise": lam_rec,
            "rJ_noise": rj + rng.normal(0, 0.05), "ret": -e100 + rng.normal(0, 0.1), "ret_unc": 0.0,
            "n_div_pol": div, "n_div_cl": div, "n_div_noise": 0, "n_traj": 4}

    for i in range(30):
        add("F18", f"P{i}", "18", None, "B" if i < 6 else "A", rng.random())
    for i in range(10):
        add("F18", f"P{i}__u30", "18", "u30", "compressed u30", rng.random())
    for f in C.FOLDS:
        for i in range(12):
            add("X5", f"Q_{C.fid(f)}_{i}", C.fid(f), None, "A", rng.random())
    for a in ("mlp", "rssm", "transformer"):
        for reg in ("tf", "ar"):
            for s in range(5):
                e100 = rng.random() + (1.0 if reg == "tf" else 0.0)
                db["models"]["AR"][f"{a}_{reg}_{s}"] = {
                    "arch": a, "regime": reg, "spec": "s7", "seed": s, "new_run": s >= 3, "E1": e100 - 2.0 * (reg == "tf"),
                    "E8": rng.random(), "E100": e100, "lam1_rec": e100, "rJ_rec": rng.random(),
                    "n_div_rec": 0, "n_lam_div": 0}
    traj = {k: {} for k in ("rJ_pol", "rJ_rec", "lam1_pol", "lam1_rec", "E100")}
    for t in table["F18"]:
        s = db["models"]["F18"][t["key"]]
        for k, v in (("rJ_pol", s["rJ_pol"]), ("rJ_rec", t["rJ"]), ("lam1_pol", s["lam1_pol"]),
                     ("lam1_rec", t["lam1"]), ("E100", t["E100"])):
            traj[k][t["key"]] = v + rng.normal(0, 1e-3, 4)
    return db, table, traj


def test_xi():
    """s1_stats.analyse on a synthetic s1_models.json (2,000 draws instead of 20,000, for time)."""
    import copy
    import s1_stats as SS
    import stats as ST
    b0 = ST.B
    ST.B = 2000
    try:
        db, table, traj = synthetic()
        o = SS.analyse(db, table, traj)
        cl, ar = o["CL"], o["AR"]
        ok1 = (cl["CL1"]["delta"] == 0 and cl["CL1"]["ci97.5"] == [0.0, 0.0] and
               cl["CL1"]["verdict"] == "UNCHANGED BY THE COMMAND SOURCE")
        ok2 = cl["CL2"]["verdict"] == "BETTER WITH POLICY COMMANDS" and abs(cl["CL2"]["rho_new"] - 1.0) < 1e-12
        ok3 = ar["AR1"]["verdict"] == "LABEL-FREE BETTER" and abs(ar["AR1"]["rho_new_mean"] - 1.0) < 1e-12
        w = SS.within([[r for r in table["X5"] if r["arena"] == C.fid(f)] for f in C.FOLDS], "lam1", "E1", "E100")
        ref = ST.boot_diff_within([[r for r in table["X5"] if r["arena"] == C.fid(f)] for f in C.FOLDS],
                                  "lam1", "E1", "E100")
        ok4 = w["ci95"] == ref["ci95"] and w["ci97.5"][0] <= w["ci95"][0] and w["ci97.5"][1] >= w["ci95"][1]
        ok5 = all(ar["AR4"][a]["both"] == 5 for a in SS.ARCHES)
        n_rep = len(SS.report(o).splitlines())
        ok6 = cl["CL6"]["n_excluded_diverged"] == sum(v["n_div_pol"] > 0 for v in db["models"]["F18"].values())
        ok7 = (cl["CL9"]["trained"]["CL1"]["n_models"] == 30 and cl["CL9"]["compressed"]["CL1"]["n_models"] == 10
               and cl["exploratory_two_level"]["n_clusters"] == 30)
        g = copy.deepcopy(db)
        g["checks"]["S1-C3"]["pass"] = False
        o3 = SS.analyse(g, table)
        ok8 = (o3["AR"]["AR1"]["verdict"].startswith("NO VERDICT") and o3["AR"]["AR1"]["rule_label"] is None and
               o3["AR"]["AR3"]["pooled"]["AR1"]["rule_label"] is None and
               not o3["CL"]["CL1"]["verdict"].startswith("NO"))
        g["checks"]["S1-C3"]["pass"], g["checks"]["S1-C1"]["pass"] = True, False
        o1 = SS.analyse(g, table)
        ok9 = all(o1[f][q]["verdict"].startswith("NO VERDICT") for f, q in (("CL", "CL1"), ("CL", "CL2"),
                                                                            ("AR", "AR1"), ("AR", "AR2")))
        g["checks"]["S1-C1"]["pass"], g["checks"]["S1-C2"]["pass"] = True, False
        o2 = SS.analyse(g, table)
        ok9 &= o2["CL"]["CL1"]["verdict"].startswith("NO VERDICT") and not o2["AR"]["AR1"]["verdict"].startswith("NO")
        del g["checks"]["S1-C4"]
        try:
            SS.analyse(g, table)
            ok10 = False
        except SystemExit:
            ok10 = True
    finally:
        ST.B = b0
    check("xi s1_stats on a synthetic s1_models.json",
          all((ok1, ok2, ok3, ok4, ok5, ok6, ok7, ok8, ok9, ok10)) and n_rep <= 60,
          f"CL1 equal ranks {ok1}; CL2 perfect {ok2}; AR1 perfect {ok3}; within 97.5 from same draws {ok4}; AR4 {ok5};"
          f" CL6 exclusions {ok6}; CL9/two-level counts {ok7}; gates C3 {ok8}, C1/C2 {ok9}; refuses without a check "
          f"{ok10}; report {n_rep} lines")


TESTS = {"i": test_i, "ii": test_ii, "iii": test_iii, "iv": test_iv, "v": test_v,
         "vi": test_vi, "vii": test_vii, "viii": test_viii, "ix": test_ix,
         "x": test_x, "xi": test_xi}

if __name__ == "__main__":
    t0 = time.time()
    for name in (sys.argv[1:] or TESTS):
        t1 = time.time()
        TESTS[name]()
        print(f"   ({name}: {time.time() - t1:.0f}s)", flush=True)
    bad = [k for k, (ok, _) in RESULTS.items() if not ok]
    print(f"{len(RESULTS) - len(bad)}/{len(RESULTS)} passed in {time.time() - t0:.0f}s")
    sys.exit(1 if bad else 0)

"""
Tests of the Step 1 closed-loop code (STEP1_PLAN.md, S1). Saves nothing. Uses only random-weight models,
constructed systems and the stored registered values; no trained simulator is driven by the policy or by noise.

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


TESTS = {"i": test_i, "ii": test_ii, "iii": test_iii, "iv": test_iv, "v": test_v}

if __name__ == "__main__":
    t0 = time.time()
    for name in (sys.argv[1:] or TESTS):
        t1 = time.time()
        TESTS[name]()
        print(f"   ({name}: {time.time() - t1:.0f}s)", flush=True)
    bad = [k for k, (ok, _) in RESULTS.items() if not ok]
    print(f"{len(RESULTS) - len(bad)}/{len(RESULTS)} passed in {time.time() - t0:.0f}s")
    sys.exit(1 if bad else 0)

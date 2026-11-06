"""
Study CL (Step 1): the label-free checks of a GRU simulator under three sources of joint commands.

Joint commands are the 12 joint position targets given at every step; the velocity command is the walking speed
and turning rate the policy is asked to follow (s1_policy.py). Every rollout starts where the registered 368-step
rollout starts: after each trajectory's 32 true history rows, on the model's own arena, with its own
normalisation. The registered functions are imported, never copied: lyap.rollout_map, lyap.first_step,
lyap.step, lyap.lyapunov and common.monitors.

Sources of joint commands after the history (forecast step 1 always keeps the recorded command, as lyap.first_step
does; forecast step k predicts row START + k - 1):
  rec    the recorded commands (the registered setting; recomputes the stored lambda_1 and r_J)
  pol    the authors' actor in closed loop (s1_policy.Policy): lambda1_pol, rJ_pol, imagined return
  cl     the same closed loop, with the policy differentiated too: lambda1_cl (Benettin, forward-mode JVPs)
  noise  recorded commands plus Gaussian noise: lambda1_noise, rJ_noise

lambda1_pol is lyap.lyapunov with the command array after the history replaced by the closed-loop commands, so the
policy's own feedback is not differentiated. lambda1_cl differentiates the composed map (policy, then simulator
step) with the policy's previous joint command in the state; its start directions are lyap.lyapunov's (same seed,
same draw, zero in the 12 added coordinates).

Divergence (new conditions only): the first forecast step at which any normalised state component is non-finite
or exceeds 1e3 in absolute value. The checks of a diverged rollout are +inf; the step is recorded (0 = none).
The imagined return of a diverged rollout is its sum up to that step, flagged.

No error is computed under policy or noisy commands: there is no recording to compare against.
"""
import time

import numpy as np
import torch
from torch.func import jvp

import common as C
import lyap
import s1_policy as SP

START, DT = C.START, C.DT
DIV_THRESHOLD = 1e3
NOISE_FRACTION = 0.1
ITEMS = ("rec", "pol", "lam_pol", "cl", "noise", "lam_noise")


# ------------------------------------------------------------------ divergence
def divergence_step(pred):
    """pred (B, T, 45) normalised. First forecast step k (1-based, row START + k - 1) that breaks the rule; 0 if none."""
    x = torch.as_tensor(pred)[:, START:]
    bad = (~torch.isfinite(x) | (x.abs() > DIV_THRESHOLD)).any(-1)          # (B, 368)
    first = torch.where(bad.any(1), bad.float().argmax(1) + 1, torch.zeros(bad.shape[0], dtype=torch.long))
    return first.numpy().astype(int)


# ------------------------------------------------------------------ closed loop
@torch.no_grad()
def closed_loop(model, st, ac, policy):
    """
    The simulator driven by `policy` after forecast step 1. Returns the predictions (B, T, 45), laid out like
    lyap.rollout_map's, and the joint commands actually applied (B, T, 12): row i of the returned array is the
    command that moved row i - 1 to row i, so lyap.rollout_map(model, st, A) reproduces the predictions.
    """
    B, T, _ = st.shape
    pred, A = st.clone(), ac.clone()
    h, s = lyap.first_step(model, st, ac)
    pred[:, START] = s
    a_prev = ac[:, START]                     # the recorded command of forecast step 1
    for i in range(START + 1, T):
        a = policy(s, a_prev, i - 1)
        A[:, i] = a
        h, s = lyap.step(model, h, s, a)
        pred[:, i] = s
        a_prev = a
    return pred, A


def lyapunov_cl(model, st, ac, policy, p=1, seed=0, dtype=torch.float64):
    """
    Closed-loop finite-time Lyapunov exponents over forecast steps 1 -> 368 (K = 367 products), Benettin's method on
    z = (h, s, a_prev) through z -> (step(h, s, pi(s, a_prev)), pi(s, a_prev)). Returns lam (B, p) in 1/s, the
    leading log stretch (B, K) and the divergence step of the float64 trajectory (B,).
    """
    model = model.to(dtype)
    st, ac = st.to(dtype), ac.to(dtype)
    B, T, _ = st.shape
    with torch.no_grad():
        h, s = lyap.first_step(model, st, ac)
    a_prev = ac[:, START]
    L, _, H = h.shape
    nhs = L * H + s.shape[1]
    n = nhs + a_prev.shape[1]
    rng = np.random.default_rng(seed)
    Q0, _ = np.linalg.qr(rng.standard_normal((nhs, max(p, 4))))          # lyap.lyapunov's draw
    Q0 = np.vstack([Q0[:, :p], np.zeros((n - nhs, p))])
    Q = torch.as_tensor(Q0, dtype=dtype)[None].repeat(B, 1, 1)
    div = np.zeros(B, dtype=int)
    logs = []
    for i in range(START + 1, T):
        r = i - 1

        def G(hh, ss, aa):
            a = policy(ss, aa, r)
            h2, s2 = lyap.step(model, hh, ss, a)
            return h2, s2, a

        def push(q):
            qh = q[:, :L * H].reshape(B, L, H).permute(1, 0, 2)
            (h2, s2, a2), (th, ts, ta) = jvp(G, (h, s, a_prev), (qh, q[:, L * H:nhs], q[:, nhs:]))
            return torch.cat([th.permute(1, 0, 2).reshape(B, L * H), ts, ta], -1), (h2, s2, a2)

        with torch.no_grad():
            outs = [push(Q[:, :, j]) for j in range(p)]
            V = torch.stack([o[0] for o in outs], -1)
            h, s, a_prev = outs[0][1]
            bad = (~torch.isfinite(s) | (s.abs() > DIV_THRESHOLD)).any(-1).numpy()
            div[(div == 0) & bad] = i - START + 1
            ok = torch.isfinite(V).all(-1).all(-1)
            V = torch.where(ok[:, None, None], V, Q)       # keep a diverged trajectory's QR finite; it is set to inf
            Qn, Rr = torch.linalg.qr(V)
            d = torch.diagonal(Rr, dim1=-2, dim2=-1)
            sg = torch.sign(d)
            sg[sg == 0] = 1
            Q = Qn * sg[:, None, :]
            logs.append(torch.log(d.abs()))
    logs = torch.stack(logs, 1)
    lam = (logs.sum(1) / (logs.shape[1] * DT)).numpy()
    model.float()
    return lam, logs[:, :, 0].numpy(), div


def _lam1(model, st, ac, div, p):
    """lyap.lyapunov's lambda_1 on the trajectories that did not diverge; +inf for the rest and for non-finite values."""
    lam = np.full(st.shape[0], np.inf)
    keep = np.flatnonzero(div == 0)
    if len(keep):
        try:
            ix = torch.as_tensor(keep)
            v = lyap.lyapunov(model, st[ix], ac[ix], p=p)[0][:, 0]
        except RuntimeError:                                       # a float64 blow-up the float32 rule missed
            v = np.array([_one(model, st[[b]], ac[[b]], p) for b in keep])
        lam[keep] = v
    model.float()
    lam[~np.isfinite(lam)] = np.inf
    return lam


def _one(model, st, ac, p):
    try:
        return float(lyap.lyapunov(model, st, ac, p=p)[0][0, 0])
    except RuntimeError:
        return np.inf


def _rJ(pred, sc, div=None):
    rJ = C.monitors(C.denorm(pred.numpy()), sc)[1]                 # (B, 368)
    m = rJ.mean(-1)
    if div is not None:
        m = np.where(div > 0, np.inf, m)
    m[~np.isfinite(m)] = np.inf
    return m, rJ


# ------------------------------------------------------------------ noise
def noise_sd():
    """Per-dimension std of the noise in raw units: 0.1 x the std of the recorded joint commands over all rows."""
    _, dat, _ = C.data()
    return NOISE_FRACTION * dat[:, C.R.ACTION_COLS].std(0)


def noisy_commands(ac):
    """Recorded commands plus noise after forecast step 1 (rows START+1..T-1); trajectory b draws with seed b."""
    B, T, _ = ac.shape
    sd = noise_sd()
    raw = SP.denorm_action(ac.double()).numpy()
    for b in range(B):
        raw[b, START + 1:] += np.random.default_rng(b).standard_normal((T - START - 1, raw.shape[-1])) * sd
    return SP.norm_action(torch.as_tensor(raw)).to(ac.dtype)


# ------------------------------------------------------------------ auxiliary outputs and the imagined return
@torch.no_grad()
def aux_outputs(model, S, A):
    """
    The model's contact and termination logits and its epistemic uncertainty for every predicted row, from one pass
    over the inputs cat(S[row - 1], A[row]) (the GRUs are causal, so this equals the step-wise imagination).
    Returns contacts (B, T, 8), terminations (B, T, 1) or None, epistemic (B, T), and the state path's
    prediction (B, T, 45); row 0 is unused.
    """
    X = torch.cat([S[:, :-1], A[:, 1:]], -1)
    ao, _ = model.auxiliary_base.memory.rnn(X)
    so, _ = model.state_base.memory.rnn(X)
    pad = lambda x: torch.cat([torch.zeros_like(x[:, :1]), x], 1)          # noqa: E731
    c = pad(torch.stack([hd.contact_layers(ao) for hd in model.auxiliary_heads]).mean(0))
    tm = None
    if getattr(model, "termination_dim", 0) > 0:
        tm = pad(torch.stack([hd.termination_layers(ao) for hd in model.auxiliary_heads]).mean(0))
    deltas = torch.stack([hd.state_mean_layers(so) for hd in model.state_heads])        # (E, B, T-1, 45)
    epi = deltas.std(0).sum(-1) if deltas.shape[0] > 1 else torch.zeros(deltas.shape[1:3], dtype=S.dtype)
    return c, tm, pad(epi[..., None])[..., 0], pad(S[:, :-1] + deltas.mean(0))


@torch.no_grad()
def imagined_return(model, pred, A, cmd, div):
    """
    Sum of the per-step imagination reward over the closed-loop forecast steps 2..368, per trajectory, up to the
    divergence step. Returns the return (B,), its 12 weighted terms (B, 12), the uncertainty penalty over the same
    steps (B,; zero for one-head models, reported apart), and the number of steps the termination head flags (B,).
    """
    B, T, _ = pred.shape
    c, tm, epi, _ = aux_outputs(model, pred, A)
    W, wu = SP.reward_weights()
    env = SP.stand_in_env(B, pred.dtype)
    ret = torch.zeros(B, dtype=torch.float64)
    terms = torch.zeros(B, len(W), dtype=torch.float64)
    unc = torch.zeros(B, dtype=torch.float64)
    nterm = torch.zeros(B, dtype=torch.long)
    for i in range(START + 1, T):
        r = i - 1
        valid = torch.as_tensor((div == 0) | (i - START + 1 < div))
        rew, tt = SP.step_reward(env, pred[:, r], A[:, r], cmd[:, r], pred[:, i], A[:, i], c[:, i])
        ret += torch.where(valid, rew.double(), 0.0)
        terms += torch.where(valid[:, None], tt.double(), 0.0)
        unc += torch.where(valid, (wu * epi[:, i] * DT).double(), 0.0)
        if tm is not None:
            nterm += (valid & torch.sigmoid(tm[:, i, 0]).round().bool()).long()
    return ret.numpy(), terms.numpy(), unc.numpy(), nterm.numpy()


# ------------------------------------------------------------------ the entry point
def score_gru(model, arena, scales, items=ITEMS, p=1, actor=None, policy=None):
    """
    Per-trajectory values for one GRU simulator on its arena. `items` selects what is computed (see ITEMS);
    "lam_pol" runs "pol" if it was not asked for. p: number of Lyapunov exponents (lambda_1 = column 0 is the
    same for every p, because lyap.lyapunov always draws the same first start direction).
    Returns a dict of numpy arrays (B,) or (B, ...) and "seconds" per item.
    """
    st, ac = arena["st"], arena["ac"]
    out, sec = {}, {}
    if any(k in items for k in ("pol", "lam_pol", "cl")) and policy is None:
        policy = SP.Policy(actor if actor is not None else SP.load_actor(), SP.arena_commands(arena))
    if "rec" in items:
        t0 = time.time()
        out["rJ_rec"], out["rJ_rec_steps"] = _rJ(lyap.rollout_map(model, st, ac), scales)
        sec["rec_rJ"] = time.time() - t0
        t0 = time.time()
        out["lam1_rec"] = lyap.lyapunov(model, st, ac, p=p)[0][:, 0]
        model.float()
        sec["rec_lam"] = time.time() - t0
    if "pol" in items or "lam_pol" in items:
        t0 = time.time()
        pred, A = closed_loop(model, st, ac, policy)
        div = divergence_step(pred)
        out["div_pol"] = div
        out["rJ_pol"], out["rJ_pol_steps"] = _rJ(pred, scales, div)
        sec["pol_rollout_rJ"] = time.time() - t0
        t0 = time.time()
        cmd = policy.cmd if hasattr(policy, "cmd") else SP.arena_commands(arena)
        out["ret"], out["ret_terms"], out["ret_unc"], out["n_term"] = imagined_return(model, pred, A, cmd, div)
        out["ret_diverged"] = div > 0
        sec["pol_return"] = time.time() - t0
        out["A_pol"] = A.numpy()
        if "lam_pol" in items:
            t0 = time.time()
            out["lam1_pol"] = _lam1(model, st, A, div, p)
            sec["lam_pol"] = time.time() - t0
    if "cl" in items:
        t0 = time.time()
        lam, _, dcl = lyapunov_cl(model, st, ac, policy, p=p)
        lam = np.where(dcl > 0, np.inf, lam[:, 0])
        lam[~np.isfinite(lam)] = np.inf
        out["lam1_cl"], out["div_cl"] = lam, dcl
        sec["cl"] = time.time() - t0
    if "noise" in items or "lam_noise" in items:
        t0 = time.time()
        acn = noisy_commands(ac)
        pred = lyap.rollout_map(model, st, acn)
        div = divergence_step(pred)
        out["div_noise"] = div
        out["rJ_noise"], out["rJ_noise_steps"] = _rJ(pred, scales, div)
        sec["noise_rollout_rJ"] = time.time() - t0
        if "lam_noise" in items:
            t0 = time.time()
            out["lam1_noise"] = _lam1(model, st, acn, div, p)
            sec["lam_noise"] = time.time() - t0
    out["seconds"] = sec
    return out

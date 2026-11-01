# copied unchanged from ../NGCAPS/scripts/lyap.py, sha256 efebc69733232188294a2a39d5b4ceeb2062fe72eae257854d0f78157f4c17f4
# copied from ../ICMAI/scripts/lyap.py, sha256 44731bc09010061a8f05cdc8312556048d13202e970ba9fd1822bdc81acdcdf2
"""
The rollout as an explicit map z_{k+1} = F_k(z_k), z = (h, s), and its finite-time Lyapunov exponents by
Benettin's method with forward-mode Jacobian-vector products (PREREG.md, "The map and its Jacobian").
"""
import numpy as np
import torch
from torch.func import jvp, vmap

from common import START, DT


def parts(model):
    rnn = model.state_base.memory.rnn
    heads = list(model.state_heads)
    return rnn, heads


def step(model, h, s, a):
    """One forecast step after the history: h (L,B,H), s (B,45), a (B,12) -> (h', s')."""
    rnn, heads = parts(model)
    out, h2 = rnn(torch.cat([s, a], -1)[:, None, :], h)
    x = out[:, -1]
    delta = torch.stack([hd.state_mean_layers(x) for hd in heads], 0).mean(0)
    return h2, s + delta


def first_step(model, st, ac, action_offset=1):
    """Forecast step 1: consumes the 32 true history rows; returns h_1, s_1 (prediction for row START)."""
    rnn, heads = parts(model)
    s_in = st[:, 0:START]
    a_in = ac[:, action_offset:START + action_offset]
    out, h = rnn(torch.cat([s_in, a_in], -1), None)
    x = out[:, -1]
    delta = torch.stack([hd.state_mean_layers(x) for hd in heads], 0).mean(0)
    return h, s_in[:, -1] + delta


@torch.no_grad()
def rollout_map(model, st, ac, action_offset=1):
    """Rollout through the explicit map; returns predictions (B, LEN, 45) like the loader's."""
    B, T, _ = st.shape
    pred = st.clone()
    h, s = first_step(model, st, ac, action_offset)
    pred[:, START] = s
    for i in range(START + 1, T):
        a = ac[:, i - 1 + action_offset]
        h, s = step(model, h, s, a)
        pred[:, i] = s
    return pred


def lyapunov(model, st, ac, p=4, seed=0, action_offset=1, dtype=torch.float64):
    """
    Leading p finite-time Lyapunov exponents per trajectory over forecast steps 1 -> 368 (K = 367 products).
    Returns lam (B, p) in 1/s, and the per-step log stretch of the leading direction (B, K).
    """
    model = model.to(dtype)
    st, ac = st.to(dtype), ac.to(dtype)
    B, T, _ = st.shape
    with torch.no_grad():
        h, s = first_step(model, st, ac, action_offset)
    L, _, H = h.shape
    n = L * H + s.shape[1]
    rng = np.random.default_rng(seed)
    # always draw 4 columns and keep the first p, so the leading start vector is the same for every p
    # (check 3 first failed because a (n,1) and an (n,4) draw differ; ANOMALIES.md A-01)
    Q0, _ = np.linalg.qr(rng.standard_normal((n, max(p, 4))))
    Q0 = Q0[:, :p]
    Q = torch.as_tensor(Q0, dtype=dtype)[None].repeat(B, 1, 1)          # (B, n, p)
    logs = []

    def F(hh, ss, aa):
        return step(model, hh, ss, aa)

    for i in range(START + 1, T):
        a = ac[:, i - 1 + action_offset]

        def push(q):                                                    # q: (B, n) one tangent per trajectory
            qh = q[:, :L * H].reshape(B, L, H).permute(1, 0, 2)
            qs = q[:, L * H:]
            (h2, s2), (th, ts) = jvp(lambda hh, ss: F(hh, ss, a), (h, s), (qh, qs))
            return torch.cat([th.permute(1, 0, 2).reshape(B, L * H), ts], -1), (h2, s2)

        with torch.no_grad():
            outs = [push(Q[:, :, j]) for j in range(p)]
            V = torch.stack([o[0] for o in outs], -1)                    # (B, n, p)
            h, s = outs[0][1]
            Qn, Rr = torch.linalg.qr(V)
            d = torch.diagonal(Rr, dim1=-2, dim2=-1)
            sg = torch.sign(d)
            sg[sg == 0] = 1
            Q = Qn * sg[:, None, :]
            logs.append(torch.log(d.abs()))
    logs = torch.stack(logs, 1)                                         # (B, K, p)
    K = logs.shape[1]
    lam = logs.sum(1) / (K * DT)
    model.float()
    return lam.numpy(), logs[:, :, 0].numpy()


def fd_check(model, st, ac, eps=1e-3, action_offset=1):
    """Check 2: jvp against a central finite difference on one step (float64)."""
    model = model.double()
    st, ac = st.double(), ac.double()
    with torch.no_grad():
        h, s = first_step(model, st, ac, action_offset)
    a = ac[:, START + action_offset]
    g = torch.Generator().manual_seed(0)
    qh = torch.randn(h.shape, generator=g, dtype=torch.float64)
    qs = torch.randn(s.shape, generator=g, dtype=torch.float64)
    nrm = torch.sqrt((qh ** 2).sum() + (qs ** 2).sum())
    qh, qs = qh / nrm, qs / nrm
    with torch.no_grad():
        _, (th, ts) = jvp(lambda hh, ss: step(model, hh, ss, a), (h, s), (qh, qs))
        hp, sp = step(model, h + eps * qh, s + eps * qs, a)
        hm, sm = step(model, h - eps * qh, s - eps * qs, a)
    fdh, fds = (hp - hm) / (2 * eps), (sp - sm) / (2 * eps)
    num = torch.sqrt(((th - fdh) ** 2).sum() + ((ts - fds) ** 2).sum())
    den = torch.sqrt((th ** 2).sum() + (ts ** 2).sum())
    model.float()
    return float(num / den)

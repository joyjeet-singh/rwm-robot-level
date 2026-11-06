"""
Study AR (Step 1): adapters that score the MLP, RSSM and Transformer simulators of the earlier reproduction
(../rwm_repro/src/baselines, imported, never copied) with this paper's measures, under the recorded joint commands.

Joint commands are the 12 joint position targets given at every step (the actions). Every adapter pairs them with
states as its model was trained (baselines/common.py action_rows_for; rssm.py _transition): the model that predicts
state row r reads joint-command row r as its newest command. That is lyap.rollout_map's pairing (action_offset = 1),
so nothing differs from the GRU measurement here.

Each adapter is a set of pure functions of (model, recurrent state, joint command):
  first(model, st, ac) -> z     consume the 32 true history rows; z after forecast step 1 (prediction of row START)
  step(model, z, a)    -> z'    one forecast step; a is the joint command of the row being predicted
  pred(z)              -> (B, 45) the newest predicted state
and z is a tuple of batch-first tensors, all of whose coordinates enter the tangent space of the Lyapunov exponent:
  MLP, Transformer  z = (S, A): the window of the last 32 states (true history rows, replaced one by one by the
                    model's own predictions) and the 32 joint commands of the same rows. The oldest command is not
                    read by the next prediction, so its tangent coordinate maps to zero. Under recorded commands the
                    command coordinates are shifted out within 32 steps (a finite-time offset, as in DECISIONS S1-4).
  RSSM              z = (h, s): the deterministic GRU latent (layers x units) and the last predicted state, which is
                    the residual base of the next prediction. The stochastic latent is not a coordinate: as in the
                    reproduction's rollout it is the prior's mode given h, never sampled. For the categorical latent
                    (every RSSM run) the mode is a one-hot argmax, piecewise constant in h, so its derivative is zero;
                    the straight-through term of rssm.py _sample is left out of the map, because it would give the
                    derivative of the class probabilities, not of the rollout. A Gaussian latent would use its mean.

Measures (score_arch), on the arena's trajectories, with exactly the registered semantics:
  E1, E8, E24, E100  evalh.window_error's windows (every position for E1, every 4th after 32 true history rows
                     otherwise), each window rolled out from its own 32 true rows, normalised MAE, float32
  E368               the mean normalised MAE of the single 368-step rollout (compute.per_model's err)
  lam1_rec           Benettin as lyap.lyapunov: float64, forward-mode JVPs, K = 367 products over forecast steps
                     1 -> 368, start direction from numpy default_rng(0) (an (n, 4) draw, QR, first column), 1/s
  rJ_rec             mean over the 368 steps of common.monitors' scaled joint residual r_J
Diagnostics, not measures: div_rec (s1_rollout.divergence_step of the float32 rollout, 0 = none) and lam_div
(the float64 Benettin trajectory became non-finite; lam1_rec is then +inf). No value is overwritten by them.
"""
import contextlib
import glob
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as Fn
from torch.func import jvp

import common as C
import evalh
import lyap
import s1_rollout as S
import baselines as BL                      # ../rwm_repro/src, on the path through common

START, LEN, DT = C.START, C.LEN, C.DT
HW = (1, 8, 24, 100)
ARCHES = ("mlp", "rssm", "transformer")


# ------------------------------------------------------------------ loading
def final_weights(run_dir):
    ws = glob.glob(os.path.join(run_dir, "weights_*.pt"))
    its = [int(re.search(r"weights_(\d+)\.pt$", w).group(1)) for w in ws]
    if not its:
        raise FileNotFoundError(run_dir)
    return os.path.join(run_dir, f"weights_{max(its)}.pt"), max(its)


def build(arch, spec):
    cfg, _, _ = C.data()
    return BL.build(arch, spec, cfg)


def load(run_dir):
    """A run's final weights in eval mode, and its record (arch, regime, spec, seed, iter, path, sha256)."""
    path, it = final_weights(run_dir)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build(ck["arch"], ck["spec"])
    m.load_state_dict(ck["model_state_dict"], strict=True)
    m.eval()
    info = {k: ck[k] for k in ("arch", "regime", "spec", "seed")}
    info.update(iter=it, path=os.path.relpath(path, C.PDM), sha256=C.sha256(path))
    return m, info


def arch_of(model):
    if isinstance(model, BL.MLPBaseline):
        return "mlp"
    if isinstance(model, BL.TransformerBaseline):
        return "transformer"
    if isinstance(model, BL.RSSMBaseline):
        return "rssm"
    if hasattr(model, "state_base"):
        return "gru"
    raise TypeError(type(model))


# ------------------------------------------------------------------ window models (MLP, Transformer)
class Window:
    @staticmethod
    def first(model, st, ac):
        H = model.history
        assert H == START
        return Window.step(model, (st[:, :H], ac[:, :H]), ac[:, H])

    @staticmethod
    def step(model, z, a):
        Sw, Aw = z
        x_action = torch.cat([Aw[:, 1:], a[:, None]], 1)          # action_rows_for: rows r - H + 1 .. r
        mean, _ = model.predict(Sw, x_action)
        return torch.cat([Sw[:, 1:], mean[:, None]], 1), x_action

    @staticmethod
    def pred(z):
        return z[0][:, -1]


# ------------------------------------------------------------------ RSSM
class RSSM:
    @staticmethod
    def mode(model, d):
        """The latent the rollout uses (rssm.py _sample(mode=True)) from a distribution of model._dist, without the
        straight-through term."""
        if model.latent == "categorical":
            probs = torch.softmax(d, dim=-1).detach()             # detached: the mode has no derivative
            onehot = Fn.one_hot(probs.argmax(-1), model.n_classes).to(probs.dtype)
            return (onehot + probs - probs).flatten(1)            # rssm.py's arithmetic: 1 + p - p is not always 1
        return d[0]

    @staticmethod
    def first(model, st, ac):
        """Filter the 32 history rows with the posterior's mode, then forecast row START (rssm.py rollout)."""
        B = st.shape[0]
        hstate = torch.zeros(model.gru_layers, B, model.deter, dtype=st.dtype)
        zl = torch.zeros(B, model.z_dim, dtype=st.dtype)
        for t in range(START):
            h, hstate, _ = model._transition(hstate, zl, ac[:, t])
            zl = RSSM.mode(model, model._posterior(h, st[:, t]))
        return RSSM._forecast(model, hstate, zl, st[:, START - 1], ac[:, START])

    @staticmethod
    def _forecast(model, hstate, zl, prev, a):
        h, hstate, prior = model._transition(hstate, zl, a)
        zn = RSSM.mode(model, prior)
        mean, _ = model._decode(h, zn, prev)
        return hstate.permute(1, 0, 2), mean

    @staticmethod
    def step(model, z, a):
        hb, prev = z                                              # hb (B, layers, units)
        hstate = hb.permute(1, 0, 2)
        zl = RSSM.mode(model, model._dist(model.prior_net(hstate[-1])))        # the prior's mode at the previous row
        return RSSM._forecast(model, hstate, zl, prev, a)

    @staticmethod
    def pred(z):
        return z[1]


# ------------------------------------------------------------------ GRU (tests only)
class GRU:
    """The paper's GRU through lyap.first_step / lyap.step, z = (h batch-first, s). Used only by the tests, to check
    that this module's rollout, windows and Benettin reproduce lyap.rollout_map, evalh and lyap.lyapunov."""
    @staticmethod
    def first(model, st, ac):
        h, s = lyap.first_step(model, st, ac)
        return h.permute(1, 0, 2), s

    @staticmethod
    def step(model, z, a):
        h, s = lyap.step(model, z[0].permute(1, 0, 2).contiguous(), z[1], a)
        return h.permute(1, 0, 2), s

    @staticmethod
    def pred(z):
        return z[1]


ADAPTER = {"mlp": Window, "transformer": Window, "rssm": RSSM, "gru": GRU}


def adapter(model):
    return ADAPTER[arch_of(model)]


# ------------------------------------------------------------------ (a) one step, (b) rollout
@torch.no_grad()
def predict_one(model, st, ac):
    """One-step prediction of row START from rows 0..START-1 and joint commands up to row START: (B, 45)."""
    ad = adapter(model)
    return ad.pred(ad.first(model, st, ac))


@torch.no_grad()
def rollout(model, st, ac):
    """The rollout driven by the joint commands `ac` after the 32 true history rows; laid out like lyap.rollout_map."""
    ad = adapter(model)
    T = st.shape[1]
    pred = st.clone()
    z = ad.first(model, st, ac)
    pred[:, START] = ad.pred(z)
    for i in range(START + 1, T):
        z = ad.step(model, z, ac[:, i])
        pred[:, i] = ad.pred(z)
    return pred


@torch.no_grad()
def window_error(model, st, ac, h, chunk=512):
    """evalh.window_error's semantics (same windows, same averaging) through this adapter: (B,)."""
    stride = evalh.STRIDE[h]
    B, T, _ = st.shape
    ks = evalh.starts(h, stride, T)
    idx = torch.as_tensor([[k - START + j for j in range(START + h)] for k in ks])
    W = len(ks)
    sw = st[:, idx].reshape(B * W, START + h, -1)
    aw = ac[:, idx].reshape(B * W, START + h, -1)
    errs = []
    for i0 in range(0, B * W, chunk):
        s_, a_ = sw[i0:i0 + chunk], aw[i0:i0 + chunk]
        p = rollout(model, s_, a_)
        errs.append((p[:, START:] - s_[:, START:]).abs().mean(-1).mean(1))
    return torch.cat(errs).reshape(B, W).mean(1).numpy()


# ------------------------------------------------------------------ (c) the step map and its Lyapunov exponent
def flat(z):
    B = z[0].shape[0]
    return torch.cat([c.reshape(B, -1) for c in z], -1)


def unflat(q, z):
    out, i = [], 0
    for c in z:
        n = c[0].numel()
        out.append(q[:, i:i + n].reshape(c.shape))
        i += n
    return tuple(out)


@contextlib.contextmanager
def jvp_mode(model):
    """
    torch 2.2's fused TransformerEncoderLayer kernel (used in eval mode without gradients) and its flash attention
    kernel have no forward-mode derivative. Inside the Lyapunov computation the Transformer runs in training mode,
    which takes the unfused layer (the same function: every dropout of the model is 0 and it has no batch statistics),
    with attention restricted to PyTorch's plain "math" kernel. Other models are left as they are.
    """
    if not isinstance(model, BL.TransformerBaseline):
        yield
        return
    was = model.training
    model.train(True)
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            yield
    finally:
        model.train(was)


def step_jvp(model, z, a, q):
    """Forward-mode JVP of the step map at z along the flat tangent q (B, n): (z', q' (B, n))."""
    ad = adapter(model)
    with jvp_mode(model):
        z2, t2 = jvp(lambda *zz: ad.step(model, zz, a), z, unflat(q, z))
    return z2, flat(t2)


def lyapunov(model, st, ac, p=1, seed=0, dtype=torch.float64):
    """
    lyap.lyapunov's Benettin method on this adapter's step map: float64, K = 367 products over forecast steps
    1 -> 368, the same start draw. A trajectory whose tangent or state becomes non-finite keeps a finite QR (its
    previous basis) and is reported with lam = +inf. Returns lam (B, p) in 1/s, the leading log stretch (B, K) and
    the non-finite flag (B,).
    """
    model = model.to(dtype)
    st, ac = st.to(dtype), ac.to(dtype)
    B, T, _ = st.shape
    with torch.no_grad():
        z = adapter(model).first(model, st, ac)
    n = flat(z).shape[1]
    rng = np.random.default_rng(seed)
    Q0, _ = np.linalg.qr(rng.standard_normal((n, max(p, 4))))          # lyap.lyapunov's draw
    Q = torch.as_tensor(Q0[:, :p], dtype=dtype)[None].repeat(B, 1, 1)
    bad = torch.zeros(B, dtype=torch.bool)
    logs = []
    for i in range(START + 1, T):
        a = ac[:, i]
        with torch.no_grad():
            outs = [step_jvp(model, z, a, Q[:, :, j]) for j in range(p)]
            V = torch.stack([o[1] for o in outs], -1)
            z = outs[0][0]
            ok = torch.isfinite(V).all(-1).all(-1) & torch.isfinite(flat(z)).all(-1)
            bad |= ~ok
            V = torch.where(ok[:, None, None], V, Q)
            Qn, Rr = torch.linalg.qr(V)
            d = torch.diagonal(Rr, dim1=-2, dim2=-1)
            sg = torch.sign(d)
            sg[sg == 0] = 1
            Q = Qn * sg[:, None, :]
            logs.append(torch.log(d.abs()))
    logs = torch.stack(logs, 1)
    lam = (logs.sum(1) / (logs.shape[1] * DT)).numpy()
    lam[bad.numpy()] = np.inf
    lam[~np.isfinite(lam)] = np.inf
    model.float()
    return lam, logs[:, :, 0].numpy(), bad.numpy()


def fd_check(model, st, ac, eps=1e-6, seed=0):
    """Relative difference between the step map's JVP and a central finite difference, at forecast step 2 (float64)."""
    model = model.double()
    st, ac = st.double(), ac.double()
    ad = adapter(model)
    with torch.no_grad():
        z = ad.first(model, st, ac)
        a = ac[:, START + 1]
        g = torch.Generator().manual_seed(seed)
        q = torch.randn(flat(z).shape, generator=g, dtype=torch.float64)
        q = q / q.norm()
        _, t = step_jvp(model, z, a, q)
        with jvp_mode(model):
            zp = ad.step(model, unflat(flat(z) + eps * q, z), a)
            zm = ad.step(model, unflat(flat(z) - eps * q, z), a)
    fd = (flat(zp) - flat(zm)) / (2 * eps)
    model.float()
    return float((t - fd).norm() / t.norm())


# ------------------------------------------------------------------ the entry point
def score_arch(run_dir, arena, scales, model=None):
    """
    Per-trajectory values for one baseline run (its final weights) on `arena` (common.arena), recorded joint commands
    only. `model` replaces loading (tests, probe). Returns a dict of numpy arrays: E1, E8, E24, E100, E368, lam1_rec,
    rJ_rec (B,); err_steps, rJ_rec_steps (B, 368); log1 (B, 367); div_rec, lam_div (B,); plus "info" and "seconds".
    """
    torch.set_num_threads(2)
    info = {}
    if model is None:
        model, info = load(run_dir)
    info["arch"] = arch_of(model)
    model.float().eval()
    st, ac = arena["st"], arena["ac"]
    out, sec = {}, {}
    t0 = time.time()
    pred = rollout(model, st, ac)
    err = (pred[:, START:] - st[:, START:]).abs().mean(-1).numpy()
    out["err_steps"], out["E368"] = err, err.mean(1)
    out["div_rec"] = S.divergence_step(pred)
    out["rJ_rec"], out["rJ_rec_steps"] = S._rJ(pred, scales)
    sec["rollout_rJ"] = time.time() - t0
    t0 = time.time()
    for h in HW:
        out[f"E{h}"] = window_error(model, st, ac, h)
    sec["windows"] = time.time() - t0
    t0 = time.time()
    lam, log1, bad = lyapunov(model, st, ac, p=1)
    out["lam1_rec"], out["log1"], out["lam_div"] = lam[:, 0], log1, bad
    sec["lam"] = time.time() - t0
    for k in ("E1", "E8", "E24", "E100", "E368"):
        v = np.asarray(out[k], dtype=np.float64)
        v[~np.isfinite(v)] = np.inf
        out[k] = v
    out["info"], out["seconds"] = info, sec
    return out

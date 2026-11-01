"""
Errors over windows (PREREG.md §4): E1 at every position, E8 / E24 / E100 from every 4th start, each window
reading the 32 true rows before its first predicted row and then running on its own predictions. Uses the
explicit map of lyap.py (the released rollout: ensemble mean fed back, action_offset = 1, float32).
"""
import torch

import common as C
import lyap

STRIDE = {1: 1, 8: 4, 24: 4, 100: 4, 368: 1}


def starts(h, stride, T=C.LEN):
    return list(range(C.START, T - h + 1, stride))


@torch.no_grad()
def window_error(model, st, ac, h, stride=None, chunk=512):
    """Per-trajectory mean error over all windows of length h, and the floor's. Returns two (B,) tensors."""
    stride = STRIDE[h] if stride is None else stride
    B, T, _ = st.shape
    ks = starts(h, stride, T)
    idx = torch.as_tensor([[k - C.START + j for j in range(C.START + h)] for k in ks])      # (W, 32 + h)
    W = len(ks)
    sw = st[:, idx].reshape(B * W, C.START + h, -1)
    aw = ac[:, idx].reshape(B * W, C.START + h, -1)
    errs = []
    for i0 in range(0, B * W, chunk):
        s_, a_ = sw[i0:i0 + chunk], aw[i0:i0 + chunk]
        hh, s = lyap.first_step(model, s_, a_)
        e = [(s - s_[:, C.START]).abs().mean(-1)]
        for i in range(C.START + 1, C.START + h):
            hh, s = lyap.step(model, hh, s, a_[:, i])
            e.append((s - s_[:, i]).abs().mean(-1))
        errs.append(torch.stack(e, 1).mean(1))
    err = torch.cat(errs).reshape(B, W).mean(1)
    return err.numpy(), window_floor(st, h, stride)


def window_floor(st, h, stride=None):
    """The floor (hold the last history row) over the same windows: (B,)."""
    stride = STRIDE[h] if stride is None else stride
    B, T, _ = st.shape
    ks = starts(h, stride, T)
    idx = torch.as_tensor([[k - C.START + j for j in range(C.START + h)] for k in ks])
    sw = st[:, idx]                                                                         # (B, W, 32 + h, 45)
    return (sw[:, :, C.START - 1:C.START] - sw[:, :, C.START:]).abs().mean(-1).mean(-1).mean(-1).numpy()

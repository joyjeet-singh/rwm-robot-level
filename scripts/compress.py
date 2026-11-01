"""
Compressed variants of a trained dynamics model, all made after training and none fine-tuned (PREREG.md §3).

  fp16      every floating-point parameter and buffer rounded to IEEE half precision and back (storage at 16 bits)
  int8w     every GRU and Linear weight matrix rounded to a symmetric per-tensor 8-bit grid,
            scale = max|W| / 127, values round(W / scale) clipped to [-127, 127], times scale; biases and the two
            learned log-std bounds untouched. Weights only: activations stay in float, so this is the weight
            part of 8-bit dynamic quantisation, not the whole of it.
  u10/u20/u30/u50   global magnitude pruning at 10/20/30/50 %: `unstructured` below, copied unchanged from
            ../PerconAI/perconai/prune.py (sha256 bda039b1f12f830b144ae65e2641b7eef2286ba20536a1c121c80d395d77dfc2)
  s192/s128 removing GRU hidden units to width 192/128 by the L2 importance score: `importance`, `keep_units`,
            `structured`, copied from the same file; the only change is that build_width takes its config
            from common.data() instead of PerconAI's harness.

Every variant leaves a float32 model that the Jacobian code can differentiate.
"""
import copy
import math

import numpy as np
import torch
from torch import nn

import common as C

VARIANTS = ("fp16", "int8w", "u10", "u20", "u30", "u50", "s192", "s128")
FULL = 256
BASES = {"state_base": ("state_heads", ("state_mean_layers.0", "state_logstd_layers.0")),
         "auxiliary_base": ("auxiliary_heads", ("contact_layers.0", "termination_layers.0"))}


# ------------------------------------------------------------------ 16-bit storage
def fp16(model):
    m = copy.deepcopy(model)
    with torch.no_grad():
        for t in list(m.parameters()) + list(m.buffers()):
            if t.is_floating_point():
                t.copy_(t.half().float())
    return m.eval(), {"kind": "fp16"}


# ------------------------------------------------------------------ the weight matrices (PerconAI definition)
def weight_matrices(model):
    """[(name, tensor)] of every nn.GRU weight_ih/weight_hh and nn.Linear weight, in named_parameters order."""
    owners = {n for n, mod in model.named_modules() if isinstance(mod, (nn.GRU, nn.Linear))}
    out = []
    for name, p in model.named_parameters():
        mod, _, leaf = name.rpartition(".")
        if mod in owners and (leaf == "weight" or leaf.startswith("weight_ih_l") or leaf.startswith("weight_hh_l")):
            out.append((name, p))
    return out


# ------------------------------------------------------------------ 8-bit weights
def int8w(model):
    m = copy.deepcopy(model)
    scales = {}
    with torch.no_grad():
        for name, w in weight_matrices(m):
            s = float(w.abs().max()) / 127.0
            if s > 0:
                w.copy_(torch.clamp(torch.round(w / s), -127, 127) * s)
            scales[name] = s
    return m.eval(), {"kind": "int8w", "scales": scales}


# ------------------------------------------------------------------ unstructured (copied unchanged)
def n_to_prune(p, n):
    return int(math.floor(p * n + 0.5))


def unstructured(model, p):
    """Global magnitude pruning at level p (module docstring). Returns (pruned deep copy, info)."""
    m = copy.deepcopy(model)
    mats = weight_matrices(m)
    pool = torch.cat([w.detach().abs().reshape(-1) for _, w in mats])
    n = pool.numel()
    k = n_to_prune(p, n)
    order = torch.argsort(pool, stable=True)
    idx = order[:k]
    mask = torch.zeros(n, dtype=torch.bool)
    mask[idx] = True
    per, off = {}, 0
    with torch.no_grad():
        for name, w in mats:
            sl = mask[off:off + w.numel()].view_as(w)
            w[sl] = 0.0
            per[name] = {"entries": w.numel(), "pruned": int(sl.sum()), "zeros_after": int((w == 0).sum())}
            off += w.numel()
    info = {"kind": "unstructured", "level": p, "pool_entries": n, "pruned_entries": k,
            "threshold_abs": float(pool[order[k - 1]]) if k else None,
            "largest_kept_abs_min": float(pool[order[k]]) if k < n else None, "per_matrix": per}
    return m.eval(), info


# ------------------------------------------------------------------ structured (copied; build_width adapted)
def _readers(model, base):
    heads, firsts = BASES[base]
    mods = dict(model.named_modules())
    names = [f"{heads}.{j}.{f}" for j in range(len(getattr(model, heads))) for f in firsts]
    for nm in names:
        assert isinstance(mods[nm], nn.Linear), nm
    return names


def importance(model, base):
    """{layer: float64 array (H,)}: the L2 norm of every weight that reads from or writes to each unit."""
    sd = {k: v.detach().double() for k, v in model.state_dict().items()}
    pre = f"{base}.memory.rnn."
    gru = getattr(model, base).memory.rnn
    h, L = gru.hidden_size, gru.num_layers
    out = {}
    for l in range(L):
        wih, whh = sd[f"{pre}weight_ih_l{l}"], sd[f"{pre}weight_hh_l{l}"]
        rows = lambda W: (W ** 2).sum(1).view(3, h).sum(0)                  # noqa: E731
        writes = rows(wih) + rows(whh)
        reads = (whh ** 2).sum(0)
        readers = [sd[f"{pre}weight_ih_l{l + 1}"]] if l + 1 < L else \
                  [sd[f"{nm}.weight"] for nm in _readers(model, base)]
        for Rm in readers:
            reads = reads + (Rm ** 2).sum(0)
        diag = torch.stack([whh[g * h + torch.arange(h), torch.arange(h)] for g in range(3)])
        out[l] = torch.sqrt(writes + reads - (diag ** 2).sum(0)).numpy()
    return out


def keep_units(score, width):
    """Indices of the `width` largest scores (ties: lower index first), ascending."""
    order = np.lexsort((np.arange(len(score)), -score))
    return sorted(int(i) for i in order[:width])


def structured(model, width):
    """Remove hidden units down to `width` in every layer of both GRUs. Returns (model, info)."""
    sd = model.state_dict()
    new = {k: v.clone() for k, v in sd.items()}
    kept = {}
    for base in BASES:
        gru = getattr(model, base).memory.rnn
        h, L = gru.hidden_size, gru.num_layers
        assert h == FULL and width <= h, (h, width)
        sc = importance(model, base)
        pre = f"{base}.memory.rnn."
        ks = {l: torch.tensor(keep_units(sc[l], width)) for l in range(L)}
        for l in range(L):
            K = ks[l]
            rows = torch.cat([K, K + h, K + 2 * h])
            wih = sd[f"{pre}weight_ih_l{l}"][rows]
            new[f"{pre}weight_ih_l{l}"] = wih if l == 0 else wih[:, ks[l - 1]]
            new[f"{pre}weight_hh_l{l}"] = sd[f"{pre}weight_hh_l{l}"][rows][:, K]
            for b in ("bias_ih", "bias_hh"):
                new[f"{pre}{b}_l{l}"] = sd[f"{pre}{b}_l{l}"][rows]
        for nm in _readers(model, base):
            new[f"{nm}.weight"] = sd[f"{nm}.weight"][:, ks[L - 1]]
        kept[base] = {str(l): ks[l].tolist() for l in range(L)}
    m = C.build_width(width, ensemble_size=model.ensemble_size)
    m.load_state_dict({k: v.contiguous() for k, v in new.items()}, strict=True)
    return m.eval(), {"kind": "structured", "width": width, "kept_units": kept}


def apply(model, variant):
    if variant == "fp16":
        return fp16(model)
    if variant == "int8w":
        return int8w(model)
    if variant[0] == "u":
        return unstructured(model, int(variant[1:]) / 100)
    if variant[0] == "s":
        return structured(model, int(variant[1:]))
    raise ValueError(variant)

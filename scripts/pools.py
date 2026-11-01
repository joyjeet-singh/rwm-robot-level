"""
Model pools (PREREG.md §3). Entry: (key, weights path, ensemble size, pool, group, variant or None, extras).
F18 pools R and T are NGCAPS's (../NGCAPS/scripts/pools.py) unchanged; pool C and the X5 variants are new.
"""
import compress as K
import common as C


def _p(d, it=2500):
    return C.run_path(d, it)


def pool_R():
    out = []
    for s in range(5):
        out.append((f"A_s{s}", _p(f"armA_seed{s}"), 1, "R", "A (M32,N8)", None, {"N": 8, "M": 32}))
    for s in range(3):
        out.append((f"A5_s{s}", _p(f"armA_seed{s}_ens5"), 5, "R", "A ensemble-5", None, {}))
    for s in range(5):
        out.append((f"nll_s{s}", _p(f"armA_seed{s}_nll"), 1, "R", "A NLL objective", None, {}))
        out.append((f"w124_s{s}", _p(f"armA_seed{s}_m49h124"), 1, "R", "A width 124", None, {}))
    for s in range(3):
        out.append((f"contam_s{s}", _p(f"armA_seed{s}_contam"), 1, "R", "A contaminated data", None, {}))
        out.append((f"dup_s{s}", _p(f"armA_seed{s}_dup"), 1, "R", "A duplicated data", None, {}))
        out.append((f"A10k_s{s}", _p(f"armA_seed{s}_10k", 10000), 1, "R", "A 10k iterations", None, {}))
        for M, N in ((1, 8), (2, 8), (8, 8), (16, 8), (32, 2), (32, 16), (32, 32)):
            out.append((f"mn_M{M}_N{N}_s{s}", _p(f"mn_M{M}_N{N}_seed{s}"), 1, "R", f"sweep M{M} N{N}", None,
                        {"N": N, "M": M}))
    return out


def pool_T():
    out = []
    for s in range(3):
        out.append((f"B_s{s}", _p(f"armB_seed{s}"), 1, "T", "B (teacher forced)", None, {}))
        out.append((f"B10k_s{s}", _p(f"armB_seed{s}_10k", 10000), 1, "T", "B 10k iterations", None, {}))
        out.append((f"mn_M32_N1_s{s}", _p(f"mn_M32_N1_seed{s}"), 1, "T", "sweep M32 N1", None, {"N": 1, "M": 32}))
    return out


def pool_C18():
    """The eight variants of each Arm A M32/N8 single model, seeds 0-4 (40 models)."""
    out = []
    for s in range(5):
        for v in K.VARIANTS:
            out.append((f"A_s{s}__{v}", _p(f"armA_seed{s}"), 1, "C", f"compressed {v}", v, {"parent": f"A_s{s}"}))
    return out


def pool_F18():
    return pool_R() + pool_T() + pool_C18()


def pool_X5():
    """55 five-fold models (A 5 seeds, A5 3, B 3 per fold) and the eight variants of every A model (200)."""
    out = []
    for f in C.FOLDS:
        for s in range(5):
            out.append((f"A_{C.fid(f)}_s{s}", _p(C.fold_dir("A", f, s)), 1, "X5", "A", None,
                        {"fold": C.fid(f), "seed": s}))
        for s in range(3):
            out.append((f"A5_{C.fid(f)}_s{s}", _p(C.fold_dir("A5", f, s)), 5, "X5", "A5", None,
                        {"fold": C.fid(f), "seed": s}))
        for s in range(3):
            out.append((f"B_{C.fid(f)}_s{s}", _p(C.fold_dir("B", f, s)), 1, "X5", "B", None,
                        {"fold": C.fid(f), "seed": s}))
        for s in range(5):
            for v in K.VARIANTS:
                out.append((f"A_{C.fid(f)}_s{s}__{v}", _p(C.fold_dir("A", f, s)), 1, "X5", f"compressed {v}", v,
                            {"fold": C.fid(f), "seed": s, "parent": f"A_{C.fid(f)}_s{s}"}))
    return out

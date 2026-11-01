"""
Per-model raw quantities for every pool (PREREG.md §4). No statistic is computed here.

  python scripts/compute.py F18     pools R, T, C on held-out episodes 1 and 8   -> results/raw_F18.npz
  python scripts/compute.py X5      the five-fold pool, each model on its fold    -> results/raw_X5.npz
  python scripts/compute.py REL     the released checkpoint on all 20 trajectories (descriptive) -> results/raw_REL.npz

Per model and trajectory: per-step error err (normalised MAE, float32 rollout), the floor, lambda_1..4 and the
leading log stretch (float64), the physics residuals r_G, r_J, r_A, and the physical errors (joint angle in
degrees, base velocity in m/s, dead-reckoned planar position in m). results/models.json holds every model's
meta-data: pool, group, variant, parent weights' SHA-256 and the variant's info.
"""
import json
import os
import sys
import time

import numpy as np
import torch

import common as C
import compress as K
import lyap
import pools

torch.set_num_threads(2)


def build(path, ens, variant):
    m = C.load(path, ens)
    info = None
    if variant is not None:
        m, info = K.apply(m, variant)
        if info.get("kind") == "int8w":
            info = {"kind": "int8w", "max_scale": max(info["scales"].values())}
        elif info.get("kind") == "unstructured":
            info = {k: info[k] for k in ("kind", "level", "pool_entries", "pruned_entries", "threshold_abs")}
        elif info.get("kind") == "structured":
            info = {"kind": "structured", "width": info["width"]}
    return m, info


def per_model(m, ar, sc):
    st, ac = ar["st"], ar["ac"]
    pred = lyap.rollout_map(m, st, ac)                                                     # float32
    err = (pred[:, C.START:] - st[:, C.START:]).abs().mean(-1).numpy()                     # (B, 368)
    P, S = C.denorm(pred.numpy()), C.denorm(st.numpy())
    rG, rJ, rA = C.monitors(P, sc)
    jdeg, vel, pos, _ = C.physical(P, S)
    lam, log1 = lyap.lyapunov(m, st, ac, p=4)                                               # float64
    return {"err": err, "lam": lam, "log1": log1, "rG": rG, "rJ": rJ, "rA": rA, "jdeg": jdeg, "vel": vel,
            "pos": pos, "finite": bool(np.isfinite(err).all() and np.isfinite(lam).all())}


def arena_extras(ar, fold_train):
    """Floors and recorded quantities shared by every model of one arena."""
    st = ar["st"]
    floor = (st[:, C.START - 1:C.START] - st[:, C.START:]).abs().mean(-1).numpy()
    S = C.denorm(st.numpy())
    hold = S.copy()
    hold[:, C.START:] = S[:, C.START - 1:C.START]
    jdeg, vel, pos, dist = C.physical(hold, S)
    sc = C.real_scales(fold_train)
    rG, rJ, rA = C.monitors(S, sc)                                       # the recording's own residuals
    return {"floor": floor, "floor_jdeg": jdeg, "floor_vel": vel, "floor_pos": pos, "dist": dist,
            "rec_rG": rG, "rec_rJ": rJ, "rec_rA": rA}, sc


def run(entries, arena_of, tag):
    t0 = time.time()
    raw, meta, done = {}, [], set()
    for key, path, ens, pool, group, variant, extra in entries:
        fold = arena_of(key)
        ar, sc = ARENAS[fold]
        if fold not in done:
            ex, _ = arena_extras(ar, C.train_episodes(fold) if fold != "all" else list(range(10)))
            for k, v in ex.items():
                raw[f"{k}_{C.fid(fold) if fold != 'all' else 'all'}"] = v
            done.add(fold)
        m, info = build(path, ens, variant)
        out = per_model(m, ar, sc)
        for k in ("err", "lam", "log1", "rG", "rJ", "rA", "jdeg", "vel", "pos"):
            raw[f"{k}_{key}"] = out[k]
        meta.append({"key": key, "pool": pool, "group": group, "variant": variant, "variant_info": info,
                     "extra": extra, "ensemble": ens, "finite": out["finite"],
                     "arena": C.fid(fold) if fold != "all" else "all", "starts": ar["starts"],
                     "sha256": C.sha256(path), "path": os.path.relpath(path, C.PDM)})
        print(tag, key, f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(os.path.join(C.RESULTS, f"raw_{tag}.npz"), **raw)
    return meta


ARENAS = {}


def setup_arena(fold):
    if fold not in ARENAS:
        if fold == "all":
            ars = [C.arena(f) for f in C.FOLDS]
            ar = {"starts": sum((a["starts"] for a in ars), []), "episode": sum((a["episode"] for a in ars), []),
                  "st": torch.cat([a["st"] for a in ars]), "ac": torch.cat([a["ac"] for a in ars])}
            ARENAS[fold] = (ar, C.real_scales(list(range(10))))
        else:
            ARENAS[fold] = (C.arena(fold), C.real_scales(C.train_episodes(fold)))
    return ARENAS[fold]


if __name__ == "__main__":
    which = sys.argv[1:] or ["F18", "X5", "REL"]
    mp = os.path.join(C.RESULTS, "models.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
    if "F18" in which:
        setup_arena((1, 8))
        meta["F18"] = run(pools.pool_F18(), lambda k: (1, 8), "F18")
    if "X5" in which:
        for f in C.FOLDS:
            setup_arena(f)
        meta["X5"] = run(pools.pool_X5(), lambda k: tuple(int(c) for c in k.split("_")[1]), "X5")
    if "REL" in which:
        setup_arena("all")
        meta["REL"] = run([("REL", C.W_RELEASED, 5, "REL", "released", None, {})], lambda k: "all", "REL")
    json.dump(meta, open(mp, "w"), indent=1)

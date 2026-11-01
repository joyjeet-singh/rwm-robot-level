"""Score every arm of study FT (prereg/PREREG_FT.md, item 4) and run checks C-FT1 to C-FT4 (item 7).

Arms per parent rid(f, s), s in 0-2 (15 parents): P (the trained model), U30 = compress.unstructured(P, 0.30), U30F
(U30 fine-tuned, runs/ft/ft_u30_<rid>), CTRL (../PerconAI/runs/e3_ft/e3ft_<rid>_w256, read only), and U50, U50F if
results/ft_probe.json's time rule queued U50F. Every model is scored by compute.per_model on its parent's fold arena,
in one run on this machine.

  ./py scripts/ft_eval.py      -> results/raw_FT.npz, results/ft_models.json, results/ft/progress.json

Resumable: models already in results/ft_models.json are skipped; both files are rewritten after every model.
C-FT1: the fine-tuned weights are reloaded; the mask is re-derived by pruning the parent again; every masked entry
must be exactly zero and the count of zeros in the weight matrices must equal the count pruned.
C-FT2: the run record lists no non-finite loss trace.  C-FT3: the run record says the settings equal the source
run's record (ft_unstructured compares them before training; on a difference it records it and does not train, and
the model is then not scored here).  C-FT4: this machine's E100 for P and U30 against
results/raw_X5.npz (cloud), per trajectory, within 1e-3 relative; a failure is reported, the scores are still used.
"""
import json
import os
import platform
import sys
import time

import numpy as np
import scipy
import torch

import common as C
import compress as K
import compute as X
import ft_unstructured as F

PERCONAI = os.path.join(C.PDM, "PerconAI")
NPZ = os.path.join(C.RESULTS, "raw_FT.npz")
META = os.path.join(C.RESULTS, "ft_models.json")
PROG = os.path.join(C.RESULTS, "ft", "progress.json")
FIELDS = ("err", "lam", "rJ") + tuple(f"E{h}" for h in X.HW)
TOL_FT4 = 1e-3


def versions():
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "machine": platform.platform(), "torch_threads": torch.get_num_threads()}


def planned_arms():
    probe = json.load(open(os.path.join(C.RESULTS, "ft_probe.json")))
    return list(probe["arms"])


def ft_record(level, fold, seed):
    p = os.path.join(C.RESULTS, "raw", "ft", f"{F.ft_id(level, fold, seed)}.json")
    return (p, json.load(open(p))) if os.path.exists(p) else (p, None)


def entries(arms):
    """(key, arm, fold, seed, builder) for every model to score; builder() -> (model, path, info)."""
    out = []
    for f in C.FOLDS:
        for s in F.SEEDS:
            rid = F.run_id(f, s)
            par = C.run_path(rid)
            for arm in arms:
                key = f"{arm}_{C.fid(f)}_s{s}"
                if arm == "P":
                    b = (lambda p=par: (C.load(p, 1), p, None))
                elif arm in ("U30", "U50"):
                    b = (lambda p=par, v=arm.lower(): (lambda mi: (mi[0], p, mi[1]))(X.build(p, 1, v)))
                elif arm in ("U30F", "U50F"):
                    lev = F.LEVELS[arm[:3].lower()]
                    wp = os.path.join(C.AIR, "runs", "ft", F.ft_id(lev, f, s), f"weights_{F.ITERS}.pt")
                    b = (lambda p=wp: (C.load(p, 1), p, None))
                elif arm == "CTRL":
                    wp = os.path.join(PERCONAI, "runs", "e3_ft", f"e3ft_{rid}_w256", f"weights_{F.ITERS}.pt")
                    b = (lambda p=wp: (C.load(p, 1), p, None))
                else:
                    raise ValueError(arm)
                out.append((key, arm, f, s, b))
    return out


def check_ft1(level, fold, seed, model):
    """C-FT1 on reloaded fine-tuned weights, with the mask re-derived from the parent."""
    pruned, pinfo = K.unstructured(C.load(C.run_path(F.run_id(fold, seed)), 1), level)
    ft = dict(K.weight_matrices(model))
    held = all(bool((ft[n].detach()[w.detach() == 0] == 0).all()) for n, w in K.weight_matrices(pruned))
    nz = F.zeros(model)
    return {"mask_entries_all_zero": held, "mask_zeros": nz, "pruned_entries": pinfo["pruned_entries"],
            "passed": bool(held and nz == pinfo["pruned_entries"])}


def save(raw, meta):
    tmp = NPZ + ".tmp.npz"
    np.savez_compressed(tmp, **raw)
    os.replace(tmp, NPZ)
    with open(META + ".tmp", "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(META + ".tmp", META)


def load_state():
    if os.path.exists(META) and os.path.exists(NPZ):
        meta = json.load(open(META))
        raw = dict(np.load(NPZ))
        return raw, meta
    return {}, {"models": {}, "checks": {"C-FT1": {}, "C-FT2": {}, "C-FT3": {}, "C-FT4": {}}, "arms": None}


def score_model(m, fold):
    ar, sc = X.setup_arena(fold)
    return X.per_model(m, ar, sc)


def ft4(raw, rawx5):
    """C-FT4 over every P and U30 model scored so far."""
    res = {}
    for f in C.FOLDS:
        for s in F.SEEDS:
            for arm, suf in (("P", ""), ("U30", "__u30")):
                k = f"E100_{arm}_{C.fid(f)}_s{s}"
                if k not in raw:
                    continue
                ref = rawx5[f"E100_A_{C.fid(f)}_s{s}{suf}"].astype(np.float64)
                ours = raw[k].astype(np.float64)
                rel = float(np.max(np.abs(ours - ref) / np.abs(ref)))
                rel_mean = float(abs(ours.mean() - ref.mean()) / abs(ref.mean()))
                res[f"{arm}_{C.fid(f)}_s{s}"] = {"max_rel_per_trajectory": rel, "rel_of_mean": rel_mean,
                                                 "passed": rel <= TOL_FT4}
    return res


def main():
    torch.set_num_threads(2)
    arms = planned_arms()
    raw, meta = load_state()
    meta["arms"] = arms
    meta["versions"] = versions()
    t0 = time.time()
    todo = entries(arms)
    for f in C.FOLDS:
        if f"floor_{C.fid(f)}" not in raw:
            ar, _ = X.setup_arena(f)
            ex, _ = X.arena_extras(ar, C.train_episodes(f))
            raw[f"floor_{C.fid(f)}"] = ex["floor"]
            for h in X.HW:
                raw[f"floor_E{h}_{C.fid(f)}"] = ex[f"floor_E{h}"]
            meta.setdefault("arenas", {})[C.fid(f)] = {"starts": ar["starts"], "episode": ar["episode"]}
    missing = []
    for i, (key, arm, f, s, build) in enumerate(todo):
        if key in meta["models"]:
            continue
        if arm in ("U30F", "U50F"):
            lev = F.LEVELS[arm[:3].lower()]
            rp, rec = ft_record(lev, f, s)
            if rec is None:
                missing.append(key)
                print(f"{key}: no run record {os.path.relpath(rp, C.AIR)}; skipped", flush=True)
                continue
            if not rec.get("trained", True):                       # C-FT3 failed before training: not scored
                if key not in meta["checks"]["C-FT3"]:
                    meta["checks"]["C-FT3"][key] = {"settings_check": rec.get("settings_check"),
                                                    "settings_diff": rec.get("settings_diff"),
                                                    "source_record": rec["source_record"], "passed": False}
                    for c in ("C-FT1", "C-FT2"):
                        meta["checks"][c][key] = {"passed": False, "why": "not trained (C-FT3 failed)"}
                    save(raw, meta)
                print(f"{key}: C-FT3 failed before training; not scored, arm excluded", flush=True)
                continue
        m, path, info = build()
        out = score_model(m, f)
        for k in FIELDS:
            raw[f"{k}_{key}"] = out[k]
        row = {"arm": arm, "parent": F.run_id(f, s), "fold": C.fid(f), "seed": s,
               **{f"E{h}": float(np.mean(out[f"E{h}"])) for h in X.HW},
               "E368": float(np.mean(out["err"])), "lam1": float(np.mean(out["lam"][:, 0])),
               "rJ": float(np.mean(out["rJ"])), "finite": out["finite"],
               "path": os.path.relpath(path, C.PDM), "sha256": C.sha256(path), "variant_info": info,
               "versions": versions()}
        meta["models"][key] = row
        if arm in ("U30F", "U50F"):
            lev = F.LEVELS[arm[:3].lower()]
            assert rec["weights_sha256"] == row["sha256"], f"{key}: weights differ from the run record"
            meta["checks"]["C-FT1"][key] = check_ft1(lev, f, s, m)
            meta["checks"]["C-FT2"][key] = {"non_finite_traces": rec["non_finite_traces"],
                                            "passed": not rec["non_finite_traces"]}
            meta["checks"]["C-FT3"][key] = {"settings_check": rec.get("settings_check"),
                                            "source_record": rec["source_record"],
                                            "passed": rec.get("settings_check") == "equal"
                                            and not rec.get("settings_diff")}
        if arm in ("P", "U30"):
            meta["checks"]["C-FT4"] = ft4(raw, np.load(os.path.join(C.RESULTS, "raw_X5.npz")))
        save(raw, meta)
        os.makedirs(os.path.dirname(PROG), exist_ok=True)
        with open(PROG, "w") as fp:
            json.dump({"done": len(meta["models"]), "of": len(todo), "last": key, "elapsed_s": time.time() - t0,
                       "missing_runs": missing}, fp, indent=1)
        print(f"{key}: E100 {row['E100']:.4f} finite {row['finite']} ({len(meta['models'])}/{len(todo)},"
              f" {time.time() - t0:.0f} s)", flush=True)
    save(raw, meta)
    bad4 = [k for k, v in meta["checks"]["C-FT4"].items() if not v["passed"]]
    print(f"scored {len(meta['models'])}/{len(todo)}; missing runs {missing}; C-FT4 failures {bad4}")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()

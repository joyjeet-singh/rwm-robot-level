"""Option C (C1): checks of trained simulators on the two held-out test sets, with the paper's functions.

    ./py scripts/c/sim_checks.py --sims sim_6M_N8_s0,sim_10k_N8_s0 --out results/c/pilot_sims.json

From the ensemble's mean prediction (lyap.rollout_map, evalh.window_error, lyap.lyapunov, common.monitors): the errors
E1, E8, E24 and E100 against holding the last state, the growth rate lambda1 and the kinematic residual r_J.

The paper's windows are 400 rows (32 history, 368 forecast). Episodes shorter than 400 rows, which on the mixed set
are those that fall early, cannot supply one and are excluded and counted; see optc/DECISIONS.md. Normalisation and the
scales of the kinematic residual come from the simulator's own rung. Per-simulator results are kept in
runs/c/<sim>/checks.json, so a re-run skips finished work; the combined file is written once, never overwritten.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(AIR, "scripts"))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import evalh  # noqa: E402
import lyap  # noqa: E402
import train_sim as TS  # noqa: E402

R, M = C.R, C.M
HORIZONS = (1, 8, 24, 100)
SETS = {"mixed": "test_mixed.npy", "expert": "test_expert.npy"}
SCALE_ROWS = 1_000_000


def load_sim(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = R.load_reference_config(R.repo_paths()["lite"])
    m = M.build_from_config(cfg, ensemble_size=int(ck["ensemble"]))
    m.load_state_dict(ck["model_state_dict"], strict=True)
    m.eval()
    return m, np.asarray(ck["state_mean"]), np.asarray(ck["state_std"]), ck


def rung_scales(rung, data_dir):
    """As common.real_scales: RMS of the joint and attitude residuals over consecutive rows of the rung's episodes."""
    a = np.load(os.path.join(data_dir, f"rung_{rung}.npy"), mmap_mode="r")
    ep = np.load(os.path.join(data_dir, f"rung_{rung}.ep.npy"))
    n = min(len(ep), SCALE_ROWS)
    S = np.asarray(a[:n, :45], dtype=np.float64)
    eG, eJ, eA = C.rule_residuals(S)
    same = ep[1:n] == ep[:n - 1]
    return {"sJ": np.sqrt((eJ[same] ** 2).mean(0)), "sA": np.sqrt((eA[same] ** 2).mean(0)), "n_pairs": int(same.sum())}


def arena(path, mean, std):
    """Non-overlapping 400-row windows from the episodes long enough to hold one."""
    a = np.load(path, mmap_mode="r")
    ep = np.asarray(a[:, 66]).astype(np.int64)
    cuts = np.flatnonzero(ep[1:] != ep[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(ep)]
    win = [s + k * C.LEN for s, e in zip(starts, ends) for k in range((e - s) // C.LEN)]
    short = int(sum((e - s) < C.LEN for s, e in zip(starts, ends)))
    if not win:
        raise SystemExit(f"{path}: no episode holds a {C.LEN}-row window")
    idx = np.asarray(win)[:, None] + np.arange(C.LEN)[None, :]
    raw = np.asarray(a[idx], dtype=np.float64)
    st = torch.as_tensor((raw[:, :, :45] - mean) / std, dtype=torch.float32)
    ac = torch.as_tensor(raw[:, :, 45:57], dtype=torch.float32)
    return {"st": st, "ac": ac, "windows": len(win), "episodes": len(starts), "episodes_too_short": short}


def check_set(m, path, mean, std, sc):
    ar = arena(path, mean, std)
    st, ac = ar["st"], ar["ac"]
    out = {k: ar[k] for k in ("windows", "episodes", "episodes_too_short")}
    with torch.no_grad():
        for h in HORIZONS:
            e, floor = evalh.window_error(m, st, ac, h)
            out[f"E{h}"], out[f"floor_E{h}"] = float(np.mean(e)), float(np.mean(floor))
        pred = lyap.rollout_map(m, st, ac)
    P = pred.numpy() * std + mean
    S = st.numpy() * std + mean
    _, rJ, _ = C.monitors(P, sc)
    _, rJ_true, _ = C.monitors(S, sc)
    out["rJ"], out["rJ_true"] = float(np.mean(rJ)), float(np.mean(rJ_true))
    lam, _ = lyap.lyapunov(copy.deepcopy(m), st, ac, p=4)         # float64; a copy so m stays float32
    out["lambda1"] = float(np.mean(lam[:, 0]))
    return out


def check_sim(name, data_dir, runs, progress=True):
    run_dir = os.path.join(runs, name)
    part = os.path.join(run_dir, "checks.json")
    if os.path.exists(part):
        return json.load(open(part))
    m, mean, std, ck = load_sim(os.path.join(run_dir, "final.pt"))
    sc = rung_scales(ck["rung"], data_dir)
    res = {"name": name, "rung": ck["rung"], "N": ck["N"], "seed": ck["seed"], "steps": ck["iter"],
           "scale_pairs": sc["n_pairs"], "sets": {}}
    t0 = time.time()
    for k, f in SETS.items():
        res["sets"][k] = check_set(m, os.path.join(data_dir, f), mean, std, sc)
        if progress:
            TS.progress(f"checks {name}: {k}", 1, 1)
    res["seconds"] = time.time() - t0
    with open(part + ".tmp", "w") as fh:
        json.dump(res, fh, indent=1)
    os.replace(part + ".tmp", part)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-dir", default=TS.DATA_DIR)
    ap.add_argument("--runs", default=TS.RUNS)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-progress", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    res = {n: check_sim(n, a.data_dir, a.runs, not a.no_progress) for n in a.sims.split(",")}
    if a.out:
        if os.path.exists(a.out):
            print(f"[sim_checks] {a.out} exists, kept")
            return
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        rec = {"what": "Option C pilot simulators, checks on the held-out MuJoCo test sets (hand-built calibrated MuJoCo "
                       "ANYmal D with a fitted PD motor; not Isaac Lab, not hardware)", "sims": res}
        with open(a.out + ".tmp", "w") as fh:
            json.dump(rec, fh, indent=1)
        os.replace(a.out + ".tmp", a.out)
    print(json.dumps({n: {k: {x: round(v[x], 4) for x in ("E8", "floor_E8", "E24", "floor_E24")}
                          for k, v in r["sets"].items()} for n, r in res.items()}))


if __name__ == "__main__":
    main()

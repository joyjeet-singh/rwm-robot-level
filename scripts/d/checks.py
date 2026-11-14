"""Option D (D0): the checks of one simulator, in common units, on both MuJoCo test sets.  ./py

    ./py scripts/d/checks.py --sim REL                    # -> results/d/checks/REL.json (written once)
    ./py scripts/d/checks.py --sim REL --arena recorded --out .cache/d_checks_REL_recorded.json   # the paper's arena

OPTIOND_PLAN.md, "Checks, in common units". With the paper's functions (lyap.rollout_map, evalh.window_error,
lyap.lyapunov, common.monitors):
  * windows: the paper's 400-row windows (32 history rows, 368 forecast steps), non-overlapping, from the test-set
    episodes long enough to hold one (scripts/c/sim_checks.arena); the expert test set is primary, the mixed secondary;
  * rollout: from the mean of the networks (the paper's map);
  * common units: every prediction is converted back to physical units and divided by the authors' scales. For a
    simulator in the authors' scales (REL, its copies, the rebuilds, the ladder) the model is used as it is, so the
    errors are the paper's own. For Option C's four (own scales) an exactly equivalent model is used whose inputs and
    outputs are in the authors' units: the change of units is folded into the first GRU layer's input weights and the
    last layer of each state head (`common_units`); the test checks it against converting the predictions;
  * errors E1, E8, E24, E100, E368 (each window reads its 32 true rows, then runs on its own predictions), and the
    hold-the-last-state floor in the same units;
  * kinematic residual r_J (and the attitude residual r_A, exploratory): common.monitors on the de-normalised
    predictions, scaled by the residuals of the expert test set's own true rows (consecutive rows of one episode);
  * growth rate lambda1: lyap.lyapunov (p = 4, float64, seed 0) in the simulator's own coordinates.

`--arena recorded` runs the paper's recorded arena instead (20 trajectories of the ten recorded episodes, scales
common.real_scales(0..9)), as scripts/compute.py did for REL; used by the tests.
The world behind the test sets is the hand-built calibrated MuJoCo ANYmal D with a fitted PD motor; not Isaac Lab, not
hardware. A result file is never overwritten.
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
SCRIPTS = os.path.dirname(HERE)
AIR = os.path.dirname(SCRIPTS)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(SCRIPTS, "c"))
import sims  # noqa: E402
import common as C  # noqa: E402
import evalh  # noqa: E402
import lyap  # noqa: E402
import sim_checks as SC  # noqa: E402  (Option C's arena)
import dprogress  # noqa: E402

DATA_DIR = os.path.join(AIR, "optc", "data")
SETS = {"expert": "test_expert.npy", "mixed": "test_mixed.npy"}
HORIZONS = (1, 8, 24, 100, 368)
OUT_DIR = os.path.join(AIR, "results", "d", "checks")


def common_units(m, mean, std):
    """A copy of m whose states are in the authors' units: s_own = (s_auth * sa + ma - mo) / so on input,
    delta_auth = delta_own * so / sa on output. Returns m itself when its scales are the authors'."""
    am, asd = sims.authors_scales()
    mean, std = np.asarray(mean, np.float64), np.asarray(std, np.float64)
    if np.array_equal(mean, am) and np.array_equal(std, asd):
        return m
    m2 = copy.deepcopy(m)
    a = torch.as_tensor(asd / std)                      # d s_own / d s_auth
    b = torch.as_tensor((am - mean) / std)              # s_own at s_auth = 0
    r = torch.as_tensor(std / asd)                      # d delta_auth / d delta_own
    with torch.no_grad():
        rnn = m2.state_base.memory.rnn
        W = rnn.weight_ih_l0.double()
        rnn.bias_ih_l0.copy_((rnn.bias_ih_l0.double() + W[:, :45] @ b).float())
        W[:, :45] = W[:, :45] * a[None, :]
        rnn.weight_ih_l0.copy_(W.float())
        for hd in m2.state_heads:
            last = hd.state_mean_layers[-1]
            last.weight.copy_((last.weight.double() * r[:, None]).float())
            last.bias.copy_((last.bias.double() * r).float())
    m2.eval()
    return m2


def expert_scales():
    """RMS of the joint and attitude residuals over consecutive true rows of one episode, expert test set."""
    a = np.load(os.path.join(DATA_DIR, SETS["expert"]), mmap_mode="r")
    S = np.asarray(a[:, :45], dtype=np.float64)
    ep = np.asarray(a[:, 66]).astype(np.int64)
    _, eJ, eA = C.rule_residuals(S)
    same = ep[1:] == ep[:-1]
    return {"sJ": np.sqrt((eJ[same] ** 2).mean(0)), "sA": np.sqrt((eA[same] ** 2).mean(0)), "n_pairs": int(same.sum())}


def recorded_arena():
    """The paper's recorded arena for REL (scripts/compute.py, setup_arena('all'))."""
    ars = [C.arena(f) for f in C.FOLDS]
    st = torch.cat([a["st"] for a in ars])
    ac = torch.cat([a["ac"] for a in ars])
    return {"st": st, "ac": ac, "windows": int(st.shape[0])}, C.real_scales(list(range(10)))


def measure(m, mean, std, st_a, ac, st_o, sc, lam=True):
    """The checks on one arena. st_a: states in the authors' units; st_o: in the simulator's own."""
    am, asd = sims.authors_scales()
    mc = common_units(m, mean, std)
    out, per = {}, {}
    with torch.no_grad():
        for h in HORIZONS:
            e, floor = evalh.window_error(mc, st_a, ac, h)
            out[f"E{h}"], out[f"floor_E{h}"] = float(np.mean(e)), float(np.mean(floor))
            per[f"E{h}"] = [float(x) for x in e]
        pred = lyap.rollout_map(mc, st_a, ac)
    P = pred.numpy().astype(np.float64) * asd + am
    S = st_a.numpy().astype(np.float64) * asd + am
    _, rJ, rA = C.monitors(P, sc)
    _, rJt, rAt = C.monitors(S, sc)
    out.update(rJ=float(np.mean(rJ)), rA=float(np.mean(rA)), rJ_true=float(np.mean(rJt)), rA_true=float(np.mean(rAt)))
    per["rJ"] = [float(x) for x in rJ.mean(1)]
    if lam:
        lm, _ = lyap.lyapunov(copy.deepcopy(m), st_o, ac, p=4)        # float64 on a copy; own coordinates
        out["lambda1"] = float(np.mean(lm[:, 0]))
        per["lambda1"] = [float(x) for x in lm[:, 0]]
    out["finite"] = bool(all(np.all(np.isfinite(v)) for v in list(out.values()) + list(per.values())
                             if not isinstance(v, bool)))
    out["per_window"] = per
    return out


def test_set(name, mean, std, max_windows=None):
    """(states in the authors' units, actions, states in own units, info) of a MuJoCo test set."""
    am, asd = sims.authors_scales()
    path = os.path.join(DATA_DIR, SETS[name])
    ar_a = SC.arena(path, am, asd)
    ar_o = SC.arena(path, mean, std)
    k = ar_a["windows"] if max_windows is None else min(max_windows, ar_a["windows"])
    info = {x: ar_a[x] for x in ("windows", "episodes", "episodes_too_short")}
    info["windows_used"] = k
    return ar_a["st"][:k], ar_a["ac"][:k], ar_o["st"][:k], info


def write_once(path, obj):
    if os.path.exists(path):
        print(f"[checks] {os.path.relpath(path, AIR)} exists, kept")
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(path + ".tmp", path)
    return True


def run(name, arena="test", out=None, max_windows=None, sets=tuple(SETS), job="record"):
    out = out or os.path.join(OUT_DIR, f"{name}.json")
    if os.path.exists(out):
        print(f"[checks] {name}: {os.path.relpath(out, AIR)} exists, skipped")
        with open(out) as fh:
            return json.load(fh)
    t0 = time.time()
    m, mean, std, rec = sims.load(name)
    res = {"what": "Option D checks in common units (errors in the authors' scales; lambda1 in the simulator's own "
                   "coordinates). Test sets from the hand-built calibrated MuJoCo ANYmal D with a fitted PD motor; not "
                   "Isaac Lab, not hardware", "simulator": rec, "arena": arena, "sets": {}}
    if arena == "recorded":
        ar, sc = recorded_arena()
        st = ar["st"][:max_windows] if max_windows else ar["st"]
        ac = ar["ac"][:max_windows] if max_windows else ar["ac"]
        if rec["scales"] != "authors":
            raise SystemExit("the recorded arena is for simulators in the authors' scales")
        r = measure(m, mean, std, st, ac, st, sc)
        r.update(windows_used=int(st.shape[0]))
        res["sets"]["recorded"] = r
    else:
        sc = expert_scales()
        res["residual_scales"] = {"from": "expert test set, consecutive true rows of one episode",
                                  "n_pairs": sc["n_pairs"], "sJ": sc["sJ"].tolist(), "sA": sc["sA"].tolist()}
        for s in sets:
            st_a, ac, st_o, info = test_set(s, mean, std, max_windows)
            r = measure(m, mean, std, st_a, ac, st_o, sc)
            r.update(info)
            res["sets"][s] = r
            dprogress.write(job, f"checks {name}: {s}")
    res["seconds"] = time.time() - t0
    res["torch"] = torch.__version__
    write_once(out, res)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="one name, or a comma list")
    ap.add_argument("--arena", choices=("test", "recorded"), default="test")
    ap.add_argument("--out", default=None, help="output file (one simulator only; tests write under .cache/)")
    ap.add_argument("--max-windows", type=int, default=None, help="tests and rehearsals only (with --out)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--job", default="record")
    a = ap.parse_args(argv)
    if (a.max_windows or a.arena == "recorded") and not a.out:
        ap.error("--max-windows and --arena recorded need --out (they are not the registered checks)")
    torch.set_num_threads(a.threads)
    for n in a.sim.split(","):
        r = run(n, a.arena, a.out, a.max_windows, job=a.job)
        print(json.dumps({"sim": n, **{s: {k: round(v[k], 4) for k in ("E1", "E8", "E100", "E368", "rJ", "lambda1")}
                                       for s, v in r["sets"].items()}}))


if __name__ == "__main__":
    main()

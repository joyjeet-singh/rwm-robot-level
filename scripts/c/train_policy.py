"""Option C (C2): train one walking policy inside a five-network simulator, through Step 2's route.

    ./py scripts/c/train_policy.py --sim sim_6M_N8_s0 --seed 0 --threads 2     # -> runs/c/sim_6M_N8_s0__p0/actor_i*.pt

Wraps scripts/s2/train_policy.py (and through it the authors' lite code) unchanged, by moving its outputs under runs/c/ and
results/c/ and adding what OPTIONC_PLAN.md ("Policies") asks for:
  * the simulator is runs/c/<sim>/final.pt, through common.load and the lite adapter;
  * the normalisation is the simulator's own (the mean and spread of its rung, stored in final.pt); actions stay raw, as
    in training and in the lite config;
  * start states are 32-step windows from the simulator's own rung, upright throughout (tilt under 30 degrees) and ending
    at least 1 s before any fall in their episode, from at most 100,000 rows (whole episodes in a fixed random order that
    depends only on the rung; see optc/DECISIONS.md, C2). They are normalised exactly as train_sim.py's loader does;
  * `--uncertainty-weight w` sets the lite reward's penalty to -w (the lite default is w = 1.0);
  * the actor is saved after exactly 500, 1,000, 1,500 and 2,000 finished iterations, as runs/c/<policy>/actor_i<n>.pt.

Names: <sim>__p<seed> at weight 1.0, <sim>__w<w>__p<seed> otherwise. As in Step 2, a policy whose result and actor exist is
skipped, and an interrupted policy restarts from iteration 0, so every policy is one uninterrupted, seeded run.
Names starting with `_test` may use other sizes and write no result file (tests only).
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AIR = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True

_spec = importlib.util.spec_from_file_location("s2_train_policy", os.path.join(AIR, "scripts", "s2", "train_policy.py"))
TP = importlib.util.module_from_spec(_spec)
sys.modules["s2_train_policy"] = TP
_spec.loader.exec_module(TP)                     # sets the lite paths (run_lite) and its torch.load
RL, C = TP.RL, TP.C
import torch  # noqa: E402

RUNS = os.path.join(AIR, "runs", "c")
RES = os.path.join(AIR, "results", "c", "train")
PROGRESS = os.path.join(AIR, "results", "c", "progress.json")
LITE_ROOT = os.path.join(RUNS, "lite")             # the lite log of a policy: LITE_ROOT/runs/lite/s2_<policy>/
DATA_DIR = os.path.join(AIR, "optc", "data")
ITERATIONS = 2000
SAVES = (500, 1000, 1500, 2000)
HISTORY = 32
FALL_MARGIN = 50                                   # steps: 1 s at 50 Hz
MAX_ROWS = 100_000
MAX_TILT_DEG = 30.0
START_SEED = 20276
RUNG_CODES = {"10k": 0, "100k": 1, "1M": 2, "6M": 3, "d6": 4}
TP.RUNS, TP.RES, TP.PROGRESS = RUNS, RES, PROGRESS
TP.AIR = RL.AIR = LITE_ROOT                        # only the lite log directory is built from these after sim_row
TP.ITERATIONS = ITERATIONS


def policy_name(sim, seed, weight=1.0):
    return f"{sim}__p{seed}" if weight == 1.0 else f"{sim}__w{weight:g}__p{seed}"


def sim_path(sim, runs=RUNS):
    p = os.path.join(runs, sim, "final.pt")
    if not os.path.exists(p):
        raise SystemExit(f"{p} does not exist")
    return p


def sim_row_of(sim, runs=RUNS):
    """A row with the fields Step 2's route reads (scripts/s2/train_policy.py: sim_row, build_sim, main)."""
    path = sim_path(sim, runs)
    return {"key": sim, "path": path, "sha256": C.sha256(path), "ensemble": 5, "variant": None, "variant_info": None,
            "arena": "all", "pool": "C", "group": "Option C simulator"}


def choose_episodes(rung, ep, max_rows=MAX_ROWS):
    """Row ranges (start, end) of the rung's episodes used for start states: all, or at most max_rows rows of them.

    Episodes are taken in a fixed random order that depends only on the rung, until the next one would not fit."""
    cuts = np.flatnonzero(ep[1:] != ep[:-1]) + 1
    starts, ends = np.r_[0, cuts], np.r_[cuts, len(ep)]
    assert len(np.unique(ep[starts])) == len(starts), "episodes are not contiguous"
    if len(ep) <= max_rows:
        return list(zip(starts.tolist(), ends.tolist()))
    order = np.random.default_rng([START_SEED, RUNG_CODES[rung]]).permutation(len(starts))
    out, total = [], 0
    for i in order:
        n = int(ends[i] - starts[i])
        if total + n > max_rows:
            break
        out.append((int(starts[i]), int(ends[i])))
        total += n
    return sorted(out)


def valid_starts(a, fall, spans, history=HISTORY, margin=FALL_MARGIN):
    """Rows s at which a start window begins: upright for all its rows, ending at least `margin` steps before any fall."""
    cos_max = np.cos(np.radians(MAX_TILT_DEG))
    out = []
    for s, e in spans:
        L = e - s
        fell = bool(fall[e - 1])
        assert not fall[s:e - 1].any(), "a fall before an episode's last row"
        last = L - history - margin if fell else L - history      # last allowed offset, the fall being at row L - 1
        if last < 0:
            continue
        g = np.asarray(a[s:e, 6:9], dtype=np.float64)
        up = (-g[:, 2] / np.linalg.norm(g, axis=1)) > cos_max
        bad = np.cumsum(np.r_[0, ~up])
        ok = bad[history:history + last + 1] - bad[:last + 1] == 0
        out.append(s + np.flatnonzero(ok))
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def build_starts(rung, mean, std, data_dir=DATA_DIR, chunk=20_000):
    """Normalised start windows: state (N, 32, 45) and action (N, 32, 12) float32, and their rung rows."""
    a = np.load(os.path.join(data_dir, f"rung_{rung}.npy"), mmap_mode="r")
    ep = np.load(os.path.join(data_dir, f"rung_{rung}.ep.npy"))
    fall = np.load(os.path.join(data_dir, f"rung_{rung}.fall.npy"))
    spans = choose_episodes(rung, ep)
    starts = valid_starts(a, fall, spans)
    if len(starts) == 0:
        raise SystemExit(f"rung {rung}: no start window passes the rule")
    assert len(starts) <= MAX_ROWS, len(starts)
    mean, std = np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)
    st, ac = np.empty((len(starts), HISTORY, 45), dtype=np.float32), np.empty((len(starts), HISTORY, 12), dtype=np.float32)
    offs = np.arange(HISTORY)
    for i in range(0, len(starts), chunk):
        rows = np.asarray(a[starts[i:i + chunk][:, None] + offs], dtype=np.float64)
        st[i:i + chunk] = (rows[:, :, :45] - mean) / std                 # as train_sim.RungWindows.batch
        ac[i:i + chunk] = rows[:, :, 45:57]
    info = {"rung": rung, "rows_in_rung": int(len(ep)), "episodes_used": len(spans),
            "rows_used": int(sum(e - s for s, e in spans)), "windows": int(len(starts)), "max_rows": MAX_ROWS,
            "history": HISTORY, "tilt_under_deg": MAX_TILT_DEG, "steps_before_fall": FALL_MARGIN}
    return torch.from_numpy(st), torch.from_numpy(ac), starts, info


def install_starts(dataset, st, ac, mean, std):
    """Replace the lite dataset's windows and normalisation; the env samples starts and (de)normalises through it."""
    n = len(st)
    dataset.state_data, dataset.action_data = st, ac
    dataset.extension_data = torch.zeros(n, HISTORY, 0)
    dataset.contact_data = torch.zeros(n, HISTORY, 0)
    dataset.termination_data = torch.zeros(n, HISTORY, 0)
    dataset.state_data_mean = torch.as_tensor(np.asarray(mean), dtype=torch.float32)
    dataset.state_data_std = torch.as_tensor(np.asarray(std), dtype=torch.float32)
    assert (dataset.action_data_mean == 0).all() and (dataset.action_data_std == 1).all(), "actions must stay raw"


def patch_lite(weight, sim, saves, out_dir):
    """Hook the lite modules: penalty weight, the simulator's start states and normalisation, and the exact-count saves."""
    import policy_training as PT
    import train as T
    ck = torch.load(sim_path(sim), map_location="cpu", weights_only=False)
    rung = ck["rung"]
    st, ac, _, info = build_starts(rung, ck["state_mean"], ck["state_std"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "starts.json"), "w") as fh:
        json.dump(info, fh)
    print(f"[train_policy] start states: {json.dumps(info)}")

    orig_resolve = T.resolve_task_config

    def resolve(task):
        cfg = orig_resolve(task)
        cfg.environment_config.uncertainty_penalty_weight = -abs(weight)
        return cfg
    T.resolve_task_config = resolve

    orig_data = T.ModelBasedExperiment.prepare_data

    def prepare_data(self, *args, **kw):
        orig_data(self, *args, **kw)
        install_starts(self.dataset, st, ac, ck["state_mean"], ck["state_std"])
    T.ModelBasedExperiment.prepare_data = prepare_data

    orig_log = PT.PolicyTraining.log

    def log(self, locs, *args, **kw):
        out = orig_log(self, locs, *args, **kw)
        n = int(locs["it"]) + 1                     # the log comes after this iteration's update
        if n in saves:
            sd = {k: v.clone() for k, v in self.alg.policy.state_dict().items() if k.startswith("actor.")}
            path = os.path.join(out_dir, f"actor_i{n}.pt")
            torch.save({"model_state_dict": sd, "iter": n - 1, "source": "lite log hook"}, path + ".tmp")
            os.replace(path + ".tmp", path)
        return out
    PT.PolicyTraining.log = log
    return info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="a simulator under runs/c/, e.g. sim_6M_N8_s0")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--uncertainty-weight", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--job", default="pilot")
    ap.add_argument("--name", default=None, help="`_test*` for tests")
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    ap.add_argument("--num_envs", type=int, default=TP.NUM_ENVS)
    ap.add_argument("--saves", default=None, help="comma list (tests only)")
    a = ap.parse_args(argv)
    name = a.name or policy_name(a.sim, a.seed, a.uncertainty_weight)
    test = name.startswith("_test")
    if not test and (a.iterations, a.num_envs, a.saves) != (ITERATIONS, TP.NUM_ENVS, None):
        raise SystemExit("the plan fixes 2,000 iterations, 4,096 robots and the saves; other sizes are for `_test*` names")
    saves = tuple(int(x) for x in a.saves.split(",")) if a.saves else SAVES
    out_dir = os.path.join(RUNS, name)
    done = os.path.join(RES, name + ".json")
    if not test and os.path.exists(done):
        print(f"[train_policy] {name}: done already, skipped")
        return json.load(open(done))

    row = sim_row_of(a.sim)
    TP.sim_row = lambda sim: row
    info = patch_lite(a.uncertainty_weight, a.sim, set(saves), out_dir)
    argv2 = ["--sim", a.sim, "--seed", str(a.seed), "--route", "rebuilt", "--job", a.job, "--threads", str(a.threads),
             "--name", name, "--iterations", str(a.iterations), "--num_envs", str(a.num_envs)]
    t0 = time.time()
    rec = TP.main(argv2)
    missing = [n for n in saves if not os.path.exists(os.path.join(out_dir, f"actor_i{n}.pt"))]
    if missing:
        raise SystemExit(f"{name}: no actor saved at iterations {missing}")
    extra = {"uncertainty_weight": a.uncertainty_weight, "start_states": info, "saves": list(saves),
             "actors_sha256": {str(n): C.sha256(os.path.join(out_dir, f"actor_i{n}.pt")) for n in saves},
             "wall_seconds_wrapper": time.time() - t0}
    with open(os.path.join(out_dir, "wrapper.json"), "w") as fh:
        json.dump(extra, fh, indent=1)
    return {**rec, **extra}


if __name__ == "__main__":
    main()

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

Amendment 1 (C2R; OPTIONC_PLAN.md, "Amendment 1") adds, for every new policy:
  * start states only from the authors'-policy episodes (source 0, no added noise) of the rung, same rule otherwise;
  * a noise floor: the policy's exploration noise is clamped to 0.05 or above after every optimiser step (the first
    pilot's weight-2.0 run crashed inside an update when it went negative);
  * `--penalty cal|<w>`: the uncertainty weight, from results/c/calibration.json by simulator name (lambda_s), or given;
  * `--fall-cost c`: a predicted fall costs a one-time c, in the units of the per-step rewards (the lite reward has no
    termination term), subtracted at the step the episode ends; this departs from the authors' reward;
  * `--max-iters n` (500, 1,000, 1,500 or 2,000), with saves every 500 iterations;
  * an early stop: after iteration 300, an arm whose mean imagined episode length (the lite log's
    Train/mean_episode_length) stays under a quarter of the limit for 100 iterations in a row is stopped and recorded
    as "collapsed"; its saves so far are kept.
  * every iteration also logs the share of imagined episodes that ended in a predicted fall.

Names: <sim>__<tag>__p<seed>, tag "cal" or "w<w>", with "_fall<c>" added when a fall cost is set (Amendment 1); the
first pilot's names were <sim>__p<seed> (weight 1.0) and <sim>__w<w>__p<seed>. As in Step 2, a policy whose result exists
is skipped, and an interrupted policy restarts from iteration 0, so every policy is one uninterrupted, seeded run.
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
ALLOWED_ITERS = (500, 1000, 1500, 2000)            # --max-iters (Amendment 1: 500 to 2,000, saves every 500)
SAVE_EVERY = 500
NOISE_FLOOR = 0.05
START_SOURCE = 0                                   # the authors' policy, no added noise (gen_data.sources)
SOURCE_COL = 67
COLLAPSE_AFTER, COLLAPSE_RUN, COLLAPSE_SHARE = 300, 100, 0.25
CALIBRATION = os.path.join(AIR, "results", "c", "calibration.json")
CALIBRATION_2S = os.path.join(AIR, "results", "c", "calibration_2S.json")
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


def choose_episodes(rung, ep, max_rows=MAX_ROWS, src=None, source=None):
    """Row ranges (start, end) of the rung's episodes used for start states: all, or at most max_rows rows of them.

    Episodes are taken in a fixed random order that depends only on the rung, until the next one would not fit.
    With `source` (Amendment 1), only the episodes of that source are eligible; `src` is the source id of each episode's
    first row, aligned with the episodes, and the order then depends on the rung and the source."""
    cuts = np.flatnonzero(ep[1:] != ep[:-1]) + 1
    starts, ends = np.r_[0, cuts], np.r_[cuts, len(ep)]
    assert len(np.unique(ep[starts])) == len(starts), "episodes are not contiguous"
    seed = [START_SEED, RUNG_CODES[rung]]
    if source is not None:
        keep = np.asarray(src) == source
        if not keep.any():
            raise SystemExit(f"rung {rung}: no episode of source {source}")
        starts, ends = starts[keep], ends[keep]
        seed = [START_SEED, RUNG_CODES[rung], 100 + int(source)]
    if int((ends - starts).sum()) <= max_rows:
        return list(zip(starts.tolist(), ends.tolist()))
    order = np.random.default_rng(seed).permutation(len(starts))
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


def episode_sources(a, ep):
    """The source id of each episode (from its first row), checked against its last row."""
    cuts = np.flatnonzero(ep[1:] != ep[:-1]) + 1
    first, last = np.r_[0, cuts], np.r_[cuts, len(ep)] - 1
    s0, s1 = np.asarray(a[first, SOURCE_COL]), np.asarray(a[last, SOURCE_COL])
    assert np.array_equal(s0, s1), "an episode changes source"
    return s0.astype(np.int64)


def build_starts(rung, mean, std, data_dir=DATA_DIR, chunk=20_000, source=None):
    """Normalised start windows: state (N, 32, 45) and action (N, 32, 12) float32, and their rung rows.

    `source=0` keeps only the authors'-policy episodes (Amendment 1, change 3)."""
    a = np.load(os.path.join(data_dir, f"rung_{rung}.npy"), mmap_mode="r")
    ep = np.load(os.path.join(data_dir, f"rung_{rung}.ep.npy"))
    fall = np.load(os.path.join(data_dir, f"rung_{rung}.fall.npy"))
    src = episode_sources(a, ep) if source is not None else None
    spans = choose_episodes(rung, ep, src=src, source=source)
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
            "history": HISTORY, "tilt_under_deg": MAX_TILT_DEG, "steps_before_fall": FALL_MARGIN,
            "source": source, "window_sources": {str(k): int(v) for k, v in
                                                 zip(*np.unique(np.asarray(a[starts, SOURCE_COL]).astype(np.int64),
                                                                return_counts=True))}}
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


class Collapsed(Exception):
    """Raised from the log hook when the early-stop rule fires (Amendment 1, change 6)."""

    def __init__(self, n, record):
        super().__init__(f"collapsed at iteration {n}")
        self.n, self.record = n, record


def collapse_at(lengths, limit, after=COLLAPSE_AFTER, run=COLLAPSE_RUN, share=COLLAPSE_SHARE):
    """The finished-iteration count n at which the early stop fires, or None.

    lengths[n - 1] is the mean imagined episode length logged after iteration n (None when no episode had ended yet).
    It fires at the first n for which iterations n - run + 1 .. n all come after iteration `after` and all have a mean
    episode length under share x limit. So the earliest stop is at n = after + run (optc/DECISIONS.md, C2R)."""
    thr = share * limit
    streak = 0
    for i, L in enumerate(lengths):
        n = i + 1
        if n <= after:
            continue
        streak = streak + 1 if (L is not None and L < thr) else 0
        if streak >= run:
            return n
    return None


def resolve_penalty(penalty, sim, paths=(CALIBRATION, CALIBRATION_2S)):
    """(weight, how): 'cal' reads lambda_s for this simulator from the calibration file that has it (results/c/
    calibration.json, or calibration_2S.json for the simulators retrained with 2 x S steps); else a number >= 0."""
    if penalty == "cal":
        found = [(p, json.load(open(p))["simulators"][sim]) for p in paths
                 if os.path.exists(p) and sim in json.load(open(p))["simulators"]]
        if len(found) != 1 or found[0][1].get("lambda") is None:
            raise SystemExit(f"{sim}: lambda must be in exactly one calibration file of {paths}; found {len(found)}")
        return float(found[0][1]["lambda"]), f"calibrated ({os.path.relpath(found[0][0], AIR)})"
    w = float(penalty)
    if not (w >= 0 and np.isfinite(w)):
        raise SystemExit(f"--penalty must be 'cal' or a number >= 0, not {penalty!r}")
    return w, "given"


def recipe_name(sim, seed, penalty, fall_cost):
    tag = "cal" if penalty == "cal" else f"w{float(penalty):g}"
    if fall_cost:
        tag += f"_fall{fall_cost:g}"
    return f"{sim}__{tag}__p{seed}"


def patch_noise_floor(floor=NOISE_FLOOR, test_init_std=None):
    """Clamp the policy's exploration noise to `floor` or above after every optimiser step (Amendment 1, change 4)."""
    from rsl_rl.algorithms import PPO
    if getattr(PPO, "_c2r_floor", None) is not None:
        PPO._c2r_floor, PPO._c2r_test_std = floor, test_init_std
        return
    orig_init = PPO.__init__

    def clamp(pol):
        with torch.no_grad():
            if getattr(pol, "noise_std_type", "scalar") == "log":
                pol.log_std.clamp_(min=float(np.log(PPO._c2r_floor)))
            else:
                pol.std.clamp_(min=PPO._c2r_floor)

    def init(self, *args, **kw):
        orig_init(self, *args, **kw)
        step = self.optimizer.step

        def stepped(*a, **k):
            out = step(*a, **k)
            clamp(self.policy)
            return out
        self.optimizer.step = stepped
        clamp(self.policy)
        if PPO._c2r_test_std is not None:            # tests only: start below the floor
            with torch.no_grad():
                self.policy.std.fill_(PPO._c2r_test_std)
    PPO.__init__ = init
    PPO._c2r_floor = floor
    PPO._c2r_test_std = test_init_std


def patch_fall_cost(cost, counter):
    """Subtract `cost` once, at the step an imagined episode ends in a predicted fall; count falls and episode ends."""
    import envs.base as B
    if getattr(B.BaseEnv, "_c2r_orig_post", None) is None:
        B.BaseEnv._c2r_orig_post = B.BaseEnv._post_imagination_step
    orig = B.BaseEnv._c2r_orig_post

    def post(self):
        flags = self.termination_flags
        fell = flags.clone().bool() if flags is not None else torch.zeros(self._num_envs, dtype=torch.bool)
        rewards, dones, extras = orig(self)
        assert not (fell & (dones == 0)).any(), "a predicted fall that does not end its episode"
        if cost:
            rewards = rewards - cost * fell.to(rewards.dtype)
        counter["falls"] += int(fell.sum())
        counter["ends"] += int((dones > 0).sum())
        counter["charged"] += float(cost) * int(fell.sum())
        return rewards, dones, extras
    B.BaseEnv._post_imagination_step = post


def patch_lite(weight, sim, saves, out_dir, fall_cost=0.0, source=START_SOURCE, floor=NOISE_FLOOR, early_stop=True,
               collapse=(COLLAPSE_AFTER, COLLAPSE_RUN, COLLAPSE_SHARE), test_init_std=None):
    """Hook the lite modules: penalty weight, the simulator's start states and normalisation, the exact-count saves,
    and (Amendment 1) the noise floor, the fall cost, the fall log and the early stop."""
    import policy_training as PT
    import train as T
    ck = torch.load(sim_path(sim), map_location="cpu", weights_only=False)
    rung = ck["rung"]
    st, ac, _, info = build_starts(rung, ck["state_mean"], ck["state_std"], source=source)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "starts.json"), "w") as fh:
        json.dump(info, fh)
    print(f"[train_policy] start states: {json.dumps(info)}")

    orig_resolve = T.resolve_task_config
    limit = [None]

    def resolve(task):
        cfg = orig_resolve(task)
        cfg.environment_config.uncertainty_penalty_weight = -abs(weight)
        limit[0] = int(cfg.environment_config.max_episode_length)
        return cfg
    T.resolve_task_config = resolve

    orig_data = T.ModelBasedExperiment.prepare_data

    def prepare_data(self, *args, **kw):
        orig_data(self, *args, **kw)
        install_starts(self.dataset, st, ac, ck["state_mean"], ck["state_std"])
    T.ModelBasedExperiment.prepare_data = prepare_data

    patch_noise_floor(floor, test_init_std)
    counter = {"falls": 0, "ends": 0, "charged": 0.0}
    patch_fall_cost(fall_cost, counter)
    lengths = []

    orig_log = PT.PolicyTraining.log

    def log(self, locs, *args, **kw):
        n = int(locs["it"]) + 1                     # the log comes after this iteration's update
        steps = self.num_steps_per_env * self.env.num_envs
        self.log_dict.update({
            "Imagination/predicted_falls": counter["falls"],
            "Imagination/episode_ends": counter["ends"],
            "Imagination/predicted_fall_share": counter["falls"] / counter["ends"] if counter["ends"] else 0.0,
            "Imagination/fall_cost_per_step": -counter["charged"] / steps,
        })
        counter.update(falls=0, ends=0, charged=0.0)
        out = orig_log(self, locs, *args, **kw)
        if n in saves:
            sd = {k: v.clone() for k, v in self.alg.policy.state_dict().items() if k.startswith("actor.")}
            path = os.path.join(out_dir, f"actor_i{n}.pt")
            torch.save({"model_state_dict": sd, "iter": n - 1, "source": "lite log hook"}, path + ".tmp")
            os.replace(path + ".tmp", path)
        lens = self.log_dict.get("Train/mean_episode_length") if len(locs["lenbuffer"]) > 0 else None
        lengths.append(float(lens) if lens is not None else None)
        c_after, c_run, c_share = collapse
        hit = collapse_at(lengths, limit[0] or 256, c_after, c_run, c_share)
        if hit is not None and early_stop:
            rec = {"collapsed_at": hit, "limit": limit[0], "threshold": c_share * (limit[0] or 256),
                   "rule": f"after iteration {c_after}, mean imagined episode length under "
                           f"{c_share:g} x the limit for {c_run} iterations in a row",
                   "last_lengths": lengths[-c_run:]}
            sd = {k: v.clone() for k, v in self.alg.policy.state_dict().items() if k.startswith("actor.")}
            torch.save({"model_state_dict": sd, "iter": n - 1, "source": "early stop"},
                       os.path.join(out_dir, f"actor_stopped_i{n}.pt"))
            raise Collapsed(n, rec)
        return out
    PT.PolicyTraining.log = log
    info["noise_floor"] = floor
    info["fall_cost"] = fall_cost
    return info


def write_collapsed(name, sim, a, weight, how, info, saves, out_dir, err, t0, res_dir=RES):
    """The record of an early-stopped arm: its lite metrics so far, and a result file marked 'collapsed'."""
    lite_dir = os.path.join(LITE_ROOT, "runs", "lite", "s2_" + name)
    os.makedirs(lite_dir, exist_ok=True)
    with open(os.path.join(lite_dir, "metrics.jsonl"), "w") as fh:
        for d in RL.LOGGED:
            fh.write(json.dumps(d, default=float) + "\n")
    have = [n for n in saves if os.path.exists(os.path.join(out_dir, f"actor_i{n}.pt"))]
    rec = {"policy": name, "sim": sim, "seed": a.seed, "status": "collapsed", "collapsed": err.record,
           "iterations_run": err.n, "max_iters": a.max_iters, "uncertainty_weight": weight, "penalty": how,
           "fall_cost": a.fall_cost, "start_states": info, "saves_planned": list(saves), "saves": have,
           "actors_sha256": {str(n): C.sha256(os.path.join(out_dir, f"actor_i{n}.pt")) for n in have},
           "mean_reward_per_iter": [d.get("Train/mean_reward") for d in RL.LOGGED],
           "wall_seconds": time.time() - t0}
    os.makedirs(res_dir, exist_ok=True)
    path = os.path.join(res_dir, name + ".json")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; not overwriting a result file")
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1)
    with open(os.path.join(out_dir, "wrapper.json"), "w") as fh:
        json.dump({k: rec[k] for k in ("status", "collapsed", "uncertainty_weight", "penalty", "fall_cost",
                                       "start_states", "saves", "actors_sha256")}, fh, indent=1)
    print(f"[train_policy] {name}: COLLAPSED at iteration {err.n}; saves kept: {have}")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="a simulator under runs/c/, e.g. sim_6M_N8_s0")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--penalty", default=None, help="'cal' (lambda_s from results/c/calibration.json) or a weight >= 0")
    ap.add_argument("--fall-cost", type=float, default=0.0, help="one-time cost of a predicted fall (Amendment 1: 4.0)")
    ap.add_argument("--max-iters", type=int, default=None, help="500, 1000, 1500 or 2000; saves every 500")
    ap.add_argument("--uncertainty-weight", type=float, default=None, help="tests only (the first pilot's flag)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--job", default="pilot")
    ap.add_argument("--name", default=None, help="`_test*` for tests")
    ap.add_argument("--iterations", type=int, default=None, help="tests only")
    ap.add_argument("--num_envs", type=int, default=TP.NUM_ENVS)
    ap.add_argument("--saves", default=None, help="comma list (tests only)")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="the early stop is for the pilot's arms; Gate C''s 10k policy runs without it")
    ap.add_argument("--test-init-std", type=float, default=None, help="tests only: start the noise below the floor")
    ap.add_argument("--test-collapse", default=None, help="tests only: after,run,share of the early stop")
    a = ap.parse_args(argv)
    test = bool(a.name and a.name.startswith("_test"))
    if not test:
        if a.penalty is None:
            raise SystemExit("Amendment 1: give --penalty cal|<w> (and --max-iters); the first pilot's flags are retired")
        if a.max_iters not in ALLOWED_ITERS:
            raise SystemExit(f"--max-iters must be one of {ALLOWED_ITERS}")
        if (a.iterations, a.num_envs, a.saves, a.uncertainty_weight, a.test_init_std,
                a.test_collapse) != (None, TP.NUM_ENVS, None, None, None, None):
            raise SystemExit("the plan fixes 4,096 robots and the saves; other settings are for `_test*` names")
        if a.fall_cost not in (0.0, 4.0):
            raise SystemExit("Amendment 1 fixes the fall cost at 4.0 (or none)")
    if a.penalty is not None:
        weight, how = resolve_penalty(a.penalty, a.sim)
    else:
        weight, how = (1.0 if a.uncertainty_weight is None else a.uncertainty_weight), "given (test)"
    iters = a.iterations if a.iterations is not None else (a.max_iters or ITERATIONS)
    if a.max_iters is None:
        a.max_iters = iters
    name = a.name or recipe_name(a.sim, a.seed, a.penalty, a.fall_cost)
    saves = tuple(int(x) for x in a.saves.split(",")) if a.saves else tuple(range(SAVE_EVERY, iters + 1, SAVE_EVERY))
    out_dir = os.path.join(RUNS, name)
    done = os.path.join(RES, name + ".json")
    if not test and os.path.exists(done):
        print(f"[train_policy] {name}: done already, skipped")
        return json.load(open(done))

    row = sim_row_of(a.sim)
    TP.sim_row = lambda sim: row
    TP.ITERATIONS = iters
    collapse = (COLLAPSE_AFTER, COLLAPSE_RUN, COLLAPSE_SHARE)
    if a.test_collapse:
        x = a.test_collapse.split(",")
        collapse = (int(x[0]), int(x[1]), float(x[2]))
    info = patch_lite(weight, a.sim, set(saves), out_dir, fall_cost=a.fall_cost, early_stop=not a.no_early_stop,
                      collapse=collapse, test_init_std=a.test_init_std)
    print(f"[train_policy] {name}: uncertainty weight {weight:g} ({how}), fall cost {a.fall_cost:g}, "
          f"{iters} iterations, saves {list(saves)}, noise floor {NOISE_FLOOR}")
    argv2 = ["--sim", a.sim, "--seed", str(a.seed), "--route", "rebuilt", "--job", a.job, "--threads", str(a.threads),
             "--name", name, "--iterations", str(iters), "--num_envs", str(a.num_envs)]
    t0 = time.time()
    try:
        rec = TP.main(argv2)
    except Collapsed as err:
        res_dir = os.path.join(AIR, ".cache", "c2r_test_res") if test else RES
        if test and os.path.exists(os.path.join(res_dir, name + ".json")):
            os.remove(os.path.join(res_dir, name + ".json"))
        return write_collapsed(name, a.sim, a, weight, how, info, saves, out_dir, err, t0, res_dir)
    missing = [n for n in saves if not os.path.exists(os.path.join(out_dir, f"actor_i{n}.pt"))]
    if missing:
        raise SystemExit(f"{name}: no actor saved at iterations {missing}")
    extra = {"status": "finished", "early_stop": not a.no_early_stop, "uncertainty_weight": weight, "penalty": how, "fall_cost": a.fall_cost,
             "max_iters": iters, "start_states": info, "saves": list(saves),
             "actors_sha256": {str(n): C.sha256(os.path.join(out_dir, f"actor_i{n}.pt")) for n in saves},
             "wall_seconds_wrapper": time.time() - t0}
    with open(os.path.join(out_dir, "wrapper.json"), "w") as fh:
        json.dump(extra, fh, indent=1)
    if not test:                                   # Step 2's route wrote the result; add the recipe to a new file
        with open(os.path.join(RES, name + ".recipe.json"), "w") as fh:
            json.dump(extra, fh, indent=1)
    return {**rec, **extra}


if __name__ == "__main__":
    main()

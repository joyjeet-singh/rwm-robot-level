"""Run the authors' lite policy training (robotic_world_model_lite/scripts/train.py) on this CPU.

    ./py scripts/lite/run_lite.py --name smoke_released --model released --num_envs 1024 --max_iterations 10
    ./py scripts/lite/run_lite.py --name smoke_A_s0 --model rebuilt:A_s0 --num_envs 1024 --max_iterations 10
    ./py scripts/lite/run_lite.py ... --holdout 1,8      # imagination start states from training episodes only

What it does and does not change (all overrides are listed in step1/notes/S4_lite.md):
  * the lite code and rsl_rl_rwm are imported from their own folders, read only (no byte-code, nothing
    is written there); the wandb package is replaced by scripts/lite/stubs/wandb;
  * every log, checkpoint and metrics file goes to runs/lite/<name>/ (git-ignored);
  * run_experiment() of train.py reads a module-level `args_cli`, so its seven calls are repeated here in
    the same order; no reward, observation, PPO setting or command range is touched;
  * `--model released` uses the authors' checkpoint through the upstream SystemDynamicsEnsemble;
    `--model rebuilt:<key>` swaps in our rebuilt simulator through scripts/lite/adapter.py.
Nothing here is imported by the paper's pipeline. The sibling folders are never written.
"""
import argparse
import json
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AIR = os.path.dirname(os.path.dirname(HERE))
PDM = os.environ.get("PDM_ROOT", os.path.dirname(AIR))
LITE = os.path.join(PDM, "robotic_world_model_lite")
RSL = os.path.join(PDM, "rsl_rl_rwm")
STUBS = os.path.join(HERE, "stubs")
SCRIPTS = os.path.dirname(HERE)

# stubs first, then the lite scripts, then rsl_rl_rwm, then our scripts (for common, adapter)
for p in (SCRIPTS, RSL, os.path.join(LITE, "scripts"), STUBS):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
sys.dont_write_bytecode = True

import torch  # noqa: E402

_orig_load = torch.load


def _load_cpu(*a, **k):
    """The released checkpoint was saved on a GPU, and train.py calls torch.load without map_location."""
    k.setdefault("map_location", "cpu")
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)


torch.load = _load_cpu

# tensordict 0.3.2 (the version installed here, no installs allowed) forbids `key in tensordict`, which
# rsl_rl.utils.resolve_obs_groups uses. Newer versions allow it. Same meaning, spelled for 0.3.2.
from tensordict import TensorDictBase  # noqa: E402
TensorDictBase.__contains__ = lambda self, key: key in self.keys()

import wandb  # noqa: E402  (the stub)

assert wandb.__file__.startswith(STUBS), wandb.__file__

LOGGED = []                                  # one dict per iteration, from PolicyTraining.log
wandb.log = lambda d=None, *a, **k: LOGGED.append(
    {kk: (float(v) if hasattr(v, "__float__") or isinstance(v, (int, float)) else str(v))
     for kk, v in (d or {}).items()})


def pick_model(spec):
    """'released' | 'rebuilt:<key or run directory name>' -> (kind, weights path or None, ensemble or None)."""
    if spec == "released":
        return "released", None, None
    kind, _, key = spec.partition(":")
    assert kind == "rebuilt" and key, spec
    table = json.load(open(os.path.join(AIR, "results", "models.json")))
    rows = [m for pool in table.values() for m in pool
            if m["key"] == key or os.path.basename(os.path.dirname(m["path"])) == key]
    assert rows, f"no model {key!r} in results/models.json"
    m = rows[0]
    return "rebuilt", os.path.join(PDM, m["path"]), int(m["ensemble"])


def window_rows(termination, history, forecast):
    """Row index of each window the lite SystemDynamicsDataset builds, in its order (train.py:130-133)."""
    reset = termination.reshape(-1).nonzero()[0]
    n = len(termination)
    w = history + forecast
    out = []
    for i in range(n - w + 1):
        if not ((reset >= i) & (reset < i + w - 1)).any():
            out.append(i)
    return out


def restrict_windows(dataset, csv_path, keep_episodes, history, forecast):
    """Keep only the dataset windows whose every row lies in `keep_episodes`.

    The lite dataset is a list of windows cut from state_action_data_0.csv, all 10 episodes. The env draws
    its imagination start states from it (envs/base.py: sample_batch). This filters the windows in place,
    after prepare_data and before training, so the upstream code is untouched. Returns (kept, total)."""
    import numpy as np
    import common
    R = common.R
    import pandas as pd
    data = pd.read_csv(csv_path, header=None).to_numpy()
    ep = R.build_episode_id(len(data))
    rows = window_rows(data[:, -1:], history, forecast)
    assert len(rows) == len(dataset), (len(rows), len(dataset))
    w = history + forecast
    ok = np.array([bool(np.isin(ep[i:i + w], list(keep_episodes)).all()) for i in rows])
    idx = torch.as_tensor(np.nonzero(ok)[0])
    for name in ("state_data", "action_data", "extension_data", "contact_data", "termination_data"):
        setattr(dataset, name, getattr(dataset, name)[idx])
    return int(ok.sum()), int(len(ok))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lite policy training on CPU (policy only, as train.py).")
    ap.add_argument("--name", required=True, help="run name; output goes to runs/lite/<name>/")
    ap.add_argument("--model", default="released", help="released | rebuilt:<key>, e.g. rebuilt:A_s0")
    ap.add_argument("--num_envs", type=int, default=8192)
    ap.add_argument("--max_iterations", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0, help="torch seed (upstream sets none)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--holdout", default="", help="episodes to exclude from imagination starts, e.g. 1,8")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)

    import train as T                              # lite/scripts/train.py (main is guarded)
    cfg = T.resolve_task_config("anymal_d_flat")
    kind, weights, ens = pick_model(a.model)

    # -- overrides (each is recorded in S4_lite.md) -------------------------------------------------
    cfg.experiment_config.device = "cpu"                                   # train.py --device
    cfg.environment_config.num_envs = a.num_envs                           # flag
    cfg.policy_training_config.max_iterations = a.max_iterations           # flag
    cfg.data_config.dataset_root = os.path.join(LITE, "assets")            # was relative to cwd
    cfg.model_architecture_config.resume_path = (
        os.path.join(LITE, "assets", "models", "pretrain_rnn_ens.pt") if kind == "released" else None)

    log_dir = os.path.join(AIR, "runs", "lite", a.name)
    os.makedirs(log_dir, exist_ok=True)
    t0 = time.time()

    exp = T.ModelBasedExperiment(**cfg.experiment_config.to_dict())
    exp.prepare_environment(**cfg.environment_config.to_dict())
    exp.prepare_model(**cfg.model_architecture_config.to_dict())
    if kind == "rebuilt":
        from adapter import load_rebuild
        sim = load_rebuild(weights, ens)
        exp.system_dynamics = sim
        exp.env.set_system_dynamics(sim)
    exp.prepare_policy(**cfg.policy_architecture_config.to_dict())
    exp.prepare_algorithm(**cfg.policy_algorithm_config.to_dict())
    exp.prepare_data(**cfg.data_config.to_dict())

    restricted = None
    if a.holdout:
        import common
        held = [int(x) for x in a.holdout.split(",")]
        keep = common.train_episodes(held)
        csv = os.path.join(LITE, "assets", "data", "state_action_data_0.csv")
        h, f = cfg.model_architecture_config.history_horizon, cfg.model_architecture_config.forecast_horizon
        restricted = restrict_windows(exp.dataset, csv, keep, h, f)
        print(f"[run_lite] windows kept {restricted[0]} of {restricted[1]} (held out episodes {held})")

    # observation only: record every step's reward and whether it is finite
    rew = {"n": 0, "finite": True, "sum": 0.0, "per_iter": []}
    step_fn = exp.env.imagination_step
    steps_per_iter = cfg.policy_training_config.num_steps_per_env

    def watched(*args, **kw):
        out = step_fn(*args, **kw)
        r = out[1]
        rew["finite"] = rew["finite"] and bool(torch.isfinite(r).all())
        rew["sum"] += float(r.mean())
        rew["n"] += 1
        if rew["n"] % steps_per_iter == 0:
            rew["per_iter"].append(rew["sum"] / steps_per_iter)
            rew["sum"] = 0.0
        return out
    exp.env.imagination_step = watched

    t1 = time.time()
    exp.train_policy(log_dir, **cfg.policy_training_config.to_dict())
    wall = time.time() - t1

    sec = [d["Perf/collection time"] + d["Perf/learning_time"] for d in LOGGED if "Perf/collection time" in d]
    with open(os.path.join(log_dir, "metrics.jsonl"), "w") as fh:
        for d in LOGGED:
            fh.write(json.dumps(d) + "\n")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 1e6 if sys.platform == "darwin" else rss / 1e3       # bytes on macOS, kB on Linux
    losses = [v for d in LOGGED for k, v in d.items() if k.startswith("Loss/") and isinstance(v, float)]
    import statistics
    summary = {
        "name": a.name, "model": a.model, "num_envs": a.num_envs, "iterations": a.max_iterations,
        "threads": a.threads, "seed": a.seed, "holdout": a.holdout or None, "windows_kept_total": restricted,
        "seconds_per_iter_mean": statistics.mean(sec) if sec else None,
        "seconds_per_iter_median": statistics.median(sec) if sec else None,
        "seconds_per_iter_each": sec,
        "setup_seconds": t1 - t0, "train_wall_seconds": wall,
        "peak_rss_mb": rss_mb,
        "step_reward_all_finite": rew["finite"], "step_reward_mean_per_iter": rew["per_iter"],
        "losses_all_finite": all(map(lambda v: v == v and abs(v) != float("inf"), losses)),
        "mean_reward_last": next((d["Train/mean_reward"] for d in reversed(LOGGED) if "Train/mean_reward" in d), None),
        "checkpoints": sorted(f for f in os.listdir(log_dir) if f.endswith(".pt")),
        "note": "timings are wall seconds per iteration (collection + learning, as the lite log prints)",
    }
    with open(os.path.join(log_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("seconds_per_iter_each", "step_reward_mean_per_iter")}))
    return summary


if __name__ == "__main__":
    main()

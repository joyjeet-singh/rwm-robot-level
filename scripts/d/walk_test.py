"""Option D (D0): the walk test of one simulator (secondary; Rule 2).  ./py

    ./py scripts/d/walk_test.py --sim REL          # -> results/d/walk/REL.json (written once)

OPTIOND_PLAN.md, "The walk test": RWM's released policy run closed-loop inside the simulator's imagination, as Option
C's calibration ran it (scripts/c/calibrate_penalty.py: the lite env and reward unchanged, its `imagine`): noise-free
actions, 1,024 robots, 256 steps, seed 20278, each robot's first episode. Three numbers: the share of imagined episodes
that end in a predicted fall, the tracking reward per step (the two tracking terms, lite units), and u_bar, the mean
disagreement between the networks (the lite measure). Also the tracking as a share of what the same policy earns per
step in MuJoCo (results/c/calibration.json's MuJoCo reference).

Start states: the authors' recorded data (the lite code's own start states), with Step 2's hold-out where Step 2 had
one (episodes 1 and 8 for the rebuilds and A5_s0), none for REL, its copies and the ladder. Option C's four simulators,
which never saw the recorded data, get all ten episodes, converted into their own scales (optd/DECISIONS.md D0-4);
for them this differs from Option C's calibration, whose start states were MuJoCo windows.
Only the policy's imagination is involved; the world (the hand-built calibrated MuJoCo ANYmal D with a fitted PD
motor; not Isaac Lab, not hardware) enters only through the MuJoCo reference.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
AIR = os.path.dirname(SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, "c"))
import train_policy as W  # noqa: E402  (Option C's: sets the lite paths through Step 2's trainer)
import calibrate_penalty as CP  # noqa: E402
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import sims  # noqa: E402
import dprogress  # noqa: E402

TP, RL, C = W.TP, W.RL, W.C
OUT_DIR = os.path.join(AIR, "results", "d", "walk")
N_ROBOTS, N_STEPS, SEED = CP.N_ROBOTS, CP.N_STEPS, CP.CAL_SEED
CALIBRATION = os.path.join(AIR, "results", "c", "calibration.json")


def to_own_scales(dataset, mean, std):
    """Re-express the dataset's windows (authors' scales) in the simulator's own; the env (de)normalises through it."""
    am = dataset.state_data_mean.double()
    asd = dataset.state_data_std.double()
    mo, so = torch.as_tensor(mean, dtype=torch.float64), torch.as_tensor(std, dtype=torch.float64)
    dataset.state_data = ((dataset.state_data.double() * asd + am - mo) / so).float()
    dataset.state_data_mean, dataset.state_data_std = mo.float(), so.float()


def experiment(name, n_envs):
    """The lite env with this simulator and its start states (as calibrate_penalty.experiment)."""
    import train as T
    from adapter import LiteDynamicsAdapter
    cfg = T.resolve_task_config("anymal_d_flat")
    cfg.experiment_config.device = "cpu"
    cfg.environment_config.num_envs = n_envs
    cfg.data_config.dataset_root = os.path.join(RL.LITE, "assets")
    cfg.model_architecture_config.resume_path = None
    exp = T.ModelBasedExperiment(**cfg.experiment_config.to_dict())
    exp.prepare_environment(**cfg.environment_config.to_dict())
    exp.prepare_model(**cfg.model_architecture_config.to_dict())
    m, mean, std, rec = sims.load(name)
    model = LiteDynamicsAdapter(m)
    exp.system_dynamics = model
    exp.env.set_system_dynamics(model)
    exp.prepare_policy(**cfg.policy_architecture_config.to_dict())
    exp.prepare_algorithm(**cfg.policy_algorithm_config.to_dict())
    exp.prepare_data(**cfg.data_config.to_dict())
    kept = None
    if rec["holdout"]:
        keep = C.train_episodes([int(x) for x in rec["holdout"].split(",")])
        csv = os.path.join(RL.LITE, "assets", "data", "state_action_data_0.csv")
        h, f = cfg.model_architecture_config.history_horizon, cfg.model_architecture_config.forecast_horizon
        kept = RL.restrict_windows(exp.dataset, csv, keep, h, f)
    if rec["scales"] == "own":
        to_own_scales(exp.dataset, mean, std)
    assert exp.env.init_dataset is exp.dataset
    starts = {"rule": "the authors' recorded data (the lite code's own start states)", "holdout": rec["holdout"] or None,
              "windows_kept_total": kept, "scales": rec["scales"]}
    return exp, starts, rec


def run(name, out=None, n_envs=N_ROBOTS, n_steps=N_STEPS, job="record"):
    out = out or os.path.join(OUT_DIR, f"{name}.json")
    if os.path.exists(out):
        print(f"[walk_test] {name}: {os.path.relpath(out, AIR)} exists, skipped")
        with open(out) as fh:
            return json.load(fh)
    import s1_policy as P
    dprogress.write(job, f"walk test {name}")
    t0 = time.time()
    actor = P.load_actor(None)
    actor.eval()
    exp, starts, rec = experiment(name, n_envs)
    r = CP.imagine(exp, actor, SEED, n_steps)
    with open(CALIBRATION) as fh:
        mj = json.load(fh)["mujoco_reference"]["tracking_per_step"]
    res = {"what": "Option D walk test (secondary): RWM's released policy, noise-free, in the simulator's imagination",
           "simulator": rec, "settings": {"robots": n_envs, "steps": n_steps, "seed": SEED,
                                          "actions": "noise-free (policy mean)", "episodes": "each robot's first"},
           "start_states": starts, "falls": r["share_fell"], "tracking_per_step": r["tracking_per_step"],
           "u_bar": r["u_bar"], "tracking_mujoco_per_step": mj, "tracking_ratio": r["tracking_per_step"] / mj,
           "imagine": r, "seconds": time.time() - t0, "torch": torch.__version__}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w") as fh:
        json.dump(res, fh, indent=1)
    os.replace(out + ".tmp", out)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="one name, or a comma list")
    ap.add_argument("--out", default=None, help="tests only (one simulator)")
    ap.add_argument("--robots", type=int, default=N_ROBOTS, help="tests only, with --out")
    ap.add_argument("--steps", type=int, default=N_STEPS, help="tests only, with --out")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--job", default="record")
    a = ap.parse_args(argv)
    if (a.robots, a.steps) != (N_ROBOTS, N_STEPS) and not a.out:
        ap.error("other sizes are for tests only (with --out)")
    torch.set_num_threads(a.threads)
    for n in a.sim.split(","):
        r = run(n, a.out, a.robots, a.steps, a.job)
        print(json.dumps({"sim": n, "falls": round(r["falls"], 4), "tracking_per_step": round(r["tracking_per_step"], 6),
                          "u_bar": round(r["u_bar"], 4), "tracking_ratio": round(r["tracking_ratio"], 3)}))


if __name__ == "__main__":
    main()

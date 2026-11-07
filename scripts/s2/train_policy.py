"""Train one Step 2 walking policy inside a learned simulator, with the authors' lite code, and export its actor.

    ./py scripts/s2/train_policy.py --sim A_s0 --seed 0            # -> runs/s2/A_s0__p0/actor.pt
    ./py scripts/s2/train_policy.py --sim REL --seed 0             # the released five-network model
    ./py scripts/s2/train_policy.py --sim A_s0__u50 --seed 0       # a compressed copy (applied here)

What it does (STEP2_PLAN.md, "Policies"):
  * wraps scripts/lite/run_lite.py unchanged: 500 iterations, 4,096 robots, `--holdout` from the simulator's fold,
    the given torch seed. Run names are `s2_<policy>`, so the lite run directory is runs/lite/s2_<policy>/;
  * the simulator: `REL` is the authors' checkpoint through run_lite's `--model released`. Any other key is a row of
    results/models.json (pool F18): its weights through common.load, with its compression variant applied by
    scripts/compress.py (run_lite itself ignores variants, so the wrapper supplies the model), then the lite
    adapter. The parent weights' sha256 and the variant's info are checked against models.json;
  * the fold: arena '18' -> `--holdout 1,8`; arena 'all' (the released model, trained on all ten episodes) -> none;
  * progress: results/s2/progress.json every 25 iterations (and after any iteration more than 5 minutes after
    the last write), from the iteration log itself;
  * export: the last checkpoint's actor weights (actor.0/2/4/6, the layout s1_policy.load_actor reads) to
    runs/s2/<policy>/actor.pt, and runs/s2/<policy>/train.json;
  * imagined return: the mean of the lite log's `Train/mean_reward` (mean return of imagined episodes that ended in
    the iteration, over its last 100) over the last 25 iterations; also the mean step reward of those iterations;
  * result: results/s2/train/<policy>.json (never overwritten). Resumable: a policy whose result and actor exist
    with matching hash is skipped. An interrupted run restarts from iteration 0, so every policy is one
    uninterrupted, seeded run.
Names starting with `_test` may use other sizes and write no result file (tests only).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
AIR = os.path.dirname(SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, "lite"))
sys.dont_write_bytecode = True

import run_lite as RL  # noqa: E402  (sets its own sys.path, torch.load and wandb stub)
import torch  # noqa: E402

for p in (SCRIPTS,):                     # our common/compress, after run_lite's order is in place
    if p not in sys.path:
        sys.path.append(p)
import common as C  # noqa: E402
import compress as K  # noqa: E402

ITERATIONS = 500
NUM_ENVS = 4096
LAST = 25                                 # iterations averaged for the imagined return
PROGRESS_EVERY = 25
RUNS = os.path.join(AIR, "runs", "s2")
RES = os.path.join(AIR, "results", "s2", "train")
PROGRESS = os.path.join(AIR, "results", "s2", "progress.json")
HOLDOUT_OF_ARENA = {"18": "1,8", "all": ""}


def policy_name(sim, seed):
    return f"{sim}__p{seed}"


def sim_row(sim):
    table = json.load(open(os.path.join(AIR, "results", "models.json")))
    if sim == "REL":
        return table["REL"][0]
    rows = [m for m in table["F18"] if m["key"] == sim]
    if len(rows) != 1:
        raise SystemExit(f"simulator {sim!r} is not one row of pool F18 in results/models.json")
    return rows[0]


def build_sim(row):
    """The rebuilt simulator of a models.json row, its variant applied, wrapped for the lite env."""
    from adapter import LiteDynamicsAdapter
    path = os.path.join(C.PDM, row["path"])
    if C.sha256(path) != row["sha256"]:
        raise SystemExit(f"weights of {row['key']} differ from results/models.json (sha256)")
    m = C.load(path, int(row["ensemble"]))
    if row["variant"]:
        m, info = K.apply(m, row["variant"])
        want = row["variant_info"] or {}
        for k, v in want.items():
            got = info.get(k)
            same = (abs(float(got) - float(v)) <= 1e-9 * max(1.0, abs(float(v)))
                    if isinstance(v, (int, float)) and not isinstance(v, bool) else got == v)
            if not same:
                raise SystemExit(f"variant {row['variant']} of {row['key']}: {k} = {got}, models.json has {v}")
        m.eval()
    return LiteDynamicsAdapter(m)


def write_progress(job, step, done, total):
    os.makedirs(os.path.dirname(PROGRESS), exist_ok=True)
    rec = {"job": job, "step": step, "done": int(done), "total": int(total),
           "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = f"{PROGRESS}.{os.getpid()}.tmp"
    json.dump(rec, open(tmp, "w"))
    os.replace(tmp, PROGRESS)


def export_actor(run_dir, out_path, iterations):
    """The last checkpoint's actor layers, saved as {'model_state_dict': actor.*, ...}."""
    cks = [f for f in os.listdir(run_dir) if f.startswith("policy_") and f.endswith(".pt")]
    last = max(cks, key=lambda f: int(f[7:-3]))
    ck = torch.load(os.path.join(run_dir, last), map_location="cpu", weights_only=False)
    if int(ck["iter"]) != iterations - 1:
        raise SystemExit(f"last checkpoint {last} is iteration {ck['iter']}, expected {iterations - 1}")
    sd = {k: v.clone() for k, v in ck["model_state_dict"].items() if k.startswith("actor.")}
    assert sorted(sd) == sorted(f"actor.{i}.{w}" for i in (0, 2, 4, 6) for w in ("weight", "bias")), sorted(sd)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    torch.save({"model_state_dict": sd, "iter": int(ck["iter"]), "source": last}, tmp)
    os.replace(tmp, out_path)
    return last


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="REL or a pool-F18 key of results/models.json")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--job", default="train", help="job name written to the progress file")
    ap.add_argument("--name", default=None, help="policy name (default <sim>__p<seed>); `_test*` for tests")
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    ap.add_argument("--num_envs", type=int, default=NUM_ENVS)
    a = ap.parse_args(argv)
    name = a.name or policy_name(a.sim, a.seed)
    test = name.startswith("_test")
    if not test and (a.iterations, a.num_envs) != (ITERATIONS, NUM_ENVS):
        raise SystemExit("the plan fixes 500 iterations and 4,096 robots; other sizes are for `_test*` names only")

    out_dir = os.path.join(RUNS, name)
    actor_path = os.path.join(out_dir, "actor.pt")
    res_path = os.path.join(RES, name + ".json")
    if not test and os.path.exists(res_path) and os.path.exists(actor_path):
        if json.load(open(res_path))["actor_sha256"] == C.sha256(actor_path):
            print(f"[train_policy] {name}: done already, skipped")
            return json.load(open(res_path))
        raise SystemExit(f"{actor_path} does not match {res_path}")
    if not test and os.path.exists(res_path):
        raise SystemExit(f"{res_path} exists but the actor does not; not overwriting a result file")

    row = sim_row(a.sim)
    holdout = HOLDOUT_OF_ARENA[str(row["arena"])]
    lite_name = "s2_" + name

    # progress every PROGRESS_EVERY iterations, from the lite iteration log (one wandb.log call per iteration)
    # (also after any iteration that ends more than 5 minutes after the last write, so a slow machine still
    # writes progress at least every 10 minutes)
    log_one = RL.wandb.log
    last_write = [time.time()]

    def logged(*args, **kw):
        log_one(*args, **kw)
        n = len(RL.LOGGED)
        if n % PROGRESS_EVERY == 0 or n == a.iterations or time.time() - last_write[0] > 300:
            write_progress(a.job, f"train {name}", n, a.iterations)
            last_write[0] = time.time()
    RL.wandb.log = logged

    # the simulator: run_lite's own route for the released model; otherwise ours, variant applied
    if a.sim == "REL":
        model_arg = "released"
    else:
        import adapter
        sim = build_sim(row)
        handed = []

        def load_rebuild(path, ens):                          # run_lite calls it after pick_model
            handed.append(sim)
            return sim
        adapter.load_rebuild = load_rebuild
        RL.pick_model = lambda spec: ("rebuilt", None, None)
        model_arg = "rebuilt:" + a.sim

    write_progress(a.job, f"train {name}", 0, a.iterations)
    t0 = time.time()
    summ = RL.main(["--name", lite_name, "--model", model_arg, "--num_envs", str(a.num_envs),
                    "--max_iterations", str(a.iterations), "--seed", str(a.seed), "--threads", str(a.threads)]
                   + (["--holdout", holdout] if holdout else []))
    wall = time.time() - t0
    if a.sim != "REL" and len(handed) != 1:
        raise SystemExit("the lite trainer did not take the simulator built here")

    run_dir = os.path.join(AIR, "runs", "lite", lite_name)
    src = export_actor(run_dir, actor_path, a.iterations)
    last = RL.LOGGED[-LAST:]
    mr = [d["Train/mean_reward"] for d in last if isinstance(d.get("Train/mean_reward"), float)]
    step_r = summ["step_reward_mean_per_iter"][-LAST:]
    rec = {
        "policy": name, "sim": a.sim, "seed": a.seed, "sim_pool": row.get("pool"), "sim_group": row.get("group"),
        "sim_variant": row.get("variant"), "sim_ensemble": int(row["ensemble"]), "sim_arena": row["arena"],
        "holdout": holdout or None, "windows_kept_total": summ["windows_kept_total"],
        "iterations": a.iterations, "num_envs": a.num_envs, "threads": a.threads,
        "imagined_return": (sum(mr) / len(mr)) if mr else None,
        "imagined_return_iterations": len(mr),
        "imagined_step_reward": (sum(step_r) / len(step_r)) if step_r else None,
        "mean_reward_per_iter": [d.get("Train/mean_reward") for d in RL.LOGGED],
        "step_reward_mean_per_iter": summ["step_reward_mean_per_iter"],
        "step_reward_all_finite": summ["step_reward_all_finite"], "losses_all_finite": summ["losses_all_finite"],
        "seconds_per_iter_median": summ["seconds_per_iter_median"], "wall_seconds": wall,
        "peak_rss_mb": summ["peak_rss_mb"], "checkpoint": src, "actor_sha256": C.sha256(actor_path),
        "lite_run_dir": os.path.relpath(run_dir, AIR),
        "what": "Step 2 policy trained in a learned simulator (lite code, unchanged); imagined return = mean "
                "Train/mean_reward over the last 25 iterations",
    }
    os.makedirs(out_dir, exist_ok=True)
    json.dump(rec, open(os.path.join(out_dir, "train.json"), "w"), indent=1)
    if not test:
        os.makedirs(RES, exist_ok=True)
        tmp = res_path + ".tmp"
        json.dump(rec, open(tmp, "w"), indent=1)
        os.replace(tmp, res_path)
    write_progress(a.job, f"trained {name}", a.iterations, a.iterations)
    print(json.dumps({k: rec[k] for k in ("policy", "holdout", "imagined_return", "imagined_step_reward",
                                          "step_reward_all_finite", "losses_all_finite", "wall_seconds")}))
    return rec


if __name__ == "__main__":
    main()

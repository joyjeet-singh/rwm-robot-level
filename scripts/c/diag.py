"""Option C diagnosis, D1-D4 (OPTIONC_PLAN.md, "Design fixed by this plan"; D0 is recomputed in the report).

Why Step 2's rebuilt-simulator policies fell. One part per process, so that torch-only parts run under ./py and the
MuJoCo parts under ./py-mj:

    ./py-mj scripts/c/diag.py --part phys    --policy A_s0__p0   # D1 physics: Step 2's 80 episodes, recorded
    ./py    scripts/c/diag.py --part falls   --policy A_s0__p0   # D1: does the simulator see each real fall?
    ./py    scripts/c/diag.py --part imag    --policy A_s0__p0   # D1: 512 imagined episodes, as policy training does
    ./py    scripts/c/diag.py --part d2      --policy A_s0__p0   # D2: distance from the simulator's training data
    ./py    scripts/c/diag.py --part d3train --seed 0            # D3: a policy in A5_s0 (five networks)
    ./py-mj scripts/c/diag.py --part d3eval  --seed 0            # D3: Step 2's evaluator on that policy
    ./py    scripts/c/diag.py --part d4                          # D4: command coverage
    ./py    scripts/c/diag.py --part report                      # results/c_diag.json and results/c_diag.txt

Everything is resumable: a part whose output exists is skipped. Work files go to runs/c/diag/ (--root for tests);
Step 2's own files are only read. `phys` drives Step 2's fidelity.run_policy_episode unchanged, through an env whose
step and reset also record, and checks each episode against Step 2's stored result.

Conventions. MuJoCo rows: x[k] is the state before joint command a[k]; x[k+1] is the state it produced, so
(x[k], a[k]) is the pair the simulator consumes last when it predicts x[k+1]. In the recorded CSV the same pair is
state row r with action row r+1. A fall on step T = steps-1 is predicted by the simulator as x[T+1]'s fall flag.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
AIR = os.path.dirname(SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, "s2"))
sys.dont_write_bytecode = True

PILOT = ("REL__p0", "A_s0__p0", "B_s0__p0", "A_s0__u50__p0")
D3_SIM = "A5_s0"
D3_SEEDS = (0, 1)
D3 = tuple(f"{D3_SIM}__p{s}" for s in D3_SEEDS)
AUTHORS = "authors"
HISTORY = 32
LAST = 50                  # forecast steps before a real fall
TOL_STEPS = 10             # 0.2 s at 0.02 s per step
K_NN = 10
N_D2 = 10000
IMAG_EPISODES = 512
IMAG_SEED = 20270
D2_SEED = 2027
D4_MC = 1000000
D4_HALF = 0.2
CAT_FULL, CAT_SHORT, CAT_TOO_SHORT = "full", "short", "too_short"

PROGRESS = os.path.join(AIR, "results", "c", "progress.json")
OUT_JSON = os.path.join(AIR, "results", "c_diag.json")
OUT_TXT = os.path.join(AIR, "results", "c_diag.txt")
RUNS_C = os.path.join(AIR, "runs", "c")
RES_D3 = os.path.join(AIR, "results", "c", "d3_train")
EVAL_D3 = os.path.join(AIR, "results", "c", "d3_eval")
LITE_ROOT = os.path.join(RUNS_C, "d3_lite")     # the lite log of a D3 policy: LITE_ROOT/runs/lite/s2_<policy>/


def write_progress(step, done=None, total=None):
    os.makedirs(os.path.dirname(PROGRESS), exist_ok=True)
    rec = {"job": "diag", "step": step, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if done is not None:
        rec.update(done=int(done), total=int(total))
    tmp = f"{PROGRESS}.{os.getpid()}.tmp"
    json.dump(rec, open(tmp, "w"))
    os.replace(tmp, PROGRESS)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    json.dump(obj, open(tmp, "w"), indent=1, default=float)
    os.replace(tmp, path)


def save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def sim_of(policy):
    return policy.rsplit("__p", 1)[0]


def actor_path(policy):
    if policy in PILOT:
        return os.path.join(AIR, "runs", "s2", policy, "actor.pt")
    return os.path.join(RUNS_C, policy, "actor.pt")


def lite_run_dir(policy):
    if policy in PILOT:
        return os.path.join(AIR, "runs", "lite", "s2_" + policy)
    return os.path.join(LITE_ROOT, "runs", "lite", "s2_" + policy)


def term_names():
    import s1_policy as P
    P.C.data()                                       # loads the lite config module that reward_weights reads
    return list(P.reward_weights()[0].keys())


# ------------------------------------------------------------------ D1, physics
class Recorder:
    """Records states, joint commands and reward terms while Step 2's run_policy_episode drives the env."""

    def __init__(self, env):
        self.env, self._step, self._reset = env, env.step, env.reset
        env.step, env.reset = self.step, self.reset
        self.xs, self.a, self.terms = [], [], []

    def reset(self, *args, **kw):
        out = self._reset(*args, **kw)
        self.xs, self.a, self.terms = [self.env.s.copy()], [], []
        return out

    def step(self, action):
        out = self._step(action)
        info = out[1]
        self.xs.append(info["state"].copy())
        self.a.append(np.array(action, dtype=np.float64))
        self.terms.append(np.array(info["terms"], dtype=np.float64))
        return out


def part_phys(a):
    import fidelity as F
    import s1_policy as P
    import torch
    torch.set_num_threads(a.threads)
    out_dir = os.path.join(a.root, "phys", a.policy)
    stored = None
    if a.policy in PILOT:
        stored = {(r["schedule"], r["start"]): r
                  for r in json.load(open(os.path.join(AIR, "results", "s2", "eval", a.policy + ".json")))["episodes"]}
    actor = F.torch_actor(P.load_actor() if a.policy == AUTHORS else P.load_actor(actor_path(a.policy)))
    env = F.make_env("cal", reward=True)
    rec = Recorder(env)
    plan = [(e, j) for e in F.EVAL_SCHEDULES for j in range(F.EVAL_STARTS)][:a.limit or None]
    mismatches = 0
    for i, (e, j) in enumerate(plan):
        path = os.path.join(out_dir, f"e{e}_j{j}.npz")
        if os.path.exists(path):
            mismatches += int(np.load(path)["mismatch"])
        else:
            r = F.run_policy_episode(env, actor, e, j)
            bad = False
            if stored is not None:
                s = stored[(e, j)]
                bad = abs(r["ret"] - s["ret"]) > 1e-6 or bool(r["fell"]) != bool(s["fell"]) or r["steps"] != s["steps"]
            save_npz(path, x=np.array(rec.xs), a=np.array(rec.a), terms=np.array(rec.terms), fell=bool(r["fell"]),
                     steps=int(r["steps"]), ret=float(r["ret"]), mismatch=bad)
            mismatches += int(bad)
        if (i + 1) % 5 == 0 or i + 1 == len(plan):
            write_progress(f"D1 physics {a.policy}", i + 1, len(plan))
    print(f"[phys] {a.policy}: {len(plan)} episodes, {mismatches} differ from Step 2's stored result")


def load_phys(root, policy):
    return [dict(np.load(p)) for p in sorted(glob.glob(os.path.join(root, "phys", policy, "e*_j*.npz")))]


# ------------------------------------------------------------------ the simulators
def load_sim(policy):
    import train_policy as TP
    return TP.build_sim(TP.sim_row(sim_of(policy)))


def state_stats():
    import common as C
    cfg = C.data()[0]
    return (np.asarray(cfg["state_data_mean"], dtype=np.float64), np.asarray(cfg["state_data_std"], dtype=np.float64))


def rollout_falls(sim, x_norm, a, end, n_forecast, mid):
    """Open-loop rollout: true 32-row history ending at row `end`, then the sent joint commands. Returns P(fall) per step."""
    import torch
    sim.reset()
    sh = torch.as_tensor(x_norm[end - HISTORY + 1:end + 1], dtype=torch.float32)[None]
    ah = torch.as_tensor(a[end - HISTORY + 1:end + 1], dtype=torch.float32)[None]
    ids = None if mid is None else torch.full((1, 1, 1), mid, dtype=torch.long)
    probs = []
    for k in range(n_forecast):
        with torch.no_grad():
            s_next, _, _, _, _, term = sim.forward(sh, ah, ids)
        probs.append(float(torch.sigmoid(term).squeeze()))
        if k + 1 < n_forecast:
            sh = s_next[:, None]
            ah = torch.as_tensor(a[end + 1 + k], dtype=torch.float32)[None, None]
    return np.array(probs)


def part_falls(a):
    import torch
    torch.set_num_threads(a.threads)
    out = os.path.join(a.root, "falls", a.policy + ".json")
    if os.path.exists(out):
        print(f"[falls] {a.policy}: exists, skipped")
        return
    sim = load_sim(a.policy)
    sm, ss = state_stats()
    eps = [r for r in load_phys(a.root, a.policy) if bool(r["fell"])][:a.limit or None]
    members = [None] + list(range(sim.ensemble_size))
    rows = []
    for i, r in enumerate(eps):
        steps = int(r["steps"])
        T = steps - 1                                # the step on which the real fall happened
        if steps >= LAST + HISTORY - 1:
            end, cat = steps - LAST, CAT_FULL        # history ends at x[end]; a[end] is the first forecast command
        elif steps >= HISTORY + 1:
            end, cat = HISTORY - 1, CAT_SHORT        # as long a forecast as the episode allows
        else:
            rows.append({"steps": steps, "category": CAT_TOO_SHORT})
            continue
        n_fc = steps - end
        x_norm = (r["x"] - sm) / ss
        row = {"steps": steps, "category": cat, "forecast_steps": n_fc, "members": {}}
        for mid in members:
            p = rollout_falls(sim, x_norm, r["a"], end, n_fc, mid)
            # forecast step k predicts the outcome of command a[end + k], so the real fall is k = T - end
            k_fall = T - end
            pred = np.nonzero(p >= 0.5)[0]            # the lite threshold: sigmoid(...).round()
            row["members"]["mean" if mid is None else str(mid)] = {
                "hit": bool(((pred >= k_fall - TOL_STEPS) & (pred <= k_fall)).any()),
                "early": bool((pred < k_fall - TOL_STEPS).any()),
                "p_max_window": float(p[max(0, k_fall - TOL_STEPS):k_fall + 1].max()),
                "p_max_all": float(p.max())}
        rows.append(row)
        if (i + 1) % 5 == 0 or i + 1 == len(eps):
            write_progress(f"D1 falls {a.policy}", i + 1, len(eps))
    write_json(out, {"policy": a.policy, "ensemble": sim.ensemble_size, "falls": rows})
    print(f"[falls] {a.policy}: {len(rows)} real falls")


# ------------------------------------------------------------------ D1, imagination
def lite_experiment(policy, n_envs):
    """The lite env and simulator of one policy, in run_lite's order and with its start-state rule."""
    import train_policy as TP
    import train as T
    RL, C = TP.RL, TP.C
    row = TP.sim_row(sim_of(policy))
    cfg = T.resolve_task_config("anymal_d_flat")
    cfg.experiment_config.device = "cpu"
    cfg.environment_config.num_envs = n_envs
    cfg.data_config.dataset_root = os.path.join(RL.LITE, "assets")
    cfg.model_architecture_config.resume_path = None
    exp = T.ModelBasedExperiment(**cfg.experiment_config.to_dict())
    exp.prepare_environment(**cfg.environment_config.to_dict())
    exp.prepare_model(**cfg.model_architecture_config.to_dict())
    sim = TP.build_sim(row)
    exp.system_dynamics = sim
    exp.env.set_system_dynamics(sim)
    exp.prepare_policy(**cfg.policy_architecture_config.to_dict())
    exp.prepare_algorithm(**cfg.policy_algorithm_config.to_dict())
    exp.prepare_data(**cfg.data_config.to_dict())
    holdout = TP.HOLDOUT_OF_ARENA[str(row["arena"])]
    if holdout:
        keep = C.train_episodes([int(x) for x in holdout.split(",")])
        csv = os.path.join(RL.LITE, "assets", "data", "state_action_data_0.csv")
        h, f = cfg.model_architecture_config.history_horizon, cfg.model_architecture_config.forecast_horizon
        RL.restrict_windows(exp.dataset, csv, keep, h, f)
    return exp


def policy_std(policy):
    import torch
    run = lite_run_dir(policy)
    cks = glob.glob(os.path.join(run, "policy_*.pt"))
    last = max(cks, key=lambda f: int(os.path.basename(f)[7:-3]))
    return torch.load(last, map_location="cpu", weights_only=False)["model_state_dict"]["std"].clone()


def imagine(exp, actor, std, sample, max_steps, seed, policy, keep_pairs):
    """The first episode of every robot, stepped as PolicyTraining.learn steps it."""
    import torch
    torch.manual_seed(seed)
    env, names = exp.env, term_names()
    N = env.num_envs
    sh, ah = env.prepare_imagination()
    alive = torch.ones(N, dtype=torch.bool)
    fell, length = torch.zeros(N, dtype=torch.bool), torch.zeros(N, dtype=torch.long)
    sums = {t: torch.zeros(N) for t in names + ["uncertainty"]}
    S, A = [], []
    env_total = 0.0
    for k in range(max_steps):
        with torch.inference_mode():
            if env.common_step_counter > 0:          # as policy_training.py:65-67 (GRU models)
                sh, ah = sh[:, -1:], ah[:, -1:]
            obs = env.get_imagination_observation(sh, ah)["policy"]
            mean = actor(obs)
            act = mean + std * torch.randn_like(mean) if sample else mean
            if keep_pairs:
                raw, _ = env.dataset.denormalize(sh[:, -1], None)
                S.append(raw[alive].clone()), A.append(act[alive].clone())
            _, rew, dones, _, sh, ah, unc = env.imagination_step(act, sh, ah)
            env_total += float(rew[alive].sum())
            dt = env._step_dt
            for t in names:
                sums[t][alive] += env.reward_term_weights[t] * env.imagination_reward_per_step[t][alive] * dt
            sums["uncertainty"][alive] += env.uncertainty_penalty_weight * unc[alive] * dt
            length[alive] += 1
            done = alive & (dones > 0)
            term = env.termination_flags
            if term is not None:
                fell[done] = term[done]
            alive &= ~done
        if (k + 1) % 50 == 0:
            write_progress(f"D1 imagination {policy}", k + 1, max_steps)
        if not alive.any():
            break
    finished = ~alive
    total = int(length.sum())
    terms = {t: float(sums[t].sum()) / max(total, 1) for t in sums}
    own_total = sum(float(sums[t].sum()) for t in sums)
    assert abs(own_total - env_total) <= 1e-3 * max(1.0, abs(env_total)), (own_total, env_total)
    res = {"episodes": N, "finished": int(finished.sum()),
           "share_fell": float(fell[finished].float().mean()) if finished.any() else None,
           "mean_length": float(length.float().mean()),
           "mean_return": float(sum(sums[t] for t in sums).mean()),
           "mean_step_reward_per_term": terms}
    return res, (torch.cat(S).numpy(), torch.cat(A).numpy()) if keep_pairs else None


def part_imag(a):
    import torch
    torch.set_num_threads(a.threads)
    import s1_policy as P
    out = os.path.join(a.root, "imag", a.policy + ".json")
    if os.path.exists(out):
        print(f"[imag] {a.policy}: exists, skipped")
        return
    exp = lite_experiment(a.policy, a.n_envs)
    net = P.load_actor(actor_path(a.policy))
    std = policy_std(a.policy)
    res = {"policy": a.policy, "ensemble": exp.system_dynamics.ensemble_size,
           "max_episode_length": int(exp.env.max_episode_length)}
    res["sampled"], pairs = imagine(exp, net, std, True, a.imag_steps, IMAG_SEED, a.policy, True)
    res["mean_action"], _ = imagine(exp, net, std, False, a.imag_steps, IMAG_SEED + 1, a.policy, False)
    save_npz(os.path.join(a.root, "imag", a.policy + ".npz"), s=pairs[0], a=pairs[1])
    write_json(out, res)
    print(f"[imag] {a.policy}: fell {res['sampled']['share_fell']} sampled, {res['mean_action']['share_fell']} mean action")


# ------------------------------------------------------------------ D2
def training_pairs(arena):
    """(state row r, action row r+1) of the simulator's training episodes: raw units, 57 columns."""
    import common as C
    _, dat, ep = C.data()
    keep = set(C.train_episodes((1, 8))) if str(arena) == "18" else set(range(C.R.N_EPISODES))
    idx = np.array([r for r in range(len(dat) - 1) if ep[r] == ep[r + 1] and int(ep[r]) in keep])
    return np.concatenate([dat[idx][:, C.R.STATE_COLS], dat[idx + 1][:, C.R.ACTION_COLS]], axis=1)


def knn_distance(query, ref, k=K_NN, chunk=1000, tag=""):
    """Each query's mean Euclidean distance to its k nearest reference rows."""
    import torch
    q, r = torch.as_tensor(query, dtype=torch.float32), torch.as_tensor(ref, dtype=torch.float32)
    out = []
    for i in range(0, len(q), chunk):
        d = torch.cdist(q[i:i + chunk], r)
        out.append(d.topk(k, dim=1, largest=False).values.mean(1))
        write_progress(f"D2 {tag}", min(i + chunk, len(q)), len(q))
    return torch.cat(out).numpy()


def subsample(a, n, seed):
    rng = np.random.default_rng(seed)
    return a if len(a) <= n else a[np.sort(rng.choice(len(a), n, replace=False))]


def part_d2(a):
    import torch
    import train_policy as TP
    torch.set_num_threads(a.threads)
    out = os.path.join(a.root, "d2", a.policy + ".json")
    if os.path.exists(out):
        print(f"[d2] {a.policy}: exists, skipped")
        return
    row = TP.sim_row(sim_of(a.policy))
    train = training_pairs(row["arena"])
    scale = train.std(0)
    assert (scale > 0).all(), "a training column has no spread"

    def dist(pairs, tag):
        finite = np.isfinite(pairs).all(1)
        q = subsample(pairs[finite], a.n_points, D2_SEED)
        d = knn_distance(q / scale, train / scale, tag=f"{a.policy} {tag}")
        return {"n_points": int(len(q)), "n_nonfinite_dropped": int((~finite).sum()), "mean": float(d.mean()),
                "median": float(np.median(d)), "p90": float(np.percentile(d, 90))}

    im = np.load(os.path.join(a.root, "imag", a.policy + ".npz"))
    res = {"policy": a.policy, "n_training_pairs": int(len(train)), "imagined": dist(np.hstack([im["s"], im["a"]]), "imagined")}
    authors = load_phys(a.root, AUTHORS)
    pairs = np.vstack([np.hstack([r["x"][:-1], r["a"]]) for r in authors])
    res["authors_mujoco"] = dist(pairs, "authors")
    res["ratio_of_means"] = res["imagined"]["mean"] / res["authors_mujoco"]["mean"]
    write_json(out, res)
    print(f"[d2] {a.policy}: imagined {res['imagined']['mean']:.2f}, authors {res['authors_mujoco']['mean']:.2f}")


# ------------------------------------------------------------------ D3
def d3_modules():
    """Step 2's trainer and evaluator, unchanged, with every output moved under runs/c/ and results/c/."""
    import train_policy as TP
    row = TP.sim_row(D3_SIM)
    TP.sim_row = lambda sim: row
    TP.RUNS, TP.RES, TP.PROGRESS = RUNS_C, RES_D3, PROGRESS
    TP.AIR = TP.RL.AIR = LITE_ROOT                 # only the lite log directory is built from these after sim_row
    return TP


def part_d3train(a):
    TP = d3_modules()
    name = f"{D3_SIM}__p{a.seed}"
    argv = ["--sim", D3_SIM, "--seed", str(a.seed), "--route", "rebuilt", "--job", "diag", "--threads", str(a.threads)]
    if a.test:
        name = f"_test_d3_{a.seed}"
        argv += ["--name", name, "--iterations", "1", "--num_envs", "64"]
    rec = TP.main(argv)
    if a.test:
        unc = [d["Imagination/epistemic_uncertainty"] for d in TP.RL.LOGGED if "Imagination/epistemic_uncertainty" in d]
        assert unc and unc[-1] > 0, f"no uncertainty in the test iteration: {unc}"
        print(f"[d3train] {name}: epistemic uncertainty {unc[-1]:.4f}, ensemble {rec['sim_ensemble']}, route {rec['route']}")


def part_d3eval(a):
    import evaluate as E
    F = E.F
    E.RUNS, E.EVAL_DIR = RUNS_C, EVAL_D3
    F.RES_DIR, F.PROGRESS = os.path.dirname(PROGRESS), PROGRESS
    E.evaluate(f"{D3_SIM}__p{a.seed}", job="diag")


# ------------------------------------------------------------------ D4
def part_d4(a):
    import s1_policy as P
    out = os.path.join(a.root, "d4.json")
    if os.path.exists(out):
        print("[d4] exists, skipped")
        return
    regimes = json.load(open(P.REGIMES_JSON))["regimes"]
    train_eps = set(P.C.train_episodes((1, 8)))

    def coverage(cmds):
        cmds = np.unique(np.round(cmds, 12), axis=0)
        edges = [np.unique(np.clip(np.concatenate([cmds[:, i] - D4_HALF, cmds[:, i] + D4_HALF, [-1.0, 1.0]]), -1, 1))
                 for i in range(3)]
        mid = [(e[:-1] + e[1:]) / 2 for e in edges]
        wid = [e[1:] - e[:-1] for e in edges]
        grid = np.stack(np.meshgrid(*mid, indexing="ij"), -1).reshape(-1, 3)
        vol = np.multiply.reduce(np.stack(np.meshgrid(*wid, indexing="ij"), -1).reshape(-1, 3), axis=1)
        inside = (np.abs(grid[:, None] - cmds[None]) <= D4_HALF).all(2).any(1)
        exact = float(vol[inside].sum() / 8.0)
        rng = np.random.default_rng(D2_SEED)
        u = rng.uniform(-1, 1, (D4_MC, 3))
        mc = np.zeros(D4_MC, dtype=bool)
        for c in cmds:
            mc |= (np.abs(u - c) <= D4_HALF).all(1)
        return {"n_distinct_commands": int(len(cmds)), "share_exact": exact, "share_monte_carlo": float(mc.mean()),
                "monte_carlo_samples": D4_MC}

    every = np.array([[r["vx"], r["vy"], r["wz"]] for r in regimes])
    fold = np.array([[r["vx"], r["vy"], r["wz"]] for r in regimes if r["ep"] in train_eps])
    res = {"half_width": D4_HALF, "n_regimes": len(regimes), "all_ten_episodes": coverage(every),
           "eight_training_episodes": coverage(fold),
           "note": "commands are the reconstructed regime means of the recorded data; uniform on [-1,1]^3 as the plan "
                   "defines it (the lite env also gives 2% of robots a zero command)"}
    write_json(out, res)
    write_progress("D4 done", 1, 1)
    print(f"[d4] {res['all_ten_episodes']}")


# ------------------------------------------------------------------ report
def falls_summary(falls):
    out = {"n_real_falls": len(falls["falls"])}
    for cat in (CAT_FULL, CAT_SHORT, CAT_TOO_SHORT):
        out[f"n_{cat}"] = sum(r["category"] == cat for r in falls["falls"])
    for cat in (CAT_FULL, CAT_SHORT):
        rows = [r for r in falls["falls"] if r["category"] == cat]
        if not rows:
            continue
        mem = [k for k in rows[0]["members"] if k != "mean"]
        s = out[cat] = {}
        s["hit_share_ensemble_mean"] = float(np.mean([r["members"]["mean"]["hit"] for r in rows]))
        s["hit_share_member_average"] = float(np.mean([np.mean([r["members"][m]["hit"] for m in mem]) for r in rows]))
        s["hit_share_any_member"] = float(np.mean([any(r["members"][m]["hit"] for m in mem) for r in rows]))
        s["early_alarm_share_member_average"] = float(np.mean([np.mean([r["members"][m]["early"] for m in mem]) for r in rows]))
        s["mean_max_fall_probability_in_window"] = float(np.mean([np.mean([r["members"][m]["p_max_window"] for m in mem])
                                                                  for r in rows]))
    return out


def phys_summary(root, policy, names):
    eps = load_phys(root, policy)
    steps = sum(int(r["steps"]) for r in eps)
    terms = np.sum([r["terms"].sum(0) for r in eps], axis=0) / max(steps, 1)
    return {"episodes": len(eps), "fall_rate": float(np.mean([bool(r["fell"]) for r in eps])),
            "mean_steps": float(np.mean([int(r["steps"]) for r in eps])),
            "episodes_differing_from_step2": int(sum(int(r["mismatch"]) for r in eps)),
            "mean_return": float(np.mean([float(r["ret"]) for r in eps])),
            "mean_step_reward_per_term": dict(zip(names, map(float, terms)))}


def d0():
    import common as C
    _, dat, _ = C.data()
    th = dat[:, C.R.CONTACT_THIGH] > 0.5
    return {"rows": int(len(dat)), "fall_rows": int((dat[:, C.R.TERMINATION] > 0.5).sum()),
            "thigh_contact_rows": int(th.any(1).sum()), "thigh_contact_rows_per_leg": th.sum(0).tolist()}


def part_report(a):
    if a.root == default_root() and (os.path.exists(OUT_JSON) or os.path.exists(OUT_TXT)):
        raise SystemExit("results/c_diag.* exist; result files are never overwritten")
    rd = lambda *p: json.load(open(os.path.join(a.root, *p)))  # noqa: E731
    names = term_names()
    res = {"what": "Option C diagnosis of Step 2's rebuilt-simulator policies; MuJoCo here is the calibrated hand-built "
                   "ANYmal D, not Isaac Lab and not hardware. Imagined-episode shares are one seeded draw of 512 episodes.",
           "D0": d0(), "D1": {}, "D2": {}, "D3": {}, "D4": rd("d4.json")}
    for p in PILOT:
        res["D1"][p] = {"imagination": rd("imag", p + ".json"), "physics": phys_summary(a.root, p, names),
                        "real_falls": falls_summary(rd("falls", p + ".json"))}
        res["D2"][p] = rd("d2", p + ".json")
    for p, seed in zip(D3, D3_SEEDS):
        ev = json.load(open(os.path.join(a.eval_dir, p + ".json")))
        res["D3"][p] = {"imagination": rd("imag", p + ".json"), "seed": seed, "fall_rate": ev["summary"]["fall_rate"],
                        "fall_rate_ci95": ev["summary"]["fall_rate_ci95"], "ret": ev["summary"]["ret"],
                        "ret_ci95": ev["summary"]["ret_ci95"], "speed_err": ev["summary"]["speed_err"],
                        "turn_err": ev["summary"]["turn_err"], "two_feet": ev["summary"]["two_feet"]}
    d4 = res["D4"]["all_ten_episodes"]
    L = ["Option C diagnosis (descriptive; hand-built calibrated MuJoCo ANYmal D, not Isaac Lab or hardware)", "",
         f"D0 recorded data: {res['D0']['rows']} rows, {res['D0']['fall_rows']} with a fall, "
         f"{res['D0']['thigh_contact_rows']} with thigh contact", "",
         "D1 imagination (512 episodes, sampled actions as in training; mean action alongside)",
         f"{'policy':16s}{'fell':>7s}{'fell/mean':>10s}{'length':>8s}{'imag return':>12s}{'uncert/step':>12s}"]
    for p in PILOT + D3:
        r = res["D1"][p]["imagination"] if p in res["D1"] else res["D3"][p]["imagination"]
        s, m = r["sampled"], r["mean_action"]
        L.append(f"{p:16s}{s['share_fell']:7.3f}{m['share_fell']:10.3f}{s['mean_length']:8.1f}{s['mean_return']:12.2f}"
                 f"{s['mean_step_reward_per_term']['uncertainty']:12.4f}")
    L += ["", "D1 physics and real falls (80 episodes; history 32, last 50 steps; hit within 0.2 s, member average)",
          f"{'policy':16s}{'falls':>7s}{'full':>6s}{'short':>7s}{'<33':>5s}{'hit full':>10s}{'hit short':>11s}"]
    for p in PILOT:
        f, ph = res["D1"][p]["real_falls"], res["D1"][p]["physics"]
        hit = lambda c: f"{f[c]['hit_share_member_average']:.2f}" if c in f else "-"  # noqa: E731
        L.append(f"{p:16s}{ph['fall_rate']:7.3f}{f['n_full']:6d}{f['n_short']:7d}{f['n_too_short']:5d}"
                 f"{hit(CAT_FULL):>10s}{hit(CAT_SHORT):>11s}")
    L += ["", "D2 distance from training data (mean over 10 neighbours, columns scaled by the data's spread)",
          f"{'policy':16s}{'imagined':>10s}{'authors':>10s}{'ratio':>7s}"]
    for p in PILOT:
        r = res["D2"][p]
        L.append(f"{p:16s}{r['imagined']['mean']:10.2f}{r['authors_mujoco']['mean']:10.2f}{r['ratio_of_means']:7.2f}")
    L += ["", "D3 the penalty alone: policies in A5_s0 (five networks, no falls in its data; Step 2's protocol)",
          f"{'policy':16s}{'falls':>7s}{'falls 95%':>18s}{'return':>9s}{'return 95%':>20s}{'speed':>8s}{'turn':>7s}"]
    for p in D3:
        r = res["D3"][p]
        fl, rl = r["fall_rate_ci95"], r["ret_ci95"]
        L.append(f"{p:16s}{r['fall_rate']:7.3f}  [{fl[0]:.3f}, {fl[1]:.3f}]{r['ret']:9.2f}  [{rl[0]:7.2f}, {rl[1]:7.2f}]"
                 f"{r['speed_err']:8.3f}{r['turn_err']:7.3f}")
    L += ["", f"D4 command coverage: {d4['share_exact']:.3f} of training commands lie within {D4_HALF} on every axis of "
              f"one of {d4['n_distinct_commands']} recorded commands"]
    assert len(L) <= 40, len(L)
    json_path = OUT_JSON if a.root == default_root() else os.path.join(a.root, "c_diag.json")
    txt_path = OUT_TXT if a.root == default_root() else os.path.join(a.root, "c_diag.txt")
    write_json(json_path, res)
    with open(txt_path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    write_progress("report written", 1, 1)
    print("\n".join(L))


# ------------------------------------------------------------------ main
def default_root():
    return os.path.join(RUNS_C, "diag")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("phys", "falls", "imag", "d2", "d3train", "d3eval", "d4", "report"))
    ap.add_argument("--policy", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--root", default=default_root(), help="work directory (tests use .cache/)")
    ap.add_argument("--limit", type=int, default=0, help="episodes or falls (tests only)")
    ap.add_argument("--n_envs", type=int, default=IMAG_EPISODES, help="imagined episodes (tests only)")
    ap.add_argument("--imag_steps", type=int, default=256, help="most imagined steps (the lite episode limit is 256)")
    ap.add_argument("--n_points", type=int, default=N_D2, help="points per D2 query set (tests only)")
    ap.add_argument("--eval_dir", default=EVAL_D3, help="D3 evaluation files (tests use .cache/)")
    ap.add_argument("--test", action="store_true", help="d3train: one iteration, 64 robots, no result file")
    a = ap.parse_args()
    {"phys": part_phys, "falls": part_falls, "imag": part_imag, "d2": part_d2, "d3train": part_d3train,
     "d3eval": part_d3eval, "d4": part_d4, "report": part_report}[a.part](a)


if __name__ == "__main__":
    main()

"""Option C (C2R, Amendment 1 changes 1-2): the calibrated uncertainty weight of each simulator, and Gate A2.

    ./py-mj scripts/c/calibrate_penalty.py mujoco     # -> results/c/authors_terms.json   (the MuJoCo reference)
    ./py    scripts/c/calibrate_penalty.py imagine    # -> results/c/calibration.json      (lambda_s and Gate A2)

The authors' policy is run closed-loop in each simulator's imagination, as policy training steps it (the lite env and
reward, unchanged; scripts/c/diag.py's `imagine`): 1,024 robots, 256 steps, noise-free actions (the policy's mean),
each robot's first episode only. Start states: for our simulators, the Amendment 1 rule (source-0 windows of the
simulator's own rung, upright, at least 1 s before any fall; scripts/c/train_policy.py); for the released model and
A5_s0, the authors' recorded data, as the lite code starts (A5_s0 was trained on those data; Step 2's hold-out applies).

  * u_bar(s): the mean ensemble disagreement per step along those episodes: the lite code's own measure (the sum over
    state dimensions of the standard deviation across the five networks' means; scripts/lite/adapter.py, equal to
    rsl_rl's SystemDynamicsEnsemble), averaged over every step every robot is in its first episode;
  * lambda_s = u_bar(REL) / u_bar(s), clipped to 0.05-2 (so lambda_REL = 1, RWM-U's weight);
  * also: the policy's imagined reward per step by term (weight x term x 0.02 s, the lite units) and its share of
    predicted falls;
  * Gate A2 (sim_6M_N8_s0 only): at most 10% of the 1,024 imagined episodes end in a predicted fall, and the two
    tracking terms together earn at least 80% of what the same policy earns per step in MuJoCo.

The MuJoCo reference re-runs the authors' policy on the walking protocol of results/c_authors.json (24 schedules x 4
starts, calibrated physics, no pushes) with Step 2's reward recorded per term (mj_env.reward_fn: the lite reward terms,
weight x term x 0.02 s); its walking score must equal results/c_authors.json's, which checks it is the same protocol.
The world is the hand-built calibrated MuJoCo ANYmal D with a fitted PD motor, not Isaac Lab and not hardware.
Both outputs are written once and never overwritten.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
MJ_OUT = os.path.join(AIR, "results", "c", "authors_terms.json")
OUT = os.path.join(AIR, "results", "c", "calibration.json")
PROGRESS = os.path.join(AIR, "results", "c", "progress.json")
SIMS = ("REL", "sim_6M_N8_s0", "sim_10k_N8_s0", "A5_s0")
GATE_SIM = "sim_6M_N8_s0"
N_ROBOTS, N_STEPS = 1024, 256
CAL_SEED = 20278
CLIP = (0.05, 2.0)
A2_FALL, A2_TRACK = 0.10, 0.80
TRACK = ("track_lin_vel_xy_exp", "track_ang_vel_z_exp")


def progress(job, step, done, total):
    os.makedirs(os.path.dirname(PROGRESS), exist_ok=True)
    with open(PROGRESS + ".tmp", "w") as fh:
        json.dump({"job": job, "step": step, "done": done, "total": total,
                   "time": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh)
    os.replace(PROGRESS + ".tmp", PROGRESS)


def write_once(path, obj):
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; result files are never overwritten")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(path + ".tmp", path)


# ------------------------------------------------------------------ MuJoCo reference (./py-mj)
def part_mujoco(a):
    import evaluate as EV
    F = EV.F
    import s1_policy as P
    import torch
    torch.set_num_threads(a.threads)
    if os.path.exists(a.mj_out):
        print(f"{a.mj_out} exists, kept")
        return
    actor = F.torch_actor(P.load_actor(None))
    env = F.make_env("cal", reward=True)
    step = env.step
    rec = []

    def stepped(action):
        out = step(action)
        rec.append(np.array(out[1]["terms"], dtype=np.float64))
        return out
    env.step = stepped
    per_ep, scores, steps = [], [], 0
    plan = [(e, j) for e in range(EV.N_SCHEDULES) for j in range(EV.N_STARTS)]
    for i, (e, j) in enumerate(plan):
        rec.clear()
        off, yaw = EV.walk_start(e, j)
        r = EV.run_episode(env, actor, EV.walk_schedule(e), off, yaw, e, j)
        t = np.array(rec)
        assert not r["fell"] and len(t) == EV.STEPS, (e, j, r["fell"], len(t))
        per_ep.append(t.sum(0))
        scores.append(r["score"])
        steps += len(t)
        if (i + 1) % 8 == 0:
            progress(a.job, "calibration: authors' policy in MuJoCo", i + 1, len(plan))
    P.C.data()
    W, _ = P.reward_weights()
    names = list(W.keys())
    mean = np.sum(per_ep, 0) / steps
    authors = json.load(open(os.path.join(AIR, "results", "c_authors.json")))
    score = float(np.mean(scores))
    same = abs(score - authors["score"]) <= 1e-9
    res = {"what": "the authors' policy in the calibrated hand-built MuJoCo ANYmal D (fitted PD motor; not Isaac Lab, "
                   "not hardware), walking protocol of results/c_authors.json, reward per step by term (weight x term "
                   "x 0.02 s, the lite units)",
           "episodes": len(plan), "steps": steps, "terms": names,
           "per_step": {n: float(v) for n, v in zip(names, mean)},
           "tracking_per_step": float(sum(mean[names.index(t)] for t in TRACK)),
           "score": score, "score_c_authors": authors["score"], "same_protocol_as_c_authors": same}
    if not same:
        raise SystemExit(f"walking score {score} differs from results/c_authors.json's {authors['score']}")
    write_once(a.mj_out, res)
    print(json.dumps({k: res[k] for k in ("tracking_per_step", "score", "same_protocol_as_c_authors")}))


# ------------------------------------------------------------------ imagination (./py)
def step2_row(sim):
    """REL or a pool-F18 row of results/models.json (Step 2's table), read directly (train_policy rebinds TP.AIR)."""
    table = json.load(open(os.path.join(AIR, "results", "models.json")))
    if sim == "REL":
        return table["REL"][0]
    rows = [m for m in table["F18"] if m["key"] == sim]
    if len(rows) != 1:
        raise SystemExit(f"{sim} is not one row of pool F18")
    return rows[0]


def experiment(sim, n_envs):
    """The lite env with this simulator and its start states (see the module notes)."""
    import torch
    import train_policy as W
    TP, RL, C = W.TP, W.RL, W.C
    import train as T
    cfg = T.resolve_task_config("anymal_d_flat")
    cfg.experiment_config.device = "cpu"
    cfg.environment_config.num_envs = n_envs
    cfg.data_config.dataset_root = os.path.join(RL.LITE, "assets")
    cfg.model_architecture_config.resume_path = None
    exp = T.ModelBasedExperiment(**cfg.experiment_config.to_dict())
    exp.prepare_environment(**cfg.environment_config.to_dict())
    exp.prepare_model(**cfg.model_architecture_config.to_dict())
    ours = sim.startswith("sim_")
    row = W.sim_row_of(sim) if ours else step2_row(sim)
    model = TP.build_sim(row)
    exp.system_dynamics = model
    exp.env.set_system_dynamics(model)
    exp.prepare_policy(**cfg.policy_architecture_config.to_dict())
    exp.prepare_algorithm(**cfg.policy_algorithm_config.to_dict())
    exp.prepare_data(**cfg.data_config.to_dict())
    if ours:
        ck = torch.load(W.sim_path(sim), map_location="cpu", weights_only=False)
        st, ac, _, info = W.build_starts(ck["rung"], ck["state_mean"], ck["state_std"], source=W.START_SOURCE)
        W.install_starts(exp.dataset, st, ac, ck["state_mean"], ck["state_std"])
        starts = {"rule": "Amendment 1: source-0 windows of the simulator's own rung", **info}
    else:
        holdout = TP.HOLDOUT_OF_ARENA[str(row["arena"])]
        kept = None
        if holdout:
            keep = C.train_episodes([int(x) for x in holdout.split(",")])
            csv = os.path.join(RL.LITE, "assets", "data", "state_action_data_0.csv")
            h, f = cfg.model_architecture_config.history_horizon, cfg.model_architecture_config.forecast_horizon
            kept = RL.restrict_windows(exp.dataset, csv, keep, h, f)
        starts = {"rule": "the authors' recorded data (the lite code's own start states)", "holdout": holdout or None,
                  "windows_kept_total": kept}
    assert exp.env.init_dataset is exp.dataset
    return exp, starts


def imagine(exp, actor, seed, n_steps=N_STEPS):
    """The first episode of every robot under noise-free actions (as scripts/c/diag.py's imagine, with disagreement)."""
    import torch
    import s1_policy as P
    P.C.data()
    names = list(P.reward_weights()[0].keys())
    torch.manual_seed(seed)
    env = exp.env
    N = env.num_envs
    sh, ah = env.prepare_imagination()
    alive = torch.ones(N, dtype=torch.bool)
    fell, length = torch.zeros(N, dtype=torch.bool), torch.zeros(N, dtype=torch.long)
    sums = {t: torch.zeros(N, dtype=torch.float64) for t in names + ["uncertainty"]}
    unc_sum = torch.zeros(N, dtype=torch.float64)
    env_total = 0.0
    for k in range(n_steps):
        with torch.inference_mode():
            if env.common_step_counter > 0:          # as policy_training.py (GRU models)
                sh, ah = sh[:, -1:], ah[:, -1:]
            obs = env.get_imagination_observation(sh, ah)["policy"]
            act = actor(obs)
            _, rew, dones, _, sh, ah, unc = env.imagination_step(act, sh, ah)
            env_total += float(rew[alive].double().sum())
            dt = env._step_dt
            for t in names:
                sums[t][alive] += (env.reward_term_weights[t] * env.imagination_reward_per_step[t][alive] * dt).double()
            sums["uncertainty"][alive] += (env.uncertainty_penalty_weight * unc[alive] * dt).double()
            unc_sum[alive] += unc[alive].double()
            length[alive] += 1
            done = alive & (dones > 0)
            if env.termination_flags is not None:
                fell[done] = env.termination_flags[done]
            alive &= ~done
        if not alive.any():
            break
    total = int(length.sum())
    own = sum(float(sums[t].sum()) for t in sums)
    assert abs(own - env_total) <= 1e-3 * max(1.0, abs(env_total)), (own, env_total)
    per_step = {t: float(sums[t].sum()) / total for t in sums}
    return {"robots": N, "steps_run": k + 1, "episodes_finished": int((~alive).sum()),
            "share_fell": float(fell.float().mean()), "falls": int(fell.sum()),
            "mean_length": float(length.float().mean()), "u_bar": float(unc_sum.sum()) / total,
            "uncertainty_weight_in_env": float(env.uncertainty_penalty_weight),
            "per_step": per_step, "tracking_per_step": sum(per_step[t] for t in TRACK),
            "total_per_step": sum(per_step.values())}


def part_imagine(a):
    import torch
    torch.set_num_threads(a.threads)
    import train_policy as W  # noqa: F401  (sets the lite paths)
    import s1_policy as P
    if os.path.exists(a.out):
        print(f"{a.out} exists, kept")
        return json.load(open(a.out))
    if not os.path.exists(a.mj_out):
        raise SystemExit(f"{a.mj_out} is missing: run the mujoco stage first (./py-mj)")
    mj = json.load(open(a.mj_out))
    actor = P.load_actor(None)
    actor.eval()
    sims = a.sims.split(",")
    if a.ref not in sims:
        sims = [a.ref] + sims
    res = {}
    for i, sim in enumerate(sims):
        progress(a.job, f"calibration: imagination in {sim}", i, len(sims))
        t0 = time.time()
        exp, starts = experiment(sim, N_ROBOTS)
        r = imagine(exp, actor, CAL_SEED)
        r.update(start_states=starts, seconds=time.time() - t0)
        res[sim] = r
        print(f"[calibrate] {sim}: u_bar {r['u_bar']:.4f}, fell {r['share_fell']:.3f}, tracking/step "
              f"{r['tracking_per_step']:.5f}, length {r['mean_length']:.1f} ({r['seconds']:.0f} s)", flush=True)
    progress(a.job, "calibration", len(sims), len(sims))
    u_ref = res[a.ref]["u_bar"]
    for sim, r in res.items():
        raw = u_ref / r["u_bar"]
        r["lambda_unclipped"] = raw
        r["lambda"] = float(min(max(raw, CLIP[0]), CLIP[1]))
        r["clipped"] = raw != r["lambda"]
    out = {"what": "Amendment 1: the calibrated uncertainty weight lambda_s = u_bar(REL) / u_bar(s), clipped to "
                   "0.05-2, from the authors' policy run closed-loop (noise-free) in each simulator's imagination; "
                   "Gate A2 on the 6M simulator. World: hand-built calibrated MuJoCo ANYmal D, fitted PD motor; not "
                   "Isaac Lab, not hardware",
           "settings": {"robots": N_ROBOTS, "steps": N_STEPS, "seed": CAL_SEED, "actions": "noise-free (policy mean)",
                        "episodes": "each robot's first", "clip": list(CLIP), "reference": a.ref,
                        "disagreement": "lite measure: sum over state dimensions of the std across the networks' means"},
           "method": "closed-loop imagination (not the open-loop fallback)",
           "mujoco_reference": {k: mj[k] for k in ("tracking_per_step", "per_step", "score", "episodes")},
           "simulators": res}
    if a.gate_sim in res:
        g = res[a.gate_sim]
        need = A2_TRACK * mj["tracking_per_step"]
        out["gate_a2"] = {"simulator": a.gate_sim, "share_fell": g["share_fell"], "fall_limit": A2_FALL,
                          "fall_ok": g["share_fell"] <= A2_FALL,
                          "tracking_per_step": g["tracking_per_step"],
                          "tracking_mujoco_per_step": mj["tracking_per_step"],
                          "tracking_ratio": g["tracking_per_step"] / mj["tracking_per_step"],
                          "tracking_needed": need, "tracking_ok": g["tracking_per_step"] >= need}
        out["gate_a2"]["passed"] = out["gate_a2"]["fall_ok"] and out["gate_a2"]["tracking_ok"]
    write_once(a.out, out)
    lines = [f"{s:16s} u_bar {r['u_bar']:.4f} lambda {r['lambda']:.4f}{' (clipped)' if r['clipped'] else ''} "
             f"fell {r['share_fell']:.3f} tracking/step {r['tracking_per_step']:.5f} total/step {r['total_per_step']:.5f}"
             for s, r in res.items()]
    if "gate_a2" in out:
        g = out["gate_a2"]
        lines.append(f"Gate A2 ({g['simulator']}): {'PASSED' if g['passed'] else 'FAILED'}; fell {g['share_fell']:.3f} "
                     f"(limit {A2_FALL}), tracking {g['tracking_ratio']:.3f} of MuJoCo's (need {A2_TRACK})")
    print("\n".join(lines))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("mujoco", "imagine"))
    ap.add_argument("--sims", default=",".join(SIMS))
    ap.add_argument("--ref", default="REL")
    ap.add_argument("--gate-sim", default=GATE_SIM)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--mj-out", default=MJ_OUT)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--job", default="calibration")
    a = ap.parse_args(argv)
    if a.stage == "mujoco":
        part_mujoco(a)
    else:
        out = part_imagine(a)
        if "gate_a2" in out and not out["gate_a2"]["passed"]:
            return 4                       # Gate A2 failed: the job retrains (exit 4 is not an error of the script)
    return 0


if __name__ == "__main__":
    sys.exit(main())

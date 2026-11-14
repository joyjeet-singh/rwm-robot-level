"""Pilot (cloud copy): train a policy in a simulator's imagination with the authors' lite code, then save actors."""
import os, sys, json, time, argparse
import numpy as np, torch
import cl
C = cl.C
sys.path.insert(0, os.path.join(cl.PDM, "rsl_rl_rwm"))
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO
from rsl_rl.utils import resolve_obs_groups
from policy_training import PolicyTraining
from adapter import LiteDynamicsAdapter
import wandb

ap = argparse.ArgumentParser()
ap.add_argument("--sim", required=True, help="REL or a checkpoint path (model_state_dict, state_mean, state_std)")
ap.add_argument("--tag", required=True); ap.add_argument("--iters", type=int, default=500)
ap.add_argument("--envs", type=int, default=4096); ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--unc", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--save-every", type=int, default=100); ap.add_argument("--noise-floor", type=float, default=0.0)
ap.add_argument("--starts", default="mj", help="mj: MuJoCo source-0 windows of the 100k rung; csv: the authors' recorded data")
ap.add_argument("--resume", default="", help="a policy_<it>.pt saved by this script: continue from it (--iters = further iterations)")
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.manual_seed(a.seed); np.random.seed(a.seed)
am, asd = cl.auth_stats()
cfg = C.data()[0]
if a.sim == "REL":
    net = C.load(C.W_RELEASED, 5); mean, std = am, asd
else:
    ck = torch.load(a.sim, map_location="cpu", weights_only=False)
    net = C.M.build_from_config(cfg, ensemble_size=5); net.load_state_dict(ck["model_state_dict"]); net.eval()
    mean, std = np.asarray(ck["state_mean"], dtype=np.float64), np.asarray(ck["state_std"], dtype=np.float64)
for p_ in net.parameters(): p_.requires_grad_(False)
starts = cl.rung_windows("100k", mean, std, source=0) if a.starts == "mj" else cl.csv_windows(mean, std)
env = cl.make_env(LiteDynamicsAdapter(net), mean, std, starts, n_envs=a.envs, unc_w=-abs(a.unc))
obs_groups = resolve_obs_groups(env.dummy_obs, {"policy": ["policy"]}, ["critic"])
ac = ActorCritic(obs=env.dummy_obs, obs_groups=obs_groups, num_actions=12, actor_hidden_dims=[128, 128, 128],
                 critic_hidden_dims=[128, 128, 128], activation="elu", init_noise_std=1.0)
alg = PPO(ac, 5, 4, 0.2, 0.99, 0.95, 1.0, 0.005, 1.0e-3, 1.0, True, "adaptive", 0.01, device="cpu")
out = os.path.join("out", a.tag); os.makedirs(out, exist_ok=True)
pt = PolicyTraining(out, env=env, alg=alg, device="cpu", num_steps_per_env=24, save_interval=a.save_every, max_iterations=a.iters)
if a.resume:
    ck_ = torch.load(a.resume, map_location="cpu", weights_only=False)
    alg.policy.load_state_dict(ck_["model_state_dict"]); alg.optimizer.load_state_dict(ck_["optimizer_state_dict"])
    alg.learning_rate = alg.optimizer.param_groups[0]["lr"]
    pt.current_learning_iteration = int(ck_["iter"]) + 1
    print("resumed from", a.resume, "at iteration", pt.current_learning_iteration, "lr", alg.learning_rate, flush=True)
hist = []
orig_log = pt.log
def log(locs, *args, **kw):
    d = pt.log_dict
    it = locs["it"]
    # the lite log fills log_dict; keep a compact record
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        orig_log(locs, *args, **kw)
    rec = {"it": it, "len": d.get("Train/mean_episode_length"), "rew": d.get("Train/mean_reward"),
           "unc": float(d.get("Imagination/epistemic_uncertainty", float("nan"))),
           "lin": float(d.get("Imagination/track_lin_vel_xy_exp", float("nan"))), "ang": float(d.get("Imagination/track_ang_vel_z_exp", float("nan"))),
           "tilt": float(d.get("Imagination/flat_orientation_l2", float("nan"))), "vz": float(d.get("Imagination/lin_vel_z_l2", float("nan"))),
           "uncr": float(d.get("Imagination/uncertainty", float("nan"))), "std": float(d.get("Policy/mean_noise_std", float("nan"))),
           "sec": locs["collection_time"] + locs["learn_time"]}
    hist.append(rec)
    if a.noise_floor > 0:
        with torch.no_grad():
            pol = alg.policy
            if hasattr(pol, "std"): pol.std.clamp_(min=a.noise_floor)
    if it % 10 == 0 or it < 3:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    if (it + 1) % a.save_every == 0:
        sd = {k: v.clone() for k, v in alg.policy.state_dict().items() if k.startswith("actor.")}
        torch.save({"model_state_dict": sd, "iter": it + 1}, os.path.join(out, f"actor_i{it+1}.pt"))
        json.dump(hist, open(os.path.join(out, "hist_resumed.json" if a.resume else "hist.json"), "w"))
pt.log = log
t0 = time.time()
pt.learn()
json.dump(hist, open(os.path.join(out, "hist_resumed.json" if a.resume else "hist.json"), "w"))
print("done", round(time.time() - t0), "s", flush=True)

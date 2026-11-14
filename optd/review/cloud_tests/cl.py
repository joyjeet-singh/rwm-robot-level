"""Closed-loop diagnostics of learned simulators in the lite imagination env (cloud copy; reads only)."""
import json
import os
import sys

import numpy as np
import torch

PDM = "/mnt/user-data/uploads/PDM"
AIR = os.path.join(PDM, "AIR 2027")
for p in (os.path.join(AIR, "scripts", "lite", "stubs"), os.path.join(PDM, "robotic_world_model_lite", "scripts"),
          os.path.join(AIR, "scripts", "lite"), os.path.join(AIR, "scripts")):
    sys.path.insert(0, p)
import common as C  # noqa: E402
from adapter import LiteDynamicsAdapter  # noqa: E402
import s1_policy as P  # noqa: E402
from envs.anymal_d_flat import AnymalDFlatEnv  # noqa: E402

DATA = os.path.join(AIR, "optc", "data")
H = 32
AUTH_MEAN = None
AUTH_STD = None


def auth_stats():
    cfg = C.data()[0]
    return np.asarray(cfg["state_data_mean"], dtype=np.float64), np.asarray(cfg["state_data_std"], dtype=np.float64)


class DS:
    """Stand-in for the lite dataset: start windows and the model's normalisation."""

    def __init__(self, st, ac, mean, std):
        self.st, self.ac = st, ac
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)

    def sample_batch(self, n, normalized=True):
        idx = torch.randint(0, len(self.st), (n,))
        return self.st[idx].clone(), self.ac[idx].clone()

    def normalize(self, state, action):
        return ((state - self.mean) / self.std if state is not None else None), action

    def denormalize(self, state, action):
        return (state * self.std + self.mean if state is not None else None), action


def csv_windows(mean, std, episodes=range(10)):
    _, dat, ep = C.data()
    starts = [s for s in range(len(dat) - H) if ep[s] == ep[s + H - 1] and ep[s] in episodes]
    idx = np.asarray(starts)[:, None] + np.arange(H)
    rows = dat[idx]
    st = torch.as_tensor((rows[:, :, :45] - mean) / std, dtype=torch.float32)
    ac = torch.as_tensor(rows[:, :, 45:57], dtype=torch.float32)
    return st, ac


def rung_windows(rung, mean, std, source=0, margin=50, tilt_deg=30.0, max_windows=60000, arr=None):
    a = np.load(os.path.join(DATA, f"rung_{rung}.npy"), mmap_mode="r") if arr is None else arr
    a = np.asarray(a)
    ep, src = a[:, 66].astype(np.int64), a[:, 67].astype(np.int64)
    fall = a[:, 65] > 0.5
    n = len(a)
    ok = np.zeros(n, dtype=bool)
    cosmax = np.cos(np.radians(tilt_deg))
    upright = -a[:, 8] >= cosmax
    # steps until the next fall within the same episode
    nxt = np.full(n, 10 ** 9)
    last = 10 ** 9
    for i in range(n - 1, -1, -1):
        if i + 1 < n and ep[i + 1] != ep[i]:
            last = 10 ** 9
        if fall[i]:
            last = i
        nxt[i] = last - i
    cu = np.cumsum(np.r_[0, (~upright).astype(np.int64)])
    for s in range(0, n - H + 1):
        e = s + H - 1
        if ep[s] != ep[e] or (source is not None and src[s] != source):
            continue
        if cu[e + 1] - cu[s] > 0 or nxt[e] < margin:
            continue
        ok[s] = True
    starts = np.flatnonzero(ok)
    if len(starts) > max_windows:
        starts = np.random.default_rng(0).choice(starts, max_windows, replace=False)
    idx = starts[:, None] + np.arange(H)
    rows = a[idx]
    st = torch.as_tensor((rows[:, :, :45] - mean) / std, dtype=torch.float32)
    ac = torch.as_tensor(rows[:, :, 45:57], dtype=torch.float32)
    return st, ac


def load(name, rung="100k", source=0):
    """-> (adapter, mean, std, (st, ac))"""
    am, asd = auth_stats()
    if name == "REL":
        net = C.load(C.W_RELEASED, 5)
        return LiteDynamicsAdapter(net), am, asd, csv_windows(am, asd)
    if name == "A5_s0":
        net = C.load(os.path.join(PDM, "rwm_repro", "runs", "armA_seed0_ens5", "weights_2500.pt"), 5)
        return LiteDynamicsAdapter(net), am, asd, csv_windows(am, asd, episodes=[0, 2, 3, 4, 5, 6, 7, 9])
    path = os.path.join(AIR, "runs", "c", name, "final.pt") if not os.path.isabs(name) else name
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = C.data()[0]
    net = C.M.build_from_config(cfg, ensemble_size=5)
    net.load_state_dict(ck["model_state_dict"], strict=True)
    net.eval()
    mean, std = np.asarray(ck["state_mean"], dtype=np.float64), np.asarray(ck["state_std"], dtype=np.float64)
    return LiteDynamicsAdapter(net), mean, std, rung_windows(rung, mean, std, source=source)


def make_env(adapter, mean, std, starts, n_envs=1024, events=True, noise=True, mean_pred=False, unc_w=-1.0,
             event_range=(48, 96), max_len=256):
    W, _ = P.reward_weights()
    env = AnymalDFlatEnv(n_envs, max_len, 0.02, W, "cpu", uncertainty_penalty_weight=unc_w, observation_noise=noise,
                         command_resample_interval_range=[100, 120],
                         event_interval_range=list(event_range) if events else None)
    ds = DS(starts[0], starts[1], mean, std)
    if mean_pred:
        fwd = adapter.forward
        adapter_forward = lambda s, a, ids=None: fwd(s, a, None)  # noqa: E731
        adapter.forward = adapter_forward
    env.set_system_dynamics(adapter)
    env.set_dataset(ds)
    env.set_init_dataset(ds, 0.8)
    return env


def imagine(env, actor, seed=20278, n_steps=256):
    """First episode of every robot, noise-free actions; per-step traces (N, T) with NaN after the episode ends."""
    torch.manual_seed(seed)
    N = env.num_envs
    sh, ah = env.prepare_imagination()
    alive = torch.ones(N, dtype=torch.bool)
    T = n_steps
    tr = {k: np.full((N, T), np.nan, dtype=np.float32) for k in
          ("lin", "ang", "vz2", "tilt2", "unc", "term_logit", "gz", "vx", "vy", "wz", "torq2", "event", "cx", "cy", "cw",
           "thigh", "feet")}
    fell = np.zeros(N, dtype=bool)
    length = np.zeros(N, dtype=np.int64)
    ds = env.dataset
    for k in range(T):
        with torch.inference_mode():
            if env.common_step_counter > 0:
                sh, ah = sh[:, -1:], ah[:, -1:]
            obs = env.get_imagination_observation(sh, ah)["policy"]
            act = actor(obs)
            # replicate the env step but keep the raw termination logits and the pre-event state
            cmd = env.base_velocity.clone()
            ev_before = (env.episode_length_buf + 1) % env.event_intervals == 0 if env.event_interval_range else \
                torch.zeros(N, dtype=torch.bool)
            _, rew, dones, _, sh, ah, unc = env.imagination_step(act, sh, ah)
            a = alive.numpy()
            rp = env.imagination_reward_per_step
            tr["lin"][a, k] = rp["track_lin_vel_xy_exp"][alive].numpy()
            tr["ang"][a, k] = rp["track_ang_vel_z_exp"][alive].numpy()
            tr["vz2"][a, k] = rp["lin_vel_z_l2"][alive].numpy()
            tr["tilt2"][a, k] = rp["flat_orientation_l2"][alive].numpy()
            tr["torq2"][a, k] = rp["dof_torques_l2"][alive].numpy()
            tr["thigh"][a, k] = rp["undesired_contacts"][alive].numpy()
            tr["unc"][a, k] = unc[alive].numpy()
            lo = env.last_obs["policy"]
            tr["vx"][a, k] = lo[alive, 0].numpy(); tr["vy"][a, k] = lo[alive, 1].numpy()
            tr["wz"][a, k] = lo[alive, 5].numpy(); tr["gz"][a, k] = lo[alive, 8].numpy()
            tr["cx"][a, k] = cmd[alive, 0].numpy(); tr["cy"][a, k] = cmd[alive, 1].numpy(); tr["cw"][a, k] = cmd[alive, 2].numpy()
            tr["event"][a, k] = ev_before[alive].float().numpy()
            length[a] += 1
            done = alive & (dones > 0)
            if env.termination_flags is not None:
                fell[done.numpy()] = env.termination_flags[done].numpy()
            alive &= ~done
        if not alive.any():
            break
    return {"tr": tr, "fell": fell, "length": length}


def summarise(r, label=""):
    tr, fell, length = r["tr"], r["fell"], r["length"]
    steps = np.isfinite(tr["lin"])
    tot = steps.sum()
    dt = 0.02
    track = (np.nansum(tr["lin"]) * 1.0 + np.nansum(tr["ang"]) * 0.5) * dt / tot
    per_robot = (np.nansum(tr["lin"], 1) + 0.5 * np.nansum(tr["ang"], 1)) * dt / np.maximum(1, steps.sum(1))
    tilt = np.degrees(np.arcsin(np.clip(np.sqrt(tr["tilt2"]), 0, 1)))
    out = {
        "fell": float(fell.mean()), "len": float(length.mean()),
        "track/step": float(track), "track med robot": float(np.median(per_robot)),
        "vz pen/step": float(-2.0 * np.nansum(tr["vz2"]) * dt / tot),
        "tilt pen/step": float(-5.0 * np.nansum(tr["tilt2"]) * dt / tot),
        "torq pen/step": float(-2.5e-5 * np.nansum(tr["torq2"]) * dt / tot),
        "u_bar": float(np.nansum(tr["unc"]) / tot),
        "robots ever tilt>30": float((np.nanmax(tilt, 1) > 30).mean()),
        "robots ever tilt>60": float((np.nanmax(tilt, 1) > 60).mean()),
        "steps tilt>30": float((tilt[steps] > 30).mean()),
        "robots ever |vz|>1": float((np.nanmax(tr["vz2"], 1) > 1.0).mean()),
        "vz2 p50/p90/p99": [float(x) for x in np.nanpercentile(tr["vz2"], [50, 90, 99])],
        "tilt p50/p90/p99 deg": [float(x) for x in np.nanpercentile(tilt, [50, 90, 99])],
    }
    if label:
        print(label, json.dumps({k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in out.items()}))
    return out


if __name__ == "__main__":
    torch.set_num_threads(2)
    actor = P.load_actor(None)
    names = sys.argv[1].split(",")
    conds = sys.argv[2].split(",") if len(sys.argv) > 2 else ["base"]
    n_envs = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    for name in names:
        for cond in conds:
            adapter, mean, std, starts = load(name)
            kw = {"base": {}, "noev": {"events": False}, "nonoise": {"noise": False},
                  "clean": {"events": False, "noise": False}, "mean": {"mean_pred": True},
                  "meanclean": {"mean_pred": True, "events": False, "noise": False}}[cond]
            env = make_env(adapter, mean, std, starts, n_envs=n_envs, **kw)
            r = imagine(env, actor)
            summarise(r, f"{name:20s} {cond:9s} starts={len(starts[0])}")

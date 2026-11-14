"""Quick test: fine-tune the released model on MuJoCo data (authors' normalisation); closed- and open-loop checks."""
import os, sys, json, time, argparse
import numpy as np, torch
import cl, ol
C = cl.C
sys.path.insert(0, os.path.join(cl.AIR, "scripts", "c"))
import train_sim as TS
T = TS.T; M = C.M

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--sources", default="all")
ap.add_argument("--freeze-aux", type=int, default=0); ap.add_argument("--steps", type=int, default=160)
ap.add_argument("--eval-every", type=int, default=40); ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--rung", default="100k"); ap.add_argument("--init", default="REL"); ap.add_argument("--N", type=int, default=8)
ap.add_argument("--save", default="")
ap.add_argument("--bad", default="", help="actor known to FALL in MuJoCo (with <actor>.mjrows.npy beside it): the fall test")
ap.add_argument("--starts", default="mj"); ap.add_argument("--step0", type=int, default=0, help="label offset when continuing")
a = ap.parse_args()
torch.set_num_threads(a.threads)
am, asd = cl.auth_stats()
cfg = C.data()[0]
arr = np.load(os.path.join(cl.DATA, f"rung_{a.rung}.npy"), mmap_mode="r")
ep = np.asarray(arr[:, 66]).astype(np.int64); src = np.asarray(arr[:, 67]).astype(np.int64); fall = np.asarray(arr[:, 65]) > 0.5
episodes = None
if a.sources == "expert":
    episodes = set(np.unique(ep[np.isin(src, [0, 1, 2])]).tolist())
elif a.sources == "src0":
    episodes = set(np.unique(ep[src == 0]).tolist())
ds = TS.RungWindows(arr, ep, fall, am, asd, 32 + a.N, episodes=episodes)
print(f"[{a.tag}] windows {len(ds)}  sources {a.sources}  freeze_aux {a.freeze_aux} lr {a.lr}", flush=True)

torch.manual_seed(0); np.random.seed(0)
model = M.build_from_config(cfg, ensemble_size=5)
if a.init == "REL":
    sd = torch.load(C.W_RELEASED, map_location="cpu", weights_only=False)["system_dynamics_state_dict"]
    model.load_state_dict(sd, strict=True)
elif os.path.exists(a.init):
    model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
params = [p for n, p in model.named_parameters() if not (a.freeze_aux and n.startswith("auxiliary"))]
opt = torch.optim.Adam(params, lr=a.lr, weight_decay=cfg["weight_decay"])
gen = torch.Generator().manual_seed(0)
weights = cfg["loss_weights"]
actor = cl.P.load_actor(None)
starts = cl.rung_windows("100k", am, asd, source=0, max_windows=20000) if a.starts == "mj" else cl.csv_windows(am, asd)
bad_actor = bad_rows = None
if a.bad:
    import mjeval, exploit
    bad_actor = mjeval.load_actor(a.bad)
    pad = np.load(a.bad + ".mjrows.npy")
    eps = []
    for r in pad:
        n = int(np.flatnonzero(np.abs(r).sum(1) > 0)[-1]) + 1
        eps.append((r[:n], bool(r[n - 1, 57] > 0.5)))
    bad_rows = exploit.windows(eps, 24)
tests = {"expert": ol.windows(np.load(os.path.join(cl.DATA, "test_expert.npy"))), "mixed": ol.windows(np.load(os.path.join(cl.DATA, "test_mixed.npy")))}
from adapter import LiteDynamicsAdapter

def evaluate(step, losses):
    model.eval()
    torch.set_num_threads(2)
    rec = {"step": step}
    if losses:
        rec["loss"] = {k: round(float(np.mean(v[-20:])), 4) for k, v in losses.items()}
    for sn, rows in tests.items():
        r = ol.errs(model, am, asd, rows)
        rec[sn] = {k: round(r[k], 3) for k in ("E1", "E8", "E24", "E100")}; rec[sn + " E1 grp"] = r["E1 by group"]
    ad = LiteDynamicsAdapter(model)
    env = cl.make_env(ad, am, asd, starts, n_envs=384)
    s = cl.summarise(cl.imagine(env, actor))
    rec["closed"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k in ("fell", "len", "track/step", "vz pen/step", "tilt pen/step", "torq pen/step", "u_bar", "robots ever tilt>30")}
    if bad_actor is not None:
        env = cl.make_env(LiteDynamicsAdapter(model), am, asd, starts, n_envs=384)
        sb = cl.summarise(cl.imagine(env, bad_actor))
        rec["bad policy, closed loop"] = {k: round(sb[k], 4) for k in ("fell", "len", "track/step", "tilt pen/step", "u_bar", "robots ever tilt>30")}
        rb = exploit.errs(model, am, asd, bad_rows, 24)
        rec["bad policy, open loop"] = {k: rb[k] for k in ("E1", "E8", "E24", "tilt at end: true / predicted (median deg)")}
    print(f"[{a.tag}] " + json.dumps(rec), flush=True)
    torch.set_num_threads(a.threads)
    model.train()

losses = {k: [] for k in ("state", "contact", "termination")}
(evaluate(0, None) if a.step0 == 0 else None)
t0 = time.time()
for it in range(a.steps):
    state, action, ext, contact, term = ds.sample(a.batch, gen)
    model.train(); model.reset(); opt.zero_grad(set_to_none=True)
    if a.freeze_aux:
        s_l = []
        for h in model.state_heads:
            s, q, b, k = model.compute_state_loss(h, state, action, teacher_forcing=(a.N == 1))
            s_l.append(s + b)
        total = torch.stack(s_l).mean()
        losses["state"].append(float(total)); losses["contact"].append(0.0); losses["termination"].append(0.0)
    else:
        terms = model.compute_loss(state, action, ext, contact, term, teacher_forcing=(a.N == 1), loss_type="mse")
        total = M.weighted_total(terms, weights)
        losses["state"].append(float(terms[0])); losses["contact"].append(float(terms[5])); losses["termination"].append(float(terms[6]))
    total.backward(); opt.step()
    if it == 4: print(f"[{a.tag}] sec/step {(time.time()-t0)/5:.1f}  first losses state {losses['state'][:5]}", flush=True)
    if (it + 1) % a.eval_every == 0:
        evaluate(a.step0 + it + 1, losses)
        if a.save:
            torch.save({"model_state_dict": model.state_dict(), "state_mean": am.tolist(), "state_std": asd.tolist(), "iter": it + 1}, a.save.replace(".pt", f"_s{a.step0+it+1}.pt"))
print(f"[{a.tag}] done {time.time()-t0:.0f}s", flush=True)

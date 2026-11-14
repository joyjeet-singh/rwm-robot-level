"""One actor, three judges: the simulator it was trained in, the released model, and MuJoCo (same protocol)."""
import sys, json, numpy as np, torch
import cl, mjref, mjeval
torch.set_num_threads(1)
am, asd = cl.auth_stats()
path = sys.argv[1]; simpath = sys.argv[2]
net = mjeval.load_actor(path)
starts = cl.rung_windows("100k", am, asd, source=0, max_windows=20000)
def judge(name):
    if name == "REL":
        ad, mean, std, _ = cl.load("REL")
    else:
        ck = torch.load(name, map_location="cpu", weights_only=False)
        m = cl.C.M.build_from_config(cl.C.data()[0], ensemble_size=5); m.load_state_dict(ck["model_state_dict"]); m.eval()
        ad, mean, std = cl.LiteDynamicsAdapter(m), np.asarray(ck["state_mean"]), np.asarray(ck["state_std"])
    st = starts if np.allclose(mean, am) else cl.rung_windows("100k", mean, std, source=0, max_windows=20000)
    env = cl.make_env(ad, mean, std, st, n_envs=384)
    r = cl.imagine(env, net)
    s = cl.summarise(r)
    return {k: round(s[k], 4) for k in ("fell", "len", "track/step", "vz pen/step", "tilt pen/step", "torq pen/step", "u_bar", "robots ever tilt>30")}
print("actor", path)
print("  in its own simulator :", json.dumps(judge(simpath)))
print("  in the released model:", json.dumps(judge("REL")))
r = mjref.run(64, events=True, noise=True, actor_net=net)
tr = r["tr"]; tot = np.isfinite(tr["lin"]).sum()
tilt = np.degrees(np.arcsin(np.clip(np.sqrt(tr["tilt2"]), 0, 1)))
print("  in MuJoCo            :", json.dumps({"fell": round(float(r["fell"].mean()), 4), "len": round(float(r["length"].mean()), 1),
      "track/step": round(float((np.nansum(tr["lin"]) + 0.5 * np.nansum(tr["ang"])) * 0.02 / tot), 4),
      "vz pen/step": round(float(-2 * np.nansum(tr["vz2"]) * 0.02 / tot), 4), "tilt pen/step": round(float(-5 * np.nansum(tr["tilt2"]) * 0.02 / tot), 4),
      "torq pen/step": round(float(-2.5e-5 * np.nansum(tr["torq2"]) * 0.02 / tot), 4), "tilt p50": round(float(np.nanpercentile(tilt, 50)), 2)}))

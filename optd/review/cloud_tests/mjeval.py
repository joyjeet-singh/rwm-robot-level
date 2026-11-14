"""Walking score of an actor in the calibrated MuJoCo model (cloud copy; the project's protocol and functions)."""
import os, sys, json, time
import numpy as np, torch
PDM = "/mnt/user-data/uploads/PDM"; AIR = os.path.join(PDM, "AIR 2027")
for p in (os.path.join(AIR, "scripts", "lite", "stubs"), os.path.join(PDM, "robotic_world_model_lite", "scripts"),
          os.path.join(AIR, "scripts"), os.path.join(AIR, "scripts", "s2"), os.path.join(AIR, "scripts", "c")):
    sys.path.insert(0, p)
import evaluate as EV
import s1_policy as P
F = EV.F
torch.set_num_threads(1)

def load_actor(path):
    if path == "authors":
        return P.load_actor(None)
    sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    import torch.nn as nn
    net = nn.Sequential(nn.Linear(48, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 12))
    keys = sorted(k for k in sd if k.startswith("actor.") and k.endswith("weight"))
    for i, k in zip((0, 2, 4, 6), keys):
        net[i].weight.data.copy_(sd[k]); net[i].bias.data.copy_(sd[k.replace("weight", "bias")])
    net.eval()
    return net

def score(path, n_sched=EV.N_SCHEDULES, n_starts=EV.N_STARTS):
    env = F.make_env("cal", reward=True)
    act = F.torch_actor(load_actor(path))
    eps = []
    for e in range(n_sched):
        for j in range(n_starts):
            off, yaw = EV.walk_start(e, j)
            eps.append(EV.run_episode(env, act, EV.walk_schedule(e), off, yaw, e, j))
    return {"policy": path, "n": len(eps), "score": float(np.mean([r["score"] for r in eps])), "falls": float(np.mean([r["fell"] for r in eps])),
            "speed_err": float(np.mean([r["speed_err"] for r in eps])), "turn_err": float(np.mean([r["turn_err"] for r in eps])),
            "ret": float(np.mean([r["ret"] for r in eps])), "two_feet": float(np.mean([r["two_feet"] for r in eps])),
            "fall_steps": [r["steps"] for r in eps if r["fell"]]}

if __name__ == "__main__":
    t = time.time()
    for p in sys.argv[1:]:
        r = score(p)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), "sec", round(time.time() - t), flush=True)

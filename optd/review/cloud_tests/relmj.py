import sys, numpy as np, torch
import cl
torch.set_num_threads(2)
actor = cl.P.load_actor(None)
am, asd = cl.auth_stats()
adapter, _, _, _ = cl.load("REL")
starts = cl.rung_windows("100k", am, asd, source=0)
for cond, kw in (("base", {}), ("clean", {"events": False, "noise": False})):
    env = cl.make_env(adapter, am, asd, starts, n_envs=512, **kw)
    r = cl.imagine(env, actor)
    cl.summarise(r, f"REL from MuJoCo starts {cond:6s} starts={len(starts[0])}")
    tr = r["tr"]
    tilt = np.degrees(np.arcsin(np.clip(np.sqrt(tr["tilt2"]), 0, 1)))
    print("   k: lin ang tilt50 verr50")
    for k in [0,4,8,16,24,32,48,64,96]:
        al = np.isfinite(tr["lin"][:,k])
        verr = np.sqrt((tr["vx"][al,k]-tr["cx"][al,k])**2 + (tr["vy"][al,k]-tr["cy"][al,k])**2)
        print(f"  {k:3d}: {np.mean(tr['lin'][al,k]):5.3f} {np.mean(tr['ang'][al,k]):5.3f} {np.percentile(tilt[al,k],50):5.2f} {np.percentile(verr,50):6.3f}")

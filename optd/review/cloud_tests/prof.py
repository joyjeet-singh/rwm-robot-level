import sys, json, numpy as np, torch
import cl
torch.set_num_threads(2)
actor = cl.P.load_actor(None)
def prof(name, **kw):
    adapter, mean, std, starts = cl.load(name)
    env = cl.make_env(adapter, mean, std, starts, n_envs=512, **kw)
    r = cl.imagine(env, actor)
    tr, fell, length = r["tr"], r["fell"], r["length"]
    tilt = np.degrees(np.arcsin(np.clip(np.sqrt(tr["tilt2"]), 0, 1)))
    print(f"== {name} {kw}  fell {fell.mean():.3f}")
    print("   k:   alive  lin   ang   tilt50 tilt90  |vz|50  |vz|90  unc50   verr50  werr50")
    for k in [0,1,2,4,8,16,24,32,48,64,96,128,192,255]:
        al = np.isfinite(tr["lin"][:,k])
        if al.sum()==0: continue
        verr = np.sqrt((tr["vx"][al,k]-tr["cx"][al,k])**2 + (tr["vy"][al,k]-tr["cy"][al,k])**2)
        werr = np.abs(tr["wz"][al,k]-tr["cw"][al,k])
        print(f"  {k:3d}: {al.mean():6.3f} {np.mean(tr['lin'][al,k]):5.3f} {np.mean(tr['ang'][al,k]):5.3f} {np.percentile(tilt[al,k],50):6.2f} {np.percentile(tilt[al,k],90):6.2f} "
              f"{np.percentile(np.sqrt(tr['vz2'][al,k]),50):7.3f} {np.percentile(np.sqrt(tr['vz2'][al,k]),90):7.3f} {np.percentile(tr['unc'][al,k],50):6.3f}  {np.percentile(verr,50):6.3f}  {np.percentile(werr,50):6.3f}")
    # termination: tilt at the final step of robots that fell, and time of fall
    f = np.flatnonzero(fell)
    if len(f):
        last = length[f]-1
        t_last = tilt[f, last]
        print(f"   fell: n={len(f)}  step of fall p10/50/90 = {np.percentile(length[f],[10,50,90])}  tilt at fall p10/50/90 = {np.round(np.percentile(t_last,[10,50,90]),1)}  share upright(<20deg) at fall = {(t_last<20).mean():.2f}")
    nf = ~fell
    mt = np.nanmax(tilt[nf],1)
    print(f"   not fell: n={nf.sum()}  ever tilt>45: {(mt>45).mean():.3f}  ever tilt>80: {(mt>80).mean():.3f}")
    return r
for name in sys.argv[1].split(","):
    prof(name, events=False, noise=False)

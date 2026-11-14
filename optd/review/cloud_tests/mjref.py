"""The authors' policy in the calibrated MuJoCo model under the lite imagination protocol (cloud copy; diagnostics)."""
import os, sys, json, time
import numpy as np, torch
PDM = "/mnt/user-data/uploads/PDM"; AIR = os.path.join(PDM, "AIR 2027")
for p in (os.path.join(AIR, "scripts", "lite", "stubs"), os.path.join(PDM, "robotic_world_model_lite", "scripts"),
          os.path.join(AIR, "scripts"), os.path.join(AIR, "scripts", "s2")):
    sys.path.insert(0, p)
import fidelity as F
import s1_policy as P
import mujoco
E = F.E
torch.set_num_threads(1)

def run(n_robots, events, noise, seed=1, n_steps=256, warm=100, actor_net=None, push=0.5, ev_range=(48, 96), cmd_range=(100, 120)):
    env = F.make_env("cal", reward=True)
    net = actor_net or P.load_actor(None)
    act = F.torch_actor(net)
    rng = np.random.default_rng(seed)
    T = n_steps
    tr = {k: np.full((n_robots, T), np.nan, dtype=np.float32) for k in ("lin", "ang", "tilt2", "vz2", "verr", "werr", "torq2")}
    terms_sum = np.zeros(12); steps = 0
    fell = np.zeros(n_robots, bool); length = np.zeros(n_robots, int)
    for i in range(n_robots):
        off = rng.uniform(-0.05, 0.05, 12); yaw = rng.uniform(-np.pi, np.pi)
        c0 = rng.uniform(-1, 1, 3)
        env.reset(joint_offset=off, yaw=yaw, cmd=c0)
        ok = True
        for k in range(warm):
            obs = E.observation(env.s, env.cmd, env.last_action)
            _, info = env.step(act(obs))
            if info["fell"]:
                ok = False; break
        if not ok:
            fell[i] = True; continue
        env.cmd = rng.uniform(-1, 1, 3)
        ci = int(rng.integers(cmd_range[0], cmd_range[1])); ei = int(rng.integers(ev_range[0], ev_range[1]))
        for k in range(T):
            s = env.s
            o = E.observation(s, env.cmd, env.last_action)
            if noise:
                o = o.copy()
                o[0:3] += rng.uniform(-1, 1, 3) * 0.1; o[3:6] += rng.uniform(-1, 1, 3) * 0.2
                o[6:9] += rng.uniform(-1, 1, 3) * 0.05; o[12:24] += rng.uniform(-1, 1, 12) * 0.01
                o[24:36] += rng.uniform(-1, 1, 12) * 1.5
            cmd = env.cmd.copy()
            _, info = env.step(act(o))
            s2 = info["state"]
            lin_err = np.sum((cmd[:2] - s2[:2]) ** 2); ang_err = (cmd[2] - s2[5]) ** 2
            tr["lin"][i, k] = np.exp(-lin_err / 0.25); tr["ang"][i, k] = np.exp(-ang_err / 0.25)
            tr["tilt2"][i, k] = s2[6] ** 2 + s2[7] ** 2; tr["vz2"][i, k] = s2[2] ** 2
            tr["verr"][i, k] = np.sqrt(lin_err); tr["werr"][i, k] = np.sqrt(ang_err)
            tr["torq2"][i, k] = np.sum(s2[33:45] ** 2)
            terms_sum += info["terms"]; steps += 1
            length[i] += 1
            if info["fell"]:
                fell[i] = True; break
            if (k + 1) % ci == 0:
                env.cmd = rng.uniform(-1, 1, 3)
            if events and (k + 1) % ei == 0:
                R = env.d.xmat[env.base].reshape(3, 3)
                dv = np.array([rng.uniform(-push, push), rng.uniform(-push, push), 0.0])
                env.d.qvel[0:3] += R @ dv
                mujoco.mj_forward(env.m, env.d)
                env.s = env.state()
    return {"tr": tr, "fell": fell, "length": length, "terms": terms_sum / max(1, steps)}

def report(r, label):
    tr = r["tr"]; steps = np.isfinite(tr["lin"]); tot = steps.sum()
    tilt = np.degrees(np.arcsin(np.clip(np.sqrt(tr["tilt2"]), 0, 1)))
    W = list(P.reward_weights()[0].keys())
    print(f"{label}: fell {r['fell'].mean():.3f} len {r['length'].mean():.1f} track/step {(np.nansum(tr['lin'])+0.5*np.nansum(tr['ang']))*0.02/tot:.4f} "
          f"vz pen {-2*np.nansum(tr['vz2'])*0.02/tot:.5f} tilt pen {-5*np.nansum(tr['tilt2'])*0.02/tot:.5f} torq pen {-2.5e-5*np.nansum(tr['torq2'])*0.02/tot:.5f} "
          f"tilt p50/p90/p99 {np.round(np.nanpercentile(tilt,[50,90,99]),2)}")
    print("   terms/step:", {k: round(float(v), 5) for k, v in zip(W, r["terms"])})
    print("   k: alive lin ang tilt50 tilt90 verr50 werr50")
    for k in [0,4,8,16,24,32,48,64,96,128,192,255]:
        al = np.isfinite(tr["lin"][:, k])
        if al.sum() == 0: continue
        print(f"  {k:3d}: {al.mean():5.3f} {np.mean(tr['lin'][al,k]):5.3f} {np.mean(tr['ang'][al,k]):5.3f} {np.percentile(tilt[al,k],50):5.2f} {np.percentile(tilt[al,k],90):5.2f} {np.percentile(tr['verr'][al,k],50):6.3f} {np.percentile(tr['werr'][al,k],50):6.3f}")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    print("mujoco", mujoco.__version__)
    t = time.time()
    report(run(n, events=False, noise=False), "MuJoCo clean")
    print("seconds", round(time.time() - t, 1))
    report(run(n, events=True, noise=True), "MuJoCo events+noise")
    report(run(n, events=True, noise=False), "MuJoCo events only")

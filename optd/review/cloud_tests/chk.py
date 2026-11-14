"""The paper's long-horizon checks in common units on the MuJoCo expert test set (diagnostics)."""
import sys, os, copy, json, numpy as np, torch
import cl
C = cl.C
import lyap
torch.set_num_threads(1)
am, asd = cl.auth_stats()
def arena(path):
    a = np.load(path); ep = a[:, 66].astype(np.int64)
    cuts = np.flatnonzero(ep[1:] != ep[:-1]) + 1
    starts = np.r_[0, cuts]; ends = np.r_[cuts, len(ep)]
    win = [s + k * C.LEN for s, e in zip(starts, ends) for k in range((e - s) // C.LEN)]
    idx = np.asarray(win)[:, None] + np.arange(C.LEN)[None, :]
    return np.asarray(a[idx], dtype=np.float64)
def scales(raw):
    S = raw[:, :, :45]
    eG, eJ, eA = C.rule_residuals(S)
    return {"sJ": np.sqrt((eJ.reshape(-1, 12) ** 2).mean(0)), "sA": np.sqrt((eA.reshape(-1, 3) ** 2).mean(0))}
for sn in ("expert", "mixed"):
    raw = arena(os.path.join(cl.DATA, f"test_{sn}.npy"))
    sc = scales(arena(os.path.join(cl.DATA, "test_expert.npy")))
    print(f"--- {sn}: {len(raw)} windows; true rJ {np.mean(C.monitors(raw[:, :, :45], sc)[1]):.3f}")
    for name in sys.argv[1].split(","):
        if name.endswith(".pt"):
            ck = torch.load(name, map_location="cpu", weights_only=False)
            net = C.M.build_from_config(C.data()[0], ensemble_size=5); net.load_state_dict(ck["model_state_dict"]); net.eval()
            mean, std = np.asarray(ck["state_mean"]), np.asarray(ck["state_std"])
        else:
            adapter, mean, std, _ = cl.load(name); net = adapter.net
        st = torch.as_tensor((raw[:, :, :45] - mean) / std, dtype=torch.float32)
        ac = torch.as_tensor(raw[:, :, 45:57], dtype=torch.float32)
        with torch.no_grad():
            pred = lyap.rollout_map(net, st, ac)
        P = pred.numpy().astype(np.float64) * std + mean
        _, rJ, rA = C.monitors(P, sc)
        e = np.abs(P - raw[:, :, :45]) / asd
        E = {h: float(e[:, C.START:C.START + h].mean()) for h in (8, 100, 368)}
        lam, _ = lyap.lyapunov(copy.deepcopy(net), st[:12], ac[:12], p=1)
        print(f"{os.path.basename(name):22s} rJ {np.mean(rJ):6.3f}  rA {np.mean(rA):6.3f}  lambda1 {np.mean(lam[:,0]):+.3f}  E8 {E[8]:.3f} E100 {E[100]:.3f} E368 {E[368]:.3f} (first-window errors, common units)")

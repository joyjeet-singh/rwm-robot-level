"""Open-loop h-step errors on MuJoCo held-out data, common units (authors' constants)."""
import sys, os, json, numpy as np, torch
import cl
C = cl.C
torch.set_num_threads(2)
am, asd = cl.auth_stats()
H, F = 32, 100
def windows(arr, stride=100):
    a = np.asarray(arr); ep = a[:,66].astype(int); fall = a[:,65] > 0.5
    starts = []
    for e in np.unique(ep):
        idx = np.flatnonzero(ep == e)
        for s in range(idx[0], idx[-1] + 2 - (H + F), stride):
            if not fall[s:s+H+F-1].any(): starts.append(s)
    idx = np.asarray(starts)[:, None] + np.arange(H + F)
    return a[idx]
def errs(net, mean, std, rows, member=None):
    st = torch.as_tensor((rows[:,:,:45] - mean) / std, dtype=torch.float32)
    ac = torch.as_tensor(rows[:,:,45:57], dtype=torch.float32)
    with torch.no_grad():
        if member is None:
            pred = net.rollout(st, ac, H, action_offset=1)
        else:
            # single-member rollout
            B, T, _ = st.shape; pred = st.clone(); net.reset()
            for i in range(H, T):
                if i > H: s_in, a_in = pred[:, i-1:i], ac[:, i:i+1]
                else: s_in, a_in = pred[:, 0:i], ac[:, 1:i+1]
                base = net.state_base(s_in, a_in)
                m, _ = net.state_heads[member](base, s_in)
                pred[:, i] = m
    P = pred.numpy().astype(np.float64) * std + mean
    T = rows[:,:,:45].astype(np.float64)
    e = np.abs(P - T) / asd                      # common units
    hold = np.abs(T[:, H-1:H, :] - T) / asd
    out = {}
    for h in (1, 8, 24, 100):
        out[f"E{h}"] = float(e[:, H:H+h].mean()); out[f"floor{h}"] = float(hold[:, H:H+h].mean())
    groups = {"v": slice(0,3), "w": slice(3,6), "g": slice(6,9), "q": slice(9,21), "qd": slice(21,33), "tau": slice(33,45)}
    out["E24 by group"] = {k: round(float(e[:, H:H+24, sl].mean()), 3) for k, sl in groups.items()}
    out["E1 by group"] = {k: round(float(e[:, H:H+1, sl].mean()), 3) for k, sl in groups.items()}
    return out
if __name__ == "__main__":
    sets = {"expert": np.load(os.path.join(cl.DATA, "test_expert.npy")), "mixed": np.load(os.path.join(cl.DATA, "test_mixed.npy"))}
    for name in sys.argv[1].split(","):
        adapter, mean, std, _ = cl.load(name)
        net = adapter.net
        for sn, arr in sets.items():
            rows = windows(arr)
            r = errs(net, mean, std, rows)
            r1 = errs(net, mean, std, rows, member=0)
            print(f"{name:20s} {sn:6s} n={len(rows):3d} mean-pred: " + " ".join(f"{k}={r[k]:.3f}" for k in ("E1","E8","E24","E100")) +
                  " | member0: " + " ".join(f"{k}={r1[k]:.3f}" for k in ("E1","E8","E24","E100")) +
                  " | floor: " + " ".join(f"{r['floor'+str(h)]:.3f}" for h in (1,8,24,100)))
            print(f"{'':20s} E1 by group {r['E1 by group']}  E24 by group {r['E24 by group']}")

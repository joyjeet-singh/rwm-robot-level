"""PREREG.md §6 checks C1-C5. Prints and stores differences only; no statistic. -> results/checks.json"""
import json
import os

import numpy as np
import torch

import common as C
import compress as K
import evalh
import lyap
import pools

torch.set_num_threads(2)
out = {}
NG = os.path.join(C.PDM, "NGCAPS", "results")

# C1: reproduce NGCAPS's stored arrays
ar18 = C.arena((1, 8))
ng18, ngx5 = np.load(os.path.join(NG, "raw_F18.npz")), np.load(os.path.join(NG, "raw_X5.npz"))
ent = {e[0]: e for e in pools.pool_F18()}
c1 = {}
for key in ("A_s0", "B_s0", "mn_M32_N32_s0", "A5_s0", "w124_s0"):
    _, path, ens, *_ = ent[key]
    m = C.load(path, ens)
    pred = lyap.rollout_map(m, ar18["st"], ar18["ac"])
    err = (pred[:, C.START:] - ar18["st"][:, C.START:]).abs().mean(-1).numpy()
    fl = (ar18["st"][:, C.START - 1:C.START] - ar18["st"][:, C.START:]).abs().mean(-1).numpy()
    lam, _ = lyap.lyapunov(m, ar18["st"], ar18["ac"], p=4)
    c1[f"F18/{key}"] = {"err_equal": bool(np.array_equal(err, ng18[f"err_{key}"])),
                        "floor_equal": bool(np.array_equal(fl, ng18[f"floor_{key}"])),
                        "lam_max_abs_diff": float(np.abs(lam - ng18[f"lam_{key}"]).max())}
for key in ("A_02_s0", "A5_34_s1", "B_79_s2"):
    kind, f, s = key.split("_")
    fold = tuple(int(c) for c in f)
    ar = C.arena(fold)
    m = C.load(C.run_path(C.fold_dir(kind, fold, int(s[1:]))), 5 if kind == "A5" else 1)
    pred = lyap.rollout_map(m, ar["st"], ar["ac"])
    err = (pred[:, C.START:] - ar["st"][:, C.START:]).abs().mean(-1).numpy()
    lam, _ = lyap.lyapunov(m, ar["st"], ar["ac"], p=4)
    c1[f"X5/{key}"] = {"err_equal": bool(np.array_equal(err, ngx5[f"err_{key}"])),
                       "lam_max_abs_diff": float(np.abs(lam - ngx5[f"lam_{key}"]).max())}
out["C1"] = c1
out["C1_pass"] = all(v["err_equal"] and v.get("floor_equal", True) and v["lam_max_abs_diff"] < 1e-12
                     for v in c1.values())

# C2: compression identities
parent = C.load(ent["A_s0"][1], 1)
ref = lyap.rollout_map(parent, ar18["st"], ar18["ac"])
u0, _ = K.unstructured(parent, 0.0)
s256, _ = K.structured(parent, 256)
f16, _ = K.fp16(parent)
i8, info = K.int8w(parent)
pp = dict(parent.named_parameters())
fp16_ok = all(torch.equal(p, pp[n].half().float()) for n, p in f16.named_parameters())
i8_excess = max(float(((dict(i8.named_parameters())[n] - w).abs().max() - info["scales"][n] / 2) /
                      max(info["scales"][n], 1e-30)) for n, w in K.weight_matrices(parent))
out["C2"] = {"u0_equal": bool(torch.equal(lyap.rollout_map(u0, ar18["st"], ar18["ac"]), ref)),
             "s256_equal": bool(torch.equal(lyap.rollout_map(s256, ar18["st"], ar18["ac"]), ref)),
             "fp16_weights_equal_half_rounded": bool(fp16_ok),
             "int8w_max_excess_over_half_scale_in_scales": i8_excess}
out["C2_pass"] = (out["C2"]["u0_equal"] and out["C2"]["s256_equal"] and fp16_ok and i8_excess <= 1e-6)

# C3: jvp against a central finite difference, float64, on two compressed variants
c3 = {}
for v in ("u30", "s128"):
    mv, _ = K.apply(C.load(ent["A_s0"][1], 1), v)
    c3[v] = lyap.fd_check(mv, ar18["st"], ar18["ac"])
out["C3_rel_err"] = c3
out["C3_pass"] = all(x < 1e-5 for x in c3.values())

# C4: the recording obeys rule G; scaled rule-J residual has RMS 1 on the scaling episodes
_, dat, ep = C.data()
eG, eJ, _ = C.rule_residuals(dat[:, C.R.STATE_COLS])
tr = C.train_episodes((1, 8))
sc = C.real_scales(tr)
same = (ep[1:] == ep[:-1]) & (ep[1:] >= 0)
inside = same & np.isin(ep[1:], tr)
rJ = np.sqrt(((eJ[inside] / sc["sJ"]) ** 2).mean(-1))
out["C4"] = {"max_rule_G_all_episodes": float(eG[same].max()), "rms_rJ_scaling_episodes": float(np.sqrt((rJ ** 2).mean()))}
out["C4_pass"] = out["C4"]["max_rule_G_all_episodes"] <= 1e-6 and abs(out["C4"]["rms_rJ_scaling_episodes"] - 1) < 1e-9

# C5: the 368-step window error equals the rollout error
e368, _ = evalh.window_error(parent, ar18["st"], ar18["ac"], 368, stride=1)
roll = (ref[:, C.START:] - ar18["st"][:, C.START:]).abs().mean(-1).mean(-1).numpy()
out["C5_max_abs_diff"] = float(np.abs(e368 - roll).max())
out["C5_pass"] = out["C5_max_abs_diff"] < 1e-6

out["all_pass"] = all(out[k] for k in ("C1_pass", "C2_pass", "C3_pass", "C4_pass", "C5_pass"))
json.dump(out, open(os.path.join(C.RESULTS, "checks.json"), "w"), indent=1)
print(json.dumps(out, indent=1))

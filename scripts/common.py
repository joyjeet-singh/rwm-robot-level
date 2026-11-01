"""
Shared definitions for the AIR 2027 study (prereg/PREREG.md).

Loaders, arenas and the rollout map follow ../NGCAPS/scripts/common.py (sha256 2f468ce9...);
the physics rules follow ../ICPAIAS/scripts/common.py (sha256 af31c1c0...). Everything under PDM outside
"AIR 2027" is READ ONLY: data and the released model in ../robotic_world_model_lite, source and trained
weights in ../rwm_repro.

PDM root: env PDM_ROOT; default the staged copy in the cloud workspace. On the Mac use
PDM_ROOT=~/Downloads/PDM.
"""
import hashlib
import os
import sys

sys.dont_write_bytecode = True  # never write __pycache__ into the read-only folders
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
AIR = os.path.dirname(HERE)
PDM = os.environ.get("PDM_ROOT", "/mnt/user-data/uploads/PDM")
RWM = os.path.join(PDM, "rwm_repro")
LITE = os.path.join(PDM, "robotic_world_model_lite")
RESULTS = os.path.join(AIR, "results")
FIGURES = os.path.join(AIR, "figures")

sys.path.insert(0, os.path.join(RWM, "src"))
import rwm_data as R  # noqa: E402
import rwm_model as M  # noqa: E402
import rwm_metrics as MET  # noqa: E402

DT = R.DT                                   # 0.02 s
START, LEN = 32, 400                        # 32 history rows, 368 forecast steps
FOLDS = ((1, 8), (0, 2), (3, 4), (5, 6), (7, 9))
N_BOOT = 20000
LIN, ANG, GRAV = R.LIN_VEL, R.ANG_VEL, R.GRAVITY   # dims 0-2, 3-5, 6-8
Q, QD = R.JOINT_POS, R.JOINT_VEL                   # dims 9-20, 21-32
_D = {}


def data():
    if not _D:
        _D["cfg"] = R.load_reference_config(LITE)
        csv = os.path.join(LITE, "assets", "data", "state_action_data_0.csv")
        _D["data"], _D["ep"] = R.load_data(csv, verbose=False)
    return _D["cfg"], _D["data"], _D["ep"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fid(fold):
    return "".join(map(str, fold))


def train_episodes(fold):
    return [e for e in range(R.N_EPISODES) if e not in fold]


RUNS = os.path.join(RWM, "runs")
W_RELEASED = os.path.join(LITE, "assets", "models", "pretrain_rnn_ens.pt")


def run_path(d, it=2500):
    return os.path.join(RUNS, d, f"weights_{it}.pt")


def fold_dir(kind, fold, s):
    """kind: 'A' (rollout-trained, one head), 'A5' (rollout-trained, five heads), 'B' (teacher-forced)."""
    arm = "B" if kind == "B" else "A"
    suf = "_ens5" if kind == "A5" else ""
    return (f"arm{arm}_seed{s}{suf}" if tuple(fold) == (1, 8) else f"x3_arm{arm}_ho{fid(fold)}_seed{s}{suf}")


def load(path, ensemble):
    """A trained dynamics model in eval mode; hidden width read from the checkpoint."""
    cfg, _, _ = data()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("system_dynamics_state_dict", ck.get("model_state_dict", ck))
    width = int(sd["state_base.memory.rnn.weight_hh_l0"].shape[1])
    if width != cfg["architecture_config"]["rnn_hidden_size"]:
        cfg = dict(cfg)
        cfg["architecture_config"] = dict(cfg["architecture_config"], rnn_hidden_size=width)
    m = M.build_from_config(cfg, ensemble_size=ensemble)
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m


def build_width(width, ensemble_size=1):
    """An untrained model with every GRU at `width` units; all else as the reference config."""
    cfg, _, _ = data()
    cfg = dict(cfg)
    cfg["architecture_config"] = dict(cfg["architecture_config"], rnn_hidden_size=width)
    return M.build_from_config(cfg, ensemble_size=ensemble_size)


def tensors(starts):
    cfg, dat, _ = data()
    idx = np.asarray(starts)[:, None] + np.arange(LEN)[None, :]
    raw = dat[idx]
    st = torch.as_tensor(R.normalise_state(raw[:, :, R.STATE_COLS], cfg["state_data_mean"],
                                           cfg["state_data_std"]), dtype=torch.float32)
    ac = torch.as_tensor(raw[:, :, R.ACTION_COLS], dtype=torch.float32)
    return st, ac


def arena(episodes):
    _, _, ep = data()
    starts = MET.non_overlapping_starts(ep, sorted(episodes), LEN)
    assert int(MET.n_independent(starts, LEN)) == len(starts) == 2 * len(episodes), starts
    st, ac = tensors(starts)
    return {"starts": [int(s) for s in starts], "episode": [int(ep[s]) for s in starts], "st": st, "ac": ac}


def denorm(x):
    cfg, _, _ = data()
    return np.asarray(x, dtype=np.float64) * cfg["state_data_std"] + cfg["state_data_mean"]


# ------------------------------------------------------------------ physics rules (ICPAIAS definitions)
def rule_residuals(S):
    """S (..., T, 45) de-normalised. Residuals for steps 1..T-1: eG (..., T-1), eJ (..., T-1, 12), eA (..., T-1, 3)."""
    g, w, q, qd = S[..., GRAV], S[..., ANG], S[..., Q], S[..., QD]
    eG = np.abs(np.linalg.norm(g[..., 1:, :], axis=-1) - 1.0)
    eJ = (q[..., 1:, :] - q[..., :-1, :]) - 0.5 * DT * (qd[..., 1:, :] + qd[..., :-1, :])
    gb, wb = 0.5 * (g[..., 1:, :] + g[..., :-1, :]), 0.5 * (w[..., 1:, :] + w[..., :-1, :])
    eA = (g[..., 1:, :] - g[..., :-1, :]) + DT * np.cross(wb, gb)
    return eG, eJ, eA


def real_scales(episodes):
    """RMS of eJ (per joint) and eA (per axis) over consecutive recorded rows inside the given episodes."""
    _, dat, ep = data()
    S = dat[:, R.STATE_COLS]
    eG, eJ, eA = rule_residuals(S)
    same = (ep[1:] == ep[:-1]) & (ep[1:] >= 0) & np.isin(ep[1:], episodes)
    return {"sJ": np.sqrt((eJ[same] ** 2).mean(0)), "sA": np.sqrt((eA[same] ** 2).mean(0)),
            "maxG": float(eG[same].max()), "n_pairs": int(same.sum())}


def monitors(S, sc):
    """Scaled per-step residuals r_G, r_J, r_A for forecast steps k = 1..368 (rows START..LEN-1)."""
    eG, eJ, eA = rule_residuals(S[..., START - 1:, :])
    rG = eG
    rJ = np.sqrt(((eJ / sc["sJ"]) ** 2).mean(-1))
    rA = np.sqrt(((eA / sc["sA"]) ** 2).mean(-1))
    return rG, rJ, rA


# ------------------------------------------------------------------ physical units (descriptive, PREREG §5)
def dead_reckon(S):
    """
    Planar position from body-frame base velocity and yaw rate, from the end of the history (row START-1),
    explicit Euler over forecast rows START..LEN-1. S (B, T, 45) de-normalised. Returns (B, 368, 2) in metres.
    The same integration is applied to the prediction and to the recording.
    """
    v, wz = S[:, START:, 0:2], S[:, START:, 5]
    psi = np.cumsum(wz * DT, axis=1)
    c, s = np.cos(psi), np.sin(psi)
    vx = c * v[..., 0] - s * v[..., 1]
    vy = s * v[..., 0] + c * v[..., 1]
    return np.stack([np.cumsum(vx * DT, 1), np.cumsum(vy * DT, 1)], -1)


def physical(P, S):
    """Per-step physical errors of prediction P against recording S (both de-normalised, (B, T, 45))."""
    jdeg = np.degrees(np.abs(P[:, START:, Q] - S[:, START:, Q])).mean(-1)          # joint angle, deg
    vel = np.abs(P[:, START:, 0:3] - S[:, START:, 0:3]).mean(-1)                  # base linear velocity, m/s
    pos = np.linalg.norm(dead_reckon(P) - dead_reckon(S), axis=-1)                 # planar position, m
    path = np.concatenate([np.zeros((S.shape[0], 1, 2)), dead_reckon(S)], 1)       # origin, then steps 1..368
    dist = np.cumsum(np.linalg.norm(np.diff(path, axis=1), axis=-1), 1)            # recorded path length, m
    return jdeg, vel, pos, dist


# ------------------------------------------------------------------ statistics helpers
def ci(v, level=95.0):
    v = np.asarray(v)
    v = v[np.isfinite(v)]
    a = (100.0 - level) / 2.0
    return [float(np.percentile(v, a)), float(np.percentile(v, 100.0 - a))]


def label3(lo_hi, pos, neg, mid="NOT SETTLED", ref=0.0):
    lo, hi = lo_hi
    return pos if lo > ref else (neg if hi < ref else mid)

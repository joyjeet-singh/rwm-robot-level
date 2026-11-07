"""
Step 2 (M2): fidelity of the hand-built MuJoCo ANYmal D model (not the data's simulator, not hardware).

(a) Replay. A window starts from recorded state row k-1 and replays the recorded joint commands of rows k, k+1, ...
    in MuJoCo (Step 1's pairing: action row i moves state row i-1 to row i, as lyap/evalh use, action_offset = 1).
    The start state is placed from the 45 recorded numbers:
    - orientation from the projected gravity (roll and pitch; yaw is not observable and is set to 0);
    - joint angles and speeds as recorded; base linear (centre-of-mass) and angular velocity as recorded, in the
      base frame;
    - base height from leg kinematics, with the lowest foot sphere just touching the ground;
    - the motor's delay buffer is filled with the targets of the previous recorded joint commands.
    Errors are evalh's: E_h is, per trajectory, the mean over all windows of length h (evalh.starts: E1 at every
    position, E8 / E24 / E100 from every 4th start) of the per-step mean absolute error over the 45 normalised
    numbers, averaged over the window's h steps. The torque error is the same average of |torque error| in N m
    over the 12 joints. MuJoCo is deterministic, so the first h steps of a longer window from the same start equal
    the h-step window; each start is replayed once, for its longest window.
(b) Gait of the authors' policy on the evaluation protocol: the ten recorded episodes' velocity-command schedules
    (s1_policy.command_per_row: each row gets its Step 0 regime's mean) x eight seeded starts (joint angles moved by
    up to +-0.05 rad, random heading). 20 s each, ending early at a fall (mj_env.MjEnv.step).
(c) Reference: the same gait statistics from the recorded data, on the same ten schedules.

Gait statistics, per episode over rows 1..end (row 0 is the reset state; a fallen episode ends at its fall), then
the mean over episodes:
- fell: the episode ended at a fall;
- two_feet: share of rows with exactly two of the four feet in contact;
- duty: share of rows with a foot in contact, averaged over the four feet;
- speed_err: |(vx, vy) - (cmd_vx, cmd_vy)| in m/s, base frame; turn_err: |wz - cmd_wz| in rad/s;
- ret (policy runs only): the physics return, the sum of the lite code's step rewards (mj_env.LiteReward).

Held-out episodes (1 and 8) and policy runs are refused until calibrate.py has frozen the constants.

  ./py-mj scripts/s2/fidelity.py --part a_uncal|a_cal|b_uncal|b_cal|c|report
"""
import argparse
import json
import os
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))
import common as C  # noqa: E402
import evalh  # noqa: E402
import mj_env as E  # noqa: E402
import actuator as ACT  # noqa: E402

AIR = C.AIR
HELD_OUT = (1, 8)
TRAIN = tuple(e for e in range(10) if e not in HELD_OUT)
HORIZONS = (1, 8, 24, 100)
MODEL_DIR = os.path.join(AIR, "step2", "model")
CAL_JSON = os.path.join(MODEL_DIR, "calibration.json")
CAL_XML = os.path.join(MODEL_DIR, "anymal_d_cal.xml")
RES_DIR = os.path.join(AIR, "results", "s2")
PROGRESS = os.path.join(RES_DIR, "progress.json")
OUT_JSON = os.path.join(AIR, "results", "s2_fidelity.json")
OUT_TXT = os.path.join(AIR, "results", "s2_fidelity.txt")
EVAL_SCHEDULES = tuple(range(10))
EVAL_STARTS = 8
EVAL_SEED = 2027                 # start j of schedule e: np.random.default_rng([EVAL_SEED, e, j])
JOINT_JITTER = 0.05              # rad
GROUPS = {"lin_vel": C.LIN, "ang_vel": C.ANG, "gravity": C.GRAV, "joint_pos": C.Q, "joint_vel": C.QD,
          "torque": list(range(33, 45))}
FOOT_COLS = list(range(61, 65))   # recorded foot contacts LF, LH, RF, RH (rwm_data.CONTACT_FOOT)


# ------------------------------------------------------------------ bookkeeping
def progress(job, step, done=None, total=None):
    os.makedirs(RES_DIR, exist_ok=True)
    rec = {"job": job, "step": step, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if done is not None:
        rec.update(done=int(done), total=int(total))
    tmp = f"{PROGRESS}.{os.getpid()}.tmp"
    json.dump(rec, open(tmp, "w"))
    os.replace(tmp, PROGRESS)


def write_new(path, obj):
    """Write a result file that must not exist yet (results are never overwritten)."""
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; result files are never overwritten")
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1, default=float)
    os.replace(tmp, path)


def frozen_calibration():
    """The frozen calibration record, after checking the calibrated model file against its recorded hash."""
    if not os.path.exists(CAL_JSON):
        raise SystemExit("calibration not frozen (step2/model/calibration.json missing): held-out episodes and "
                         "policy runs are refused")
    cal = json.load(open(CAL_JSON))
    if not cal.get("frozen") or C.sha256(CAL_XML) != cal["cal_xml_sha256"]:
        raise SystemExit("calibration record not frozen or calibrated model changed since freezing")
    return cal


def require_allowed(episodes):
    if set(episodes) & set(HELD_OUT):
        frozen_calibration()


def make_env(which, reward=False):
    """'uncal': M1's model with its placeholder constants; 'cal': the frozen calibrated model."""
    if which == "uncal":
        return E.MjEnv(xml=E.MODEL_XML, reward=reward)
    if which == "cal":
        frozen_calibration()
        return E.MjEnv(xml=CAL_XML, reward=reward)
    raise ValueError(which)


# ------------------------------------------------------------------ (a) replay
def gravity_quat(g):
    """Quaternion (w, x, y, z) with yaw 0 whose projected gravity R^T (0, 0, -1) is the unit vector of g."""
    g = np.asarray(g, dtype=np.float64)
    g = g / np.linalg.norm(g)
    th = np.arcsin(np.clip(g[0], -1.0, 1.0))           # pitch
    ph = np.arctan2(-g[1], -g[2])                       # roll;  R = Ry(th) Rx(ph)
    qy = np.array([np.cos(th / 2), 0.0, np.sin(th / 2), 0.0])
    qx = np.array([np.cos(ph / 2), np.sin(ph / 2), 0.0, 0.0])
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, qy, qx)
    return q


class Replayer:
    """Open-loop replay of recorded joint commands from a recorded state, in one MuJoCo model."""

    def __init__(self, env):
        assert isinstance(env.motor, ACT.DelayedPD), "replay supports the PD fallback motor model"
        self.env = env
        m = env.m
        self.ipos = m.body_ipos[env.base].copy()
        self.radius = np.array([m.geom_size[g][0] for g in env.g_foot])
        self.warn0 = 0

    def set_state(self, s, prev_actions):
        """s: recorded raw state (45,); prev_actions: earlier recorded joint-command rows, oldest first."""
        env, m, d = self.env, self.env.m, self.env.d
        mujoco.mj_resetData(m, d)
        q = gravity_quat(s[6:9])
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        R = R.reshape(3, 3)
        d.qpos[0:3] = 0.0
        d.qpos[3:7] = q
        d.qpos[env.qadr] = E.DEFAULT_POS + s[9:21]
        mujoco.mj_kinematics(m, d)
        low = min(d.geom_xpos[g][2] - r for g, r in zip(env.g_foot, self.radius))
        d.qpos[2] = -low                                     # lowest foot sphere touches the ground
        w = np.asarray(s[3:6], dtype=np.float64)
        d.qvel[3:6] = w                                      # free joint: angular velocity, body frame
        d.qvel[0:3] = R @ s[0:3] - np.cross(R @ w, R @ self.ipos)   # frame-origin velocity, world frame
        d.qvel[env.vadr] = s[21:33]
        mujoco.mj_forward(m, d)
        # motor: physics-step targets of the previous commands (4 per policy step), newest last
        mot = env.motor
        mot.reset()
        tg = [E.DEFAULT_POS + E.ACTION_SCALE * np.asarray(a, dtype=np.float64)
              for a in prev_actions for _ in range(E.DECIMATION)]
        mot._buf = [t.copy() for t in tg[len(tg) - mot.delay:]] if mot.delay > 0 else []
        assert len(mot._buf) == mot.delay, "not enough previous joint commands for the motor delay"
        env.torque = np.asarray(s[33:45], dtype=np.float64).copy()
        self.warn0 = int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)

    def run(self, actions):
        """Replay joint-command rows; returns the raw states after each policy step (n, 45)."""
        env, m, d = self.env, self.env.m, self.env.d
        out = np.empty((len(actions), 45))
        for i, a in enumerate(actions):
            target = E.DEFAULT_POS + E.ACTION_SCALE * np.asarray(a, dtype=np.float64)
            for _ in range(E.DECIMATION):
                applied, _ = env.motor(target, d.qpos[env.qadr], d.qvel[env.vadr])
                d.ctrl[:] = applied
                mujoco.mj_step(m, d)
                env.torque = applied
            out[i] = env.state()
        return out

    def unstable(self):
        return int(self.env.d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) > self.warn0


def window_plan(horizons=HORIZONS, T=C.LEN):
    """{start k: longest h whose evalh windows start at k}, and {h: evalh starts}."""
    starts = {h: evalh.starts(h, evalh.STRIDE[h], T) for h in horizons}
    plan = {}
    for h in sorted(horizons):
        for k in starts[h]:
            plan[k] = h
    return plan, starts


def n_motor_rows(motor):
    return -(-motor.delay // E.DECIMATION)


def raw_arena(episodes):
    """The evalh arena for these episodes (two non-overlapping 400-row trajectories each), plus raw float64 rows."""
    ar = C.arena(list(episodes))
    _, dat, _ = C.data()
    idx = np.asarray(ar["starts"])[:, None] + np.arange(C.LEN)[None, :]
    ar["raw"] = dat[idx]                                    # (B, 400, 66) float64
    return ar


def window_errors(predict, st, horizons=HORIZONS, raw=None, on_traj=None):
    """
    evalh's window errors for a generic predictor. predict(b, k, h) -> (h, 45) raw predicted states for rows
    k..k+h-1 of trajectory b. st: (B, 400, 45) normalised recorded states (the arena's). Returns per horizon the
    per-trajectory E_h, torque error (N m), per-group errors, and window counts.
    """
    cfg, _, _ = C.data()
    mu, sd = np.asarray(cfg["state_data_mean"]), np.asarray(cfg["state_data_std"])
    st = np.asarray(st, dtype=np.float64)
    B = st.shape[0]
    plan, starts = window_plan(horizons, st.shape[1])
    acc = {h: {"E": np.zeros(B), "torque_Nm": np.zeros(B), "groups": {g: np.zeros(B) for g in GROUPS}}
           for h in horizons}
    bad = np.zeros(B, dtype=int)
    for b in range(B):
        for k, L in sorted(plan.items()):
            P, unstable = predict(b, k, L)
            bad[b] += int(unstable)
            Pn = (P - mu) / sd
            ae = np.abs(Pn - st[b, k:k + L])                 # (L, 45) normalised
            tq = np.abs(P[:, 33:45] - (raw[b, k:k + L, 33:45] if raw is not None else st[b, k:k + L, 33:45] * sd[33:45] + mu[33:45]))
            for h in horizons:
                if k in starts[h] and h <= L:
                    acc[h]["E"][b] += ae[:h].mean()
                    acc[h]["torque_Nm"][b] += tq[:h].mean()
                    for g, cols in GROUPS.items():
                        acc[h]["groups"][g][b] += ae[:h, cols].mean()
        if on_traj is not None:
            on_traj(b + 1, B)
    out = {}
    for h in horizons:
        W = len(starts[h])
        out[h] = {"E": acc[h]["E"] / W, "torque_Nm": acc[h]["torque_Nm"] / W,
                  "groups": {g: v / W for g, v in acc[h]["groups"].items()}, "n_windows_per_traj": W}
    return out, bad


def mujoco_predictor(rep, ar):
    raw = ar["raw"]
    nprev = max(1, n_motor_rows(rep.env.motor))

    def predict(b, k, h):
        prev = raw[b, k - nprev:k, 45:57]
        rep.set_state(raw[b, k - 1, :45], prev)
        P = rep.run(raw[b, k:k + h, 45:57])
        return P, rep.unstable() or not np.all(np.isfinite(P))
    return predict


def replay(env, episodes, horizons=HORIZONS, traj=None, on_traj=None):
    """Fidelity (a) on the evalh arena of `episodes` (optionally a subset of its trajectories)."""
    require_allowed(episodes)
    ar = raw_arena(episodes)
    if traj is not None:
        ar = {"starts": [ar["starts"][i] for i in traj], "episode": [ar["episode"][i] for i in traj],
              "st": ar["st"][traj], "ac": ar["ac"][traj], "raw": ar["raw"][traj]}
    rep = Replayer(env)
    t0 = time.time()
    res, bad = window_errors(mujoco_predictor(rep, ar), ar["st"].numpy(), horizons, ar["raw"], on_traj)
    floor = {h: evalh.window_floor(ar["st"], h) for h in horizons}
    return {"episodes": list(episodes), "starts": ar["starts"], "traj_episode": ar["episode"],
            "seconds": time.time() - t0, "unstable_windows": bad.tolist(),
            "horizons": {str(h): {"mean": float(r["E"].mean()), "per_traj": r["E"].tolist(),
                                  "torque_Nm": float(r["torque_Nm"].mean()),
                                  "groups": {g: float(v.mean()) for g, v in r["groups"].items()},
                                  "floor": float(np.mean(floor[h])), "n_windows_per_traj": r["n_windows_per_traj"]}
                         for h, r in res.items()}}


# ------------------------------------------------------------------ (b), (c) gait
def schedule(e, n=1000):
    """(n, 3) velocity command of episode e's rows (s1_policy.command_per_row); the last row held if shorter."""
    import s1_policy as P
    _, _, ep = C.data()
    rows = np.where(ep == e)[0]
    cmd = P.command_per_row()[rows]
    if len(cmd) < n:
        cmd = np.concatenate([cmd, np.repeat(cmd[-1:], n - len(cmd), 0)])
    return cmd[:n]


def eval_start(e, j):
    """Start j of schedule e: joint offsets (12,) and heading."""
    rng = np.random.default_rng([EVAL_SEED, e, j])
    return rng.uniform(-JOINT_JITTER, JOINT_JITTER, 12), float(rng.uniform(-np.pi, np.pi))


def gait_stats(v_xy, wz, cmd, feet):
    """Per-episode statistics over the given rows (see the module text)."""
    nf = feet.sum(1)
    return {"rows": int(len(wz)), "two_feet": float(np.mean(nf == 2)), "duty": float(feet.mean()),
            "feet_down_hist": [float(np.mean(nf == i)) for i in range(5)],
            "speed_err": float(np.mean(np.linalg.norm(v_xy - cmd[:, 0:2], axis=1))),
            "turn_err": float(np.mean(np.abs(wz - cmd[:, 2])))}


def run_policy_episode(env, actor, e, j, steps=1000):
    """One evaluation episode of a policy (actor: obs (48,) -> joint command (12,)) on schedule e, start j."""
    cmd_all = schedule(e, steps + 1)
    off, yaw = eval_start(e, j)
    env.reset(joint_offset=off, yaw=yaw, cmd=cmd_all[0])
    V, W, CM, F = [], [], [], []
    ret, fell = 0.0, False
    for k in range(steps):
        env.cmd = np.asarray(cmd_all[k])
        obs = E.observation(env.s, env.cmd, env.last_action)
        a = actor(obs)
        _, info = env.step(a)
        s = info["state"]
        V.append(s[0:2]), W.append(s[5]), CM.append(cmd_all[k + 1]), F.append(info["contacts"][4:8] > 0.5)
        ret += info.get("reward", 0.0)
        if not np.all(np.isfinite(s)):
            raise FloatingPointError(f"non-finite state, schedule {e} start {j} step {k}")
        if info["fell"]:
            fell = True
            break
    st = gait_stats(np.array(V), np.array(W), np.array(CM), np.array(F, dtype=float))
    return {"schedule": e, "start": j, "fell": fell, "steps": env.t, "ret": ret, **st}


def torch_actor(net):
    import torch

    def act(obs):
        with torch.no_grad():
            return net(torch.as_tensor(obs, dtype=torch.float32)[None])[0].double().numpy()
    return act


def summarise(eps):
    keys = ("two_feet", "duty", "speed_err", "turn_err")
    out = {"n": len(eps), "fall_rate": float(np.mean([r["fell"] for r in eps]))}
    out.update({k: float(np.mean([r[k] for r in eps])) for k in keys})
    if "ret" in eps[0]:
        out["ret"] = float(np.mean([r["ret"] for r in eps]))
    out["per_schedule"] = {str(e): {k: float(np.mean([r[k] for r in eps if r["schedule"] == e]))
                                    for k in keys + ("fell",)} for e in sorted({r["schedule"] for r in eps})}
    return out


def gait_policy(which, actor=None, schedules=EVAL_SCHEDULES, starts=EVAL_STARTS, on_ep=None):
    """(b): the authors' policy (default) on the evaluation protocol."""
    import s1_policy as P
    frozen_calibration()
    env = make_env(which, reward=True)
    actor = actor or torch_actor(P.load_actor())
    t0, eps = time.time(), []
    for e in schedules:
        for j in range(starts):
            eps.append(run_policy_episode(env, actor, e, j))
            if on_ep is not None:
                on_ep(len(eps), len(schedules) * starts)
    return {"model": which, "seconds": time.time() - t0, "summary": summarise(eps), "episodes": eps}


def gait_recorded(schedules=EVAL_SCHEDULES):
    """(c): the recorded data's gait statistics on the same schedules (rows 1..end of each episode)."""
    _, dat, ep = C.data()
    eps = []
    for e in schedules:
        rows = np.where(ep == e)[0]
        cmd = schedule(e, len(rows))
        r = rows[1:]
        st = gait_stats(dat[r][:, 0:2], dat[r][:, 5], cmd[1:], (dat[r][:, FOOT_COLS] > 0.5).astype(float))
        eps.append({"schedule": e, "start": 0, "fell": bool(dat[rows, 65].max() > 0.5), **st})
    return {"summary": summarise(eps), "episodes": eps}


# ------------------------------------------------------------------ job parts
def part_path(name):
    return os.path.join(RES_DIR, f"fid_{name}.json")


def run_part(name):
    path = part_path(name)
    if os.path.exists(path):
        print(f"{name}: done already ({path})")
        return
    if name in ("a_uncal", "a_cal"):
        which = name[2:]
        env = make_env(which)
        frozen_calibration()
        res = {"model": which, "held_out": replay(env, HELD_OUT, on_traj=lambda i, n: progress(
            "calib", f"fidelity {name}", i, n))}
    elif name in ("b_uncal", "b_cal"):
        res = gait_policy(name[2:], on_ep=lambda i, n: progress("calib", f"gait {name}", i, n) if i % 5 == 0 else None)
    elif name == "c":
        res = gait_recorded()
    else:
        raise SystemExit(f"unknown part {name}")
    write_new(path, res)
    progress("calib", f"{name} done")
    print(f"{name}: wrote {path}")


def report():
    if os.path.exists(OUT_JSON) and os.path.exists(OUT_TXT):
        print("report: done already")
        return
    cal = frozen_calibration()
    P = {n: json.load(open(part_path(n))) for n in ("a_uncal", "a_cal", "b_uncal", "b_cal", "c")}
    strip = lambda b: {k: v for k, v in b.items() if k != "episodes"}  # noqa: E731
    out = {
        "what": "Step 2 fidelity of the hand-built MuJoCo ANYmal D model (scripts/s2/fidelity.py); "
                "validation, not results",
        "calibration": {k: cal[k] for k in ("chosen", "objective", "cal_xml_sha256")},
        "replay_held_out": {"uncalibrated": P["a_uncal"]["held_out"], "calibrated": P["a_cal"]["held_out"]},
        "gait_authors_policy": {"uncalibrated": strip(P["b_uncal"]), "calibrated": strip(P["b_cal"])},
        "gait_recorded": P["c"],
        "episodes_files": {n: os.path.relpath(part_path(n), AIR) for n in P},
    }
    write_new(OUT_JSON, out)
    L = ["Step 2 fidelity (hand-built MuJoCo ANYmal D; validation, not results)",
         "calibrated constants: " + ", ".join(f"{k}={v}" for k, v in cal["chosen"].items()),
         f"training objective E8 = {cal['objective']['E8']:.4f}, E1 = {cal['objective']['E1']:.4f}",
         "", "(a) replay of recorded joint commands, held-out episodes 1 and 8 (4 trajectories)",
         f"{'h':>4} {'uncal E':>9} {'cal E':>9} {'floor':>9} {'uncal Nm':>9} {'cal Nm':>9}"]
    for h in HORIZONS:
        u, c = P["a_uncal"]["held_out"]["horizons"][str(h)], P["a_cal"]["held_out"]["horizons"][str(h)]
        L.append(f"{h:>4} {u['mean']:9.4f} {c['mean']:9.4f} {c['floor']:9.4f} {u['torque_Nm']:9.2f} "
                 f"{c['torque_Nm']:9.2f}")
    L.append("unstable windows (uncal, cal): "
             f"{sum(P['a_uncal']['held_out']['unstable_windows'])}, {sum(P['a_cal']['held_out']['unstable_windows'])}")
    L += ["", "(b) authors' policy, 10 schedules x 8 starts; (c) recorded data, same schedules",
          f"{'':>12} {'falls':>7} {'2 feet':>7} {'duty':>7} {'speed':>7} {'turn':>7} {'return':>8}"]
    for lab, s in (("MuJoCo uncal", P["b_uncal"]["summary"]), ("MuJoCo cal", P["b_cal"]["summary"]),
                   ("recorded", P["c"]["summary"])):
        L.append(f"{lab:>12} {s['fall_rate']:7.3f} {s['two_feet']:7.3f} {s['duty']:7.3f} {s['speed_err']:7.3f} "
                 f"{s['turn_err']:7.3f} {s.get('ret', float('nan')):8.2f}")
    L += ["speed in m/s, turn in rad/s; 2 feet = share of time with exactly two feet down",
          "Gate 1 is applied by M3 (G1 falls < 10%, G2 |2 feet - recorded| <= 0.10, G3 speed <= 1.5 x recorded)"]
    assert len(L) <= 40
    tmp = OUT_TXT + ".tmp"
    open(tmp, "w").write("\n".join(L) + "\n")
    if os.path.exists(OUT_TXT):
        raise SystemExit(f"{OUT_TXT} exists")
    os.replace(tmp, OUT_TXT)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get("S2_THREADS", "2")))
    report() if a.part == "report" else run_part(a.part)


if __name__ == "__main__":
    main()

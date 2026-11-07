"""
Step 2 (M1) tests of the MuJoCo ANYmal D model and environment (STEP2_PLAN.md M1, tests a-f) and the speed probe.

  ./py-mj scripts/s2/test_mj.py            # run, print
  ./py-mj scripts/s2/test_mj.py --write    # also write results/s2/m1_tests.json (never overwritten)

(a) The model compiles; total mass and the whole-robot inertia about its centre of mass (all joints at zero) match
    the URDF to 1e-6, computed two independent ways: a direct per-link sum over the URDF's tree, and MuJoCo's own
    URDF import (in memory, double precision, every link kept as a body).
(b) Joint order and signs: actuators and state joints are in the data's order (rwm_data.JOINT_NAMES); at the default
    pose each foot lies under its own corner of the trunk; after 2 s of standing on zero actions the signs of the
    hip-flexion (HFE) and knee (KFE) torques match the signs of the recorded mean torques over training-episode rows
    with all four feet down. Hip abduction (HAA) torques are reported, not tested: their recorded mean is small and
    changes sign between summaries.
(c) The motor wrapper equals Isaac Lab's computation (IdealPDActuator.compute, DCMotor._clip_effort, the delay
    buffer's lag clamped to the history available), re-implemented here in torch from the quoted source, on random
    inputs including joint speeds beyond the clip's range.
(d) Zero actions for 10 s: no fall, and the base height stays within 2 cm of its value at 1 s (the first second is
    the drop from the 0.6 m start onto the feet).
(e) The observation builder equals s1_policy.observation on a constructed state.
(f) The authors' policy runs for 20 s (1000 policy steps, training episode 0's velocity-command schedule) without
    errors and with finite values. Nothing about walking is measured here (that is Gate 1).
Probe: wall-clock seconds per 20-second episode with the policy (2 threads).
"""
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

import actuator as ACT
import build_model as BM
import mj_env as E

torch.set_num_threads(2)
OUT = os.path.join(BM.AIR, "results", "s2", "m1_tests.json")
RES = {}


def composite(masses, coms, inertias):
    M = sum(masses)
    c = sum(m * x for m, x in zip(masses, coms)) / M
    I = np.zeros((3, 3))
    for m, x, Ii in zip(masses, coms, inertias):
        r = x - c
        I += Ii + m * (r @ r * np.eye(3) - np.outer(r, r))
    return M, c, I


def urdf_direct():
    """Per-link world inertias by forward kinematics over the URDF tree at zero joint angles (no fusion)."""
    links, joints = BM.parse()
    T = {"base": np.eye(4)}
    pending = list(joints)
    while pending:
        for j in list(pending):
            if j["parent"] in T:
                T[j["child"]] = T[j["parent"]] @ j["T"]      # zero angle: revolute adds no rotation
                pending.remove(j)
    ms, cs, Is = [], [], []
    for ln, d in links.items():
        if d["inertial"] is None:
            continue
        m, To, I = d["inertial"]
        W = T[ln] @ To
        ms.append(m)
        cs.append(W[:3, 3])
        Is.append(W[:3, :3] @ I @ W[:3, :3].T)
    return composite(ms, cs, Is)


def mujoco_composite(m):
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    if m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        d.qpos[0:7] = [0, 0, 0, 1, 0, 0, 0]               # base at the origin for the comparison
    mujoco.mj_kinematics(m, d)
    ms, cs, Is = [], [], []
    for b in range(1, m.nbody):
        R = d.ximat[b].reshape(3, 3)
        ms.append(m.body_mass[b])
        cs.append(d.xipos[b].copy())
        Is.append(R @ np.diag(m.body_inertia[b]) @ R.T)
    return composite(ms, cs, Is)


def mujoco_urdf_import():
    r = ET.parse(BM.URDF).getroot()
    for l in r.findall("link"):
        for v in l.findall("visual") + l.findall("collision"):
            l.remove(v)
    for j in r.findall("joint"):
        if j.get("name") in BM.PAYLOAD_JOINTS:
            j.set("type", "fixed")
    # fusestatic="false": MuJoCo 3.10's fusing of this URDF moves the fused centres of mass (whole-robot centre of
    # mass by 1.2 cm, inertia by 0.43 kg m^2; step2/notes/M1.md), so the unfused import (97 bodies) is compared.
    r.insert(0, ET.fromstring('<mujoco><compiler discardvisual="true" fusestatic="false"/></mujoco>'))
    r.insert(1, ET.fromstring('<link name="world"/>'))
    r.insert(2, ET.fromstring('<joint name="fb" type="floating"><parent link="world"/><child link="base"/></joint>'))
    return mujoco.MjModel.from_xml_string(ET.tostring(r, encoding="unicode"))


def test_a():
    m = mujoco.MjModel.from_xml_path(E.MODEL_XML)
    M0, c0, I0 = mujoco_composite(m)
    M1, c1, I1 = urdf_direct()
    M2, c2, I2 = mujoco_composite(mujoco_urdf_import())
    err = {"mass_vs_urdf": abs(M0 - M1), "com_vs_urdf": float(np.abs(c0 - c1).max()),
           "inertia_vs_urdf": float(np.abs(I0 - I1).max()), "mass_vs_mj_import": abs(M0 - M2),
           "com_vs_mj_import": float(np.abs(c0 - c2).max()), "inertia_vs_mj_import": float(np.abs(I0 - I2).max())}
    ok = all(v < 1e-6 for v in err.values()) and m.nu == 12 and m.njnt == 13
    return ok, {"total_mass_kg": M0, "inertia_diag_kgm2": np.diag(I0).tolist(), "max_abs_errors": err,
                "nbody": m.nbody, "njnt": m.njnt, "nu": m.nu, "timestep": m.opt.timestep}


def recorded_stance_torque_signs():
    import common as C
    _, d, ep = C.data()
    rows = np.isin(ep, ACT.TRAIN_EPISODES) & (d[:, 61:65].sum(1) == 4)
    return d[rows, 33:45].mean(0), int(rows.sum())


def test_b():
    import common  # noqa: F401  (puts rwm_repro/src on the path)
    import rwm_data
    env = E.MjEnv(reward=False)
    names_act = [mujoco.mj_id2name(env.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(env.m.nu)]
    order_ok = names_act == list(rwm_data.JOINT_NAMES) == E.JOINTS
    env.reset()
    corner_ok = True
    for leg, g in zip(E.LEGS, env.g_foot):
        p = env.d.xmat[env.base].reshape(3, 3).T @ (env.d.geom_xpos[g] - env.d.xpos[env.base])
        sx, sy = (1 if leg[1] == "F" else -1), (1 if leg[0] == "L" else -1)
        corner_ok &= bool(np.sign(p[0]) == sx and np.sign(p[1]) == sy and p[2] < -0.3)
    for _ in range(100):
        _, info = env.step(np.zeros(12))
    sim = info["state"][33:45]
    rec, n = recorded_stance_torque_signs()
    idx = list(range(4, 12))                                    # HFE, KFE
    sign_ok = bool(np.all(np.sign(sim[idx]) == np.sign(rec[idx])))
    ok = order_ok and corner_ok and sign_ok and info["contacts"][4:].sum() == 4
    return ok, {"joint_order": names_act, "feet_under_own_corners": corner_ok,
                "standing_torque_Nm": np.round(sim, 3).tolist(), "recorded_stance_mean_torque_Nm": np.round(rec, 3).tolist(),
                "recorded_rows": n, "hfe_kfe_signs_match": sign_ok}


def isaac_reference(targets, qs, qds, kp, kd, delay):
    """Isaac Lab's DelayedPD -> IdealPD -> DCMotor clip, from the quoted source (torch)."""
    sat, elim, vlim = ACT.SATURATION, ACT.EFFORT_LIMIT, ACT.VELOCITY_LIMIT
    vel_at_effort_lim = vlim * (1 + elim / sat)
    hist, out = [], []
    for tgt, q, qd in zip(targets, qs, qds):
        tgt, joint_pos, joint_vel = (torch.as_tensor(x, dtype=torch.float64) for x in (tgt, q, qd))
        hist.append(tgt)
        lag = min(delay, len(hist) - 1)                          # lag clamped to the pushes so far
        control_pos = hist[-1 - lag]
        error_pos = control_pos - joint_pos
        error_vel = torch.zeros_like(joint_vel) - joint_vel
        computed_effort = kp * error_pos + kd * error_vel + 0.0
        jv = torch.clip(joint_vel, min=-vel_at_effort_lim, max=vel_at_effort_lim)
        torque_speed_top = sat * (1.0 - jv / vlim)
        torque_speed_bottom = sat * (-1.0 - jv / vlim)
        max_effort = torch.clip(torque_speed_top, max=elim)
        min_effort = torch.clip(torque_speed_bottom, min=-elim)
        out.append(torch.clip(computed_effort, min=min_effort, max=max_effort).numpy())
    return np.array(out)


def test_c():
    rng = np.random.default_rng(0)
    mot = ACT.load()
    worst = 0.0
    for trial in range(20):
        T = 60
        targets = np.repeat(rng.uniform(-1.5, 1.5, (T // 4, 12)), 4, axis=0)
        qs = rng.uniform(-1.5, 1.5, (T, 12))
        qds = rng.uniform(-20, 20, (T, 12))
        mot.reset()
        ours = np.array([mot(t, q, qd)[0] for t, q, qd in zip(targets, qs, qds)])
        ref = isaac_reference(targets, qs, qds, mot.kp, mot.kd, mot.delay)
        worst = max(worst, float(np.abs(ours - ref).max()))
    return worst < 1e-9, {"max_abs_diff_Nm": worst, "kp": mot.kp, "kd": mot.kd, "delay_physics_steps": mot.delay}


def test_d():
    env = E.MjEnv(reward=False)
    env.reset()
    h, fell = [], False
    for _ in range(500):
        _, info = env.step(np.zeros(12))
        h.append(info["base_height"])
        fell |= bool(info["fell"])
    h = np.array(h)
    drift = float(np.abs(h[49:] - h[49]).max())
    return (not fell) and drift < 0.02, {"fell": fell, "height_at_1s_m": float(h[49]),
                                         "max_drift_after_1s_m": drift}


def test_e():
    import s1_policy as P
    rng = np.random.default_rng(1)
    cfg = P.C.data()[0]
    sm, ss = np.asarray(cfg["state_data_mean"]), np.asarray(cfg["state_data_std"])
    worst = 0.0
    for _ in range(50):
        x = sm + ss * rng.standard_normal(45)
        cmd, a = rng.uniform(-1, 1, 3), rng.standard_normal(12)
        ours = E.observation(x, cmd, a)
        ref = P.observation(torch.as_tensor(((x - sm) / ss)[None]), torch.as_tensor(cmd[None]),
                            P.norm_action(torch.as_tensor(a[None])))[0].numpy()
        worst = max(worst, float(np.abs(ours - ref).max()))
    return worst < 1e-9, {"max_abs_diff": worst}


def test_f():
    import json as _j
    import s1_policy as P
    C = P.C
    reg = [r for r in _j.load(open(P.REGIMES_JSON))["regimes"] if r["ep"] == 0]
    start = reg[0]["start"]
    actor = P.load_actor()
    env = E.MjEnv()
    obs = env.reset()
    finite = True
    t0 = time.time()
    for k in range(1000):
        row = start + k
        cmd = next(([r["vx"], r["vy"], r["wz"]] for r in reg if r["start"] <= row <= r["end"]), [0.0, 0.0, 0.0])
        env.cmd = np.asarray(cmd)
        obs = E.observation(env.s, env.cmd, env.last_action)
        with torch.no_grad():
            a = actor(torch.as_tensor(obs, dtype=torch.float32)[None])[0].double().numpy()
        obs, info = env.step(a)
        finite &= bool(np.all(np.isfinite(obs)) and np.all(np.isfinite(info["state"]))
                       and np.isfinite(info["reward"]) and np.all(np.isfinite(env.d.qacc)))
    secs = time.time() - t0
    return finite and env.t == 1000, {"steps": env.t, "all_finite": finite, "seconds_per_20s_episode": secs,
                                      "threads": torch.get_num_threads()}


def main():
    ok_all = True
    for name, fn in [("a", test_a), ("b", test_b), ("c", test_c), ("d", test_d), ("e", test_e), ("f", test_f)]:
        t0 = time.time()
        ok, info = fn()
        info["seconds"] = round(time.time() - t0, 2)
        RES[name] = {"pass": bool(ok), **info}
        ok_all &= ok
        print(f"({name}) {'PASS' if ok else 'FAIL'}", json.dumps(info, default=float)[:300])
    RES["all_pass"] = bool(ok_all)
    RES["model_sha256"] = __import__("hashlib").sha256(open(E.MODEL_XML, "rb").read()).hexdigest()
    if "--write" in sys.argv:
        if os.path.exists(OUT):
            raise SystemExit(f"{OUT} exists; result files are never overwritten")
        json.dump(RES, open(OUT, "w"), indent=1, default=float)
        print("wrote", OUT)
    print("ALL PASS" if ok_all else "SOME FAILED")


if __name__ == "__main__":
    main()

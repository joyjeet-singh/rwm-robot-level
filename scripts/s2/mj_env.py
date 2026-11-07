"""
Step 2 (M1): the MuJoCo ANYmal D walking environment (one robot, numpy). Hand-built physics model, not the data's
own simulator and not hardware.

Isaac Lab flat velocity task settings it follows (step2/notes/M0_facts.md b):
- physics step 0.005 s, `decimation` = 4 physics steps per policy step (0.02 s), episode 20 s;
- joint command (target) = default pose + 0.5 * action, every joint, held over the policy step; the motor model
  (scripts/s2/actuator.py) runs at every physics step;
- reset to the default pose at base height 0.6 m (optional joint offsets and heading for the evaluation starts);
- fall: Isaac Lab's `illegal_contact` on body `base`, threshold 1 N: the episode ends when the norm of the net
  contact force on the trunk exceeds 1 N at any physics step of the policy step (Isaac Lab takes the maximum over
  its contact-force history).

The 45-number state, in the data's order and units (lite anymal_d_flat.py:58-63):
  0:3 base linear velocity of the base's centre of mass, in the base frame (Isaac Lab 2.1 root_lin_vel_b);
  3:6 base angular velocity, base frame; 6:9 gravity direction in the base frame;
  9:21 joint angles minus the default pose; 21:33 joint speeds; 33:45 joint torques applied at the policy step's
  last physics step (the recorded torque's timing, scripts/s2/actuator.py). Joints in the data's order:
  HAA x4 (LF, LH, RF, RH), HFE x4, KFE x4.
The 48 observations: exactly as scripts/s1_policy.py `observation` builds them from that state, the velocity
command and the previous action. Foot and thigh contacts: net contact force on the body's shapes > 1 N at the last
physics step (Isaac Lab ContactSensor force_threshold), order thigh LF, LH, RF, RH then foot LF, LH, RF, RH as
the data's 8 contact columns. The reward is the lite code's own terms and weights (s1_policy.step_reward).
"""
import os

import mujoco
import numpy as np

import actuator as ACT

HERE = os.path.dirname(os.path.abspath(__file__))
AIR = os.path.dirname(os.path.dirname(HERE))
MODEL_XML = os.path.join(AIR, "step2", "model", "anymal_d.xml")

LEGS = ("LF", "LH", "RF", "RH")
JOINTS = [f"{leg}_{j}" for j in ("HAA", "HFE", "KFE") for leg in LEGS]
DEFAULT = {"HAA": {"F": 0.0, "H": 0.0}, "HFE": {"F": 0.4, "H": -0.4}, "KFE": {"F": -0.8, "H": 0.8}}
DEFAULT_POS = np.array([DEFAULT[n[3:]][n[1]] for n in JOINTS])
ACTION_SCALE = 0.5
DECIMATION = 4
STEP_DT = 0.02
EPISODE_S = 20.0
START_Z = 0.6
CONTACT_THRESHOLD = 1.0      # N
FALL_THRESHOLD = 1.0         # N


def set_constants(m, friction=None, timeconst=None, armature=None, damping=None):
    """The four calibration constants (STEP2_PLAN.md calibration grid). None leaves the model's value."""
    if friction is not None:
        m.geom_friction[:, 0] = friction                 # all shapes and the floor (MuJoCo takes the larger)
    if timeconst is not None:
        m.geom_solref[:, 0] = timeconst
    leg_dofs = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS]
    if armature is not None:
        m.dof_armature[leg_dofs] = armature
    if damping is not None:
        m.dof_damping[leg_dofs] = damping


class LiteReward:
    """The lite code's reward for one robot: s1_policy.step_reward on a stand-in env (air-time memory per episode)."""

    def __init__(self):
        import torch
        import s1_policy as P
        self.torch, self.P = torch, P
        cfg = P.C.data()[0]
        self.sm = np.asarray(cfg["state_data_mean"], dtype=np.float64)
        self.ss = np.asarray(cfg["state_data_std"], dtype=np.float64)
        self.reset()

    def reset(self):
        self.env = self.P.stand_in_env(1, dtype=self.torch.float64)

    def __call__(self, s_prev, a_prev, cmd, s_next, a, contacts):
        t = lambda x: self.torch.as_tensor(np.asarray(x, dtype=np.float64)[None])  # noqa: E731
        logits = np.where(np.asarray(contacts) > 0.5, 20.0, -20.0)        # sigmoid().round() gives the 0/1 back
        total, terms = self.P.step_reward(self.env, t((s_prev - self.sm) / self.ss), t(a_prev), t(cmd),
                                          t((s_next - self.sm) / self.ss), t(a), t(logits))
        return float(total[0]), terms[0].numpy()


def observation(state, cmd, last_action):
    """48 observations, s1_policy.observation's layout, raw units."""
    return np.concatenate([state[0:3], state[3:6], state[6:9], np.asarray(cmd, dtype=np.float64),
                           state[9:21], state[21:33], np.asarray(last_action, dtype=np.float64)])


class MjEnv:
    def __init__(self, motor=None, constants=None, xml=MODEL_XML, reward=True):
        self.m = mujoco.MjModel.from_xml_path(xml)
        set_constants(self.m, **(constants or {}))
        self.d = mujoco.MjData(self.m)
        self.motor = motor if motor is not None else ACT.load()
        name = lambda t, n: mujoco.mj_name2id(self.m, t, n)  # noqa: E731
        J = mujoco.mjtObj.mjOBJ_JOINT
        self.qadr = np.array([self.m.jnt_qposadr[name(J, j)] for j in JOINTS])
        self.vadr = np.array([self.m.jnt_dofadr[name(J, j)] for j in JOINTS])
        self.act = np.array([name(mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in JOINTS])
        assert list(self.act) == list(range(12)), "actuators must be in the data's joint order"
        self.base = name(mujoco.mjtObj.mjOBJ_BODY, "base")
        G = mujoco.mjtObj.mjOBJ_GEOM
        self.g_trunk = name(G, "trunk")
        self.g_floor = name(G, "floor")
        self.g_thigh = [name(G, f"{leg}_thigh") for leg in LEGS]
        self.g_foot = [name(G, f"{leg}_foot") for leg in LEGS]
        self.g_shank = [name(G, f"{leg}_shank") for leg in LEGS]
        self.reward_fn = LiteReward() if reward else None
        self.cmd = np.zeros(3)
        self._f6 = np.zeros(6)

    # -------------------------------------------------------------- state
    def _geom_forces(self):
        """Net contact force vector (world frame) per geom, from all current contacts."""
        F = np.zeros((self.m.ngeom, 3))
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            mujoco.mj_contactForce(self.m, self.d, i, self._f6)
            fw = c.frame.reshape(3, 3).T @ self._f6[:3]       # force on geom2 from geom1, world frame
            F[c.geom1] -= fw
            F[c.geom2] += fw
        return F

    def contacts(self, F=None):
        """8 contacts: thigh LF, LH, RF, RH, then foot (sphere + shank tube, the URDF FOOT link) LF, LH, RF, RH."""
        F = self._geom_forces() if F is None else F
        th = [np.linalg.norm(F[g]) > CONTACT_THRESHOLD for g in self.g_thigh]
        ft = [np.linalg.norm(F[a] + F[b]) > CONTACT_THRESHOLD for a, b in zip(self.g_foot, self.g_shank)]
        return np.array(th + ft, dtype=np.float64)

    def state(self):
        d, b = self.d, self.base
        R = d.xmat[b].reshape(3, 3)                        # base (link) frame -> world
        w_local = d.qvel[3:6].copy()                       # free joint: angular velocity in the body frame
        w_world = R @ w_local
        v_origin = d.qvel[0:3]                             # free joint: linear velocity of the frame origin, world
        v_com = v_origin + np.cross(w_world, d.xipos[b] - d.xpos[b])
        return np.concatenate([R.T @ v_com, w_local, R.T @ np.array([0.0, 0.0, -1.0]),
                               d.qpos[self.qadr] - DEFAULT_POS, d.qvel[self.vadr], self.torque])

    # -------------------------------------------------------------- episode
    def reset(self, joint_offset=None, yaw=0.0, cmd=(0.0, 0.0, 0.0)):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[0:3] = [0.0, 0.0, START_Z]
        self.d.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        q = DEFAULT_POS.copy()
        if joint_offset is not None:
            q = q + np.asarray(joint_offset)
        self.d.qpos[self.qadr] = q
        mujoco.mj_forward(self.m, self.d)
        self.motor.reset()
        if self.reward_fn is not None:
            self.reward_fn.reset()
        self.torque = np.zeros(12)
        self.last_action = np.zeros(12)
        self.cmd = np.asarray(cmd, dtype=np.float64)
        self.t = 0
        self.s = self.state()
        return observation(self.s, self.cmd, self.last_action)

    def step(self, action):
        """One policy step. Returns (observation, info); info has state, contacts, fell, reward, terms."""
        action = np.asarray(action, dtype=np.float64)
        target = DEFAULT_POS + ACTION_SCALE * action
        trunk_force = 0.0
        for _ in range(DECIMATION):
            applied, _ = self.motor(target, self.d.qpos[self.qadr], self.d.qvel[self.vadr])
            self.d.ctrl[:] = applied
            mujoco.mj_step(self.m, self.d)
            self.torque = applied
            F = self._geom_forces()
            trunk_force = max(trunk_force, float(np.linalg.norm(F[self.g_trunk])))
        self.t += 1
        s_prev, a_prev = self.s, self.last_action
        self.s = self.state()
        con = self.contacts(F)
        fell = trunk_force > FALL_THRESHOLD
        info = {"state": self.s, "contacts": con, "fell": fell, "trunk_force": trunk_force,
                "base_height": float(self.d.qpos[2]), "time_out": self.t * STEP_DT >= EPISODE_S - 1e-9}
        if self.reward_fn is not None:
            info["reward"], info["terms"] = self.reward_fn(s_prev, a_prev, self.cmd, self.s, action, con)
        self.last_action = action
        return observation(self.s, self.cmd, self.last_action), info

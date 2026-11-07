"""
Step 2 (M1): the motor model. The ANYdrive network file could not be obtained (step2/assets/manifest.json,
actuator.status FAILED), so this is STEP2_PLAN.md's fallback: a PD controller with a fixed delay, fitted to the
recorded torques of the training episodes.

Isaac Lab source this re-implements (v3.0.0, isaaclab/actuators/, quoted in step2/notes/M1_sources.md):
- IdealPDActuator.compute: effort = stiffness * (target_pos - pos) + damping * (target_vel - vel) + effort_ff,
  then clipped by `_clip_effort(effort, joint_vel)`. Here target_vel = 0 and effort_ff = 0, as
  JointPositionAction sends.
- DCMotor._clip_effort (ActuatorNetLSTM inherits from DCMotor, so the ANYdrive net is clipped the same way):
      joint_vel  = clip(joint_vel, -vel_at_effort_lim, +vel_at_effort_lim)
      top        = saturation * ( 1 - joint_vel / velocity_limit)
      bottom     = saturation * (-1 - joint_vel / velocity_limit)
      effort     = clip(effort, max(bottom, -effort_limit), min(top, effort_limit))
  with vel_at_effort_lim = velocity_limit * (1 + effort_limit / saturation). ANYMAL_D: saturation_effort 120,
  actuator_effort_limit 80, actuator_velocity_limit 7.5 (step2/notes/M0_facts.md a).
- DelayedPDActuator: the position target passes through a delay buffer before the PD law; the lag is clamped to
  the history available, so just after a reset the oldest stored target is used.

Fit (`fit`, training episodes 0,2,3,4,5,6,7,9 only; never 1 or 8):
- Timing. A recorded row r holds state r and joint command A[r]; the policy's output from state r is A[r+1]
  (Step 1's registered pairing, confirmed here: step2/notes/M1.md), and A[r+1] is sent during the step from row r
  to row r+1. The torque stored in row r+1 is taken to be the torque applied at that step's last physics step,
  computed from the joint state 3/4 of the way from row r to row r+1 (linear interpolation).
- Delay d (physics steps of 0.005 s): the target used at physics step s is the command of the policy step that
  contained step s - d. Kp and Kd (one pair shared by all 12 joints, as one network serves all 12 in Isaac Lab)
  are fitted by least squares for each d in 0..11 on rows whose recorded torque is below 75 N m (clip inactive).
  The d with the smallest error is kept; delays the torques cannot tell apart give identical errors, and the
  smallest such d is kept.
"""
import json
import os

import numpy as np

SATURATION, EFFORT_LIMIT, VELOCITY_LIMIT = 120.0, 80.0, 7.5
VEL_AT_EFFORT_LIM = VELOCITY_LIMIT * (1.0 + EFFORT_LIMIT / SATURATION)
DECIMATION = 4
TRAIN_EPISODES = (0, 2, 3, 4, 5, 6, 7, 9)
HERE = os.path.dirname(os.path.abspath(__file__))
AIR = os.path.dirname(os.path.dirname(HERE))
FIT_JSON = os.path.join(AIR, "results", "s2", "pd_fit.json")


def clip_effort(effort, joint_vel):
    """DCMotor._clip_effort, in numpy."""
    v = np.clip(joint_vel, -VEL_AT_EFFORT_LIM, VEL_AT_EFFORT_LIM)
    top = SATURATION * (1.0 - v / VELOCITY_LIMIT)
    bottom = SATURATION * (-1.0 - v / VELOCITY_LIMIT)
    max_e = np.minimum(top, EFFORT_LIMIT)
    min_e = np.maximum(bottom, -EFFORT_LIMIT)
    return np.minimum(np.maximum(effort, min_e), max_e)


class DelayedPD:
    """PD with a fixed delay of `delay` physics steps on the position target, per joint, with reset."""

    def __init__(self, kp, kd, delay, n=12):
        self.kp, self.kd, self.delay, self.n = float(kp), float(kd), int(delay), n
        self.reset()

    def reset(self):
        self._buf = []                          # newest last; at most delay + 1 targets

    def __call__(self, target, joint_pos, joint_vel):
        """One physics step: push the target, read the delayed one, return (applied, computed) torque."""
        self._buf.append(np.array(target, dtype=np.float64))
        if len(self._buf) > self.delay + 1:
            self._buf.pop(0)
        lag = min(self.delay, len(self._buf) - 1)
        tgt = self._buf[-1 - lag]
        computed = self.kp * (tgt - joint_pos) + self.kd * (0.0 - joint_vel)
        return clip_effort(computed, joint_vel), computed


def load(path=FIT_JSON):
    p = json.load(open(path))
    return DelayedPD(p["kp"], p["kd"], p["delay_physics_steps"])


# ------------------------------------------------------------------ fit to the recorded torques
def _design(d, ep, delay, episodes=TRAIN_EPISODES, w=0.75):
    """Rows k (torque of the step from row k-1 to k): X = (target - q, -qd), y = recorded torque."""
    jp, jv, tau, act = d[:, 9:21], d[:, 21:33], d[:, 33:45], d[:, 45:57]
    X1, X2, Y = [], [], []
    for e in episodes:
        rows = np.where(ep == e)[0]
        lo = 3 + (delay + DECIMATION) // DECIMATION
        k = rows[lo:]
        # last physics step of the step (k-1 -> k) is s = 4(k-1)+3 (local to the episode, index-free);
        # with delay d its target is the command of the policy step containing s - d, i.e. step (k-1) - m,
        # m = ceil((d - 3) / 4) clipped at 0; the command sent during step (j -> j+1) is A[j + 1].
        m = max(0, -(-(delay - (DECIMATION - 1)) // DECIMATION))
        tgt = 0.5 * act[k - m]                       # joint_pos is relative to the default pose (scale 0.5)
        q = jp[k - 1] + w * (jp[k] - jp[k - 1])
        qd = jv[k - 1] + w * (jv[k] - jv[k - 1])
        X1.append(tgt - q)
        X2.append(-qd)
        Y.append(tau[k])
    return np.concatenate(X1), np.concatenate(X2), np.concatenate(Y)


def fit(max_delay=11, write=True):
    import common as C
    _, d, ep = C.data()
    rows = []
    for delay in range(max_delay + 1):
        x1, x2, y = (a.ravel() for a in _design(d, ep, delay))
        ok = np.abs(y) < 75.0
        A = np.stack([x1[ok], x2[ok]], 1)
        (kp, kd), *_ = np.linalg.lstsq(A, y[ok], rcond=None)
        pred = A @ np.array([kp, kd])
        rmse = float(np.sqrt(np.mean((pred - y[ok]) ** 2)))
        r2 = float(1 - np.mean((pred - y[ok]) ** 2) / np.var(y[ok]))
        rows.append({"delay": delay, "kp": float(kp), "kd": float(kd), "rmse": rmse, "r2": r2, "n": int(ok.sum())})
    best_rmse = min(r["rmse"] for r in rows)
    best = min((r for r in rows if r["rmse"] <= best_rmse + 1e-12), key=lambda r: r["delay"])
    out = {
        "what": "PD-with-delay fallback motor model fitted to recorded torques (STEP2_PLAN.md motor model)",
        "episodes": list(TRAIN_EPISODES), "unsaturated_rule": "|recorded torque| < 75 N m",
        "timing": "torque row k = last physics step of step k-1 -> k, state interpolated w = 0.75",
        "kp": best["kp"], "kd": best["kd"], "delay_physics_steps": best["delay"],
        "delay_s": best["delay"] * 0.005, "rmse_Nm": best["rmse"], "r2": best["r2"],
        "clip": {"saturation_effort": SATURATION, "effort_limit": EFFORT_LIMIT, "velocity_limit": VELOCITY_LIMIT},
        "grid": rows,
    }
    if write:
        os.makedirs(os.path.dirname(FIT_JSON), exist_ok=True)
        if os.path.exists(FIT_JSON):
            raise SystemExit(f"{FIT_JSON} exists; result files are never overwritten")
        json.dump(out, open(FIT_JSON, "w"), indent=1)
    return out


if __name__ == "__main__":
    o = fit()
    for r in o["grid"]:
        print(r["delay"], round(r["kp"], 2), round(r["kd"], 3), round(r["rmse"], 3), round(r["r2"], 4))
    print("chosen delay", o["delay_physics_steps"], "kp", round(o["kp"], 2), "kd", round(o["kd"], 3))

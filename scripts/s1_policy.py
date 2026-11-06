"""
Study CL (Step 1): the authors' policy as a source of joint commands for a GRU simulator.

- The actor is the released checkpoint's `model_state_dict` actor (48 -> 128 -> 128 -> 128 -> 12, ELU; layout as
  ../rwm_repro/scripts/task1c_policy_test.py:50 build_actor). Its mean output is used: no action noise.
- The observation is built exactly as ../robotic_world_model_lite/scripts/envs/anymal_d_flat.py:39-56
  get_imagination_observation builds it, from the simulator's current (normalised) state, denormalised with the
  model's normalisation, the velocity command and the previous joint command; without observation noise
  (step1/DECISIONS.md S1-2). `authors_observation` calls the authors' own method on a stand-in object; the
  test checks `observation` against it.
- The velocity command (walking speed vx, vy and turning rate wz the policy is asked to follow) is not recorded.
  As task1c builds it (command_per_row, task1c l.65), each row gets the mean of the Step 0 velocity regime that
  covers it (../rwm_repro/results/step0_regimes.json).
- Timing (task1c, offset -1, step1/notes/S0_inventory.md item a): fed state row r and the joint command of row r,
  the actor's output stands for recorded joint-command row r + 1. In lyap.rollout_map that row is the command
  that moves state row r to row r + 1 (action_offset = 1), so the closed loop keeps the registered pairing.
- `step_reward` calls the authors' `_compute_imagination_reward_terms` (anymal_d_flat.py:97-163) on the stand-in
  and weights the terms as envs/base.py:160-171 does (w * term * step_dt), with the configured weights
  (anymal_d_flat_cfg.py:15-29).
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

import common as C

sys.path.insert(0, os.path.join(C.LITE, "scripts"))
from envs.anymal_d_flat import AnymalDFlatEnv  # noqa: E402

TASK1C_JSON = os.path.join(C.RWM, "results", "task1c_policy.json")
REGIMES_JSON = os.path.join(C.RWM, "results", "step0_regimes.json")
OBS_DIM, ACT_DIM = 48, 12


# ------------------------------------------------------------------ actor
def load_actor(path=None):
    """The released actor in eval mode, float32."""
    path = path or C.W_RELEASED
    sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    net = nn.Sequential(nn.Linear(OBS_DIM, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(),
                        nn.Linear(128, 128), nn.ELU(), nn.Linear(128, ACT_DIM))
    with torch.no_grad():
        for i in (0, 2, 4, 6):
            net[i].weight.copy_(sd[f"actor.{i}.weight"])
            net[i].bias.copy_(sd[f"actor.{i}.bias"])
    net.eval()
    for q in net.parameters():
        q.requires_grad_(False)
    return net


# ------------------------------------------------------------------ velocity command
_CMD = {}


def command_per_row():
    """(10000, 3) velocity command per CSV row: the covering Step 0 regime's mean (vx, vy, wz); zero elsewhere."""
    if "cmd" not in _CMD:
        _, dat, _ = C.data()
        regimes = json.load(open(REGIMES_JSON))["regimes"]
        cmd = np.zeros((len(dat), 3))
        for r in regimes:
            cmd[r["start"]:r["end"] + 1] = [r["vx"], r["vy"], r["wz"]]
        _CMD["cmd"] = cmd
    return _CMD["cmd"]


def arena_commands(ar):
    """(B, LEN, 3) float64 velocity command for every row of every arena trajectory."""
    cmd = command_per_row()
    idx = np.asarray(ar["starts"])[:, None] + np.arange(C.LEN)[None, :]
    return torch.as_tensor(cmd[idx], dtype=torch.float64)


# ------------------------------------------------------------------ normalisation (the model's own)
def _stats(dtype):
    cfg, _, _ = C.data()
    f = lambda k: torch.as_tensor(np.asarray(cfg[k], dtype=np.float64), dtype=dtype)  # noqa: E731
    return f("state_data_mean"), f("state_data_std"), f("action_data_mean"), f("action_data_std")


def denorm_state(s):
    sm, ss, _, _ = _stats(s.dtype)
    return s * ss + sm


def denorm_action(a):
    _, _, am, asd = _stats(a.dtype)
    return a * asd + am


def norm_action(a):
    """The model's input normalisation of a joint command (identity for this config: mean 0, std 1)."""
    _, _, am, asd = _stats(a.dtype)
    return (a - am) / asd


# ------------------------------------------------------------------ observation
def observation(s, cmd, a_prev):
    """
    get_imagination_observation, differentiable: s (B,45) normalised state, cmd (B,3) velocity command,
    a_prev (B,12) previous joint command as given to the model. Returns (B,48) in raw units.
    """
    x = denorm_state(s)
    return torch.cat([x[:, 0:3], x[:, 3:6], x[:, 6:9], cmd.to(s.dtype), x[:, 9:21], x[:, 21:33],
                      denorm_action(a_prev)], dim=1)


class _Dataset:
    def denormalize(self, state, action):
        return (denorm_state(state) if state is not None else None,
                denorm_action(action) if action is not None else None)


def stand_in_env(B, dtype=torch.float32):
    """The authors' AnymalDFlatEnv without its constructor: only what the observation and reward methods read."""
    env = AnymalDFlatEnv.__new__(AnymalDFlatEnv)
    env._num_envs = B
    env.device = "cpu"
    env._step_dt = C.data()[0]["step_dt"]
    env.observation_noise = False
    env.dataset = _Dataset()
    env.base_velocity = torch.zeros(B, 3, dtype=dtype)
    env._init_additional_imagination_attributes()
    return env


def authors_observation(env, s, cmd, a_prev):
    """The authors' method, for the test: state and action histories of length 1."""
    env.base_velocity = cmd.to(s.dtype)
    return env.get_imagination_observation(s[:, None], a_prev[:, None])["policy"]


def reward_weights():
    mod = sys.modules["configs"]                       # loaded by rwm_data.load_reference_config (C.data())
    e = mod.AnymalDFlatConfig.EnvironmentConfig()
    return dict(e.reward_term_weights), float(e.uncertainty_penalty_weight)


def step_reward(env, s_prev, a_prev, cmd, s_next, a, contact_logits):
    """
    One imagination step's reward, as policy_training.py:68-70 orders the calls: the observation of the current
    state (sets last_obs and obs_last_action), then the reward terms of the predicted next state with the joint
    command a. Returns (B,) weighted sum and (B, 12) weighted terms, each already multiplied by step_dt.
    """
    W, _ = reward_weights()
    authors_observation(env, s_prev, cmd, a_prev)
    parsed = env._parse_imagination_states(denorm_state(s_next))
    contacts = env._parse_contacts(contact_logits)
    env._compute_imagination_reward_terms(parsed, denorm_action(a), None, contacts)
    terms = torch.stack([W[k] * env.imagination_reward_per_step[k] * env._step_dt for k in W], -1)
    return terms.sum(-1), terms


# ------------------------------------------------------------------ the policy as a source of joint commands
class Policy:
    """
    pi(s, a_prev, r): the joint command for the step that moves state row r to row r + 1, from the normalised
    state s at row r, the previous joint command a_prev (the one that produced row r) and the velocity command
    of row r. Output normalised like the model's inputs. Works in float32 and float64 (actor copied per dtype).
    """

    def __init__(self, actor, cmd):
        self.actor, self.cmd = actor, cmd
        self._nets = {d: load_actor_like(actor).to(d) for d in (torch.float32, torch.float64)}

    def net(self, dtype):
        return self._nets[dtype]

    def __call__(self, s, a_prev, r):
        obs = observation(s, self.cmd[:, r], a_prev)
        return norm_action(self.net(s.dtype)(obs))


def load_actor_like(actor):
    net = nn.Sequential(nn.Linear(OBS_DIM, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(),
                        nn.Linear(128, 128), nn.ELU(), nn.Linear(128, ACT_DIM))
    net.load_state_dict(actor.state_dict())
    net.eval()
    for q in net.parameters():
        q.requires_grad_(False)
    return net

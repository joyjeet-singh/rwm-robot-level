"""Option C (C2R): tests of the Amendment 1 fixes in scripts/c/train_policy.py and of scripts/c/pilot2_rule.py.

    ./py scripts/c/test_c2r.py [part ...]     # parts: short floor starts fallcost earlystop rule (default: all)

  * short: one iteration with 64 robots in each pilot simulator, nonzero uncertainty, penalty and fall cost on;
  * floor: the noise floor holds when the noise is forced below it (a direct optimiser step, and a run started at 0.01);
  * starts: the start states are all from source 0 (6M and 10k rungs, and the short runs' records);
  * fallcost: with the fall cost on, an episode that ends in a predicted fall is charged 4.0 exactly once;
  * earlystop: the rule fires on fixtures, not on REL__p0's recorded metrics; a run stopped by it keeps its record;
  * rule: scripts/c/pilot2_rule.py on fixtures, every branch.
Test runs are named `_test*`; they write under runs/c/_test*, runs/c/lite/ and .cache/ only.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
PY = os.path.join(AIR, "py")
LOGS = os.path.join(AIR, ".cache", "logs")


def ok(msg):
    print(f"ok: {msg}", flush=True)


def lite_metrics(name):
    p = os.path.join(AIR, "runs", "c", "lite", "runs", "lite", "s2_" + name, "metrics.jsonl")
    return [json.loads(x) for x in open(p)]


def run_train(name, args, log):
    cmd = [PY, os.path.join(HERE, "train_policy.py"), "--seed", "0", "--name", name, "--num_envs", "64",
           "--threads", "2", "--job", "test", *args]
    rc = subprocess.run(cmd, cwd=AIR, capture_output=True, text=True)
    os.makedirs(LOGS, exist_ok=True)
    open(os.path.join(LOGS, log), "w").write(rc.stdout + rc.stderr)
    assert rc.returncode == 0, rc.stderr[-2000:]
    return rc


def part_short():
    for sim in ("sim_6M_N8_s0", "sim_10k_N8_s0"):
        name = f"_test_c2r_{sim}"
        run_train(name, ["--sim", sim, "--penalty", "0.5", "--fall-cost", "4", "--iterations", "1", "--saves", "1"],
                  f"c2r_short_{sim}.log")
        d = os.path.join(AIR, "runs", "c", name)
        assert os.path.exists(os.path.join(d, "actor_i1.pt"))
        m = [r for r in lite_metrics(name) if "Imagination/epistemic_uncertainty" in r][-1]
        u, pen = m["Imagination/epistemic_uncertainty"], m.get("Imagination/uncertainty", 0.0)
        assert u > 0, u
        assert "Imagination/predicted_fall_share" in m and "Imagination/fall_cost_per_step" in m, sorted(m)
        falls, steps = m["Imagination/predicted_falls"], 24 * 64
        assert abs(m["Imagination/fall_cost_per_step"] + 4.0 * falls / steps) < 1e-12
        assert m["Policy/mean_noise_std"] >= 0.05
        st = json.load(open(os.path.join(d, "starts.json")))
        assert st["source"] == 0 and list(st["window_sources"]) == ["0"], st
        w = json.load(open(os.path.join(d, "wrapper.json")))
        assert w["uncertainty_weight"] == 0.5 and w["fall_cost"] == 4.0
        ok(f"one iteration, 64 robots, {sim}: disagreement {u:.3f}, penalty term {pen:.4f}, predicted falls {falls}, "
           f"{st['windows']} start windows, all source 0")


def part_floor():
    import torch
    import train_policy as W
    W.patch_noise_floor(0.05)
    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import ActorCritic
    from tensordict import TensorDict
    obs = TensorDict({"policy": torch.zeros(4, 48)}, batch_size=[4])
    pol = ActorCritic(obs, {"policy": ["policy"], "critic": ["policy"]}, 12, init_noise_std=1.0)
    alg = PPO(pol, learning_rate=1.0)
    assert float(pol.std.min()) >= 0.05
    with torch.no_grad():
        pol.std.fill_(0.01)                        # forced below the floor
    pol.std.grad = torch.ones_like(pol.std) * 10.0  # a step that pushes it further down (towards negative)
    alg.optimizer.step()
    assert abs(float(pol.std.min()) - 0.05) < 1e-7 and float(pol.std.max()) < 0.0501, pol.std
    ok("a direct optimiser step from 0.01, pushed down: the noise is clamped to 0.05")

    name = "_test_c2r_floor"
    run_train(name, ["--sim", "sim_10k_N8_s0", "--penalty", "1", "--iterations", "2", "--saves", "2",
                     "--test-init-std", "0.01"], "c2r_floor.log")
    m = [r for r in lite_metrics(name) if "Policy/mean_noise_std" in r]
    std = [r["Policy/mean_noise_std"] for r in m]
    assert all(s >= 0.05 - 1e-9 for s in std), std
    run = os.path.join(AIR, "runs", "c", "lite", "runs", "lite", "s2_" + name)
    cks = sorted(f for f in os.listdir(run) if f.startswith("policy_") and f.endswith(".pt"))
    sd = torch.load(os.path.join(run, cks[-1]), map_location="cpu", weights_only=False)["model_state_dict"]
    assert float(sd["std"].min()) >= 0.05 - 1e-9, sd["std"]
    ok(f"a run started at noise 0.01: logged noise {['%.3f' % s for s in std]}, saved minimum {float(sd['std'].min()):.3f}")


def part_starts():
    import train_policy as W
    for rung in ("6M", "10k"):
        a = np.load(os.path.join(W.DATA_DIR, f"rung_{rung}.npy"), mmap_mode="r")
        _, _, starts, info = W.build_starts(rung, np.zeros(45), np.ones(45), source=0)
        src = np.asarray(a[starts[:, None] + np.arange(W.HISTORY), W.SOURCE_COL])
        assert (src == 0).all() and len(starts) > 0
        assert info["rows_used"] <= W.MAX_ROWS
        _, _, again, _ = W.build_starts(rung, np.zeros(45), np.ones(45), source=0)
        assert np.array_equal(starts, again)
        ok(f"rung {rung}: {len(starts)} start windows, every row of every window from source 0, repeatable")


def make_env(n=4):
    import torch
    from tensordict import TensorDict
    import train_policy as W  # noqa: F401  (lite paths)
    import s1_policy as P
    P.C.data()
    from envs.anymal_d_flat import AnymalDFlatEnv
    weights, _ = P.reward_weights()
    env = AnymalDFlatEnv(n, 256, 0.02, weights, "cpu", -1.0, False, [100, 120], [48, 96])

    class Dyn:
        ensemble_size = 5

        def reset_partial(self, ids):
            pass
    env.system_dynamics = Dyn()
    env.system_dynamics_model_ids = torch.zeros(1, n, 1, dtype=torch.long)
    env.common_step_counter = 0
    env._init_imagination_reward_buffer()
    env._init_intervals()
    env._init_additional_imagination_attributes()
    env._init_imagination_command()
    env.extras = {}
    env.last_obs = TensorDict({"policy": torch.zeros(n, env.observation_dim)}, batch_size=[n])
    env.epistemic_uncertainty = torch.zeros(n)
    return env


def part_fallcost():
    import torch
    import train_policy as W
    for cost in (4.0, 0.0):
        env = make_env()
        counter = {"falls": 0, "ends": 0, "charged": 0.0}
        W.patch_fall_cost(cost, counter)
        R, D = [], []
        for k in range(6):
            env.termination_flags = torch.tensor([k == 2, False, k == 4, False])
            r, d, _ = env._post_imagination_step()
            R.append(r.clone()), D.append(d.clone())
        R, D = torch.stack(R), torch.stack(D)
        want = torch.zeros(6, 4, dtype=R.dtype)
        want[2, 0] = want[4, 2] = -cost
        assert torch.equal(R, want), R            # every other term is zero here, so the reward is the charge alone
        assert int(D[2, 0]) == 1 and int(D[4, 2]) == 1 and int(D.sum()) == 2
        assert counter["falls"] == 2 and counter["charged"] == 2 * cost
        ok(f"fall cost {cost:g}: each predicted fall charged {cost:g} exactly once, at the step its episode ends; "
           f"no other step charged")
    W.patch_fall_cost(0.0, {"falls": 0, "ends": 0, "charged": 0.0})
    # a time-out without a fall is not charged; a fall at the time-out step is charged once
    env = make_env()
    counter = {"falls": 0, "ends": 0, "charged": 0.0}
    W.patch_fall_cost(4.0, counter)
    env.episode_length_buf[:] = 255
    env.termination_flags = torch.tensor([True, False, False, False])
    r, d, _ = env._post_imagination_step()
    assert r.tolist() == [-4.0, 0.0, 0.0, 0.0] and d.tolist() == [1, 1, 1, 1]
    ok("at the 256-step limit: a fall is charged once, a time-out is not")


def part_earlystop():
    import train_policy as W
    lim = 256
    assert W.collapse_at([10.0] * 1500, lim) == 400
    assert W.collapse_at([None] * 5 + [10.0] * 1495, lim) == 400
    x = [10.0] * 1500
    x[349] = 100.0                                # iteration 350 is long: the streak restarts at 351
    assert W.collapse_at(x, lim) == 450
    assert W.collapse_at([10.0] * 399, lim) is None
    assert W.collapse_at([63.9] * 400, lim) == 400 and W.collapse_at([64.0] * 1500, lim) is None
    assert W.collapse_at([10.0] * 300 + [200.0] * 1200, lim) is None
    ok("early stop on fixtures: fires at 400 at the earliest, restarts after a long iteration, threshold 64 strict")

    def lens(path):
        rows = [json.loads(x) for x in open(os.path.join(AIR, path, "metrics.jsonl"))]
        return [r.get("Train/mean_episode_length") for r in rows if "Imagination/epistemic_uncertainty" in r]
    rel = W.collapse_at(lens("runs/lite/s2_REL__p0"), lim)
    assert rel is None, rel
    p6 = W.collapse_at(lens("runs/c/lite/runs/lite/s2_sim_6M_N8_s0__p0"), lim)
    p10 = W.collapse_at(lens("runs/c/lite/runs/lite/s2_sim_10k_N8_s0__p0"), lim)
    w4 = W.collapse_at(lens("runs/c/lite/runs/lite/s2_sim_6M_N8_s0__w4__p0"), lim)
    ok(f"REL__p0's recorded metrics: does not fire. First pilot, for the record: 6M fires at {p6}, 10k at {p10}, "
       f"weight 4.0 at {w4}")

    name = "_test_c2r_stop"
    run_train(name, ["--sim", "sim_10k_N8_s0", "--penalty", "1", "--iterations", "6", "--saves", "1",
                     "--test-collapse", "1,2,1.0"], "c2r_stop.log")
    rec = json.load(open(os.path.join(AIR, ".cache", "c2r_test_res", name + ".json")))
    assert rec["status"] == "collapsed" and rec["iterations_run"] < 6 and rec["saves"] == [1], rec
    assert os.path.exists(os.path.join(AIR, "runs", "c", name, f"actor_stopped_i{rec['iterations_run']}.pt"))
    assert len(lite_metrics(name)) >= rec["iterations_run"]
    ok(f"a run stopped by the rule (test thresholds) at iteration {rec['iterations_run']}: record 'collapsed', save "
       f"kept, lite metrics written")


def part_rule():
    import pilot2_rule as R
    R.self_test()
    ok("pilot2_rule on fixtures: every branch")


PARTS = {"short": part_short, "floor": part_floor, "starts": part_starts, "fallcost": part_fallcost,
         "earlystop": part_earlystop, "rule": part_rule}

if __name__ == "__main__":
    import torch
    torch.set_num_threads(2)
    for p in (sys.argv[1:] or list(PARTS)):
        PARTS[p]()
    print("all requested parts passed")

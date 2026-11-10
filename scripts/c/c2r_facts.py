"""Option C (C2R, step 3): checks of Amendment 1's reading of the lite code and the first pilot.

    ./py scripts/c/c2r_facts.py      # -> results/c/c2r_facts.json (written once)

  * how the lite log scales its reward terms: from the code (envs/base.py) and, numerically, the ratio of the logged
    mean return (Train/mean_reward) to the sum of the logged terms times the limit's duration;
  * whether the lite reward has any cost for a predicted fall (its reward terms and its step function);
  * the share of the first pilot's start windows by data source (the start rule of C2: every source), and the windows
    the Amendment 1 rule (source 0 only) leaves.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
OUT = os.path.join(AIR, "results", "c", "c2r_facts.json")
LITE = os.path.join(os.path.dirname(AIR), "robotic_world_model_lite", "scripts")
RUNS = {"6M": "runs/c/lite/runs/lite/s2_sim_6M_N8_s0__p0", "10k": "runs/c/lite/runs/lite/s2_sim_10k_N8_s0__p0",
        "6M_w4": "runs/c/lite/runs/lite/s2_sim_6M_N8_s0__w4__p0", "REL": "runs/lite/s2_REL__p0",
        "A5_p0": "runs/c/d3_lite/runs/lite/s2_A5_s0__p0", "A5_p1": "runs/c/d3_lite/runs/lite/s2_A5_s0__p1"}


def group_of(sources):
    """source id -> group, from the committed data manifest."""
    return {int(s["id"]): s["group"] for s in sources}


def grep(path, needle):
    return [i + 1 for i, line in enumerate(open(path)) if needle in line]


def main():
    if os.path.exists(OUT):
        print(f"{OUT} exists, kept")
        return
    import train_policy as W
    man = json.load(open(os.path.join(AIR, "optc", "data_manifest.json")))
    groups = group_of(man["sources"])
    base = os.path.join(LITE, "envs", "base.py")
    flat = os.path.join(LITE, "envs", "anymal_d_flat.py")
    cfgf = os.path.join(LITE, "configs", "anymal_d_flat_cfg.py")
    res = {"what": "C2R step 3: Amendment 1's reading checked against the lite code, the lite metrics and the data"}

    # 1. scaling
    ratios = {}
    for k, d in RUNS.items():
        rows = [json.loads(x) for x in open(os.path.join(AIR, d, "metrics.jsonl"))][1:]
        out = {}
        for n in (100, 500, 1000, len(rows)):
            if n > len(rows):
                continue
            r = rows[n - 1]
            terms = sum(v for t, v in r.items() if t.startswith("Imagination/") and t not in
                        ("Imagination/epistemic_uncertainty", "Imagination/num_valid_imagination_envs"))
            mr = r.get("Train/mean_reward")
            out[str(n)] = {"mean_return": mr, "sum_terms": terms, "sum_terms_x_5.12s": terms * 256 * 0.02,
                           "mean_length": r.get("Train/mean_episode_length")}
        ratios[k] = out
    res["scaling"] = {
        "code": {
            "per_step_reward": "weight x term x step_dt (0.02 s), summed over terms; the penalty is "
                               "uncertainty_weight x disagreement x step_dt",
            "per_step_reward_lines": f"envs/base.py:{grep(base, 'rewards += self.reward_term_weights')}",
            "logged_term": "episode sum of (weight x term x step_dt), averaged over the episodes reset at the latest "
                           "reset, divided by the limit's duration (256 x 0.02 s = 5.12 s); the lite log then averages "
                           "this over every step of the iteration",
            "logged_term_line": f"envs/base.py:{grep(base, 'episodic_sum_avg / (self._max_episode_length')}",
            "meaning": "a rate per second over the 256-step limit: an episode that ends early contributes only the "
                       "steps it had, so logged terms shrink with episode length. The per-step reward of an episode "
                       "of L steps is the logged value x 5.12 / L (in reward units per 0.02 s step).",
        },
        "numeric_check": ratios}

    # 2. fall cost
    weights_line = grep(cfgf, "reward_term_weights")
    res["fall_cost"] = {
        "reward_terms": "12 terms (tracking x2, vertical speed, roll/pitch rate, torques, joint accelerations, action "
                        "rate, feet air time, thigh contacts, stand still, tilt, joint limits at weight 0) plus the "
                        "uncertainty penalty; none is a termination term",
        "config_line": f"configs/anymal_d_flat_cfg.py:{weights_line}",
        "termination": "a predicted fall (sigmoid of the fall head, rounded) only ends the episode (dones) and resets "
                       "it; no reward is attached",
        "termination_lines": f"envs/anymal_d_flat.py:{grep(flat, 'def _parse_terminations')}, envs/base.py:"
                             f"{grep(base, 'terminated = self.termination_flags')}",
        "has_fall_cost": False}

    # 3. start-state sources of the first pilot (every source) and of the Amendment 1 rule (source 0)
    st = {}
    for rung in ("6M", "10k"):
        zeros, ones = np.zeros(45), np.ones(45)
        _, _, _, first = W.build_starts(rung, zeros, ones)
        _, _, _, new = W.build_starts(rung, zeros, ones, source=0)
        by_group = {}
        for s, n in first["window_sources"].items():
            g = groups[int(s)]
            by_group[g] = by_group.get(g, 0) + n
        tot = sum(by_group.values())
        st[rung] = {"first_pilot": {"windows": first["windows"], "episodes": first["episodes_used"],
                                    "share_by_group": {g: n / tot for g, n in sorted(by_group.items())}},
                    "amendment1_source0": {k: new[k] for k in ("windows", "episodes_used", "rows_used",
                                                               "window_sources")}}
    res["start_states"] = st
    res["start_states_note"] = ("the first pilot's start states replaced the lite dataset's windows, and the lite env's "
                                "init_dataset is the same object (train.py prepare_data), so every start came from the "
                                "rung's windows; none came from the authors' recorded data")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"start_states": st, "scaling_6M": ratios["6M"]}, indent=1)[:3000])


if __name__ == "__main__":
    main()

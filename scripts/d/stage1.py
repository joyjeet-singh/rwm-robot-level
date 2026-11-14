"""Option D (D1 step 1): what Stage 1 gave, for the note and the registration.  ./py scripts/d/stage1.py

Reads results/d_record.json and the stored lite training logs; writes results/d/stage1.json (written once).
  * REL's policies' walking scores;
  * Step 2's nine simulators by the design's verdict rule (a policy fails below 0.15; a simulator fails when at least
    two of its policies failed): how many failed, and how often the two seeds disagree (where the ladder would train
    a third policy);
  * the design's stop rule (scripts/d/train_policy.stop_at) on the stored per-iteration mean imagined episode lengths
    of Step 2's 22 policies and of Option C's pilot policies: which it would have stopped, and when.
Descriptive; nothing is trained or measured.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import train_policy as D  # noqa: E402  (the pure parts only)

FAIL_SCORE = 0.15
STEP2_NINE = ("REL", "REL__fp16", "REL__int8w", "REL__u10", "REL__u20", "REL__u30", "REL__u50", "REL__s192",
              "REL__s128")
OPTC_LOGS = {
    "sim_6M_N8_s0__p0": "runs/c/lite/runs/lite/s2_sim_6M_N8_s0__p0/metrics.jsonl",
    "sim_10k_N8_s0__p0": "runs/c/lite/runs/lite/s2_sim_10k_N8_s0__p0/metrics.jsonl",
    "sim_6M_N8_s0__w2__p0": "runs/c/lite/runs/lite/s2_sim_6M_N8_s0__w2__p0/metrics.jsonl",
    "sim_6M_N8_s0__w4__p0": "runs/c/lite/runs/lite/s2_sim_6M_N8_s0__w4__p0/metrics.jsonl",
    "A5_s0__p0": "runs/c/d3_lite/runs/lite/s2_A5_s0__p0/metrics.jsonl",
    "A5_s0__p1": "runs/c/d3_lite/runs/lite/s2_A5_s0__p1/metrics.jsonl",
}
OUT = os.path.join(AIR, "results", "d", "stage1.json")


def stop_of(path):
    L = D.lengths_from_metrics(os.path.join(AIR, path))
    if L is None:
        return {"log": path, "stored": False}
    return {"log": path, "stored": True, "iterations_logged": len(L), "stop_at": D.stop_at(L),
            "missing_lengths": sum(v is None for v in L)}


def main():
    rec = json.load(open(os.path.join(AIR, "results", "d_record.json")))
    S = rec["simulators"]
    out = {"what": "Option D, D1 step 1: what Stage 1 gave (descriptive)", "fail_score": FAIL_SCORE,
           "stop_rule": D.STOP}
    out["REL"] = {p["policy"]: p["score"] for p in S["REL"]["policies"]}
    nine = {}
    for s in STEP2_NINE:
        reg = [p for p in S[s]["policies"] if p["role"] == "Step 2 registered"]
        f = [p["score"] < FAIL_SCORE for p in reg]
        nine[s] = {"scores": [p["score"] for p in reg], "failed": f, "verdict": "failed" if sum(f) >= 2 else
                   "did not fail", "seeds_disagree": sum(f) == 1}
    out["step2_nine"] = nine
    out["step2_failed"] = sum(v["verdict"] == "failed" for v in nine.values())
    out["step2_disagree"] = sum(v["seeds_disagree"] for v in nine.values())
    stops = {}
    for s, v in S.items():
        for p in v["policies"]:
            if p.get("role", "").startswith("Step 2"):
                r = stop_of(f"runs/lite/s2_{p['policy']}/metrics.jsonl")
                r.update(role=p["role"], score=p.get("score"), record_says=p.get("stop_rule_would_fire_at"))
                stops[p["policy"]] = r
    for p, path in OPTC_LOGS.items():
        stops[p] = dict(stop_of(path), role="Option C pilot")
    out["stop_rule_on_stored_logs"] = stops
    out["stopped_step2_registered"] = sorted(p for p, r in stops.items()
                                             if r["role"] == "Step 2 registered" and r.get("stop_at"))
    out["stopped_step2_pilot"] = sorted(p for p, r in stops.items() if r["role"] == "Step 2 pilot" and r.get("stop_at"))
    out["stopped_optc"] = sorted(p for p, r in stops.items() if r["role"] == "Option C pilot" and r.get("stop_at"))
    out["agrees_with_record"] = all(r["stop_at"] == r["record_says"] for r in stops.values()
                                    if "record_says" in r and r["stored"])
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists; not overwriting")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("REL", "step2_failed", "step2_disagree", "stopped_step2_registered",
                                          "stopped_step2_pilot", "stopped_optc", "agrees_with_record")}))
    for p, r in stops.items():
        if r.get("stop_at") or not r["stored"]:
            print(p, r.get("stop_at"), r.get("score"), r["stored"], r.get("iterations_logged"))


if __name__ == "__main__":
    main()

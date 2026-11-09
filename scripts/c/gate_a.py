"""Option C (C2): Gate A. On the expert test set, the 6M rollout-trained seed-0 simulator's 8-step and 24-step errors must be
below those of holding the last state (OPTIONC_PLAN.md, "Gates and the pilot").

    ./py scripts/c/gate_a.py        # reads results/c/pilot_sims.json, writes results/c/gate_a.json once; exit 1 on failure
"""
import json
import os
import sys

AIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(AIR, "results", "c", "pilot_sims.json")
OUT = os.path.join(AIR, "results", "c", "gate_a.json")


def main():
    e = json.load(open(SRC))["sims"]["sim_6M_N8_s0"]["sets"]["expert"]
    rows = {h: {"error": e[h], "hold_last_state": e["floor_" + h], "below": e[h] < e["floor_" + h]} for h in ("E8", "E24")}
    passed = all(r["below"] for r in rows.values())
    rec = {"what": "Gate A: sim_6M_N8_s0 on the expert test set, against holding the last state", "source": "results/c/pilot_sims.json",
           "windows": e["windows"], "episodes": e["episodes"], "horizons": rows, "passed": passed}
    if os.path.exists(OUT):
        if json.load(open(OUT)) != rec:
            raise SystemExit(f"{OUT} exists and differs")
    else:
        with open(OUT, "w") as fh:
            json.dump(rec, fh, indent=1)
    print(json.dumps(rec))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

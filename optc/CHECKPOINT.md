# Option C checkpoint

**Last session:** C2R, COMPLETE (2026-11-11 00:13).

**Next action:** the launcher runs the second pilot's job `optc/jobs/pilot.zsh`, then starts C3. Sessions must not start the job.

**Blockers:** none yet.
- Gate A2 failed once (`results/c/calibration.json`), so the job first retrains `sim_6M_N8_s0` and `sim_10k_N8_s0` with 2 × S = 5,144 steps (as `…__2S`), then checks and recalibrates them.
- If Gate A2 fails again, the job exits 3, BLOCKED (`pilot2_rule.py a2` writes DECISIONS, LOG and `results/c_pilot.*`).

**State**
- Calibration: λ = 0.30 (6M), 0.23 (10k), 0.25 (`A5_s0`), from ū = 0.46 in the released model. Closed loop; the fallback was not used.
- `scripts/c/train_policy.py` has the Amendment 1 changes:
  - `--penalty cal|<w>`, `--fall-cost`, `--max-iters`, saves every 500;
  - the noise floor, source-0 starts and the early stop.
- `scripts/c/pilot2_rule.py` implements Gate A2's branch, Gate B', the iteration rule' and Gate C'. `scripts/c/test_c2r.py` passes in full.
- `results/c_pilot1.*`: the first pilot's record. The job adds the surviving saves' scores (`pilot1_record.py --measured`).

**For the job** (estimates, not measured): about 16 h to retrain the simulators, 13 h for the four arms, 4 h for the 10k policy, plus measuring; 35–40 h in all.

**For later sessions**
- **C3:**
  - `results/c_pilot.txt` holds the verdicts. If Gate A2 passed only after 2 × S, S becomes 2 × S for the whole pool.
  - If Gate C' is not passed, add the 2k rung and record it before registering.
  - Declare the first pilot, Amendment 1 and the second pilot as known.
  - If the chosen arm has a fall cost, the paper must say that it departs from the authors' reward.
- **C3:** `scripts/c/evaluate.py`'s pilot list names the second pilot's policies only. Main-pool names pass once `prereg/PREREG_C.md` is committed.
- **C3:** the 10k rung has only 3 source-0 episodes (2,907 start windows; DECISIONS C2R 12).
- **Descriptive:** in imagination, the 6M simulator charges the authors' walk 0.089 per step for vertical speed, against 0.00007 in MuJoCo. This fits its kinematic residual of 9.1.
- The RWM-U and RWM quotes in `optc/notes/C0_facts.md` are unconfirmed; check them against the papers before citing.

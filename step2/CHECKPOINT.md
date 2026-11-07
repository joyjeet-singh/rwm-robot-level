# Checkpoint
Last session: M2, COMPLETE (2026-11-07).
Next: the launcher runs job calib (step2/jobs/calib.zsh). It writes results/s2/calib/, step2/model/calibration.json and anymal_d_cal.xml (frozen, read-only), results/s2/fid_*.json, results/s2_fidelity.json and results/s2_fidelity.txt. Then M3 applies Gate 1 to `gait_authors_policy.calibrated.summary` against `gait_recorded.summary` (step2/notes/M2.md).
Blockers: none. Later sessions must load step2/model/anymal_d_cal.xml, not MjEnv(constants=...) (DECISIONS.md M2).

# Step 2 log
## M0 status: COMPLETE (2026-11-07 23:50)
- Tag air2027-step2-start on air2027-improve's commit; branch step2; plan committed.
- Facts in step2/notes/M0_facts.md; ANYdrive URL and a few items UNVERIFIED (see DECISIONS.md).
- step2/jobs/fetch.zsh written, passes zsh -n; not run (launcher runs it).
## M1 status: COMPLETE (2026-11-07 23:39)
- step2/model/anymal_d.xml built from the URDF (57.03 kg; box trunk, capsule legs, sphere feet; dt 0.005; 12 motors ±80 N·m).
- ANYdrive missing: PD fallback fitted on training eps: Kp 82.6, Kd 4.93, delay 4 physics steps, R² 0.93 (results/s2/pd_fit.json).
- scripts/s2/{build_model,actuator,mj_env,test_mj}.py; tests a–f all pass (results/s2/m1_tests.json).
- MuJoCo 3.10 fusestatic URDF import shifts fused COMs (0.43 kg·m²); model built without it.
- Probe: 1.7 s per 20-s episode with the authors' policy (2 threads).
- For M2: contact time constant 0.005 is raised to 0.01 by MuJoCo's refsafe (DECISIONS.md M1).
## M2 status: COMPLETE (2026-11-07 23:50)
- scripts/s2/fidelity.py (replay E1/E8/E24/E100 + torque error; gait of the authors' policy; recorded reference), scripts/s2/calibrate.py (108-point grid, training arena, resumable, freeze), scripts/s2/test_fid.py.
- Tests a–e pass (results/s2/m2_tests.json); windows equal evalh's (floor to 1.3e-7); held-out refused.
- Grid points are compiled from model text: run-time armature leaves MuJoCo's derived quantities stale (DECISIONS.md M2).
- step2/jobs/calib.zsh written, passes zsh -n; a dry run of all parts on redirected paths passed. Not run (launcher runs it). Estimated 15–25 min.
## M3 status: COMPLETE (2026-11-08 00:25)
- Gate 1 passed (calibrated model, 80 episodes): falls 1.25% (<10%); two feet 0.789 vs 0.844 recorded (gap 5.5 pts); speed error 0.069 vs 0.072 m/s (ratio 0.95).
- scripts/s2/train_policy.py (wraps run_lite; applies compression variants itself, as run_lite ignores them), evaluate.py (M2 protocol, bootstrap 95% intervals, pre-registration guard), pilot_report.py.
- Tests a–e pass (scripts/s2/test_m3.py, results/s2/m3_tests.json, ~35 s).
- step2/jobs/pilot.zsh written, zsh -n passes, report dry-run on test outputs. Not run (launcher runs it). Estimated 4–5 h.
- Committed the calib job's outputs (fidelity results, calibration, frozen model) with this session.
## M4 status: COMPLETE (2026-11-08 04:33)
- Gate 2: P1 passed (REL 10% falls), P2 failed (A_s0, B_s0, A_s0__u50 fell in all episodes), P3 passed -> fallback pool registered: REL + 8 compressed copies, seeds 1 and 2 (18 policies).
- Copies' checks computed before registration with registered functions (results/s2/checks.json; in-sample, all 20 trajectories).
- prereg/PREREG_S2.md + scripts/s2_stats.py registered (b51a8e5), hash results/prereg_hash_s2.json (2afa114), chmod a-w.
- air-stats-reviewer: 1 MUST + 6 SHOULD FIX, all fixed before registration.
- scripts/s2/test_s2.py all pass; scripts/s2_main.py resumable; train_policy.py gains --route rebuilt and REL__<variant>.
- step2/jobs/main.zsh: 2 workers x 2 threads (probe: side by side ~15% faster); about 20 h. Not run (launcher runs it).

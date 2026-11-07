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

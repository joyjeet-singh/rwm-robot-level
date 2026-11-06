# Step 1 log

## S0 status: COMPLETE (2026-11-06 14:00)
Tag air2027-step1-start at 6887672; branch step1; plan committed b96ab1f. Env matches R0 (py3.11.15, torch 2.2.2, numpy 1.26.4, scipy 1.13.1, tensordict 0.3.2; pypdf ok); disk 35 GiB free.
Notes: step1/notes/S0_inventory.md, S0_known.md. Key finding: ICPAIAS already computed rJ/P_on/E368 for 24 baselines (exploratory) — PREREG_S1 must declare it.
Lite: train.py needs wandb stub, rsl_rl via sys.path, CUDA→CPU (torch.load without map_location), and blocks waiting for assets/data CSVs.
Open: baselines_eval.json contents only skimmed.

## S1 status: COMPLETE (2026-11-06 15:10)
Added scripts/s1_policy.py, s1_rollout.py (score_gru), test_s1.py, s1_probe.py, and results/s1_probe.json; commit e52aa65.
All 13 tests in test_s1.py pass in 15 s. task1c R² was reproduced exactly. The linear exponents are within 4e-4 of the true values. A_s0 and B_s0 match the stored λ1 and r̄J within 3.2e-5 relative. The replay closed loop reproduces the open-loop rollout exactly.
Probe: 8.0 s per one-head model and 11.3 s per five-head model. Projected 0.22 h for F18 and 0.58 h for X5, not counting weight loading.
Choices are in DECISIONS S1-1…7, which S3 must register: p=1, no observation noise, the noise seed, the λ1_cl start directions, the imagined-return scope (with the authors' dof_acc quirk), divergence, and the probe scope.
No trained simulator was driven by the policy or by noise.

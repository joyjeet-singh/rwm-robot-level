# Step 1 log

## S0 status: COMPLETE (2026-11-06 14:00)
Tag air2027-step1-start at 6887672; branch step1; plan committed b96ab1f. Env matches R0 (py3.11.15, torch 2.2.2, numpy 1.26.4, scipy 1.13.1, tensordict 0.3.2; pypdf ok); disk 35 GiB free.
Notes: step1/notes/S0_inventory.md, S0_known.md. Key finding: ICPAIAS already computed rJ/P_on/E368 for 24 baselines (exploratory) — PREREG_S1 must declare it.
Lite: train.py needs wandb stub, rsl_rl via sys.path, CUDA→CPU (torch.load without map_location), and blocks waiting for assets/data CSVs.
Open: baselines_eval.json contents only skimmed.

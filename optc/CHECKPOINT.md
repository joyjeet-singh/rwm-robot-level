# Option C checkpoint

**Last session:** C1, COMPLETE.

**Next action:** the launcher runs the job `optc/jobs/data.zsh`, then starts C2. Sessions must not start the job.

**Blockers:** none.

**State**
- Root cause: `optc/notes/C1_rootcause.md`. D3: the penalty alone did not fix it (seed 0 did not fall but did not walk; seed 1 fell every time).
- Code: `scripts/c/gen_data.py`, `train_sim.py`, `sim_checks.py`, `probe_c1.py`, `test_c1.py`. All tests passed: generator equals Step 2's evaluator (difference 0), windows and fall rule, first 20 losses equal the rebuild's (difference 0), resume to identical weights, checks on both test sets.
- Tested on small data only. The full 6M generation, the rung files and the 2,572-step trainings have not run.
- `results/c/probe.json`: S = 2,572; one training at a time, 4 threads (`optc/DECISIONS.md`, C1 item 14). 6M transitions should take about 0.7 h to generate.
- The job writes `optc/data_manifest.json`, `optc/data/` (shards, `rung_*.npy`, test sets), `runs/c/sim_6M_N8_s0` and `runs/c/sim_10k_N8_s0`, and `results/c/pilot_sims.json`.

**For later sessions**
- **C2 must commit `optc/data_manifest.json`** after the job; it did not exist when C1 committed.
- Batch size is 256 (the rebuild's), not the 1,024 in `optc/notes/C0_facts.md` (`optc/DECISIONS.md`, C1 item 1).
- Simulators are named `sim_<rung>_N<8|1>_s<seed>`; `final.pt` holds `state_mean` and `state_std`, the rung's own normalisation. Step 2's route (`common.load`, the lite adapter) assumes the config's normalisation, so C2's policy route must use these values.
- D6's training data is rung `d6` (`train_sim.py --rung d6`).
- `results/c_diag.json`, `results/c_diag.txt` and the rest of `results/c/` are not committed; C1 staged only `results/c/probe.json`.
- The tools refuse `mkdir` under `results/`; create result folders from scripts with `os.makedirs`.
- The RWM-U and RWM quotes in `optc/notes/C0_facts.md` are unconfirmed; check them against the papers before citing.

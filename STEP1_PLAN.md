# AIR 2027 — Step 1: closed-loop and generality studies on the CPU (unattended plan)

Step 1 adds what can be done for free, on this Mac, with assets the PDM folder already holds:

- **Study CL** (pre-registered). Do the label-free checks still rank simulators by long-horizon error when a policy driving the simulator supplies the joint commands (closed loop), instead of the recording? The policy is the authors' own, stored in the released checkpoint.
- **Study AR** (pre-registered). Do the registered comparisons of checks hold for simulators of other designs? The pool is the 26 MLP, RSSM and Transformer models trained in the earlier reproduction, plus 12 trained here. The new ones give each design five seeds of both training methods (one step ahead, and on its own rollouts), as the paper's main models have.
- **Study TB** (descriptive). Do the conclusions depend on training length? This uses the 10,000-iteration models and the training-loss records, which already exist.
- **Task LP** (engineering). Get the authors' lite code training a walking policy inside our simulators on this CPU, and time it. This prepares Step 2, the physics-simulator test.

Results go into the paper's appendix as new sections A8–A10. The body changes only where a sentence becomes out of date, and then without adding length. Nothing is submitted, uploaded, pushed or e-mailed. Everything runs on the CPU, at no cost.

Two terms are used throughout:
- **joint commands** (the actions): the 12 joint position targets given to the legs at every step;
- **velocity command**: the walking speed and turning rate the policy is asked to follow.

**For the author — how to run it.** In a terminal (the app's terminal or Terminal.app):

```
cd ~/Downloads/PDM/"AIR 2027"
chmod +x go_s1.zsh
caffeinate -i ./go_s1.zsh auto     # every session and the background job; safe to re-run after a stop
./go_s1.zsh status                 # progress, from another terminal
```

Expect about a day in all (roughly 20–26 hours). Most of it is a background job of about 13–15 hours:
- scoring every simulator: about 3–4 hours;
- training the 12 new models: about 10 hours, going by the earlier reproduction's own training times on this Mac.

The job has no time limit. The launcher stops it only if it makes no progress for an hour.

If a session stops at Claude's usage limit, the launcher waits until the reset time the message gives, and then resumes; this doesn't use up the session's retries. It stops only if it can't read the reset time, for example on a weekly limit.

Keep the Mac plugged in with the lid open, and don't change anything under `~/Downloads/PDM` while it runs.

If the run stops:
- **BLOCKED:** read `step1/DECISIONS.md` and write your answer there. Then run `./go_s1.zsh S<n> resume`, and `auto` again.
- **Job S3 failed or stalled:** read the end of `.cache/logs/job_S3.log`, fix the cause if there is one, then run `auto` again. The job resumes where it stopped.

When it finishes, start with `step1/REPORT.md`.

**How work is split.** The launcher starts one headless Claude Code session per block. Each session receives only the common rules and its own block.

| Session | Model | Effort | Max turns | Work |
|---|---|---|---|---|
| S0 | Sonnet 5.5 | medium | 60 | branch, plan committed, environment, inventory of every asset used |
| S1 | Opus 5.5 | high | 150 | closed-loop measurement code and tests (no trained simulator driven by the policy) |
| S2 | Opus 5.5 | high | 150 | adapters and training code for the MLP, RSSM and Transformer simulators, and tests |
| S3 | Opus 5.5 | high | 150 | pre-register studies CL and AR; job driver; analysis code; review; job |
| job S3 | — | — | — | training 12 new models and scoring every simulator, in the background (13–15 h; no time limit) |
| S4 | Sonnet 5.5 | high | 120 | the authors' lite policy training running on this CPU, and timings |
| S5 | Sonnet 5.5 | medium | 60 | training length (descriptive) |
| S6 | Opus 5.5 | high | 120 | registered analysis; results into the appendix and the paper (waits for the job) |
| S7 | Sonnet 5.5 | high | 80 | claims, references, build checks, supplement |
| S8 | Opus 5.5 | high | 100 | fresh-eyes review of the new material; report |

Opus does the work where a mistake would cost the most:
- the measurement and training code;
- the registration;
- the registered analysis and the paper text;
- the final review.

Sonnet does the inventory, the engineering port, the descriptive analysis and the checks.

The subagents are the ones installed in `.claude/agents/` during revision round 1. Their models are pinned in their files:

| Subagent | Model | Used by |
|---|---|---|
| air-reader | Sonnet 5.5 | S0 |
| air-stats-reviewer | Opus 5.5 | S3, S8 |
| air-claim-checker | Sonnet 5.5 | S7 |
| air-referee | Opus 5.5 | S8 |

<!-- BEGIN COMMON -->
## Common rules (every session)

### C1 What this round is
Step 1 adds three CPU-only studies and one engineering task to the AIR 2027 paper:
- **Study CL** (pre-registered): checks with a policy in the loop.
- **Study AR** (pre-registered): other simulator architectures, including 12 newly trained models.
- **Study TB** (descriptive): training length.
- **Task LP:** the authors' lite policy training running on this CPU.

Sessions S0–S8 work in order; yours is the block after these rules. Earlier sessions' outcomes are in `step1/LOG.md`, `step1/CHECKPOINT.md` and `step1/notes/`. The plan is `STEP1_PLAN.md`; its launcher is `go_s1.zsh`.

**Terms.** Keep these two distinct in code, notes and text:
- **joint commands** are the actions: the 12 joint position targets given at every step;
- the **velocity command** is the walking speed and turning rate a policy is asked to follow.

**The paper now.** Revision round 1 is complete (`revision/REPORT.md`). The two-column check is exactly six pages with no slack, so a body change must not add length.

### C2 Running unattended
- **Never ask the user anything.** When this plan does not settle a choice:
  - take the more conservative option (fewer or weaker claims, no new data);
  - record it in `step1/DECISIONS.md` under your session id;
  - go on.

  Stop BLOCKED only when going on would break a rule below or a registered design.
- **Never start `claude`, `go_s1.zsh`, `go_rev.zsh`, `caffeinate` or a job yourself.** Anything that would run longer than about 10 minutes goes into `step1/jobs/<your id>.zsh`, which the launcher runs.
- **On a resumed attempt,** first read:
  - `step1/CHECKPOINT.md`;
  - `tail -20 step1/LOG.md`;
  - your notes in `step1/notes/`;
  - `git log --oneline -15`.

  Continue from the first unfinished step; never redo a finished one.
- **Before ending,** with any status:
  - commit (C7);
  - append your status line to `step1/LOG.md`;
  - overwrite `step1/CHECKPOINT.md` (C9).

### C3 The boundary
- **Where you may write.** Read anything under `~/Downloads/PDM`, except:
  - never open, copy or list the contents of key files (`../rwm_repro/sshkey*`);
  - don't read `../rwm_private/`.

  Create, change or delete files only inside this folder (`AIR 2027`). The launcher compares the rest of `~/Downloads/PDM` before and after every session and job, and stops the run if anything changed.
- **The sibling folders are read-only,** including `../rwm_repro`, `../robotic_world_model_lite` and `../rsl_rl_rwm`.
  - Never run their scripts or a `main()`.
  - Use their code through `sys.path`, or copy a file into `scripts/` with a header line giving its source path and sha256.
  - Code you import must not write into its own folder (logs, checkpoints, caches): point every such path into this folder.
- **Never edited, copied over, touched or re-permissioned:**
  - `prereg/PREREG.md`;
  - `prereg/PREREG_FT.md`;
  - `scripts/ft_stats.py`;
  - once S3 commits them, `prereg/PREREG_S1.md` and `scripts/s1_stats.py`.
- **No installs.** Everything needed is in `./py`:
  - torch 2.2.2 and tensordict 0.3.2;
  - numpy 1.26.4, scipy 1.13.1 and pypdf, the last two in `.pylib`.

  Never `pip install` anything; an earlier numpy install had to be undone. W&B is replaced by a stub module in `scripts/lite/stubs/`.
- **No sudo. No external actions:** never push, upload, e-mail, submit or post.

### C4 Integrity
- **No result is typed into `paper/*.template.tex`.**
  - Every value comes from a result file through `scripts/paper_numbers.py`, with `air_numbers.py`, `revision_numbers.py` and, new, `s1_numbers.py`.
  - `build_paper.py` refuses to build when a `{{key}}` has no value.
- **Registered verdicts, old and new, stand as registered.** Study TB, and any analysis not in `prereg/PREREG_S1.md`, are descriptive or exploratory. Label them so everywhere: text, tables, captions, appendix.
- **Registration before data.** Until S3 commits `prereg/PREREG_S1.md`:
  - no trained simulator may be driven by the policy's joint commands or by noisy joint commands;
  - no trained MLP, RSSM or Transformer simulator may have an error, a growth rate or a kinematic residual computed. Only test (vi) may compare their one-step predictions with the reproduction's own code;
  - no new MLP, RSSM or Transformer model may be trained beyond test (ix)'s 50 iterations.

  Tests use untrained (random-weight) models, constructed systems, and values that already exist.
- **After the registration:**
  - Problems found go to `ANOMALIES.md`: date, what, effect, what was done.
  - Result files are written only by scripts, never by hand.
- **Resampling:** resample whole trajectories or whole models, as the registration says; never seed × trajectory.
- **Existing results:**
  - The stored results were computed in a Linux cloud workspace.
  - Never overwrite an existing result file. New values go into new files: `results/s1_*`, `results/raw_S1.npz` and `results/s1/`.
- **Never edit a shell script while it runs.**

### C5 Environment
- **Python:** always `./py`. It runs the PDM interpreter `../.venv-rwm311` (Python 3.11), and sets `PDM_ROOT`, `PYTHONPATH`, no bytecode, and LaTeX on `PATH`.
- **Machine:** Intel i5, 4 logical cores, 8 GB memory. Call `torch.set_num_threads(2)` in any computation.
- **While job S3 runs** (sessions S4 and S5), keep your own computations under about 10 minutes, with at most 2 threads.
- **No poppler:** PDF text comes from pypdf through `scripts/pdf_tools.py`.
- **Long output:** send it to a log in `.cache/logs/`, and print only its tail.
- **The job's progress file**, `results/s1/progress.json`, belongs to the job alone. Never write it: the launcher watches it to tell a slow job from a stalled one.

### C6 Context and token discipline
- **How to read:**
  - read only what your block names;
  - grep first, and read at most about 200 lines at a time;
  - print JSON fields with `./py -c`, never whole files or arrays;
  - print at most about 40 lines from any command.
- **Never read these whole:**
  - `REVISION_PLAN.md` and `revision/review/*`;
  - `results/paper_numbers.json` and `results/paper_numbers.md`;
  - any `results/raw_*.npz`;
  - `history.bundle`;
  - logs (use `tail`);
  - any JSON over 100 KB (most `../rwm_repro/results/*.json` are);
  - `../rwm_repro/PAPER*.md` and `../rwm_repro/FINDINGS_LEDGER.md`.
- **Subagents:**
  - Use only those your block names.
  - Never pass a `model` parameter on an Agent call.
  - Give each a self-contained brief: exact files, the question, and what to return. Ask for no more than you need.
- **Don't re-read** a file you have already read in this session unless it changed.

### C7 Git
- Work on branch `step1`, which S0 creates.
- Commit at the end of your session and wherever your block says, with the subject "S<n>: <what>".
- **Staging:** stage only the paths your session changed, with `git add <paths>`. Never `git add -A`: job S3 may be writing.
- **Never** push, checkout, restore, reset --hard, stash, clean or rebase, or rewrite history in any other way.
- **Never commit:** `runs/`, `.cache/`, `.pylib/`, `paper/build/`, `submission/` or `history.bundle`.
- **Tags:** only `air2027-step1-start` (S0) and `air2027-step1` (S8).

### C8 Style
- Readers are roboticists. Use plain language, and define every unavoidable term at first use.
- One idea per sentence, with at most two numbers-with-intervals in a sentence.
- State failures as plainly as successes. Never claim more than the numbers show.
- Name the scope wherever a claim could be read more widely: one robot, one trotting gait, simulated data, no physics-simulator test of any policy.
- Notes in `step1/notes/` are at most 300 words unless your block says otherwise.

### C9 Ending a session
- **Status line.** Append to `step1/LOG.md` the line `## S<n> status: COMPLETE (YYYY-MM-DD HH:MM)`, with PARTIAL or BLOCKED in place of COMPLETE where needed.
  - Then add at most 8 lines: what was done, files, commit hashes, open points.
  - The launcher reads this line, so write it exactly in that form.
- **Checkpoint.** Overwrite `step1/CHECKPOINT.md` with the last session and its status, the next action, and any blockers.
- **BLOCKED:** add to `step1/DECISIONS.md` what is blocked, the options, and your recommendation.
- **PARTIAL:** commit what is done, and write the exact next step in `step1/CHECKPOINT.md`.
<!-- END COMMON -->

<!-- BEGIN S0 -->
## S0 — Branch, plan, environment and inventory (Sonnet 5.5, medium)

**Goal:**
- a branch for Step 1;
- the plan committed before any Step 1 data exist;
- a confirmed environment;
- an inventory of every asset Step 1 uses, including everything that already exists which the registration must declare.

**Read:** `revision/CHECKPOINT.md` and `revision/notes/R0_env.md`. List folders with `ls`.

**Steps**
1. **Git.**
   - Run `git status --porcelain | head -20` and `git log --oneline -3`. Known differences are `figures/fig_horizon.*` and untracked files; leave them.
   - Tag the current commit `air2027-step1-start`.
   - Run `git switch -c step1`; on a resumed attempt, `git switch step1`. Record the hash.
   - Commit `STEP1_PLAN.md` and `go_s1.zsh` alone, as "S0: Step 1 plan (written before any Step 1 data)".
   - If missing, create in `step1/`:
     - `LOG.md` and `DECISIONS.md`, each with a title line;
     - `CHECKPOINT.md`;
     - the folders `notes/`, `jobs/` and `review/`.
2. **Environment.**
   - Print the versions of Python, torch, numpy, scipy and tensordict through `./py`; they should match R0.
   - Run `./py -c "import pypdf, tensordict"`.
   - Record free disk space.
3. **Inventory.** Use air-reader for anything outside this folder: one brief per item, each reply at most 300 words, with paths, line numbers and keys. Inside this folder, grep and use `./py -c` yourself.
   - **a. The authors' policy** (`../robotic_world_model_lite/assets/models/pretrain_rnn_ens.pt`):
     - the keys holding the actor, any observation normaliser, and the dynamics model;
     - how `../rwm_repro/scripts/task1c_policy_test.py` loads the actor and builds observations and velocity commands (function names, lines);
     - task1c's result file, and the keys holding its mean R² at offsets −1 and 0;
     - which offset fits better, that is, which recorded row the actor's output corresponds to.
   - **b. Velocity commands:** the Step 0 regime file in `../rwm_repro/results/`, written by `../rwm_repro/scripts/step0_velocity_regimes.py`. Record its fields, and the segment boundaries and per-segment commands for all ten episodes.
   - **c. Imagination observation and reward:**
     - in `../robotic_world_model_lite/scripts/envs/anymal_d_flat.py`, the lines of `get_imagination_observation` and `_compute_imagination_reward_terms`;
     - the reward-weight field in `configs/anymal_d_flat_cfg.py`, and how the env maps the weights to terms;
     - which model outputs give the feet contacts.
   - **d. Other architectures:**
     - **Runs.** Every folder `../rwm_repro/runs/baseline_*` and its final weights file.
       - 26 are expected: MLP 9, RSSM 8, Transformer 9.
       - `baselines_probe` is a timing probe, not a model.
       - Also search the other PDM folders for baseline runs.
     - **Per run,** from `../rwm_repro/results/baseline_run_*.json` (print keys first; the curves are long):
       - regime (`tf`: trained one step ahead; `ar`: trained on its own rollouts), spec, seed and iterations;
       - parameter count, wall-clock time, and torch threads if recorded.
     - **Specs:** what "s7", "matched", "x1v1" and "x1v2" mean, from the reproduction's notes (for example `BASELINE_SPECS.md`).
     - **Classes** in `../rwm_repro/src/baselines/` (`common.py`, `mlp.py`, `rssm.py`, `transformer.py`):
       - constructor arguments, and the `predict` and `rollout` signatures;
       - the joint-command pairing: confirm that `rollout` takes `action_offset=1`, as `scripts/lyap.py` does;
       - the normalisation used.
     - **Training script,** `../rwm_repro/scripts/train_baseline.py`:
       - confirm it trains on the seed-0 split (episodes 1 and 8 held out);
       - confirm it computes no error on held-out data;
       - confirm it guards `main()`, and writes only to `--out-dir` when given one;
       - list the functions a copy would reuse (`build_everything`, `loss_weights`, `train_iterations`).
   - **e. What already exists:**
     - Search `../rwm_repro/results`, `../NGCAPS/results`, `../ICPAIAS/results`, `../PerconAI/results` and this folder's `results/`:
       - for any growth rate (lyap, lam) or kinematic residual (rJ) computed for a baseline model;
       - for any rollout driven by a policy or by noisy commands (search "closed", "policy", "noise", "I4").
     - `../rwm_repro/results/action_sensitivity*` is known; record its intervention list and headline readings only.
     - Record "none found", or the exact files and keys.
   - **f. Training length:**
     - the summary keys of `../rwm_repro/results/training_tail_slopes*`;
     - whether the 10,000-iteration models (`../rwm_repro/runs/*_10k*`) appear in `results/model_table.json`;
     - their 2,500-iteration counterparts (same arm and seed), and their held-out episodes.
   - **g. The lite code** (`../robotic_world_model_lite/scripts/train.py` and what it calls):
     - its flags, and how it loads a pretrained dynamics model and trains only the policy;
     - the device and `num_envs` fields;
     - where it writes logs, checkpoints and videos;
     - every import missing from `./py` (wandb, others);
     - the interface `../rsl_rl_rwm/rsl_rl/modules/system_dynamics.py` gives the env (`forward`, `reset`, `reset_partial`, `ensemble_size`).
   - **h. Stored values in this folder:**
     - the fields of `results/raw_F18.npz` and `results/raw_X5.npz` (`lam`, `rJ`, `E1`, `E8`, `E24`, `E100`, …) and their shapes;
     - how rows map to models (`results/models.json`, `results/model_table.json`).

     Print shapes and keys only.
4. **Write the notes.**
   - `step1/notes/S0_inventory.md`: a table with the columns item, path, format and keys, and notes; at most 150 lines.
   - `step1/notes/S0_known.md` (at most 300 words): what already exists that the registration must declare.
5. Commit "S0: branch, environment and inventory".
<!-- END S0 -->

<!-- BEGIN S1 -->
## S1 — Closed-loop measurement code and tests (Opus 5.5, high)

**Goal:** code that computes, for any GRU simulator in the pools:
- the two label-free checks, the growth rate λ1 and the kinematic residual r̄J, under three sources of joint commands:
  - recorded;
  - from the authors' policy in closed loop;
  - recorded plus noise;
- the closed-loop growth rate;
- the imagined return.

Tests must pass. No trained simulator is driven by the policy or by noise (C4).

**Read:**
- `step1/notes/S0_inventory.md` (items a–c and h) and `S0_known.md`.
- In `scripts/`:
  - `lyap.py` (`rollout_map`, `first_step`, `step`, `lyapunov`);
  - `compute.py` (`build`, `per_model`, `setup_arena`);
  - `common.py` (`load`, `arena`, `denorm`, `monitors`, normalisation);
  - `evalh.py` and `pools.py`.
- `../rwm_repro/scripts/task1c_policy_test.py`, and the lines S0 names in `anymal_d_flat.py`.

**Definitions to implement.** All rollouts start where the registered 368-step rollout starts:
- after each trajectory's 32 true history steps;
- on each model's own arena;
- with each model's own normalisation.

**Recorded joint commands** (the registered setting).
- The analysis uses the stored values (S0 item h).
- The new code reproduces them by calling `per_model`'s own functions: `lyap.rollout_map`, `lyap.lyapunov` and `C.monitors`. Import them; never copy them.

**Timing.**
- `lyap.rollout_map` predicts row i from the state at row i−1 and the joint command `ac[:, i]` (`action_offset=1`). The closed loop must keep this pairing.
- Use task1c's alignment (S0 item a) to decide which recorded row the actor's output stands for.
- Test (iv) checks the result.

**Policy joint commands (closed loop).**
- At each forecast step after the first, the authors' actor receives an observation built exactly as `get_imagination_observation` builds it, from:
  - the simulator's current predicted state, denormalised;
  - the velocity command of the recorded segment covering that step, as task1c builds it;
  - its own previous joint command. At the first closed-loop step, this is the recorded joint command that the alignment assigns to the step before.
- The actor's mean output, with no noise and normalised like the model's inputs, is the next joint command.
- The first forecast step keeps the recorded joint command, as `lyap.first_step` does.
- **λ1_pol:** `lyap.lyapunov`, with the joint-command array after the history replaced by the closed-loop joint commands. The policy's own feedback is not differentiated, so this is the simulator's error growth on the states a policy visits.
- **r̄J_pol:** `C.monitors` on the closed-loop states, as `per_model` does for recorded commands.

**Closed-loop growth rate λ1_cl.**
- Use Benettin's method with forward-mode JVPs through the composed map: the policy, then the simulator step.
- The map's state includes the policy's previous joint command.
- Use float64, with the same start directions, seed and length as `lyap.lyapunov`.

**Noisy recorded joint commands.**
- After the history, add Gaussian noise to the recorded joint commands.
- Draw it once per trajectory, in raw units, with `numpy.random.default_rng(trajectory_index)`.
- The per-dimension std is 0.1 × the std of the recorded joint commands over all 10,000 rows.
- Every model sees the same noisy sequence.
- Compute λ1_noise and r̄J_noise as above.

**Divergence** (new conditions only).
- A rollout diverges at the first step where any normalised state component is non-finite, or exceeds 1e3 in absolute value.
- Its checks are then +inf, so they rank worst (`stats._clean`). Record the step.

**Imagined return.**
- Use the per-step reward of `_compute_imagination_reward_terms`, with the configured weights.
- Sum it over the closed-loop forecast steps, with contacts from the model's contact outputs (S0 item c).
- Average over trajectories.
- A diverged rollout gets the sum up to divergence, and a flag.

**No errors.** Errors such as E100 are not defined under policy or noisy joint commands, because there is no recording to compare against.

**Files**
- `scripts/s1_policy.py`: actor loading, observation, velocity command.
- `scripts/s1_rollout.py`:
  - closed loop, noise and divergence;
  - λ1_pol, λ1_noise, λ1_cl, r̄J_pol and r̄J_noise;
  - imagined return;
  - the recomputed recorded-command values.

  Its single entry point, `score_gru(model, arena, scales) -> dict`, returns per-trajectory values.
- `scripts/test_s1.py`.

**Tests.** All must pass in at most 10 minutes, saving nothing outside `.cache/`.
- **(i)** `s1_policy` reproduces task1c's mean R² at offsets −1 and 0 to 1e-4. Read the reference values from task1c's result file; don't type them.
- **(ii)** Linear systems with known exponents:
  - the λ1 code on a constructed linear map gives log|largest eigenvalue|/dt to 1e-3;
  - the closed-loop code with a linear policy u = Kx gives the same for A + BK.
- **(iii)** For two F18 models, one rollout-trained and one teacher-forced, the new code's recorded-command λ1 and r̄J equal the stored values within 1e-4 relative. These values already exist.
- **(iv)** On a random-weight GRU of the release's architecture, the closed-loop code with a "policy" that returns the recorded joint commands reproduces the open-loop rollout: states equal to 1e-10, in float64.
- **(v)** The divergence rule triggers on a constructed blow-up.

Never drive a trained simulator by the policy or by noise in this session.

**Probe.** Time `score_gru`, item by item, on a random-weight GRU of the release's architecture, on F18's arena. That arena has four trajectories; X5's arenas also have four per split.

With a script, write `results/s1_probe.json`, giving:
- seconds per model for each item;
- the projected hours for F18 (97 models) and X5 (255).

**Finish.** Write `step1/notes/S1.md`, and commit "S1: closed-loop measurement code and tests (no trained simulator driven by the policy)".
<!-- END S1 -->

<!-- BEGIN S2 -->
## S2 — Adapters and training code for the other architectures (Opus 5.5, high)

**Goal:**
- the MLP, RSSM and Transformer simulators can be scored with this paper's measures, under recorded joint commands;
- the job can train new ones exactly as the earlier reproduction trained its own.

Tests must pass. No trained baseline has an error, growth rate or residual computed, and no new model is trained beyond test (ix) (C4).

**Read:**
- `step1/notes/S0_inventory.md` (item d) and `S0_known.md`.
- In `scripts/`: `evalh.py`, `lyap.py` and `common.py` (normalisation, arenas, `monitors`).
- `../rwm_repro/src/baselines/*.py` and `../rwm_repro/scripts/train_baseline.py`.

**1. Adapters** (`scripts/s1_arch.py`): one per class. Each loads a run's final weights and provides:
- (a) one-step prediction from a history window;
- (b) the rollout driven by given joint commands;
- (c) a step map on the model's own recurrent state, for JVPs:
  - MLP and Transformer: the window of past predicted states and joint commands;
  - RSSM: the deterministic latent, using means and no sampling.

Each model pairs joint commands with states as it was trained (S0 item d). If that differs from `lyap.rollout_map`'s pairing, follow the model, and say so in the notes.

The adapter computes:
- E1, E8, E24 and E100, with exactly `evalh`'s semantics: windows from every fourth position after 32 true history steps, and E1 at every position;
- E368, from the single 368-step rollout;
- λ1_rec, with the same Benettin settings as `lyap.lyapunov`: float64, the same start directions, seed and length;
- r̄J_rec, with `C.monitors`.

The single entry point is `score_arch(run_dir, arena, scales) -> dict`. Every run is scored on F18's arena (episodes 1 and 8 held out). If S0 found a run trained on another split, leave it out and say so.

**2. Training code** (`scripts/s1_train_baseline.py`):
- It is a copy of `../rwm_repro/scripts/train_baseline.py`, with a header line giving its source path and sha256.
- Change only what the job needs:
  - outputs go to `runs/s1_baselines/<run>/`: the weights and the run's JSON;
  - a one-line summary per run goes to `results/s1/baseline_training.json`: iterations, wall time, final loss, threads, and whether the weights are finite;
  - a callback lets the job write its progress every 50 iterations;
  - the number of torch threads is a parameter, 2 by default.
- Keep the reproduction's training settings: the seed-0 split, the 40-row windows, the seeding, Adam, batch 256, 2,500 iterations, and the loss.
- New runs are named `s1_baseline_<arch>_<regime>_s7_seed<s>`.

**Tests** (add to `scripts/test_s1.py`; at most 10 minutes in all):
- **(vi)** For each class, one-step predictions on 100 recorded windows equal those of `../rwm_repro`'s own baseline code to 1e-6 relative. Compare predictions only, never an error measure.
- **(vii)** The step map's JVP matches finite differences to 1e-4, on random weights.
- **(viii)** On random weights, `score_arch` returns finite values of the expected shapes.
- **(ix)** For each class, the copy trains `tf_s7` with seed 0 for 50 iterations, writing only to `.cache/`.
  - Its per-iteration training loss must equal the stored curve in `../rwm_repro/results/baseline_run_<arch>_tf_s7_seed0.json` to 1e-4 relative.
  - These curves already exist, and the loss is on training data only.
  - If it fails, check the torch thread count first, and record what you find.

**Probe.** Through a script, add to `results/s1_probe.json`:
- for each class, the time of `score_arch` on random weights, and the projected minutes for all runs (26 existing, 12 new);
- for each class, the time of 20 training iterations at 2 threads; and the time of two RSSM runs side by side, at 1 thread each;
- the projected hours to train the 12 new runs, one at a time and two at a time.

For comparison, the reproduction's recorded training times on this Mac were:
- under 0.1 s per iteration for the MLP;
- about 1 s for the Transformer;
- 2.2–2.7 s for the RSSM.

That comes to about 10 hours for the 12 new runs, one at a time.

**Finish.** Write `step1/notes/S2.md`, and commit "S2: adapters and training code for other architectures (no baseline scored)".
<!-- END S2 -->

<!-- BEGIN S3 -->
## S3 — Register studies CL and AR; prepare the job (Opus 5.5, high)

**Read:**
- `step1/notes/S0_known.md`, `S0_inventory.md`, `S1.md` and `S2.md`.
- `results/s1_probe.json`.
- `scripts/stats.py`: `spearman`, `_clean`, `boot_diff` and `boot_diff_within`. Import these; never copy them.
  - `boot_diff` is a paired model bootstrap: 20,000 draws, `default_rng(0)`, 95% and 97.5% intervals.
  - `boot_diff_within` is for X5.
- The docstrings of `scripts/s1_rollout.py` and `scripts/s1_arch.py`.
- `prereg/PREREG.md`: only Q1's and Q2's label lines (`grep -n "LABEL-FREE"`).
- `results/stats_report.txt`: the registered results, 15 lines.

**Design to register** in `prereg/PREREG_S1.md`. Write it in plain language, and fill the bracketed items.

1. **Questions.**
   - **CL:** when a policy, not the recording, supplies the joint commands, do the label-free checks still rank simulators by their 100-step error under recorded commands?
   - **AR:** do the registered comparisons of checks hold for simulators of other architectures?
2. **Simulators.**
   - **CL:** F18's 97 simulators (primary) and X5's 255 (secondary), with keys from `results/model_table.json`.
   - **AR:** every baseline run S0 listed ([n]; 26 expected), plus 12 new runs. The job trains the new runs after this registration: seeds 3 and 4 of `tf_s7` and `ar_s7`, for each of the MLP, RSSM and Transformer.
     - **Why the new runs:** with them, each architecture has five seeds of both training methods, as the paper's main models (`armA`, seeds 0–4) have. With only three seeds per method, a ranking within one architecture would rest on 8 or 9 models.
     - The `matched`, `x1v1` and `x1v2` runs are kept as they are, and not extended.
     - All runs are trained on F18's split, and scored with their final weights on F18's arena.
   - The released model is scored for reference only.
3. **Policy and commands:**
   - the authors' actor [keys], deterministic;
   - the velocity command per recorded segment [file];
   - the observation as in `anymal_d_flat.py` [lines];
   - the timing rule from S1.
4. **Measures:** as defined in S1 and S2, quoted verbatim, including the divergence rule.
   - For F18 and X5, the recorded-command values (λ1_rec, r̄J_rec, E1, E8, E100) are the stored registered values [files, fields]. The job recomputes them only for check S1-C1.
   - The target is the stored E100. For AR, it is the E100 computed by `score_arch`.
5. **Statistics:**
   - Spearman correlation with average ranks; non-finite values rank worst.
   - Intervals from `stats.boot_diff`, where both correlations come from the same draw of models.
   - On X5, `stats.boot_diff_within` over the five splits.
   - For AR, `stats.boot_diff_within` over the three architectures: correlations within each architecture, averaged, with models resampled within their own architecture.
6. **Questions and rules.**
   - **Primary family CL** (97.5% intervals, on F18):
     - **CL1:** d_r = ρ(r̄J_pol, E100) − ρ(r̄J_rec, E100).
     - **CL2:** d_λ = ρ(λ1_pol, E100) − ρ(λ1_rec, E100).
     - **Labels**, applied in this order:
       1. If the interval lies inside [−0.10, +0.10]: UNCHANGED BY THE COMMAND SOURCE. Add "small decrease" or "small increase" if it also excludes 0.
       2. Otherwise, if its upper end is below 0: WORSE WITH POLICY COMMANDS.
       3. Otherwise, if its lower end is above 0: BETTER WITH POLICY COMMANDS.
       4. Otherwise: NOT SETTLED.
     - **The margin, 0.10, is fixed by this plan.** It is smaller than the smallest registered difference that earned a label: Q4_vs_E8 (+0.115) and Q3_vs_E1 (−0.117) in `results/stats_report.txt`. Quote those lines.
   - **Primary family AR** (97.5% intervals; within architectures and averaged, as in item 5):
     - **AR1:** ρ(λ1_rec, E100) − ρ(E1, E100), which is Q1's question. Q1's labels: LABEL-FREE BETTER if the interval lies above 0, ONE-STEP BETTER if below, NOT SETTLED otherwise.
     - **AR2:** ρ(r̄J_rec, E100) − ρ(E8, E100), which is Q4's second question. Q2's labels: LABEL-FREE BETTER, TRAINING-HORIZON BETTER, NOT SETTLED.
   - **Secondary** (95% intervals, not corrected for multiplicity, reported as secondary):
     - **CL3:** rank agreement ρ(check_pol, check_rec), for each check, on F18.
     - **CL4:** CL1 and CL2 on X5, with CL1's labels.
     - **CL5:** ρ(λ1_cl, E100) − ρ(λ1_rec, E100), on F18.
     - **CL6:** ρ(imagined return, E100), on F18. A positive value means that less accurate simulators flatter the policy.
     - **CL7:** divergence counts by group (rollout-trained, trained one step ahead, compressed), on F18 and X5.
     - **CL8:** CL1 and CL2 with noisy recorded joint commands in place of the policy's, on F18.
     - **CL9:** CL1 and CL2 within F18's trained (57) and compressed (40) simulators separately.
     - **AR3:** AR1 and AR2 pooled across the three architectures (one ranking of all baseline runs), and on the extended pool (F18 plus the baselines).
     - **AR4:** for each architecture, over seeds 0–4 of `tf_s7` and `ar_s7`: is the teacher-forced run better one step ahead, and worse at 100 steps? Descriptive, paired by seed.
   - **Exploratory** (labelled so): two-level intervals for CL1 and CL2, as in `scripts/revision_analyses.py`.
7. **Checks.** The job runs them, and `s1_stats.py` reads their results.
   - **S1-C1:** the recomputed recorded-command λ1 and r̄J agree with the stored ones. This is checked for all F18 models, and for every fifth X5 model in table order.
     - For at least 95% of these models, |Δλ1| ≤ 1e-3 per second, and |Δr̄J| ≤ 1e-3 relative. Both values non-finite counts as agreement.
     - The rank correlation is at least 0.999 for each pool and check.
     - Models outside the tolerance are listed.
   - **S1-C2:** test (i) passes at the job's start.
   - **S1-C3:** test (vi) passes at the job's start.
   - **S1-C4:** every model in the pools was scored. A missing model, or a new run whose training loss became non-finite, is excluded and listed.
   - **S1-C5:** test (ix) passes at the job's start. If it fails, no new run is trained; AR uses the existing runs only, and says so.
   - **If S1-C1, S1-C2 or S1-C3 fails,** the new code does not reproduce the registered measures:
     - `s1_stats.py` gives no verdict for the affected family, and says why;
     - S6 stops BLOCKED;
     - a failed S1-C3 affects only AR.
8. **No time limit.** Everything registered runs. Record:
   - the probe's projection;
   - whether new runs train one at a time or two at a time. Two at a time, with one thread each, only if the probe shows it finishes at least 30% sooner.

   The launcher stops the job only if it makes no progress for an hour.
9. **What was known**, from `S0_known.md`:
   - the paper's values under recorded commands (Table 2, `tab:checks`);
   - task1c's result: the actor reproduces the recorded actions only partly;
   - `action_sensitivity`'s readings: the simulators respond strongly to joint commands;
   - the existing baseline runs' errors in the reproduction's own error measure, and their training-loss curves;
   - whatever else S0 found.

   State plainly that no closed-loop rollout, no noisy-command check, and no baseline growth rate or residual existed.
10. **After the commit,** neither this file nor `scripts/s1_stats.py` changes. A needed change goes to `ANOMALIES.md` with a new commit, and the paper says so.

**Steps**
1. **Write `scripts/s1_eval.py`, the job driver.** It works in four phases:
   1. **Checks:** tests (i), (vi) and (ix), recording S1-C2, S1-C3 and S1-C5.
   2. **Study CL:** scores F18, X5 and the released model (for reference only), recording S1-C1.
   3. **Training:** trains the 12 new runs with `scripts/s1_train_baseline.py`.
   4. **Study AR:** scores every baseline run, and records S1-C4.

   Requirements:
   - **Resumable.** It skips models already in `results/s1_models.json`, and runs whose final weights exist. An interrupted training run restarts from iteration 0, so it stays identical to an uninterrupted one.
   - **Progress.** It rewrites `results/s1/progress.json` after each model and every 50 training iterations, with:
     - the phase and the item;
     - counts done and left;
     - the time.

     It writes progress only from the work itself, never from a timer, so that a hung job shows as stalled.
   - **Outputs:** `results/raw_S1.npz`, and `results/s1_models.json` with software versions.
   - **Threads:** torch uses 2 threads. Training may instead run two at a time with one thread each, as item 8 records.
2. **Write `scripts/s1_stats.py`.**
   - It implements items 5–8 exactly.
   - It writes `results/s1_stats.json`, and a report of at most 60 lines in `results/s1_report.txt`.
   - It refuses to run if a check result is missing.
3. **Extend `scripts/test_s1.py`.**
   - Rule labels on constructed intervals: inside the margin; inside it and excluding 0; below; above; straddling.
   - `s1_stats` on a synthetic `s1_models.json`.
   - All tests must pass.
4. **Write `prereg/PREREG_S1.md`** from the design above.
5. **Review.** Ask air-stats-reviewer to review `prereg/PREREG_S1.md`, `scripts/s1_stats.py`, `scripts/s1_eval.py` and `scripts/test_s1.py` against this block's design.
   - Fix every MUST FIX and SHOULD FIX, and rerun the tests.
   - Record the review in `step1/DECISIONS.md`.
6. **Register.**
   1. Commit these files together:
      - `prereg/PREREG_S1.md`;
      - `scripts/s1_policy.py`, `s1_rollout.py`, `s1_arch.py`, `s1_train_baseline.py`, `s1_eval.py`, `s1_stats.py` and `test_s1.py`;
      - `results/s1_probe.json`.

      Use the subject "S3: PRE-REGISTER studies CL and AR (no trained simulator driven by the policy yet)".
   2. With a one-line `./py` command, write `results/prereg_hash_s1.json` in the format of `results/prereg_hash.json`, adding the commit hash. Commit it.
   3. Run `chmod a-w prereg/PREREG_S1.md scripts/s1_stats.py`.
7. **Write `step1/jobs/S3.zsh`,** modelled on `revision/jobs/R3.zsh`. The launcher runs it in the background, saving its output in `.cache/logs/job_S3.log`. It must:
   - `cd` to this folder, and check it is there;
   - run `./py scripts/s1_eval.py`;
   - print one line per phase, and exit non-zero on failure;
   - pass `zsh -n`.
8. Write `step1/notes/S3.md` with the registration commit's hash and time. Commit "S3: job for studies CL and AR".
<!-- END S3 -->

<!-- BEGIN S4 -->
## S4 — The authors' lite policy training on this CPU (Sonnet 5.5, high)

**Goal:** train a walking policy inside any of our simulators, on this Mac, with the authors' lite code unchanged in behaviour, and time it. This prepares Step 2; nothing goes into the paper. Job S3 is running, so keep each run under about 10 minutes, with 2 threads.

**Read:**
- `step1/notes/S0_inventory.md` (item g).
- In `../robotic_world_model_lite/scripts/`, only the parts S0 names: `train.py`, `policy_training.py`, `envs/base.py` and the configs.
- `../rsl_rl_rwm/rsl_rl/modules/system_dynamics.py`: the interface only.
- `scripts/common.py` (`load`).

**Steps**
1. **Write `scripts/lite/run_lite.py`.**
   - It imports the lite code and `rsl_rl_rwm` from their folders through `sys.path`, read-only.
   - It puts `scripts/lite/stubs/` first on the path. The stub `wandb` package provides no-op `init`, `log`, `finish`, `config`, `Table` and `run`.
   - It overrides `device="cpu"`, and takes `num_envs` and `max_iterations` as flags.
   - It sends every log, checkpoint and video to `runs/lite/<name>/`, which git ignores. Writing into the lite folder would stop the run (C3).
   - It loads either the released checkpoint, or one of our simulators through `scripts/lite/adapter.py`.
2. **Write `scripts/lite/adapter.py`.** It wraps our rebuilt simulator in the interface the lite env expects: `forward`, `reset`, `reset_partial` and `ensemble_size`.
   - **Test:** on 200 recorded windows, the adapter's next-state predictions equal our rebuild's to 1e-6 relative.
   - `reset` and `reset_partial` behave as in `SystemDynamicsEnsemble`.
3. **Keep the method unchanged.** Change no reward, observation, PPO setting or command range. Record every override in `step1/notes/S4_lite.md`.
4. **Smoke runs.**
   - (a) The released model, with 1,024 parallel robots, for 10 iterations.
   - (b) One rebuilt parent (`armA_seed0`), with the same settings.

   Both must run without error and give finite rewards. Record the seconds per iteration and the peak memory.
5. **Project for Step 2:** the hours to train one policy for 500 iterations on this Mac, at 1,024 and at 4,096 parallel robots. Recommend the CPU or a free notebook GPU accordingly.
6. **Note a Step 2 requirement.** The lite code draws imagination start states from `state_action_data_0.csv`, which includes the held-out episodes. Step 2 must restrict them to each model's training episodes. Say how to do that through the runner or the adapter, without touching the upstream code.
7. **If the lite code cannot run without an install:**
   - record exactly what is missing, and what would be installed;
   - skip steps 4–5;
   - still end COMPLETE. This task is preparation, and Step 2 will decide.
8. **Finish.**
   - Write `scripts/lite/test_lite.py`: adapter equality, plus a 2-iteration run.
   - Write `step1/notes/S4_lite.md` (at most 400 words): commands, overrides, timings, projections and open issues.
   - Commit "S4: lite policy training on this CPU".
<!-- END S4 -->

<!-- BEGIN S5 -->
## S5 — Training length (Sonnet 5.5, medium)

**Goal:** describe whether the paper's conclusions depend on training length. Use values that already exist, or are cheap to compute under recorded commands. This is descriptive, not registered. Job S3 is running, so keep each computation under about 10 minutes, with 2 threads.

**Read:**
- `step1/notes/S0_inventory.md` (items f and h).
- Through `./py -c`: the summary of `../rwm_repro/results/training_tail_slopes*`, and `results/model_table.json`.
- `results/raw_F18.npz` and `results/ft_models.json`, through code.

**Steps**
1. **Write `scripts/s1_budget.py`.** It writes `results/s1_budget.json`, and `results/s1_budget.txt` (at most 40 lines), covering:
   - **(a)** how many models were still improving at 2,500 iterations and at 10,000, from the stored slope file;
   - **(b)** for each method (rollout-trained, teacher-forced) and seed: E1, E8, E24, E100, E368, λ1 and r̄J, at 2,500 and at 10,000 iterations. If the 10,000-iteration models have no stored values, score them with `compute.build` and `compute.per_model` on their arena (about 7 s each), and say so;
   - **(c)** at 10,000 iterations, per seed and on average:
     - Is teacher forcing still better one step ahead, and worse at 100 steps?
     - Is λ1 still positive for the teacher-forced models, and negative for the rollout-trained ones?
   - **(d)** the fine-tuning study's control: how much skill 500 more training iterations added, from existing files.

   With three seeds, give no intervals. Label everything descriptive, and say which values existed before this plan.
2. Write `step1/notes/S5.md`, and commit "S5: training length (descriptive)".
<!-- END S5 -->

<!-- BEGIN S6 -->
## S6 — Registered analysis and results into the paper (Opus 5.5, high)

The launcher starts this session only after job S3 has finished.

**Read:**
- `prereg/PREREG_S1.md` (whole).
- `results/s1_report.txt`, and `results/s1_stats.json` through `./py -c`.
- `results/s1_budget.txt`.
- `tail -30 .cache/logs/job_S3.log`.
- `paper/appendix.template.tex`: the section headings and the last 40 lines.
- `paper/body.template.tex`, only the lines matching "recorded commands", "policy" or "closed loop" (grep, with 3 lines of context). These are in:
  - the abstract;
  - the introduction's scope sentence;
  - Discussion item (3);
  - Limitations;
  - the Conclusion.
- `scripts/revision_numbers.py`, the first 60 lines: its `add(put, sig, sgn, pct, cib, dp, load, rel)` is called from `scripts/paper_numbers.py`.

**Steps**
1. **Confirm the job finished,** and that checks S1-C1 to S1-C4 are recorded.
   - Handle a failure as `PREREG_S1.md` says, and record it in `ANOMALIES.md`.
   - If a failed check withholds the CL verdicts, stop BLOCKED.
2. **Run `./py scripts/s1_stats.py`.** It is registered; do not edit it. Its verdicts are the result, whatever they are.
3. **Write `scripts/s1_numbers.py`** with the same `add(...)` pattern, called from `paper_numbers.py` after `revision_numbers`. Register every number the new text uses, with its source file and field.
4. **Write the appendix** in plain language. Label secondary and exploratory results, and give the time of the registration commit, from git.
   - **A8, "Checks with a policy in the loop":**
     - the design, in about five sentences;
     - a table of each check's ρ with E100 under recorded, policy and noisy joint commands, on F18 and X5;
     - the verdicts of CL1 and CL2;
     - the secondary readings, including divergence counts and the sign of CL6;
     - what was known beforehand.
   - **A9, "Other simulator architectures":**
     - the pool: the existing and the newly trained runs, and how the new ones were trained;
     - a table of ρ with E100 for the four checks;
     - the verdicts of AR1 and AR2;
     - the AR4 reading.
   - **A10, "Training length":** from `s1_budget`, labelled descriptive.
5. **Edit the body without adding length;** the two-column check has no slack.
   - **Discussion and Conclusion.** Discussion (3) says the checks "were tested only with recorded commands, not with commands from a policy". The Conclusion says neither "has been tested in closed loop with a policy". Replace those words with the CL result and a pointer to appendix A8.
   - **Limitations.** Keep "no policy trained", which is still true. Change only words the result makes false.
   - **Abstract and introduction.** Change them only if a sentence there is now false, and don't lengthen the abstract.
   - **AR or TB.** If either changes a claim in the body, replace only the words affected.
   - **Length.** If the two-column PDF exceeds six pages, shorten the new words. If it is still over, leave the body unchanged, and say so in the notes.
6. **Claims.** Add assertions to `scripts/check_claims.py` for every new sentence: verdicts, directions and counts.
7. **Finish.**
   - Build. The two-column PDF must have at most six pages.
   - Write `step1/notes/S6.md`, including the list of new keys.
   - Commit "S6: Step 1 results in the paper". Include `results/raw_S1.npz`, `s1_models.json`, `s1_stats.json`, `s1_report.txt`, `results/s1/` and `s1_budget.*`, but not `runs/`.
<!-- END S6 -->

<!-- BEGIN S7 -->
## S7 — Claims, references, build checks, supplement (Sonnet 5.5, high)

**Read:**
- `step1/notes/S6.md`, and the assertions S6 added to `scripts/check_claims.py`.
- The template changes: `git diff air2027-step1-start -- paper/*.template.tex`, in parts of at most 200 lines.
- `SUBMIT.md`: the supplement paragraph only.

**Steps**
1. **Run the full pipeline.** Every step must pass:
   - `./py scripts/paper_numbers.py`;
   - `./py scripts/build_paper.py`, twice, giving identical text through `scripts/pdf_tools.py`;
   - `./py scripts/check_claims.py`;
   - `./py scripts/check_bib.py` (strict);
   - `./py scripts/submission_checks.py`.
2. **Check against the evidence.** Ask air-claim-checker to check two things against `results/s1_stats.json`, `results/s1_budget.json`, `prereg/PREREG_S1.md` and the key list in `step1/notes/S6.md`:
   - appendix sections A8–A10 in `paper/appendix.tex`;
   - the changed body sentences in `paper/body.tex`.

   Fix each finding minimally, and never strengthen a claim. Log every change in `step1/notes/S7.md`, then rerun step 1.
3. **Rebuild the supplement.** Round 1 built `submission/AIR2027_supplement.zip` by hand.
   - **Learn its contents** with `unzip -l` and `unzip -p … README_SUPPLEMENT.md`. Keep a copy as `.cache/AIR2027_supplement_round1.zip`.
   - **Write `scripts/build_supplement.py`.** It rebuilds the same contents with the same exclusions, adding:
     - `prereg/PREREG_S1.md`;
     - the `s1_*` scripts and `test_s1.py`;
     - `scripts/lite/`, stubs included;
     - the new result files;
     - a Step 1 section in `README_SUPPLEMENT.md`, with the registration time.
   - **Scan the new zip.** Unzip it into `.cache/supp_check`, and scan it for the identifying patterns of `scripts/submission_checks.py`, plus "/Users/", "Downloads" and "joyjeetsingh". On a hit:
     - neutralise it in the copy only;
     - rebuild and rescan;
     - record what changed.
   - Never include `history.bundle` or the `.git` folder.
4. Commit "S7: checks and supplement".
<!-- END S7 -->

<!-- BEGIN S8 -->
## S8 — Fresh-eyes review and report (Opus 5.5, high)

**Read:**
- `step1/notes/S4_lite.md`, `S5.md`, `S6.md` and `S7.md`.
- `prereg/PREREG_S1.md` and `results/s1_report.txt`.

**Steps**
1. **Extract only the new material.** With `scripts/pdf_tools.py`, write the text of appendix sections A8–A10, plus the changed body sentences, to `step1/review/new_material.txt` (at most 400 lines).
2. **Get two reviews, in parallel.**
   - air-referee reviews `step1/review/new_material.txt` only.
   - air-stats-reviewer checks whether `results/s1_stats.json` follows `prereg/PREREG_S1.md` exactly: rules, pools and exclusions. It reads only those two files and `results/s1_report.txt`.
3. **Act on the reviews.**
   - Fix text issues and overclaims.
   - Run no new analyses; registered verdicts stand.
   - Decline anything else with a reason, under "S8 declined" in `step1/DECISIONS.md`.
4. **Recheck.** Rerun S7's step 1: everything must pass, and the two-column PDF must have at most six pages.
5. **Update the documents:**
   - `README.md`: the new scripts and result files, and the Step 1 run order;
   - `ANOMALIES.md`: anything from S0–S7;
   - `SUBMIT.md`: if the supplement changed.
6. **Write `step1/REPORT.md`** for the author, in plain language and at most 500 words:
   - what was done;
   - the verdicts of CL1, CL2, AR1 and AR2, and the main secondary readings;
   - what the training-length analysis showed;
   - the lite-code timings, and the Step 2 recommendation: CPU or a free notebook GPU, and how many hours;
   - what the author should check;
   - one optional author action: ask the RWM authors for the policy and settings that produced the released data. That would allow a closed-loop test against ground truth.
7. **Close the round.**
   1. Commit "S8: Step 1 complete", and tag it `air2027-step1`.
   2. Regenerate the history bundle with `git bundle create history.bundle --all`, then run `git bundle verify history.bundle`.
   3. Overwrite `step1/CHECKPOINT.md`: "Step 1 complete. Next: read step1/REPORT.md; Step 2 needs a decision on a GPU."
<!-- END S8 -->

## Traceability: Step 1 items → review weaknesses → sessions

| Step 1 item | Weakness it addresses | Sessions |
|---|---|---|
| (a) Checks with a policy in the loop (study CL) | The label-free checks were tested only with recorded commands; no policy in the loop | S1, S3, job S3, S6 |
| (b) MLP, RSSM and Transformer simulators, 12 of them trained here (study AR) | Evidence from one simulator design only | S2, S3, job S3, S6 |
| (c) Training length (study TB) | The parent models were not fully trained | S5, S6 |
| (d) Lite policy training on the CPU (task LP) | A prerequisite for robot-level evidence (Step 2) | S4 |

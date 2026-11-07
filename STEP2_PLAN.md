# AIR 2027 — Step 2: robot-level evidence in a hand-built MuJoCo ANYmal D (unattended plan)

**The question.** Do the checks that rank learned simulators by long-horizon error also rank them by how well a policy trained inside them walks in a physics simulator?

**The route.** Everything runs on this Mac, at no cost:
1. Build an ANYmal D model in MuJoCo by hand, from ANYbotics' published robot description, with the same learned motor model Isaac Lab uses for ANYmal D.
2. Check it against the recorded data, and calibrate a few physical constants on training episodes only.
3. Train walking policies inside about 25 of the paper's learned simulators, then measure every policy in the MuJoCo model.
4. Write the result into the paper's Step 2 slot.

**Two go/no-go checks are fixed in this plan before any MuJoCo number exists:**
- **Gate 1:** the RWM authors' own policy must walk in the MuJoCo model much as it does in the recorded data. If it doesn't, the hand-built physics is too far from the data's source to judge anything.
- **Gate 2:** a four-policy pilot must show that policies trained in learned simulators can walk, and that they differ. If they can't walk, a fallback pool is used: the released model and its compressed copies.

**For the author — how to run it.** In a terminal (the app's terminal or Terminal.app):

```
cd ~/Downloads/PDM/"AIR 2027"
chmod +x go_s2.zsh
caffeinate -i ./go_s2.zsh auto     # every session and job in order; safe to re-run after a stop
./go_s2.zsh status                 # progress, from another terminal
```

**Time:** about 8–10 days, nearly all of it unattended.
- Building and checking the model: about 2–3 days.
- The pilot: about 8 hours.
- Training and measuring about 50 policies: about 4 days. This is the longest part, so keep the Mac plugged in with the lid open.
- The analysis and paper: about 1 day.

That leaves margin before the 20 December cut-off. The jobs have no time limit, and the launcher stops one only if it makes no progress for an hour. If Claude's usage limit is hit, the launcher waits for the reset.

**Downloads and installs, done once by a job the launcher runs, not by Claude:**
- ANYbotics' `anymal_d_simple_description` (BSD 3-Clause licence);
- the ANYdrive motor network file from NVIDIA's public Isaac Lab assets. It is used locally, never redistributed;
- MuJoCo and four small helper packages, into `AIR 2027/.pylib-mj`. Your Python environment is not touched.

**If it stops BLOCKED** (for example at a gate): read `step2/DECISIONS.md` and write your answer there. Then run `./go_s2.zsh M<n> resume`, and `auto` again.

**How work is split.** One headless Claude Code session per block, plus four jobs the launcher runs between sessions.

| Step | Model | Effort | Max turns | Work |
|---|---|---|---|---|
| M0 | Sonnet 5.5 | medium | 80 | branch; facts (Isaac Lab's ANYmal D settings, the robot description, MuJoCo for this Mac); write the fetch job |
| job fetch | — | — | — | downloads and the MuJoCo install, about 10 minutes |
| M1 | Opus 5.5 | high | 180 | build the MuJoCo ANYmal D model and the walking environment; tests |
| M2 | Opus 5.5 | high | 150 | fidelity measures and the calibration code; write the calibration job |
| job calib | — | — | — | calibration on training episodes; fidelity on held-out ones; the authors' policy in MuJoCo, about 1–3 hours |
| M3 | Opus 5.5 | high | 100 | Gate 1; policy training and evaluation code; write the pilot job |
| job pilot | — | — | — | four pilot policies trained and measured, about 8 hours |
| M4 | Opus 5.5 | high | 150 | Gate 2; register the main study; write the main job |
| job main | — | — | — | about 50 policies trained and measured, about 4 days |
| M5 | Opus 5.5 | high | 150 | registered analysis; the Step 2 slot, appendix and scope sentences |
| M6 | Sonnet 5.5 | high | 120 | claim assertions, all checks, supplement, documents |
| M7 | Opus 5.5 | high | 100 | fresh-eyes review, fixes, report, tag |

Opus does the physics modelling, the gates, the registration and the writing. Sonnet does the fact-finding, the checks and the paperwork.

| Subagent | Model | Used by |
|---|---|---|
| air-bib-checker (web) | Sonnet 5.5 | M0, M1 |
| air-reader (read-only files) | Sonnet 5.5 | M0 |
| air-stats-reviewer | Opus 5.5 | M4, M7 |
| air-claim-checker | Sonnet 5.5 | M6 |
| air-referee | Opus 5.5 | M7 |

## Design fixed by this plan (committed by M0 before any MuJoCo number exists)

**Simulator pool:** 25 single-network simulators from the main pool (F18's split: episodes 1 and 8 held out).
- the five seeds of the released configuration;
- three teacher-forced models;
- three trained with N = 1, three with N = 2 and three with N = 32;
- the eight compressed copies of seed 0.

Every member has one network, so the lite code's uncertainty penalty is inactive for all of them alike.

**Policies:**
- two seeds per simulator, 50 policies in all;
- the authors' lite code, unchanged apart from Step 1's shims: 500 iterations, 4,096 robots;
- start states only from the simulator's own training episodes (`--holdout 1,8`).

**Measuring a policy in MuJoCo:**
- **Command schedules:** the ten recorded episodes' velocity-command schedules, 20 s each.
- **Starts:** eight perturbed starts per schedule, seeded. Joint angles move by up to ±0.05 rad, and heading is random.
- **Falls:** an episode ends at a fall, by Isaac Lab's rule for this task (M0 finds it).
- **Measures:**
  - **primary:** physics return, which is the lite code's own reward terms and weights computed from MuJoCo states and foot contacts, averaged over the 80 episodes;
  - **secondary:** fall rate, speed-tracking error (m/s) and turn-tracking error (rad/s);
  - a simulator's score is the mean over its two policies.

**Gate 1, the physics model** (after calibration; the authors' policy on the 80 evaluation episodes):
- **G1:** it falls in fewer than 10% of episodes;
- **G2:** its share of time with two feet down is within 10 percentage points of the recorded data's;
- **G3:** its mean speed-tracking error is at most 1.5 times the recorded data's own error on the same schedules.

**Gate 2, the pilot.** Four policies, one seed each: the released model, seed 0, one teacher-forced model, and the 50%-pruned copy of seed 0.
- **P1:** the policy trained in the released model falls in fewer than 25% of episodes;
- **P2:** at least one of the three policies trained in rebuilt simulators falls in fewer than 50%;
- **P3:** the four physics returns are not all inside each other's 95% intervals.

**If the gates fail:**
- **G1–G3 still fail after calibration, or P1 fails:** stop BLOCKED with a diagnosis.
- **P2 or P3 fails:** register the fallback pool instead. It is the released model and its eight compressed copies, two policy seeds each, all five-network, so the penalty is active alike.

**Calibration** (training episodes only, never episodes 1 and 8):
- a grid over four constants:
  - foot friction: 0.6, 0.8, 1.0 or 1.2;
  - contact stiffness time constant: 0.005, 0.01 or 0.02 s;
  - joint armature: 0, 0.02 or 0.05 kg·m²;
  - joint damping: 0, 0.5 or 1.0 N·m·s/rad;
- the objective is the mean 8-step error on training episodes, ties broken by one-step error;
- the chosen values are frozen before any held-out number or policy is measured.

**Motor model:**
- **Main:** Isaac Lab's ANYdrive network. Its input is the target-minus-actual joint angle and the joint speed; it keeps its memory per joint; it is clipped as Isaac Lab clips it; it runs at every 0.005 s physics step.
- **Fallback,** if the file can't be obtained: a PD controller with a fixed delay, fitted to the recorded torques of the training episodes. The method is recorded.

<!-- BEGIN COMMON -->
## Common rules (every session)

### C1 What this round is
Step 2 tests whether the paper's checks predict how well policies trained inside learned simulators walk in physics. The physics is a MuJoCo ANYmal D model built by hand in this round. Sessions M0–M7 and four jobs run in order; yours is the block after these rules.
- Earlier outcomes are in `step2/LOG.md`, `step2/CHECKPOINT.md` and `step2/notes/`.
- The design, the gates and the calibration grid are fixed in `STEP2_PLAN.md` ("Design fixed by this plan"). Never change them; a needed change goes to `step2/DECISIONS.md` and stops the run BLOCKED.
- **Terms.** Keep these two distinct:
  - **joint commands** are the actions, the 12 joint position targets;
  - the **velocity command** is the walking speed and turning rate a policy follows.

### C2 Running unattended
- **Never ask the user anything.** When this plan does not settle a choice:
  - take the more conservative option;
  - record it in `step2/DECISIONS.md` under your session id;
  - go on.

  Stop BLOCKED when going on would break a rule or the fixed design, or when a gate fails as the plan says.
- **Never start** `claude`, any `go_*.zsh` launcher, `caffeinate` or a job yourself. Anything over about 10 minutes goes into a job file the plan names, and the launcher runs it.
- **On a resumed attempt,** first read `step2/CHECKPOINT.md`, `tail -20 step2/LOG.md`, your notes, and `git log --oneline -15`. Continue from the first unfinished step.
- **Before ending,** with any status: commit (C7), append your status line to `step2/LOG.md`, and overwrite `step2/CHECKPOINT.md` (C9).

### C3 What you may change
- **Where you may write.** Write only inside this folder (`AIR 2027`).
  - The launcher compares the rest of `~/Downloads/PDM` before and after every session and job.
  - Never open key files (`../rwm_repro/sshkey*`), and don't read `../rwm_private/`.
- **Sibling folders** (`../robotic_world_model_lite`, `../rsl_rl_rwm`, `../rwm_repro`) are read-only. Use their code through `sys.path`, as `scripts/lite/` already does.
- **Read-only from earlier rounds:**
  - every registration (`prereg/*`), plus `scripts/ft_stats.py` and `scripts/s1_stats.py`;
  - once M4 commits them, `prereg/PREREG_S2.md` and `scripts/s2_stats.py`.
- **Downloads and installs happen only in `step2/jobs/fetch.zsh`,** which the launcher runs. They go only into `step2/assets/` and `.pylib-mj/`, with versions, sources and sha256 sums recorded in `step2/assets/manifest.json`.
  - Sessions never run `pip`, `curl` or `git clone` themselves.
  - The ANYdrive file is never committed and never put in the supplement.
- **No sudo. No external actions:** never push, upload, e-mail, submit or post.

### C4 Integrity
- **Registration before data.** Until M4 commits `prereg/PREREG_S2.md`, no policy trained in a main-pool simulator may be measured in MuJoCo. The gate measurements, of the authors' policy and the four pilot policies, are validation: they are declared in the registration, and are not results.
- **Calibration** uses training episodes only. Held-out numbers are computed once, after the constants are frozen.
- **No number is typed into a template.** Values reach the paper through `{{key}}`s, from result files written by scripts.
- **Registered verdicts** stand as registered. Secondary, exploratory and descriptive results are labelled so everywhere.
- **New result files only:** `results/s2_*` and `results/s2/`. Never overwrite an existing result file.
- **Never edit a shell script while it runs.**

### C5 Environment
- **Python:**
  - `./py` for torch work: training policies, the lite code, the learned simulators;
  - `./py-mj` (written by the fetch job) is the same interpreter plus `.pylib-mj`, for MuJoCo.
- **Machine:** Intel i5, 4 logical cores, 8 GB. Sessions use at most 2 threads; jobs may use 4 when no session runs.
- **Long output:** send it to `.cache/logs/`, and print only its tail.
- **Job progress:** every job rewrites `results/s2/progress.json` at least every 10 minutes, from the work itself, never from a timer. The launcher stops a job that makes no progress for an hour. A job is resumable: it skips finished work.

### C6 Context and token discipline
- **How to read:** read only what your block names; grep first; at most about 200 lines at a time; print JSON fields with `./py -c`; at most about 40 lines of output.
- **Never read these whole:** other rounds' plans; `results/paper_numbers.*`; any `results/raw_*.npz`; the robot description's mesh files; `history.bundle`; logs (use `tail`).
- **Subagents:** only those your block names, and never a `model` parameter. Give each a self-contained brief.

### C7 Git
- Work on branch `step2`, which M0 creates from the improvement round's last commit. Commit subjects are "M<n>: <what>".
- **Staging:** stage only your paths, never `git add -A`. Never push, checkout, restore, reset --hard, stash, clean or rebase.
- **Never commit:** `runs/`, `.cache/`, `.pylib*`, `step2/assets/` (it has its own manifest), `paper/build/`, `submission/` or `history.bundle`.
- **Tags:** only `air2027-step2-start` (M0) and `air2027-step2` (M7).

### C8 Style
- **Readers are roboticists.** Use plain language; define terms; one idea per sentence; at most two numbers-with-intervals in a sentence.
- **Honesty.** State failures as plainly as successes.
- **Scope.** The physics is a hand-built model of the robot, not the data's own simulator, and not hardware. Say so wherever a claim could be read more widely.
- **Notes** are at most 300 words unless the block says otherwise.

### C9 Ending a session
- **Status line.** Append to `step2/LOG.md` the line `## M<n> status: COMPLETE (YYYY-MM-DD HH:MM)`, with PARTIAL or BLOCKED in place of COMPLETE where needed.
  - Then add at most 8 lines.
  - The launcher reads this line, so write it exactly in that form.
- **Checkpoint.** Overwrite `step2/CHECKPOINT.md` with the last session and status, the next action, and any blockers.
- **BLOCKED:** add to `step2/DECISIONS.md` what is blocked, the options, and your recommendation.
- **PARTIAL:** commit what is done, and write the exact next step in `step2/CHECKPOINT.md`.
<!-- END COMMON -->

<!-- BEGIN M0 -->
## M0 — Branch, facts and the fetch job (Sonnet 5.5, medium)

**Read:**
- `improve/CHECKPOINT.md`;
- `step1/notes/S4_lite.md`;
- `ls scripts/lite/ scripts/`.

**Steps**
1. **Git.**
   - Check that HEAD is tagged `air2027-improve`; if not, record what is there.
   - Tag `air2027-step2-start`, then run `git switch -c step2` (resumed: `git switch step2`).
   - Commit `STEP2_PLAN.md` and `go_s2.zsh` alone: "M0: Step 2 plan (written before any MuJoCo number)".
   - Create `step2/` with `LOG.md`, `DECISIONS.md`, `CHECKPOINT.md`, `notes/`, `jobs/` and `assets/`.
2. **Facts from the web.** Ask air-bib-checker for each of the following, separately, at most 300 words each, with exact quotes and URLs.
   - **a. Isaac Lab's ANYmal D robot settings** (`isaaclab_assets/robots/anymal.py`, current release):
     - the motor-network file name and its download URL on NVIDIA's public asset server;
     - its effort, saturation and speed limits;
     - the default joint angles;
     - the starting base height;
     - the joint names.
   - **b. Isaac Lab's flat-ground velocity task for ANYmal D:**
     - physics step and decimation;
     - joint-command scale and offset;
     - the observation terms, in order, with any scaling;
     - the fall (termination) rule;
     - the command ranges;
     - the friction of the ground and the feet.
   - **c. The robot description:** the commit hash of `github.com/ANYbotics/anymal_d_simple_description`, its licence, and which files hold the URDF and the collision shapes.
   - **d. MuJoCo:** the newest version with a wheel for macOS on Intel (x86_64) and Python 3.11, and the pure-Python packages that `mujoco` imports at load time.
3. **Facts from local files.** Use air-reader for the sibling folders, and grep this folder yourself.
   - The joint order and signs in the recorded data: column names, or the mean standing angles.
   - Whether the 45 recorded state numbers include the 12 joint torques.
   - Where the lite code computes the imagination reward and its weights.
   - How `scripts/s1_policy.py` builds the 48 observation numbers.
4. **Write `step2/notes/M0_facts.md`** (at most 500 words, plus the quotes).
5. **Write `step2/jobs/fetch.zsh`.** It must:
   - `git clone` the description into `step2/assets/anymal_d/`, at the recorded commit;
   - download the ANYdrive file into `step2/assets/actuator/`, recording its sha256. If the download fails, record that and continue: M1 then uses the PD fallback;
   - run `../.venv-rwm311/bin/python -m pip install --target .pylib-mj --no-deps mujoco==<ver>` and the helper packages, each pinned;
   - write `./py-mj`: like `./py`, but with `.pylib-mj` also on `PYTHONPATH`;
   - check it: `./py-mj -c "import mujoco, numpy, torch; m = mujoco.MjModel.from_xml_string('<mujoco/>'); print(mujoco.__version__, numpy.__version__)"`. numpy must still be 1.26.4;
   - write `step2/assets/manifest.json`, print one line per item, and pass `zsh -n`.
6. Commit "M0: facts and fetch job".
<!-- END M0 -->

<!-- BEGIN M1 -->
## M1 — The MuJoCo ANYmal D model and walking environment (Opus 5.5, high)

**Read:**
- `step2/notes/M0_facts.md` and `step2/assets/manifest.json`;
- the description's URDF (grep the joints, links, inertias and collision elements; never the meshes);
- `scripts/s1_policy.py`, for the observation builder.

If the fetch job recorded a failure, follow the fallback in `STEP2_PLAN.md`, or stop BLOCKED if MuJoCo itself failed.

**Build**
- **`step2/model/anymal_d.xml`:**
  - the URDF's bodies, 12 hinge joints, inertias and joint limits;
  - simplified collision shapes from the URDF: sphere feet, capsule legs, box trunk;
  - a ground plane;
  - a 0.005 s time step;
  - one torque motor per joint, limited to Isaac Lab's effort limit.
  - The URDF can be imported by MuJoCo's compiler, then edited. Ask air-bib-checker for MuJoCo Menagerie's ANYmal C model (`anybotics_anymal_c`) if its settings help as a reference.
- **`scripts/s2/actuator.py`:** the ANYdrive network wrapper. It takes the target-minus-actual angle and the joint speed, keeps per-joint memory with a reset, and clips as Isaac Lab clips. Or the fitted PD fallback.
- **`scripts/s2/mj_env.py`:**
  - reset to the default pose;
  - joint command = default + scale × action;
  - `decimation` physics steps per policy step, with the motor model at every physics step;
  - the 48 observations exactly as `s1_policy.py` builds them, in the data's joint order;
  - Isaac Lab's fall rule;
  - the 45-number state in the data's order and units, joint torques included;
  - foot contacts;
  - the lite reward terms, imported from the lite code where possible.

**Tests** (`scripts/s2/test_mj.py`, at most 10 minutes):
- **(a)** The model compiles. Total mass and inertia match the URDF to 1e-6.
- **(b)** Joint order and signs: the default pose's signs match the recorded standing pattern.
- **(c)** The motor wrapper equals Isaac Lab's computation on random inputs, re-implemented from the quoted source.
- **(d)** With zero actions, the robot stands for 10 s: the base height drifts less than 2 cm, and it doesn't fall.
- **(e)** The observation builder equals `s1_policy.py`'s on a constructed state.
- **(f)** The environment runs the authors' policy for 20 s without errors and with finite values. Measure nothing about walking here; that is Gate 1.

**Probe:** the seconds per 20-second episode.

**Finish:** write `step2/notes/M1.md`, and commit "M1: MuJoCo ANYmal D model and environment".
<!-- END M1 -->

<!-- BEGIN M2 -->
## M2 — Fidelity measures and calibration code (Opus 5.5, high)

**Read:**
- `step2/notes/M1.md`;
- `scripts/evalh.py` and `scripts/common.py`: the windows, the error measure and the arenas;
- the calibration grid and Gate 1 in `STEP2_PLAN.md`.

**Build `scripts/s2/fidelity.py`:**
- **(a) Replay.** From recorded states, replay the recorded joint commands in MuJoCo.
  - The base height comes from leg kinematics, with the lowest foot on the ground.
  - Orientation comes from the projected gravity.
  - Compute E1, E8, E24 and E100 with exactly `evalh`'s windows, plus the torque error.
  - Do this for any chosen episodes.
- **(b) Gait of the authors' policy.** Run it on the evaluation protocol (ten schedules × eight starts). Measure falls, the share of time with two feet down, duty factor, and speed- and turn-tracking errors.
- **(c) Reference.** The same gait statistics computed from the recorded data on the same schedules.

**Build `scripts/s2/calibrate.py`:**
- the fixed grid;
- the objective is the 8-step error on training episodes (all except 1 and 8);
- it is resumable;
- it writes `step2/model/calibration.json` and `step2/model/anymal_d_cal.xml`.

**Tests** (at most 10 minutes):
- replay with recorded commands on one training window runs, and gives finite values;
- the window positions equal `evalh`'s;
- held-out episodes are refused by `calibrate.py`.

**Write `step2/jobs/calib.zsh`.** In order:
1. calibrate;
2. freeze the constants;
3. fidelity (a) on the held-out trajectories, for the uncalibrated and the calibrated model;
4. (b) and (c).

It writes `results/s2_fidelity.json`, and `results/s2_fidelity.txt` (at most 40 lines), with progress as in C5.

**Finish:** write `step2/notes/M2.md`, and commit "M2: fidelity and calibration".
<!-- END M2 -->

<!-- BEGIN M3 -->
## M3 — Gate 1 and the pilot job (Opus 5.5, high)

**Read:**
- `results/s2_fidelity.txt`;
- `step2/notes/M2.md`;
- Gate 1, Gate 2 and the design in `STEP2_PLAN.md`;
- `step1/notes/S4_lite.md`.

**Steps**
1. **Gate 1.** Apply G1–G3 to the calibrated model. Record the reading in `step2/notes/M3.md`.
   - If it fails, stop BLOCKED, with the diagnosis and these options:
     - a wider calibration;
     - the Isaac Lab route (see the end of this plan);
     - stopping Step 2.
2. **Write `scripts/s2/train_policy.py`.**
   - It wraps `scripts/lite/run_lite.py`: 500 iterations, 4,096 robots, `--holdout` from the simulator's fold, a seed.
   - It exports the trained actor to `runs/s2/<policy>/actor.pt`, plus its imagined return from the last iterations.
   - It writes progress every 25 iterations.
3. **Write `scripts/s2/evaluate.py`.**
   - It measures one exported actor in `anymal_d_cal.xml` with the fixed protocol.
   - It writes `results/s2/eval/<policy>.json`, and is resumable.
4. **Tests** (at most 10 minutes): a 2-iteration training run exports an actor that `evaluate.py` can measure, on one schedule and one start.
5. **Write `step2/jobs/pilot.zsh`.** It trains the four pilot policies, one seed each, then measures them. It writes `results/s2_pilot.json` and `.txt`.
6. **Finish:** write `step2/notes/M3.md`, and commit "M3: Gate 1 passed; pilot job".
<!-- END M3 -->

<!-- BEGIN M4 -->
## M4 — Gate 2, registration and the main job (Opus 5.5, high)

**Read:**
- `results/s2_pilot.txt` and `results/s2_fidelity.txt`;
- the design in `STEP2_PLAN.md`;
- `scripts/stats.py`: `spearman`, `_clean` and `boot_diff`. Import these; never copy them;
- `results/model_table.json`, through `./py -c`, to map the pool to model ids.

**Steps**
1. **Gate 2.** Apply P1–P3.
   - If P1 fails, stop BLOCKED.
   - If P2 or P3 fails, use the fallback pool.
   - Record the reading.
2. **Write `prereg/PREREG_S2.md`,** in plain language:
   - **The question.**
   - **The pool,** the exact model ids, by the plan's rule.
   - **Training settings** and **the MuJoCo protocol,** quoted from the plan.
   - **Measures:** physics return (primary); falls and tracking (secondary).
   - **Checks:** E1, E8, E100, λ1 and r̄J under recorded commands; λ1 and r̄J under policy commands; the imagined return. All are stored values.
   - **Statistics:**
     - the simulator is the unit, with policy seeds averaged;
     - Spearman correlation, with checks negated so that higher is better;
     - paired bootstrap over simulators, 20,000 draws, `default_rng(0)`.
   - **Premise (95%). R0:** ρ(−E100, return). SUPPORTED if the interval lies above 0, CONTRADICTED if below, NOT SETTLED otherwise.
   - **Primary family (97.5%):**
     - **R1:** ρ(−λ1, return) − ρ(−E1, return), with labels LABEL-FREE BETTER / ONE-STEP BETTER / NOT SETTLED;
     - **R2:** ρ(−r̄J, return) − ρ(−E8, return), with labels LABEL-FREE BETTER / TRAINING-HORIZON BETTER / NOT SETTLED.
   - **Secondary (95%):**
     - R0–R2 with the fall rate as the outcome;
     - the same with checks computed under the policy's commands;
     - ρ(imagined return, physics return);
     - selection: the physics return of the top three simulators chosen by each check, against the best three;
     - the rollout-trained subset alone.
   - **Exploratory:** two-level intervals, with policy seeds inside simulators.
   - **Expected outcome:** with 25 simulators, the differences may well end NOT SETTLED. Say so here.
   - **What was known:** Gate 1's and the pilot's readings, and all earlier results.
3. **Write `scripts/s2_stats.py`, `scripts/s2_main.py` and tests.**
   - `s2_main.py` trains, exports and measures every policy, and is resumable.
   - The tests use synthetic results and constructed intervals for every label.
4. **Review.** Ask air-stats-reviewer to review all of these against this block, in at most 400 words. Fix every MUST FIX and SHOULD FIX.
5. **Register.**
   1. Commit them together: "M4: PRE-REGISTER Step 2 (no main-pool policy measured)".
   2. Write `results/prereg_hash_s2.json` and commit it.
   3. Run `chmod a-w prereg/PREREG_S2.md scripts/s2_stats.py`.
6. **Write `step2/jobs/main.zsh`.**
   - It runs `./py scripts/s2_main.py`, using 4 threads (no session runs during it), or two policies side by side if a 10-minute probe shows that is faster.
   - Record the choice.
7. Commit "M4: main job".
<!-- END M4 -->

<!-- BEGIN M5 -->
## M5 — Analysis and the paper (Opus 5.5, high)

**Read:**
- `prereg/PREREG_S2.md`;
- `tail -30 .cache/logs/job_s2_main.log`;
- `results/s2_fidelity.txt`;
- the Step 2 slot, the abstract's scope sentence, Limitations and the Conclusion in `paper/body.template.tex` (grep);
- the appendix headings.

**Steps**
1. **Run `./py scripts/s2_stats.py`.** It is registered; do not edit it. Its verdicts are the result, whatever they are.
2. **Write `scripts/s2_numbers.py`,** with the `add(...)` pattern, and call it from `paper_numbers.py`. Register every new key.
3. **Step 2 slot.** Replace `\stepslot` with one paragraph and one small table: R0–R2, with the physics model's scope stated plainly. It must fit the slot's space.
4. **Update other text** only where it is now false:
   - the abstract's scope ("no physics-simulator test");
   - Limitations;
   - the Conclusion;
   - the Discussion sentence about policy performance.

   The abstract stays at most 200 words.
5. **Appendix:**
   - **A11, "A hand-built MuJoCo ANYmal D":** the build, the motor model, calibration, Gate 1, and the model's own E1/E8/E100 on the held-out trajectories (descriptive);
   - **A12, "Policies trained in the simulators":** the pool, training, protocol, the pilot, all results.
6. **Fit.** `slack.py` must PASS with the reserve set to 0, since the slot is now used.
7. **Finish:** write `step2/notes/M5.md` with the new keys and claims, and commit "M5: Step 2 results in the paper".
<!-- END M5 -->

<!-- BEGIN M6 -->
## M6 — Claims, checks, supplement and documents (Sonnet 5.5, high)

**Read:**
- `step2/notes/M5.md`;
- the start of `scripts/check_claims.py`, and its claim descriptions (grep).

**Steps**
1. **Claims.** Add an assertion for every new claim. All must pass.
2. **Full pipeline,** with every step passing: `paper_numbers.py`, `build_paper.py` twice (identical text), `slack.py`, `check_claims.py`, `check_numbers.py`, `check_bib.py` and `submission_checks.py`.
3. **Check against the evidence.** Ask air-claim-checker to check the Step 2 slot, appendices A11–A12 and the changed sentences against `results/s2_*` and `prereg/PREREG_S2.md`. Fix each finding minimally.
4. **Supplement.**
   - Add `scripts/s2/`, the MJCF model with the BSD licence notice of the description, `prereg/PREREG_S2.md` and the `s2` results.
   - Leave out the ANYdrive file; give its URL and sha256 instead.
   - Rebuild, and scan as before. It must have 0 hits.
5. **Documents.** Update `README.md`, `SUBMIT.md` and `ANOMALIES.md`.
6. Commit "M6: checks, supplement, documents".
<!-- END M6 -->

<!-- BEGIN M7 -->
## M7 — Fresh-eyes review, report and tag (Opus 5.5, high)

**Read:** `step2/notes/M5.md` and `prereg/PREREG_S2.md`.

**Steps**
1. **Extract the main text** of the two-column check to `step2/review/body.txt`.
2. **Get two reviews, in parallel.**
   - air-referee reviews `body.txt`.
   - air-stats-reviewer checks `results/s2_stats.json` against `prereg/PREREG_S2.md`, and the Step 2 sentences against both.
3. **Act on the reviews.** Fix the text. Run no new analysis. Decline anything else with a reason, under "M7 declined".
4. **Rerun the pipeline.** Everything must pass.
5. **Write `step2/REPORT.md`** (at most 500 words, plain language):
   - Gate 1, the pilot, and the verdicts;
   - what the physics model can and cannot show;
   - the referee's score;
   - what the author must check.
6. **Close the round.**
   1. Commit "M7: Step 2 complete", and tag it `air2027-step2`.
   2. Run `git bundle create history.bundle --all` and `git bundle verify history.bundle`.
   3. Write `step2/CHECKPOINT.md`: "Step 2 complete. Next: read step2/REPORT.md; submit by 31 Dec."
<!-- END M7 -->

## The alternative kept in reserve

If Gate 1 fails, the route still open is the one in the earlier version of this plan:
- measure the same policies in Isaac Lab, the data's own simulator family, on a rented RTX GPU;
- about $3–5 for two short sessions, at Vast.ai's listed $0.38–0.67 per hour for an RTX 4090 (6 Nov 2026).

Everything from M3 on would carry over, apart from where the measuring runs.

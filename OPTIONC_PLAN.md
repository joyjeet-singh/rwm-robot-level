# AIR 2027 — Option C: the right data, measured where it was made (unattended plan)

## Why Step 2's rebuilt-simulator policies fell

**What happened** (`results/s2_pilot.txt`). The three policies trained in the paper's rebuilt simulators fell in all 80 MuJoCo episodes. One trained in RWM's released model fell in 10%.
- **In their simulators, the failed policies looked better.** They earned 1.8–2.5 times the imagined reward of the released model's policy: 3.8–5.1 against 2.07.
- **In MuJoCo, they barely stood.** They had exactly two feet down only 18–25% of the time, against 57% for the released model's policy and 79% for the authors' own policy.

A policy that scores well in its simulator but fails in physics has found the simulator's mistakes and used them.

**Why, as far as the evidence goes, strongest first:**
1. **The data held no falls.**
   - In all ten recorded episodes the fall column is zero; the rebuild's data loader even checks this.
   - A simulator can learn to predict a fall only from examples of falls, so the rebuilt simulators' fall predictor learned "never". In imagination, falling then cost little or nothing. D1 measures how often each simulator predicts a fall.
   - The released model was trained by RWM's authors on their own, much larger data. That data came from many policies during training, so it very likely includes falls.
2. **Nothing kept the policy where the simulator was right.**
   - The lite code subtracts the five networks' disagreement from the reward, with weight 1.0, the value RWM-U calibrated.
   - A single-network simulator has no disagreement, so it had no penalty. Step 1's notes had said so and recommended the five-network rebuilds (`A5_*`).
   - Step 2's plan used single-network simulators anyway, to keep the runs alike. That was a mistake in the plan.
   - The released model has five networks, so its policies were penalised.
3. **The data were narrow.** About 8,000 transitions from one walking policy on about 21 commands, against RWM's 6 million from many policies. RWM-U found that data from a single expert overfits, and that data mixed from many stages of training works best.
4. **Policies were trained briefly.** 500 iterations with 4,096 robots, half the 8,192 the lite code uses by default. Even the released model's policies fell in 10–45% of episodes, against about 1% for the authors' own.

**What Option C does about it:**
1. **Test the diagnosis first** (D0–D6 below), on Step 2's simulators and on new ones.
2. **Make the right data.** Your calibrated MuJoCo ANYmal D passed Step 2's Gate 1, so it can stand in for the world. It generates "mixed" data like RWM-U's:
   - **who acts:** the authors' policy on every command; the same policy with noise; Step 2's 22 policies of every quality; and random exploration;
   - **conditions:** falls, pushes and varied physics;
   - **amount:** up to 6 million transitions, RWM's amount.
3. **Train simulators as RWM-U does,** with five networks so the penalty works. Train them on 10 thousand, 100 thousand, 1 million (RWM-U's amount) and 6 million transitions.
4. **Train policies inside them and measure the policies in the same MuJoCo.** Data source and test bed are one physics, so nothing is lost between them.

The data ladder answers "how much is enough". The pool then tests the paper's checks on simulators whose policies range from falling to walking, which is what a ranking test needs.

**For the author — how to run it.** In a terminal (the app's terminal or Terminal.app):

```
cd ~/Downloads/PDM/"AIR 2027"
chmod +x go_c.zsh
caffeinate -i ./go_c.zsh auto     # every session and job in order; safe to re-run after a stop
./go_c.zsh status                 # progress, from another terminal
```

You can keep using the Mac. Jobs check only the folders their code reads, so working in your other PDM projects won't stop them. The short Claude sessions check all of PDM. If one stops because something outside AIR 2027 changed and the change was yours, run `auto` again.

**Time.** About 7–10 days, nearly all unattended, at no cost:
- diagnosis, about 4 hours;
- data and the two pilot simulators, about 10 hours;
- pilot policies, 8–17 hours;
- the main pool, 4–7 days, depending on how many policy iterations the pilot shows are needed;
- the paper, about 1 day.

Started this week, it ends around 20 November, a month before the 20 December cut-off. Nothing is downloaded or installed: MuJoCo and the robot model are in place from Step 2.

**If it stops BLOCKED** (for example at a gate):
1. Read `optc/DECISIONS.md`.
2. Write your answer there.
3. Run `./go_c.zsh C<n> resume`, then `auto` again.

**10 November: the first pilot failed, and Amendment 1 changes the plan.**
- The pilot policy fell in every episode. In its simulator it lost reward on every step and a fall cost nothing, so it learned to end its episodes.
- Amendment 1 (in the design below) first checks that the simulator can hold a good walk, then tries four training recipes in a second pilot. A new session, C2R, sets this up.
- This adds one to two days; the round should end around 21–23 November.
- Continue with `caffeinate -i ./go_c.zsh auto`.

**How work is split.**

| Step | Model | Effort | Max turns | Work |
|---|---|---|---|---|
| C0 | Opus 5.5 | high | 150 | branch; facts; diagnosis code; write the diagnosis job |
| job diag | — | — | — | D0–D4, including two policies in `A5_s0`; about 4 h |
| C1 | Opus 5.5 | high | 180 | root-cause note; data generator; simulator trainer; tests and probes; write the data job |
| job data | — | — | — | 6M transitions, then the two pilot simulators and their checks; about 10 h |
| C2 | Opus 5.5 | high | 150 | Gate A; walking score; policy wrapper; the first pilot's job |
| job pilot (first) | — | — | — | failed on 10 November; see Amendment 1 |
| C2R | Opus 5.5 | high | 150 | Amendment 1: calibration, Gate A2 and fixes; the second pilot's job |
| job pilot | — | — | — | the second pilot: four policies to 1,500 iterations, then one more; 20–30 h |
| C3 | Opus 5.5 | high | 150 | Gates B and C; register; write the main job |
| job main | — | — | — | the remaining simulators and checks; 32 policies trained and measured; 4–7 days |
| C4 | Opus 5.5 | high | 150 | registered analysis; the paper |
| C5 | Sonnet 5.5 | high | 120 | claim assertions, all checks, supplement, documents |
| C6 | Opus 5.5 | high | 100 | fresh-eyes review, fixes, report, tag |

Opus does the work where an error would spoil the result: the diagnosis, the data and training code, the gates, the registration and the writing. Sonnet does the checks and the paperwork.

| Subagent | Model | Used by |
|---|---|---|
| air-reader (read-only files) | Sonnet 5.5 | C0, C1 |
| air-bib-checker (web) | Sonnet 5.5 | C0 |
| air-stats-reviewer | Opus 5.5 | C3, C6 |
| air-claim-checker | Sonnet 5.5 | C5 |
| air-referee | Opus 5.5 | C6 |

## Design fixed by this plan (committed by C0 before any Option C number exists)

### Diagnosis
**In the diagnosis job,** on Step 2's simulators and policies:
- **D0, what the data held (C0).** Count the rows in the ten recorded episodes with a fall (column 65) and with any thigh contact (columns 57–60).
- **D1, can the simulator see a fall?** For Step 2's four pilot policies (`REL__p0`, `A_s0__p0`, `B_s0__p0`, `A_s0__u50__p0`):
  - **in imagination:** run 512 imagined episodes in each policy's own simulator, as policy training does. Record the share that end in a predicted fall, and the mean reward per term.
  - **in physics:** re-run the 80 MuJoCo pilot episodes with Step 2's evaluator and seeds.
    - For each real fall, give the simulator the true 32-step history before the last 50 steps, and the joint commands the policy actually sent.
    - Record whether it predicts a fall within 0.2 s of the real one.
    - Record the reward per term in MuJoCo.
- **D2, distance from the data.** For each pilot policy: the mean distance from its imagined state–action pairs to their 10 nearest neighbours in its simulator's training data, with each column scaled by the training data's spread. Compare it with the same distance for the authors' policy's MuJoCo pairs.
- **D3, the penalty alone.**
  - Train two policies in `A5_s0`, the existing five-network rebuild. It was trained on the same 8,000 transitions, so it has the penalty but still no falls in its data.
  - Seeds 0 and 1; 500 iterations; 4,096 robots.
  - Measure them on Step 2's 80-episode protocol, and apply D1's imagination part to them.
- **D4, command coverage.** The share of policy-training commands (uniform in −1 to 1 on each axis) that lie within 0.2, on every axis, of a command in the recorded data.

**In the main job,** reported as descriptive:
- **D5, data without the penalty.** Two policies in the 6M rollout-trained seed-0 simulator, with the uncertainty weight set to 0.
- **D6, expert-only data from MuJoCo.**
  - A five-network simulator, trained like the pool's, on 8 episodes of the authors' policy alone: about 8,000 transitions, with no falls, like the recording.
  - Two policies, with the pool's uncertainty weight.

**How the pieces fit.** Each comparison isolates one cause:

| Comparison | What it tests |
|---|---|
| Step 2 against D3 | the penalty, at small data |
| D3 against D6 | where the data came from |
| D6 against the 10k rung | falls and variety, at the same amount of data |
| D5 against the 6M pool member | the penalty, at large data |
| the ladder | the amount of data |

### Data from MuJoCo (the calibrated model)
- **Episodes:** 20 s at 50 Hz, ended early by a fall (Step 2's fall rule).
- **Commands:** speed, sideways speed and turn, each uniform in −1 to 1, with a new command every 10 s, as in policy training.
- **Physics:**
  - half the episodes use the calibrated model unchanged;
  - the other half draw foot friction from 0.4–0.8, an added base payload of −5 to +5 kg, and motor gains ±10%;
  - every training episode gets a push of up to 0.5 m/s every 10–15 s.
- **Sources, by share of transitions:**
  - 30%: the authors' policy;
  - 25%: the authors' policy with Gaussian action noise of 0.1 or 0.3, half each;
  - 25%: Step 2's 22 trained policies, in equal shares;
  - 20%: exploration, half smooth random joint commands around the standing pose, half the authors' policy with noise 0.6.
- **Format:**
  - the recorded data's 66 columns (45 state, 12 action, 8 contacts, fall), plus an episode id and a source id;
  - float32 shards in `optc/data/` (not committed), with a committed manifest, `optc/data_manifest.json`.
- **Amounts:**
  - 6 million training transitions;
  - the 1M, 100k and 10k rungs are nested random subsets of whole episodes, drawn within each source so that the shares hold.
- **Held out** (never trained on):
  - **the mixed test set:** 40 more episodes from the same generator;
  - **the expert test set:** 20 episodes of the authors' policy, calibrated physics, no pushes (like the paper's recorded data);
  - **D6's 8 training episodes:** made like the expert test set, with other seeds.

### Simulators
- **Trainer:** a copy of the rebuild's trainer (`../rwm_repro/src/rwm_train.py`, verified equal to the authors' losses and gradients). Only the data loader changes:
  - a window of 32 + N steps never crosses an episode end, and may contain a fall only at its last step (the lite loader's rule);
  - normalisation comes from each rung's own training data.
- **Model:**
  - RWM's GRU, 2 × 256;
  - five members on a shared trunk (the rebuild's `ensemble_size = 5`, as in `A5`);
  - predicts the state, the contacts and the fall.
- **Training:**
  - the rebuild's batch size and learning rate;
  - the same number of steps, S, for every simulator: the larger of 2,500 (the `A5` setting) and the most steps that fit in 8 hours with two simulators training at once (C1's probe);
  - keep the final weights.
- **Pool, 14 simulators:**
  - rollout-trained (N = 8), seeds 0 and 1, at each rung (10k, 100k, 1M and 6M): 8;
  - teacher-forced (N = 1), seed 0, at each rung: 4;
  - 30%- and 50%-pruned copies of the 6M rollout-trained seed 0, not retrained: 2.

### Policies
- **Code and settings:**
  - the authors' lite code, through Step 2's route for rebuilt models;
  - 4,096 robots;
  - the lite reward unchanged, including the uncertainty weight of 1.0, unless the pilot's retry rule changes it.
- **Start states:** 32-step windows from the simulator's own rung. Each must be upright throughout (tilt under 30°) and end at least 1 s before any fall. Use at most 100,000 rows.
- **Iterations:** set by the pilot rule below.
- **Seeds:** 1 and 2 for the pool, 28 policies. The pilot uses seed 0.

### Walking score (the primary outcome)
- **Each step** scores exp(−(‖v_xy − c_xy‖² + (ω_z − c_ω)²)/0.25). This combines the task's two tracking terms.
  - v and ω are the base-frame speed and turn rate; c is the command.
  - After a fall, every remaining step scores 0.
- **Averages:** an episode scores the mean over its 1,000 steps, from 0 to 1. A policy scores the mean over its episodes; a simulator, the mean over its two policies.
- **Reference points:**
  - perfect tracking scores 1;
  - standing perfectly still scores about 0.07 on the protocol's commands. The evaluator computes this exactly for the protocol's schedules and reports it.
- **Protocol:**
  - calibrated physics, no pushes, no observation noise;
  - 24 fixed command schedules, uniform over the training range with a new command every 10 s, × 4 seeded starts = 96 episodes.
- **Secondary measures:** fall rate, speed and turn errors, physics return, and the ten recorded schedules.

### Checks
Computed on both test sets with the paper's functions, from the ensemble's mean prediction:
- the errors E1, E8, E24 and E100;
- the growth rate λ1 and the kinematic residual r̄J;
- λ1 and r̄J under each policy's own commands;
- the imagined return;
- the ensemble's disagreement along the policy's imagined rollouts (new, secondary).

### Gates and the pilot
- **Gate A (C2).** On the expert test set, the 6M rollout-trained seed-0 simulator's 8-step and 24-step errors must be below those of holding the last state. Otherwise, stop BLOCKED.
- **Pilot.**
  - Train seed-0 policies in the 6M and 10k rollout-trained seed-0 simulators, to 2,000 iterations.
  - Save them at 500, 1,000, 1,500 and 2,000 iterations, and measure each save with the full protocol.
  - Measure the authors' policy with the same protocol.
- **Gate B.** At 2,000 iterations, the 6M policy falls in under 20% of episodes and scores at least half the authors' policy.
- **Retry** (automatic; pilot only). If Gate B fails, train two more 6M seed-0 policies to 2,000 iterations, with the same saves:
  - if the 6M policy fell in 20% of episodes or more, with the uncertainty weight at 2.0 and at 4.0;
  - if it fell less but scored too low, with the weight at 0.5 and at 0.25.

  Keep the weight whose policy scores highest, provided it passes Gate B. If none passes, stop BLOCKED.
- **Iteration rule.** The main study uses the smallest saved count at which the kept 6M policy scores at least 90% of its own 2,000-iteration score.
- **Gate C** (not a stop). The 10k policy scores below the 6M policy, with their 95% intervals over episodes apart. If not, add a 2k rung (rollout-trained seeds 0 and 1, and teacher-forced seed 0), and record it before registering.

### Amendment 1 (10 November 2026, revised the same day; after the first pilot, before registration)

**What the first pilot showed** (`results/c/pilot_decision.json`, `results/c/eval/`, the lite metrics in `runs/c/lite/`):
- **The result.** The 6M simulator's policy (weight 1.0, 2,000 iterations) fell in all 96 episodes, scoring 0.0004 against 0.94 for the authors' policy. At 500 iterations it had fallen in 85%.
- **In imagination, it learned to end its episodes.** Imagined episodes shrank from about half the 256-step limit at iteration 500 to about one step at the end. The 10k policy did the same within 100 iterations, and D3's policies in `A5_s0` within 20.
- **Why ending paid.** Two things held together:
  - **the reward per step was negative.** In the 6M run the penalties outweighed the tracking rewards from the first iteration to the last. Early on, the vertical-speed and tilt penalties were larger than the uncertainty charge; later the uncertainty charge was the largest;
  - **a fall was free.** The lite reward has no cost for a predicted fall, so a policy losing reward every step gains by falling at once.
- **Why the released model's policy escaped.** Its reward per step was also negative for about its first 250 iterations. But the released model almost never predicts a fall (D1: 0.2% of imagined episodes), so its policy had no way out and had to learn to walk. Our simulators predict falls readily: `A5_s0` in every D1 episode, and ours were trained on data full of falls.
- **What is not established.** At the start of training the networks disagree about as much in the released model (about 2.0–2.2 per step) as in the 6M simulator (2.3); `A5_s0` and the 10k simulator are near 2.8. The released model's 0.6 is its value once the policy walks. No policy has yet walked in our simulators, so their value there is unknown. Calibration (change 1) measures it.
- **Two suspects for the early penalties:**
  - **start states.** The first pilot started imagination anywhere in the mixed data, including states from noisy and failing policies. The lite code starts from the authors' walking data;
  - **the simulator.** It was trained on about a tenth of its data, once: 2,572 steps × 256 windows against 6 million. Its kinematic residual on held-out data is about 9, against about 0.6–1.0 for the data themselves. Its prediction errors are good (Gate A), so this is a suspect, not a finding. Training also ran twice as fast as the probe assumed, so S used 4 hours of the 8 allowed.
- **The first pilot's retry rule pointed the wrong way** ("it fell, so raise the penalty").
  - Weight 2.0 crashed after about 1,500 iterations, when the policy's exploration noise went negative.
  - Weight 4.0 finished with a negative imagined return and was not measured.

**Changes.** These replace the design above wherever they differ.
1. **Calibrated penalty.** Each simulator s gets its own weight λ_s = ū_REL / ū_s, clipped to 0.05–2.
   - **ū** is the mean disagreement per step along the authors' policy, run closed-loop in that simulator's imagination: 1,024 robots, 256 steps, noise-free actions, from the simulator's start states (change 3). For the released model, start from the recorded data.
   - This gives the authors' walking policy the same uncertainty charge in every simulator as it pays in the released model at RWM-U's weight of 1.0.
   - λ_s comes from the simulator alone, before any policy is trained in it.
   - **Fallback.** If the authors' policy cannot run in imagination, use open-loop replay instead, and record which in `optc/DECISIONS.md`. The replay feeds the expert test set's joint commands through the simulator from the true 32-step history, for 100 steps. For the released model, it uses the recorded data's commands.
2. **Gate A2: can the simulator hold a good walk?** The authors' policy walks in MuJoCo without falling. In the same calibration run, the 6M seed-0 simulator must agree:
   - at most 10% of the 1,024 imagined episodes end in a predicted fall;
   - the two tracking terms together earn at least 80% of what the same policy earns per step in MuJoCo (`results/c_authors.json`, or a re-run on the expert test set's commands).

   **If it fails,** the simulator is the problem, and no policy setting can fix it. The pilot job then first retrains `sim_6M_N8_s0` and `sim_10k_N8_s0` with 2 × S steps, recomputes their checks and calibration, and applies Gate A2 again. If it passes, S becomes 2 × S for every simulator. If it fails again, stop BLOCKED.
3. **Start states** come only from the authors'-policy episodes (source 0, no added noise) in the simulator's own rung: upright, and at least 1 s before any fall. The lite code also starts imagination from the authors' recorded data.
4. **Noise floor.** Keep the policy's exploration noise at 0.05 or above: a clamp after each update, or the lite code's own log-scale option if it has one.
5. **Fall cost** (in two of the pilot's arms). A predicted fall costs a one-time 4.0, in the units of the per-step rewards: about 130 steps of perfect tracking. Use the lite reward's own termination term if it has one; otherwise subtract it at the step the episode ends. This departs from the authors' reward, and the paper must say so if it is used.
6. **Second pilot.** Four policies in the 6M simulator, seed 0, 1,500 iterations, saved at 500, 1,000 and 1,500; each save is measured with the full protocol.

   | Arm | Penalty | Fall cost |
   |---|---|---|
   | P1 | λ_6M | none |
   | P2 | λ_6M | 4.0 |
   | P3 | 0 | none |
   | P4 | 0 | 4.0 |

   - **Early stop.** After iteration 300, an arm whose mean imagined episode length stays under a quarter of the limit for 100 iterations in a row is stopped and recorded as "collapsed". Its saves so far are still measured.
   - For the record, also measure the first pilot's surviving saves: weight 2.0 at 500, 1,000 and 1,500, and weight 4.0 at 500 to 2,000.
7. **Gate B'.** Check P1, then P2, then P3, then P4.
   - An arm passes if its best save falls in under 20% of episodes and scores at least half the authors' policy.
   - The first to pass sets the main study's recipe for every simulator: its own λ_s or 0, with or without the fall cost.
   - If none passes, stop BLOCKED.
8. **Iteration rule'.** The main study uses the smallest saved count at which the chosen arm scores at least 90% of its best save.
9. **Gate C'** (not a stop). After Gate B', the job trains one policy in the 10k simulator with the chosen recipe and iteration count. It must score below the chosen arm's best save, with their 95% intervals over episodes apart. If not, add the 2k rung, as before.
10. **The main job.**
    - Every policy uses the chosen recipe, with the noise floor and the start-state rule above.
    - **D3' (descriptive).** Repeat D3 in `A5_s0` with the chosen recipe: two policies, 500 iterations, Step 2's protocol.
    - **D5 swaps the penalty:** λ = 0 if the main study uses one, λ_6M if it does not.
11. **Registration** declares the first pilot, this amendment and the second pilot as known.

<!-- BEGIN COMMON -->
## Common rules (every session)

### C1 What this round is
Option C finds why Step 2's rebuilt-simulator policies fell, then repeats the policy test with the right data.
- **The world** is your calibrated MuJoCo ANYmal D, which passed Step 2's Gate 1. It makes the data and it tests the policies.
- **The method:**
  - five-network simulators are trained on mixed MuJoCo data, at four sizes;
  - policies are trained inside them with RWM-U's uncertainty penalty;
  - each policy's walking is measured back in MuJoCo.
- **Order.** Sessions C0–C6 and four jobs run in order; yours is the block after these rules.
- **Earlier outcomes** are in `optc/LOG.md`, `optc/CHECKPOINT.md` and `optc/notes/`.
- **The fixed design.** These are fixed in `OPTIONC_PLAN.md` ("Design fixed by this plan"):
  - the diagnosis, data, pool and policies;
  - the walking score, gates, retry and iteration rules.

  Never change them. A needed change goes to `optc/DECISIONS.md` and stops the run BLOCKED.
- **Amendment 1** (end of the design) adds Gate A2 and replaces the penalty weight, the start states and the pilot rules. Where it differs from the design above it, the amendment wins.
- **Terms.** Keep these distinct:
  - **joint commands** are the policy's actions;
  - the **velocity command** is the speed and turn the policy is asked for;
  - a **transition** is one 0.02 s step of data.

### C2 Running unattended
- **Never ask the user anything.** When this plan does not settle a choice:
  - take the more conservative option;
  - record it in `optc/DECISIONS.md`;
  - go on.

  Stop BLOCKED when going on would break a rule or the fixed design, or when a gate fails as the plan says.
- **Never start** `claude`, any `go_*.zsh` launcher, `caffeinate` or a job yourself. Anything over about 10 minutes goes into the job file your block names. The launcher runs it after you finish.
- **On a resumed attempt,** first read `optc/CHECKPOINT.md`, `tail -20 optc/LOG.md`, your notes, and `git log --oneline -15`.
- **Before ending,** with any status: commit (C7), append your status line to `optc/LOG.md`, and overwrite `optc/CHECKPOINT.md` (C9).

### C3 What you may change
- **Write only inside this folder.** The launcher compares the rest of `~/Downloads/PDM` before and after every session.
- **Never open key files** (`../rwm_repro/sshkey*`), and don't read `../rwm_private/`.
- **Sibling folders are read-only.** Use their code through `sys.path`, or copy a file into `scripts/c/` with a header giving its source path and sha256.
- **Read-only here:**
  - every registration (`prereg/*`);
  - `scripts/ft_stats.py`, `scripts/s1_stats.py` and `scripts/s2_stats.py`;
  - Step 2's models, policies and results;
  - once C3 commits them, `prereg/PREREG_C.md` and `scripts/c_stats.py`.
- **No downloads, no installs, no sudo.**
- **No external actions:** never push, upload, e-mail, submit or post.

### C4 Integrity
- **Registration before results.** Until C3 commits `prereg/PREREG_C.md`:
  - no main-pool policy may be measured in MuJoCo;
  - no main-pool check may be set against any walking score.

  The diagnosis, Gate A and the pilot are declared in the registration as known.
- **Held-out episodes** never enter training, start states or the choice of S, weights or iterations.
- **No number is typed into a template.** Values reach the paper through `{{key}}`s, from result files written by scripts.
- **Registered verdicts** stand as registered. Label secondary, exploratory and descriptive results as such.
- **New result files only:** `results/c_*` and `results/c/`. Never overwrite an existing result file.
- **Never edit a shell script while it runs.**

### C5 Environment
- **Python:** `./py` for torch work; `./py-mj` (from Step 2) for MuJoCo.
- **Machine:** Intel i5, 4 logical cores, 8 GB of memory.
  - Sessions use at most 2 threads.
  - Jobs use 2 workers × 2 threads, or 1 × 4 if C1's probe says memory requires it.
  - Keep any one array under about 2 GB: data are float32 and memory-mapped.
- **Jobs:**
  - rewrite `results/c/progress.json` at least every 10 minutes, from the work itself (every 50 simulator steps or 25 policy iterations, at most);
  - are resumable, and skip finished work;
  - are stopped by the launcher if they make no progress for an hour.
- **Where things go:** simulators, policies and their logs under `runs/c/` (not committed); data shards in `optc/data/`.
- **Long output:** send it to `.cache/logs/`, and print only its tail.

### C6 Context and token discipline
- **How to read:** only what your block names; grep first; at most about 200 lines at a time. Print JSON fields with `./py -c`; at most about 40 lines of output.
- **Never read these whole:**
  - other rounds' plans;
  - `results/paper_numbers.*` and `results/raw_*.npz`;
  - data shards (print shapes only);
  - `history.bundle`;
  - logs (use `tail`).
- **Subagents:** only those your block names, and never a `model` parameter. Give each a self-contained brief.

### C7 Git
- **Branch.** Work on branch `optc`, which C0 creates from Step 2's last commit. Commit subjects are "C<n>: <what>".
- **Staging.** Stage only your paths. Never `git add -A`, push, checkout, restore, reset --hard, stash, clean or rebase.
- **Never commit:** `runs/`, `.cache/`, `.pylib*`, `optc/data/`, `step2/assets/`, `paper/build/`, `submission/` or `history.bundle`. C0 adds `optc/data/` and `.pylib-mj/` to `.gitignore`.
- **Tags:** only `air2027-optc-start` (C0) and `air2027-optc` (C6).

### C8 Style
- **Readers are roboticists.** Plain language; define terms; at most two numbers-with-intervals in a sentence.
- **Honesty.** State failures as plainly as successes, including Step 2's design mistake (single-network simulators).
- **Scope.** The world here is a hand-built MuJoCo model of ANYmal D, with a fitted PD motor. It is not Isaac Lab and not hardware; say so wherever a claim could be read more widely.
- **Notes:** at most 300 words unless the block says otherwise.

### C9 Ending a session
- **Status line.** Append `## C<n> status: COMPLETE (YYYY-MM-DD HH:MM)` to `optc/LOG.md`, with PARTIAL or BLOCKED in place of COMPLETE where needed.
  - Take the time from `date '+%F %H:%M'`, never by guessing.
  - Add at most 8 lines after it.
- **Checkpoint.** Overwrite `optc/CHECKPOINT.md` with the last session and status, the next action, and any blockers.
- **BLOCKED:** add to `optc/DECISIONS.md` what is blocked, the options, and your recommendation.
- **PARTIAL:** commit what is done, and write the exact next step in `optc/CHECKPOINT.md`.
<!-- END COMMON -->

<!-- BEGIN C0 -->
## C0 — Branch, facts and the diagnosis job (Opus 5.5, high)

**Read:**
- `step2/REPORT.md` and `results/s2_pilot.txt`;
- `step2/notes/M3.md` and `step2/notes/M4.md`;
- `step1/notes/S4_lite.md`;
- `ls scripts/s2/ scripts/lite/`.

**Steps**
1. **Git.**
   - Check that HEAD carries the tag `air2027-step2` (`git describe --tags --exact-match`); if not, record what is there.
   - Tag `air2027-optc-start`, then run `git switch -c optc` (on a resumed attempt: `git switch optc`).
   - Commit `OPTIONC_PLAN.md` and `go_c.zsh` alone: "C0: Option C plan (written before any Option C number)".
   - Create `optc/` with `LOG.md`, `DECISIONS.md`, `CHECKPOINT.md`, `notes/`, `jobs/`, `data/` and `review/`.
   - Add `optc/data/` and `.pylib-mj/` to `.gitignore` if absent, and commit that with the folder.
2. **Facts.**
   - Ask air-reader to report, with file and line:
     - **lite code (config and imagination):** robots; iterations; steps per robot per iteration; imagination length; the command range and resampling; the uncertainty weight's name and value; how an imagined episode ends on a fall (fall predictor, state limits, or both); how start states are loaded and chosen;
     - **the rebuild's trainer and loader** (`../rwm_repro/src/rwm_train.py` and `rwm_data.py`): batch size; learning rate; how the ensemble size and forecast length N are set; what the loader assumes about the data, such as the zero-fall check;
     - **Step 2's policy route** (`scripts/s2/train_policy.py --route rebuilt`): how it loads a model, sets start states, and saves.
   - **D0.** Count the rows with a fall and with any thigh contact in the recorded data, with `./py`.
   - Ask air-bib-checker, with quotes:
     - for RWM-U (arXiv 2504.16680) on ANYmal D: the uncertainty weight, imagination length, policy iterations, robots, ensemble size, and dataset sizes and composition;
     - for RWM (arXiv 2501.10100): the amount and source of the pretraining data.
   - Write `optc/notes/C0_facts.md` (at most 400 words).
3. **Write `scripts/c/diag.py`** for D1–D4, exactly as the design defines them.
   - D3 trains through `scripts/s2/train_policy.py --route rebuilt` with `A5_s0`, and is measured with `scripts/s2/evaluate.py` on Step 2's 80-episode protocol.
   - Outputs: `results/c_diag.json`, and `results/c_diag.txt` (at most 40 lines).
4. **Tests** (at most 10 minutes):
   - D1 runs for 10 steps on one policy;
   - D2 runs on 1,000 points;
   - D4 runs in full;
   - D3's route loads `A5_s0` and runs one iteration with 64 robots, with nonzero uncertainty.
5. **Write `optc/jobs/diag.zsh`,** with progress as in C5. It must be resumable and pass `zsh -n`.
6. Commit "C0: facts and diagnosis job".
<!-- END C0 -->

<!-- BEGIN C1 -->
## C1 — Root cause, data generator, trainer and the data job (Opus 5.5, high)

**Read:**
- `results/c_diag.txt` and `optc/notes/C0_facts.md`;
- `scripts/s2/mj_env.py` and `scripts/s2/evaluate.py` (grep first, then the parts you need);
- `../rwm_repro/src/rwm_data.py` and `rwm_train.py`, in parts (air-reader can summarise them).

**Steps**
1. **Root-cause note.** Write `optc/notes/C1_rootcause.md` (at most 400 words):
   - what D0–D4 showed;
   - which causes they support or weaken;
   - what stays open until D5 and D6.

   If D3's policies walk (falls under 20%), the penalty alone fixed it. Record this; the plan continues unchanged.
2. **Data generator, `scripts/c/gen_data.py`,** exactly as the design says:
   - sources and shares, commands, physics variety and pushes;
   - the 66 columns, plus episode id and source id;
   - shards in `optc/data/`, with `optc/data_manifest.json` (counts per source, seeds, sha256 sums);
   - the nested rungs, as lists of episode ids;
   - the two test sets and D6's episodes;
   - 2 workers; resumable.
3. **Trainer, `scripts/c/train_sim.py`.** Copy the rebuild's trainer and change only the loader: the window rule, rung normalisation, five members, and N = 8 or N = 1.
4. **Tests** (at most 10 minutes):
   - 20 s of the authors' policy from the generator equals Step 2's evaluator, with the same seed and commands, to 1e-6;
   - no training window crosses an episode end, and a fall appears only at a window's last step;
   - on the recorded data, `train_sim.py` reproduces the rebuild's first 20 losses to 1e-4, with the same seed and windows.
5. **Probes,** written to `results/c/probe.json`:
   - transitions per second per worker, with two workers running;
   - seconds per step for five-member training with N = 8 and with N = 1 on the 1M rung, two trainings at once;
   - peak memory.

   Set S by the design's rule and record it.
   If two trainings at once exceed 6 GB, the jobs train one simulator at a time with 4 threads. Record that too.
6. **Write `optc/jobs/data.zsh`:**
   - generate everything;
   - train the two pilot simulators (6M and 10k, rollout-trained, seed 0);
   - compute their checks on both test sets, into `results/c/pilot_sims.json`.
7. Commit "C1: root cause, data generator, trainer, data job".
<!-- END C1 -->

<!-- BEGIN C2 -->
## C2 — Gate A, walking score and the pilot job (Opus 5.5, high)

**Read:**
- `results/c/pilot_sims.json` and `results/c/probe.json`;
- the design's walking score, gates, retry and iteration rules;
- `scripts/s2/train_policy.py` and `scripts/s2/evaluate.py`, in parts.

**Steps**
1. **Gate A.** Apply it. If it fails, stop BLOCKED with a diagnosis in `optc/DECISIONS.md`.
2. **Walking score, `scripts/c/evaluate.py`,** built from Step 2's evaluator:
   - the score exactly as defined, with the 96-episode protocol, the secondary measures and the standing-still reference;
   - it refuses main-pool policies until `prereg/PREREG_C.md` is committed.
   - **Tests:** the authors' policy gives a finite score; a policy that holds the standing pose scores near the standing-still reference.
   - Measure the authors' policy into `results/c_authors.json`. This is Gate B's reference.
3. **Policy wrapper, `scripts/c/train_policy.py`.** It wraps Step 2's route, adding the start-state rule and an uncertainty-weight flag, and saving at 500, 1,000, 1,500 and 2,000 iterations.
   - **Tests:**
     - one iteration with 64 robots on the 10k pilot simulator, with nonzero uncertainty;
     - through the wrapper, the 6M pilot simulator's one-step predictions on 100 windows from its rung equal `train_sim.py`'s own, to 1e-6. This catches a mismatch in normalisation or loading.
4. **Write `optc/jobs/pilot.zsh`:**
   - train the two pilot policies on 2 workers, and measure every save;
   - apply Gate B and the retry rule mechanically, with `scripts/c/pilot_rule.py`;
   - if the rule calls for it, train and measure the two retry policies;
   - write `results/c_pilot.json`, and `results/c_pilot.txt` (at most 40 lines).
5. Commit "C2: Gate A, walking score, pilot job".
<!-- END C2 -->

<!-- BEGIN C2R -->
## C2R — Amendment 1: calibration, fixes and the second pilot's job (Opus 5.5, high)

The first pilot failed; read "Amendment 1" in the design before anything else.

**Read:**
- Amendment 1;
- `results/c/pilot_decision.json` and `optc/CHECKPOINT.md`;
- `tail -40 .cache/logs/job_c_pilot.log`;
- `scripts/c/train_policy.py`, `scripts/c/pilot_rule.py`, `scripts/c/diag.py` (its imagination part) and `optc/jobs/pilot.zsh`, in parts.

**Steps**
1. **Commit the amendment.** Commit `OPTIONC_PLAN.md` and `go_c.zsh` (changed by the author for Amendment 1) alone: "C2R: Amendment 1 (after the first pilot)".
   - Also commit the diagnosis results that C1 left uncommitted.
   - Move `.cache/logs/job_c_pilot.log` to `.cache/logs/job_c_pilot_first.log`.
2. **The first pilot's record.** Write `scripts/c/pilot1_record.py`, which writes `results/c_pilot1.json` and `results/c_pilot1.txt` (at most 40 lines). Never overwrite `results/c/pilot_decision.json`.
   - Gate B at weight 1.0;
   - per policy, from the lite metrics: imagined episode length, mean disagreement, and each reward term, at iterations 0, 100, 500, 1,000 and the end;
   - the same for `REL__p0` and D3's policies, for comparison;
   - the first pilot's surviving saves are measured by the job (step 7) and added there.
3. **Check the amendment's reading.** With `./py`, confirm from the lite metrics and code, and record in `optc/notes/C2R.md`:
   - how the logged reward terms are scaled (per step, per second or per episode);
   - whether the lite reward has any cost for a predicted fall;
   - the share of first-pilot start states by data source.

   If a fact contradicts Amendment 1's account, record it in `optc/DECISIONS.md`. Stop BLOCKED only if it makes a change in the amendment impossible.
4. **Calibration and Gate A2, `scripts/c/calibrate_penalty.py`.**
   - Compute ū and λ_s exactly as Amendment 1 defines them, using the lite code's own disagreement measure, for `REL`, `sim_6M_N8_s0`, `sim_10k_N8_s0` and `A5_s0`.
   - For each, also report the authors' policy's imagined reward per step by term, and its share of predicted falls.
   - Apply Gate A2 to `sim_6M_N8_s0`. Write `results/c/calibration.json` and commit it before the job runs.
   - If this takes more than 10 minutes, it moves into the job instead.
5. **Fixes in `scripts/c/train_policy.py`:**
   - the noise floor;
   - the start-state rule (source 0 only);
   - a `--fall-cost` flag;
   - the early stop for collapsed arms;
   - a `--max-iters` flag (500 to 2,000) with saves every 500;
   - λ from `results/c/calibration.json` by simulator name, or an explicit value.

   **Tests:**
   - one iteration with 64 robots in each pilot simulator, with nonzero uncertainty;
   - the noise floor holds when the noise is forced below it;
   - the start states are all from source 0;
   - with the fall cost on, an episode that ends in a predicted fall is charged 4.0 exactly once;
   - the early stop fires on a fixture, and not on `REL__p0`'s recorded metrics.
6. **Rule, `scripts/c/pilot2_rule.py`.** It implements Gate B', the iteration rule' and Gate C', writes `results/c/pilot2_decision.json`, and writes `results/c_pilot.json` and `.txt` (at most 40 lines). Test every branch on fixtures. A BLOCKED result is recorded in `optc/DECISIONS.md` and `optc/LOG.md`, and the job exits 3.
7. **Write a new `optc/jobs/pilot.zsh`:**
   - if Gate A2 failed: retrain the two pilot simulators with 2 × S steps, redo their checks and calibration, and apply Gate A2 again (exit 3 if it fails);
   - P1 and P2 on two workers, then P3 and P4;
   - measure every save as soon as its policy finishes;
   - measure the first pilot's surviving saves, and add them to `results/c_pilot1.*`;
   - apply Gate B' and the iteration rule', then train and measure the 10k policy for Gate C';
   - run `pilot2_rule.py`.

   It is resumable and skips finished work, with progress as in C5, and must pass `zsh -n`.
8. **Record.** Write `optc/DECISIONS.md` entries for every choice. Commit "C2R: calibration, fixes, second pilot job".
<!-- END C2R -->

<!-- BEGIN C3 -->
## C3 — Gates B and C, registration and the main job (Opus 5.5, high)

**Read:**
- `results/c_pilot.txt`, `results/c_pilot1.txt`, `results/c/calibration.json`, `results/c_diag.txt` and `optc/notes/C1_rootcause.md`;
- the design in `OPTIONC_PLAN.md`, including Amendment 1;
- in `scripts/stats.py`, `spearman`, `_clean` and `boot_diff`. Import these; never copy them.

**Steps**
1. **Gates.** Use Amendment 1's rules, which `results/c/pilot2_decision.json` already applies.
   - **Gate B':** if no arm passed, stop BLOCKED with the options.
   - Record the chosen recipe (penalty rule and fall cost), S, the iteration count and Gate C'.
2. **Write `prereg/PREREG_C.md`:**
   - the question;
   - the pool, with any Gate C rung;
   - the settings: S, N, iterations, the penalty rule with its calibration procedure, the fall cost if used, the noise floor, robots and start states;
   - the protocol, the walking score, and the checks on both test sets.
   - **Statistics:** the unit is the simulator; Spearman correlation with every check turned higher-is-better; paired bootstrap over simulators, 20,000 draws, `default_rng(0)`.
   - **Premise (95%). R0:** ρ(−E100 on the expert test set, walking score). SUPPORTED, CONTRADICTED or NOT SETTLED.
   - **Primary family (97.5% each), on the expert test set, with the labels used in Step 2:**
     - **R1:** ρ(−λ1, score) − ρ(−E1, score);
     - **R2:** ρ(−r̄J, score) − ρ(−E8, score).
   - **Secondary (95%):**
     - R0–R2 on the mixed test set;
     - E100 on the mixed against the expert test set;
     - ensemble disagreement along policy rollouts against the walking score;
     - the imagined return against the walking score;
     - selection of the best three simulators by each check;
     - fall rate as the outcome.
   - **Descriptive:** the walking score against the data rung, D5, D6 and the root cause.
   - **What was known:** the diagnosis, Gates A and A2, the first pilot, Amendment 1 and the second pilot.
   - **Expected outcome:** with about 14 simulators, the differences between checks may end NOT SETTLED. Say so here. The premise test and the data ladder are informative either way.
3. **Write `scripts/c_stats.py`, `scripts/c_main.py` and tests** with synthetic results.
   - `c_main.py` trains the remaining simulators, including D6's, and makes the pruned copies.
   - It computes all checks, and each simulator's λ_s with `calibrate_penalty.py` before its policies.
   - It then trains and measures the 28 pool policies, the four D5 and D6 policies, and D3's two policies.
   - It is resumable, and uses the worker layout C1 recorded.
4. **Review.** Ask air-stats-reviewer to review all of these against this block, in at most 400 words. Fix every MUST FIX and SHOULD FIX.
5. **Register.**
   1. Commit them together: "C3: PRE-REGISTER Option C (no main-pool policy measured)".
   2. Write `results/prereg_hash_c.json` and commit it.
   3. Run `chmod a-w prereg/PREREG_C.md scripts/c_stats.py`.
6. **Write `optc/jobs/main.zsh`,** and commit "C3: main job".
<!-- END C3 -->

<!-- BEGIN C4 -->
## C4 — Analysis and the paper (Opus 5.5, high)

**Read:**
- `prereg/PREREG_C.md` and `optc/notes/C1_rootcause.md`;
- `tail -30 .cache/logs/job_c_main.log`;
- in `paper/body.template.tex`, by grep: the Step 2 paragraph, the abstract's scope sentence, Limitations and the Conclusion;
- the appendix headings.

**Steps**
1. **Run `./py scripts/c_stats.py`.** It is registered; do not edit it. Its verdicts are the result, whatever they are.
2. **Numbers.** Write `scripts/c_numbers.py` (the `add(...)` pattern), and call it from `paper_numbers.py`.
3. **Main text.** Replace the Step 2 paragraph with one paragraph and one small table:
   - why the first test failed, in one sentence (the root cause);
   - the walking scores along the data ladder;
   - R0–R2.

   Change the abstract's scope sentence, Limitations and the Conclusion only where they are now false. The abstract stays at most 200 words.
4. **Appendix:**
   - move Step 2 into an appendix section, with the root cause: D0–D6, D3', the comparison table, the first pilot and Amendment 1;
   - add a section "Learned simulators of a known physics model", covering the data, simulators, policies and all results.
5. **Fit.** `slack.py` must print PASS (reserve 0).
6. **Finish.** Write `optc/notes/C4.md` with the new keys and claims, and commit "C4: Option C in the paper".
<!-- END C4 -->

<!-- BEGIN C5 -->
## C5 — Claims, checks, supplement and documents (Sonnet 5.5, high)

**Read:**
- `optc/notes/C4.md`;
- the start of `scripts/check_claims.py`, and its claim descriptions (grep).

**Steps**
1. **Claims.** Add an assertion for every new claim. All must pass.
2. **Full pipeline,** every step passing:
   - `paper_numbers.py`;
   - `build_paper.py`, twice, giving identical text;
   - `slack.py`, `check_claims.py`, `check_numbers.py`, `check_bib.py` and `submission_checks.py`.
3. **Check against the evidence.** Ask air-claim-checker to check the new paragraph, the table and the appendix sections against `results/c_*` and `prereg/PREREG_C.md`. Fix each finding minimally.
4. **Supplement.**
   - Add `scripts/c/`, `prereg/PREREG_C.md`, the `c` results and the data manifest.
   - Leave the data out: the generator recreates it from its seeds.
   - Rebuild the supplement and scan it. The scan must find 0 hits.
5. **Documents.** Update `README.md`, `SUBMIT.md` and `ANOMALIES.md`.
6. Commit "C5: checks, supplement, documents".
<!-- END C5 -->

<!-- BEGIN C6 -->
## C6 — Fresh-eyes review, report and tag (Opus 5.5, high)

**Read:** `optc/notes/C4.md`, `prereg/PREREG_C.md` and `results/c_stats.txt`.

**Steps**
1. **Extract the main text** of the two-column check to `optc/review/body.txt`.
2. **Get two reviews, in parallel.**
   - air-referee reviews `body.txt`.
   - air-stats-reviewer checks `results/c_stats.json` against `prereg/PREREG_C.md`, and the new sentences against both.
3. **Act on the reviews.** Fix the text, with no new analysis. Decline anything else with a reason, under "C6 declined".
4. **Rerun the pipeline.** Everything must pass.
5. **Write `optc/REPORT.md`** (at most 500 words, plain language):
   - the root cause;
   - how much data walking needed;
   - the verdicts;
   - what the MuJoCo world can and cannot show;
   - the referee's score;
   - what the author must check.
6. **Close the round.**
   1. Commit "C6: Option C complete", and tag it `air2027-optc`.
   2. Run `git bundle create history.bundle --all` and `git bundle verify history.bundle`.
   3. Write in `optc/CHECKPOINT.md`: "Option C complete. Next: read optc/REPORT.md; submit by 31 Dec."
<!-- END C6 -->

## Sources behind the diagnosis
- **Step 2:** `results/s2_pilot.txt` and `step2/REPORT.md` (falls, imagined returns, two feet down).
- **The recorded data's fall column:** `../rwm_repro/src/rwm_data.py`, which asserts it is all zero.
- **Step 1's warning:** `step1/notes/S4_lite.md` and `step1/LOG.md` ("use 5-head rebuilds (A5_*) so the uncertainty penalty is active").
- **The reward:** `step2/notes/M0_facts.md`. The uncertainty term's weight is −1.0, and the lite default is 8,192 robots (`step1/notes/S4_lite.md`).
- **RWM** (arXiv 2501.10100): 6M pretraining transitions from policies trained under varied dynamics; "the policy often exploits minor world model errors".
- **RWM-U** (arXiv 2504.16680):
  - the uncertainty penalty with weight 1.0 is "well-calibrated";
  - at 0.5 or less, policies "attain high rewards in imagination but exploit model inaccuracies";
  - mixed data beats expert-only data, which overfits;
  - its datasets are about 1M transitions.

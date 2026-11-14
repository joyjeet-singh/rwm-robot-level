# AIR 2027 — Option D, early-failure form (unattended plan)

Option D replaces what is left of Option C. Option C stays on its branch (`optc`), and nothing is deleted. This version replaces the Option D plan of 13 November, which had not been started.

## What this version changes

The earlier Option D plan built 26 adapted simulators at fixed amounts of adaptation, and trained two policies in every one, whatever happened. That would have taken about five days, and much of the time would have gone to simulators far from the point where policies start to fall.

This version reaches the same question in about three days. It is built on five rules:
1. **Record first.** The results we already have are written into result files on your Mac before anything new is trained (`RESULTS_SO_FAR.md` lists them).
2. **Cheapest decisive test first.** Nothing long starts until a short test has passed.
3. **Every costly step has a stop rule,** written here before the run.
4. **A failure is established once, then left alone.** A simulator's verdict comes from two policies, or three when the two disagree, so that one unlucky policy cannot decide it. After that: no further try, no diagnosis, no deeper version of the same thing.
5. **Search, don't sweep.** Each family of simulators gets three tests placed around the point where it starts to fail, not a fixed list of step counts.

**Revised on 14 November: policies per simulator, and all seven runs.** Two changes from the first draft of this version:
- **Every tested simulator gets two policies, and a third when those two disagree.** The first draft gave a simulator whose first policy failed no second policy.
- **All seven runs of the list may be done** (the first draft: six), and no run starts after 56 hours of job time (the first draft: 30). This was your choice.

Why two policies, and a third after a split (`optd/REVIEW.md` section 13 has the evidence):
- The search places most tests near the point where policies start to fall. There, one policy is close to a coin toss, and each verdict steers the next test.
- A third policy only after a split gives the same verdict as three policies for every simulator, at about three-quarters of the cost.
- In Step 2, each of the nine simulators' two policies agreed on failed or not, but those simulators lie far from the boundary. The two policies of the walking copy nearest to it scored 0.25 and 0.53.
- In a simulated version of the ladder, this rule settled the registered tests more often than one policy for a failed simulator did, for about 12 to 15 hours more. Beyond three policies, each extra policy helped less than the same hours spent on more simulators.

**Where a failure is caught, and what it then costs:**

| What can fail | When it is caught | What happens | Time lost |
|---|---|---|---|
| The code | by tests and a short rehearsal of the whole chain, before each job | the session fixes it, or stops after three failed fixes | minutes |
| A fine-tuning run whose loss is not finite | at that step | the run ends; what it has saved is kept | minutes |
| A policy that cannot even walk in imagination | at about iteration 100 of 500 | it is stopped, measured and counted as failed | about 30 minutes, not 2 hours |
| A simulator whose policies learn in imagination but fall in MuJoCo | after its two policies, or three when they disagree | it is recorded as failed, and the search moves on | about 4 hours; 6 with a third policy |
| A family with no boundary to find (all three of its simulators fail, or none does) | after its three tests | its repeat run is skipped | about 15 hours saved |
| A job that runs long | at its cap of 56 hours | no new run starts | bounded |
| A session that keeps failing | at its third failed fix | BLOCKED with a short note; the launcher tries a session at most three times | bounded |
| A crash that is the code's fault, not the simulator's | at once: the same code is run on the released model for a few iterations | the job stops with an error and resumes after the fix; nothing is recorded as failed | minutes |

**What the review behind this plan found** (details: `optd/REVIEW.md`; the full record: `RESULTS_SO_FAR.md`):
- Option C stopped because our from-scratch simulators could not hold up RWM's released policy. Two mistakes in my design caused that: the scaling of the states, and far too little training for this Mac.
- The released model has no such problem, and its training can be continued on MuJoCo data ("fine-tuned") in minutes.
- Doing so halved its prediction error, and the policy trained in the adapted model then fell in 86% of MuJoCo episodes. The released model's policy, trained the same way, walks.
- So lower error gave a worse simulator for training. That is the paper's question, and one pair of simulators hints that the paper's long-horizon checks saw it while the short-horizon errors did not.

**What Option D does.**
1. **The record** (about 3 hours). It measures the walking score of Step 2's 22 existing policies, and the checks of the 17 simulators we already have, into result files.
2. **The ladder** (about two and a half days). It adapts the released model by different amounts, in up to seven runs. In each run it tests three simulators, chosen one after another so that they close in on the amount of adaptation at which policies start to fall. Each tested simulator gets two policies, and a third when those two disagree.
3. **The registered question:** which checks, computed before any policy is trained, predicted which simulators give walking policies?
4. **The paper** (about half a day).

**For the author — how to run it.** In a terminal (the app's terminal or Terminal.app):

```
cd ~/Downloads/PDM/"AIR 2027"
chmod +x go_d.zsh
caffeinate -i ./go_d.zsh auto     # every session and job in order; safe to re-run after a stop
./go_d.zsh status                 # progress and the latest results, from another terminal
```

You can keep using the Mac, as before. If a session stops because something outside AIR 2027 changed and the change was yours, run `auto` again.

**Time.** About three days, and at most four, nearly all unattended, at no cost. The record is complete about 5 hours after the start. The first new result (does the first adapted simulator's first policy walk on your Mac?) arrives about 9 hours after the start, and that simulator's verdict, from two policies, about 11 hours after the start.

**If it stops BLOCKED.**
1. Read `optd/DECISIONS.md`.
2. Write your answer there.
3. Run `./go_d.zsh D<n> resume`, then `auto` again.

**Two choices you can change before starting,** by editing the design below:
- **Policies per simulator: two, and a third when they disagree.** Changing this back to "one, and a second only for a simulator that did not fail" saves roughly 15 to 20 hours, but lets a single policy steer the search wherever it lands on the wrong side of 0.15. Three policies for every simulator would cost roughly 15 to 20 hours more than this rule and change no verdict; it would only make each simulator's mean score a little steadier.
- **All seven runs, and no run starts after 56 hours.** The two workers each do whole runs of about 16 hours. When both repeat runs (`W_s1`, `M_s1`) take place, there are seven runs to do: the seventh starts at about 48 hours and adds about 16. Otherwise there are at most six, and the ladder ends about 16 hours sooner. The cap of 56 hours leaves room for the seventh run to start even if the first six run a little slow. Changing this back to "at most six runs, and no run starts after 40 hours" saves those 16 hours; in the simulated ladder, the seventh run gained about as much as a third policy for every simulator would.

**How work is split.**

| Step | Model | Effort | Max turns | Work |
|---|---|---|---|---|
| D0 | Opus 5.5 | high | 200 | branch; facts; code for the record and for fine-tuning; tests; the record job |
| job record | — | — | — | walking scores of 22 existing policies; checks of 17 existing simulators; about 3 h |
| D1 | Opus 5.5 | high | 200 | policy wrapper; verdict rule, search rule and driver; statistics; registration; rehearsal; the ladder job |
| job ladder | — | — | — | up to seven runs of three simulators each, two or three policies per simulator; about two and a half days, capped |
| D2 | Opus 5.5 | high | 150 | registered analysis; the paper |
| D3 | Sonnet 5.5 | high | 120 | claim assertions, all checks, supplement, documents |
| D4 | Opus 5.5 | high | 100 | fresh-eyes review, fixes, report, tag |

| Subagent | Model | Used by |
|---|---|---|
| air-reader (read-only files) | Sonnet 5.5 | D0 |
| air-stats-reviewer | Opus 5.5 | D1, D4 |
| air-claim-checker | Sonnet 5.5 | D3 |
| air-referee | Opus 5.5 | D4 |

## Design fixed by this plan (committed by D0 before any Option D number exists on this Mac)

### Terms
- **Released model:** RWM's own five-network simulator, `pretrain_rnn_ens.pt`, called `REL`.
- **Fine-tune:** continue training a model from its current weights on new data. A **step** is one update on one batch of 256 windows.
- **Authors' scales:** the fixed mean and spread per state in the lite config (`anymal_d_flat_cfg.py`, `state_data_mean` and `state_data_std`).
- **Step 2's recipe:** the authors' lite policy training as Step 2 ran it (see Policies).
- **Walking score:** Option C's measure of a policy in MuJoCo. RWM's released policy scores 0.942; standing still scores 0.069.
- **Failed** (of a policy): its walking score is below 0.15, or it was stopped for collapsing in imagination, or its training failed by its simulator (see Policies).
- **Failed** (of a simulator): at least two of its policies failed. This is the simulator's **verdict**.

### Kept, unchanged
- **The world:** the calibrated MuJoCo ANYmal D of Step 2.
- **The walking score** and its 96-episode protocol (`scripts/c/evaluate.py`).
- **Step 2's simulators and policies:** `REL` and its eight compressed copies with two policies each; and the pilot's four policies.
- **Option C's data and test sets** in `optc/data/`: the rung `rung_1M`, and the test sets `test_expert.npy` and `test_mixed.npy`.
- **Option C's simulators:** the four in `runs/c/`, and `A5_s0`. They get checks, but no new policies.
- **Code:** `scripts/c/`, `scripts/s2/` and `scripts/lite/`. Import them; change them only by wrapping.
- **Records:** Option C's sessions, its two pilots and Gate A2 stay as they are, and the paper reports them.

### Stage 1: the record
Measured for what already exists, with nothing trained:
- **Walking score, fall rate and Step 2's physics return** of 22 policies: Step 2's 18 registered ones and its 4 pilot ones.
- **Checks and the walk test** (both defined below) of 17 simulators: `REL`; its eight compressed copies; Step 2's three single-network rebuilds (`A_s0`, `B_s0`, `A_s0__u50`); `A5_s0`; Option C's four.
- **One table,** `results/d_record.txt` with `results/d_record.json`: a row per simulator with how it was made, its checks, its walk test, and how each policy trained in it did. Option C's pilot policies enter from its own result files.

### Stage 2: the ladder
**Runs.** A run is one fine-tuning of the released model. Every run keeps the authors' scales and all five networks.

| Order | Run | Data | Learning rate | Windows | Seed | Starts only if |
|---|---|---|---|---|---|---|
| 1 | `W_s0` | walking | 1e-4 | rollout | 0 | always |
| 2 | `M_s0` | mixed | 1e-4 | rollout | 0 | always |
| 3 | `Wtf_s0` | walking | 1e-4 | teacher-forced | 0 | always |
| 4 | `W_s1` | walking | 1e-4 | rollout | 1 | `W_s0` found a boundary |
| 5 | `M_s1` | mixed | 1e-4 | rollout | 1 | `M_s0` found a boundary |
| 6 | `Wg_s0` | walking | 1e-5 | rollout | 0 | always |
| 7 | `Mg_s0` | mixed | 1e-5 | rollout | 0 | always |

- **Mixed data:** every row of Option C's rung `rung_1M`. 45% of it comes from weaker policies, heavy action noise and random motion, with many falls.
- **Walking data:** the rows of `rung_1M` whose source is 0, 1 or 2: RWM's released policy without action noise, and with noise 0.1 and 0.3.
- **Rollout windows:** 32 + 8 rows; the model predicts 8 steps from its own outputs. **Teacher-forced:** 32 + 1 rows; it predicts one step from true data.
- **A run found a boundary** when at least one of its tested simulators failed and at least one did not.
- **Limits.** All seven runs of the list may be done. No run starts after 56 hours of job time. A run that has started finishes its three tests. A run that does not start is recorded with the reason.

**Simulators.** A run is saved when it reaches 5, 20, 40, 80, 160, 320 and 1,280 steps. Each save is a simulator, named `<run>_d<steps>`, in `runs/d/<simulator>/final.pt`.

**The search: three tests per run.** A test of a simulator is: compute its checks; train and measure its policies (below); give its verdict.
1. Test the simulator at 80 steps.
2. If it failed, test 20 steps. Otherwise test 320 steps.
3. The third test depends on the first two:

| 80 steps | Second test | Third test |
|---|---|---|
| failed | 20 failed | 5 |
| failed | 20 did not fail | 40 |
| did not fail | 320 failed | 160 |
| did not fail | 320 did not fail | 1,280 |

A run is fine-tuned only as far as its next test needs.

**Policies per tested simulator: two, and a third when they disagree.**
- The run's worker trains the policies with seeds 1 and 2, one after the other, and measures each as soon as it is trained.
- If exactly one of the two failed, it trains a third, with seed 3.
- The verdict is the majority: the simulator failed when at least two of its policies failed. The search waits for it.
- Every policy counts: a failed policy is measured and kept like any other, and enters its simulator's mean score.

**Fine-tuning.**
- **Start:** the released model's weights, all of them. **Scales:** the authors', never the data's own.
- **Loss and optimiser:** the rebuild's trainer, which equals the authors' loss; Adam, weight decay 1e-5, batch 256, no gradient clipping. Everything is trained.
- **Windows:** inside one episode, with a fall only at the last row (the lite loader's rule).
- **Seed:** sets the order of the batches.
- **One fine-tuning at a time,** on 2 threads.
- **A loss that is not finite** ends the run. Its tested simulators stay in the pool.

**Policies: Step 2's recipe, with one stop rule.**
- **Code:** `scripts/lite/run_lite.py`, as `scripts/s2/train_policy.py` calls it with `--route rebuilt`.
- **Settings:** 500 iterations; 4,096 robots; start states from the authors' recorded data with no episode held out, as for `REL`; the lite config's uncertainty weight; 2 threads. No noise floor, no fall cost.
- **Seeds:** 1 and 2, as in Step 2; 3 for a third policy.
- **The stop rule (collapse in imagination).** From iteration 50 on, training stops once the mean imagined episode length (`Train/mean_episode_length`) has stayed under 128 steps, half the limit, for 50 iterations in a row. The earliest stop is at iteration 100.
- **A stopped policy** is measured like any other, from its last weights, and counts as failed.
- **A training run that fails by its simulator** in another way (a loss or reward that is not finite, or the same crash inside the lite code twice) gets a walking score of 0 and a fall rate of 1, as Step 2's rule says, and counts as failed.
- **The crash guard.** Before a crash is counted as the simulator's, the same code is run on `REL` for 3 iterations. If that crashes too, the fault is the code's: the job stops with an error, and nothing is recorded as failed.
- **No other early stop.** A policy that still trains in imagination runs its 500 iterations. Stopping it on an early MuJoCo score would misjudge slow learners: in the review the released model's own policy scored 0.08 at 200 iterations and 0.64 at 500.

**Workers.** Two workers, each taking the next run in the order above and doing its tests, and each test's policies, one after another. Policy training and fine-tuning use 2 threads each.

### Outcome and pools
- **Outcome:** the walking score. The unit is the simulator: the mean over its two or three policies.
- **Primary pool:** `REL`, with its two registered policies (seeds 1 and 2), and every ladder simulator that was tested (at most 21).
- **Pooled:** the primary pool and Step 2's eight compressed copies, with the two policies each that Step 2 trained.
- **Descriptive only:** Option C's four simulators, `A5_s0`, and Step 2's three single-network rebuilds.

### Checks, in common units
Computed for a ladder simulator before its first policy is trained, on the expert test set (primary) and the mixed test set (secondary):
- **Windows and rollout:** the paper's 400-row windows, and the paper's rollout from the mean of the five networks.
- **Common units.** Every prediction is converted back to physical units and then divided by the authors' scales. For the ladder and for Step 2's simulators this is the paper's own error. For Option C's four simulators it puts their errors in the same units.
- **Errors:** E1, E8, E24, E100 and E368, with the hold-the-last-state floor in the same units.
- **Kinematic residual r̄J** (and the attitude residual r̄A, exploratory): scaled by the residuals of the expert test set's own true rows.
- **Growth rate λ1:** the paper's function and settings, in the simulator's own coordinates.

**The walk test,** a closed-loop check: RWM's released policy run inside the simulator's imagination, as Option C's calibration ran it (noise-free actions, 1,024 robots, 256 steps, seed 20278, start states from the authors' recorded data). It gives three numbers: the share of imagined episodes that end in a fall, the tracking reward per step, and ū, the mean disagreement between the five networks.

### Sanity gates (the only places a job stops BLOCKED)
- **In the record job:** RWM's released policy scores what `results/c_authors.json` says, to 1e-6; and `REL`'s two registered policies score at least 0.3 on average. The review measured 0.605 and 0.574 on a copy.
- **In the ladder job:** neither `W_s0` nor `M_s0` can produce a simulator at 80 steps with finite weights.

### The registered analysis (D1 writes it out in full)
- **Statistics:** Spearman correlation across simulators, every check turned so that higher is better; paired bootstrap over simulators, 20,000 draws, `default_rng(0)`; Step 2's labels and decision rule.
- **Premise (95%). R0:** ρ(−E100, score), expert test set, primary pool.
- **Primary family (97.5% each), expert test set, primary pool:**
  - **R1:** ρ(−λ1, score) − ρ(−E1, score);
  - **R2:** ρ(−r̄J, score) − ρ(−E8, score).
- **Secondary (95%):**
  - R0–R2 on the pooled pool, and on the mixed test set;
  - ρ(−E368, score) − ρ(−E8, score);
  - the walk test: ρ(tracking, score), ρ(−falls, score) and ρ(−ū, score), each also minus ρ(−E8, score);
  - R1 and R2 on `REL` with the seed-0 runs alone;
  - R0–R2 with each simulator's seed-1 policy alone, to show how much the extra policies change the answer;
  - "failed" as the outcome: how well each check separates failed simulators from the others, as the area under the ROC curve;
  - fall rate as the outcome, and Step 2's physics return as the outcome;
  - the best three simulators chosen by each check.
- **Descriptive:** the ladder table (per run: each tested step count with its checks, its verdict, and each policy's score and fall rate); how often a test needed a third policy; the descriptive simulators.

<!-- BEGIN COMMON -->
## Common rules (every session)

### Rule 1: what this round is
Option D tests the paper's checks on simulators adapted from the released model, in early-failure form.
- **The world** is the calibrated MuJoCo ANYmal D of Step 2. It made the data and it tests the policies.
- **Stage 1, the record,** measures what already exists. **Stage 2, the ladder,** adapts the released model in up to seven runs and tests three simulators per run, each with two policies, or three when the two disagree.
- **The policies** use Step 2's recipe with one added stop rule, so Step 2's policies can be pooled with the new ones.
- **Order.** Sessions D0–D4 and two jobs run in order; yours is the block after these rules.
- **Earlier outcomes** are in `optd/LOG.md`, `optd/CHECKPOINT.md` and `optd/notes/`. Option C's are in `optc/`.
- **The fixed design** is in `OPTIOND_PLAN.md` under "Design fixed by this plan". Never change it. A needed change goes to `optd/DECISIONS.md` and stops the run BLOCKED.
- **`optd/REVIEW.md` and `RESULTS_SO_FAR.md`** hold what was known before this round. Numbers marked "cloud only" there are context. Re-measure on this Mac before using any of them.
- **Terms.** Keep these distinct:
  - **joint commands** are the policy's actions;
  - the **velocity command** is the speed and turn the policy is asked for;
  - a **transition** is one 0.02 s step of data.

### Rule 2: failing early
- **A failed simulator is a result.** Record it and go on. Never tune, retrain, repair or re-test a simulator because its policies fall, and never investigate why they fall. The design's second and third policies are part of a test, not a re-test; train no others.
- **Three strikes.** If the same test or step still fails after three different fixes, stop BLOCKED with a note of at most 10 lines. Do not try a fourth.
- **Rehearse before a job.** A session that writes a job first runs the whole chain of that job at toy size (minutes), and writes the job only when the rehearsal passes.
- **Secondary parts never block.** If a secondary check or analysis cannot be made to pass its tests, record that in `optd/DECISIONS.md`, leave it out, and go on. The primary path is: the walking score; fine-tuning; the checks E1, E8, E100, λ1 and r̄J; the policies; the search; and R0–R2.
- **Stop BLOCKED** only when going on would break a rule or the fixed design, at a sanity gate, or on the third strike.

### Rule 3: running unattended
- **Never ask the user anything.** When this plan does not settle a choice, take the more conservative option, record it in `optd/DECISIONS.md`, and go on.
- **Never start** `claude`, any `go_*.zsh` launcher, `caffeinate` or a job yourself. Anything over about 10 minutes goes into the job file your block names. The launcher runs it after you finish.
- **A job that stops at a sanity gate** records the reason in `optd/DECISIONS.md` and `optd/LOG.md`, and exits with code 3. Any other failure exits with another code.
- **On a resumed attempt,** first read `optd/CHECKPOINT.md`, `tail -20 optd/LOG.md`, your notes, and `git log --oneline -15`.
- **Before ending,** with any status: commit (Rule 8), append your status line to `optd/LOG.md`, and overwrite `optd/CHECKPOINT.md` (Rule 10).

### Rule 4: what you may change
- **Write only inside this folder.** The launcher compares the rest of `~/Downloads/PDM` before and after every session.
- **Never open key files** (`../rwm_repro/sshkey*`), and don't read `../rwm_private/`.
- **Sibling folders are read-only.** Use their code through `sys.path`, or copy a file into `scripts/d/` with a header giving its source path and sha256.
- **Read-only here:**
  - every registration (`prereg/*`);
  - `scripts/ft_stats.py`, `scripts/s1_stats.py` and `scripts/s2_stats.py`;
  - Step 2's and Option C's models, policies, data and results;
  - `scripts/c/`, `scripts/s2/` and `scripts/lite/` (import or wrap them; new code goes in `scripts/d/`);
  - once D1 commits them, `prereg/PREREG_D.md` and `scripts/d_stats.py`.
- **No downloads, no installs, no sudo.** Never `pip install` anything: not in the sibling folders, and not into `../.venv-rwm311`.
- **No external actions:** never push, upload, e-mail, submit or post.

### Rule 5: integrity
- **Registration before results.** Until D1 commits `prereg/PREREG_D.md`, no policy may be trained in a ladder simulator, except test runs of at most 3 iterations with `_test*` names whose actors are deleted.
- **Known before registration,** and declared in it: Step 2's results; Option C's records; `optd/REVIEW.md`; and everything the record job measures.
- **Held-out episodes** never enter fine-tuning or any choice.
- **The search uses only the walking scores** of a simulator's own policies, through its verdict. No check may decide which simulator is tested or stopped, or whether a third policy is trained.
- **No number is typed into a template.** Values reach the paper through `{{key}}`s, from result files written by scripts.
- **Registered verdicts** stand as registered. Label secondary, exploratory and descriptive results as such.
- **New result files only:** `results/d_*` and `results/d/`. Never overwrite an existing result file.
- **Never edit a shell script while it runs.**

### Rule 6: environment
- **Python:** `./py` for torch work; `./py-mj` (from Step 2) for MuJoCo.
- **Machine:** Intel i5, 4 logical cores, 8 GB of memory. Sessions use at most 2 threads. Jobs use two workers of 2 threads.
- **Jobs:**
  - rewrite `results/d/progress.json` at least every 10 minutes, from the work itself;
  - append one line per measured policy to `results/d_ladder.txt` (the ladder job), so that results can be read as they come;
  - are resumable, and skip finished work;
  - are stopped by the launcher if they make no progress for an hour.
- **Where things go:** simulators, policies and their logs under `runs/d/` and `runs/lite/d_*` (not committed).
- **Long output:** send it to `.cache/logs/`, and print only its tail.

### Rule 7: context and token discipline
- **How to read:** only what your block names; grep first; at most about 200 lines at a time. Print JSON fields with `./py -c`; at most about 40 lines of output.
- **Never read these whole:**
  - other rounds' plans;
  - `results/paper_numbers.*` and `results/raw_*.npz`;
  - data files (print shapes only);
  - `history.bundle`;
  - logs (use `tail`).
- **Subagents:** only those your block names, and never a `model` parameter. Give each a self-contained brief.

### Rule 8: git
- **Branch.** Work on branch `optd`, which D0 creates from Option C's branch. Commit subjects are "D<n>: <what>".
- **Staging.** Stage only your paths. Never `git add -A`, push, checkout, restore, reset --hard, stash, clean or rebase.
- **Never commit:** `runs/`, `.cache/`, `.pylib*`, `optc/data/`, `step2/assets/`, `paper/build/`, `submission/` or `history.bundle`. `step2/assets/` holds the ANYdrive file, which is never committed or passed on.
- **Tags:** only `air2027-optd-start` (D0) and `air2027-optd` (D4).

### Rule 9: style
- **Readers are roboticists.** Plain language; define terms; at most two numbers-with-intervals in a sentence.
- **Honesty.** State failures as plainly as successes. That includes Option C's from-scratch simulators failing the walk test, the two design mistakes behind it (scaling and training amount), and every ladder simulator that failed.
- **Scope.** The world here is a hand-built MuJoCo model of ANYmal D, with a fitted PD motor. It is not Isaac Lab and not hardware; say so wherever a claim could be read more widely.
- **Double-blind.** Nothing in the paper or the supplement may identify the author: no names, user names, home paths or repository links.
- **Notes:** at most 300 words unless the block says otherwise.

### Rule 10: ending a session
- **Status line.** Append `## D<n> status: COMPLETE (YYYY-MM-DD HH:MM)` to `optd/LOG.md`, with PARTIAL or BLOCKED in place of COMPLETE where needed.
  - Take the time from `date '+%F %H:%M'`, never by guessing.
  - Add at most 8 lines after it.
- **Checkpoint.** Overwrite `optd/CHECKPOINT.md` with the last session and status, the next action, and any blockers.
- **BLOCKED:** add to `optd/DECISIONS.md` what is blocked, the options, and your recommendation.
- **PARTIAL:** commit what is done, and write the exact next step in `optd/CHECKPOINT.md`.
<!-- END COMMON -->

<!-- BEGIN D0 -->
## D0 — Branch, facts, code for the record and for fine-tuning, tests and the record job (Opus 5.5, high)

**Read:**
- `RESULTS_SO_FAR.md` and `optd/REVIEW.md`;
- `tail -12 optc/LOG.md` and `optc/CHECKPOINT.md`;
- `ls scripts/c/ scripts/s2/ optd/review/cloud_tests/`;
- in parts, by grep: `scripts/c/train_sim.py` (`RungWindows`, `load_rung`, `one_step`, `setup`), `scripts/c/sim_checks.py`, `scripts/c/calibrate_penalty.py` (`experiment`, `imagine`) and `scripts/c/evaluate.py`;
- `optd/review/cloud_tests/ft.py` (what the review ran; cloud paths, reference only).

**Steps**
1. **Git.**
   - On branch `optc` (on a resumed attempt: `optd`), commit what Option C's last job left uncommitted: its results, `optc/LOG.md` and `optc/DECISIONS.md`. Subject: "Option C: second pilot blocked at Gate A2".
   - Tag `air2027-optd-start`, then run `git switch -c optd`.
   - Commit `OPTIOND_PLAN.md`, `go_d.zsh`, `RESULTS_SO_FAR.md` and `optd/REVIEW.md` with `optd/review/` alone: "D0: Option D plan and the record so far (written before any Option D number on this Mac)".
   - Create `optd/LOG.md`, `DECISIONS.md`, `CHECKPOINT.md`, `notes/` and `jobs/`.
2. **Facts.** Ask air-reader to confirm, with file and line, and write `optd/notes/D0_facts.md`:
   - the authors' fixed scales, and that the lite dataset uses them when they are given;
   - Step 2's recipe: what `scripts/s2/train_policy.py` passes to `run_lite.py`; that Step 2's 18 registered policies used `--route rebuilt`; where each policy's actor, training record and training log are;
   - whether all 22 actors (18 registered, 4 pilot) exist, and whether the 18 match the sha256 that `results/s2_main.json` records;
   - that the lite code logs `Train/mean_episode_length` every iteration, and where each Step 2 run stored it;
   - the columns of Option C's rungs (state, action, contacts, fall, episode, source), and which sources are RWM's released policy;
   - how `scripts/s2/train_policy.py` builds a compressed copy of `REL`, and where Step 2's three single-network rebuilds and `A5_s0` are.

   If a fact contradicts this plan's design, record it and stop BLOCKED. A missing actor is not a contradiction: list it, and leave its policy out of the record.
3. **Code, in `scripts/d/`.**
   - `sims.py` (`./py`): one place that knows every simulator. `load(name)` returns the model in eval mode, its mean and spread, and a short record. It covers the 17 existing simulators, with Option C's four in their own scales, and any ladder simulator in `runs/d/`.
   - `checks.py` (`./py`): the checks in common units on both test sets, into `results/d/checks/<simulator>.json`. Use the paper's functions (`lyap.rollout_map`, `evalh.window_error`, `lyap.lyapunov`, `common.monitors`).
   - `walk_test.py` (`./py`): the walk test, with Option C's functions, into `results/d/walk/<simulator>.json`. It is secondary (Rule 2).
   - `evaluate.py` (`./py-mj`): the walking score of an actor file, with Option C's functions and protocol, into `results/d/eval/<policy>.json`. It also records the fall rate and Step 2's physics return. Before `prereg/PREREG_D.md` is committed, it measures only RWM's released policy, Step 2's 22 policies and `_test*` names.
   - `record.py` (`./py`): the record's table, from the files above and from Option C's result files.
   - `finetune.py` (`./py`): one run, as the design says, from Option C's trainer. It advances a run to a given step count, saves a simulator at each step count of the design's list that it passes, stops on a loss that is not finite, and is resumable from its last save, which also stores the optimiser and the random states.
4. **Tests, `scripts/d/test_d0.py`** (at most 10 minutes; all must pass):
   - `checks.py` gives `REL`, on the paper's recorded arena, the E8, λ1 and r̄J stored for it in `results/model_table.json`;
   - `sims.load("REL")` equals the released checkpoint, and `sims.load("REL__u10")` reproduces the variant record Step 2 stored;
   - `walk_test.py` on `REL` reproduces `results/c/calibration.json`'s tracking and ū;
   - `evaluate.py` on RWM's released policy, 2 schedules × 1 start, gives the same episode scores as `scripts/c/evaluate.py`;
   - walking windows come only from sources 0, 1 and 2; no window crosses an episode end; a fall appears only at a window's last row; no test-set episode is in the rung;
   - at step 0 and with the same seed, `finetune.py`'s loss on a fixed batch equals the rebuild's loss for the released model;
   - a run advanced to 3 steps and then to 5 equals a run advanced straight to 5, weight for weight;
   - a save holds the authors' scales, loads with `common.load`, and runs one step inside the lite adapter;
   - a teacher-forced step and a gentle step run, and change the weights.
5. **Rehearse the record job** at toy size: one policy on 2 schedules, one simulator's checks on 4 windows, then `record.py`.
6. **Write `optd/jobs/record.zsh`** (progress as in Rule 6; resumable; passes `zsh -n`):
   1. the record job's first sanity gate (RWM's released policy);
   2. the 22 policies measured; then its second sanity gate (`REL`'s two policies);
   3. checks and the walk test for the 17 simulators;
   4. `record.py`.
7. Commit "D0: facts, record and fine-tuning code, tests, record job".
<!-- END D0 -->

<!-- BEGIN D1 -->
## D1 — Policy wrapper, search, statistics, registration, rehearsal and the ladder job (Opus 5.5, high)

**Read:**
- `results/d_record.txt` and `optd/notes/D0_facts.md`;
- the design's Stage 2, Outcome and pools, and the registered analysis;
- in parts: `scripts/s2/train_policy.py`, and in `scripts/c/train_policy.py` the early stop (`collapse_at`, `Collapsed`, and where the log hook raises it);
- in `scripts/stats.py`, `spearman`, `_clean`, `boot_diff` and `boot_rho` (import these; never copy them);
- `prereg/PREREG_S2.md`, the headings, the statistics section and the decision rule only.

**Steps**
1. **Record what stage 1 gave,** in `optd/notes/D1.md` (at most 200 words):
   - the walking scores of `REL`'s policies;
   - how many of Step 2's nine simulators would count as failed, and how often its two policy seeds disagree on that (a disagreement is where the ladder would train a third policy);
   - which of Step 2's 22 policies and of Option C's pilot policies the design's stop rule would have stopped, and when, from their stored training logs (do this once `train_policy.py` has the rule). The registration declares any of Step 2's 18 among them.
2. **`scripts/d/train_policy.py`** (`./py`): Step 2's recipe for a ladder simulator, or for any of Step 2's nine.
   - It does what `scripts/s2/train_policy.py` does for a rebuilt simulator with no hold-out, and only swaps in the simulator from `sims.load`. Build the simulator before `run_lite` sets its seed, as Step 2's wrapper does, so that the random numbers are the same.
   - It adds the design's stop rule through the log hook, and exports the stopped policy's weights.
   - It writes the actor to `runs/d/policies/<policy>/actor.pt` and the record to `results/d/train/<policy>.json`. Policy names are `<simulator>__p<seed>`.
   - It applies the design's rule for a run that fails by its simulator.
   - Before `prereg/PREREG_D.md` is committed, it accepts only `_test*` names with at most 3 iterations.
3. **`scripts/d/ladder.py`** (`./py`): the ladder job's driver.
   - **The verdict rule** is one pure function: from a test's measured policies so far, it returns the next seed to train (2 after 1; 3 after a split) or the verdict.
   - **The search rule** is another: from a run's verdicts so far, it returns the next step count, or "done".
   - **A run's loop:** fine-tune to the next step count; checks and walk test; policies as the verdict rule asks, each measured with `evaluate.py` as soon as it is trained; record the test and its verdict; repeat.
   - **The queue:** two workers; a free worker claims the next run of the design's order whose start condition can be decided; the design's limits apply; a skipped run is recorded with its reason.
   - **State:** one file per run in `results/d/ladder/`, so that the job resumes exactly where it stopped. One line per measured policy goes to `results/d_ladder.txt`.
   - **A lock** keeps fine-tuning to one run at a time.
   - **The crash guard** of the design, with a `_test*` name for its 3-iteration run.
   - **A toy mode** replaces fine-tuning, training and measuring with stand-ins, for tests.
4. **`scripts/d_stats.py`** (`./py`): the registered analysis, with tests on synthetic results. It must run on any pool of four or more simulators, and must say plainly when a comparison cannot be computed.
5. **Tests, `scripts/d/test_d1.py`** (at most 15 minutes):
   - **the recipe is Step 2's:** with one thread, 256 robots, 3 iterations and seed 1, `scripts/d/train_policy.py` on `REL` and `scripts/s2/train_policy.py --sim REL --route rebuilt` log the same losses and rewards, to 1e-6. Use new `_test_d*` names, which write only new folders under `runs/`; delete those folders afterwards;
   - **the stop rule,** on made-up histories of episode lengths: the earliest stop is at iteration 100; it needs 50 iterations in a row; iterations before 50 do not count;
   - **the verdict rule:** two failed; neither failed; a split settled by the third policy either way; a stopped policy counts as failed; a score of exactly 0.15 does not;
   - **the search rule:** every row of the design's table;
   - **the driver, in toy mode:** all four search paths; a run whose loss is not finite; both start conditions; all seven runs when both start conditions hold; the 56-hour limit, with a started run finishing its tests; seeds 1 and 2 for every tested simulator, and seed 3 only after a split; the crash guard, both ways; a stop and a resume between two policies of one test;
   - **`d_stats.py`** on synthetic pools: a clear case of each label, and a pool of four.
6. **Write `prereg/PREREG_D.md`:**
   - the question; the world; the runs; the search, the verdict rule and the stop rules; the limits; the pools;
   - Step 2's recipe, the seeds, the collapse rule, and the rule for a run that fails by its simulator;
   - the walking score and its protocol; the checks in common units; the walk test;
   - the registered analysis, exactly as the design lists it, with Step 2's labels and decision rule;
   - **what was known before registering:** Step 2's results; Option C's records; everything in `optd/REVIEW.md`, including policies trained in cloud copies of the released model and of a walking-data fine-tune; everything in `results/d_record.json`;
   - **limits to state:**
     - the pool is built step by step from the policies' scores, so it is not a fixed sample, and the intervals are optimistic;
     - a simulator's score is the mean of only two or three policies, and a third is trained only after a split, so how many policies a simulator has depends on their scores; say how often Step 2's two seeds disagreed (step 1);
     - near the boundary even a majority of three can misjudge a simulator;
     - simulators saved from one run are not independent, and every one descends from `REL`;
     - one thing, how far a simulator was moved from `REL`, may drive every check and the score together;
     - 500 iterations is the authors' default and leaves policies immature;
     - the design was chosen after the review saw one fine-tuned simulator fail, and the number of policies per simulator after it measured Step 2's pairs and three seeds in that simulator.
7. **Review.** Ask air-stats-reviewer to review the registration, `d_stats.py`, the verdict rule and the search rule against this block and the design, in at most 400 words. Fix every MUST FIX and SHOULD FIX.
8. **Register.**
   1. Commit them together: "D1: PRE-REGISTER Option D (no ladder policy trained)".
   2. Write `results/prereg_hash_d.json` and commit it.
   3. Run `chmod a-w prereg/PREREG_D.md scripts/d_stats.py`.
9. **Rehearse the ladder job** at toy size, with real code: one `_test_d*` run fine-tuned by 2 steps; its checks on 4 windows; two policies of 64 robots and 2 iterations each; each measured on one schedule; the verdict rule applied; the driver's state written and resumed. Delete what the rehearsal made.
10. **Write `optd/jobs/ladder.zsh`** (resumable; passes `zsh -n`): the driver with two workers; then checks and the walk test for any tested simulator that lacks them; then `results/d_ladder.json` and the final `results/d_ladder.txt`.
11. Commit "D1: policy wrapper, search, registration, ladder job".
<!-- END D1 -->

<!-- BEGIN D2 -->
## D2 — Analysis and the paper (Opus 5.5, high)

**Read:**
- `results/d_ladder.txt`, `results/d_record.txt`, `prereg/PREREG_D.md` and `optc/notes/C1_rootcause.md`;
- `tail -30 .cache/logs/job_d_ladder.log`;
- in `paper/body.template.tex`, by grep: the Step 2 paragraph, the abstract's scope sentence, Limitations and the Conclusion;
- the appendix headings.

**Steps**
1. **Run `./py scripts/d_stats.py`.** It is registered; do not edit it. Its verdicts are the result, whatever they are.
2. **Numbers.** Write `scripts/d_numbers.py` (the `add(...)` pattern), and call it from `paper_numbers.py`. It covers the record, the ladder and Option C's results.
3. **Main text.** Replace the Step 2 paragraph with one paragraph and one small table:
   - Step 2 and Option C in one sentence each: what was tried and how it failed;
   - what adapting the released model did to its short-horizon error and to its policies' walking;
   - R0–R2, and the walk test.

   Change the abstract's scope sentence, Limitations and the Conclusion only where they are now false. The abstract stays at most 200 words.
4. **Appendix:**
   - move Step 2 into an appendix section, with its root cause;
   - add a section "Simulators of a known physics model", covering:
     - the record: every simulator that has had a policy trained in it, and how those policies did;
     - the from-scratch simulators, the walk test they failed, and why (scaling and training amount);
     - the ladder: the data, the runs, the search, the verdict rule and the stop rules, the ladder table, and all results;
     - the pooled analysis with Step 2's simulators.
5. **Fit.** `slack.py` must print PASS (reserve 0).
6. **Finish.** Write `optd/notes/D2.md` with the new keys and claims, and commit "D2: Option D in the paper".
<!-- END D2 -->

<!-- BEGIN D3 -->
## D3 — Claims, checks, supplement and documents (Sonnet 5.5, high)

**Read:**
- `optd/notes/D2.md`;
- the start of `scripts/check_claims.py`, and its claim descriptions (grep).

**Steps**
1. **Claims.** Add an assertion for every new claim. All must pass.
2. **Full pipeline,** every step passing:
   - `paper_numbers.py`;
   - `build_paper.py`, twice, giving identical text;
   - `slack.py`, `check_claims.py`, `check_numbers.py`, `check_bib.py` and `submission_checks.py`.
3. **Check against the evidence.** Ask air-claim-checker to check the new paragraph, the table and the appendix sections against `results/d_*`, `results/c_*` and `prereg/PREREG_D.md`. Fix each finding minimally.
4. **Supplement.**
   - Add `scripts/c/`, `scripts/d/`, `prereg/PREREG_D.md`, the `c` and `d` results and Option C's data manifest.
   - Leave the data and the simulators out: Option C's generator and `scripts/d/finetune.py` recreate them from their seeds.
   - Rebuild the supplement and scan it. The scan must find 0 hits.
5. **Documents.** Update `README.md`, `SUBMIT.md` and `ANOMALIES.md`.
6. Commit "D3: checks, supplement, documents".
<!-- END D3 -->

<!-- BEGIN D4 -->
## D4 — Fresh-eyes review, report and tag (Opus 5.5, high)

**Read:** `optd/notes/D2.md`, `prereg/PREREG_D.md` and `results/d_stats.txt`.

**Steps**
1. **Extract the main text** of the two-column check to `optd/review/body.txt`.
2. **Get two reviews, in parallel.**
   - air-referee reviews `body.txt`.
   - air-stats-reviewer checks `results/d_stats.json` against `prereg/PREREG_D.md`, and the new sentences against both.
3. **Act on the reviews.** Fix the text, with no new analysis. Decline anything else with a reason, under "D4 declined".
4. **Rerun the pipeline.** Everything must pass.
5. **Write `optd/REPORT.md`** (at most 500 words, plain language):
   - why the from-scratch simulators failed, and what replaced them;
   - what adapting the released model did to its errors and to the policies trained in it;
   - which runs were skipped or stopped early, and why;
   - the verdicts;
   - what the MuJoCo world can and cannot show;
   - the referee's score;
   - what the author must check.
6. **Close the round.**
   1. Commit "D4: Option D complete", and tag it `air2027-optd`.
   2. Run `git bundle create history.bundle --all` and `git bundle verify history.bundle`.
   3. Write in `optd/CHECKPOINT.md`: "Option D complete. Next: read optd/REPORT.md; submit by 31 Dec."
<!-- END D4 -->

## Sources behind this plan
- **The record so far:** `RESULTS_SO_FAR.md`.
- **The gate that stopped Option C:** `.cache/logs/job_c_pilot.log`, with `results/c/calibration.json` and `results/c/calibration_2S.json`.
- **Policies that collapse in imagination:** `results/c_pilot1.txt` and `results/c_diag.txt` (imagined episodes of 1 to 23 steps in `sim_10k_N8_s0` and `A5_s0`).
- **The authors' scales:** `../robotic_world_model_lite/scripts/configs/anymal_d_flat_cfg.py` (`state_data_mean`, `state_data_std`).
- **Step 2's recipe and policies:** `scripts/s2/train_policy.py`, `scripts/lite/run_lite.py` and `results/s2_main.json`.
- **This review's tests and numbers:** `optd/REVIEW.md`.

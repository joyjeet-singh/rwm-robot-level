# Option D decisions

Choices this plan did not settle, and anything blocked (Rule 3, Rule 10).

## D0 (14 Nov 2026): choices the plan did not settle (none blocks)
- **D0-1 r̄J test tolerance.** On the paper's recorded arena, `checks.py` gives `REL` the E8 and λ1 of `results/model_table.json` (relative 4e-7 and 1e-10). Its r̄J (3.2786) is 0.13% below the stored 3.2827 (computed on the cloud), and equal to the value Step 2 recomputed on this Mac (`results/s2/checks.json`, validation). The test requires equality with the Mac value and 0.5% with the stored one. The gap is float drift over 368 steps; E368 shows the same (0.36%).
- **D0-2 Fine-tuning optimiser.** Fresh Adam state (the review's choice). The authors' stored optimiser state is not used. Weight decay 1e-5 is the reference config's value.
- **D0-3 Missing episode lengths (for D1).** The lite code logs `Train/mean_episode_length` only when an episode has ended in that iteration (about 2% of iterations have none). In the record's descriptive use of the stop rule, a missing value breaks a run of short iterations. That is conservative: it gives fewer stops. Recommendation for D1: use the same rule in the policy wrapper and register it.
- **D0-4 Walk-test start states for Option C's four.** The design says to use the authors' recorded data. These simulators never saw that data, so they get all ten episodes, converted into their own scales. This differs from Option C's calibration, which started them from MuJoCo windows, so their walk-test numbers are not comparable with `results/c/calibration.json`. The rebuilds and `A5_s0` keep Step 2's hold-out (episodes 1 and 8). `REL`, its copies and the ladder hold nothing out.
- **D0-5 Common units for Option C's four.** The change of units is folded into an equivalent model: the first GRU layer's input weights and the last layer of each state head. Tested against converting the predictions directly: largest difference 2.4e-6 in the authors' units. λ1 is computed on the original model, in its own coordinates, as the design says.
- **D0-6 Windows and λ1.** λ1 uses every 400-row window of each test set: 40 expert and 67 mixed (the review used 12). Mixed episodes shorter than 400 rows cannot hold a window, so they are left out and counted (Option C's rule).
- **D0-7 Fall rate and physics return.** "Fall rate" is the share of the 96 walking episodes that end in a fall. The physics return, and a second fall rate, come from Step 2's protocol (10 recorded schedules × 8 starts). The rehearsal re-measured `REL__p1`'s physics return as 17.342, equal to Step 2's stored value.
- **D0-8 Pilots and Option C's policies.** `REL__p0` was trained by the released route, not the rebuilt one; the record says so. `A5_s0`'s two policies were never given a walking score. The record carries their fall rate and return from `results/c/d3_eval/` and does not measure them; the 22 are the measured set.
- **D0-9 Fine-tuning resume state.** It is written every 100 steps, as well as at each save and at the end of each call. Results are unchanged: a run advanced to 3 steps and then to 5 equals one advanced straight to 5, weight for weight.
- **D0-10 Rehearsal.** Sessions may not run job files (the launcher's deny list). The rehearsal therefore ran the job's toy-size commands one by one, from the `REHEARSAL=1` branch of `optd/jobs/record.zsh`. The job file itself passes `zsh -n`. A failing gate was checked separately: it exits 3 and writes its note.

## D1 (14 Nov 2026): choices the plan did not settle (none blocks)
- **D1-1 Missing episode lengths.** D0-3's rule is registered: an iteration with no logged `Train/mean_episode_length` breaks a run of short iterations (fewer stops; conservative).
- **D1-2 "The same crash".** The same exception type, inside the lite code, on two attempts. Two different crashes are not a result: the job stops with an error (conservative: nothing is recorded as failed).
- **D1-3 Job time.** The 56-hour limit counts the driver's running time, summed over resumes (`results/d/ladder/job.json`), not wall-clock time including any time the job was not running.
- **D1-4 The queue.** A free worker skips a run whose start condition cannot yet be decided (its parent still running) and takes the next in the order; the skipped run is decided when its parent ends. Started runs are resumed first after a restart.
- **D1-5 Stopped and not finite.** A stopped policy whose logged losses are not finite counts as failed by its simulator (score 0, fall rate 1, not measured), as a finished one would.
- **D1-6 Checks during the job.** If a ladder simulator's checks or walk test cannot be computed, the driver records it and goes on (checks never steer the search); the job's last step retries them. A simulator still without a check leaves the analyses that use it, listed.
- **D1-7 Analysis details the design lists but does not define.** "R0–R2 on the pooled pool, and on the mixed test set" = two variants (pooled pool on the expert set; primary pool on the mixed set). Labels added: LONG-HORIZON BETTER / TRAINING-HORIZON BETTER (E368 vs E8); WALK TEST BETTER / TRAINING-HORIZON BETTER; AUC against 0.5 (ABOVE / BELOW CHANCE), NOT COMPUTABLE with fewer than 2 of either kind or more than 10% one-kind draws. Best three: Step 2's selection method on E1, E8, E100, E368, λ1, r̄J. A pool of fewer than 4 simulators is NOT COMPUTABLE. `REL` and Step 2's copies take their "failed" verdict from their two registered policies.
- **D1-8 A progress file.** A smoke run of Step 2's wrapper (`_test_d1s2`, before the test was written) rewrote the untracked progress file `results/s2/progress.json`. It holds no result. The test now redirects that file to `.cache/`.
- **D1-9 After the statistics review (air-stats-reviewer).** The physics-return fill uses the measured policies of the primary pool (the pool analysed), finite values only, and applies only to policies failed by their simulator; a measured policy without a physics return is missing data. A check stored as null is missing (the simulator leaves that analysis, listed); a computed non-finite value ranks worst. A missing outcome (no seed-1 policy, no physics return) leaves only that simulator out. The primary pool counts tests that reached a verdict. Step 2's policies enter without the stop rule (stated as a limit). Work done before a sanity gate is descriptive. All are in the registration.

## D2 (17 Nov 2026): choices the plan did not settle (none blocks)
- **D2-1 The registered analysis already existed.** `results/d_stats.json` and `.txt` were on disk (17 Nov 12:16, before any D2 session ran; three earlier D2 launches failed at sign-in and wrote nothing). No launcher or job file names `d_stats`, so their writer is unrecorded. Rule 5 forbids overwriting, so D2 ran `scripts/d_stats.py` (unedited, read-only) into `.cache/d2/` instead: both files are byte-identical to the existing ones. The existing files are the result.
- **D2-2 The verdict table.** Its physics-test rows (R0–R2) now give Option D's registered verdicts (walking score, 19 simulators). Step 2's R0–R2 stay in appendix A12's text with their intervals; A12 no longer points to the table for them.
- **D2-3 Fit (reserve 0).** The new paragraph and table made the two-column check 29 lines too long. Besides the paragraph, table, abstract sentence, Limitations and Conclusion, D2 compressed wording elsewhere without removing any result: Fig. 2's caption; "five registrations" (was four, now false); one Results sentence of untested speculation (the Q6 "possible cause"); one Discussion sentence repeating the Results ("within one training method…"); the run-time-alarm sentence; the imagined-return sentence; "judges choices". `slack.py`: PASS, 6 pages, abstract 199 words.
- **D2-4 Walking-score anchors in the paragraph.** The released policy's 0.942 and standing still's 0.069 stay; the released model's own policies' 0.59 is in the paragraph, not the table caption (space).
- **D2-5 Option C's training amount.** "About 0.1 of one pass" = 2,572 steps × the 256-window batch (`results/c/probe.json`) / the 6M rung's rows (`optc/data/rung_6M.stats.json`); rows stand in for windows, which are slightly fewer, so the share is a slight underestimate.

## D3 (17 Nov 2026): choices the plan did not settle (none blocks)
- **D3-1 Claim-check findings not applied.** #2 (secondary variants in the body), #6 (tracking range in the body), #8, #9: the body has no free space (reserve 0) and A13 carries the content; see `optd/review/D3_claimcheck.md`.
- **D3-2 Supplement paths.** One result file (`results/d/train/M_s0_d20__p1.json`, a crash traceback) holds the working copy's absolute path. The working file is untouched; the staged copy is rewritten (`build_supplement.py`, `neutralise_paths`) and the README inside says so.
- **D3-3 Hash file.** `results/prereg_hash_d.json` lists `results/d_record.json` and `results/d/stage1.json`; neither is altered by the neutralisation (they hold no paths).

## D4 (17 Nov 2026): review actions (none blocks)
- **D4-1 Text fixed after the reviews** (`optd/review/D4_referee.md`, `D4_stats.md`; no new analysis). Recommendation scoped to
  prediction error; abstract and conclusion say "one- and 8-step error" and "13 of 18 searched"; "registered" -> "primary" where
  only primary tests stayed unsettled; "in reverse" -> "partly in reverse (descriptive)"; the released model's 0.59 labelled a
  policy score; "imagined tracking ranked it well"; failure rule "≥2 policies below 0.15"; Table IV caption; the fine-tuning
  amount named as a possible confound (Limitations, A13); A13 wording (R0 premise, R1–R2 primary; 2- and 3-iteration tests).
- **D4-2 Fit.** The fixes ran 2 lines over. Cut: "forecast" (intro, a paraphrase of [6]); "(descriptive; weights only)" ->
  "Rounding only the weights ... (descriptive)"; "on {{n_P}} GRU simulators" left the abstract (the count is in Sec. III).
  `slack.py` PASS, free 0.06 textheight.

## D4 declined
- **Referee 1b (abstract scope longer), 3 (novelty against [29]), 4 rest (Table I caption, E_h definition, jargon), 5 rest
  (38% vs 39%, four-times-longer models, "five registrations", Q5/Q1-24 rows, Fig. 2 caption, "101 trained simulators"):**
  the body has no free space; the numbers come from result files and pass `check_claims.py`/`check_numbers.py` (38% and 39% are
  different quantities); the definitions are in Sec. III and the appendices.
- **Referee 1c ("registered" overstates; self-reported commits):** the paper already says the registrations have no
  independent timestamp (Sec. III); the hashes are in the supplement.
- **Referee 2 (Q1 depends on pool make-up; label-free checks see longer rollouts; Table III dagger):** a new analysis
  would be needed; registered verdicts stand as registered; the Limitations name the shared parents and data.
- **Referee 6 (supplement git metadata):** the supplement holds no `.git`; `submission_checks.py` scan 0 hits.
- **Stats 8 (d_stats.txt AUC lines omit dropped draws; ci97.5 for secondary):** `scripts/d_stats.py` and its results are
  registered and read-only; the paper quotes only 95% intervals for secondary results.

# Rewrite decisions

## W0
- Step 2 slot is drawn inline (a framed `\parbox`, one column wide, 0.30 textheight, with \medskip around), not as a float: conservative for the fit check, since an inline box cannot move to fill a column gap.
- `scripts/slack.py` writes `results/slack.json` and updates `results/build.json` (the reserve entry) via build_paper.py; no registered result file touched.
- HEAD at start was c699388, tagged air2027-step1 (as expected). `figures/fig_horizon.*` left unstaged.

## W1
- **Title (for the author):** kept, "One Step Ahead Is Not Enough: Testing a Learned Quadruped Simulator at the Horizon It Is Used" (conservative). Alternatives in OUTLINE part 2: "When One-Step Checks Mislead: Vetting a Learned Quadruped Simulator for Long Imagined Rollouts"; "Checking Learned Legged-Robot Simulators at the Horizon of Use".
- Abstract leads with the training-choice reversal (descriptive/earlier registered) and pruning plus fine-tuning (registered), then the checks; it drops the ensemble-spread and checklist sentences (both stay in the body). The draft is 192 whitespace words. When Step 2 arrives, about 10 words must go.
- Budget: target 3,400 prose words, including the abstract; cap 3,550. This is estimated from the current 3,950 words that fill six pages exactly, minus the 19-line slot, about 5 lines for the new Table 3, and a margin. `slack.py` is the arbiter.
- Prose names "main pool (F18)" and "five-split pool (X5)"; tables keep F18/X5.
- Table 2 is restructured (ρ only, adding the policy-command and architecture columns); its x2 split columns go to A3. Table 3 is new (registered verdicts, one column). Fig. 2 moves to the appendix only as a last resort.
- Related work keeps every citation (no refs.bib change planned).
- Step 2 question: does a simulator's 2-second error, and do the two label-free checks, predict how well a policy trained inside it follows its velocity command in a physics simulator?
- The OUTLINE word count is 1,598 alphanumeric tokens; table pipes and dashes are not counted.
- **air-stats-reviewer on parts 3 and 7:**
  - MUST FIX 1–3 fixed:
    - A8 (now A7) gives the Holm qualifier.
    - C14 gives the AR3 extended pool (LABEL-FREE BETTER, SEC).
    - C7 cites q6_* for the Q6 verdict, with the 10/20% shifts as EXPL.
  - SHOULD FIX 4–12 fixed:
    - "averaged over three other architectures", and "GRU" in place of "recurrent".
    - A6 is scoped to the GRU pool and calls Q3 secondary.
    - A3 reads "descriptive; partly registered earlier"; the RWM ablation citation (Sec. 4.3, App. A.4.1) stays in the introduction.
    - A4 reads "restored nearly all", with the FT2 key; the K values are noted as X5.
    - C2 now says "the same reversal".
    - C16 reads UNCHANGED within the 0.10 margin.
    - C17 gives the label WORSE (SEC) and ρ≈0 (DESC).
    - C18 limits non-divergence to the GRU pools.
    - C13 adds Holm. d4_F18_non_worse = 15 was confirmed.
  - Must-not additions 13–16 added.
  - NOTE 17: keys x3_n_eps, x3_p, k_h1/k_h100 and s1_art_rssm_* exist in paper_numbers.md.
  - NOTE 18 declined: n_trained (101) and n_compressed (200) cover F18 and X5 (paper_numbers.md), not F18 only.
  - NOTE 19: C15 notes CL1 is driven by the trained models (CL9). The two-level interval stays in the appendix (EXPL).

## W2
- **No new appendix section.** OUTLINE part 5 asked for a new section "Training choices, details" after A1. Inserting it would renumber A2–A10, which the body, the appendix and check_claims descriptions cite as text. Conservative: the material went into A1 (retitled "Training settings, models and rebuild") as a paragraph "Training choices, details"; numbering A1–A10 unchanged.
- **x2 split columns** were not copied into a new A3 table: A4's `tab:app-two` already prints every x2_tc_* and x2_c_* value. The body points to appendix A4.
- **Table 3 pools** read F18, X5 and AR (AR = the three other architectures, averaged; defined in the caption). AR and CL estimates keep three decimals, as in A8–A9, so the same value is printed alike in body and appendix.
- **rw_numbers.py** adds only formatting re-prints (signed, two decimals) of values in `results/s1_stats.json`, so Table 2 reads alike across columns; no new values.
- **Spare space:** the reserve build still has room for a 0.50 textheight slot; nothing was added back, leaving margin for Step 2.
- **Wording kept from the old text, flagged for W3:** "moderate pruning adds error but barely moves the growth rate" sits beside Q6 NOT SETTLED (as before); it describes the size of the shift, not a verdict.

## W4
**Step 2 candidates (requested by the referee; not run):**
- Train a policy inside a few simulators (teacher-forced, rollout-trained, 30% pruned) and test it in a physics simulator. This is Step 2's planned question (OUTLINE part 8).
- Q1 restricted to rollout-trained models only (no compressed copies, no one-step-trained). The nearest registered result, Q5 (without the one-step-trained models), is now in Sec. 4.2.
- A training-episode error baseline (8- or 100-step error on the training episodes) as a further check to compare.

**W4 declined (with reason):**
- Referee 1, drop or gather the status tags: C4 requires labels wherever a result appears. Only small changes were made (definitions added, labels corrected).
- Referee 2, the released model's positive growth rate in the abstract: no room (197 of 200 words). It is now in checklist item (3) and is already in Sec. 4.1 and Limitations.
- Referee 5, "label-free" is motivated by new commands, yet the checks use the held-out history and commands: the checks need only the history and joint commands, which exist for new commands. The text is unchanged; the baseline is a Step 2 candidate.
- Referee 6, a sentence on what is new against [26] and [29], and DayDreamer: the contribution bullets already state the registered comparison. No space (free slot 0.32). New references are outside this round (refs.bib may only drop or fix).
- Referee 7, change the title, and cut checklist item (4) and the Uncertainty paragraph: the title was kept by W1, the scope is stated in the abstract and the introduction, and item (4) rests on appendix A6. Less change is the conservative choice.
- Referee 8, the floor at 368 steps: Fig. 1 already shows it; no space.
- Minor: renaming TRAINING-HORIZON BETTER (it is the registered label); AR/CL/F18/X5 names (defined in Sec. 3); Table 1 "40 on X5" (declined for space).
- Stats 9, the Holm family excludes the 24- and 368-step readings: the text already says secondary verdicts are uncorrected and Holm covers "the 7 secondary tests of Q3–Q6". Unchanged.
- Stats 4, write "primary" for "registered" in the abstract: "registered" is accurate and shorter. Unchanged.
- Supplement not rebuilt in W4 (check_claims.py changed): left to W5's final build.

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

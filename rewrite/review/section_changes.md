# What changed in the main text, section by section
`latexdiff` is not installed, so this list replaces a marked-up PDF. Old text: `git show air2027-rewrite-start:paper/body.template.tex`
(4,386 words incl. markup); new: `paper/body.template.tex` (4,231). Nothing outside the template changed what a number says.

## Whole paper
- Rebuilt around one story: one-step checks picked the wrong training and compression choices; label-free checks were
  compared on two pools, four architectures and two command sources; a short checklist follows.
- Evidence now covers the original study, revision round 1 and Step 1 (studies CL, AR, TB).
- Fits six two-column pages **with** the Step 2 slot (a blank framed box, 0.30 of the text height, one column wide,
  at the end of Sec. 4). Free space after the last review: 0.32 of text height.
- Every claim is asserted in `check_claims.py` (107 -> 127 assertions; none removed).
- Title unchanged.

## Abstract (197 words, limit 200)
- Leads with the training-choice reversal (descriptive / earlier registration), then pruning and fine-tuning
  (registered), then the label-free checks.
- Names the pool and the recorded commands for the residual and growth-rate results; Step 1 results added.
- Dropped the ensemble-spread and checklist sentences (both stay in the body).

## 1 Introduction
- Shortened; two questions and scope stated early; three contributions (where one-step checks mislead; the
  registered comparison of label-free checks; a checklist). Reversals are called "descriptive", not "registered".

## 2 Related work
- Four short paragraphs; every citation kept.

## 3 System and protocol
- Pools renamed in the text: main pool (F18) and five-split pool (X5). Models now include Step 1's MLP, RSSM and
  Transformer. Joint commands and velocity command kept distinct. Holm correction and Spearman correlation defined.
- Model list, rebuild checks, measure details and training-choice details moved to appendix A1.

## 4.1 One-step checks reverse training and compression choices (was "Where one step and the horizon of use disagree")
- Table 1 and Fig. 1 kept, shorter captions. Adds the longer-training (TB) and other-architecture checks.
- Pruning shifts and width removal moved to A5.

## 4.2 Which check ranks simulators without recorded futures?
- Table 2 restructured: rank correlation only; rows F18 recorded/policy, closed loop, X5, MLP, RSSM, Transformer.
- **New Table 3:** the 12 registered verdicts with pool, command source and Holm dagger.
- Adds Q5 (secondary; without the one-step-trained models) with its Holm failure; Q6 is NOT SETTLED and no longer
  described as "barely moves".
- Five-split failure and its likely cause kept; policy-command results (CL) added.
- Ensemble spread, variance collapse and the released model moved to A6.

## Step 2 slot
- New. Blank framed box at the end of Sec. 4 (becomes 4.3).

## 5 Discussion
- "Why one-step checks mislead here" and a checklist of four items. Item 3 notes the released model's positive
  growth rate (so a negative one certifies nothing).

## Limitations
- Now names: one robot and trotting gait, simulated data, no physics-simulator test, command source, four
  trajectories from two episodes, no correction across registrations, the 3 of 13 verdicts lost in the new pools, few
  runs per architecture, TB.

## 6 Conclusion and AI statement
- Shorter. The AI-use statement is still there and still must be edited by the author.

## Appendix (single-column PDF only)
- A1 retitled "Training settings, models and rebuild"; A5 gained pruning shifts and width removal; A6 retitled
  "Uncertainty and run-time signals". No section renumbered; nothing registered left the paper.

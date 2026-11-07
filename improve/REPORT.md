# Improvement round: report for the author

## What changed
No experiment was run; no registration or result file was touched.
- Abstract rewritten (185 words; status stated once per group; scope includes "no physics-simulator test").
- The recommendation now leads with checking several steps ahead on held-out episodes. Label-free checks come second, as a fallback.
- The kinematic residual is set against iKCE. The limits of the registrations are stated. Model counts are reconciled (57 trained on the main pool, 55 on the second, 11 on both, 352 pool entries).
- The imagined-return signal is mentioned as secondary and as a candidate for a follow-up study.
- Tables 1 and 3 regrouped, appendix page breaks, Figure 1 with one line.
- Section list: `improve/review/section_changes.md` (no `latexdiff`).

## Facts corrected (sources: `results/external_facts.json`, each with URL, version and quoted line)
- RWM and RWM-U imagine **100 steps** per iteration, not 24 (RWM Table S11; RWM-U Table S11). 24 is the released code's PPO collection length, so it is no longer called RWM's horizon.
- RWM's own comparison says its multi-step model beats teacher forcing (RWM Sec. 4.3). Our text now says this.
- Data: 7,991 of our transitions against RWM's 6 million (RWM Table 1) = 0.13%. "About 750 times more" for the released model was weakened to "RWM reports", since its data are unrecorded.
- iKCE (2607.05966) is tested on one model and does not rank models. Our use differs.

## Simulators trained the same way (exploratory, added after the registered results)
Among the 48 rollout-trained simulators of the main pool, correlation with 100-step error: kinematic residual +0.85, growth rate +0.76, 8-step error +0.38, 1-step error -0.18. On the second pool the growth rate does not carry over (-0.21). Among compressed copies every check ranks well. The case for label-free checks is mixed, and the text says so.

## Referee (a subagent)
4/10, weak reject. Main points: thin novelty beside RWM's ablation, small data scale, an unequal comparison (8-step error shares start positions with 100-step error), and an overclaiming title or conclusion. Fixed: verdict wording, labels, conclusion scope. Declined, with reasons: `improve/DECISIONS.md`.

## Before you submit
1. Put the AI-drafted paragraphs in your own words.
2. Check the AI-use statement. It omits the AI-run reviews and reference checks, and says only "the author(s)".
3. Check the corrected facts against the three papers; quotes came through a page summariser.
4. Check the supplement's version-control history for names.
5. Fit: six two-column pages with the Step 2 slot, zero spare room.

## Next
Read this, then decide Step 2 (`STEP2_PLAN.md`): policies trained inside scored simulators and tested in a physics simulator.

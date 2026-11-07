# Section-by-section changes in the improvement round (I2-I4)

`latexdiff` is not installed, so this list stands in for `paper/build/body_diff.pdf`. To see exact wording, run
`git diff air2027-improve-start air2027-improve -- paper/body.template.tex paper/appendix.template.tex`.
Line numbers refer to `paper/body.template.tex` now. Edits are in the templates only; every number is still a key.

## Abstract (template lines ~9-26)
- Rewritten (185 words): three groups of claims, status stated once per group; recommendation led by checking
  several steps ahead; scope sentence ("no physics-simulator test").
- Residual marked "(secondary)" (I4); imagined-return comparison marked exploratory.

## 1 Introduction (l.28-60)
- RWM and RWM-U imagine 100 steps per iteration (key ext_rwm_imag_steps), not 24. 24 steps is only an intermediate horizon.
- Data fraction stated as 7,991 over RWM's 6 million transitions (0.13%), from external_facts.json.

## 2 Related Work (l.61-80)
- New: iKCE placed against the paper's kinematic residual (same idea, different use: ranking simulators).
- RWM's teacher-forcing statement corrected to what RWM says (RWM-AR beats RWM-TF).
- Repeat of RAMBO dropped (still cited in the Introduction).

## 3 System and Protocol (l.81-142)
- Models section: counts reconciled (57 trained on F18, 55 on X5, 11 on both, 352 pool entries).
- Registration limits and status kinds (registered / secondary / descriptive / exploratory) stated.
- Data fraction wording (7,991 / 6 million).

## 4.1 One-step checks (l.144-220)
- Table 1: grouped rows (Training / Compression / Uncertainty); the counts column moved to the caption.
- Caption no longer calls 24 steps RWM's horizon; Fig. 1 draws one line at 100 steps (RWM, RWM-U).
- Released-model wording: "unrecorded data (RWM reports about 750 times our transitions)".
- Verdict wording tightened: AR3 pooled "not settled"; closed-loop CL5 "ranked worse, with rho near zero".

## 4.2 Which check ranks simulators (l.221-335)
- New paragraph: ranking among simulators trained the same way (exploratory, A4 table app-within).
- Imagined return mentioned as a secondary signal and "a candidate for a follow-up study".
- Table 3 regrouped (scriptsize); X5 residual "not separated from" 8-step error; "two of three other networks".
- Recommendation order: (1) check 8 or more steps ahead on held-out episodes; (3) label-free checks as fallback.

## 5 Discussion (l.336-365) and 6 Conclusion (l.366-)
- Discussion: "were not shown to rank architectures together"; limits of the two registrations.
- Conclusion: "the safer way to rank these simulators"; scope of the claim; policy performance named as open.

## Appendix
- A3: sentence that E8 shares start positions with E100 (split-half check beside it).
- A4: new table tab:app-within (rankings within training method, exploratory).
- A5, A6, A8, A9, A10: details moved here from the main text to fit (nothing removed from the paper); released-model wording in A6.
- FloatBarrier (page break fallback, placeins absent) before each appendix section.

## Code
- New: scripts/im_facts.py (results/external_facts.json), scripts/im_within.py (results/im_within.*), scripts/im_numbers.py.
- make_figures.py: Fig. 1 single horizon line. check_claims.py: +I2/I4 claims (159 checks).

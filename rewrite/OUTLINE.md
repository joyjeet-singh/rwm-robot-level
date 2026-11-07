# W1 outline of the new main text
Ids: `rewrite/notes/W0_evidence.md`. Status: PRIM, SEC (95%, uncorrected), EXPL, DESC, EARLY (earlier registration).

## 1 Story
1. One-step checks picked the wrong training choices: teacher forcing and one-step training look better one step ahead, 2–2.6 times as wrong at 2 s (E1–E3, DESC/EARLY), also after 4x longer training and in MLP and Transformer, not RSSM, simulators (E59, E61, DESC).
2. Same trap in pruning: 30% pruned keeps 90% of skill one step ahead, 39% at 2 s; fine-tuning restores it (E6 DESC; E10–E11 PRIM).
3. The growth rate (no recorded future needed) ranked 2-s error better than one-step error on the main pool and averaged over three other architectures, two carrying it (E24, E56 PRIM); not separated from 8-step error (E25); failed on mostly compressed copies (E28 SEC; likely cause E7).
4. The kinematic residual ranked the main pool best with recorded commands (E29 SEC, E39 EXPL), less well under the authors' policy (E45 PRIM); the growth rate's change: not settled (E46); through the policy, worse (E49 SEC).
5. Practice: train and validate at the horizon of use; fine-tune after pruning; screen with label-free checks; negative λ1 certifies nothing (E40, E41).
6. Scope: one robot and trotting gait, flat ground, simulated data; policy only in imagination; no physics test.

## 2 Title
Keep: "One Step Ahead Is Not Enough: Testing a Learned Quadruped Simulator at the Horizon It Is Used". Alternatives: "When One-Step Checks Mislead: Vetting a Learned Quadruped Simulator for Long Imagined Rollouts"; "Checking Learned Legged-Robot Simulators at the Horizon of Use".

## 3 Abstract plan (draft is 192 words by whitespace; keep ≤195 by `slack.py`)
| # | Sentence | Keys | Status |
|---|---|---|---|
| A1 | Learned simulators let legged robots learn in long imagined rollouts, yet are often trained, compressed and checked one step ahead. | – | context |
| A2 | We rebuilt the Robotic World Model's (RWM) open quadruped simulator and measured N trained and M compressed simulators. | n_trained (F18+X5), n_compressed | DESC |
| A3 | One-step checks picked the wrong training choices: teacher forcing and one-step training were better one step ahead but r1 and r2 times as wrong at 2 s (descriptive; partly registered earlier), also after longer training and in two of three other architectures. | ab_h100_ratio, N1overN8_E100; s1_tb_*, s1_ar4_* (no number shown) | DESC/EARLY |
| A4 | Pruning 30% of weights kept K1 of the skill one step ahead, K100 at 2 s (descriptive); fine-tuning restored nearly all (registered). | K_u30_h1, K_u30_h100 (X5); ft_FT1_verdict, ft_FT2_verdict | DESC; PRIM |
| A5 | The error growth rate, needing no recorded future, ranked simulators by 2-second error better than one-step error (registered): on n_P GRU simulators and, averaged, over three other architectures (two carry it). | n_P; q1_verdict, s1_ar1_label | PRIM |
| A6 | On the GRU pool it was not separated from 8-step error (registered) and failed a secondary replication on mostly compressed copies. | q2_verdict; q3_1_verdict | PRIM; SEC |
| A7 | A kinematic residual ranked the n_P best with recorded commands (descriptive; its lead over 8-step error failed multiple-test correction), less well with the RWM authors' policy (registered). | q4_1_rho_new, a2c_*; s1_cl1_label | DESC; EXPL; PRIM |
| A8 | Scope: simulated data, one robot and trotting gait, no physics-simulator test. | – | scope |
Body only: gradient match, ensemble spread, checklist. The RWM ablation is cited in the introduction (li2025rwm, Sec. 4.3, App. A.4.1).

## 4 Section plan
Now 3,950 words and four floats fill six pages; needed: 19-line slot, 5 lines (Table 3), margin. **Target 3,400 words incl. abstract, cap 3,550**; `slack.py` decides.
| Section | Purpose | Words |
|---|---|---|
| 1 Introduction | context; two questions; scope; 3 contributions (where one-step checks mislead; registered comparison of label-free checks over two pools, four architectures, two command sources; checklist) | 450 |
| 2 Related work | four paragraphs as lists; keep every citation | 160 |
| 3 System and protocol | data; simulator; rebuild; "main pool (F18)", "five-split pool (X5)"; models incl. Step 1; measures (joint vs velocity command); three registrations; wider intervals, Holm (1 sentence) | 640 |
| 4.1 One-step checks reverse training and compression choices (Table 1, Fig. 1) | objective, horizon, robustness (TB, AR4); λ1 falls with N; pruning, FT, 8-bit; uncertainty, 2 sentences | 700 |
| 4.2 Which check ranks simulators without recorded futures? (Tables 2–3, Fig. 2) | main pool; five-split failure and cause; other architectures; policy commands | 700 |
| Step 2 slot | end of Sec. 4 | 19 lines |
| 5 Discussion | why one-step checks mislead here; checklist of 4 | 230 |
| Limitations | scope; command source; few trajectories; dependence; λ1 caveats; few runs per architecture; TB | 190 |
| 6 Conclusion, AI statement | | 130 |

## 5 Moves
**To the appendix** (with their body-only keys):
- diff_*, win_*, stale_* → A1, new paragraph "Rebuild and verification"; full model list (n_R, n_T, hp_*) → A1.
- ph_*, dist_h100, x3_n_eps, x3_p, N32_E1/E100, ng_q4_* → new section "Training choices, details" after A1.
- width removal, perc_*, lamshift_u10/u20/u50/s128, lampos_* → A5.
- variance collapse, per-horizon multiplier, x7_summary, cov2_perh_h100, cov_nom2 → A6, retitled "Uncertainty and run-time signals".
- x2 split columns of Table 2 → A3 (small table; body keeps one sentence).
- rel_E1, rel_E100, rel_floor_E100, released_floor_E1 → A6; body keeps rel_lam.

**Condensed:** contributions; setup; 4.2's exploratory paragraph → one pointer; Limitations.
**Cut only because the appendix has it:** pre-registration knowledge (A2); lost-verdict list (A4); Q1/Q2 at 24 and 368 steps except one clause on Q2_E24; Q5, Q6 detail; FT secondaries (A7); CL3–CL9 (A8); AR3–AR4 (A9); TB numbers (A10).

## 6 Tables and figures
- **Table 1, Fig. 1:** keep; shorter caption; no regeneration.
- **Table 2 (tab:checks), changed:** ρ with E100 only, no verdict rows. Rows E1, E8, λ1, λ1 through the policy, r_J. Columns: needs recorded future; main pool, recorded (q1_rho_*, q2_rho_e8, q4_1_rho_new); main pool, policy (s1_t_F18_*_pol, _lam1_cl; E1, E8 "--"); X5 (q3_*_rho_*, x1_rJ (E)); MLP, RSSM, Transformer (s1_art_*).
- **Table 3 (tab:verdicts), new, one column:** question, pool, difference [interval], verdict, status: Q1, Q2, Q3 vs E1, Q3 vs E8, Q4 vs E1, Q4 vs E8 (Holm note), AR1, AR2, CL1, CL2, FT1, FT2. Verdicts from keys only; table* if too wide.
- **Fig. 2:** keep, shorter caption; if over, last resort: to appendix.

## 7 Claims (status; keys)
- C1 TF better at 1 step, worse at 24/100/368 (DESC; ab_h*_ratio); rollout better on all episodes at 368 (EARLY; x3_n_eps, x3_p).
- C2 N=1 shows the same reversal (DESC; N1overN8_E*).
- C3 At 10,000 iterations TF still better at 1, worse at 100 steps, 3/3 seeds; loss still falling (DESC; s1_tb_*).
- C4 MLP, Transformer: TF better at E1, worse at E100, 5/5 pairs; RSSM worse at E100 in 0/5 (DESC; s1_ar4_*).
- C5 λ1 falls with N (DESC N*_lam; EARLY ng_q4_rho); N≥8 rebuilt models contract, TF ones do not (DESC); released model λ1>0, maybe on training episodes (DESC; rel_lam).
- C6 30% pruning K at 1/100/368 steps (DESC; K_u30_*); FT1 FINE-TUNING RESTORES SKILL, FT2 NEARLY AS SKILLED (PRIM; ft_FT1_*, ft_FT2_*).
- C7 Moderate pruning adds error; Q6 (30%-pruned less stable, X5) NOT SETTLED (SEC; q6_*); 10/20% shifts EXPL (lamshift_*). 8-bit keeps K_int8w_h100 (DESC; weights only).
- C8 Ensemble spread ranks error but is too narrow, more with horizon (DESC; dis_spear_h100, k_h*).
- C9 Table 2's ρ values (DESC).
- C10 Q1 LABEL-FREE BETTER, Q2 NOT SETTLED (PRIM); at 24 steps E8 ahead (SEC; q2_e24_verdict).
- C11 Residual vs E1 and vs E8 LABEL-FREE BETTER (SEC); only vs E1 survives Holm (EXPL; a2c_*).
- C12 One-step error ranks trained models backwards; every check ranks compressed copies well (EXPL; x2_*).
- C13 X5: ONE-STEP BETTER and TRAINING-HORIZON BETTER over λ1 (SEC); first unresolved under wider intervals and Holm (EXPL); cause stated as likely; d4_F18_non_worse of d4_F18_worse_total had λ1≤0 (DESC); residual level with E8 on X5 (EXPL; x1_rJ_minus_E8).
- C14 AR1 LABEL-FREE BETTER (PRIM), carried by MLP and Transformer; in the RSSM E1 ranked better (DESC; s1_art_rssm_*). AR2 NOT SETTLED (PRIM). Pooled runs (n=38) NOT SETTLED; extended with F18 (n=135) LABEL-FREE BETTER (both SEC; s1_ar3_*).
- C15 CL1 WORSE WITH POLICY COMMANDS (PRIM; s1_cl1_*, s1_t_F18_rJ_rec/_pol), against E100 under recorded commands, driven by trained models (SEC CL9); CL2 NOT SETTLED (PRIM).
- C16 Noisy recorded commands: both UNCHANGED, within the 0.10 margin (SEC; s1_cl8_*); X5 residual again worse (SEC; s1_cl4_*).
- C17 λ1 through the policy: WORSE WITH POLICY COMMANDS (SEC; s1_cl5_*); ρ≈0 (DESC; s1_t_F18_lam1_cl).
- C18 Lower imagined return in less accurate simulators (SEC; s1_cl6_rho); no GRU rollout diverged under policy, closed-loop or noisy commands (DESC; s1_cl7_*).
- Later: C1–C18, run-time alarms (EARLY, A6), a2_n_lost (EXPL).

**Must not say**
- "the residual is the best check" without main pool, recorded commands, Holm, level on X5, weaker under the policy.
- "no difference", "similar", "as good as", "robust" for any NOT SETTLED (Q2, Q6, AR2, CL2, FT one-step).
- "holds across architectures" or "generalises" without the RSSM; anything beyond four architectures with few runs each.
- the growth rate "beats" or "matches" 8-step error.
- CL as error under policy commands: E100 is under recorded commands.
- imagined return as walking performance; anything about policies in physics, hardware or sim-to-real.
- a negative growth rate as a guarantee; "the released model is unstable".
- "fully trained", "converged"; TB as registered.
- pruning made "free"; K above 100% as better than the parent.
- "teacher forcing always fails"; "one-step error is useless"; "novel", "first".
- pruning or the command source "leaves λ1 unchanged" (Q6, CL2 NOT SETTLED; UNCHANGED only on X5 and noisy commands); UNCHANGED as "no effect".
- λ1 ranks mixed architectures together (AR3 n=38).
- Q1_E24, Q5_vs_E1, Q3_vs_E1 without "lost under two-level intervals".

## 8 Step 2 slot
- **Where:** end of Sec. 4, before Discussion (`\stepslot`), to become 4.3.
- **Question:** does a simulator's 2-second error, and do the two label-free checks, predict how well a policy trained inside it follows its velocity command in a physics simulator?
- **Holds:** a paragraph and a small table.
- **Sentences that change:** A8 becomes result plus scope (about 10 words must go: A3's robustness clause or A1); the introduction's scope sentence and contributions (one more); Discussion opening and checklist item 2; Limitations on "no controller tested"; the Conclusion's last sentence.

## 9 Author's voice
Introduction paragraph 1 and the two-questions paragraph; the Discussion's "why one-step checks mislead here"; the Conclusion; the title.

# W2 claims list: result-bearing sentences of the new main text, in order
Mark: N = new to the main text, C = changed wording or scope, K = kept. Status: PRIM, SEC, EXPL, DESC, EARLY.
For W3: `check_claims.py` asserts by description; rows marked N or C need a new or reworded assertion.

| # | Sec. | Quote (short) | Claim | Keys | Status | Mark |
|---|---|---|---|---|---|---|
| 1 | Abs | "were better one step ahead but {{ab_h100_ratio}} and {{N1overN8_E100}} times as wrong at 2 s" | TF and N=1 better at E1, worse at E100 | ab_h1_ratio<1, N1overN8_E1<1, ab_h100_ratio, N1overN8_E100 | DESC/EARLY | C |
| 2 | Abs | "also after longer training and in two of three other architectures" | TB 3/3 seeds; AR4 MLP 5/5, Transformer 5/5, RSSM 0/5 | s1_tb_nseeds, s1_ar4_*_both | DESC | N |
| 3 | Abs | "Pruning 30% ... kept {{K_u30_h1}} ..., {{K_u30_h100}} at 2 s" | K at h1, h100 (X5) | K_u30_h1, K_u30_h100 | DESC | K |
| 4 | Abs | "fine-tuning restored nearly all of it (registered)" | FT1 RESTORES, FT2 NEARLY AS SKILLED | ft_FT1_verdict, ft_FT2_verdict | PRIM | N |
| 5 | Abs | "growth rate ... better than one-step error (registered) on {{n_P}} GRU simulators and, averaged, over three other architectures (two carry it)" | Q1 LABEL-FREE BETTER; AR1 LABEL-FREE BETTER; MLP, Transformer lam1>E1, RSSM not | q1_verdict, s1_ar1_label, s1_art_*_lam1/E1 | PRIM (+DESC split) | C |
| 6 | Abs | "not separated from 8-step error (registered) and failed a secondary replication" | Q2 NOT SETTLED; Q3 vs E1 ONE-STEP BETTER | q2_verdict, q3_1_verdict | PRIM; SEC | C |
| 7 | Abs | "A kinematic residual ranked them best with recorded commands (descriptive; ... failed multiple-test correction)" | rho_rJ highest on F18; Q4_vs_E8 fails Holm | q4_1_rho_new > q2_rho_e8, q1_rho_lam, q1_rho_e1; a2c_q4_vs_e8_survives=no | DESC; EXPL | C |
| 8 | Abs | "and less well under RWM's policy (registered)" | CL1 WORSE WITH POLICY COMMANDS | s1_cl1_label | PRIM | N |
| 9 | 1 | "measured {{n_trained}} trained simulators and {{n_compressed}} compressed copies" | counts | n_trained, n_compressed | DESC | K |
| 10 | 1 | "The reversal persists after four times longer training and in two of three other architectures, and fine-tuning repairs the pruned copies (registered)" | as 2 and 4 | s1_tb_*, s1_ar4_*, ft_FT1/FT2 | DESC; PRIM | N |
| 11 | 3 | "These data are {{budget_pct}} of what the original model was trained on" | data share | budget_pct | DESC | K |
| 12 | 3 | "main pool (F18) ... {{n_P}} simulators ... five-split pool ... {{n_X}}" | pool sizes | n_P, n_X | DESC | C |
| 13 | 3 | "{{n_R}} rollout-trained ... {{n_T}} ... {{n_C}} compressed ... study AR used {{s1_ar_n}}" | model counts | n_R, n_T, n_C, s1_ar_n | DESC | C |
| 14 | 3 | "widens the intervals on correlations by a median factor of {{a2_median_ratio}}" | two-level width ratio | a2_median_ratio | EXPL | K |
| 15 | 4.1 | "rollout training beats teacher forcing at the horizon of use, and a longer training horizon helps" | ab_h100_ratio>1, N1overN8_E100>1 | ab_*, N1overN8_* | DESC | K |
| 16 | 4.1 | "teacher forcing's one-step error is {{ab_h1_pct_better}} lower (descriptive), as an earlier registered analysis also found" | TF better at h1 | ab_h1_pct_better, TMLR x3_h1_gap<0 | DESC; EARLY | K |
| 17 | 4.1 | "By 8 steps teacher forcing is worse on {{ab_h8_nworse}} of the 20 trajectories, and the gap grows with the horizon" | ratio grows 8<24<100<368 | ab_h8_nworse, ab_h*_ratio | DESC | C |
| 18 | 4.1 | "at 368 steps an earlier registered rule found rollout training better on all {{x3_n_eps}} held-out episodes" | sign test all episodes | x3_n_eps (x3_p in A1) | EARLY | K |
| 19 | 4.1 | "joint angles are off by {{ph_jdeg_h100_B}}° against {{ph_jdeg_h100_A}}°" | TF worse in joint angle at 2 s | ph_jdeg_h100_* | DESC | K |
| 20 | 4.1 | "Training with N=1 shows the same reversal" | N1overN8_E1<1, N1overN8_E100>1 | N1overN8_* | DESC | C |
| 21 | 4.1 | "Trained four times longer, teacher forcing was still better one step ahead and worse at 100 steps in each of {{s1_tb_nseeds}} seeds, while the training loss was still falling in {{s1_tb_fall_10000}} of {{s1_tb_n_10000}} runs" | TB conclusions | s1_tb_* | DESC | N |
| 22 | 4.1 | "better one step ahead and worse at 100 steps in {{s1_ar4_mlp_both}} of ... MLP ... {{s1_ar4_transformer_both}} of ... Transformer" | AR4 pairs | s1_ar4_mlp_*, s1_ar4_transformer_* | DESC | N |
| 23 | 4.1 | "In an RSSM ... better one step ahead in {{s1_ar4_rssm_tf_better_E1}} ... worse at 100 steps in {{s1_ar4_rssm_tf_worse_E100}} of them" | RSSM no reversal | s1_ar4_rssm_* | DESC | N |
| 24 | 4.1 | "positive for N=1 ({{N1_lam}}/s) and falls through ... to {{N32_lam}}/s" | lam1 monotone in N | N*_lam | DESC | K |
| 25 | 4.1 | "a rule committed before that analysis found the same order" | NGCAPS Q4 rho<0 | ng_q4_rho (now in A1) | EARLY | C |
| 26 | 4.1 | "Every rebuilt model trained on ... eight or more steps was contracting, and no teacher-forced one was" | sign of lam1 by method | stats.json D1 | DESC | K |
| 27 | 4.1 | "trained on {{scale_ratio}} times more data, it has λ1={{rel_lam}}/s" | released lam1>0 | scale_ratio, rel_lam | DESC | C (moved from 4.2) |
| 28 | 4.1 | "keeps {{K_u30_h1}} ... but {{K_u30_h100}} at 100 steps, and by 368 steps ... worse than predicting no change" | K_u30 h1/h100, h368<0 | K_u30_* | DESC | K |
| 29 | 4.1 | "restored the 100-step skill from {{ft_FT1_Knoft}} to {{ft_FT1_Kft}} (FINE-TUNING RESTORES SKILL)" | FT1 | ft_FT1_* | PRIM | K |
| 30 | 4.1 | "NEARLY AS SKILLED as their parents fine-tuned the same way ({{ft_FT2_K}} {{ft_FT2_ci}})" | FT2 | ft_FT2_* | PRIM | K |
| 31 | 4.1 | "Skill above 100% mostly reflects the extra training, not the pruning (descriptive)" | P vs CTRL K<100% while U30F vs P >100% | ft_K_P_vs_CTRL_*, ft_K_U30F_vs_P_* | DESC | N |
| 32 | 4.1 | "at 30% the registered test could not separate the shift ({{lamshift_u30}}/s ...) from zero (NOT SETTLED; secondary)" | Q6 | q6_verdict, lamshift_u30 | SEC | C |
| 33 | 4.1 | "no copy pruned by 10–30% has λ1>0" | lampos_u10/u20/u30=0 | lampos_* | DESC | K |
| 34 | 4.1 | "Rounding weights to 8 bits keeps {{K_int8w_h100}} of the 100-step skill" | 8-bit K | K_int8w_h100 | DESC | N (was table only) |
| 35 | 4.1 | "spread ranks error well (Spearman {{dis_spear_h100}}) but is too narrow, by a factor of {{k_h1}} one step ahead and {{k_h100}} at 100 steps" | k grows with horizon | dis_spear_h100, k_h1, k_h100 | DESC | C |
| 36 | 4.1 | "Each head's own variance collapses" | variance collapse | (A6) | DESC/EARLY | K |
| 37 | 4.2 | "one-step error ranked ... poorly ({{q1_rho_e1}}) and the growth rate well ({{q1_rho_lam}})" | rho values | q1_rho_* | DESC | K |
| 38 | 4.2 | "The first primary question found the growth rate better (LABEL-FREE BETTER)" | Q1 | q1_verdict | PRIM | K |
| 39 | 4.2 | "could not separate it from the 8-step error (NOT SETTLED), which ranked ahead of it at 24 steps (secondary)" | Q2; Q2_E24 TRAINING-HORIZON BETTER | q2_verdict, q2_e24_verdict | PRIM; SEC | C |
| 40 | 4.2 | "one-step error ranks the {{x2_tc_n}} trained models backwards ({{x2_tc_E1}}; exploratory); every check ranks the compressed copies well" | x2 split | x2_tc_E1 (CI<0), x2_c_* | EXPL | K |
| 41 | 4.2 | "The kinematic residual ranked best ({{q4_1_rho_new}}; descriptive)" | rJ highest rho F18 | q4_1_rho_new | DESC | C |
| 42 | 4.2 | "registered rules favoured it over both E1 and E8 (secondary), but only its lead over E1 survives Holm's correction (exploratory)" | Q4 both LABEL-FREE BETTER; Holm | q4_*_verdict, a2c_q4_* | SEC; EXPL | C |
| 43 | 4.2 | "most of the {{n_X_fold}} simulators per split are compressed copies" | 40 of 51 | n_X_fold | DESC | C |
| 44 | 4.2 | "one-step error ranked better than it (ONE-STEP BETTER; secondary), and the 8-step error better still" | Q3 both | q3_1_verdict, q3_8_verdict | SEC | K |
| 45 | 4.2 | "The first reading is not resolved under the wider intervals, but the growth rate did not beat one-step error in any reading" | two-level loses Q3_vs_E1; a2b also negative | a2_q3_vs_e1_holds, a2b_q3_vs_e1_* | EXPL | C |
| 46 | 4.2 | "of the {{d4_F18_worse_total}} main-pool simulators worse than the floor at 100 steps, {{d4_F18_non_worse}} had λ1≤0" | d4 | d4_F18_* | DESC | K |
| 47 | 4.2 | "level with the 8-step error here ({{x1_rJ_minus_E8}} ...; exploratory)" | X5 rJ vs E8 CI spans 0 | x1_rJ_minus_E8* | EXPL | K |
| 48 | 4.2 | "AR1 found the growth rate better than one-step error (LABEL-FREE BETTER), carried by the MLP and the Transformer" | AR1; per-arch rho | s1_ar1_label, s1_art_* | PRIM; DESC | N |
| 49 | 4.2 | "In the RSSM, one-step error ranked them better ({{rw_t_rssm_E1}} against {{rw_t_rssm_lam1}}; descriptive)" | RSSM E1>lam1 | rw_t_rssm_* | DESC | N |
| 50 | 4.2 | "AR2 ... was NOT SETTLED" | AR2 | s1_ar2_label | PRIM | N |
| 51 | 4.2 | "neither comparison was settled and the growth rate did not rank the runs (ρ={{s1_art_pooled_lam1}}); with the main pool added, AR1 was LABEL-FREE BETTER (both secondary)" | AR3 pooled NOT SETTLED x2; extended AR1 | s1_ar3_* | SEC | N |
| 52 | 4.2 | "The residual then ranked less well (WORSE WITH POLICY COMMANDS; {{rw_t_F18pol_rJ}} against {{q4_1_rho_new}})" | CL1 | s1_cl1_*, rw_t_F18pol_rJ | PRIM | N |
| 53 | 4.2 | "a drop carried by the trained models (secondary) and seen again on the five-split pool (secondary)" | CL9 trained WORSE, compressed UNCHANGED; CL4 CL1 WORSE | s1_cl9_*, s1_cl4_cl1_label | SEC | N |
| 54 | 4.2 | "The growth rate's change was NOT SETTLED" | CL2 | s1_cl2_label | PRIM | N |
| 55 | 4.2 | "With small noise ... both checks were UNCHANGED BY THE COMMAND SOURCE (within ±{{s1_margin}}; secondary)" | CL8 both | s1_cl8_cl1_label, s1_cl8_cl2_label | SEC | N |
| 56 | 4.2 | "(closed loop), the growth rate was WORSE WITH POLICY COMMANDS (secondary) and did not rank the simulators (ρ={{rw_t_F18pol_lam1cl}})" | CL5 | s1_cl5_label, rw_t_F18pol_lam1cl | SEC; DESC | N |
| 57 | 4.2 | "Simulators with higher error gave the policy lower imagined return (ρ={{s1_cl6_rho}}; secondary)" | CL6 rho<0 | s1_cl6_* | SEC | N |
| 58 | 4.2 | "none of the {{s1_cl7_nmodels}} GRU simulators diverged under policy, closed-loop or noisy commands" | CL7 | s1_cl7_ndiv=0 | DESC | N |
| 59 | 5 | "the simulators they produced amplify them (λ1>0)" | TF and N=1 lam1>0 | N1_lam, D1 | DESC | N |
| 60 | 5 | "One-step error therefore ranked training choices backwards, while it ranked copies of the same parent well" | x2_tc_E1<0, x2_c_E1>0 | x2_* | EXPL | N |
| 61 | 5 | checklist (3): "the residual ranked RWM-style simulators best with recorded commands but less well under a policy; a positive growth rate flagged the one-step-trained simulators, a negative one certifies nothing, and growth rates did not rank different architectures together" | as 41, 52, 26, 46, 51 | as cited | mixed | C |
| 62 | 5 | checklist (4): "run-time alarms, which lost to a step counter or a fixed schedule in earlier registered analyses" | NGCAPS WORSE THAN COUNTER; ICPAIAS SCHEDULE BETTER | ng_q3_*, icp_h7_* | EARLY | C |
| 63 | Lim | "leave {{a2_n_lost}} of {{a2_n_tests}} registered verdicts unresolved (exploratory)" | two-level | a2_n_lost, a2_n_tests | EXPL | K |
| 64 | Lim | "Each other architecture has {{rw_t_rssm_n}} or {{rw_t_mlp_n}} runs, and the longer-training check has {{s1_tb_nseeds}} seeds and no intervals" | counts | rw_t_*_n, s1_tb_nseeds | DESC | N |
| 65 | 6 | "one-step checks favoured teacher forcing, one-step training and a pruned model that, without fine-tuning, loses most of its skill by 2 s" | as 1, 28 | | DESC | K |
| 66 | 6 | "on the main pool the growth rate ranked RWM-style simulators better than one-step error but was not separated from 8-step error, and a negative value certifies nothing" | Q1, Q2, d4 | | PRIM; DESC | C |
| 67 | 6 | "the kinematic residual ranked them best with recorded commands and less well under a policy" | 41, CL1 | | DESC; PRIM | C |
| T2 | 4.2 | Table 2: rho with E100; new columns F18 policy (CL), architectures (AR) | | q*_rho_*, rw_t_*, x1_rJ | DESC (X5 r_J EXPL) | N |
| T3 | 4.2 | Table 3: registered verdicts Q1–Q4, AR1–AR2, CL1–CL2, FT1–FT2; † fails Holm (Q3 vs E1, Q4 vs E8) | | as rows | PRIM/SEC; † EXPL | N |

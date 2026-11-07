# I2 claims: new or changed result-bearing sentences (for I3's check_claims.py and air-claim-checker)
Format: section | short quote | claim | keys | status. Values: results/paper_numbers.json.

## Abstract
- Abs | "{{ab_h100_ratio}} and {{N1overN8_E100}} times as wrong at 2 s, the {{ext_rwm_imag_steps}}-step horizon RWM imagines" | TF and N=1 worse than N=8 at 100 steps; RWM imagines 100 steps | ab_h100_ratio, N1overN8_E100, ext_rwm_imag_steps | descriptive; external fact
- Abs | "pruning 30% ... kept {{K_u30_h1}} ... but {{K_u30_h100}} at 2 s" | unchanged values | K_u30_h1, K_u30_h100 | descriptive
- Abs | "fine-tuning restored the pruned copies' skill (registered)" | FT1 verdict "fine-tuning restores skill" | ft_FT1_verdict | registered (FT, primary)
- Abs | "growth rate ranked them ... better than one-step error, but was not separated from 8-step error and failed a replication" | Q1 label-free better; Q2 not settled; Q3 vs E1 one-step better | q1_verdict, q2_verdict, q3_1_verdict | registered (Q1,Q2 primary; Q3 secondary)
- Abs | "a kinematic residual ranked them better than one-step error, but less well under policy-chosen commands" | Q4 vs E1 label-free better; CL1 worse with policy commands | q4_1_verdict, s1_cl1_label | registered (Q4 secondary; CL1 primary)
- Abs | "We therefore recommend checking 8 or more steps ahead ... with these checks as a fallback" | recommendation; rests on Q2, AR2 not settled and Q3 vs E8 training-horizon better | q2_verdict, s1_ar2_label, q3_8_verdict | recommendation (no new result)

## Introduction / Related Work
- Intro | "trains a walking policy inside it in imagined rollouts of {{ext_rwm_imag_steps}} steps (2 s) per update [RWM Table S11], as does RWM-U ({{ext_rwmu_imag_steps}} steps)" | 100 and 100 | ext_rwm_imag_steps, ext_rwmu_imag_steps | external facts (quotes still flagged unchecked against raw HTML)
- Intro | "at the horizon of use they are worse, also after longer training and in two of three other architectures. Fine-tuning repairs the pruned copies." | as before (labels dropped from the bullet) | s1_ar4_*, s1_tb_nseeds, ft_FT1_verdict | descriptive / registered
- RW | "iKCE ... asks, without a recorded future, whether imagined positions follow from imagined velocities ...; it diagnosed one model ({{ext_ikce_task}}), whereas we use the residual to rank many simulators" | iKCE: one model, DreamerV3 walker-walk, no ranking | ext_ikce_task, ext_ikce_n_models; external_facts ikce_ranks_models == False | external fact

## Section 3
- Data | "The main pool's training episodes hold {{ext_ours_transitions}} transitions, {{ext_budget_fraction}} of the {{ext_rwm_pretrain}} RWM pretrained on [Table 1]" | 7,991 / 6,000,000 = 0.13% | ext_ours_transitions, ext_budget_fraction, ext_rwm_pretrain | external fact + stored count
- Models | "Counted once, the {{n_trained}} trained and {{n_compressed}} compressed GRU simulators fill both pools: main pool's {{n_P}} are {{im_f18_trained}} trained and {{n_C}} copies, five-split pool's {{n_X}} are {{im_x5_trained}} and {{n_compressed}}, {{im_both_trained}} trained models and the {{n_C}} copies are in both ...; study AR adds {{s1_ar_n}}" | 57+40=97; 55+200=255; 57+55-11=101; F18 copies subset of X5 copies; AR 38 not in 101 | n_trained, n_compressed, n_P, n_X, n_C, im_f18_trained, im_x5_trained, im_both_trained, s1_ar_n | count (asserted in im_numbers.py)
- Stats | "registrations are commits in our own version control, ... times are self-reported ... order but no independent timestamp, and the answer to Q1 was largely predictable from earlier analyses (A2)" | statement of limits | (A2 keys pre_ng_*, pre_icmai_*) | statement
- Stats | "Results are marked registered / descriptive / exploratory" | status kinds stated once | - | statement

## Section 4.1
- 4.1 | "RWM and RWM-U imagine {{ext_rwm_imag_steps}} steps" ; Fig. 1 caption "Vertical line: the {{ext_rwm_imag_steps}}-step horizon RWM and RWM-U imagine" | 100 | ext_rwm_imag_steps (== ext_rwmu_imag_steps) | external fact
- 4.1 | "agree with RWM's ablation on {{ext_budget_fraction}} of its pretraining data" | 0.13% | ext_budget_fraction | external fact
- 4.1 | "Longer training and other networks gave the same reversal (descriptive) ... in each of {{s1_tb_nseeds}} seeds (A10), as in 5 of 5 MLP, 5 of 5 Transformer; RSSM better one step in 5 of 5 but worse at 100 steps in 0" | unchanged values | s1_tb_nseeds, s1_ar4_* | descriptive
- 4.1 | "trained on unrecorded data (RWM reports {{scale_ratio}} times our transitions)" | 750 = 6M / 7,991 rounded | scale_ratio | external fact, weakened
- 4.1 | "fine-tuning ... restored ... from {{ft_FT1_Knoft}} to {{ft_FT1_Kft}}, and the fine-tuned copies were nearly as skilled as their parents fine-tuned the same way (... Table 3)" | FT1 restores, FT2 nearly as skilled | ft_FT1_verdict, ft_FT2_verdict | registered (plain words)
- 4.1 | "(not settled; secondary, Table 3)" growth-rate shift at 30% | Q6 not settled | q6_verdict | registered secondary

## Section 4.2
- 4.2 | "The first primary question found the growth rate better, and the second could not separate it from the 8-step error, which ranked ahead of it at 24 steps (Table 3)" | Q1 label-free better; Q2 not settled; Q2_E24 training-horizon better | q1_verdict, q2_verdict, S["Q2_E24"] | registered (plain words)
- 4.2 | "Every check ranked the main pool's {{im_c_n}} compressed copies well ({{im_c_lam1}} to {{im_c_E8}})" | c: E1 .89, E8 .96, lam1 .82, rJ .94 all >= .82 | im_c_* | EXPLORATORY
- 4.2 | "the residual ranked its {{im_a_n}} rollout-trained simulators better than 8-step error did ({{im_a_rJ}} against {{im_a_E8}})" | a: rJ .85 > E8 .38, diff +0.47 [+0.13,+0.81] excludes 0 | im_a_rJ, im_a_E8, im_a_drJ(_ci) | EXPLORATORY
- 4.2 | "On the five-split pool's {{im_b_n}} rollout-trained simulators ... the growth rate failed ({{im_b_lam1}}) and the residual's lead over 8-step error was unclear" | b: lam1 -0.21 (CI includes 0); rJ-E8 +0.20 [-0.13,+0.56] includes 0 | im_b_lam1(_ci), im_b_drJ_ci | EXPLORATORY
- 4.2 | "one-step error ranked better than it, and the 8-step error better still (secondary, Table 3)" | Q3 one-step better; training-horizon better | q3_1_verdict, q3_8_verdict | registered secondary
- 4.2 | "The growth rate ranked better than one-step error (AR1) ... The residual against the 8-step error (AR2) was not settled (Table 3)" | AR1 label-free better; AR2 not settled | s1_ar1_label, s1_ar2_label | registered primary
- 4.2 | "Ranked across architectures together, the growth rate did not rank the runs (rho = {{s1_art_pooled_lam1}}; secondary, A9)" | unchanged | s1_art_pooled_lam1 | secondary
- 4.2 | "The residual then ranked less well (... ) ...; the growth rate's change was not settled (Table 3)" | CL1 worse with policy commands; CL2 not settled | s1_cl1_label, s1_cl2_label | registered primary
- 4.2 | "With small noise ... both checks were unchanged, and ... closed loop, the growth rate did not rank the simulators (secondary, A8)" | CL8 unchanged; CL5; lam1cl rho -0.01 | s1_cl8_cl1_label, s1_cl5_label, rw_t_F18pol_lam1cl | secondary
- 4.2 | "This signal needs a policy and a reward but no recording, and was nominally the strongest under policy commands, ahead of the residual's {{rw_t_F18pol_rJ}} (descriptive); not registered as a check; candidate for a follow-up study" | |CL6 rho| .86 = im_d_negret > rw_t_F18pol_rJ .83 > rw_t_F18pol_lam1 .76 > |lam1cl| | s1_cl6_rho, im_d_negret, rw_t_F18pol_rJ, rw_t_F18pol_lam1, rw_t_F18pol_lam1cl | CL6 secondary; comparison descriptive

## Discussion / Conclusion
- Disc (1) | "no label-free check beat 8-step error in a primary test (Table 3)" | Q2 and AR2 (the primary tests against E8) not settled | q2_verdict, s1_ar2_label | registered primary
- Disc (3) | "Within one training method it [residual] beat 8-step error on the main pool's rollout-trained simulators, while the growth rate failed on the five-split pool's (exploratory)" | as 4.2 | im_a_drJ_ci, im_b_lam1 | EXPLORATORY
- Disc (4) | "re-synchronising on a score lost to a fixed schedule (A6)" | earlier registered verdict "schedule better" | (appendix A6 key) | earlier registered
- Concl | "the growth rate beat one-step error on the main pool but failed its replication, and the kinematic residual ranked best with recorded commands (descriptive) but less well under a policy" | Q1, Q3, q4_1_rho_new highest of Table 2 F18 column, CL1 | q1_verdict, q3_1_verdict, q4_1_rho_new, s1_cl1_label | registered / descriptive

## Appendix A4 (new paragraph and Table tab:app-within): all EXPLORATORY; keys im_{a,b,c}_*; A6: released model "unrecorded data (RWM reports {{scale_ratio}} times as many transitions as our main pool's)".

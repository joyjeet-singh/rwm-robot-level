# W0 evidence digest (values and labels copied from the result files; nothing computed)
Status codes: PRIM = registered primary, SEC = registered secondary, EXPL = exploratory, DESC = descriptive, EARLY = earlier study (reused, not re-registered).
Sources: SR = results/stats_report.txt, FT = results/ft_report.txt, S1 = results/s1_report.txt, TB = results/s1_budget.txt, RX = results/revision_exploratory.txt.
Keys are `{{key}}` names in results/paper_numbers.json (family given when several share a stem).
Pool P = F18 trained+compressed (97 simulators); X5 = five-split replication; R = rollout-trained, C = compressed.

## A. Training choices, horizon, compression, uncertainty (4.1)
E1  | Teacher forcing (TF) is better one step ahead, worse at the horizon of use (ratio of 100-step error 2.0x; N=1 vs N=8 2.6x) | EARLY + descriptive on X5 | ab_h1_*, ab_h100_ratio, N1overN8_E100 | stats.json D3; TMLR x3_h1_gap (-0.0090)
E2  | One-step error ranks training choices backwards | EARLY / DESC | ab_*, N1overN8_* | stats.json
E3  | TF position and joint-angle error exceed rollout training's over 2 s; neither beats constant velocity on position over 2 s | DESC | ph_*, dist_h100 | figure_facts.json
E4  | Growth rate lambda_1 falls with training horizon N (rank corr. -0.97 over N models; committed before analysis) | EARLY (NGCAPS) | ng_q4_rho, ng_q4_ci, N*_lam | NGCAPS/results/main.json
E5  | Every rebuilt model trained on 8+ step rollouts contracts; every TF/N=1 model expands | DESC | lampos_*, rel_lam | stats.json
E6  | Pruning 30% without retraining: K 90% at 1 step, 83% at 24, 39% at 100, -41% at 368 (worse than no change) | DESC (K = skill retained) | K_u30_h1/h24/h100/h368 | stats.json D2
E7  | Moderate pruning (10-30%) adds error but barely changes lambda_1: 30% shift +0.03/s; 10 and 20% similar, intervals barely exclude zero (exploratory, A5); no copy pruned 10-30% has lambda_1>0; heavier pruning and unit removal raise it | DESC / EXPL | lamshift_u10/u20/u30/u50/s128, lampos_* | stats.json
E8  | 8-bit quantisation keeps most skill (K 97.0% at 100 steps); fp16 100% | DESC | K_int8w_*, K_fp16_* | stats.json
E9  | Width-192 copy without fine-tuning has K below one half at every horizon | DESC | K_s192_*, perc_K_first_w192_* | stats.json; perc_* from the narrow-copy runs
E10 | Fine-tuning (500 its) restores 30%-pruned skill: FT1 d=0.639 [0.544, 0.760], U30F 1.022, U30 0.383 -> FINE-TUNING RESTORES SKILL | PRIM (FT, 97.5%) | ft_FT1_* | FT
E11 | FT2 K_ctrl(U30F,100)=0.999 [0.995, 1.004] -> NEARLY AS SKILLED | PRIM (FT, 97.5%) | ft_FT2_* | FT
E12 | FT1_h24 d=0.189 [0.166, 0.216] RESTORES; FT2_h24 1.000 [0.998, 1.001] NEARLY AS SKILLED | SEC (95%, uncorrected) | ft_FT1_h24_*, ft_FT2_h24_* | FT
E13 | FT1_h368 d=1.346 [1.021, 1.764] RESTORES; FT2_h368 0.994 [0.979, 1.006] NEARLY AS SKILLED | SEC | ft_FT1_h368_*, ft_FT2_h368_* | FT
E14 | 50% arm: FT1_u50_h100 d=1.944 [1.713, 2.285] RESTORES; FT2_u50_h100 0.990 [0.984, 0.997] NEARLY AS SKILLED | SEC | ft_FT1_u50_h100_*, ft_FT2_u50_h100_* | FT
E15 | One-step reading of fine-tuned copy: K_orig h1 1.023, h100 1.022; d=0.001 [-0.012, 0.009] -> NOT SETTLED | SEC | ft_onestep_* | FT
E16 | Fine-tuning changes lambda_1: U30F-P -0.0238 [-0.049, -0.002] DECREASES; U30-P 0.0038 [-0.062, 0.068] NOT SETTLED | SEC | ft_change_lam1_* | FT
E17 | Residual r_J: U30F-P -0.0521 [-0.093, -0.012] DECREASES; U30-P 2.2521 [1.998, 2.500] INCREASES | SEC | ft_change_rJ_* | FT
E18 | K above 100% mostly reflects fine-tuning itself: control P vs CTRL K = 0.975/0.990/0.990/0.978/0.956 (h1/8/24/100/368); U30F vs P 1.023/1.010/1.009/1.022/1.039 | DESC | ft_K_P_vs_CTRL_*, ft_K_U30F_vs_P_* | FT
E19 | U30 vs P K: 0.899/0.901/0.820/0.383/-0.307 (h1/8/24/100/368; h368 CI [-0.716, 0.007]) | DESC | ft_K_U30_vs_P_* | FT
E20 | Released ensemble's spread: disagreement among heads ranks 100-step error (Spearman +0.90); coverage 83.5% at 100 steps vs 95.45% nominal for +-2 sigma; within band at 6 of 6 horizons (central) | DESC | dis_spear_h100, cov2_perh_h100, cov_nom2, x7_summary | stats.json
E21 | Variance collapse: each head's variance shrinks (loss minimised at sigma=0); released heads too small by a factor that grows with horizon; per-horizon multiplier restores 1-sigma coverage only | DESC / EARLY | synth_under 21.7, synth_nll 0.98, synth_span 25, cov1_*, cov2_perh_h100 | rwm_repro paper_numbers.json (meaning of synth_* to confirm there)
E22 | Stale command (late command feed): one-step error +55%; 100-step change 8.2% not resolved on 20 trajectories | DESC | stale_h1, stale_h100, stale_*_ci | stats.json
E23 | Rebuild matches the release: gradients equal (max diff 0); 352 of 9,961 training windows splice episode ends (win_cross/win_naive) | EARLY (verification) | diff_*, win_* | rwm_repro

## B. Which check predicts the 100-step error (4.2); registered in prereg/PREREG.md
E24 | Q1 (PRIM): LABEL-FREE BETTER delta/mean=+0.4780 ci97.5=[0.196, 0.789]; rho_new 0.796, rho_base (E1) 0.318 | PRIM | q1_verdict, q1_delta, q1_ci975, q1_rho_lam, q1_rho_e1 | SR
E25 | Q2 (PRIM): NOT SETTLED delta/mean=-0.0400 ci97.5=[-0.201, 0.141]; rho_new 0.796, rho_base (E8) 0.836 | PRIM | q2_verdict, q2_delta, q2_ci975 | SR
E26 | Q1_E24: LABEL-FREE BETTER +0.3574 [0.132, 0.600]; Q2_E24: TRAINING-HORIZON BETTER -0.1620 [-0.251, -0.081] | SEC (95%) | q1_e24_*, q2_e24_* | SR
E27 | Q1_E368: LABEL-FREE BETTER +0.5619 [0.310, 0.831]; Q2_E368: NOT SETTLED +0.0535 [-0.086, 0.215] | SEC | q1_e368_*, q2_e368_* | SR
E28 | Q3 vs E1 (X5): ONE-STEP BETTER -0.1170 [-0.216, -0.026]; vs E8: TRAINING-HORIZON BETTER -0.2843 [-0.370, -0.217] (growth rate replication failed) | SEC | q3_1_*, q3_8_* | SR
E29 | Q4 vs E1: LABEL-FREE BETTER +0.6330 [0.411, 0.880]; vs E8: LABEL-FREE BETTER +0.1150 [0.022, 0.233] (rho_new 0.951 vs E8 0.836) (kinematic residual r_J) | SEC | q4_1_*, q4_8_* | SR
E30 | Q5 (without one-step-trained models, 88 models): vs E1 LABEL-FREE BETTER +0.2683 [0.043, 0.512]; vs E8 NOT SETTLED +0.0120 [-0.160, 0.198] | SEC | q5_1_*, q5_8_* | SR
E31 | Q6 (pruning w/o retraining less stable): NOT SETTLED +0.0272 [-0.022, 0.075] | SEC | q6_verdict, q6_mean, q6_ci | SR
E32 | Holm over Q3-Q6 (family-wise 5%): only Q3_vs_E8 and Q4_vs_E1 survive; Q4_vs_E8 p_holm=0.0530, Q3_vs_E1 0.0556, Q5_vs_E1 0.0556 do not (within 2 MC SE) | EXPL | a2c_* (n_survive 2) | RX A2c
E33 | Wider two-level intervals (clusters x trajectories): 3 of 13 registered verdicts lost (Q1_E24, Q5_vs_E1, Q3_vs_E1 exclude 0 no more); Q1 97.5% two-level [+0.09, +1.11] x1.7; median width ratio 1.5 | EXPL | a2_n_lost, a2_n_tests, a2_median_ratio | RX A2a
E34 | X5 without split {1,8}: Q3_vs_E1 -0.143, Q3_vs_E8 -0.300 hold; rJ vs E8 +0.022 [0.002, 0.048]* (five splits +0.011 [-0.010, 0.034]); growth rate still no better than E1 | EXPL | a2b_* | RX A2b
E35 | r_J over 100 steps minus E8, vs E100: F18 all +0.109 [+0.035, +0.205]*; X5 +0.013 [-0.003, +0.033]; r_J over 8 steps minus E8: F18 all -0.006 [-0.076, 0.063], X5 -0.073 [-0.120, -0.040]* | EXPL | a1_*, a1t_* | RX A1
E36 | Horizon-matched growth rate: lam1_8 vs E100 F18 all +0.21 [-0.03, 0.43]; lam1_100 +0.77; lam1_368 +0.80; r_J_100 +0.94 | EXPL | a1_*, a1t_* | RX A1
E37 | Split-half (check from one trajectory, E100 from the other): rho barely changes (F18 all E8 +0.77, lam1 +0.73, rJ +0.91; lam1 loses 0.044 [-0.087, -0.013]*) | EXPL | a3_* | RX A3
E38 | Split by what differs: one-step error ranks the 57 trained F18 models backwards (rho -0.50, CI [-0.71, -0.20]); every check ranks the 40 compressed copies well (E1 +0.89) | EXPL | x2_tc_*, x2_c_* | RX A1; stats.json
E39 | r_J best on the five-split pool, level with E8 (r_J minus E8 +0.01 [-0.01, +0.03]) | EXPL | x1_rJ, x1_rJ_minus_E8 | stats.json
E40 | Of 17 F18 simulators worse than holding the last state at 100 steps, 15 have lambda_1<=0 (so a negative growth rate certifies nothing); of 12 with lambda_1>0, 2 are worse | DESC | d4_F18_worse_total, d4_F18_non_worse, d4_F18_pos_n, d4_F18_pos_worse | stats.json
E41 | Released model (about 750x more data): low one-step and 100-step error (0.083, 0.40), yet lambda_1 +0.70/s on these trajectories (not contracting); training rows unrecorded, so not held-out | DESC | scale_ratio, rel_E1, rel_E100, rel_lam | stats.json
E42 | Regret of picking the model by each check (100-step error above the best): E1 0.33, E8 0.016, lambda_1 0.00048, r_J 0.0057 (regret_E1_ratio 2.8) | DESC | regret_E1, regret_E8, regret_lam1, regret_rJ | stats.json
E43 | Knowledge before registration: growth rate ranked trained F18 models (NGCAPS); residual ranked trained five-split models (ICMAI/ICPAIAS) | EARLY | pre_ng_RT_*, pre_icp_*, pre_icmai_* | NGCAPS, ICPAIAS, ICMAI results
E44 | Run-time signals (label-free AUROC): physics residual 0.54, disagreement 0.59 (icp_*); NGCAPS growth-signal AUROC 0.50; none above 0.65 on A-ens; re-synchronising on a score adds error versus a fixed schedule | DESC / EARLY (A6) | icp_auc_phys, icp_auc_dis, ng_q3_auc | ICPAIAS, NGCAPS

## C. Revision round 1 (revision/REPORT.md) stands behind E32-E39 (all exploratory, nothing registered; Q1-Q6 reproduce to 2e-16)

## D. Step 1 (prereg/PREREG_S1.md; checks S1-C1..C5 all PASS) -- appendix A8-A10 only; almost absent from main text
E45 | CL1 (PRIM): WORSE WITH POLICY COMMANDS d=-0.120 ci97.5=[-0.220, -0.052]; rho_pol +0.832, rho_rec +0.951 (F18, n=97) | PRIM | s1_cl1_* | S1
E46 | CL2 (PRIM): NOT SETTLED d=-0.032 ci97.5=[-0.113, +0.041]; rho_pol +0.764, rho_rec +0.796 | PRIM | s1_cl2_* | S1
E47 | CL3 rank agreement recorded vs policy: lam1 +0.930 [0.873, 0.961]; rJ +0.832 [0.728, 0.897] | SEC | s1_cl3_* | S1
E48 | CL4 (X5): CL1 WORSE WITH POLICY COMMANDS d=-0.063 [-0.105, -0.033]; CL2 UNCHANGED BY THE COMMAND SOURCE d=+0.011 [-0.042, 0.064] | SEC | s1_cl4_* | S1
E49 | CL5 closed-loop lam1: WORSE WITH POLICY COMMANDS d=-0.809 [-1.034, -0.564], rho_cl=-0.012 (did not rank the simulators) | SEC | s1_cl5_* | S1
E50 | CL6 imagined return vs E100: rho=-0.857 [-0.909, -0.770] n=97 (return fell as error rose) | SEC | s1_cl6_* | S1
E51 | CL7 no model diverged under policy, closed-loop or noisy commands (F18 0/0/0 of 97; X5 0/0/0 of 255) | DESC | s1_cl7_* | S1
E52 | CL8 noisy commands: CL1 UNCHANGED d=-0.002 [-0.011, 0.006]; CL2 UNCHANGED d=-0.029 [-0.074, 0.003] | SEC | s1_cl8_* | S1
E53 | CL9 trained (n=57): CL1 WORSE d=-0.285 [-0.492, -0.137], CL2 NOT SETTLED; compressed (n=40): CL1 UNCHANGED d=-0.021, CL2 NOT SETTLED | SEC | s1_cl9_* | S1
E54 | Two-level (57 clusters x 4 trajectories): CL1 ci97.5=[-0.282, -0.002]; CL2 [-0.139, +0.112] | EXPL | s1_tl_* | S1
E55 | Released model (reference only): lam1 pol +1.597 rec +0.700; rJ pol +0.701 rec +3.279; diverged pol 0 of 20 | DESC | s1_rel_* | S1
E56 | AR1 (PRIM): LABEL-FREE BETTER d=+0.796 ci97.5=[+0.368, +1.025]; per architecture (MLP, RSSM, Transformer) rho_check=[0.841, 0.259, 0.868], rho_base=[-0.692, 0.881, -0.61] | PRIM | s1_ar1_* | S1
E57 | AR2 (PRIM): NOT SETTLED d=+0.110 ci97.5=[-0.081, +0.356]; rho_check [0.764, 0.937, 0.984] vs rho_base [0.764, 0.86, 0.731] | PRIM | s1_ar2_* | S1
E58 | AR3 pooled with RWM-style GRU (n=38): AR1 NOT SETTLED d=-0.164 [-0.793, 0.561]; AR2 NOT SETTLED d=-0.202 [-0.526, 0.079]; extended (n=135): AR1 LABEL-FREE BETTER +0.467 [0.230, 0.711]; AR2 NOT SETTLED +0.049 | SEC | s1_ar3_* | S1
E59 | AR4 teacher forcing vs rollout (5 pairs each): MLP and Transformer TF better at E1 5/5, worse at E100 5/5; RSSM better at E1 5/5, worse at E100 0/5 | DESC | s1_ar4_* | S1
E60 | Diverged under recorded commands: MLP 3 runs, RSSM 0, Transformer 0 (teacher-forced runs) | DESC | s1_ar_ndiv, s1_ar_div_* | S1
E61 | Training length (Study TB; 3 seeds, no intervals): still falling 6/6 at 2,500 its, 5/6 at 10,000; at 10,000 TF still better at 1 step, worse at 100, lam1>0 (TF) in 3/3 seeds | DESC | s1_tb_* | S1 (s1_report.txt, s1_budget.txt)
E62 | Study TB, seed means (rollout 2500 -> 10000): E100 0.251 -> 0.181; E368 0.335 -> 0.201; TF E100 0.576 -> 0.464; TF E368 2.161 -> 0.960; lam1 rollout -0.425 -> -0.740; TF 0.675 -> 1.531 | DESC | s1_tb_* | S1
E63 | Control P vs CTRL (+500 its) K = 0.975/0.990/0.990/0.978/0.956 | DESC | s1_tb_K_min/K_max, ft_K_P_vs_CTRL_* | S1; FT
E64 | Step 1 job ran 11 h, started after registration commit (2026-11-06 12:02 UTC) | process | s1_job_hours, s1_reg_time, s1_reg_commit | S1 / s1_budget.txt

## Notes for the rewrite
- The strongest registered results: E24 (Q1), E10-E11 (FT1, FT2), E45 (CL1), E56 (AR1). Not settled and must stay so: E25, E31, E46, E57.
- E24 and E25 are scope-limited to pool P (F18, 97 simulators, recorded commands, one robot, one gait, GRU architecture); E56-E58 add MLP/Transformer/RSSM.
- E32-E39, E54 are exploratory; E1-E9, E18-E23, E41, E51, E55, E59-E62 descriptive; label them wherever they appear (abstract included).
- Keys not individually confirmed: E55 s1_rel_* (real names: s1_rel_lam1_pol, s1_rel_lam1_rec, s1_rel_rJ_pol, s1_rel_rJ_rec, s1_rel_ntraj), E21 meaning of synth_*. Grep paper_numbers.json before use.

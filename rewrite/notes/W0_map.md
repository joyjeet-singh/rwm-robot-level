# W0 map of the current paper (before the rewrite)
Line ranges are in `paper/body.template.tex` (369 lines) and `paper/appendix.template.tex` (530 lines). Words = prose only
(tables, figures, `\cite`, `\ref` excluded; each `{{key}}` counted as 1). Regenerate with: `./py .cache/map.py <file>` (not committed).
Build now: single column 19 pp, two-column check 6 pp (NO free space), reserve check (0.30 slot) 7 pp, abstract 198 words.
The Step 2 slot is `\stepslot` (empty in normal builds); with none in the body, the reserve build puts it before `\section{Discussion}`.
Main text prose is about 4,000 words in 6 pages. Step 1 (CL, AR) is in the main text only in 3 sentences (L47, L340, L359-361), no table.

## body.template.tex
| Lines | Words | Keys | Part |
|---|---|---|---|
| 8-21 | 198 | 7 | Abstract (n_trained, n_compressed, ab_h100_ratio, N1overN8_E100, K_u30_h1/h100, n_P) |
| 28-64 | 490 | 3 | 1 Introduction (sec:intro) |
| 65-84 | 189 | 0 | 2 Related Work; paras: Compounding error (66), Uncertainty for model-based control (70), Stability of recurrent networks (75), Physics and compression (80) |
| 85 | | | 3 System and Protocol (sec:setup) |
| 86-92 | 111 | 3 | Robot data (rows, hp_*) |
| 93-98 | 79 | 0 | Simulator |
| 99-108 | 145 | 11 | Rebuild and verification (diff_*, a2_*, win_cross/win_naive, stale_*) |
| 109-114 | 79 | 0 | Held-out data |
| 115-128 | 235 | 10 | Models (n_R, n_C, hp_*) |
| 129-144 | 263 | 0 | Measures |
| 145-167 | 345 | 10 | Statistics and registration (a2_n_lost, a2_n_tests, a2_median_ratio, pre_*) |
| 168 | | | 4 Results (sec:results) |
| 169-254 | 33+ | 29 | 4.1 Where one step and the horizon of use disagree (sec:decisions): tab/fig 169-209, then paras below |
| 210-222 | 191 | 13 | Training choices (K_*, ab_*, ft_*) |
| 223-231 | 107 | 10 | Training horizon and stability (ph_*, lamshift_*, ng_q4_*, N*_lam) |
| 232-246 | 261 | 18 | Compression (K_*, lamshift_*, lampos_*, perc_*, ft_FT1/FT2) |
| 247-254 | 127 | 4 | Uncertainty (descriptive) (dis_spear_h100, cov2_perh_h100, x7_summary) |
| 255-331 | 541 | 63 | 4.2 Which check predicts error at the horizon of use? (sec:checks): tab/fig 294-330; keys q1-q6, x1, x2, x3, d4, rel, a1, a2b, a2c, a3 |
| 332 | | | 5 Discussion (sec:discussion) |
| 333-345 | 224 | 0 | A checklist from this case study |
| 346-356 | 190 | 1 | Limitations (a2_*) |
| 357-363 | 94 | 0 | 6 Conclusion |
| 364-370 | 43 | 0 | Reproducibility and use of AI tools; then `\bibliography{refs}` |

## Tables and figures in the main text
- `tab:decisions` (L173, widetable, 4.1): "Each decision judged one step ahead and at longer horizons. Error ratio: error of the one-step-trained..."
- `fig:horizon` (L200, widefigure, 4.1): "Error at each forecast step on the 20 held-out trajectories of the five splits (models averaged within..."
- `tab:checks` (L294, widetable, 4.2): "Spearman correlation with $E_{100}$ across simulators. F18: 4 held-out trajectories. X5: 20,..."
- `fig:checks` (L322, widefigure, 4.2): "One-step error, growth rate and kinematic residual (left to right) against 100-step error for the {{n_P}} simulators..."

## appendix.template.tex (single-column submission only; NOT in the two-column check)
| Lines | Words | Section / tables |
|---|---|---|
| 6-37 | 176 | A1 Training settings (app:A1); tab:app-train |
| 38-51 | 203 | A2 What was known before the registration (app:A2) |
| 52-109 | 146 | A3 Horizon-matched and split-half checks (app:A3); tab:app-horizon, tab:app-split (keys a1_, a1t_, a3_) |
| 110-186 | 195 | A4 Wider intervals, five-split pool without {1,8}, Holm (app:A4); tab:app-two, tab:app-tests (a2_, a2b_, a2c_) |
| 187-247 | 165 | A5 Compression and the growth rate (app:A5); tab:app-comp, tab:app-perc |
| 248-260 | 195 | A6 Run-time signals (app:A6) |
| 261-360 | ~650 | A7 Study FT (app:A7): Design 264, Time rule and checks 278, Results 287, What was known 352; tab:app-ft, tab:app-ft-desc (ft_*) |
| 361-446 | ~880 | A8 Checks with a policy in the loop (Study CL, app:A8): Scope 365, Design 366, Registered verdicts 411, Secondary readings 417, Exploratory 436, What was known 442; tab:app-cl (s1_cl*) |
| 447-517 | ~630 | A9 Other simulator architectures (Study AR, app:A9): Scope 450, Design 451, Registered verdicts 493, Secondary and descriptive 500, What was known 513; tab:app-ar (s1_ar*) |
| 518-530 | 197 | A10 Training length (descriptive) (app:A10) (s1_tb_*) |

## check_claims.py assertions by section (line numbers in scripts/check_claims.py; 340 lines)
- Abstract: L129 (TF and N=1 worse than N=8 at 2 s); L156 (primary 1 and 2), L161, L163 (Holm), L165.
- 4.1 Training choices (L26-38, 74-79, 96, 132): TF vs rollout at 1/8/24/100/368 steps, position and joint-angle error, stale command (L58, L130).
- 4.1 Training horizon and stability (L39-51, 84-90, 134-138, 230): N=1 vs N=8, lambda_1 falls with N, N=32, NGCAPS paths.
- 4.1 Compression (L52-55, 88, 98, 139-152, 192-207, 223-236): pruning K, shift, fine-tuning FT1/FT2, quantisation.
- 4.1 Uncertainty (L56-57, 153-155): spread grows with horizon, synthetic data, per-horizon multiplier.
- 4.2 Checks (L91-110, 156-182): Q1-Q6 verdicts, r_J, E8, split pools, exploratory A1-A3, d4 (L180), re-synchronising (L182).
- Discussion/Conclusion (L102, 238, 267-270): released model, CL1/CL2 sentences.
- Appendices A7 (L192-219), A8 (L258-293), A9 (L296-327), A10 (L331-340).
- Limitations (L219): only 30% and 50% pruned copies fine-tuned.
(Assertions are keyed by description string, so a moved sentence keeps its assertion; renamed sentences need the description updated.)

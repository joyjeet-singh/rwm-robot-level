# R8 — claims encoded in scripts/check_claims.py (67 assertions; 46 before)

Removed (their sentences no longer exist after R6):
- "30%-pruned copy crosses the floor ... after tens of steps" (replaced by the line below)
- "fp16 keeps essentially all skill"; "coverage at 100 steps below nominal"; "per-member rollout raises coverage"
- "one- and two-step-trained models ... several times the best 100-step error"; "choosing by E1 picked N=1 or N=2"
- "abstract: TF about twice as wrong" (replaced by "TF and N=1 both worse than N=8 at 2 s")

Added (sentence -> assertion):
- Fig. 1 caption, pruned copy crosses the floor, starts within a few per cent -> U_cross exists; U_k1 < 1.1 A_k1
- Abstract, "times as wrong at 2 s" -> B/A > 1 and N=1 E100 > N=8 E100
- Sec. 3, stale command, one-step effect largest; held-out pair resolves 100 steps -> h1 > h100; CI lower > 0
- Sec. 4.1, rollout better on all held-out episodes (sign test) -> eps_pos == n_eps; p < 0.01
- Sec. 4.1, longer horizons more stable (rho(N, lambda_1)) -> NG Q4 rho < 0, CI upper < 0
- Sec. 4.1, every rollout-trained model of 8+ steps contracts, no teacher-forced one -> lam1 < 0 / > 0 over F18 rows
- Sec. 4.2, no copy pruned by 10-30% has lambda_1 > 0 -> n_positive == 0 for u10, u20, u30
- Sec. 4.2, 30% shift unresolved, 10 and 20% smaller but resolved -> CI spans 0 / above 0; means smaller
- Sec. 4.2 and checklist, moderate pruning adds error while barely changing lambda_1 -> K(100) < 1; shifts < 1/4 of u50
- Sec. 4.2, heavier pruning and unit removal raise it; some width-192/128 copies unstable -> CI > 0; n_positive > 0
- Checklist, pruned model has lost most skill by 2 s -> K(u30, 100) < 0.5
- Sec. 4.2, fine-tuned width 192 keeps most, width 128 not settled -> pooled CI lower > 0.5; spans 0.5
- Sec. 4.2, scratch better at 128, not separated at 192 -> pooled Q4 labels
- Sec. 4.2, 8-bit keeps most of 368-step skill -> CI lower > 0.5
- Sec. 4.3, synthetic under-estimate, likelihood loss recovers -> ratio > 5; NLL ratio within 0.1 of 1
- Sec. 4.3, multiplier misses the tails -> 2-sigma coverage < nominal
- Sec. 5, primary 1 favours growth rate, primary 2 not settled -> Q1, Q2 verdicts, rho order
- Sec. 5, five splits: one-step error ranks better than growth rate -> Q3_vs_E1 verdict, CI < 0
- Sec. 5, r_J over E8 on F18: registered yes, Holm no -> Q4_vs_E8 verdict; rejects_zero False
- Sec. 3, Holm: two of seven survive, named -> survive set equals {Q3_vs_E8, Q4_vs_E1}
- Sec. 3, wider intervals: three of 13 lose their reading, named -> holds-set complement is {Q1_E24, Q5_vs_E1, Q3_vs_E1}
- Sec. 5 and A3, r_J over 100 steps still ranks well -> rho > 0.9 on F18 and X5
- Sec. 5 and A3, r_J over 8 steps level with E8 on F18, below on X5 -> CI spans 0; CI upper < 0
- Sec. 5 and A3, growth rate needs the long rollout -> 8-step rho at least 0.3 below the 100-step rho
- Sec. 5, X5 without split {1,8}: growth rate still does not beat E1 -> CI lower <= 0
- Sec. 5, X5 without split {1,8}: r_J level or slightly ahead of E8 (wording changed, see R8.md) -> CI lower >= 0, upper < 0.1
- Sec. 5, most F18 simulators worse than the floor have lambda_1 <= 0 -> nonpos count > pos count > 0
- Sec. 5 and A6, re-synchronising on a score adds error -> H7 CI lower > 0

Not encoded (no order, sign or threshold, or already checked by the number build): hyperparameter equality across
25 parents, ratios and widths of intervals, "tenths of 1/s" start-direction dependence (left to the claim checker).

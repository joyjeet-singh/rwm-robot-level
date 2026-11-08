# M7: air-referee on step2/review/body.txt (fresh-eyes review; read only body.txt). Score: borderline

1. [claims] "Policies from three rebuilt simulators fell in every pilot episode" (p4): no rebuilt simulator was shown to yield a walking policy; absent from Abstract and Limitations. Add a clause to Limitations.
2. [claims] E100 never validated as the target; R0 uninformative; E100 uses recorded commands, so the title is right about horizon, not use. Say "100-step error under recorded commands".
3. [claims, clarity] "no check was shown to predict" reads as a finding; heading "Walking in a physics simulator" conflicts with "favours standing still" and falls; paragraph never says a policy was trained inside each simulator or how many; "best" undefined; "the authors' policy" ambiguous; "in-sample" unexplained; "9 copies" is the released model plus eight.
4. [fairness] Nine models with one parent are effectively one unit; state that the pilot preceded registration; R1/R2 end at exactly 0.00; both estimates favour the labelled checks and the text never says so.
5. [fairness] Q1 compares unequal information (E1 one step; label-free checks over whole rollouts; E8 a prefix of E100); drop or qualify "cheap".
6. [claims] "We recommend checking 8 or more steps ahead" (p1) has no E8-versus-E1 row; "the safer way" rests on descriptive correlations.
7. [claims] "One-step checks reverse training and compression choices": nothing reverses for compression. Retitle; fix "picked the wrong choices".
8. [fairness] Q1 depends on pool mix; bootstrap treats copies as independent; registration of a predictable answer; AR selection; does AR1 survive without the diverged MLP runs?
9. [claims] "their simulators amplify them (λ1 > 0)": the released model is positive too; say "in our low-data rebuilds".
10. [claims] "less well under policy-chosen commands" omits that the target remained recorded-command error.
11. [novelty] First half thin; lead with the check comparison; shorten Sec. IV-A.
12. [clarity] "registered/secondary/descriptive" in the Abstract; Holm, "parent", "reading", "wider intervals", Q5, Q6; CL versus "closed loop"; cut "Counted once..."; dense "each joint in units of its residual".
13. [inconsistencies] R0's sign undefined; Table III caption does not cover R1/R2; "TRAINING-HORIZON BETTER" unexplained; "3 of 13 registered verdicts" not countable from Table III; the 2.0 and 2.6 in the Abstract come from different pools.
14. [identity] Author names absent; risk is the "supplementary history": scrub commit names and paths.

Score: borderline

# R6 — text moved out of Sec. 4.3 for appendix A6 (unchanged, from body.template.tex at commit 01145a5, lines 331-342)

The original Sec. 4.3 in full; R6 replaced it with three sentences. R7 places this in appendix A6. Placeholders are kept as they were.

```latex
\subsection{What label-free signals cannot do at run time}\label{sec:runtime}
A check that ranks whole simulators need not flag individual bad steps. Earlier analyses of these models, on
the same held-out trajectories of the five splits, found that none of the label-free signals is a usable alarm for a
step that is worse than holding the last state. On the rollout-trained ensembles, a combined kinematic-consistency
score reached an area under the ROC curve (AUROC; 0.5 is chance) of {{icp_auc_phys}} {{icp_auc_phys_ci}}, ensemble
disagreement {{icp_auc_dis}} and a plain step counter {{icp_auc_counter}}. Averaged over models of all three kinds, the growth
rate read step by step (over the last eight steps) reached {{ng_q3_auc}}, below the step counter's {{ng_q3_counter}}
(difference {{ng_q3_delta}} {{ng_q3_ci}}; registered: \textsc{worse than counter}). Re-synchronising the simulator with the recorded state when the
kinematic score fired gave {{icp_h7_excess}} {{icp_h7_ci}} more 368-step error than the same number of
re-synchronisations on a fixed schedule (registered: \textsc{schedule better}). Physics helps as a repair only where it is exact: rescaling the predicted
gravity direction to unit length changed the 368-step error by {{icp_G}} {{icp_G_ci}}, while forcing the joint
angles onto the approximate integration rule multiplied it by {{icp_J_x}}.
```

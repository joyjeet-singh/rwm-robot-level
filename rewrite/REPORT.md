# Rewrite report (W0–W5)

**The new story.** One-step checks picked the wrong training choices: models trained one step at a time, or with
teacher forcing, look better one step ahead but are 2–2.6 times as wrong at 2 seconds (descriptive and earlier
registrations; also after longer training and in MLP and Transformer simulators, not RSSM). Pruning shows the same
trap, and fine-tuning restores skill (registered). Two checks that need no recorded future, the kinematic residual and
the growth rate, were then compared. Both ranked the main pool well with recorded commands. Both failed on mostly
compressed copies, worse under the authors' policy, and one change is NOT SETTLED. Practice: train and validate at the
horizon of use, and do not read a negative growth rate as a certificate. Scope: one robot, one gait, simulated data,
no physics-simulator test.

**Title:** kept, "One Step Ahead Is Not Enough: Testing a Learned Quadruped Simulator at the Horizon It Is Used".
Alternatives: "When One-Step Checks Mislead: Vetting a Learned Quadruped Simulator for Long Imagined Rollouts";
"Checking Learned Legged-Robot Simulators at the Horizon of Use".

**Moved to the appendix** (nothing left the paper): model list, rebuild checks, measure details (A1); pruning shifts
and width removal (A5); ensemble spread, variance collapse, released model (A6).

**Fit** (`slack.py`): two-column 6 pages; with the slot 6 pages; free space 0.32 of text height; abstract 197 words
(limit 200); PASS. `check_claims.py`: 127 of 127. The single-column PDF is the one to upload.

**Step 2 slot:** end of Sec. 4 (the box after Fig. 2, before the Discussion), 0.30 of text height, one column wide.
The sentences that must change when it is filled are in `rewrite/OUTLINE.md` part 8.

**Referee** (a fresh-eyes agent, W4): 3/5, borderline. Done: Q5 added (without the one-step-trained models), the
released model's positive growth rate stated, limitations name the two episodes, the missing correction across
registrations and the 3 lost verdicts, plus several scope fixes from a statistics reviewer. Declined, with reasons in
`DECISIONS.md`: the title, dropping checklist item 4 and the Uncertainty paragraph, a novelty comparison with two
related papers (no space). The training-episode baseline and the policy-in-simulator test are Step 2 candidates.

**Put in your own words before submitting:** the first paragraph of the Introduction and the two-questions paragraph;
the Discussion's "why one-step checks mislead here"; the Conclusion; the title. Also edit the **AI-use statement** (end
of the paper) to describe your own role correctly, and check AIR 2027's rules on AI-assisted writing.

**Watch for:** the abstract's 2.0 and 2.6 ratios come from different pools (Table 1 says which).

**Next steps.**
1. Read this report, `rewrite/review/section_changes.md` and the PDF.
2. Make Step 2's registration test the main text's claims (outline part 8): does a simulator's 2-second error, and do
   the two label-free checks, predict how well a policy trained inside it follows its velocity command in a physics
   simulator?
3. Withdraw the overlapping submissions, then submit between 15 Nov and 31 Dec 2026 (`SUBMIT.md`).

"""
I1 (improvement round): facts taken from other papers, written to results/external_facts.json.

These are constants, not measurements: each carries its value, the source's URL and version, the section it comes from,
and the line quoted.  They were read by session I0 (improve/notes/I0_facts.md) through a page summariser, so every
record has "quote_checked_against_raw_html": false until a reader has compared the quote with the source text.
Facts about the released code are not here: they come from the local copies and are recorded in I0_facts.md.
The paper's text may use these only through the keys im_numbers.py defines.
"""
import json
import os

import common as C

OUT = os.path.join(C.RESULTS, "external_facts.json")
RWM = {"url": "https://arxiv.org/abs/2501.10100", "version": "v5 (14 Dec 2025); v4 read too, no difference in these items",
       "paper": "Robotic World Model (RWM)"}
RWMU = {"url": "https://arxiv.org/abs/2504.16680", "version": "v3 (8 Jan 2026)", "paper": "RWM-U (uncertainty-aware RWM)"}
IKCE = {"url": "https://arxiv.org/abs/2607.05966", "version": "v1 (7 Jul 2026; only version)",
        "paper": "iKCE (imagined kinematic consistency error)"}

FACTS = {
    "rwm_imag_steps": dict(RWM, value=100, unit="imagined steps per iteration", section="Table S11 (MBPO-PPO)",
                           quote="imagination steps per iteration: 100"),
    "rwm_imag_envs": dict(RWM, value=4096, unit="imagined environments", section="Table S11 (MBPO-PPO)",
                          quote="imagination environments: 4096"),
    "rwm_history": dict(RWM, value=32, unit="steps", section="Sec. 4.1",
                        quote="The model is trained with history horizon M=32 and forecast horizon N=8."),
    "rwm_forecast": dict(RWM, value=8, unit="steps", section="Sec. 4.1",
                         quote="The model is trained with history horizon M=32 and forecast horizon N=8."),
    "rwm_pretrain_transitions": dict(RWM, value=6000000, unit="state transitions", section="Table 1 (RWM pretraining)",
                                     quote="state transitions: 6M"),
    "rwm_ar_beats_tf": dict(RWM, value="RWM-AR outperforms its teacher-forcing counterpart (RWM-TF)", unit="statement",
                            section="Sec. 4.3",
                            quote="RWM-AR significantly outperforms its teacher-forcing counterpart (RWM-TF)."),
    "rwm_tf_poor_ar": dict(RWM, value="N=1 is teacher forcing: fast, poor autoregressive performance", unit="statement",
                           section="App. A.4.1 (varying N)",
                           quote="this setting leads to poor autoregressive performance"),
    "rwmu_imag_steps": dict(RWMU, value=100, unit="imagined steps per iteration", section="App. A.3.2, Table S11",
                            quote="imagination steps per iteration: 100 (Intro: \"100-step episodic rollouts\")"),
    "rwmu_penalty": dict(RWMU, value="r~ = r - lambda*u, u = ensemble variance (epistemic uncertainty)", unit="statement",
                         section="Sec. 4.2, Eq. 5", quote="r~ = r - lambda*u"),
    "ikce_definition": dict(IKCE, value="departure of each predicted next state from the kinematic extrapolation of its "
                            "predecessor (closed-form constant-velocity or constant-acceleration predictor)",
                            unit="statement", section="Sec. III-A",
                            quote="iKCE measures how far each predicted next state departs from the kinematic "
                                  "extrapolation of its predecessor"),
    "ikce_necessary": dict(IKCE, value="necessary but not sufficient to certify dynamic imagination", unit="statement",
                           section="Sec. III-A / Sec. V",
                           quote="necessary but not sufficient to certify dynamic imagination; structural, not "
                                 "predictive, by design"),
    "ikce_n_models": dict(IKCE, value=1, unit="models evaluated", section="App. A, Table II",
                          quote="DreamerV3 on DMC walker-walk (one model, one task)"),
    "ikce_task": dict(IKCE, value="DreamerV3 on DMC walker-walk", unit="statement", section="App. A, Table II",
                      quote="DreamerV3 on DMC walker-walk"),
    "ikce_ranks_models": dict(IKCE, value=False, unit="used to rank models", section="whole paper (read to the end)",
                              quote="no ranking result found; it diagnoses one model (imagined against real-physics "
                                    "rollouts; friction sweep)"),
}


def main():
    for f in FACTS.values():
        f["quote_checked_against_raw_html"] = False
        f["read_by"] = "I0 (improve/notes/I0_facts.md)"
    json.dump(FACTS, open(OUT, "w"), indent=1)
    print(len(FACTS), "facts ->", os.path.relpath(OUT, C.AIR))


if __name__ == "__main__":
    main()

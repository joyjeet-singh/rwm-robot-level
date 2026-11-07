# Step 2 decisions

## M0
- **ANYdrive URL unverified.** The agent found the file name but not the full URL (Isaac Lab reads the asset root at run time). Conservative choice: `fetch.zsh` tries two guessed NVIDIA public roots and records success or failure; failure is not fatal (M1 uses the PD fallback per the plan). Recommendation: M1 reads `manifest.json` first.
- **Helper package pins** (absl-py, etils, glfw, pyopengl, importlib_resources, zipp) are my knowledge, not checked against PyPI; a failed pin shows in the fetch output and the manifest check. mujoco 3.10.0 is the pin from PyPI.
- **Isaac Lab v3.0.0 flat task uses the Newton backend;** the facts above are used as the task's settings only.
- **Local mean angles** the agent gave are whole-file means, not standing angles; joint order was taken from rwm_repro/src/rwm_data.py instead.

## M1
- **Motor model: PD fallback** (manifest: ANYdrive FAILED). One Kp, Kd pair shared by all 12 joints, as one network serves all 12 in Isaac Lab. Fitted by least squares on training-episode rows with |torque| < 75 N·m. Delay grid 0–11 physics steps. The recorded torques only separate the groups 0–3, 4–7 and 8–11; the smallest delay in the best group (4) is kept.
- **Timing assumption:** row r's torque is the one applied at the last physics step of the step into row r. The policy's output from state r is action row r+1 (Step 1's pairing). A fair rerun, with each convention's own previous action, favoured it (R² 0.40 vs 0.30).
- **Collision shapes:** the thigh capsule's radius is half the thigh box's width (0.0465 m), not the 0.07 m of the URDF's knee cylinders. So knees touch the ground slightly later. Drive housings are omitted. The payload is not a collision shape. There is no self-collision. Feet use condim 3 (sliding friction only), not Menagerie's 6.
- **Fall rule:** trunk net contact force > 1 N at any physics step of the policy step. Isaac Lab takes the maximum over its history; its history length is unverified. Contacts are net force > 1 N at the last physics step. Foot contact counts the sphere and the shank tube, which together are the URDF's FOOT link.
- **Placeholder constants** until M2 calibrates them: friction 0.8 (Isaac Lab's robot static friction), time constant 0.02 s (MuJoCo's default), armature 0, damping 0. `mj_env.set_constants` sets them. Friction applies to every shape and to the floor, because MuJoCo takes the larger of the two.
- **Note for M2 (design unchanged):** with MuJoCo's default `refsafe`, a contact time constant below 2 × timestep (0.01 s) is raised to 0.01 s. So the grid points 0.005 and 0.01 give identical physics. Recommendation: keep refsafe on (stable), evaluate the grid as fixed, and break the resulting tie by the plan's rule. If still tied, take the larger value, 0.01.
- **Subagent use:** air-bib-checker also fetched Isaac Lab's actuator and articulation source (quotes in `step2/notes/M1_sources.md`), beyond the Menagerie model the block names.

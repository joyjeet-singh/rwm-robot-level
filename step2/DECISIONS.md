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

## M2
- **Replay start state** (plan: height from leg kinematics, lowest foot on the ground; orientation from gravity). Yaw is unobservable and set to 0. "On the ground" means the lowest foot sphere just touches it, with no penetration. The motor's delay buffer is filled with the targets of the previous recorded joint commands. The recorded torque is copied into the state (it is an output and has no effect on the dynamics).
- **Windows:** each start is replayed once, for its longest evalh window. Shorter windows are its first h steps. MuJoCo is deterministic, and a test checks that the prefixes are equal. Positions and averaging are evalh's (tested against `evalh.window_floor`).
- **Grid points are evaluated from compiled model text, not from `set_constants` at run time.** Setting armature at run time leaves the compiler's derived quantities stale (`dof_invweight0`, `body_invweight0`, `dof_M0`, `actuator_acc0`). These enter the contact solver. The text evaluated for each point is the same text `--freeze` writes, and its hash is checked. Later sessions must use `step2/model/anymal_d_cal.xml`, not `MjEnv(constants=...)`.
- **Calibrated model file** is M1's XML with the defaults and joint attributes edited. `MjSpec.to_xml` rounds every number to 6 digits, which would undo M1's inertia precision.
- **Exact ties** (0.005 and 0.01 s run identically under refsafe): after E8 and E1, the larger time constant wins, then the earlier grid point (M1's recommendation).
- **Gait statistics:** rows 1 to the end, or to the fall; per-episode means, then the mean over episodes.
  - two feet = exactly two of four feet in contact;
  - duty = contact share, averaged over the four feet;
  - speed error = |(vx, vy) − command| in the base frame; turn error = |wz − command|.
  - The velocity command is Step 0's regime means (it is not recorded). Episode 0 has 999 rows, so its last command is held.
- **Evaluation starts:** `default_rng([2027, e, j])`; all 12 joints get a uniform ±0.05 rad offset; heading is uniform on (−π, π).
- **Refusals:** held-out replays (calibrated *and* uncalibrated) and every policy run are refused until `calibration.json` is frozen and its XML hash matches.
- **Extra, descriptive:** (b) also runs on the uncalibrated model, to help a diagnosis if Gate 1 fails. Gate 1 reads the calibrated run only.

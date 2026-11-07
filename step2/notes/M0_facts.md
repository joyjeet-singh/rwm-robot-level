# M0 facts (2026-11-07)

Web facts come from subagent summaries of the pages, not byte-for-byte reads; the fetch tool summarises. Items marked UNVERIFIED must be rechecked by M1 against the files fetched.

## a. Isaac Lab ANYmal D robot (v3.0.0, released 2026-10-20)
Source: github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab_assets/isaaclab_assets/robots/anymal.py
- `network_file=f"{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt"`, with `ISAACLAB_NUCLEUS_DIR = f"{ISAAC_NUCLEUS_DIR}/IsaacLab"`.
- **Full download URL: UNVERIFIED.** The US asset root is read at run time from `isaaclab.python.kit`; no literal URL is in `assets.py`. `fetch.zsh` tries two guessed roots (omniverse-content-production S3, `.../Assets/Isaac/IsaacLab/ActuatorNets/ANYbotics/...`). If both fail the PD fallback applies (see DECISIONS.md).
- Docstring: no public ANYmal-D actuator net exists; the ANYmal-C net is used.
- Limits: `saturation_effort=120.0`, `actuator_effort_limit=80.0`, `actuator_velocity_limit=7.5`; `soft_joint_pos_limit_factor` 0.95.
- Default joint angles: HAA 0.0; F_HFE 0.4; H_HFE -0.4; F_KFE -0.8; H_KFE 0.8. Start `pos=(0,0,0.6)`.
- Joints: `.*HAA`, `.*HFE`, `.*KFE`.

## b. Flat velocity task (v3.0.0; commit e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4; `isaaclab_tasks/core/velocity/`)
- `sim.dt = 0.005`, `decimation = 4` (control step 0.02 s, as the lite code's `step_dt`), episode 20 s.
- Action: `JointPositionActionCfg(joint_names=[".*"], scale=0.5, use_default_offset=True)`: target = default + 0.5 × action.
- Policy observations in order, no scaling: base_lin_vel, base_ang_vel, projected_gravity, velocity_commands, joint_pos_rel, joint_vel_rel, last_action (flat removes height_scan). Training noise: ±0.1, ±0.2, ±0.05, none, ±0.01, ±1.5, none.
- **Fall rule:** `illegal_contact`, body `base`, threshold 1.0 (N), plus time-out.
- Commands: lin_vel_x, lin_vel_y, ang_vel_z each (-1, 1); heading (-π, π); resampled every 10 s.
- Friction: terrain static 1.0, dynamic 1.0 (multiply); startup event sets all robot bodies static 0.8, dynamic 0.6. Whether the flat plane keeps terrain 1.0/1.0: UNVERIFIED.
- Caveat: v3.0.0 flat config switches to the Newton MuJoCo-Warp backend; the data predates it.

## c. Description
github.com/ANYbotics/anymal_d_simple_description, `master`, commit d5bf4dc3a81af81a16b9598fc15c98e339f45a33 (2025-02-15). BSD 3-Clause, "Copyright 2023, ANYbotics AG".
- URDF: `urdf/anymal.urdf` (plain URDF). No collision directory: collision shapes are 44 inline primitives (boxes, cylinders, spheres). Meshes `meshes/*.dae` are visual only; MuJoCo cannot load them and they are not needed.
- Leg joints `{LF,RF,LH,RH}_{HAA,HFE,KFE}`, effort 80, velocity 8.5. HAA limits about -0.785..0.611; HFE/KFE ±9.42 (unlimited). Two payload joints are revolute (fix them). Axes ±x, signs differ by leg.
- UNVERIFIED: stray text after `base_to_lidar_parent` may make the XML malformed.

## d. MuJoCo
Newest cp311 macOS x86_64 wheel: **3.10.0** (`mujoco-3.10.0-cp311-cp311-macosx_10_16_x86_64.whl`, sha256 62b7e9fa…409f, uploaded 2026-06-22). 3.14–3.17 are arm64 only; 3.11/3.12 not fully checked. Requires: absl-py, etils[epath], glfw, numpy (unbounded), pyopengl. With `--no-deps`, etils[epath] also needs importlib_resources, zipp (fsspec, typing_extensions already installed). Helper pins in fetch.zsh come from my knowledge, not the agent: absl-py 2.1.0, etils 1.7.0, glfw 2.7.0, pyopengl 3.1.7, importlib_resources 6.4.0, zipp 3.19.2.

## Local facts
- **Joint order** (rwm_repro/src/rwm_data.py:52): `[f"{leg}_{j}" for j in (HAA,HFE,KFE) for leg in [LF,LH,RF,RH]]`: HAA×4 (LF,LH,RF,RH), then HFE×4, then KFE×4. Signs not recorded.
- **State 45** (lite anymal_d_flat.py ~58-63): lin_vel 0:3, ang_vel 3:6, gravity 6:9, joint_pos 9:21, joint_vel 21:33, **joint torque 33:45 (yes, included)**. Torque mean ±2..12, std 15. CSV: 66 columns = 45 state, 12 action, 8 contact (4 thigh, 4 foot), 1 termination. DT = 0.02.
- **Imagination reward** (lite envs/anymal_d_flat.py ~98-163; weights configs/anymal_d_flat_cfg.py:16-30): track_lin_vel_xy_exp 1.0, track_ang_vel_z_exp 0.5, lin_vel_z_l2 -2.0, ang_vel_xy_l2 -0.05, dof_torques_l2 -2.5e-5, dof_acc_l2 -2.5e-7, action_rate_l2 -0.01, feet_air_time 0.5, undesired_contacts -1.0, stand_still -1.0, flat_orientation_l2 -5.0, dof_pos_limits 0, uncertainty -1.0. Tracking uses exp(-err/0.25). Quirk: joint_acc uses obs columns 12:24 (joint_pos, not joint_vel).
- **48 observations** (scripts/s1_policy.py:103-110): state[0:3], state[3:6], state[6:9], command(3), state[9:21], state[21:33], previous action(12); state denormalised, action identity, no noise. Torques are not observed.
- Reward is applied by s1_policy.py:145-157 `step_reward`, each term times step_dt.

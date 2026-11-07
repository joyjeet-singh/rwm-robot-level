# M1 sources (quotes as returned by air-bib-checker's fetch tool, which summarises; not diffed byte for byte)

Isaac Lab v3.0.0, `source/isaaclab/isaaclab/actuators/`:
- `IdealPDActuator.compute`: `error_pos = control_action.joint_positions - joint_pos`; `error_vel = control_action.joint_velocities - joint_vel`; `self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts`; `self.applied_effort = self._clip_effort(self.computed_effort, joint_vel)`.
- `DCMotor.__init__`: `self._vel_at_effort_lim = self.actuator_velocity_limit * (1 + self.actuator_effort_limit / self._saturation_effort)`.
- `DCMotor._clip_effort`: `joint_vel = torch.clip(joint_vel, min=-self._vel_at_effort_lim, max=self._vel_at_effort_lim)`; `torque_speed_top = self._saturation_effort * (1.0 - joint_vel / self.actuator_velocity_limit)`; `torque_speed_bottom = self._saturation_effort * (-1.0 - joint_vel / self.actuator_velocity_limit)`; `max_effort = torch.clip(torque_speed_top, max=self.actuator_effort_limit)`; `min_effort = torch.clip(torque_speed_bottom, min=-self.actuator_effort_limit)`; `return torch.clip(effort, min=min_effort, max=max_effort)`.
- `actuator_net.py`: `class ActuatorNetLSTM(DCMotor)`; its `compute` feeds `(joint_positions - joint_pos, joint_vel)` to the LSTM and returns `self._clip_effort(self.computed_effort, joint_vel)`.
- `DelayedPDActuator.compute` passes positions, velocities and efforts through `DelayBuffer.compute`, then `super().compute`. `DelayBuffer.compute` (partial lines): `lag = torch.minimum(self._time_lags, self._num_pushes)`: the lag is clamped to the history stored.

Isaac Lab v2.1.0 `articulation_data.py`: `root_lin_vel_w` returns `self.root_state_w[:, 7:10]` ("the articulation root's center of mass frame"); `root_lin_vel_b = quat_rotate_inverse(self.root_quat_w, self.root_lin_vel_w)`; `projected_gravity_b` uses `root_link_quat_w`. v3.0.0 has only explicit `root_link_*`/`root_com_*` variants (alias not found).

MuJoCo Menagerie `anybotics_anymal_c` (main): `<option cone="elliptic" impratio="100"/>`; joints `damping="1" frictionloss="0.1"`; foot `sphere size="0.03" condim="6" friction="0.8 0.02 0.01"`; position actuators `kp="100" forcerange="-80 80"`; trunk collision box.

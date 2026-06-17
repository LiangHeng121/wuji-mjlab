# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Kinematics-bias action with two modes (switch via cfg.mode):

- "offset"  : target = ref_qpos[t] + action * action_scale          (uniform scale)
- "wdelta"  : DexTrack-exact accumulative residual (use_kinematics_bias_wdelta).
              Per-group speed scales, residual accumulates over the episode:
                delta_delta = speed_scale * step_dt * action * 2.0
                accum_delta += delta_delta
                target = ref_qpos[t] + accum_delta
              speed_scale = [glb_trans]*3 + [glb_rot]*3 + [dof_speed]*20 (q26 order).
              Mirrors allegro_hand_tracking_generalist.py lines 12519-12526 so the
              ONLY intended difference from DexTrack is the simulator (MuJoCo vs PhysX).

Both clamp the final target to the soft joint limits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions.actions import (
  JointPositionAction,
  JointPositionActionCfg,
)


class KinematicsBiasAction(JointPositionAction):
  cfg: "KinematicsBiasActionCfg"

  def __init__(self, cfg: "KinematicsBiasActionCfg", env):
    super().__init__(cfg, env)
    self._mode = cfg.mode
    self._action_scale = cfg.action_scale
    self._command_name = cfg.command_name

    soft = self._entity.data.soft_joint_pos_limits[:, self._target_ids]
    self._lower_limits = soft[..., 0]
    self._upper_limits = soft[..., 1]

    # Per-joint speed scale for wdelta mode, in the action's target-joint order.
    names = [self._entity.joint_names[i] for i in self._target_ids]
    scale = []
    for n in names:
      if n in ("WRJ0x", "WRJ0y", "WRJ0z"):
        scale.append(cfg.glb_trans_vel_scale)
      elif n in ("WRJ0rx", "WRJ0ry", "WRJ0rz"):
        scale.append(cfg.glb_rot_vel_scale)
      else:
        scale.append(cfg.dof_speed_scale)
    self._speed_scale = torch.tensor(scale, device=env.device).unsqueeze(0)  # (1, n)
    self._accum_delta = torch.zeros(
      (self.num_envs, len(self._target_ids)), device=env.device
    )

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    clamped = torch.clamp(actions, -1.0, 1.0)
    ref = self._env.command_manager.get_term(self._command_name).ref_qpos
    ref = ref[:, self._target_ids]

    if self._mode == "wdelta":
      # DexTrack accumulative residual (control_freq_inv folded into step_dt; x2.0 const).
      delta_delta = self._speed_scale * self._env.step_dt * clamped * 2.0
      self._accum_delta = self._accum_delta + delta_delta
      target = ref + self._accum_delta
    else:  # "offset"
      target = ref + clamped * self._action_scale

    self._processed_actions = torch.clamp(
      target, self._lower_limits, self._upper_limits
    )

  @property
  def processed_action(self) -> torch.Tensor:
    return self._processed_actions

  def reset(self, env_ids: torch.Tensor) -> None:
    super().reset(env_ids)
    self._accum_delta[env_ids] = 0.0  # clear accumulated residual on episode reset


@dataclass(kw_only=True)
class KinematicsBiasActionCfg(JointPositionActionCfg):
  command_name: str = "motion"
  mode: str = "offset"  # "offset" (current) | "wdelta" (DexTrack-exact)
  # offset mode:
  action_scale: float = 0.1
  # wdelta mode (DexTrack wuji values):
  glb_trans_vel_scale: float = 0.5
  glb_rot_vel_scale: float = 0.5
  dof_speed_scale: float = 20.0

  def build(self, env) -> KinematicsBiasAction:
    return KinematicsBiasAction(self, env)

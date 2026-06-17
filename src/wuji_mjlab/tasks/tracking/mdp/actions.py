# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Kinematics-bias action (DexTrack ``use_kinematics_bias_wdelta``).

Joint position target = reference qpos[t] + action * scale, where the moving
reference comes from the HandObjectMotionCommand. This is a residual on top of
the kinematic reference trajectory, not a residual on a static default pose.
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
    self._action_scale = cfg.action_scale
    self._command_name = cfg.command_name

    soft = self._entity.data.soft_joint_pos_limits[:, self._target_ids]
    self._lower_limits = soft[..., 0]
    self._upper_limits = soft[..., 1]

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    clamped = torch.clamp(actions, -1.0, 1.0)

    # Current reference qpos (robot joint order), aligned to this action's joints.
    ref = self._env.command_manager.get_term(self._command_name).ref_qpos
    ref = ref[:, self._target_ids]

    target = ref + clamped * self._action_scale
    self._processed_actions = torch.clamp(
      target, self._lower_limits, self._upper_limits
    )

  @property
  def processed_action(self) -> torch.Tensor:
    return self._processed_actions


@dataclass(kw_only=True)
class KinematicsBiasActionCfg(JointPositionActionCfg):
  action_scale: float = 0.5
  command_name: str = "motion"

  def build(self, env) -> KinematicsBiasAction:
    return KinematicsBiasAction(self, env)

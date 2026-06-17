# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Reward terms for hand+object tracking.

Deliberately minimal / "naive" set: hand joint tracking + object position +
object orientation, all bounded exp(-err/std^2) kernels. NO contact guidance /
softclip / idle-hold (those were PhysX-era patches; add back only if MuJoCo
physics actually needs them).
"""

from __future__ import annotations

import torch

from mjlab.utils.lab_api.math import quat_error_magnitude


def joint_tracking_exp(env, command_name: str = "motion", std: float = 0.5) -> torch.Tensor:
  """exp(-||robot_joint_pos - ref_qpos||^2 / std^2)."""
  cmd = env.command_manager.get_term(command_name)
  robot = env.scene[cmd.cfg.hand_entity_name]
  err = torch.sum(torch.square(cmd.ref_qpos - robot.data.joint_pos), dim=-1)
  return torch.exp(-err / std**2)


def object_pos_tracking_exp(env, command_name: str = "motion", std: float = 0.05) -> torch.Tensor:
  """exp(-||object_pos - ref_obj_pos||^2 / std^2)."""
  cmd = env.command_manager.get_term(command_name)
  obj = env.scene[cmd.cfg.object_entity_name]
  err = torch.sum(torch.square(cmd.ref_obj_pos - obj.data.root_link_pos_w), dim=-1)
  return torch.exp(-err / std**2)


def object_ori_tracking_exp(env, command_name: str = "motion", std: float = 0.5) -> torch.Tensor:
  """exp(-quat_err(object, ref_obj)^2 / std^2)."""
  cmd = env.command_manager.get_term(command_name)
  obj = env.scene[cmd.cfg.object_entity_name]
  err = quat_error_magnitude(cmd.ref_obj_quat, obj.data.root_link_quat_w) ** 2
  return torch.exp(-err / std**2)

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Observation terms for hand+object tracking."""

from __future__ import annotations

import torch


def object_pos_b(env, command_name: str = "motion") -> torch.Tensor:
  """Object position relative to the env origin (drops the per-env offset)."""
  cmd = env.command_manager.get_term(command_name)
  obj = env.scene[cmd.cfg.object_entity_name]
  return obj.data.root_link_pos_w - env.scene.env_origins


def object_quat_w(env, command_name: str = "motion") -> torch.Tensor:
  """Object orientation (world frame, wxyz)."""
  cmd = env.command_manager.get_term(command_name)
  obj = env.scene[cmd.cfg.object_entity_name]
  return obj.data.root_link_quat_w


def object_pos_error(env, command_name: str = "motion") -> torch.Tensor:
  """Reference object position minus achieved object position."""
  cmd = env.command_manager.get_term(command_name)
  obj = env.scene[cmd.cfg.object_entity_name]
  return cmd.ref_obj_pos - obj.data.root_link_pos_w

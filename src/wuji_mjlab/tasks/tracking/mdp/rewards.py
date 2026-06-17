# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Reward terms ported faithfully from DexTrack's base reward.

Mirrors the Isaac Gym ``AllegroHandTrackingGeneralist`` base reward (default
config, train_free_hand=False, w_obj_ornt=False, smoothness=0), i.e. WITHOUT the
pinall3 / cgsmooth / B2-contact / softclip / idle patches. The total is:

    r = -rew_delta_hand_pose_coef * (0.6*L1(d_trans) + 0.1*L1(d_rot) + 0.1*L1(d_finger))
        - rew_finger_obj_dist_coef * (finger_dist + palm_dist_rew_w * palm_dist)
        + goal_hand_rew      # = -2*goal_dist, only when grasping (flag==2)
        + bonus              # = 1/(1+10*goal_dist) if goal_dist<=0.05 and flag==2

flag = (finger_dist <= 0.12*nf) + (palm_dist <= 0.12); ==2 means hand is grasping.
Distances use the object root as the handle point. Base reward uses raw negative
L1/L2 penalties (NOT exp kernels) -- matching DexTrack.
"""

from __future__ import annotations

import torch

# 4 fingers used in the base reward (th, ff, mf, rf = finger1..4); pinky excluded
# unless FIX_FINGER5 (a pinall3 switch) is on.
_TIP_SITES = (
  "right_finger1_tip",
  "right_finger2_tip",
  "right_finger3_tip",
  "right_finger4_tip",
)
_PALM_BODY = "right_palm_link"
_NUM_FINGERS = 4
_FINGER_DIST_CLAMP = 0.6 * _NUM_FINGERS  # 2.4 m
_FINGER_DIST_THRES = 0.12 * _NUM_FINGERS  # 0.48 m
_PALM_GRIP_THRES = 0.12  # m
_PALM_DIST_CLAMP = 0.5  # m


def _grasp_geometry(env, command_name: str):
  """Shared geometry: finger_dist, palm_dist, grasp flag, goal_dist."""
  cmd = env.command_manager.get_term(command_name)
  robot = env.scene[cmd.cfg.hand_entity_name]
  obj = env.scene[cmd.cfg.object_entity_name]

  cache = getattr(env, "_grasp_idx", None)
  if cache is None:
    tip_idx = robot.find_sites(list(_TIP_SITES), preserve_order=True)[0]
    palm_idx = robot.find_bodies([_PALM_BODY], preserve_order=True)[0][0]
    cache = (torch.as_tensor(tip_idx, device=env.device, dtype=torch.long), palm_idx)
    env._grasp_idx = cache
  tip_idx, palm_idx = cache

  obj_pos = obj.data.root_link_pos_w  # (E,3)
  tips = robot.data.site_pos_w[:, tip_idx]  # (E,4,3)
  palm = robot.data.body_link_pos_w[:, palm_idx]  # (E,3)

  finger_dist = torch.norm(obj_pos.unsqueeze(1) - tips, p=2, dim=-1).sum(dim=-1)
  finger_dist = finger_dist.clamp(max=_FINGER_DIST_CLAMP)
  palm_dist = torch.norm(obj_pos - palm, p=2, dim=-1).clamp(max=_PALM_DIST_CLAMP)

  # Finger penalty only counts when the palm is already close (DexTrack gating).
  finger_dist = torch.where(
    palm_dist <= _PALM_GRIP_THRES, finger_dist, torch.zeros_like(finger_dist)
  )
  flag = (finger_dist <= _FINGER_DIST_THRES).int() + (palm_dist <= _PALM_GRIP_THRES).int()
  goal_dist = torch.norm(cmd.ref_obj_pos - obj_pos, p=2, dim=-1)
  return finger_dist, palm_dist, flag, goal_dist


def hand_pose_tracking(
  env, command_name: str = "motion",
  trans_coef: float = 0.6, rot_coef: float = 0.1, finger_coef: float = 0.1,
) -> torch.Tensor:
  """-(0.6*L1(d_trans) + 0.1*L1(d_rot) + 0.1*L1(d_finger)); weight = rew_delta_hand_pose_coef."""
  cmd = env.command_manager.get_term(command_name)
  robot = env.scene[cmd.cfg.hand_entity_name]
  dq = cmd.ref_qpos - robot.data.joint_pos  # (E,26), q26 order (== robot joint order)
  d_trans = torch.norm(dq[:, :3], p=1, dim=-1)
  d_rot = torch.norm(dq[:, 3:6], p=1, dim=-1)
  d_finger = torch.norm(dq[:, 6:], p=1, dim=-1)
  return -(trans_coef * d_trans + rot_coef * d_rot + finger_coef * d_finger)


def finger_object_distance(
  env, command_name: str = "motion", palm_dist_rew_w: float = 2.0
) -> torch.Tensor:
  """-(finger_dist + palm_dist_rew_w * palm_dist); weight = rew_finger_obj_dist_coef."""
  finger_dist, palm_dist, _, _ = _grasp_geometry(env, command_name)
  return -(finger_dist + palm_dist_rew_w * palm_dist)


def object_pos_tracking(env, command_name: str = "motion") -> torch.Tensor:
  """goal_hand_rew = -2*goal_dist, only when grasping (flag==2); weight 1.0."""
  _, _, flag, goal_dist = _grasp_geometry(env, command_name)
  return torch.where(flag == 2, -2.0 * goal_dist, torch.zeros_like(goal_dist))


def object_inplace_bonus(env, command_name: str = "motion") -> torch.Tensor:
  """+1/(1+10*goal_dist) if goal_dist<=0.05 and grasping (flag==2); weight 1.0."""
  _, _, flag, goal_dist = _grasp_geometry(env, command_name)
  b = torch.where(
    goal_dist <= 0.05, 1.0 / (1.0 + 10.0 * goal_dist), torch.zeros_like(goal_dist)
  )
  return torch.where(flag == 2, b, torch.zeros_like(goal_dist))

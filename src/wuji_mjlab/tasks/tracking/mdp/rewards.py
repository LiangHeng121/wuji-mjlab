# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""DexTrack base reward + pinall3 switches (no cgsmooth/B2/softclip/idle patches).

pinall3 = base reward (hand_pose, finger/palm-obj dist, object pos, in-place
bonus) plus:
  - RELAX_PALM:  palm_grip_thres 0.12 -> 0.22, palm_dist_rew_w 2.0 -> 0.0
                 (drops the palm-distance penalty; palm is positioned by
                  PALM_POS_REW instead)
  - FIX_FINGER5: include the pinky -> 5 fingertips, thresholds x5
  - FINGER_POS_REW (coef 1.0): -sum_5 ||sim_tip_i - ref_tip_i||
  - PALM_POS_REW  (coef 1.0): -||sim_palm - ref_palm||

The FINGER_POS / PALM_POS terms densely track the reference grasp SHAPE in world
space (ref fingertip/palm positions verified to match sim FK <=2 mm), giving an
always-on gradient that forms the grasp independent of where the object falls --
this is what the gated object terms need to bootstrap.
"""

from __future__ import annotations

import torch

# pinall3: FIX_FINGER5 -> all 5 fingertips (th, ff, mf, rf, lf = finger1..5).
_TIP_SITES = tuple(f"right_finger{i}_tip" for i in range(1, 6))
_PALM_BODY = "right_palm_link"
_NUM_FINGERS = 5
_FINGER_DIST_CLAMP = 0.6 * _NUM_FINGERS  # 3.0 m
_FINGER_DIST_THRES = 0.12 * _NUM_FINGERS  # 0.60 m
_PALM_GRIP_THRES = 0.22  # m (RELAX_PALM)
_PALM_DIST_CLAMP = 0.5  # m


def _hand_points(env, command_name: str):
  """Shared lookups: command, robot, sim fingertip (E,5,3) + palm (E,3), obj (E,3)."""
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

  sim_tips = robot.data.site_pos_w[:, tip_idx]  # (E,5,3)
  sim_palm = robot.data.body_link_pos_w[:, palm_idx]  # (E,3)
  obj_pos = obj.data.root_link_pos_w  # (E,3)
  return cmd, robot, sim_tips, sim_palm, obj_pos


def _grasp_geometry(env, command_name: str):
  """finger_dist, palm_dist, grasp flag, goal_dist."""
  cmd, _, sim_tips, sim_palm, obj_pos = _hand_points(env, command_name)
  finger_dist = torch.norm(obj_pos.unsqueeze(1) - sim_tips, p=2, dim=-1).sum(dim=-1)
  finger_dist = finger_dist.clamp(max=_FINGER_DIST_CLAMP)
  palm_dist = torch.norm(obj_pos - sim_palm, p=2, dim=-1).clamp(max=_PALM_DIST_CLAMP)
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
  """-(0.6*L1(d_trans) + 0.1*L1(d_rot) + 0.1*L1(d_finger)); weight=rew_delta_hand_pose_coef."""
  cmd = env.command_manager.get_term(command_name)
  robot = env.scene[cmd.cfg.hand_entity_name]
  dq = cmd.ref_qpos - robot.data.joint_pos  # (E,26), q26 order
  d_trans = torch.norm(dq[:, :3], p=1, dim=-1)
  d_rot = torch.norm(dq[:, 3:6], p=1, dim=-1)
  d_finger = torch.norm(dq[:, 6:], p=1, dim=-1)
  return -(trans_coef * d_trans + rot_coef * d_rot + finger_coef * d_finger)


def finger_object_distance(
  env, command_name: str = "motion", palm_dist_rew_w: float = 0.0
) -> torch.Tensor:
  """-(finger_dist + palm_dist_rew_w*palm_dist); RELAX_PALM -> palm_dist_rew_w=0."""
  finger_dist, palm_dist, _, _ = _grasp_geometry(env, command_name)
  return -(finger_dist + palm_dist_rew_w * palm_dist)


def object_pos_tracking(env, command_name: str = "motion") -> torch.Tensor:
  """-2*goal_dist, only when grasping (flag==2)."""
  _, _, flag, goal_dist = _grasp_geometry(env, command_name)
  return torch.where(flag == 2, -2.0 * goal_dist, torch.zeros_like(goal_dist))


def object_inplace_bonus(env, command_name: str = "motion") -> torch.Tensor:
  """+1/(1+10*goal_dist) if goal_dist<=0.05 and grasping (flag==2)."""
  _, _, flag, goal_dist = _grasp_geometry(env, command_name)
  b = torch.where(
    goal_dist <= 0.05, 1.0 / (1.0 + 10.0 * goal_dist), torch.zeros_like(goal_dist)
  )
  return torch.where(flag == 2, b, torch.zeros_like(goal_dist))


def finger_pos_tracking(env, command_name: str = "motion") -> torch.Tensor:
  """FINGER_POS_REW: -sum_5 ||sim_tip_i - ref_tip_i|| (always on); weight=FINGER_POS_COEF."""
  cmd, _, sim_tips, _, _ = _hand_points(env, command_name)
  return -torch.norm(sim_tips - cmd.ref_fingertip_pos, p=2, dim=-1).sum(dim=-1)


def palm_pos_tracking(env, command_name: str = "motion") -> torch.Tensor:
  """PALM_POS_REW: -||sim_palm - ref_palm|| (always on); weight=PALM_POS_COEF."""
  cmd, _, _, sim_palm, _ = _hand_points(env, command_name)
  return -torch.norm(sim_palm - cmd.ref_palm_pos, p=2, dim=-1)

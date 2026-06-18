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

from mjlab.utils.lab_api.math import quat_apply

# FIX_FINGER5 -> all 5 fingertips (th, ff, mf, rf, lf = finger1..5).
_TIP_SITES = tuple(f"right_finger{i}_tip" for i in range(1, 6))
_PALM_BODY = "right_palm_link"
_NUM_FINGERS = 5  # wuji has 5 fingers -> thresholds use x5 in BOTH original & pinall3
_FINGER_DIST_CLAMP = 0.6 * _NUM_FINGERS  # 3.0 m
_FINGER_DIST_THRES = 0.12 * _NUM_FINGERS  # 0.60 m
# grip thres: original 0.12, pinall3 (RELAX_PALM) 0.22. n_finger_sum: original sums
# only th,ff,mf,rf (4); pinall3 (FIX_FINGER5) sums all 5. (thresholds stay x5.)
_PALM_DIST_CLAMP = 0.5  # m


def _hand_points(env, command_name: str):
  """Shared lookups: command, robot, sim fingertip (E,5,3) + palm (E,3), obj (E,3)."""
  cmd = env.command_manager.get_term(command_name)
  robot = env.scene[cmd.cfg.hand_entity_name]

  cache = getattr(env, "_grasp_idx", None)
  if cache is None:
    tip_idx = robot.find_sites(list(_TIP_SITES), preserve_order=True)[0]
    palm_idx = robot.find_bodies([_PALM_BODY], preserve_order=True)[0][0]
    cache = (torch.as_tensor(tip_idx, device=env.device, dtype=torch.long), palm_idx)
    env._grasp_idx = cache
  tip_idx, palm_idx = cache

  sim_tips = robot.data.site_pos_w[:, tip_idx]  # (E,5,3)
  sim_palm = robot.data.body_link_pos_w[:, palm_idx]  # (E,3)
  obj_pos = cmd.obj_pos  # (E,3) active object (multi-object aware)
  return cmd, robot, sim_tips, sim_palm, obj_pos


def _grasp_geometry(env, command_name: str, grip_thres: float, n_finger_sum: int):
  """finger_dist, palm_dist, grasp flag, goal_dist.

  grip_thres: palm-grip threshold (original 0.12, pinall3/RELAX_PALM 0.22).
  n_finger_sum: # fingertips summed into finger_dist (original 4, pinall3 5).
  """
  cmd, _, sim_tips, sim_palm, obj_pos = _hand_points(env, command_name)
  tips = sim_tips[:, :n_finger_sum]  # (E, n, 3)
  finger_dist = torch.norm(obj_pos.unsqueeze(1) - tips, p=2, dim=-1).sum(dim=-1)
  finger_dist = finger_dist.clamp(max=_FINGER_DIST_CLAMP)
  palm_dist = torch.norm(obj_pos - sim_palm, p=2, dim=-1).clamp(max=_PALM_DIST_CLAMP)
  finger_dist = torch.where(
    palm_dist <= grip_thres, finger_dist, torch.zeros_like(finger_dist)
  )
  flag = (finger_dist <= _FINGER_DIST_THRES).int() + (palm_dist <= grip_thres).int()
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
  env, command_name: str = "motion", palm_dist_rew_w: float = 0.0,
  grip_thres: float = 0.22, n_finger_sum: int = 5,
) -> torch.Tensor:
  """-(finger_dist + palm_dist_rew_w*palm_dist); RELAX_PALM -> palm_dist_rew_w=0."""
  finger_dist, palm_dist, _, _ = _grasp_geometry(env, command_name, grip_thres, n_finger_sum)
  return -(finger_dist + palm_dist_rew_w * palm_dist)


def object_pos_tracking(
  env, command_name: str = "motion", grip_thres: float = 0.22, n_finger_sum: int = 5
) -> torch.Tensor:
  """-2*goal_dist, only when grasping (flag==2)."""
  _, _, flag, goal_dist = _grasp_geometry(env, command_name, grip_thres, n_finger_sum)
  return torch.where(flag == 2, -2.0 * goal_dist, torch.zeros_like(goal_dist))


def object_inplace_bonus(
  env, command_name: str = "motion", grip_thres: float = 0.22, n_finger_sum: int = 5
) -> torch.Tensor:
  """+1/(1+10*goal_dist) if goal_dist<=0.05 and grasping (flag==2)."""
  _, _, flag, goal_dist = _grasp_geometry(env, command_name, grip_thres, n_finger_sum)
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


def fair_reward_metric(env, command_name: str = "motion") -> torch.Tensor:
  """Config-independent FAIR reward (DexTrack fair@0.22 canonical base coefs:
  hand_pose 0.6/0.1/0.1, finger/palm-obj dist palm_dist_rew_w=2.0 + grip 0.22 +
  4-finger sum, gated obj pos + in-place bonus; NO pinall3 / NO patches).

  Logged to extras['log']['fair_reward'] for a comparable live wandb/tb curve.
  Reports the EPISODE-SUM (DexTrack reward_fair semantics): per-env fair is
  accumulated over the episode and, on reset, the mean sum of the just-finished
  episodes is published -- directly comparable to the single-seq fair (~200), not
  the per-step mean (~0.7). The reward manager runs AFTER termination, so
  env.reset_buf already holds this step's dones. Returns zeros -> no training
  contribution (this is a metric, not a training term).
  """
  hp = hand_pose_tracking(env, command_name, 0.6, 0.1, 0.1)
  fo = finger_object_distance(env, command_name, palm_dist_rew_w=2.0,
                              grip_thres=0.22, n_finger_sum=4)
  op = object_pos_tracking(env, command_name, grip_thres=0.22, n_finger_sum=4)
  ib = object_inplace_bonus(env, command_name, grip_thres=0.22, n_finger_sum=4)
  fair = 0.5 * hp + 0.3 * fo + 1.0 * op + 1.0 * ib

  accum = getattr(env, "_fair_accum", None)
  if accum is None or accum.shape[0] != fair.shape[0]:
    accum = torch.zeros_like(fair)
    env._fair_last = torch.zeros((), device=fair.device)
  accum = accum + fair
  done = env.reset_buf.bool()
  if bool(done.any()):
    env._fair_last = accum[done].mean()       # mean episode-sum of finished episodes
    accum = accum.clone()
    accum[done] = 0.0
  env._fair_accum = accum

  # Publish every step: the reset step's extras['log'] gets wiped by _reset_idx,
  # but the value persists in env._fair_last and is re-published next step.
  log = env.extras.setdefault("log", {}) if isinstance(env.extras, dict) else None
  if log is not None:
    log["fair_reward"] = env._fair_last
  return torch.zeros_like(fair)


# ----- cgsmooth_b2_softclip patches (on top of pinall3) ---------------------


def contact_guide(env, command_name: str = "motion", beta: float = 8.0) -> torch.Tensor:
  """B2 contact guidance: +mean over contact fingers of exp(-beta*d), where d is
  the sim fingertip -> reference contact point (object-local point transformed by
  the LIVE object pose). Gated per finger by the true contact flag. weight=CONTACT_COEF.

    cg_value = sum_f flag_f * exp(-beta * d_f) / (sum_f flag_f + 1e-6)
  """
  cmd, _, sim_tips, _, obj_pos = _hand_points(env, command_name)
  obj_quat = cmd.obj_quat  # (E,4) wxyz active object (multi-object aware)
  local = cmd.ref_contact_local  # (E,5,3) object-local
  # world contact point = obj_pos + R(obj_quat) @ local, per finger.
  q = obj_quat.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 4)
  world = quat_apply(q, local.reshape(-1, 3)).reshape(-1, 5, 3) + obj_pos.unsqueeze(1)
  d = torch.norm(sim_tips - world, p=2, dim=-1)  # (E,5)
  flag = cmd.ref_contact_flag  # (E,5)
  v = torch.exp(-beta * d)
  cg = (flag * v).sum(dim=-1)
  fs = flag.sum(dim=-1)
  return cg / (fs + 1e-6)


def soft_joint_limit(env, command_name: str = "motion") -> torch.Tensor:
  """SOFT_LIMIT: -sum_finger [relu(lo-q)^2 + relu(q-hi)^2] over the 20 finger joints
  (skip the 6 base WRJ0* DOF). Anti reverse-joint. weight=SOFT_LIMIT_COEF.
  """
  cmd = env.command_manager.get_term(command_name)
  robot = env.scene[cmd.cfg.hand_entity_name]
  q = robot.data.joint_pos[:, 6:]  # finger joints (q26 order: 6 base first)
  lim = robot.data.soft_joint_pos_limits[:, 6:]  # (E, nfinger, 2)
  below = torch.clamp(lim[..., 0] - q, min=0.0)
  above = torch.clamp(q - lim[..., 1], min=0.0)
  return -(below * below + above * above).sum(dim=-1)


def action_rate_l2(env, command_name: str = "motion") -> torch.Tensor:
  """ACTION_RATE: -[||a_t - a_{t-1}||^2 + ||a_t - 2a_{t-1} + a_{t-2}||^2] (1st+2nd order
  residual rate, penalises twitchy actions). weight=ACTION_RATE_COEF.
  """
  am = env.action_manager
  a, p, pp = am.action, am.prev_action, am.prev_prev_action
  first = torch.sum((a - p) ** 2, dim=-1)
  second = torch.sum((a - 2.0 * p + pp) ** 2, dim=-1)
  return -(first + second)

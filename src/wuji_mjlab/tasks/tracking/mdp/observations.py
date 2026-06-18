# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Observation terms.

Two obs sets (selected by env_cfg action/obs mode):
- "simple": the compact 114-d set that the first successful run used.
- "full":   a faithful port of DexTrack's pure_state_wref_wdelta obs (repose_z
            is False -> NO frame rotation; use_local_canonical_state -> hand base
            translation is expressed relative to the object). Pure scale factors
            (unscale / vel_obs_scale) are omitted because mjlab normalizes obs.
            Components mirror DexTrack; minor redundant/ambiguous terms (the dup
            goal-delta, the 7-d hand-goal pose delta) are approximated/omitted and
            noted -- the policy is retrained, so byte-identity isn't required.
"""

from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg


# --- terminations ---------------------------------------------------------

def object_unstable(
  env, command_name: str = "motion", max_pos: float = 3.0, max_vel: float = 80.0
) -> torch.Tensor:
  """Reset envs whose active object or hand state has gone NaN / out of bounds.

  Rare physics blowups (a finger clipping an object -> huge contact force) at scale
  produce NaN that would otherwise trip mjlab's strict obs-NaN assert and kill the
  WHOLE run. Terminating those envs runs BEFORE reward/obs, so they reset to a clean
  reference state and the obs stays finite (Isaac Gym / DexTrack tolerate blowups the
  same way via reset). Returns (E,) bool."""
  cmd = env.command_manager.get_term(command_name)
  robot = env.scene[cmd.cfg.hand_entity_name]
  p, v = cmd.obj_pos, cmd.obj_linvel
  qv = robot.data.joint_vel
  bad = (torch.isnan(p).any(-1) | torch.isnan(v).any(-1) | torch.isnan(qv).any(-1)
         | (p.abs() > max_pos).any(-1) | (v.abs() > max_vel).any(-1))
  return bad

# --- simple obs (the successful baseline) ---------------------------------


def object_pos_b(env, command_name: str = "motion") -> torch.Tensor:
  cmd = env.command_manager.get_term(command_name)
  return cmd.obj_pos - env.scene.env_origins


def object_quat_w(env, command_name: str = "motion") -> torch.Tensor:
  cmd = env.command_manager.get_term(command_name)
  return cmd.obj_quat


def object_pos_error(env, command_name: str = "motion") -> torch.Tensor:
  cmd = env.command_manager.get_term(command_name)
  return cmd.ref_obj_pos - cmd.obj_pos


# --- full DexTrack-faithful obs -------------------------------------------

_FINGER_BODIES = tuple(f"right_finger{i}_link4" for i in range(1, 6))  # th,ff,mf,rf,lf
_PALM_BODY = "right_palm_link"


def _idx(env, command_name):
  cmd = env.command_manager.get_term(command_name)
  robot = env.scene[cmd.cfg.hand_entity_name]
  obj = cmd.obj  # objs[0] in both modes; object state via cmd.obj_* accessors
  cache = getattr(env, "_fullobs_idx", None)
  if cache is None:
    fin = robot.find_bodies(list(_FINGER_BODIES), preserve_order=True)[0]
    palm = robot.find_bodies([_PALM_BODY], preserve_order=True)[0][0]
    cache = (torch.as_tensor(fin, device=env.device, dtype=torch.long), palm)
    env._fullobs_idx = cache
  return cmd, robot, obj, cache[0], cache[1]


def ho_hand_qpos(env, command_name: str = "motion") -> torch.Tensor:
  """Hand qpos (26), base translation made object-relative (use_local_canonical_state)."""
  cmd, robot, obj, _, _ = _idx(env, command_name)
  qp = robot.data.joint_pos.clone()
  qp[:, :3] = qp[:, :3] - cmd.obj_pos
  return qp


def ho_hand_qvel(env, command_name: str = "motion") -> torch.Tensor:
  cmd, robot, _, _, _ = _idx(env, command_name)
  return robot.data.joint_vel


def ho_fingertip_state(env, command_name: str = "motion") -> torch.Tensor:
  """5 fingertips x (pos3 + quat4 + linvel3 + angvel3) = 65."""
  cmd, robot, _, fin, _ = _idx(env, command_name)
  pos = robot.data.body_link_pos_w[:, fin]
  quat = robot.data.body_link_quat_w[:, fin]
  lin = robot.data.body_link_lin_vel_w[:, fin]
  ang = robot.data.body_link_ang_vel_w[:, fin]
  return torch.cat([pos, quat, lin, ang], dim=-1).reshape(env.num_envs, -1)


def ho_palm_state(env, command_name: str = "motion") -> torch.Tensor:
  """Palm pos (object-relative, 3) + palm euler (3) = base rot joints q26[3:6]."""
  cmd, robot, obj, _, palm = _idx(env, command_name)
  palm_pos = robot.data.body_link_pos_w[:, palm] - cmd.obj_pos
  palm_euler = robot.data.joint_pos[:, 3:6]  # fly base rot == palm orientation
  return torch.cat([palm_pos, palm_euler], dim=-1)


def ho_object_state(env, command_name: str = "motion") -> torch.Tensor:
  """obj pos_delta(3) + quat(4) + linvel(3) + angvel(3) + goal_pos_delta(3) = 16."""
  cmd, _, obj, _, _ = _idx(env, command_name)
  pos_delta = cmd.ref_obj_pos - cmd.obj_pos
  return torch.cat(
    [
      pos_delta,
      cmd.obj_quat,
      cmd.obj_linvel,
      cmd.obj_angvel,
      pos_delta,  # DexTrack repeats the goal pos delta
    ],
    dim=-1,
  )


def ho_delta_qpos(env, command_name: str = "motion") -> torch.Tensor:
  """Current hand qpos minus reference qpos (26)."""
  cmd, robot, _, _, _ = _idx(env, command_name)
  return robot.data.joint_pos - cmd.ref_qpos


def ho_next_ref_qpos(env, command_name: str = "motion") -> torch.Tensor:
  """Next-frame reference hand qpos (26) -- use_future_ref_as_obs_goal."""
  return env.command_manager.get_term(command_name).next_ref_qpos


def ho_cumulative_delta(env, command_name: str = "motion") -> torch.Tensor:
  """Accumulated residual delta-targets (26) from the wdelta action (0 in offset mode)."""
  cmd, robot, _, _, _ = _idx(env, command_name)
  act = env.action_manager.get_term("joint_pos")
  accum = getattr(act, "_accum_delta", None)
  if accum is None:
    return torch.zeros((env.num_envs, robot.num_joints), device=env.device)
  return accum


def ho_obj_latent(env, command_name: str = "motion") -> torch.Tensor:
  """Precomputed 256-d object latent feature (DexTrack w_obj_latent_features)."""
  return env.command_manager.get_term(command_name).obj_latent

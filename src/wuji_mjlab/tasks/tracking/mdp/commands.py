# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Hand + object motion-tracking command.

Loads a DexTrack/GRAB retargeted reference (q26 hand state + object pose,
300 frames) and, each control step, exposes the current reference frame
(``ref_qpos``, ``ref_obj_pos``, ``ref_obj_quat``). On episode reset it writes
the reference frame-0 state into the sim (hand joints + object freejoint).

The wuji fly hand has NO free-joint root: all 26 DOF are joints
(6 base [3 slide + 3 hinge] + 5 fingers x 4), so the whole hand reference is
``joint_pos`` and the base is positioned purely by writing joint state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

# q26 column order (must match right_fly_mjlab.xml joint definition order).
Q26_JOINT_NAMES: tuple[str, ...] = (
  "WRJ0x",
  "WRJ0y",
  "WRJ0z",
  "WRJ0rx",
  "WRJ0ry",
  "WRJ0rz",
) + tuple(
  f"right_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)
)


class HandObjectMotionCommand(CommandTerm):
  """Per-frame hand+object reference advancer (see module docstring)."""

  cfg: "HandObjectMotionCommandCfg"

  def __init__(self, cfg: "HandObjectMotionCommandCfg", env):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.hand_entity_name]
    self.obj: Entity = env.scene[cfg.object_entity_name]

    data = np.load(cfg.motion_file, allow_pickle=True).item()
    q = np.asarray(data["robot_delta_states_weights_np"], dtype=np.float32)  # (T,26) q26
    otrans = np.asarray(data["object_transl"], dtype=np.float32)  # (T,3)
    oquat_xyzw = np.asarray(data["object_rot_quat"], dtype=np.float32)  # (T,4) xyzw
    oquat_wxyz = oquat_xyzw[:, [3, 0, 1, 2]]  # -> MuJoCo wxyz

    # Permute q26 columns into the robot's actual joint order.
    name_to_q = {n: i for i, n in enumerate(Q26_JOINT_NAMES)}
    missing = [n for n in self.robot.joint_names if n not in name_to_q]
    if missing:
      raise ValueError(f"robot joints not in q26 layout: {missing}")
    perm = [name_to_q[n] for n in self.robot.joint_names]

    self._ref_qpos = torch.tensor(q[:, perm], device=self.device)  # (T, njoint) robot order
    self._ref_obj_pos = torch.tensor(otrans, device=self.device)  # (T,3)
    self._ref_obj_quat = torch.tensor(oquat_wxyz, device=self.device)  # (T,4) wxyz
    self.time_step_total = int(self._ref_qpos.shape[0])

    # Reference fingertip + palm world positions (FPOS data), for the pinall3
    # FINGER_POS_REW / PALM_POS_REW terms. Verified to match sim FK <=2 mm.
    lk = data["link_key_to_link_pos"]
    lk = lk.item() if hasattr(lk, "item") and not isinstance(lk, dict) else lk
    tips = np.stack(
      [lk[f"right_finger{i}_tip_link"] for i in range(1, 6)], axis=1
    ).astype(np.float32)  # (T,5,3) order th,ff,mf,rf,lf
    self._ref_tip_pos = torch.tensor(tips, device=self.device)  # (T,5,3)
    self._ref_palm_pos = torch.tensor(
      np.asarray(lk["right_palm_link"], dtype=np.float32), device=self.device
    )  # (T,3)

    # Object latent feature (DexTrack w_obj_latent_features): precomputed 256-d
    # per-sequence embedding. Constant for a single sequence.
    if cfg.obj_latent_file:
      import os
      seq = os.path.basename(cfg.motion_file)
      seq = seq.replace("wuji_passive_active_info_", "").replace("_nf_300.npy", "")
      feat_dict = np.load(cfg.obj_latent_file, allow_pickle=True).item()
      if seq not in feat_dict:
        raise KeyError(f"object latent for '{seq}' not in {cfg.obj_latent_file}")
      self._obj_latent = torch.tensor(
        np.asarray(feat_dict[seq], dtype=np.float32), device=self.device
      )  # (256,)
    else:
      self._obj_latent = None

    # B2 contact guidance (cgsmooth_b2_softclip): per-frame contact flag (T,5) +
    # contact point in OBJECT-LOCAL frame (T,5,3), order th,ff,mf,rf,lf. Generated
    # by wuji_pipeline/generate_contact_guidance_grab2.py ("grab_truth_nearest":
    # GRAB true contact flag + nearest in-slice vertex to the wuji fingertip).
    if cfg.contact_file:
      cd = np.load(cfg.contact_file, allow_pickle=True).item()
      self._contact_flag = torch.tensor(
        np.asarray(cd["contact_flag"], dtype=np.float32), device=self.device
      )  # (T,5)
      self._contact_local = torch.tensor(
        np.asarray(cd["contact_pos_local"], dtype=np.float32), device=self.device
      )  # (T,5,3) object-local
    else:
      self._contact_flag = None
      self._contact_local = None

    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_obj_pos"] = torch.zeros(self.num_envs, device=self.device)

  # -- reference accessors (current frame, per-env) --------------------------

  @property
  def ref_qpos(self) -> torch.Tensor:
    return self._ref_qpos[self.time_steps]

  @property
  def ref_obj_pos(self) -> torch.Tensor:
    # mjlab sim positions here are env-local (site_pos_w / root_link_pos_w do NOT
    # include env_origins), so references must NOT add env_origins either.
    return self._ref_obj_pos[self.time_steps]

  @property
  def ref_obj_quat(self) -> torch.Tensor:
    return self._ref_obj_quat[self.time_steps]

  @property
  def ref_fingertip_pos(self) -> torch.Tensor:
    # (E,5,3) env-local frame; order th,ff,mf,rf,lf
    return self._ref_tip_pos[self.time_steps]

  @property
  def ref_palm_pos(self) -> torch.Tensor:
    return self._ref_palm_pos[self.time_steps]

  @property
  def next_ref_qpos(self) -> torch.Tensor:
    # DexTrack uses progress_buf+1 as the obs goal (use_future_ref_as_obs_goal).
    nxt = torch.clamp(self.time_steps + 1, max=self.time_step_total - 1)
    return self._ref_qpos[nxt]

  @property
  def obj_latent(self) -> torch.Tensor:
    return self._obj_latent.unsqueeze(0).expand(self.num_envs, -1)

  @property
  def ref_contact_flag(self) -> torch.Tensor:
    # (E,5) contact flag at current frame; order th,ff,mf,rf,lf.
    return self._contact_flag[self.time_steps]

  @property
  def ref_contact_local(self) -> torch.Tensor:
    # (E,5,3) contact point in object-local frame at current frame.
    return self._contact_local[self.time_steps]

  @property
  def has_contact(self) -> bool:
    return self._contact_flag is not None

  @property
  def command(self) -> torch.Tensor:
    # What the policy sees as the goal: the target joint pose.
    return self.ref_qpos

  # -- CommandTerm hooks -----------------------------------------------------

  def _update_metrics(self) -> None:
    self.metrics["error_joint_pos"] = torch.norm(
      self.ref_qpos - self.robot.data.joint_pos, dim=-1
    )
    self.metrics["error_obj_pos"] = torch.norm(
      self.ref_obj_pos - self.obj.data.root_link_pos_w, dim=-1
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Single-trajectory: always (re)start at frame 0.
    self.time_steps[env_ids] = 0
    t = self.time_steps[env_ids]

    # Write hand joint state = reference frame 0 (clip to soft limits).
    jp = self._ref_qpos[t].clone()
    soft = self.robot.data.soft_joint_pos_limits[env_ids]
    jp = torch.clip(jp, soft[:, :, 0], soft[:, :, 1])
    jv = torch.zeros_like(jp)
    self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

    # Write object freejoint = reference object pose (zero velocity).
    # Env-local frame: no env_origins (see ref_obj_pos note).
    pos = self._ref_obj_pos[t]
    quat = self._ref_obj_quat[t]
    vel = torch.zeros(len(env_ids), 6, device=self.device)
    root_state = torch.cat([pos, quat, vel], dim=-1)
    self.obj.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.obj.reset(env_ids=env_ids)

  def _update_command(self) -> None:
    self.time_steps += 1
    ended = torch.where(self.time_steps >= self.time_step_total)[0]
    if ended.numel() > 0:
      self._resample_command(ended)


@dataclass(kw_only=True)
class HandObjectMotionCommandCfg(CommandTermCfg):
  motion_file: str = ""
  hand_entity_name: str = "robot"
  object_entity_name: str = "object"
  obj_latent_file: str = ""  # obj_type_to_obj_feat.npy (DexTrack w_obj_latent_features)
  contact_file: str = ""  # contact_grab2/<seq>_contact.npy (B2 contact guidance)

  def build(self, env) -> HandObjectMotionCommand:
    return HandObjectMotionCommand(self, env)

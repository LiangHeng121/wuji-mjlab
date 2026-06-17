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

    # Sequence list: multi-trajectory generalist if motion_files is given, else a
    # single sequence (motion_file). All tensors are stacked (S, T, ...) and each
    # env is assigned a sequence via self.env_seq (see _resample_command).
    motion_files = list(cfg.motion_files) if cfg.motion_files else [cfg.motion_file]
    contact_files = list(cfg.contact_files) if cfg.contact_files else (
      [cfg.contact_file] if cfg.contact_file else []
    )
    if contact_files and len(contact_files) != len(motion_files):
      raise ValueError("contact_files must align 1:1 with motion_files")

    name_to_q = {n: i for i, n in enumerate(Q26_JOINT_NAMES)}
    missing = [n for n in self.robot.joint_names if n not in name_to_q]
    if missing:
      raise ValueError(f"robot joints not in q26 layout: {missing}")
    perm = [name_to_q[n] for n in self.robot.joint_names]

    feat_dict = (
      np.load(cfg.obj_latent_file, allow_pickle=True).item() if cfg.obj_latent_file else None
    )

    import os
    qs, ops, oqs, tps, pps, lats, cfs, cls = [], [], [], [], [], [], [], []
    for mf in motion_files:
      data = np.load(mf, allow_pickle=True).item()
      q = np.asarray(data["robot_delta_states_weights_np"], dtype=np.float32)  # (T,26)
      oquat_xyzw = np.asarray(data["object_rot_quat"], dtype=np.float32)  # (T,4) xyzw
      qs.append(q[:, perm])
      ops.append(np.asarray(data["object_transl"], dtype=np.float32))  # (T,3)
      oqs.append(oquat_xyzw[:, [3, 0, 1, 2]])  # -> MuJoCo wxyz (T,4)
      lk = data["link_key_to_link_pos"]
      lk = lk.item() if hasattr(lk, "item") and not isinstance(lk, dict) else lk
      tps.append(np.stack(
        [lk[f"right_finger{i}_tip_link"] for i in range(1, 6)], axis=1
      ).astype(np.float32))  # (T,5,3) th,ff,mf,rf,lf
      pps.append(np.asarray(lk["right_palm_link"], dtype=np.float32))  # (T,3)
      if feat_dict is not None:
        seq = os.path.basename(mf).replace(
          "wuji_passive_active_info_", "").replace("_nf_300.npy", "")
        if seq not in feat_dict:
          raise KeyError(f"object latent for '{seq}' not in {cfg.obj_latent_file}")
        lats.append(np.asarray(feat_dict[seq], dtype=np.float32))  # (256,)

    lens = {a.shape[0] for a in qs}
    if len(lens) != 1:
      raise ValueError(f"all sequences must share length; got {lens}")
    self.num_seqs = len(qs)
    self.time_step_total = int(qs[0].shape[0])

    self._ref_qpos = torch.tensor(np.stack(qs), device=self.device)  # (S,T,J)
    self._ref_obj_pos = torch.tensor(np.stack(ops), device=self.device)  # (S,T,3)
    self._ref_obj_quat = torch.tensor(np.stack(oqs), device=self.device)  # (S,T,4)
    self._ref_tip_pos = torch.tensor(np.stack(tps), device=self.device)  # (S,T,5,3)
    self._ref_palm_pos = torch.tensor(np.stack(pps), device=self.device)  # (S,T,3)
    self._obj_latent = (
      torch.tensor(np.stack(lats), device=self.device) if lats else None  # (S,256)
    )

    # B2 contact guidance: per-frame contact flag (S,T,5) + object-local contact
    # point (S,T,5,3), order th,ff,mf,rf,lf. From generate_contact_guidance_grab2.
    if contact_files:
      for cfile in contact_files:
        cd = np.load(cfile, allow_pickle=True).item()
        cfs.append(np.asarray(cd["contact_flag"], dtype=np.float32))  # (T,5)
        cls.append(np.asarray(cd["contact_pos_local"], dtype=np.float32))  # (T,5,3)
      self._contact_flag = torch.tensor(np.stack(cfs), device=self.device)  # (S,T,5)
      self._contact_local = torch.tensor(np.stack(cls), device=self.device)  # (S,T,5,3)
    else:
      self._contact_flag = None
      self._contact_local = None

    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    # Per-env sequence assignment (random; resampled on episode reset).
    self.env_seq = torch.randint(
      0, self.num_seqs, (self.num_envs,), device=self.device, dtype=torch.long
    )

    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_obj_pos"] = torch.zeros(self.num_envs, device=self.device)

  # -- reference accessors (current frame, per-env) --------------------------

  # All accessors index by (per-env sequence, per-env frame) -> (E, ...).

  @property
  def ref_qpos(self) -> torch.Tensor:
    return self._ref_qpos[self.env_seq, self.time_steps]

  @property
  def ref_obj_pos(self) -> torch.Tensor:
    # mjlab sim positions here are env-local (site_pos_w / root_link_pos_w do NOT
    # include env_origins), so references must NOT add env_origins either.
    return self._ref_obj_pos[self.env_seq, self.time_steps]

  @property
  def ref_obj_quat(self) -> torch.Tensor:
    return self._ref_obj_quat[self.env_seq, self.time_steps]

  @property
  def ref_fingertip_pos(self) -> torch.Tensor:
    # (E,5,3) env-local frame; order th,ff,mf,rf,lf
    return self._ref_tip_pos[self.env_seq, self.time_steps]

  @property
  def ref_palm_pos(self) -> torch.Tensor:
    return self._ref_palm_pos[self.env_seq, self.time_steps]

  @property
  def next_ref_qpos(self) -> torch.Tensor:
    # DexTrack uses progress_buf+1 as the obs goal (use_future_ref_as_obs_goal).
    nxt = torch.clamp(self.time_steps + 1, max=self.time_step_total - 1)
    return self._ref_qpos[self.env_seq, nxt]

  @property
  def obj_latent(self) -> torch.Tensor:
    return self._obj_latent[self.env_seq]  # (E,256)

  @property
  def ref_contact_flag(self) -> torch.Tensor:
    # (E,5) contact flag at current frame; order th,ff,mf,rf,lf.
    return self._contact_flag[self.env_seq, self.time_steps]

  @property
  def ref_contact_local(self) -> torch.Tensor:
    # (E,5,3) contact point in object-local frame at current frame.
    return self._contact_local[self.env_seq, self.time_steps]

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
    # Restart at frame 0 and (multi-seq) draw a fresh sequence for these envs.
    self.time_steps[env_ids] = 0
    if self.num_seqs > 1:
      self.env_seq[env_ids] = torch.randint(
        0, self.num_seqs, (len(env_ids),), device=self.device, dtype=torch.long
      )
    s = self.env_seq[env_ids]
    t = self.time_steps[env_ids]

    # Write hand joint state = reference frame 0 (clip to soft limits).
    jp = self._ref_qpos[s, t].clone()
    soft = self.robot.data.soft_joint_pos_limits[env_ids]
    jp = torch.clip(jp, soft[:, :, 0], soft[:, :, 1])
    jv = torch.zeros_like(jp)
    self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

    # Write object freejoint = reference object pose (zero velocity).
    # Env-local frame: no env_origins (see ref_obj_pos note).
    pos = self._ref_obj_pos[s, t]
    quat = self._ref_obj_quat[s, t]
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
  motion_file: str = ""  # single-sequence (ignored if motion_files is set)
  motion_files: tuple[str, ...] = ()  # multi-sequence generalist (per-env assignment)
  hand_entity_name: str = "robot"
  object_entity_name: str = "object"
  obj_latent_file: str = ""  # obj_type_to_obj_feat.npy (DexTrack w_obj_latent_features)
  contact_file: str = ""  # single-seq contact_grab2/<seq>_contact.npy (B2)
  contact_files: tuple[str, ...] = ()  # multi-seq contact, aligned 1:1 with motion_files

  def build(self, env) -> HandObjectMotionCommand:
    return HandObjectMotionCommand(self, env)

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
    # Object(s). Single-object: one entity. Multi-object generalist: several
    # entities (one per object type) ALL present in every env -- per env, ONE is
    # "active" (matches that env's sequence's object) and the others are parked far
    # away (mujoco-warp shares one compiled model across envs, so we can't swap the
    # mesh per env; we put every object in every env and activate one).
    # Swap mode (path-(c)): ONE object body whose mesh is swapped per-world via
    # geom_dataid (replaces the multi-entity + park scheme). Single entity.
    self.swap = bool(cfg.swap_object_names)
    self.multi_obj = bool(cfg.object_entity_names) and not self.swap
    if self.multi_obj:
      self.objs: list[Entity] = [env.scene[n] for n in cfg.object_entity_names]
      self.num_objs = len(self.objs)
      self.obj: Entity = self.objs[0]  # for cfg compatibility
    else:
      self.obj = env.scene[cfg.object_entity_name]
      self.objs = [self.obj]
      self.num_objs = 1

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

    # Reference hinge angles come from retargeting as RAW values that jump by 2*pi
    # when a joint crosses the +/-pi seam (esp. the base WRJ0r{x,y,z} on rotation-
    # heavy motions: drink/pour/eat). The position actuator then sees a full-turn
    # target error -> huge torque -> the whole hand+object explode. Unwrap along
    # time so every joint target is continuous. No-op for joints that never cross
    # the seam (fingers, base slides). Fixes BOTH training and eval.
    self._ref_qpos = torch.tensor(np.unwrap(np.stack(qs), axis=1), device=self.device)  # (S,T,J)
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

    # Per-sequence object index (multi-object): which object entity each sequence
    # uses. Single-object -> all zeros.
    if self.multi_obj or self.swap:
      so = np.asarray(cfg.seq_object_idx, dtype=np.int64)
      if so.shape[0] != self.num_seqs:
        raise ValueError(f"seq_object_idx ({so.shape[0]}) != num_seqs ({self.num_seqs})")
      self.seq_obj = torch.tensor(so, device=self.device)
    else:
      self.seq_obj = torch.zeros(self.num_seqs, dtype=torch.long, device=self.device)
    # Parked pose per slot: x = 10 + 2j on the floor (env-local), far from the hand.
    self._park = torch.tensor(
      [[10.0 + 2.0 * j, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0] for j in range(self.num_objs)],
      device=self.device,
    )  # (num_objs, 7) pos(3)+quat(4)

    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    # Per-env sequence assignment (random; resampled on episode reset).
    self.env_seq = torch.randint(
      0, self.num_seqs, (self.num_envs,), device=self.device, dtype=torch.long
    )
    self.env_obj = self.seq_obj[self.env_seq]  # per-env active object index

    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_obj_pos"] = torch.zeros(self.num_envs, device=self.device)

    if self.swap:
      self._setup_swap(env)

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

  # -- active object state (per env; single-object = the one entity) -----------

  def _active(self, attr: str) -> torch.Tensor:
    if not self.multi_obj:
      return getattr(self.obj.data, attr)
    stacked = torch.stack([getattr(o.data, attr) for o in self.objs], dim=0)  # (J,E,..)
    e = torch.arange(self.num_envs, device=self.device)
    return stacked[self.env_obj, e]

  @property
  def obj_pos(self) -> torch.Tensor:
    return self._active("root_link_pos_w")

  @property
  def obj_quat(self) -> torch.Tensor:
    return self._active("root_link_quat_w")

  @property
  def obj_linvel(self) -> torch.Tensor:
    return self._active("root_link_lin_vel_w")

  @property
  def obj_angvel(self) -> torch.Tensor:
    return self._active("root_link_ang_vel_w")

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
      self.ref_obj_pos - self.obj_pos, dim=-1
    )

  # -- swap mode (path-(c): per-world geom_dataid mesh swap) ------------------

  def _setup_swap(self, env) -> None:
    """Resolve the single object's geom/body indices + each slot's mesh ids and
    physical params, expand the per-world model fields, write initial assignment.
    cfg.swap_object_names[k]=slot k (aligned w/ seq_object_idx); cfg.swap_params
    [name]={rbound,aabb,mass,idiag,com,iquat} (from get_grab_multiobj_swap_cfg)."""
    self._sim = env.sim
    mjm = env.sim.mj_model
    pre = self.cfg.object_entity_name + "/"  # mjlab attach prefix
    self._vis_geom = mjm.geom(pre + "obj_visual").id
    self._col_geom = mjm.geom(pre + "obj_col").id
    self._obj_body = mjm.body(pre + "obj").id
    obj_jnt = int(mjm.body(pre + "obj").jntadr[0])  # object free joint
    self._obj_dofadr = int(mjm.jnt_dofadr[obj_jnt])  # its 6 dofs: dofadr:dofadr+6
    names = list(self.cfg.swap_object_names)
    params = self.cfg.swap_params or {}
    d = self.device
    keys = ("vmesh", "cmesh", "rb", "aabb", "mass", "idiag", "com", "iquat",
            "dofw", "bodyw", "subm")
    col = {k: [] for k in keys}
    for n in names:
      col["vmesh"].append(mjm.mesh(pre + f"{n}_visual").id)
      col["cmesh"].append(mjm.mesh(pre + f"{n}_col").id)
      p = params[n]
      col["rb"].append(p["rbound"]); col["aabb"].append(p["aabb"])
      col["mass"].append(p["mass"]); col["idiag"].append(p["idiag"])
      col["com"].append(p["com"]); col["iquat"].append(p["iquat"])
      col["dofw"].append(p["dof_invweight0"]); col["bodyw"].append(p["body_invweight0"])
      col["subm"].append(p["subtreemass"])
    self._slot_vis_mesh = torch.tensor(col["vmesh"], dtype=torch.int32, device=d)
    self._slot_col_mesh = torch.tensor(col["cmesh"], dtype=torch.int32, device=d)
    self._slot_rbound = torch.tensor(col["rb"], dtype=torch.float32, device=d)
    self._slot_aabb = torch.tensor(np.stack(col["aabb"]), dtype=torch.float32, device=d)
    self._slot_mass = torch.tensor(col["mass"], dtype=torch.float32, device=d)
    self._slot_idiag = torch.tensor(np.stack(col["idiag"]), dtype=torch.float32, device=d)
    self._slot_com = torch.tensor(np.stack(col["com"]), dtype=torch.float32, device=d)
    self._slot_iquat = torch.tensor(np.stack(col["iquat"]), dtype=torch.float32, device=d)
    self._slot_dofw = torch.tensor(np.stack(col["dofw"]), dtype=torch.float32, device=d)
    self._slot_bodyw = torch.tensor(np.stack(col["bodyw"]), dtype=torch.float32, device=d)
    self._slot_subm = torch.tensor(col["subm"], dtype=torch.float32, device=d)
    # Expand only the per-world fields we write. The object is a SEPARATE kinematic
    # tree (free body) -> block-diagonal mass matrix -> its invweight is independent
    # of the hand and equals the single-object reference (computed in _obj_phys). So
    # we write the object's invweight indices DIRECTLY and never call the global
    # recompute_constants(set_const), which would also recompute the hand's 26-dof
    # invweight every reset (and churn d.qpos/kinematics) -> hand control unstable.
    env.sim.expand_model_fields((
      "geom_dataid", "geom_rbound", "geom_aabb",
      "body_mass", "body_inertia", "body_ipos", "body_iquat",
      "body_subtreemass", "body_invweight0", "dof_invweight0",
    ))
    all_ids = torch.arange(self.num_envs, device=d)
    self._write_swap(all_ids, self.env_obj)  # first episode uses env_seq

  def _write_swap(self, env_ids: torch.Tensor, slots: torch.Tensor) -> None:
    """Write the active object slot's mesh + physical params for env_ids."""
    m = self._sim.model
    m.geom_dataid[env_ids, self._vis_geom] = self._slot_vis_mesh[slots]
    m.geom_dataid[env_ids, self._col_geom] = self._slot_col_mesh[slots]
    m.geom_rbound[env_ids, self._col_geom] = self._slot_rbound[slots]
    m.geom_aabb[env_ids, self._col_geom] = self._slot_aabb[slots]
    m.body_mass[env_ids, self._obj_body] = self._slot_mass[slots]
    m.body_inertia[env_ids, self._obj_body] = self._slot_idiag[slots]
    m.body_ipos[env_ids, self._obj_body] = self._slot_com[slots]
    m.body_iquat[env_ids, self._obj_body] = self._slot_iquat[slots]
    # Solver invweight for the OBJECT only (precomputed per object); the hand's
    # invweight stays at its correct compile-time value (never touched). No global
    # recompute -> hand dofs are not corrupted.
    da = self._obj_dofadr
    m.dof_invweight0[env_ids, da:da + 6] = self._slot_dofw[slots]
    m.body_invweight0[env_ids, self._obj_body] = self._slot_bodyw[slots]
    m.body_subtreemass[env_ids, self._obj_body] = self._slot_subm[slots]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Restart at frame 0 and (multi-seq) draw a fresh sequence for these envs.
    self.time_steps[env_ids] = 0
    if self.cfg.rsi_prob > 0.0:
      # RSI (Reference State Initialization, default OFF): with prob rsi_prob start
      # from a random mid-trajectory frame -- hand+object are teleported to the
      # reference at that frame below (the existing s,t placement code), so the
      # policy gets to experience held/lifted states it could never reach by
      # exploration from frame 0 (DeepMimic-style bootstrap).
      n = len(env_ids)
      tmax = max(int(self.cfg.rsi_max_frac * (self.time_step_total - 1)), 1)
      rand_t = torch.randint(0, tmax, (n,), device=self.device)
      use = torch.rand(n, device=self.device) < self.cfg.rsi_prob
      self.time_steps[env_ids] = torch.where(use, rand_t, torch.zeros_like(rand_t))
    if self.num_seqs > 1:
      self.env_seq[env_ids] = torch.randint(
        0, self.num_seqs, (len(env_ids),), device=self.device, dtype=torch.long
      )
    self.env_obj[env_ids] = self.seq_obj[self.env_seq[env_ids]]
    s = self.env_seq[env_ids]
    t = self.time_steps[env_ids]

    # Write hand joint state = reference frame 0 (clip to soft limits).
    jp = self._ref_qpos[s, t].clone()
    soft = self.robot.data.soft_joint_pos_limits[env_ids]
    jp = torch.clip(jp, soft[:, :, 0], soft[:, :, 1])
    jv = torch.zeros_like(jp)
    self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

    # Object freejoint(s). Env-local frame: no env_origins (see ref_obj_pos note).
    # Active object -> reference pose; inactive objects -> parked far away. For
    # multi-object each slot is written for all reset envs (ref where active, park
    # where not). Zero velocity.
    ref_pos = self._ref_obj_pos[s, t]
    ref_quat = self._ref_obj_quat[s, t]
    vel = torch.zeros(len(env_ids), 6, device=self.device)
    if self.swap:
      # Single body: write the reference pose, then swap its mesh + physical
      # params per-world to the active object slot (no park; only one body).
      self.obj.write_root_state_to_sim(
        torch.cat([ref_pos, ref_quat, vel], dim=-1), env_ids=env_ids)
      self.obj.reset(env_ids=env_ids)
      self._write_swap(env_ids, self.env_obj[env_ids])
      return
    active = self.env_obj[env_ids]  # (n,) which slot is active per reset env
    for j, obj in enumerate(self.objs):
      is_active = (active == j).unsqueeze(-1)  # (n,1)
      pos = torch.where(is_active, ref_pos, self._park[j, :3].unsqueeze(0))
      quat = torch.where(is_active, ref_quat, self._park[j, 3:].unsqueeze(0))
      obj.write_root_state_to_sim(torch.cat([pos, quat, vel], dim=-1), env_ids=env_ids)
      obj.reset(env_ids=env_ids)

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
  object_entity_name: str = "object"  # single-object mode
  object_entity_names: tuple[str, ...] = ()  # multi-object: one entity per obj type
  swap_object_names: tuple[str, ...] = ()  # path-(c) swap: object name per slot
  swap_params: dict | None = None  # per-object phys (from get_grab_multiobj_swap_cfg)
  seq_object_idx: tuple[int, ...] = ()  # per-seq object slot (aligned w/ motion_files)
  obj_latent_file: str = ""  # obj_type_to_obj_feat.npy (DexTrack w_obj_latent_features)
  contact_file: str = ""  # single-seq contact_grab2/<seq>_contact.npy (B2)
  contact_files: tuple[str, ...] = ()  # multi-seq contact, aligned 1:1 with motion_files
  # RSI ablation (default OFF): prob of starting an episode at a random reference
  # frame in [0, rsi_max_frac*T) instead of frame 0 (hand+object teleported there).
  rsi_prob: float = 0.0
  rsi_max_frac: float = 0.8

  def build(self, env) -> HandObjectMotionCommand:
    return HandObjectMotionCommand(self, env)

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji fly-hand cubesmall tracking env config (thin binding layer)."""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from wuji_mjlab.tasks.tracking.config.wuji_hand.robot import get_wuji_fly_hand_cfg
from wuji_mjlab.tasks.tracking.grab_object_cfg import get_grab_object_cfg
from wuji_mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

# Single cubesmall reference (FPOS retarget, same data used on the Isaac Gym side).
_MOTION_FILE = (
  "/home/liangh/DexTrack/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_FPOS_v1/"
  "data/wuji_passive_active_info_ori_grab_s2_cubesmall_inspect_1_nf_300.npy"
)
_OBJECT_NAME = "cubesmall"  # real GRAB mesh (matches the retargeted reference)
_OBJ_LATENT_FILE = "/home/liangh/DexTrack/assets/obj_type_to_obj_feat.npy"
_SEQ = "ori_grab_s2_cubesmall_inspect_1"
# B2 contact guidance data (true-flag + nearest in-slice vertex) for cgsmooth.
_CONTACT_FILE = (
  "/home/liangh/DexTrack/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_FPOS_v1/"
  f"contact_grab2/{_SEQ}_contact.npy"
)


def wuji_hand_cubesmall_tracking_env_cfg(
  play: bool = False, num_envs: int = 4096,
  action_mode: str = "offset", obs_mode: str = "full",
  scale_rewards_by_dt: bool = False, finger_kp_scale: float = 8.0,
  obj_friction: float | None = None, reward_mode: str = "pinall3",
) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg(
    num_envs=num_envs, action_mode=action_mode, obs_mode=obs_mode,
    scale_rewards_by_dt=scale_rewards_by_dt, reward_mode=reward_mode,
  )

  cfg.scene.entities = {
    "robot": get_wuji_fly_hand_cfg(finger_kp_scale=finger_kp_scale),
    "object": get_grab_object_cfg(
      _OBJECT_NAME, **({"friction": obj_friction} if obj_friction is not None else {})
    ),
  }
  cfg.commands["motion"].motion_file = _MOTION_FILE
  cfg.commands["motion"].obj_latent_file = _OBJ_LATENT_FILE
  if reward_mode == "cgsmooth_b2_softclip":  # only this mode needs contact data
    cfg.commands["motion"].contact_file = _CONTACT_FILE
  cfg.viewer.body_name = "right_palm_link"

  if play:
    cfg.scene.num_envs = 4
    cfg.observations["policy"].enable_corruption = False

  if not Path(_MOTION_FILE).exists():
    raise FileNotFoundError(f"reference motion not found: {_MOTION_FILE}")
  return cfg


# ----- multi-sequence cubesmall generalist (all subjects, exclude offhand) -----
_DATA_DIR = Path(
  "/home/liangh/DexTrack/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_FPOS_v1"
)
_MOTION_DIR = _DATA_DIR / "data"
_CONTACT_DIR = _DATA_DIR / "contact_grab2"


def _cubesmall_sequences() -> list[str]:
  """All cubesmall seq names with motion+contact data, excluding 'offhand'."""
  seqs = []
  for p in sorted(_MOTION_DIR.glob("wuji_passive_active_info_*cubesmall*_nf_300.npy")):
    seq = p.name.replace("wuji_passive_active_info_", "").replace("_nf_300.npy", "")
    if "offhand" in seq:
      continue
    if not (_CONTACT_DIR / f"{seq}_contact.npy").exists():
      continue  # need contact for cgsmooth; keep the set consistent across configs
    seqs.append(seq)
  if not seqs:
    raise FileNotFoundError(f"no cubesmall sequences under {_MOTION_DIR}")
  return seqs


def wuji_hand_cubesmall_multi_tracking_env_cfg(
  play: bool = False, num_envs: int = 4096,
  action_mode: str = "wdelta", obs_mode: str = "full",
  scale_rewards_by_dt: bool = False, finger_kp_scale: float = 1.0,
  reward_mode: str = "pinall3",
) -> ManagerBasedRlEnvCfg:
  """Generalist over all cubesmall sequences (s1..s10 inspect/lift/pass, NO offhand).
  Each env is assigned a random sequence, resampled on episode reset."""
  cfg = make_tracking_env_cfg(
    num_envs=num_envs, action_mode=action_mode, obs_mode=obs_mode,
    scale_rewards_by_dt=scale_rewards_by_dt, reward_mode=reward_mode,
  )
  cfg.scene.entities = {
    "robot": get_wuji_fly_hand_cfg(finger_kp_scale=finger_kp_scale),
    "object": get_grab_object_cfg(_OBJECT_NAME),
  }
  seqs = _cubesmall_sequences()
  motion_files = tuple(str(_MOTION_DIR / f"wuji_passive_active_info_{s}_nf_300.npy") for s in seqs)
  cfg.commands["motion"].motion_files = motion_files
  cfg.commands["motion"].obj_latent_file = _OBJ_LATENT_FILE
  if reward_mode == "cgsmooth_b2_softclip":
    cfg.commands["motion"].contact_files = tuple(
      str(_CONTACT_DIR / f"{s}_contact.npy") for s in seqs
    )
  cfg.viewer.body_name = "right_palm_link"

  if play:
    cfg.scene.num_envs = 4
    cfg.observations["policy"].enable_corruption = False
  return cfg

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


def wuji_hand_cubesmall_tracking_env_cfg(
  play: bool = False, num_envs: int = 4096,
  action_mode: str = "offset", obs_mode: str = "full",
  scale_rewards_by_dt: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg(
    num_envs=num_envs, action_mode=action_mode, obs_mode=obs_mode,
    scale_rewards_by_dt=scale_rewards_by_dt,
  )

  cfg.scene.entities = {
    "robot": get_wuji_fly_hand_cfg(),
    "object": get_grab_object_cfg(_OBJECT_NAME),
  }
  cfg.commands["motion"].motion_file = _MOTION_FILE
  cfg.commands["motion"].obj_latent_file = _OBJ_LATENT_FILE
  cfg.viewer.body_name = "right_palm_link"

  if play:
    cfg.scene.num_envs = 4
    cfg.observations["policy"].enable_corruption = False

  if not Path(_MOTION_FILE).exists():
    raise FileNotFoundError(f"reference motion not found: {_MOTION_FILE}")
  return cfg

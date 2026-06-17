# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji fly-hand cubesmall tracking env config (thin binding layer)."""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from wuji_mjlab.assets.objects.inhand_object.object_cfg import get_inhand_object_cfg
from wuji_mjlab.tasks.tracking.config.wuji_hand.robot import get_wuji_fly_hand_cfg
from wuji_mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

# Single cubesmall reference (FPOS retarget, same data used on the Isaac Gym side).
_MOTION_FILE = (
  "/home/liangh/DexTrack/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_FPOS_v1/"
  "data/wuji_passive_active_info_ori_grab_s2_cubesmall_inspect_1_nf_300.npy"
)
_CUBESMALL_EDGE_M = 0.05  # ~5 cm cube


def wuji_hand_cubesmall_tracking_env_cfg(
  play: bool = False, num_envs: int = 4096
) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg(num_envs=num_envs)

  cfg.scene.entities = {
    "robot": get_wuji_fly_hand_cfg(),
    "object": get_inhand_object_cfg(edge_m=_CUBESMALL_EDGE_M),
  }
  cfg.commands["motion"].motion_file = _MOTION_FILE
  cfg.viewer.body_name = "right_palm_link"

  if play:
    cfg.scene.num_envs = 4
    cfg.observations["policy"].enable_corruption = False

  if not Path(_MOTION_FILE).exists():
    raise FileNotFoundError(f"reference motion not found: {_MOTION_FILE}")
  return cfg

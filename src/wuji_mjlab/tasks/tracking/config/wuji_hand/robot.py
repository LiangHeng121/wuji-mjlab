# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji fly-hand EntityCfg for tracking: 20 finger DOF + 6-DOF floating base.

Uses ``right_fly_mjlab.xml`` (the vendor hand + a 6-DOF base of 3 slide + 3
hinge joints, WRJ0x/y/z/rx/ry/rz). All 26 joints are position-actuated; the
base is positioned purely by joint state (no free-joint root).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

_FLY_XML = (
  Path(__file__).resolve().parents[4]
  / "assets"
  / "robots"
  / "wuji_hand"
  / "mjcf"
  / "right_fly_mjlab.xml"
)
assert _FLY_XML.exists(), f"missing fly hand xml: {_FLY_XML}"


def _get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(_FLY_XML))
  mesh_root = (_FLY_XML.parent / spec.meshdir).resolve()
  assets: dict[str, bytes] = {}
  for pattern in ("*.STL", "*.stl"):
    for f in mesh_root.glob(pattern):
      assets[f"{spec.meshdir}/{f.name}"] = f.read_bytes()
  spec.assets = assets
  return spec


WUJI_FLY_HAND_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlActuatorCfg(target_names_expr=(".*",)),  # all 26 joints (base + fingers)
  ),
  soft_joint_pos_limit_factor=0.9,
)

# Episode reset always overwrites this with the reference frame-0 state, so the
# home pose only needs to be valid. Base joints default to 0 (hand at origin).
WUJI_FLY_HAND_HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    "WRJ0.*": 0.0,
    ".*_finger1_joint1": 0.34,
    ".*_finger1_joint2": -0.08,
    ".*_finger1_joint3": 0.97,
    ".*_finger1_joint4": 0.78,
    ".*_finger2_joint1": 0.93,
    ".*_finger2_joint2": 0.17,
    ".*_finger2_joint3": 0.96,
    ".*_finger2_joint4": 0.80,
    ".*_finger3_joint1": 1.02,
    ".*_finger3_joint2": -0.02,
    ".*_finger3_joint3": 0.83,
    ".*_finger3_joint4": 0.76,
    ".*_finger4_joint1": 0.98,
    ".*_finger4_joint2": -0.16,
    ".*_finger4_joint3": 0.71,
    ".*_finger4_joint4": 0.88,
    ".*_finger5_joint1": 0.83,
    ".*_finger5_joint2": -0.21,
    ".*_finger5_joint3": 0.76,
    ".*_finger5_joint4": 1.23,
  },
  joint_vel={".*": 0.0},
)


def get_wuji_fly_hand_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=WUJI_FLY_HAND_HOME,
    spec_fn=_get_spec,
    articulation=WUJI_FLY_HAND_ARTICULATION,
  )

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Hand+object motion-tracking task configuration (robot-agnostic factory).

Minimal single-trajectory tracking: a kinematics-bias action drives the hand to
follow a GRAB-retargeted q26 reference while a free-floating object should be
carried along its reference pose. Robot/object entities + the motion file are
bound in ``config/<robot>/env_cfgs.py``.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from wuji_mjlab.tasks.tracking import mdp
from wuji_mjlab.tasks.tracking.mdp.actions import KinematicsBiasActionCfg
from wuji_mjlab.tasks.tracking.mdp.commands import HandObjectMotionCommandCfg


def make_tracking_env_cfg(num_envs: int = 4096) -> ManagerBasedRlEnvCfg:
  """Create the base hand+object tracking config."""

  ##
  # Observations
  ##

  def _obs_terms(noisy: bool) -> dict[str, ObservationTermCfg]:
    jp_noise = Unoise(n_min=-0.01, n_max=0.01) if noisy else None
    jv_noise = Unoise(n_min=-0.5, n_max=0.5) if noisy else None
    return {
      "command": ObservationTermCfg(
        func=mdp.generated_commands, params={"command_name": "motion"}
      ),
      "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, noise=jp_noise),
      "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=jv_noise),
      "object_pos": ObservationTermCfg(
        func=mdp.object_pos_b, params={"command_name": "motion"}
      ),
      "object_quat": ObservationTermCfg(
        func=mdp.object_quat_w, params={"command_name": "motion"}
      ),
      "object_pos_error": ObservationTermCfg(
        func=mdp.object_pos_error, params={"command_name": "motion"}
      ),
      "actions": ObservationTermCfg(func=mdp.last_action),
    }

  observations = {
    "policy": ObservationGroupCfg(
      terms=_obs_terms(noisy=True),
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=_obs_terms(noisy=False),
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": KinematicsBiasActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      action_scale=0.5,
      command_name="motion",
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "motion": HandObjectMotionCommandCfg(
      resampling_time_range=(1.0e9, 1.0e9),  # never time-based resample
      hand_entity_name="robot",
      object_entity_name="object",
      motion_file="",  # set per-robot
    )
  }

  ##
  # Rewards
  ##

  rewards: dict[str, RewardTermCfg] = {
    "joint_tracking": RewardTermCfg(
      func=mdp.joint_tracking_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.5},
    ),
    "object_pos_tracking": RewardTermCfg(
      func=mdp.object_pos_tracking_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.05},
    ),
    "object_ori_tracking": RewardTermCfg(
      func=mdp.object_ori_tracking_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.5},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-2),
  }

  ##
  # Terminations
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  }

  ##
  # Assemble
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=num_envs,
      env_spacing=0.75,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events={},
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # set per-robot
      distance=0.7,
      fovy=55.0,
      elevation=-15.0,
      azimuth=130.0,
    ),
    sim=SimulationCfg(
      nconmax=200,
      njmax=800,
      mujoco=MujocoCfg(
        timestep=0.0083,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=2,  # control dt = 0.0166 s ~= GRAB 60 Hz reference
    episode_length_s=5.0,  # ~300 control steps
  )

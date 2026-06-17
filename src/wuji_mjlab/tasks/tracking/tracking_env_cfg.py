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


def make_tracking_env_cfg(
  num_envs: int = 4096, action_mode: str = "offset", obs_mode: str = "full",
  scale_rewards_by_dt: bool = False, reward_mode: str = "pinall3",
) -> ManagerBasedRlEnvCfg:
  """Create the base hand+object tracking config.

  action_mode: "offset" (uniform residual, action_scale 0.1) or "wdelta"
    (DexTrack-exact accumulative per-group residual).
  obs_mode: "full" (default; faithful DexTrack pure_state_wref_wdelta obs, ~499-d
    incl 256-d object latent) or "simple" (the 114-d set the first success used).
  scale_rewards_by_dt: False (default) matches DexTrack/rl_games (no dt scaling ->
    episode returns ~hundreds). True is the original mjlab default (dt-scaled ->
    single-digit returns; the first success used True).
  reward_mode: which DexTrack reward lineage to use:
    - "original": base only (hand_pose 0.6/0.1/0.1, finger/palm-obj dist with
      palm_dist_rew_w=2.0 + grip 0.12 + 4 fingers, gated obj pos + in-place bonus).
    - "pinall3" (default): base + RELAX_PALM (palm_dist_rew_w 0, grip 0.22) +
      FIX_FINGER5 (5 fingers) + FINGER_POS_REW(1.0) + PALM_POS_REW(1.0).
    - "cgsmooth_b2_softclip": pinall3 + contact_guide(B2, beta=8) + HAND_EMA(0.4,
      in the action) + action_rate(0.0005) + soft_joint_limit(0.5).
  """

  ##
  # Observations
  ##

  def _simple_terms(noisy: bool) -> dict[str, ObservationTermCfg]:
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

  def _full_terms() -> dict[str, ObservationTermCfg]:
    # Faithful DexTrack pure_state_wref_wdelta obs (see observations.py).
    p = {"command_name": "motion"}
    return {
      "hand_qpos": ObservationTermCfg(func=mdp.ho_hand_qpos, params=p),
      "hand_qvel": ObservationTermCfg(func=mdp.ho_hand_qvel, params=p),
      "fingertips": ObservationTermCfg(func=mdp.ho_fingertip_state, params=p),
      "palm": ObservationTermCfg(func=mdp.ho_palm_state, params=p),
      "actions": ObservationTermCfg(func=mdp.last_action),
      "object": ObservationTermCfg(func=mdp.ho_object_state, params=p),
      "delta_qpos": ObservationTermCfg(func=mdp.ho_delta_qpos, params=p),
      "next_ref_qpos": ObservationTermCfg(func=mdp.ho_next_ref_qpos, params=p),
      "cumulative_delta": ObservationTermCfg(func=mdp.ho_cumulative_delta, params=p),
      "obj_latent": ObservationTermCfg(func=mdp.ho_obj_latent, params=p),
    }

  if obs_mode == "simple":
    policy_terms, critic_terms = _simple_terms(noisy=True), _simple_terms(noisy=False)
  else:
    policy_terms = critic_terms = _full_terms()

  observations = {
    "policy": ObservationGroupCfg(
      terms=policy_terms, concatenate_terms=True,
      enable_corruption=(obs_mode == "simple"),
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms, concatenate_terms=True, enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  # HAND_EMA finger-target low-pass is part of the cgsmooth patch only.
  hand_ema_coef = 0.4 if reward_mode == "cgsmooth_b2_softclip" else 0.0

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": KinematicsBiasActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      mode=action_mode,
      # offset mode: small residual — the zero-residual reference already lifts the
      # cube (with kp x8), so constrain the policy to small corrections around it.
      action_scale=0.1,
      # wdelta mode: per-group speed scales. DexTrack wuji uses dof_speed_scale=20,
      # but that gives finger residual ~+-0.66 rad/step -> too twitchy in MuJoCo
      # (bang-bang actions, hand jitters). Reduced to 5 (~+-0.17 rad/step) to test.
      glb_trans_vel_scale=0.5,
      glb_rot_vel_scale=0.5,
      dof_speed_scale=5.0,
      hand_ema_coef=hand_ema_coef,
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

  # DexTrack reward lineage, selected by reward_mode (weights = DexTrack coefs;
  # funcs return the signed reward so weights are positive coef magnitudes).
  if reward_mode == "original":
    # base: tighter palm grip (0.12), 4-finger sum, palm-distance penalty ON.
    grip, n_sum, palm_w = 0.12, 4, 2.0
  else:  # pinall3 and cgsmooth_b2_softclip share the RELAX_PALM/FIX_FINGER5 base.
    grip, n_sum, palm_w = 0.22, 5, 0.0

  rewards: dict[str, RewardTermCfg] = {
    "hand_pose_tracking": RewardTermCfg(  # rew_delta_hand_pose_coef = 0.5
      func=mdp.hand_pose_tracking,
      weight=0.5,
      params={"command_name": "motion",
              "trans_coef": 0.6, "rot_coef": 0.1, "finger_coef": 0.1},
    ),
    "finger_object_distance": RewardTermCfg(  # rew_finger_obj_dist_coef = 0.3
      func=mdp.finger_object_distance,
      weight=0.3,
      params={"command_name": "motion", "palm_dist_rew_w": palm_w,
              "grip_thres": grip, "n_finger_sum": n_sum},
    ),
    "object_pos_tracking": RewardTermCfg(  # goal_hand_rew, gated by grasp flag
      func=mdp.object_pos_tracking,
      weight=1.0,
      params={"command_name": "motion", "grip_thres": grip, "n_finger_sum": n_sum},
    ),
    "object_inplace_bonus": RewardTermCfg(  # in-place bonus, gated by grasp flag
      func=mdp.object_inplace_bonus,
      weight=1.0,
      params={"command_name": "motion", "grip_thres": grip, "n_finger_sum": n_sum},
    ),
  }

  # FAIR metric: config-independent comparable reward, logged to wandb/tb (weight
  # 1.0 but the func returns 0 -> no training contribution; see fair_reward_metric).
  rewards["fair_reward_metric"] = RewardTermCfg(
    func=mdp.fair_reward_metric, weight=1.0, params={"command_name": "motion"},
  )

  if reward_mode != "original":  # pinall3 dense fingertip/palm tracking terms
    rewards["finger_pos_tracking"] = RewardTermCfg(  # FINGER_POS_COEF=1.0
      func=mdp.finger_pos_tracking, weight=1.0, params={"command_name": "motion"},
    )
    rewards["palm_pos_tracking"] = RewardTermCfg(  # PALM_POS_COEF=1.0
      func=mdp.palm_pos_tracking, weight=1.0, params={"command_name": "motion"},
    )

  if reward_mode == "cgsmooth_b2_softclip":  # B2 contact + action-rate + soft-limit
    rewards["contact_guide"] = RewardTermCfg(  # CONTACT_COEF=1.0, CONTACT_BETA=8
      func=mdp.contact_guide, weight=1.0,
      params={"command_name": "motion", "beta": 8.0},
    )
    rewards["action_rate"] = RewardTermCfg(  # ACTION_RATE_COEF=0.0005
      func=mdp.action_rate_l2, weight=0.0005, params={"command_name": "motion"},
    )
    rewards["soft_joint_limit"] = RewardTermCfg(  # SOFT_LIMIT_COEF=0.5
      func=mdp.soft_joint_limit, weight=0.5, params={"command_name": "motion"},
    )

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
      # Per-env contact/constraint buffer caps. These dominate mujoco-warp
      # memory; 200/800 was wildly over-provisioned (cut 22000-env VRAM ~2x).
      # 128/640 leaves margin over a single-cube grasp (reorient, far more
      # contact-heavy, uses 180) while fitting 22000 envs comfortably.
      nconmax=96,
      njmax=512,
      mujoco=MujocoCfg(
        timestep=0.0083,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=2,  # control dt = 0.0166 s ~= GRAB 60 Hz reference
    episode_length_s=5.0,  # ~300 control steps
    scale_rewards_by_dt=scale_rewards_by_dt,
  )

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji fly-hand tracking task registration."""

from mjlab.tasks.registry import register_mjlab_task

from wuji_mjlab.rl.runner import WujiOnPolicyRunner

from .env_cfgs import wuji_hand_cubesmall_tracking_env_cfg
from .rsl_rl.ppo import wuji_hand_tracking_ppo_runner_cfg

# DEFAULT: DexTrack-faithful — full obs + wdelta accumulative action (finger
# dof_speed_scale reduced 20->5 for MuJoCo) + kp x8 + magnitude-aligned rewards.
# Verified: cube lifted to 0.407 m, 2 mm tracking, low jitter (0.062/step).
register_mjlab_task(
  task_id="WujiHand_Tracking_Cubesmall",
  env_cfg=wuji_hand_cubesmall_tracking_env_cfg(num_envs=4096, action_mode="wdelta"),
  play_env_cfg=wuji_hand_cubesmall_tracking_env_cfg(play=True, action_mode="wdelta"),
  rl_cfg=wuji_hand_tracking_ppo_runner_cfg(max_iterations=10000),
  runner_cls=WujiOnPolicyRunner,
)

# Variant: offset action (uniform residual 0.1). Also works (0.408 m); the proven
# first config before wdelta was tuned.
register_mjlab_task(
  task_id="WujiHand_Tracking_Cubesmall_Offset",
  env_cfg=wuji_hand_cubesmall_tracking_env_cfg(num_envs=4096, action_mode="offset"),
  play_env_cfg=wuji_hand_cubesmall_tracking_env_cfg(play=True, action_mode="offset"),
  rl_cfg=wuji_hand_tracking_ppo_runner_cfg(run_name="Tracking_Offset", max_iterations=10000),
  runner_cls=WujiOnPolicyRunner,
)

# REPRODUCTION baseline: the EXACT first successful config (simple 114-d obs +
# offset + dt-scaled rewards, as it actually ran).
register_mjlab_task(
  task_id="WujiHand_Tracking_Cubesmall_Simple",
  env_cfg=wuji_hand_cubesmall_tracking_env_cfg(
    num_envs=4096, obs_mode="simple", scale_rewards_by_dt=True),
  play_env_cfg=wuji_hand_cubesmall_tracking_env_cfg(
    play=True, obs_mode="simple", scale_rewards_by_dt=True),
  rl_cfg=wuji_hand_tracking_ppo_runner_cfg(run_name="Tracking_Simple", max_iterations=10000),
  runner_cls=WujiOnPolicyRunner,
)

# (wdelta is now the default WujiHand_Tracking_Cubesmall above.)

# Ablation: ORIGINAL vendor finger kp (x1). Open-loop can't lift (grip too weak),
# but does the trained policy learn to compensate via the residual?
register_mjlab_task(
  task_id="WujiHand_Tracking_Cubesmall_OrigKp",
  env_cfg=wuji_hand_cubesmall_tracking_env_cfg(
    num_envs=4096, action_mode="wdelta", finger_kp_scale=1.0),
  play_env_cfg=wuji_hand_cubesmall_tracking_env_cfg(
    play=True, action_mode="wdelta", finger_kp_scale=1.0),
  rl_cfg=wuji_hand_tracking_ppo_runner_cfg(run_name="Tracking_OrigKp", max_iterations=10000),
  runner_cls=WujiOnPolicyRunner,
)

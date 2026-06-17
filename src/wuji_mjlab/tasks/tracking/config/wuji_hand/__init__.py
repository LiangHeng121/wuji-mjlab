# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji fly-hand tracking task registration."""

from mjlab.tasks.registry import register_mjlab_task

from wuji_mjlab.rl.runner import WujiOnPolicyRunner

from .env_cfgs import wuji_hand_cubesmall_tracking_env_cfg
from .rsl_rl.ppo import wuji_hand_tracking_ppo_runner_cfg

# DEFAULT: full (DexTrack-faithful) obs + offset action (the proven action/kp/reward).
register_mjlab_task(
  task_id="WujiHand_Tracking_Cubesmall",
  env_cfg=wuji_hand_cubesmall_tracking_env_cfg(num_envs=4096),  # obs full, action offset
  play_env_cfg=wuji_hand_cubesmall_tracking_env_cfg(play=True),
  rl_cfg=wuji_hand_tracking_ppo_runner_cfg(max_iterations=10000),
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

# DexTrack-faithful: full obs + wdelta accumulative action ("only the simulator differs").
register_mjlab_task(
  task_id="WujiHand_Tracking_Cubesmall_Wdelta",
  env_cfg=wuji_hand_cubesmall_tracking_env_cfg(num_envs=4096, action_mode="wdelta"),
  play_env_cfg=wuji_hand_cubesmall_tracking_env_cfg(play=True, action_mode="wdelta"),
  rl_cfg=wuji_hand_tracking_ppo_runner_cfg(run_name="Tracking_Wdelta", max_iterations=10000),
  runner_cls=WujiOnPolicyRunner,
)

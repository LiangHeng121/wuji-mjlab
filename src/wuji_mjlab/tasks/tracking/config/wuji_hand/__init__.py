# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji fly-hand tracking task registration."""

from mjlab.tasks.registry import register_mjlab_task

from wuji_mjlab.rl.runner import WujiOnPolicyRunner

from .env_cfgs import wuji_hand_cubesmall_tracking_env_cfg
from .rsl_rl.ppo import wuji_hand_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="WujiHand_Tracking_Cubesmall",
  env_cfg=wuji_hand_cubesmall_tracking_env_cfg(num_envs=4096),
  play_env_cfg=wuji_hand_cubesmall_tracking_env_cfg(play=True),
  rl_cfg=wuji_hand_tracking_ppo_runner_cfg(max_iterations=10000),
  runner_cls=WujiOnPolicyRunner,
)

# DexTrack-exact action mode (use_kinematics_bias_wdelta): per-group accumulative
# residual. Aim is "only the simulator differs" from DexTrack.
register_mjlab_task(
  task_id="WujiHand_Tracking_Cubesmall_Wdelta",
  env_cfg=wuji_hand_cubesmall_tracking_env_cfg(num_envs=4096, action_mode="wdelta"),
  play_env_cfg=wuji_hand_cubesmall_tracking_env_cfg(play=True, action_mode="wdelta"),
  rl_cfg=wuji_hand_tracking_ppo_runner_cfg(run_name="Tracking_Wdelta", max_iterations=10000),
  runner_cls=WujiOnPolicyRunner,
)

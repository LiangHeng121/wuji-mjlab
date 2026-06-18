# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji fly-hand tracking task registration."""

from mjlab.tasks.registry import register_mjlab_task

from wuji_mjlab.rl.runner import WujiOnPolicyRunner

from .env_cfgs import (
  wuji_hand_3obj_multi_tracking_env_cfg,
  wuji_hand_cubesmall_multi_tracking_env_cfg,
  wuji_hand_cubesmall_tracking_env_cfg,
  wuji_hand_multi_tracking_env_cfg,
)
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
# but the trained policy DOES learn to compensate via the residual (max z 0.409).
register_mjlab_task(
  task_id="WujiHand_Tracking_Cubesmall_OrigKp",
  env_cfg=wuji_hand_cubesmall_tracking_env_cfg(
    num_envs=4096, action_mode="wdelta", finger_kp_scale=1.0),
  play_env_cfg=wuji_hand_cubesmall_tracking_env_cfg(
    play=True, action_mode="wdelta", finger_kp_scale=1.0),
  rl_cfg=wuji_hand_tracking_ppo_runner_cfg(run_name="Tracking_OrigKp", max_iterations=10000),
  runner_cls=WujiOnPolicyRunner,
)

# ----- 3-way REWARD comparison (all kp x1 + wdelta; only the reward differs) -----
# original (base) vs pinall3 (dense fingertip/palm tracking) vs cgsmooth_b2_softclip
# (pinall3 + B2 contact guide + HAND_EMA + action_rate + soft joint limit).
for _rid, _rmode in (
  ("Original", "original"),
  ("Pinall3", "pinall3"),
  ("CGSmooth", "cgsmooth_b2_softclip"),
):
  register_mjlab_task(
    task_id=f"WujiHand_Tracking_Cubesmall_Cmp_{_rid}",
    env_cfg=wuji_hand_cubesmall_tracking_env_cfg(
      num_envs=4096, action_mode="wdelta", finger_kp_scale=1.0, reward_mode=_rmode),
    play_env_cfg=wuji_hand_cubesmall_tracking_env_cfg(
      play=True, action_mode="wdelta", finger_kp_scale=1.0, reward_mode=_rmode),
    rl_cfg=wuji_hand_tracking_ppo_runner_cfg(
      run_name=f"Tracking_Cmp_{_rid}", max_iterations=10000),
    runner_cls=WujiOnPolicyRunner,
  )

# ----- multi-sequence cubesmall generalist (all subjects, NO offhand) ----------
# 3 reward configs again, for running on 3 GPUs in parallel. kp x1 + wdelta.
for _rid, _rmode in (
  ("Original", "original"),
  ("Pinall3", "pinall3"),
  ("CGSmooth", "cgsmooth_b2_softclip"),
):
  register_mjlab_task(
    task_id=f"WujiHand_Tracking_CubesmallMulti_{_rid}",
    env_cfg=wuji_hand_cubesmall_multi_tracking_env_cfg(
      num_envs=4096, reward_mode=_rmode),
    play_env_cfg=wuji_hand_cubesmall_multi_tracking_env_cfg(
      play=True, reward_mode=_rmode),
    rl_cfg=wuji_hand_tracking_ppo_runner_cfg(
      run_name=f"Tracking_CubesmallMulti_{_rid}", max_iterations=10000),
    runner_cls=WujiOnPolicyRunner,
  )

# ----- single-object specialists: cup / apple (cgsmooth_b2_softclip, kp x1) -----
# cup: all non-offhand (drink/lift/pour). apple: all non-offhand (eat/lift/pass,
# matching the DexTrack run -- eat kept per user). Both use B2 cgsmooth recipe.
for _oid, _obj, _excl in (
  ("Cup", "cup", ()),
  ("Apple", "apple", ()),
):
  register_mjlab_task(
    task_id=f"WujiHand_Tracking_{_oid}Multi_CGSmooth",
    env_cfg=wuji_hand_multi_tracking_env_cfg(
      object_name=_obj, num_envs=4096, reward_mode="cgsmooth_b2_softclip",
      extra_exclude=_excl),
    play_env_cfg=wuji_hand_multi_tracking_env_cfg(
      object_name=_obj, play=True, reward_mode="cgsmooth_b2_softclip",
      extra_exclude=_excl),
    rl_cfg=wuji_hand_tracking_ppo_runner_cfg(
      run_name=f"Tracking_{_oid}Multi_CGSmooth", max_iterations=10000),
    runner_cls=WujiOnPolicyRunner,
  )

# ----- multi-OBJECT generalist: cubesmall + cup + apple (cgsmooth_b2_softclip) --
register_mjlab_task(
  task_id="WujiHand_Tracking_3Obj_CGSmooth",
  env_cfg=wuji_hand_3obj_multi_tracking_env_cfg(num_envs=4096),
  play_env_cfg=wuji_hand_3obj_multi_tracking_env_cfg(play=True),
  rl_cfg=wuji_hand_tracking_ppo_runner_cfg(
    run_name="Tracking_3Obj_CGSmooth", max_iterations=10000),
  runner_cls=WujiOnPolicyRunner,
)

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji fly-hand tracking task registration."""

from mjlab.tasks.registry import register_mjlab_task

from wuji_mjlab.rl.runner import WujiOnPolicyRunner

from .env_cfgs import (
  wuji_hand_3obj_multi_tracking_env_cfg,
  wuji_hand_3obj_swap_tracking_env_cfg,
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

# ----- cup grip-strength experiment: kp x8 (direction 2 -- heavier object needs
# more finger force to actually lift off the floor; kp x1 fingers touch the cup
# but can't bear its 97g weight). Same cgsmooth recipe otherwise.
register_mjlab_task(
  task_id="WujiHand_Tracking_CupMulti_CGSmooth_Kp8",
  env_cfg=wuji_hand_multi_tracking_env_cfg(
    object_name="cup", num_envs=4096, reward_mode="cgsmooth_b2_softclip",
    finger_kp_scale=8.0),
  play_env_cfg=wuji_hand_multi_tracking_env_cfg(
    object_name="cup", play=True, reward_mode="cgsmooth_b2_softclip",
    finger_kp_scale=8.0),
  rl_cfg=wuji_hand_tracking_ppo_runner_cfg(
    run_name="Tracking_CupMulti_CGSmooth_Kp8", max_iterations=10000),
  runner_cls=WujiOnPolicyRunner,
)

# ----- cup direction 1: contact-gated grasp (require >=2 fingers REALLY touching
# the cup for object-track/bonus credit; distance flag fires on floor-hover, ~45%
# of cup's bonus was loose-contact). kp x1, same cgsmooth recipe otherwise.
register_mjlab_task(
  task_id="WujiHand_Tracking_CupMulti_CGSmooth_Contact",
  env_cfg=wuji_hand_multi_tracking_env_cfg(
    object_name="cup", num_envs=4096,
    reward_mode="cgsmooth_b2_softclip_contact"),
  play_env_cfg=wuji_hand_multi_tracking_env_cfg(
    object_name="cup", play=True, reward_mode="cgsmooth_b2_softclip_contact"),
  rl_cfg=wuji_hand_tracking_ppo_runner_cfg(
    run_name="Tracking_CupMulti_CGSmooth_Contact", max_iterations=10000),
  runner_cls=WujiOnPolicyRunner,
)

# ----- contact-gated cgsmooth_b2_softclip, single-object: cubesmall + apple -----
for _oid, _obj in (("Cubesmall", "cubesmall"), ("Apple", "apple")):
  register_mjlab_task(
    task_id=f"WujiHand_Tracking_{_oid}Multi_CGSmooth_Contact",
    env_cfg=wuji_hand_multi_tracking_env_cfg(
      object_name=_obj, num_envs=4096, reward_mode="cgsmooth_b2_softclip_contact"),
    play_env_cfg=wuji_hand_multi_tracking_env_cfg(
      object_name=_obj, play=True, reward_mode="cgsmooth_b2_softclip_contact"),
    rl_cfg=wuji_hand_tracking_ppo_runner_cfg(
      run_name=f"Tracking_{_oid}Multi_CGSmooth_Contact", max_iterations=10000),
    runner_cls=WujiOnPolicyRunner,
  )

# ----- apple grip-strength ABLATION: kp x8 + contact-gated. Identical to the
# apple contact specialist above EXCEPT finger_kp_scale 1->8 (the ONLY variable).
# Does more finger force help apple lift? The doc says kp x1 suffices (3obj proved
# it), but verify directly on the single-object apple specialist. Run at the same
# env count as the apple contact specialist (~23000) to isolate kp.
register_mjlab_task(
  task_id="WujiHand_Tracking_AppleMulti_CGSmooth_Contact_Kp8",
  env_cfg=wuji_hand_multi_tracking_env_cfg(
    object_name="apple", num_envs=4096,
    reward_mode="cgsmooth_b2_softclip_contact", finger_kp_scale=8.0),
  play_env_cfg=wuji_hand_multi_tracking_env_cfg(
    object_name="apple", play=True,
    reward_mode="cgsmooth_b2_softclip_contact", finger_kp_scale=8.0),
  rl_cfg=wuji_hand_tracking_ppo_runner_cfg(
    run_name="Tracking_AppleMulti_CGSmooth_Contact_Kp8", max_iterations=10000),
  runner_cls=WujiOnPolicyRunner,
)

# ----- apple from-scratch ABLATION FAMILY (all default-off switches; baseline =
# apple contact specialist 0/8). Each toggles ONE mechanism to test why apple
# won't lift (escape hatch / floor-farming / exploration deadlock / mass). All
# apple, kp x1, cgsmooth_b2_softclip_contact, launched at 8000 env externally.
_AP_BASE = dict(object_name="apple", reward_mode="cgsmooth_b2_softclip_contact")
_AP_ABLATIONS = {
  "R1": dict(r1_ungated_obj_pos=True),
  "R2": dict(r2_bonus_lift_only=True),
  "R3": dict(r3_guide_goal_gate=0.10),
  "R123": dict(r1_ungated_obj_pos=True, r2_bonus_lift_only=True,
               r3_guide_goal_gate=0.10),
  "MassCur": dict(mass_curriculum=True),
  "RSI": dict(rsi_prob=0.5),
  "ET": dict(et_left_behind=True),
  "Friction": dict(obj_friction=1.0),
  # Noise handled below (ppo init_std, not an env switch).
}
for _x, _kw in _AP_ABLATIONS.items():
  register_mjlab_task(
    task_id=f"WujiHand_Tracking_AppleMulti_CGSmooth_Contact_{_x}",
    env_cfg=wuji_hand_multi_tracking_env_cfg(num_envs=4096, **_AP_BASE, **_kw),
    play_env_cfg=wuji_hand_multi_tracking_env_cfg(play=True, **_AP_BASE, **_kw),
    rl_cfg=wuji_hand_tracking_ppo_runner_cfg(
      run_name=f"Tracking_AppleMulti_CGSmooth_Contact_{_x}", max_iterations=10000),
    runner_cls=WujiOnPolicyRunner,
  )
# Noise: exploration-only variant (init_std 0.5->0.8), env identical to baseline.
register_mjlab_task(
  task_id="WujiHand_Tracking_AppleMulti_CGSmooth_Contact_Noise",
  env_cfg=wuji_hand_multi_tracking_env_cfg(num_envs=4096, **_AP_BASE),
  play_env_cfg=wuji_hand_multi_tracking_env_cfg(play=True, **_AP_BASE),
  rl_cfg=wuji_hand_tracking_ppo_runner_cfg(
    run_name="Tracking_AppleMulti_CGSmooth_Contact_Noise",
    max_iterations=10000, init_std=0.8),
  runner_cls=WujiOnPolicyRunner,
)

# ----- single-sequence specialists (contact-gated, kp x1): cup + apple lift -----
for _oid, _obj, _seq in (
  ("Cup", "cup", "ori_grab_s8_cup_lift"),
  ("Apple", "apple", "ori_grab_s2_apple_lift"),
):
  register_mjlab_task(
    task_id=f"WujiHand_Tracking_{_oid}Single_CGSmooth_Contact",
    env_cfg=wuji_hand_multi_tracking_env_cfg(
      object_name=_obj, num_envs=4096,
      reward_mode="cgsmooth_b2_softclip_contact", only_seq=_seq),
    play_env_cfg=wuji_hand_multi_tracking_env_cfg(
      object_name=_obj, play=True,
      reward_mode="cgsmooth_b2_softclip_contact", only_seq=_seq),
    rl_cfg=wuji_hand_tracking_ppo_runner_cfg(
      run_name=f"Tracking_{_oid}Single_CGSmooth_Contact", max_iterations=10000),
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

# 3-object generalist, contact-gated grasp.
register_mjlab_task(
  task_id="WujiHand_Tracking_3Obj_CGSmooth_Contact",
  env_cfg=wuji_hand_3obj_multi_tracking_env_cfg(
    num_envs=4096, reward_mode="cgsmooth_b2_softclip_contact"),
  play_env_cfg=wuji_hand_3obj_multi_tracking_env_cfg(
    play=True, reward_mode="cgsmooth_b2_softclip_contact"),
  rl_cfg=wuji_hand_tracking_ppo_runner_cfg(
    run_name="Tracking_3Obj_CGSmooth_Contact", max_iterations=10000),
  runner_cls=WujiOnPolicyRunner,
)

# 3-object generalist, PATH-(c) per-world geom_dataid SWAP (single object body,
# mesh swapped per-world instead of 3 bodies + park). Same contact-gated recipe;
# single-object contact buffers. Compare against the park version above.
register_mjlab_task(
  task_id="WujiHand_Tracking_3Obj_CGSmooth_Contact_Swap",
  env_cfg=wuji_hand_3obj_swap_tracking_env_cfg(
    num_envs=4096, reward_mode="cgsmooth_b2_softclip_contact"),
  play_env_cfg=wuji_hand_3obj_swap_tracking_env_cfg(
    play=True, reward_mode="cgsmooth_b2_softclip_contact"),
  rl_cfg=wuji_hand_tracking_ppo_runner_cfg(
    run_name="Tracking_3Obj_CGSmooth_Contact_Swap", max_iterations=10000),
  runner_cls=WujiOnPolicyRunner,
)

# ----- specialist->generalist online (DAgger) distillation -------------------
# Scalable architecture: N single-object sim instances + one shared student (NOT
# multi-object-in-one-sim; park/swap don't scale). The registered env_cfg builds
# ONE single-object env (cubesmall); DistillRunner builds the rest (cup, apple),
# stitches them into a MultiVecEnv, loads the 3 frozen specialists, and adds a BC
# loss to PPO (WUJI_DISTILL_COEF, default 1.0; =0 -> pure multi-instance PPO
# generalist = scalable successor to park 3obj). Teachers routed per-sample by the
# 256-d object latent. See rl/distill_runner.py + rl/multi_vecenv.py.
from wuji_mjlab.rl.distill_runner import DistillRunner  # noqa: E402

register_mjlab_task(
  task_id="WujiHand_Tracking_Distill_3obj",
  env_cfg=wuji_hand_multi_tracking_env_cfg(
    object_name="cubesmall", num_envs=4096,
    reward_mode="cgsmooth_b2_softclip_contact"),
  play_env_cfg=wuji_hand_multi_tracking_env_cfg(
    object_name="cubesmall", play=True,
    reward_mode="cgsmooth_b2_softclip_contact"),
  rl_cfg=wuji_hand_tracking_ppo_runner_cfg(
    run_name="Tracking_Distill_3obj", max_iterations=10000),
  runner_cls=DistillRunner,
)

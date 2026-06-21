# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""RL configuration for the Wuji fly-hand tracking task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def wuji_hand_tracking_ppo_runner_cfg(
  run_name: str = "Tracking",
  max_iterations: int = 10000,
) -> RslRlOnPolicyRunnerCfg:
  # Aligned to DexTrack wuji (HumanoidPPOSupervised.yaml + multi script): net v4,
  # lr 5e-4 adaptive (kl 0.008), horizon 32, mini_epochs 5, critic_coef 4,
  # clip_value, entropy 0. The ONLY intended differences from DexTrack are the
  # reward (deliberate) and the simulator (MuJoCo vs PhysX).
  net_v4 = (8192, 4096, 2048, 1024, 512, 256, 128)
  return RslRlOnPolicyRunnerCfg(
    obs_groups={"actor": ("policy",), "critic": ("critic",)},
    actor=RslRlModelCfg(
      hidden_dims=net_v4,  # DexTrack net_type='v4'
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "SoftplusGaussianDistribution",
        "init_std": 0.5,
        "min_std": 0.2,
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=net_v4,  # rl_games net v4 is a shared trunk; match its capacity
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=4.0,           # DexTrack critic_coef=4
      use_clipped_value_loss=True,   # DexTrack clip_value=True
      clip_param=0.2,                # e_clip=0.2
      entropy_coef=0.0,              # DexTrack entropy_coef=0.0
      num_learning_epochs=5,         # mini_epochs=5
      num_mini_batches=32,           # 40000*32 / 40000 = 32 minibatches
      learning_rate=5.0e-4,          # DexTrack learning_rate=5e-4
      schedule="adaptive",           # lr_schedule=adaptive
      desired_kl=0.008,              # kl_threshold=0.008
      gamma=0.99,
      lam=0.95,                      # tau=0.95
      max_grad_norm=1.0,             # grad_norm=1.0
    ),
    experiment_name="wuji_tracking",
    logger="wandb",
    wandb_project="wuji_tracking_mjlab",
    run_name=run_name,
    save_interval=250,  # net-v4 ckpt ~0.2-1GB; 50 filled the 3.6T disk -> all-jobs-crash
    num_steps_per_env=32,            # horizon_length=32
    max_iterations=max_iterations,
  )

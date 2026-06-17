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
  return RslRlOnPolicyRunnerCfg(
    obs_groups={"actor": ("policy",), "critic": ("critic",)},
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "SoftplusGaussianDistribution",
        "init_std": 0.5,
        "min_std": 0.2,
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=0.5,
      use_clipped_value_loss=False,
      clip_param=0.2,
      entropy_coef=0.001,
      num_learning_epochs=4,
      num_mini_batches=32,
      learning_rate=1.0e-4,
      schedule="fixed",
      gamma=0.99,
      lam=0.95,
      max_grad_norm=1.0,
    ),
    experiment_name="wuji_tracking",
    logger="wandb",
    wandb_project="wuji_tracking_mjlab",
    run_name=run_name,
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=max_iterations,
  )

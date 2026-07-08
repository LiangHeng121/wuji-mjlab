# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""DistillRunner: specialist->generalist online (DAgger) distillation runner.

The registered distill task builds ONE single-object env (the first object) via the
normal train-script path. This runner then builds the remaining single-object envs,
stitches all of them into a MultiVecEnv (one shared student sees every object's
experience), loads the frozen per-object specialist teachers, and injects
(teachers, teacher_latents, distill_coef) into the PPO algorithm so PPO.update adds
a behavior-cloning term. See docs/mjlab_migration_plan.md and rl/multi_vecenv.py.

Scalable by design: N single-object instances (mujoco-warp fast path, high env-cap),
not N objects crammed into one sim. Hundreds of objects = cycle k instances.

distill_coef is read from env var WUJI_DISTILL_COEF (default 1.0) so it can be
swept without re-registering. coef=0 makes this a pure multi-instance PPO generalist
(the scalable successor to the park 3obj env).
"""

from __future__ import annotations

import copy
import glob
import os

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from wuji_mjlab.rl.multi_vecenv import MultiVecEnv
from wuji_mjlab.rl.runner import WujiOnPolicyRunner
from wuji_mjlab.tasks.tracking.config.wuji_hand.env_cfgs import (
  wuji_hand_multi_tracking_env_cfg,
)

# Distill config: object order == teacher order. First object is env #0 (built by
# the train script); the rest are built here. Teacher ckpts glob-resolved (latest).
_OBJECTS = ("cubesmall", "cup", "duck", "elephant", "mouse", "phone", "train",
            "alarmclock", "flute")  # 9obj TOPO scale-up (no apple); coef0 pure PPO
_TEACHER_GLOBS = {
  "cubesmall": "logs/rsl_rl/wuji_tracking/*CubesmallMulti_CGSmooth_Contact*/model_*.pt",
  "cup": "logs/rsl_rl/wuji_tracking/*CupMulti_CGSmooth_Contact*/model_*.pt",
  # apple teacher = the MassCur specialist (3/8 lift), NOT the standard AppleMulti
  # specialist (0/8). The 0/8 teacher's BC loss would fight MassCur's lifting; the
  # 3/8 teacher reinforces it. _latest_ckpt picks the highest-iter MassCur ckpt.
  "apple": "logs/rsl_rl/wuji_tracking/*AppleMulti_CGSmooth_Contact_MassCur_Env8000*/model_*.pt",
}
_REWARD_MODE = "cgsmooth_b2_softclip_contact"


def _latest_ckpt(pattern: str) -> str:
  hits = glob.glob(pattern)
  if not hits:
    raise FileNotFoundError(f"no teacher ckpt for pattern: {pattern}")
  # sort by iter number in model_<N>.pt
  return sorted(hits, key=lambda p: int(p.split("_")[-1][:-3]))[-1]


class DistillRunner(WujiOnPolicyRunner):
  def __init__(self, env, train_cfg: dict, log_dir=None, device: str = "cpu"):
    # `env` is the wrapped single-object env #0 (object _OBJECTS[0]), already built
    # by the train script from the distill task's env_cfg. Match its env count.
    num_envs = env.num_envs

    # construct_algorithm() POPS keys (class_name, ...) out of the cfg it consumes,
    # so snapshot a pristine copy NOW — before super().__init__ mutates train_cfg —
    # to rebuild each teacher runner below.
    pristine_cfg = copy.deepcopy(train_cfg)

    # Build the remaining single-object envs on the same device. MassCur
    # (WUJI_MASS_CUR=1) is applied APPLE-ONLY: only the apple sub-env anneals its
    # mass, so the new-vs-old comparison is attributable to the apple curriculum
    # alone (cube/cup already lift fine and are left untouched).
    mass_cur = os.environ.get("WUJI_MASS_CUR") == "1"
    sub_envs = [env]
    for obj in _OBJECTS[1:]:
      cfg = wuji_hand_multi_tracking_env_cfg(
        object_name=obj, num_envs=num_envs, reward_mode=_REWARD_MODE,
        mass_curriculum=(mass_cur and obj == "apple"))
      cfg.scene.num_envs = num_envs
      base = ManagerBasedRlEnv(cfg=cfg, device=device)
      sub_envs.append(RslRlVecEnvWrapper(base, clip_actions=env.clip_actions))

    multi = MultiVecEnv(sub_envs)

    # Build the shared student on the combined env (this constructs actor/critic/PPO).
    super().__init__(multi, train_cfg, log_dir, device)

    # Load frozen teachers (one per object, order == _OBJECTS) + their object latents.
    distill_coef = float(os.environ.get("WUJI_DISTILL_COEF", "1.0"))
    teachers = []
    latents = []
    # coef0 = pure PPO: BC loss is off, so teachers are never used. Skip loading
    # them entirely -- lets us scale to objects that have NO specialist teacher
    # (_TEACHER_GLOBS/_latest_ckpt would otherwise crash on a missing ckpt).
    if distill_coef > 0.0:
      for k, obj in enumerate(_OBJECTS):
        ckpt = _latest_ckpt(_TEACHER_GLOBS[obj])
        # Build a throwaway runner on this object's sub-env to load the teacher policy.
        tcfg = copy.deepcopy(pristine_cfg)
        # MjlabOnPolicyRunner (not raw OnPolicyRunner) so it strips None cnn_cfg/
        # distribution_cfg from the cfg before MLPModel construction.
        tmp = MjlabOnPolicyRunner(sub_envs[k], tcfg, None, device)
        tmp.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
        pol = tmp.get_inference_policy(device=device)
        for p in pol.parameters():
          p.requires_grad_(False)
        teachers.append(pol)
        # Object latent (constant per single-object env): last 256 dims are the latent.
        cmd = sub_envs[k].unwrapped.command_manager.get_term("motion")
        latents.append(cmd.obj_latent[0].detach().clone())
        print(f"[distill] teacher[{k}] {obj} <- {ckpt}")

    self.alg.teachers = teachers
    self.alg.teacher_latents = torch.stack(latents).to(device) if latents else None
    self.alg.distill_coef = distill_coef
    print(f"[distill] distill_coef={distill_coef}  num_envs/obj={num_envs}  "
          f"objects={_OBJECTS}")

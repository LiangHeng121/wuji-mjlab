# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""MultiVecEnv: present N independent single-object sim instances as ONE VecEnv.

Rationale (see docs/mjlab_migration_plan.md "specialist->generalist 在线蒸馏"):
mujoco-warp cannot host heterogeneous objects in a single compiled model (park/swap
don't scale to hundreds of objects). Instead we run N *pure single-object* sims
(each a fast mujoco-warp instance, high env-cap, no park/swap overhead) and stitch
their experience along the batch dim so ONE shared student network + one PPO+BC
update sees all objects. Object identity reaches the net via the 256-d obj latent
already in the obs; scaling to hundreds of objects = cycle k single-object instances.

Implements the rsl_rl VecEnv contract used by OnPolicyRunner (see
mjlab/rl/vecenv_wrapper.py): num_envs / num_actions / device / max_episode_length /
get_observations()->TensorDict / reset()->(TensorDict, dict) / step(actions).
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


class MultiVecEnv:
  """Concatenate N RslRlVecEnvWrapper instances into one VecEnv.

  All sub-envs must share the same obs/action layout (same hand, same obs terms,
  same action dim) — they differ only in the object. Envs are concatenated in the
  batch dimension in the given order, so ``env_teacher_id[j]`` gives the teacher
  index (== sub-env index) for global env j.
  """

  def __init__(self, envs: list[RslRlVecEnvWrapper]):
    assert len(envs) >= 1
    self.envs = envs
    self.device = envs[0].device
    self.num_actions = envs[0].num_actions
    self.max_episode_length = envs[0].max_episode_length
    self._sizes = [e.num_envs for e in envs]
    self.num_envs = sum(self._sizes)
    # boundaries for slicing a global action tensor back to each sub-env
    self._bounds = []
    o = 0
    for s in self._sizes:
      self._bounds.append((o, o + s))
      o += s
    # per-global-env teacher/object index (contiguous blocks)
    tid = torch.cat([
      torch.full((s,), k, dtype=torch.long, device=self.device)
      for k, s in enumerate(self._sizes)
    ])
    self.env_teacher_id = tid  # (num_envs,)

    # Consistency checks across sub-envs.
    for e in envs[1:]:
      assert e.num_actions == self.num_actions, "sub-envs disagree on num_actions"
      assert e.device == self.device, "sub-envs on different devices"

  # -- VecEnv contract -------------------------------------------------------

  @property
  def cfg(self):
    # rsl_rl reads env.cfg (is_finite_horizon, etc.) and Logger reads env.cfg.
    return self.envs[0].cfg

  @property
  def unwrapped(self):
    return self.envs[0].unwrapped

  @property
  def episode_length_buf(self) -> torch.Tensor:
    return torch.cat([e.episode_length_buf for e in self.envs])

  @episode_length_buf.setter
  def episode_length_buf(self, value: torch.Tensor) -> None:
    for (a, b), e in zip(self._bounds, self.envs):
      e.episode_length_buf = value[a:b]

  def seed(self, seed: int = -1) -> int:
    for i, e in enumerate(self.envs):
      e.seed(seed + i if seed >= 0 else seed)
    return seed

  def _cat_obs(self, obs_list: list[TensorDict]) -> TensorDict:
    keys = list(obs_list[0].keys())
    out = {k: torch.cat([o[k] for o in obs_list], dim=0) for k in keys}
    return TensorDict(out, batch_size=[self.num_envs])

  def get_observations(self) -> TensorDict:
    return self._cat_obs([e.get_observations() for e in self.envs])

  def reset(self) -> tuple[TensorDict, dict]:
    outs = [e.reset() for e in self.envs]
    obs = self._cat_obs([o for o, _ in outs])
    return obs, {}

  def step(self, actions: torch.Tensor):
    obs_list, rew_list, done_list = [], [], []
    extras: dict = {}
    time_outs = []
    logs: list[dict] = []  # per-instance extras["log"] (fair_reward etc.)
    for (a, b), e in zip(self._bounds, self.envs):
      o, r, d, ex = e.step(actions[a:b])
      obs_list.append(o)
      rew_list.append(r)
      done_list.append(d)
      if "time_outs" in ex:
        time_outs.append(ex["time_outs"])
      if isinstance(ex.get("log"), dict):
        logs.append(ex["log"])
    obs = self._cat_obs(obs_list)
    rew = torch.cat(rew_list, dim=0)
    dones = torch.cat(done_list, dim=0)
    if len(time_outs) == len(self.envs):
      extras["time_outs"] = torch.cat(time_outs, dim=0)
    # Merge per-instance logged scalars (e.g. Episode_Reward/*) so the logger
    # sees them. Each single-object env publishes its own metrics; report the
    # mean across instances as the generalist's curve.
    merged: dict = {}
    if logs:
      for key in {k for lg in logs for k in lg}:
        vals = [lg[key] for lg in logs if key in lg]
        merged[key] = sum(vals) / len(vals)
    # fair_reward needs special handling: each sub-env's reward manager writes
    # fair into extras["log"], but ManagerBasedRlEnv._reset_idx WIPES the whole
    # "log" dict on any step that resets ANY env. At high env-counts (staggered
    # resets) that is *every* step, so fair almost never survives into ex["log"]
    # (and the mean-of-present-instances scheme above then drops it entirely).
    # The value persists in env._fair_last (never touched by _reset_idx), so read
    # it directly and publish EVERY step -> stable Episode/fair_reward curve, and
    # guaranteed present in the logger's key-defining ep_extras[0]. See
    # tasks/tracking/mdp/rewards.py:fair_reward_metric.
    fair_vals = [
      fl for e in self.envs
      if (fl := getattr(e.unwrapped, "_fair_last", None)) is not None
    ]
    if fair_vals:
      merged["fair_reward"] = sum(fair_vals) / len(fair_vals)
    if merged:
      extras["log"] = merged
    return obs, rew, dones, extras

  def close(self) -> None:
    for e in self.envs:
      e.close()

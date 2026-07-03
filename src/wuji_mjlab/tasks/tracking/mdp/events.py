# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Event terms for tracking-task ablations (all default-off; registered only in
the corresponding ablation tasks)."""

from __future__ import annotations

import torch


def object_mass_curriculum(
  env,
  env_ids: torch.Tensor | None,
  start_scale: float = 0.3,
  end_frac: float = 0.6,
  total_steps: int = 320_000,
  entity_name: str = "object",
) -> None:
  """MassCur ablation: anneal the object's mass+inertia from start_scale x true
  value to 1.0 x over the first end_frac of training (total_steps = iters*32).

  GLOBAL (same value in every world) -- we write index [0, ...] of the UNEXPANDED
  per-world model fields, which broadcasts to all worlds. Solver invweights are
  written directly with the analytic free-body scaling (invweight ~ 1/mass, ~1/
  inertia): the object is a separate kinematic tree so its invweight is
  independent of the hand -- no global recompute_constants (the swap lesson:
  set_const also rewrites the hand's 26-dof invweight and churns qpos).
  Runs as a "reset" event: a few scalar writes per reset batch, negligible cost.
  """
  st = getattr(env, "_mass_cur", None)
  if st is None:
    mjm = env.sim.mj_model
    body = mjm.body(entity_name + "/obj")
    bid = body.id
    dofadr = int(mjm.jnt_dofadr[int(body.jntadr[0])])  # free joint -> 6 dofs
    dev = env.device
    st = dict(
      bid=bid, dofadr=dofadr, scale=-1.0,
      mass=float(mjm.body_mass[bid]),
      inertia=torch.tensor(mjm.body_inertia[bid], device=dev, dtype=torch.float32),
      subm=float(mjm.body_subtreemass[bid]),
      dofw=torch.tensor(
        mjm.dof_invweight0[dofadr:dofadr + 6], device=dev, dtype=torch.float32),
      bodyw=torch.tensor(
        mjm.body_invweight0[bid], device=dev, dtype=torch.float32),
    )
    env._mass_cur = st

  frac = min(1.0, env.common_step_counter / max(end_frac * total_steps, 1.0))
  k = start_scale + (1.0 - start_scale) * frac
  if abs(k - st["scale"]) < 1e-4:
    return  # unchanged since last reset batch
  st["scale"] = k

  m = env.sim.model
  bid, da = st["bid"], st["dofadr"]
  m.body_mass[0, bid] = st["mass"] * k
  m.body_inertia[0, bid] = st["inertia"] * k
  m.body_subtreemass[0, bid] = st["subm"] * k
  m.dof_invweight0[0, da:da + 6] = st["dofw"] / k
  m.body_invweight0[0, bid] = st["bodyw"] / k

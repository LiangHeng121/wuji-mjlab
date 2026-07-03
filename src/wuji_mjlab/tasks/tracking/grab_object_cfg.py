# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Generic GRAB object importer for mjlab.

Turns a GRAB ``contact_meshes/<name>.ply`` into a free-floating mjlab
``EntityCfg``: a visual mesh geom (full resolution) + collision geom(s).

MuJoCo collides a mesh as its CONVEX HULL, so concave objects (cup, bowl, duck,
flute, ...) are convex-decomposed with CoACD into several convex collision geoms
(otherwise fingers cannot enter cavities). Near-convex objects (cubesmall,
apple, banana) use a single convex hull.

Meshes are scaled x1.25 (GRAB -> sim convention, matching the contact-guidance
pipeline) so the geometry lines up with the reference object pose.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from mjlab.entity import EntityCfg

GRAB_MESH_DIR = Path(
  "/data/home/liangheng/DexTrack/GRAB/unzipped/tools/object_meshes/contact_meshes"
)
GRAB_SCALE = 1.25
# DexTrack/Isaac Gym used a uniform rigid_obj_density=500 for ALL objects (not
# real per-object masses). Match it: the reference trajectories were generated
# under uniform-density physics, so consistency beats physical accuracy here.
DEFAULT_DENSITY = 500.0

# Hand contact params win (priority=1 on the hand); object is priority 0.
_OBJ_FRICTION = (0.3, 0.005, 0.0001)
_OBJ_SOLREF = (0.015, 1.0)
_OBJ_SOLIMP = (0.85, 0.85, 0.001, 0.5, 2.0)  # MjSpec wants full 5-vector


def _add_inline_mesh(spec: mujoco.MjSpec, name: str, tm: trimesh.Trimesh) -> None:
  mesh = spec.add_mesh()
  mesh.name = name
  mesh.uservert = tm.vertices.astype(float).flatten().tolist()
  mesh.userface = tm.faces.astype(int).flatten().tolist()


def _convex_parts(
  tm: trimesh.Trimesh, threshold: float, max_hulls: int
) -> list[trimesh.Trimesh]:
  """Convex collision parts: hull if convex, else CoACD decomposition."""
  if tm.is_convex:
    return [tm.convex_hull]
  try:
    import coacd

    cmesh = coacd.Mesh(tm.vertices, tm.faces)
    result = coacd.run_coacd(cmesh, threshold=threshold, max_convex_hull=max_hulls)
    parts = [
      trimesh.Trimesh(vertices=np.asarray(v), faces=np.asarray(f)).convex_hull
      for (v, f) in result
    ]
    return parts or [tm.convex_hull]
  except Exception as e:  # noqa: BLE001
    print(f"[grab_object] CoACD unavailable/failed ({e}); using single convex hull")
    return [tm.convex_hull]


def _build_spec(
  ply: Path,
  scale: float,
  density: float,
  rgba: tuple[float, float, float, float],
  coacd_threshold: float,
  max_hulls: int,
  friction: float | None = None,
  convex_hull: bool = False,
) -> mujoco.MjSpec:
  fric = (friction, _OBJ_FRICTION[1], _OBJ_FRICTION[2]) if friction is not None else _OBJ_FRICTION
  tm = trimesh.load(str(ply), force="mesh")
  tm.apply_scale(scale)

  spec = mujoco.MjSpec()
  _add_inline_mesh(spec, "obj_visual", tm)
  # convex_hull=True -> ONE convex hull (collision). CoACD's thin/degenerate hulls
  # for deep-concave objects (cup) create near-singular contacts -> velocity blows
  # up to 1e4+ -> NaN. A single hull is rock-stable; fine for grasp-from-outside
  # objects (cup/apple held by the body/rim, not the cavity).
  parts = [tm.convex_hull] if convex_hull else _convex_parts(tm, coacd_threshold, max_hulls)
  for i, p in enumerate(parts):
    _add_inline_mesh(spec, f"obj_col_{i}", p)

  body = spec.worldbody.add_body()
  body.name = "obj"
  body.add_freejoint()

  # All geoms are massless; the body's mass + inertia are set EXPLICITLY from
  # the true (non-convex) mesh below. This avoids two MuJoCo pitfalls:
  #  - per-part density double-counts mass where CoACD pieces overlap, and
  #  - a single mesh geom's auto-mass uses the CONVEX HULL volume (way too
  #    heavy for hollow objects like a cup).
  gv = body.add_geom()
  gv.type = mujoco.mjtGeom.mjGEOM_MESH
  gv.meshname = "obj_visual"
  gv.group = 2
  gv.rgba = list(rgba)
  gv.contype = 0
  gv.conaffinity = 0
  gv.density = 0.0

  for i in range(len(parts)):
    gc = body.add_geom()
    gc.type = mujoco.mjtGeom.mjGEOM_MESH
    gc.meshname = f"obj_col_{i}"
    gc.group = 3
    gc.density = 0.0
    gc.friction = list(fric)
    gc.solref = list(_OBJ_SOLREF)
    gc.solimp = list(_OBJ_SOLIMP)
    gc.condim = 3
    gc.priority = 0

  _set_true_inertial(body, tm, density)
  return spec


def _set_true_inertial(body, tm: trimesh.Trimesh, density: float) -> None:
  """Set body mass + inertia from the true mesh volume (not the convex hull)."""
  mass = float(tm.volume * density)
  com = np.asarray(tm.center_mass, dtype=float)
  inertia = np.asarray(tm.moment_inertia, dtype=float) * density  # about COM
  evals, evecs = np.linalg.eigh(inertia)
  if np.linalg.det(evecs) < 0:  # ensure a proper rotation
    evecs[:, 0] *= -1.0
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, evecs.flatten())
  body.mass = mass
  body.ipos = com.tolist()
  body.iquat = quat.tolist()
  body.inertia = np.clip(evals, 1e-9, None).tolist()
  body.explicitinertial = True


def _obj_phys(name: str, scale: float, density: float) -> dict:
  """Per-object physical params for a swap slot: compile a 1-body-1-hull reference
  model (with the SAME explicit true-mesh inertial used in the swap body) and read
  the collision geom's rbound/aabb + body inertial + the solver invweight constants
  (dof_invweight0 / body_invweight0 / body_subtreemass). All written per-world for
  the active object, so broadphase, dynamics AND solver impedance match the swapped
  mesh -- WITHOUT a global recompute_constants (which would clobber the hand's dofs,
  since the object is a separate kinematic tree -> block-diagonal mass matrix ->
  the object's invweight equals this single-object reference's)."""
  tm = trimesh.load(str(GRAB_MESH_DIR / f"{name}.ply"), force="mesh")
  tm.apply_scale(scale)
  hull = tm.convex_hull  # cube is_convex, cup/apple forced single hull -> 1 hull each
  mass = float(tm.volume * density)
  com = np.asarray(tm.center_mass, dtype=float)
  inertia = np.asarray(tm.moment_inertia, dtype=float) * density
  evals, evecs = np.linalg.eigh(inertia)
  if np.linalg.det(evecs) < 0:
    evecs[:, 0] *= -1.0
  iquat = np.zeros(4)
  mujoco.mju_mat2Quat(iquat, evecs.flatten())
  idiag = np.clip(evals, 1e-9, None)

  spec = mujoco.MjSpec()
  _add_inline_mesh(spec, "h", hull)
  b = spec.worldbody.add_body()
  b.add_freejoint()
  # explicit true-mesh inertial = exactly what the swap body gets per-world, so the
  # compile-time invweight below is consistent with the per-world mass/inertia write.
  b.mass = mass
  b.ipos = com.tolist()
  b.iquat = iquat.tolist()
  b.inertia = idiag.tolist()
  b.explicitinertial = True
  g = b.add_geom()
  g.type = mujoco.mjtGeom.mjGEOM_MESH
  g.meshname = "h"
  m = spec.compile()  # body 1 = object, geom 0 = hull, dofs 0:6 = free joint
  return dict(
    rbound=float(m.geom_rbound[0]),
    aabb=np.array(m.geom_aabb[0], dtype=np.float32).reshape(2, 3).copy(),
    mass=mass,
    idiag=idiag.astype(np.float32),
    com=com.astype(np.float32),
    iquat=iquat.astype(np.float32),
    dof_invweight0=np.array(m.dof_invweight0[:6], dtype=np.float32).copy(),  # (6,)
    body_invweight0=np.array(m.body_invweight0[1], dtype=np.float32).copy(),  # (2,)
    subtreemass=float(m.body_subtreemass[1]),
  )


def _build_swap_spec(
  names: tuple[str, ...], scale: float, density: float,
  rgba: tuple[float, float, float, float], friction: float | None,
) -> mujoco.MjSpec:
  """ONE free body with a single visual + single collision geom, but a mesh pool
  holding every object's visual + convex-hull mesh. Per-world ``geom_dataid``
  selects which object each env collides (path-(c) swap, replaces park). All
  objects are topology-identical (1 visual + 1 hull), required for same-model
  swap. Body inertial is overwritten per-world at reset."""
  fric = (friction, _OBJ_FRICTION[1], _OBJ_FRICTION[2]) if friction is not None else _OBJ_FRICTION
  spec = mujoco.MjSpec()
  for n in names:
    tm = trimesh.load(str(GRAB_MESH_DIR / f"{n}.ply"), force="mesh")
    tm.apply_scale(scale)
    _add_inline_mesh(spec, f"{n}_visual", tm)
    _add_inline_mesh(spec, f"{n}_col", tm.convex_hull)
  body = spec.worldbody.add_body()
  body.name = "obj"
  body.add_freejoint()
  gv = body.add_geom()
  gv.name = "obj_visual"
  gv.type = mujoco.mjtGeom.mjGEOM_MESH
  gv.meshname = f"{names[0]}_visual"
  gv.group = 2
  gv.rgba = list(rgba)
  gv.contype = 0
  gv.conaffinity = 0
  gv.density = 0.0
  gc = body.add_geom()
  gc.name = "obj_col"
  gc.type = mujoco.mjtGeom.mjGEOM_MESH
  gc.meshname = f"{names[0]}_col"
  gc.group = 3
  gc.density = 0.0
  gc.friction = list(fric)
  gc.solref = list(_OBJ_SOLREF)
  gc.solimp = list(_OBJ_SOLIMP)
  gc.condim = 3
  gc.priority = 0
  # explicit inertial = first object (overwritten per-world at reset).
  tm0 = trimesh.load(str(GRAB_MESH_DIR / f"{names[0]}.ply"), force="mesh")
  tm0.apply_scale(scale)
  _set_true_inertial(body, tm0, density)
  return spec


def get_grab_multiobj_swap_cfg(
  names: tuple[str, ...],
  scale: float = GRAB_SCALE,
  density: float = DEFAULT_DENSITY,
  rgba: tuple[float, float, float, float] = (0.85, 0.3, 0.2, 1.0),
  init_pos: tuple[float, float, float] = (0.0, 0.0, 0.1),
  friction: float | None = None,
) -> tuple[EntityCfg, dict]:
  """Single-body multi-mesh object entity (path-(c) geom_dataid swap) + the
  per-object physical params table the command writes per-world. Returns
  (EntityCfg, {name: {rbound, aabb, mass, idiag, com, iquat}})."""
  for n in names:
    if not (GRAB_MESH_DIR / f"{n}.ply").exists():
      raise FileNotFoundError(f"GRAB mesh not found: {GRAB_MESH_DIR / f'{n}.ply'}")
  params = {n: _obj_phys(n, scale, density) for n in names}
  cfg = EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=init_pos, rot=(1.0, 0.0, 0.0, 0.0)),
    spec_fn=partial(_build_swap_spec, tuple(names), scale, density, rgba, friction),
  )
  return cfg, params


def get_grab_object_cfg(
  name: str,
  scale: float = GRAB_SCALE,
  density: float = DEFAULT_DENSITY,
  rgba: tuple[float, float, float, float] = (0.85, 0.3, 0.2, 1.0),
  init_pos: tuple[float, float, float] = (0.0, 0.0, 0.1),
  coacd_threshold: float = 0.05,
  max_hulls: int = 24,
  friction: float | None = None,
  convex_hull: bool = False,
) -> EntityCfg:
  """Build a free-floating EntityCfg for GRAB object ``name`` (e.g. 'cubesmall').
  convex_hull=True forces a single convex collision hull (stable for deep-concave
  objects like cup whose CoACD hulls blow up; see _build_spec)."""
  ply = GRAB_MESH_DIR / f"{name}.ply"
  if not ply.exists():
    raise FileNotFoundError(f"GRAB mesh not found: {ply}")
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=init_pos, rot=(1.0, 0.0, 0.0, 0.0)),
    spec_fn=partial(
      _build_spec, ply, scale, density, rgba, coacd_threshold, max_hulls, friction,
      convex_hull,
    ),
  )

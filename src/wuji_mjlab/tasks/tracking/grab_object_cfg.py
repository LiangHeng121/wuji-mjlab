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
  "/home/liangh/DexTrack/GRAB/unzipped/tools/object_meshes/contact_meshes"
)
GRAB_SCALE = 1.25
# GRAB cube density back-calculated from cube.xml baseline (~683 kg/m^3); a
# reasonable default for the light plastic GRAB props.
DEFAULT_DENSITY = 683.0

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
) -> mujoco.MjSpec:
  tm = trimesh.load(str(ply), force="mesh")
  tm.apply_scale(scale)

  spec = mujoco.MjSpec()
  _add_inline_mesh(spec, "obj_visual", tm)
  parts = _convex_parts(tm, coacd_threshold, max_hulls)
  for i, p in enumerate(parts):
    _add_inline_mesh(spec, f"obj_col_{i}", p)

  body = spec.worldbody.add_body()
  body.name = "obj"
  body.add_freejoint()

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
    gc.density = density
    gc.friction = list(_OBJ_FRICTION)
    gc.solref = list(_OBJ_SOLREF)
    gc.solimp = list(_OBJ_SOLIMP)
    gc.condim = 3
    gc.priority = 0

  return spec


def get_grab_object_cfg(
  name: str,
  scale: float = GRAB_SCALE,
  density: float = DEFAULT_DENSITY,
  rgba: tuple[float, float, float, float] = (0.85, 0.3, 0.2, 1.0),
  init_pos: tuple[float, float, float] = (0.0, 0.0, 0.1),
  coacd_threshold: float = 0.05,
  max_hulls: int = 24,
) -> EntityCfg:
  """Build a free-floating EntityCfg for GRAB object ``name`` (e.g. 'cubesmall')."""
  ply = GRAB_MESH_DIR / f"{name}.ply"
  if not ply.exists():
    raise FileNotFoundError(f"GRAB mesh not found: {ply}")
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=init_pos, rot=(1.0, 0.0, 0.0, 0.0)),
    spec_fn=partial(
      _build_spec, ply, scale, density, rgba, coacd_threshold, max_hulls
    ),
  )

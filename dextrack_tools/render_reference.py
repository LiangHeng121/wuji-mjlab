"""Kinematic replay of the REFERENCE trajectory: each frame teleport hand qpos +
object pose to the retargeted reference (no policy, no grasp physics) and render.
Shows what the GRAB-retargeted motion the policy is imitating actually looks like.
Usage: render_reference.py <TASK> <seq_name> <out.mp4>
"""
from __future__ import annotations

import sys

import imageio
import numpy as np
import torch

import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

TASK, SEQ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]

env_cfg, _ = prepare_task_cfgs(TASK, [], play=True)
env_cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array")
cmd = env.command_manager.get_term("motion")
robot = env.scene[cmd.cfg.hand_entity_name]
NAMES = [p.split("info_")[1].split("_nf_300")[0] for p in cmd.cfg.motion_files]
sidx = NAMES.index(SEQ)
ids = torch.tensor([0], device="cuda:0")
env.reset()
if hasattr(cmd, "env_seq"):
  cmd.env_seq[:] = sidx
  if hasattr(cmd, "env_obj"):
    cmd.env_obj[:] = cmd.seq_obj[sidx]

objs = cmd.objs if hasattr(cmd, "objs") else [env.scene["object"]]
active_slot = int(cmd.seq_obj[sidx]) if hasattr(cmd, "seq_obj") else 0
T = cmd.time_step_total
frames, oz = [], []
for t in range(T):
  # hand -> reference qpos
  jp = cmd._ref_qpos[sidx, t].unsqueeze(0)
  robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
  # active object -> reference pose; park others far
  for j, o in enumerate(objs):
    if j == active_slot:
      root = torch.cat([cmd._ref_obj_pos[sidx, t], cmd._ref_obj_quat[sidx, t],
                        torch.zeros(6, device="cuda:0")]).unsqueeze(0)
    else:
      root = torch.cat([cmd._park[j], torch.zeros(6, device="cuda:0")]).unsqueeze(0)
    o.write_root_state_to_sim(root, env_ids=ids)
  env.sim.forward()
  img = env.render()
  if img is not None:
    frames.append(np.asarray(img))
  oz.append(float(cmd._ref_obj_pos[sidx, t, 2]))

imageio.mimwrite(OUT, frames, fps=30, quality=8)
oz = np.array(oz)
print(f"WROTE {OUT} | seq={SEQ} 参考物体z: 起{oz[0]:.2f} 峰{oz.max():.2f} 末{oz[-1]:.2f} "
      f"举升帧(z>0.15)={int((oz>0.15).sum())}/{T}")

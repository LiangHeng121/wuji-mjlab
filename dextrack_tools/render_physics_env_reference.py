"""Physics reference render using the ACTUAL training env.

Steps the real WujiHand_Tracking env with ZERO policy residual (so the action
target = reference q26[t], pure kinematics bias) under full MuJoCo physics with
the properly contact-tuned cube. Shows whether the open-loop reference grasps
and carries the object in the exact physics the policy will train in.
"""

import os
import numpy as np
import torch
import imageio

from mjlab.envs import ManagerBasedRlEnv
from wuji_mjlab.tasks.tracking.config.wuji_hand.env_cfgs import (
    wuji_hand_cubesmall_tracking_env_cfg,
)

OUT = "/home/liangh/DexTrack/mjlab_fly_cubesmall_PHYSICS_reference.mp4"

cfg = wuji_hand_cubesmall_tracking_env_cfg(play=True)
cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
adim = env.action_manager.total_action_dim

obs, _ = env.reset()
obj = env.scene["object"]
cmd = env.command_manager.get_term("motion")

frames = []
obj_z = []
ref_z = []
for i in range(300):
    act = torch.zeros((env.num_envs, adim), device=env.device)  # zero residual
    env.step(act)
    img = env.render()
    if img is not None:
        frames.append(np.asarray(img))
    obj_z.append(float(obj.data.root_link_pos_w[0, 2]))
    ref_z.append(float(cmd.ref_obj_pos[0, 2]))

if frames:
    imageio.mimwrite(OUT, frames, fps=30, quality=8)
    print("WROTE", OUT, "frames", len(frames))
else:
    print("render() returned None -- no frames")

oz = np.array(obj_z)
rz = np.array(ref_z)
print(f"object z achieved: start {oz[0]:.3f} max {oz.max():.3f} end {oz[-1]:.3f}")
print(f"object z reference: start {rz[0]:.3f} max {rz.max():.3f} end {rz[-1]:.3f}")
err = np.abs(oz - rz)
print(f"obj-z tracking err: mean {err.mean():.3f} max {err.max():.3f}")
print("(reference grasp HOLDS under physics)" if oz.max() > 0.15 and err.mean() < 0.1
      else "(open-loop reference does NOT fully hold -- expected; RL should fix)")

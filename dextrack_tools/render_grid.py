"""Tile ONE policy's rollouts on several fixed sequences into a single grid video.

Default: CGSmooth on its top-2 bonus sequences per task type (lift/inspect/pass),
6 sequences -> 3 rows (task type) x 2 cols, one combined mp4.
"""
from __future__ import annotations

import glob
import types
from dataclasses import asdict

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from wuji_mjlab.tasks.tracking.config.wuji_hand.env_cfgs import _cubesmall_sequences
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

TASK = "WujiHand_Tracking_CubesmallMulti_CGSmooth"
OUT = "/home/liangh/DexTrack/mjlab_cgsmooth_grid.mp4"
# 2 rows x 3 cols (landscape): cols = lift/inspect/pass, rows = that type's #1/#2.
ROWS = [
  [("ori_grab_s2_cubesmall_lift", "s2 lift"),
   ("ori_grab_s8_cubesmall_inspect_1", "s8 inspect"),
   ("ori_grab_s1_cubesmall_pass_1", "s1 pass")],
  [("ori_grab_s8_cubesmall_lift", "s8 lift"),
   ("ori_grab_s2_cubesmall_inspect_1", "s2 inspect"),
   ("ori_grab_s6_cubesmall_pass_1", "s6 pass")],
]
NAMES = _cubesmall_sequences()


def locked(idx):
  def _resample(self, env_ids):
    self.time_steps[env_ids] = 0
    self.env_seq[env_ids] = idx
    s, t = self.env_seq[env_ids], self.time_steps[env_ids]
    jp = torch.clip(self._ref_qpos[s, t].clone(),
                    self.robot.data.soft_joint_pos_limits[env_ids][:, :, 0],
                    self.robot.data.soft_joint_pos_limits[env_ids][:, :, 1])
    self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)
    root = torch.cat([self._ref_obj_pos[s, t], self._ref_obj_quat[s, t],
                      torch.zeros(len(env_ids), 6, device=self.device)], dim=-1)
    self.obj.write_root_state_to_sim(root, env_ids=env_ids)
    self.obj.reset(env_ids=env_ids)
  return _resample


def lab(frame, text):
  img = Image.fromarray(frame)
  d = ImageDraw.Draw(img)
  d.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
  d.text((6, 4), text, fill=(255, 255, 255))
  return np.asarray(img)


ckpt = sorted(glob.glob(f"logs/rsl_rl/wuji_tracking/*CubesmallMulti_CGSmooth/model_*.pt"),
              key=lambda x: int(x.split("_")[-1][:-3]))[-1]
print("CKPT", ckpt)
env_cfg, agent_cfg = prepare_task_cfgs(TASK, [], play=True)
env_cfg.scene.num_envs = 1
base = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array")
env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
r = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cuda:0")
r.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
pol = r.get_inference_policy(device="cuda:0")
cmd = base.command_manager.get_term("motion")

panel = {}  # (ri,ci) -> frames
for ri, row in enumerate(ROWS):
  for ci, (seq, short) in enumerate(row):
    cmd._resample_command = types.MethodType(locked(NAMES.index(seq)), cmd)
    base.reset()
    obs = env.get_observations()
    fr = []
    for _ in range(300):
      obs, _, _, _ = env.step(pol(obs))
      img = base.render()
      if img is not None:
        fr.append(lab(np.asarray(img), f"CGSmooth | {short}"))
    panel[(ri, ci)] = fr
    print(f"rendered {seq} ({len(fr)} frames)")

n = min(len(v) for v in panel.values())
grid = []
for i in range(n):
  rows_img = [np.concatenate([panel[(ri, ci)][i] for ci in range(len(ROWS[ri]))], axis=1)
              for ri in range(len(ROWS))]
  grid.append(np.concatenate(rows_img, axis=0))
imageio.mimwrite(OUT, grid, fps=30, quality=8)
print(f"WROTE {OUT} ({n} frames, 2x3 grid: rows=top1/top2, cols=lift/inspect/pass)")

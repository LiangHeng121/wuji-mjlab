"""Side-by-side comparison: same sequence, 3 reward policies (Original|Pinall3|
CGSmooth) horizontally concatenated into one video per sequence, with labels.

Each policy is locked to the SAME sequence (monkeypatched _resample_command), so
the panels differ only by the trained reward. Usage:
  render_compare.py <out_prefix> [seq_name1 seq_name2 ...]
"""
from __future__ import annotations

import glob
import sys
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

OUT_PREFIX = sys.argv[1]
SEQ_NAMES = sys.argv[2:] or [
  "ori_grab_s10_cubesmall_inspect_1",  # high lift (~90cm)
  "ori_grab_s3_cubesmall_inspect_1",   # mid
  "ori_grab_s5_cubesmall_pass_1",      # pass
]
POLICIES = [
  ("Original", "WujiHand_Tracking_CubesmallMulti_Original"),
  ("Pinall3", "WujiHand_Tracking_CubesmallMulti_Pinall3"),
  ("CGSmooth", "WujiHand_Tracking_CubesmallMulti_CGSmooth"),
]
NAMES = _cubesmall_sequences()
SEQ_IDX = {n: NAMES.index(n) for n in SEQ_NAMES}


def make_locked_resample(idx):
  def _resample(self, env_ids):
    self.time_steps[env_ids] = 0
    self.env_seq[env_ids] = idx
    s, t = self.env_seq[env_ids], self.time_steps[env_ids]
    jp = self._ref_qpos[s, t].clone()
    soft = self.robot.data.soft_joint_pos_limits[env_ids]
    jp = torch.clip(jp, soft[:, :, 0], soft[:, :, 1])
    self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)
    pos, quat = self._ref_obj_pos[s, t], self._ref_obj_quat[s, t]
    root = torch.cat([pos, quat, torch.zeros(len(env_ids), 6, device=self.device)], dim=-1)
    self.obj.write_root_state_to_sim(root, env_ids=env_ids)
    self.obj.reset(env_ids=env_ids)
  return _resample


def label(frame, text):
  img = Image.fromarray(frame)
  d = ImageDraw.Draw(img)
  d.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
  d.text((6, 4), text, fill=(255, 255, 255))
  return np.asarray(img)


# frames[label][seq_name] = list of (H,W,3)
frames = {lbl: {} for lbl, _ in POLICIES}
for lbl, task in POLICIES:
  ckpt = sorted(glob.glob(f"logs/rsl_rl/wuji_tracking/*CubesmallMulti_{lbl}/model_*.pt"),
                key=lambda x: int(x.split("_")[-1][:-3]))[-1]
  print(f"[{lbl}] {ckpt}")
  env_cfg, agent_cfg = prepare_task_cfgs(task, [], play=True)
  env_cfg.scene.num_envs = 1
  base = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array")
  env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
  r = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cuda:0")
  r.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  pol = r.get_inference_policy(device="cuda:0")
  cmd = base.command_manager.get_term("motion")
  for nm, idx in SEQ_IDX.items():
    cmd._resample_command = types.MethodType(make_locked_resample(idx), cmd)
    base.reset()
    obs = env.get_observations()
    fr = []
    for _ in range(300):
      obs, _, _, _ = env.step(pol(obs))
      img = base.render()
      if img is not None:
        fr.append(label(np.asarray(img), lbl))
    frames[lbl][nm] = fr
  del env, base, r
  torch.cuda.empty_cache()

for nm in SEQ_NAMES:
  trio = [frames[lbl][nm] for lbl, _ in POLICIES]
  n = min(len(t) for t in trio)
  combined = [np.concatenate([trio[0][i], trio[1][i], trio[2][i]], axis=1) for i in range(n)]
  out = f"{OUT_PREFIX}_{nm}.mp4"
  imageio.mimwrite(out, combined, fps=30, quality=8)
  print(f"WROTE {out} ({n} frames, 3-up: Original|Pinall3|CGSmooth)")

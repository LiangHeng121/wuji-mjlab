"""Eval ALL sequences of a single-object task by episode bonus, then render the
top-N (best-performing) sequences. Usage:
  eval_render_top.py <TASK> <object> <ckpt_glob_tag> <out_prefix> [top_n]
"""
from __future__ import annotations

import glob
import sys
import types
from dataclasses import asdict

import imageio
import numpy as np
import torch

import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from wuji_mjlab.tasks.tracking.config.wuji_hand.env_cfgs import _object_sequences
from wuji_mjlab.tasks.tracking.mdp import rewards as R
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

TASK, OBJ, TAG, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
TOPN = int(sys.argv[5]) if len(sys.argv) > 5 else 6
SEQS = _object_sequences(OBJ)
wd = sorted(glob.glob(f"logs/rsl_rl/wuji_tracking/*{TAG}_CGSmooth/model_*.pt"),
            key=lambda x: int(x.split("_")[-1][:-3]))[-1]
print("CKPT", wd)

env_cfg, agent_cfg = prepare_task_cfgs(TASK, [], play=True)
env_cfg.scene.num_envs = 1
base = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array")
env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
r = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cuda:0")
r.load(wd, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
pol = r.get_inference_policy(device="cuda:0")
cmd = base.command_manager.get_term("motion")
NAMES = [n for n in
         [p.split("info_")[1].split("_nf_300")[0] for p in cmd.cfg.motion_files]]


def lock(idx):
  def _r(self, ids):
    self.time_steps[ids] = 0
    self.env_seq[ids] = idx
    s, t = self.env_seq[ids], self.time_steps[ids]
    jp = torch.clip(self._ref_qpos[s, t].clone(),
                    self.robot.data.soft_joint_pos_limits[ids][:, :, 0],
                    self.robot.data.soft_joint_pos_limits[ids][:, :, 1])
    self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
    self.robot.reset(env_ids=ids)
    self.obj.write_root_state_to_sim(
      torch.cat([self._ref_obj_pos[s, t], self._ref_obj_quat[s, t],
                 torch.zeros(len(ids), 6, device=self.device)], dim=-1), env_ids=ids)
    self.obj.reset(env_ids=ids)
  return _r


# pass 1: bonus per sequence (no frames)
scored = []
for nm in SEQS:
  cmd._resample_command = types.MethodType(lock(NAMES.index(nm)), cmd)
  base.reset(); obs = env.get_observations(); bonus = 0.0; mz = 0.0
  for _ in range(300):
    obs, _, _, _ = env.step(pol(obs))
    bonus += float(R.object_inplace_bonus(base, "motion", grip_thres=0.22, n_finger_sum=5)[0])
    mz = max(mz, float(cmd.obj_pos[0, 2]))
  scored.append((nm, bonus, mz))
scored.sort(key=lambda x: -x[1])
print("RANK (bonus/300, max_z):")
for nm, b, mz in scored:
  print(f"  {nm:<34} bonus={b:6.1f}({b/3:3.0f}%) max_z={mz:.2f}")

# pass 2: render top-N
for nm, b, mz in scored[:TOPN]:
  cmd._resample_command = types.MethodType(lock(NAMES.index(nm)), cmd)
  base.reset(); obs = env.get_observations(); frames = []
  for _ in range(300):
    obs, _, _, _ = env.step(pol(obs))
    img = base.render()
    if img is not None:
      frames.append(np.asarray(img))
  out = f"{OUT}_{nm}.mp4"
  imageio.mimwrite(out, frames, fps=30, quality=8)
  print(f"WROTE {out} bonus={b/3:.0f}% max_z={mz:.2f}")

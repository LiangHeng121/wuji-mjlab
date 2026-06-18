"""Render a multi-seq generalist policy on SEVERAL fixed sequences (one video each).

The multi-seq command assigns env_seq randomly; here we monkeypatch its
_resample_command to LOCK a chosen sequence index, so each render shows the same
policy on a specific cubesmall sequence. Usage:
  render_multi_seqs.py <TASK> "<ckpt glob>" <out_prefix> [n_seqs]
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
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

TASK = sys.argv[1]
PAT = sys.argv[2]
OUT_PREFIX = sys.argv[3]
N_SEQS = int(sys.argv[4]) if len(sys.argv) > 4 else 6

wd = sorted(glob.glob(PAT), key=lambda x: int(x.split("_")[-1][:-3]))[-1]
print("CKPT", wd)

env_cfg, agent_cfg = prepare_task_cfgs(TASK, [], play=True)
env_cfg.scene.num_envs = 1
base = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array")
env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
r = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cuda:0")
r.load(wd, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
pol = r.get_inference_policy(device="cuda:0")

cmd = base.command_manager.get_term("motion")
obj = base.scene["object"]


def make_locked_resample(idx: int):
  def _resample(self, env_ids):
    self.time_steps[env_ids] = 0
    self.env_seq[env_ids] = idx  # LOCKED (no randint)
    s = self.env_seq[env_ids]
    t = self.time_steps[env_ids]
    jp = self._ref_qpos[s, t].clone()
    soft = self.robot.data.soft_joint_pos_limits[env_ids]
    jp = torch.clip(jp, soft[:, :, 0], soft[:, :, 1])
    self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)
    pos = self._ref_obj_pos[s, t]
    quat = self._ref_obj_quat[s, t]
    root = torch.cat(
      [pos, quat, torch.zeros(len(env_ids), 6, device=self.device)], dim=-1
    )
    self.obj.write_root_state_to_sim(root, env_ids=env_ids)
    self.obj.reset(env_ids=env_ids)
  return _resample


# Pick spread-out sequence indices for variety.
nseq = cmd.num_seqs
idxs = sorted(set(int(round(i)) for i in np.linspace(0, nseq - 1, N_SEQS)))
from wuji_mjlab.tasks.tracking.config.wuji_hand.env_cfgs import _cubesmall_sequences
names = _cubesmall_sequences()

for k in idxs:
  cmd._resample_command = types.MethodType(make_locked_resample(k), cmd)
  obs, _ = base.reset()
  obs = env.get_observations()
  frames, oz, rz = [], [], []
  for i in range(300):
    obs, _, _, _ = env.step(pol(obs))
    img = base.render()
    if img is not None:
      frames.append(np.asarray(img))
    oz.append(float(obj.data.root_link_pos_w[0, 2]))
    rz.append(float(cmd.ref_obj_pos[0, 2]))
  oz, rz = np.array(oz), np.array(rz)
  nm = names[k] if k < len(names) else f"seq{k}"
  out = f"{OUT_PREFIX}_{nm}.mp4"
  imageio.mimwrite(out, frames, fps=30, quality=8)
  print(f"WROTE {out} | seq={nm} max_z={oz.max():.3f} ref_max={rz.max():.3f} "
        f"z_err={np.mean(np.abs(oz - rz)):.3f}")

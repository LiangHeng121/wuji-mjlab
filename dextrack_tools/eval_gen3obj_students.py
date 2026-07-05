"""Eval a MultiVecEnv generalist student (Gen3obj coef=0 arms, or distill coef=1)
on all 22 lift sequences, per single-object env (the student sees single-object obs).

Usage: eval_gen3obj_students.py <ckpt> <data_ver> <label>
  data_ver = GRAB_Tracking_PK_WUJI_FPOS_v1 | GRAB_Tracking_PK_WUJI_TOPO_v1
Success = max_z >= ref_peak - 0.05.  cwd = wuji-mjlab, cuda:0 via CVD.
"""
from __future__ import annotations
import os, sys
CKPT, DATA_VER, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ["WUJI_DATA_VER"] = DATA_VER  # MUST precede wuji_mjlab.tasks import

import types
from dataclasses import asdict
import torch

import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

# object -> (single-object task, lift seqs)
OBJS = [
  ("cube", "WujiHand_Tracking_CubesmallMulti_CGSmooth_Contact",
   [f"ori_grab_s{i}_cubesmall_lift" for i in (10, 1, 2, 5, 6, 8)]),
  ("cup", "WujiHand_Tracking_CupMulti_CGSmooth_Contact",
   [f"ori_grab_s{i}_cup_lift" for i in (10, 1, 3, 4, 5, 6, 8, 9)]),
  ("apple", "WujiHand_Tracking_AppleMulti_CGSmooth_Contact",
   [f"ori_grab_s{i}_apple_lift" for i in (1, 2, 3, 4, 6, 7, 8, 9)]),
]


def make_locked(lock_idx):  # single-object env: pin each env to its own seq
  def _resample(self, env_ids):
    self.time_steps[env_ids] = 0
    self.env_seq[env_ids] = lock_idx[env_ids]
    self.env_obj[env_ids] = self.seq_obj[self.env_seq[env_ids]]
    s, t = self.env_seq[env_ids], self.time_steps[env_ids]
    soft = self.robot.data.soft_joint_pos_limits[env_ids]
    jp = torch.clip(self._ref_qpos[s, t].clone(), soft[:, :, 0], soft[:, :, 1])
    self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)
    ref_pos, ref_quat = self._ref_obj_pos[s, t], self._ref_obj_quat[s, t]
    vel = torch.zeros(len(env_ids), 6, device=self.device)
    self.obj.write_root_state_to_sim(
      torch.cat([ref_pos, ref_quat, vel], dim=-1), env_ids=env_ids)
    self.obj.reset(env_ids=env_ids)
  return _resample


print(f"[{LABEL}] ckpt={CKPT}  data={DATA_VER}", flush=True)
rows, tot_ok, tot_n = [], 0, 0
for oname, task, seqs in OBJS:
  env_cfg, agent_cfg = prepare_task_cfgs(task, [], play=True)
  env_cfg.scene.num_envs = len(seqs)
  base = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
  r = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cuda:0")
  r.load(CKPT, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  pol = r.get_inference_policy(device="cuda:0")
  cmd = base.command_manager.get_term("motion")
  mf = list(cmd.cfg.motion_files)
  lock_idx = torch.tensor(
    [next(i for i, m in enumerate(mf) if nm in m) for nm in seqs], device="cuda:0")
  cmd._resample_command = types.MethodType(make_locked(lock_idx), cmd)
  base.reset()
  obs = env.get_observations()
  maxz = torch.full((len(seqs),), -1e9, device="cuda:0")
  for _ in range(300):
    obs, _, _, _ = env.step(pol(obs))
    maxz = torch.maximum(maxz, cmd.obj_pos[:, 2])
  refpeak = cmd._ref_obj_pos[lock_idx][:, :, 2].max(dim=1).values
  ok = (maxz >= refpeak - 0.05)
  for nm, mz, rp, o in zip(seqs, maxz.tolist(), refpeak.tolist(), ok.tolist()):
    rows.append((nm, mz, rp, o))
  n_ok = int(ok.sum()); tot_ok += n_ok; tot_n += len(seqs)
  print(f"  {oname}: {n_ok}/{len(seqs)}", flush=True)
  del env, base, r
  torch.cuda.empty_cache()

print(f"\n===== {LABEL}  lift max_z  (data={DATA_VER.split('_')[-1]}) =====")
for nm, mz, rp, o in rows:
  print(f"{nm:<26}{mz:>8.3f} {'OK' if o else '.':>3}{rp:>9.3f}")
print("-" * 48)
print(f"{LABEL}: {tot_ok}/{tot_n} lifted")

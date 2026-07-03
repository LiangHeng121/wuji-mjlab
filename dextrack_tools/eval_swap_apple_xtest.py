"""Diagnose swap apple 0/8: is it the swap ENV (mis-instantiates apple) or the
swap TRAINING (policy never learned apple)? Load BOTH the swap ckpt and the PARK
ckpt into the SWAP env, eval apple-lift (8 seqs, one per env). If park-policy also
fails apple in swap-env -> env bug; if park-policy lifts -> swap training issue.
Also dumps apple's z-range (moved vs static) + finger-contact count.  cwd=wuji-mjlab.
"""
from __future__ import annotations
import glob
import types
from dataclasses import asdict

import torch

import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

CK = "logs/rsl_rl/wuji_tracking"
SWAP = "WujiHand_Tracking_3Obj_CGSmooth_Contact_Swap"


def latest(pat):
  return max(glob.glob(pat), key=lambda x: int(x.split("model_")[-1][:-3]))


CKPTS = [
  ("swap8000_ckpt", latest(f"{CK}/*Swap_Env8000*/model_*.pt")),
  ("park_ckpt_in_swap_env", f"{CK}/2026-06-21_06-50-25_Tracking_3Obj_CGSmooth_Contact/model_9999.pt"),
]
APPLE = [f"ori_grab_s{i}_apple_lift" for i in (1, 2, 3, 4, 6, 7, 8, 9)]
N = len(APPLE)


def make_locked(lock_idx):
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
    self._write_swap(env_ids, self.env_obj[env_ids])
  return _resample


for label, ckpt in CKPTS:
  print(f"\n===== [{label}] {ckpt}", flush=True)
  env_cfg, agent_cfg = prepare_task_cfgs(SWAP, [], play=True)
  env_cfg.scene.num_envs = N
  base = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
  r = (load_runner_cls(SWAP) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cuda:0")
  try:
    r.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  except Exception as e:
    print(f"  strict load failed ({e}); retry strict=False")
    r.load(ckpt, load_cfg={"actor": True}, strict=False, map_location="cuda:0")
  pol = r.get_inference_policy(device="cuda:0")
  cmd = base.command_manager.get_term("motion")
  mf = list(cmd.cfg.motion_files)
  lock_idx = torch.tensor(
    [next(i for i, m in enumerate(mf) if nm in m) for nm in APPLE], device="cuda:0")
  cmd._resample_command = types.MethodType(make_locked(lock_idx), cmd)
  base.reset()
  obs = env.get_observations()
  maxz = torch.full((N,), -1e9, device="cuda:0")
  minz = torch.full((N,), 1e9, device="cuda:0")
  for _ in range(300):
    obs, _, _, _ = env.step(pol(obs))
    z = cmd.obj_pos[:, 2]
    maxz = torch.maximum(maxz, z)
    minz = torch.minimum(minz, z)
  refpeak = cmd._ref_obj_pos[lock_idx][:, :, 2].max(dim=1).values
  mz, mnz, rp = maxz.cpu().numpy(), minz.cpu().numpy(), refpeak.cpu().numpy()
  nok = 0
  for i, nm in enumerate(APPLE):
    ok = mz[i] >= rp[i] - 0.05
    nok += ok
    print(f"  {nm:<22} max_z={mz[i]:.3f} z_range={mz[i]-mnz[i]:.3f} ref={rp[i]:.3f} {'✓' if ok else '·'}")
  print(f"  >>> {label}: {nok}/8")
  del env, base, r
  torch.cuda.empty_cache()

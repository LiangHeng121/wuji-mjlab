"""Rollout the 4 configs on the 8 apple-lift seqs, record per-step object z.

Produces dextrack_tools/figs/apple_ztraj.npz with:
  phys_z[label][seq]  -> (300,) physics object z per step
  ref_z[label][seq]   -> (300,) reference object z aligned to current time_step
  ref_peak[seq], max_z[label][seq], lifted[label][seq]

cwd = wuji-mjlab.  Set CUDA_VISIBLE_DEVICES to an idle GPU.
"""
from __future__ import annotations
import types
from dataclasses import asdict

import numpy as np
import torch

import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

import glob as _glob


def latest_model(dir_glob: str) -> str:
  """Pick the highest-iter model_*.pt among dirs matching dir_glob (robust to
  intermediate-ckpt cleanup)."""
  dirs = sorted(_glob.glob(dir_glob))
  assert dirs, f"no dir matches {dir_glob}"
  d = dirs[-1]
  models = _glob.glob(f"{d}/model_*.pt")
  assert models, f"no model_*.pt in {d}"
  return max(models, key=lambda p: int(p.split("model_")[-1].split(".pt")[0]))


CK = "logs/rsl_rl/wuji_tracking"
# label, task, ckpt-dir-glob (latest model auto-picked), is_3obj
RUNS = [
  ("apple8000kp1", "WujiHand_Tracking_AppleMulti_CGSmooth_Contact",
   latest_model(f"{CK}/*AppleMulti_CGSmooth_Contact_Env8000*"), False),
  ("apple22000kp1", "WujiHand_Tracking_AppleMulti_CGSmooth_Contact",
   latest_model(f"{CK}/2026-06-21_06-50-25_Tracking_AppleMulti_CGSmooth_Contact"), False),
  ("apple22000kp8", "WujiHand_Tracking_AppleMulti_CGSmooth_Contact_Kp8",
   latest_model(f"{CK}/*AppleMulti_CGSmooth_Contact_Kp8_Env22000*"), False),
  ("3obj", "WujiHand_Tracking_3Obj_CGSmooth_Contact",
   latest_model(f"{CK}/2026-06-21_06-50-25_Tracking_3Obj_CGSmooth_Contact"), True),
]
APPLE_LIFT = [f"ori_grab_s{i}_apple_lift" for i in (1, 2, 3, 4, 6, 7, 8, 9)]
NSTEP = 300


def make_locked(idx: int, is_3obj: bool):
  def _resample(self, env_ids):
    self.time_steps[env_ids] = 0
    self.env_seq[env_ids] = idx
    self.env_obj[env_ids] = self.seq_obj[self.env_seq[env_ids]]
    s, t = self.env_seq[env_ids], self.time_steps[env_ids]
    soft = self.robot.data.soft_joint_pos_limits[env_ids]
    jp = torch.clip(self._ref_qpos[s, t].clone(), soft[:, :, 0], soft[:, :, 1])
    self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)
    ref_pos, ref_quat = self._ref_obj_pos[s, t], self._ref_obj_quat[s, t]
    vel = torch.zeros(len(env_ids), 6, device=self.device)
    if is_3obj:
      active = self.env_obj[env_ids]
      for j, obj in enumerate(self.objs):
        is_act = (active == j).unsqueeze(-1)
        pos = torch.where(is_act, ref_pos, self._park[j, :3].unsqueeze(0))
        quat = torch.where(is_act, ref_quat, self._park[j, 3:].unsqueeze(0))
        obj.write_root_state_to_sim(torch.cat([pos, quat, vel], dim=-1), env_ids=env_ids)
        obj.reset(env_ids=env_ids)
    else:
      self.obj.write_root_state_to_sim(
        torch.cat([ref_pos, ref_quat, vel], dim=-1), env_ids=env_ids)
      self.obj.reset(env_ids=env_ids)
  return _resample


phys_z, ref_z, max_z, lifted = {}, {}, {}, {}
ref_peak = {}

for label, task, ckpt, is_3obj in RUNS:
  print(f"\n=== {label} ({task}) ===", flush=True)
  env_cfg, agent_cfg = prepare_task_cfgs(task, [], play=True)
  env_cfg.scene.num_envs = 1
  base = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
  r = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cuda:0")
  r.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  pol = r.get_inference_policy(device="cuda:0")
  cmd = base.command_manager.get_term("motion")
  mf = list(cmd.cfg.motion_files)
  phys_z[label], ref_z[label], max_z[label], lifted[label] = {}, {}, {}, {}
  for nm in APPLE_LIFT:
    idx = next(i for i, m in enumerate(mf) if nm in m)
    cmd._resample_command = types.MethodType(make_locked(idx, is_3obj), cmd)
    base.reset()
    obs = env.get_observations()
    ref_full = cmd._ref_obj_pos[idx, :, 2].detach().cpu().numpy()  # (T,)
    T = ref_full.shape[0]
    pz, rz = [], []
    for _ in range(NSTEP):
      obs, _, _, _ = env.step(pol(obs))
      pz.append(float(cmd.obj_pos[0, 2]))
      t = int(cmd.time_steps[0].item())
      rz.append(float(ref_full[min(t, T - 1)]))
    pz = np.asarray(pz, dtype=np.float32)
    rz = np.asarray(rz, dtype=np.float32)
    phys_z[label][nm] = pz
    ref_z[label][nm] = rz
    rp = float(ref_full.max())
    ref_peak[nm] = rp
    mz = float(pz.max())
    max_z[label][nm] = mz
    lifted[label][nm] = bool(mz >= rp - 0.05)
    print(f"  {nm:<22} max_z={mz:.3f} ref_peak={rp:.3f} "
          f"{'lifted' if lifted[label][nm] else '.'}", flush=True)
  del env, base, r
  torch.cuda.empty_cache()

# flatten into arrays for npz (labels x seqs x NSTEP)
labels = [r[0] for r in RUNS]
PZ = np.stack([[phys_z[l][s] for s in APPLE_LIFT] for l in labels])  # (L,S,300)
RZ = np.stack([[ref_z[l][s] for s in APPLE_LIFT] for l in labels])
MZ = np.array([[max_z[l][s] for s in APPLE_LIFT] for l in labels])   # (L,S)
LF = np.array([[lifted[l][s] for s in APPLE_LIFT] for l in labels])
RP = np.array([ref_peak[s] for s in APPLE_LIFT])

import os
os.makedirs("dextrack_tools/figs", exist_ok=True)
np.savez("dextrack_tools/figs/apple_ztraj.npz",
         labels=np.array(labels), seqs=np.array(APPLE_LIFT),
         phys_z=PZ, ref_z=RZ, max_z=MZ, lifted=LF, ref_peak=RP)
print("\nsaved dextrack_tools/figs/apple_ztraj.npz")

"""Q3 eval (parallel): 3obj generalist full 22 lift-sequence max_z, three-way.
Each of the 22 lift sequences is pinned to its own env (num_envs=22) and evaluated
in ONE 300-step rollout per ckpt.  Success = max_z >= ref_peak - 0.05.
  park8000  : historical baseline (doc 16/22), 3 bodies + PARK non-active.
  swap8000  : path-(c) swap, single body per env (no park pollution).
  swap22000 : swap at 22000 env (mid ckpt, converged).
cwd = wuji-mjlab.
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
PARK = "WujiHand_Tracking_3Obj_CGSmooth_Contact"
SWAP = "WujiHand_Tracking_3Obj_CGSmooth_Contact_Swap"


def latest(pat):
  return max(glob.glob(pat), key=lambda x: int(x.split("model_")[-1][:-3]))


RUNS = [
  ("park8000",  PARK, f"{CK}/2026-06-21_06-50-25_Tracking_3Obj_CGSmooth_Contact/model_9999.pt", "park"),
  ("swap8000",  SWAP, latest(f"{CK}/*Swap_Env8000*/model_*.pt"), "swap"),
  ("swap22000", SWAP, latest(f"{CK}/*Swap_Env22000*/model_*.pt"), "swap"),
]
GROUPS = [
  ("cube",  [f"ori_grab_s{i}_cubesmall_lift" for i in (10, 1, 2, 5, 6, 8)]),
  ("cup",   [f"ori_grab_s{i}_cup_lift" for i in (10, 1, 3, 4, 5, 6, 8, 9)]),
  ("apple", [f"ori_grab_s{i}_apple_lift" for i in (1, 2, 3, 4, 6, 7, 8, 9)]),
]
SEQS = [s for _, g in GROUPS for s in g]
N = len(SEQS)


def make_locked(lock_idx, mode):
  def _resample(self, env_ids):
    self.time_steps[env_ids] = 0
    self.env_seq[env_ids] = lock_idx[env_ids]  # each env pinned to its own seq
    self.env_obj[env_ids] = self.seq_obj[self.env_seq[env_ids]]
    s, t = self.env_seq[env_ids], self.time_steps[env_ids]
    soft = self.robot.data.soft_joint_pos_limits[env_ids]
    jp = torch.clip(self._ref_qpos[s, t].clone(), soft[:, :, 0], soft[:, :, 1])
    self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)
    ref_pos, ref_quat = self._ref_obj_pos[s, t], self._ref_obj_quat[s, t]
    vel = torch.zeros(len(env_ids), 6, device=self.device)
    if mode == "swap":
      self.obj.write_root_state_to_sim(
        torch.cat([ref_pos, ref_quat, vel], dim=-1), env_ids=env_ids)
      self.obj.reset(env_ids=env_ids)
      self._write_swap(env_ids, self.env_obj[env_ids])
    else:
      active = self.env_obj[env_ids]
      for j, obj in enumerate(self.objs):
        is_act = (active == j).unsqueeze(-1)
        pos = torch.where(is_act, ref_pos, self._park[j, :3].unsqueeze(0))
        quat = torch.where(is_act, ref_quat, self._park[j, 3:].unsqueeze(0))
        obj.write_root_state_to_sim(torch.cat([pos, quat, vel], dim=-1), env_ids=env_ids)
        obj.reset(env_ids=env_ids)
  return _resample


results = {}  # label -> (maxz[N], refpeak[N])
for label, task, ckpt, mode in RUNS:
  print(f"[{label}] {ckpt}", flush=True)
  env_cfg, agent_cfg = prepare_task_cfgs(task, [], play=True)
  env_cfg.scene.num_envs = N
  base = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
  r = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cuda:0")
  r.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  pol = r.get_inference_policy(device="cuda:0")
  cmd = base.command_manager.get_term("motion")
  mf = list(cmd.cfg.motion_files)
  lock_idx = torch.tensor(
    [next(i for i, m in enumerate(mf) if nm in m) for nm in SEQS], device="cuda:0")
  cmd._resample_command = types.MethodType(make_locked(lock_idx, mode), cmd)
  base.reset()
  obs = env.get_observations()
  maxz = torch.full((N,), -1e9, device="cuda:0")
  for _ in range(300):
    obs, _, _, _ = env.step(pol(obs))
    maxz = torch.maximum(maxz, cmd.obj_pos[:, 2])
  refpeak = cmd._ref_obj_pos[lock_idx][:, :, 2].max(dim=1).values
  results[label] = (maxz.cpu().numpy(), refpeak.cpu().numpy())
  nok = int((maxz >= refpeak - 0.05).sum())
  print(f"  {label}: {nok}/{N}", flush=True)
  del env, base, r
  torch.cuda.empty_cache()

labels = [r[0] for r in RUNS]
print("\n===== 3obj full-22 max_z (✓ = mz >= ref_peak - 0.05) =====")
print(f"{'sequence':<26}" + "".join(f"{l:>15}" for l in labels) + f"{'ref':>8}")
counts = {l: {g: 0 for g, _ in GROUPS} for l in labels}
si = 0
for gname, gseqs in GROUPS:
  for nm in gseqs:
    rp = results[labels[0]][1][si]
    row = f"{nm:<26}"
    for l in labels:
      mz = results[l][0][si]
      ok = mz >= rp - 0.05
      counts[l][gname] += ok
      row += f"{mz:>10.3f} {'✓' if ok else '·'}  "
    print(row + f"{rp:>8.3f}")
    si += 1
  print("-" * 78)
for gname, gseqs in GROUPS:
  print(f"{'  ' + gname:<26}" + "".join(
    f"{str(counts[l][gname]) + '/' + str(len(gseqs)):>15}" for l in labels))
print(f"{'TOTAL n/22':<26}" + "".join(
  f"{str(sum(counts[l].values())) + '/22':>15}" for l in labels))

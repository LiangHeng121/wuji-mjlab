"""TOPO A/B eval: apple-lift max_z for the TOPO-data specialist vs known FPOS baseline.

FPOS baseline (exp1, same config, FPOS data) = 0/8 lifted. Question: does TOPO's
low-penetration retarget data (1.2mm vs 17.6mm) let the apple specialist lift?

Success = max_z >= ref_peak - 0.05. Single-object apple env, TOPO data loaded via
WUJI_DATA_VER (set BEFORE importing wuji_mjlab.tasks so env_cfgs picks it up).
Object reference trajectory is the same GRAB motion in both → ref_peak comparable
to the FPOS baseline. cwd = wuji-mjlab. Run on an idle GPU (cuda:0 via CVD).
"""
from __future__ import annotations
import os
os.environ["WUJI_DATA_VER"] = "GRAB_Tracking_PK_WUJI_FPOS_v1"  # MUST be before import

import types
from dataclasses import asdict

import torch

import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

CK = "logs/rsl_rl/wuji_tracking"
RUNS = [
  ("masscur", "WujiHand_Tracking_AppleMulti_CGSmooth_Contact",
   "logs/rsl_rl/wuji_tracking/2026-07-04_02-01-03_Tracking_AppleMulti_CGSmooth_Contact_MassCur_Env8000/model_3250.pt", False),
]
APPLE_LIFT = [f"ori_grab_s{i}_apple_lift" for i in (1, 2, 3, 4, 6, 7, 8, 9)]


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
    self.obj.write_root_state_to_sim(
      torch.cat([ref_pos, ref_quat, vel], dim=-1), env_ids=env_ids)
    self.obj.reset(env_ids=env_ids)
  return _resample


results = {}
for label, task, ckpt, is_3obj in RUNS:
  env_cfg, agent_cfg = prepare_task_cfgs(task, [], play=True)
  env_cfg.scene.num_envs = 1
  base = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
  r = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cuda:0")
  r.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  pol = r.get_inference_policy(device="cuda:0")
  cmd = base.command_manager.get_term("motion")
  mf = list(cmd.cfg.motion_files)
  for nm in APPLE_LIFT:
    idx = next(i for i, m in enumerate(mf) if nm in m)
    cmd._resample_command = types.MethodType(make_locked(idx, is_3obj), cmd)
    base.reset()
    obs = env.get_observations()
    oz = []
    for _ in range(300):
      obs, _, _, _ = env.step(pol(obs))
      oz.append(float(cmd.obj_pos[0, 2]))
    max_z = max(oz)
    ref_peak = float(cmd._ref_obj_pos[idx, :, 2].max())
    results[(label, nm)] = (max_z, ref_peak, max_z >= ref_peak - 0.05)
  del env, base, r
  torch.cuda.empty_cache()

print("\n===== apple-lift max_z  MassCur@9999-FINISHED nominal-mass  FPOS  (✓ = max_z >= ref_peak - 0.05) =====")
print(f"{'sequence':<22}{'apple_topo':>16}{'ref_peak':>10}")
for nm in APPLE_LIFT:
  mz, rp, ok = results[("masscur", nm)]
  print(f"{nm:<22}{mz:>12.3f} {'✓' if ok else '·':>3}{rp:>10.3f}")
n = sum(1 for nm in APPLE_LIFT if results[("masscur", nm)][2])
print("-" * 48)
print(f"{'n/8 lifted (FPOS baseline=0/8)':<22}{str(n) + '/8':>16}")

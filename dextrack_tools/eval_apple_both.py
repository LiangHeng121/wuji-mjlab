"""Per-seq apple-lift eval with BOTH criteria, to find disagreements:
  max_z (old):   pass if max_z >= ref_peak - 0.05
  track (new):   pass if mean ||obj_pos - ref_obj_pos|| < 0.05 m
Prints per seq and flags DISAGREE = track-pass but max_z-fail.
Usage: eval_apple_both.py <ckpt> <data_ver>
"""
from __future__ import annotations
import os, sys
CKPT, DATA_VER = sys.argv[1], sys.argv[2]
os.environ["WUJI_DATA_VER"] = DATA_VER

import types
from dataclasses import asdict
import torch

import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

TASK = "WujiHand_Tracking_AppleMulti_CGSmooth_Contact"
SEQS = [f"ori_grab_s{i}_apple_lift" for i in (1, 2, 3, 4, 6, 7, 8, 9)]


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
  return _resample


env_cfg, agent_cfg = prepare_task_cfgs(TASK, [], play=True)
env_cfg.scene.num_envs = len(SEQS)
base = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
r = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cuda:0")
r.load(CKPT, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
pol = r.get_inference_policy(device="cuda:0")
cmd = base.command_manager.get_term("motion")
mf = list(cmd.cfg.motion_files)
lock_idx = torch.tensor([next(i for i, m in enumerate(mf) if nm in m) for nm in SEQS], device="cuda:0")
cmd._resample_command = types.MethodType(make_locked(lock_idx), cmd)
base.reset()
obs = env.get_observations()
maxz = torch.full((len(SEQS),), -1e9, device="cuda:0")
err_sum = torch.zeros(len(SEQS), device="cuda:0")
err_max = torch.zeros(len(SEQS), device="cuda:0")
# error at the frame where the REFERENCE object is highest (the lift moment)
ref_z_traj = cmd._ref_obj_pos[lock_idx][:, :, 2]  # [n, T]
peak_t = ref_z_traj.argmax(dim=1)                  # per-seq ref-peak timestep
err_at_peak = torch.zeros(len(SEQS), device="cuda:0")
for t in range(300):
  obs, _, _, _ = env.step(pol(obs))
  e = torch.linalg.norm(cmd.obj_pos - cmd.ref_obj_pos, dim=-1)
  maxz = torch.maximum(maxz, cmd.obj_pos[:, 2])
  err_sum += e
  err_max = torch.maximum(err_max, e)
  at_peak = (peak_t == t)
  err_at_peak = torch.where(at_peak, e, err_at_peak)
refpeak = ref_z_traj.max(dim=1).values
mean_err = err_sum / 300

print("\n==== apple-lift: max_z(old) vs 各种track误差 ====")
print(f"{'seq':<22}{'max_z':>7}{'refpk':>7}{'mzOK':>5}{'meanE':>7}{'maxE':>7}{'peakE':>7}")
for i, nm in enumerate(SEQS):
  mz, rp = maxz[i].item(), refpeak[i].item()
  mzok = mz >= rp - 0.05
  print(f"{nm:<22}{mz:>7.3f}{rp:>7.3f}{'✓' if mzok else '·':>5}"
        f"{mean_err[i].item():>7.3f}{err_max[i].item():>7.3f}{err_at_peak[i].item():>7.3f}")
print(f"\nmax_z通过: {sum(1 for i in range(len(SEQS)) if maxz[i]>=refpeak[i]-0.05)}/8")
print("校准: 看 maxE / peakE 能否把 max_z✓ 和 ✗ 分开")
for th in (0.03, 0.05, 0.08, 0.10):
  nmax = sum(1 for i in range(len(SEQS)) if err_max[i] < th)
  npk = sum(1 for i in range(len(SEQS)) if err_at_peak[i] < th)
  print(f"  阈值{th}: maxE通过={nmax}/8  peakE通过={npk}/8")

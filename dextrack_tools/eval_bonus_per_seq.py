"""Per-sequence test bonus: roll out each of the 3 multi-seq policies locked to
each given sequence, report episode-SUM object_inplace_bonus (consistent def:
grip 0.22, 5-finger) + max_z + fair, so the 3 policies are comparable per seq.
"""
from __future__ import annotations

import glob
import types
from dataclasses import asdict

import numpy as np
import torch

import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from wuji_mjlab.tasks.tracking.config.wuji_hand.env_cfgs import _cubesmall_sequences
from wuji_mjlab.tasks.tracking.mdp import rewards as R
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

import os
SEQS = _cubesmall_sequences() if os.environ.get("ALL_SEQS") else [
  "ori_grab_s10_cubesmall_inspect_1",
  "ori_grab_s3_cubesmall_inspect_1",
  "ori_grab_s5_cubesmall_pass_1",
]
POLICIES = [
  ("Original", "WujiHand_Tracking_CubesmallMulti_Original"),
  ("Pinall3", "WujiHand_Tracking_CubesmallMulti_Pinall3"),
  ("CGSmooth", "WujiHand_Tracking_CubesmallMulti_CGSmooth"),
]
NAMES = _cubesmall_sequences()
IDX = {n: NAMES.index(n) for n in SEQS}


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


results = {}  # (lbl, seq) -> dict
for lbl, task in POLICIES:
  ckpt = sorted(glob.glob(f"logs/rsl_rl/wuji_tracking/*CubesmallMulti_{lbl}/model_*.pt"),
                key=lambda x: int(x.split("_")[-1][:-3]))[-1]
  it = ckpt.split("_")[-1][:-3]
  env_cfg, agent_cfg = prepare_task_cfgs(task, [], play=True)
  env_cfg.scene.num_envs = 1
  base = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
  r = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cuda:0")
  r.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  pol = r.get_inference_policy(device="cuda:0")
  cmd = base.command_manager.get_term("motion")
  obj = base.scene["object"]
  for nm, idx in IDX.items():
    cmd._resample_command = types.MethodType(locked(idx), cmd)
    base.reset()
    obs = env.get_observations()
    bonus = 0.0
    oz, rz = [], []
    for _ in range(300):
      obs, _, _, _ = env.step(pol(obs))
      # consistent bonus def for all policies: grip 0.22, 5-finger.
      bonus += float(R.object_inplace_bonus(base, "motion", grip_thres=0.22,
                                            n_finger_sum=5)[0])
      oz.append(float(obj.data.root_link_pos_w[0, 2]))
      rz.append(float(cmd.ref_obj_pos[0, 2]))
    oz, rz = np.array(oz), np.array(rz)
    results[(lbl, nm)] = dict(it=it, bonus=bonus, hold=bonus / 300 * 100,
                              max_z=oz.max(), ref=rz.max(),
                              zerr=float(np.mean(np.abs(oz - rz))))
  del env, base, r
  torch.cuda.empty_cache()

def cell(d):
  return f"{d['bonus']:>6.1f}({d['hold']:>3.0f}%)"


labels = [lbl for lbl, _ in POLICIES]
print(f"\n===== PER-SEQUENCE TEST BONUS (episode-sum/300, consistent grip0.22/5-finger; "
      f"iter {'/'.join(results[(l, SEQS[0])]['it'] for l in labels)}) =====")
print(f"{'sequence':<34}" + "".join(f"{l:>14}" for l in labels) + "   best")
for nm in sorted(SEQS):
  vals = {l: results[(l, nm)]['bonus'] for l in labels}
  best = max(vals, key=vals.get)
  print(f"{nm:<34}" + "".join(f"{cell(results[(l, nm)]):>14}" for l in labels) + f"   {best}")

print("\n----- 按 CGSmooth bonus 从高到低排序 -----")
print(f"{'sequence':<34}{'CGSmooth':>12}{'Pinall3':>12}{'Original':>12}")
for nm in sorted(SEQS, key=lambda n: results[("CGSmooth", n)]['bonus'], reverse=True):
  print(f"{nm:<34}" + f"{results[('CGSmooth',nm)]['bonus']:>12.1f}"
        + f"{results[('Pinall3',nm)]['bonus']:>12.1f}"
        + f"{results[('Original',nm)]['bonus']:>12.1f}")

print("\n----- 汇总 (满分300) -----")
for l in labels:
  bs = [results[(l, n)]['bonus'] for n in SEQS]
  wins = sum(1 for n in SEQS if max(labels, key=lambda x: results[(x, n)]['bonus']) == l)
  fails = sum(1 for n in SEQS if results[(l, n)]['hold'] < 50)
  print(f"{l:>10}: 均值 {np.mean(bs):>6.1f} | 最佳序列数 {wins:>2}/{len(SEQS)} | 失败(<50%) {fails}")

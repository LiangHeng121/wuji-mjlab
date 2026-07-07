"""Full-sequence eval: object POSITION TRACKING ERROR across ALL 84 training seqs
(cube 26 + cup 31 + apple 27), not just the 22 lift seqs. Works for every action
type (lift/pass/inspect/drink/pour/eat/offhand) since tracking error is defined
regardless of whether the object is lifted.

Per seq: mean over the 300-step rollout of ||obj_pos - ref_obj_pos||. Success =
mean_err < THRESH (default 0.05 m). Reports per-object success + error stats and a
few threshold cuts. Usage: eval_gen3obj_fullseq.py <ckpt> <data_ver> <label>
"""
from __future__ import annotations
import os, sys
CKPT, DATA_VER, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ["WUJI_DATA_VER"] = DATA_VER

import types
from dataclasses import asdict
import torch

import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from wuji_mjlab.tasks.tracking.config.wuji_hand.env_cfgs import _object_sequences
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

THRESH = 0.05  # m; success if mean object-pos error below this
OBJS = [
  ("cube", "WujiHand_Tracking_CubesmallMulti_CGSmooth_Contact", "cubesmall"),
  ("cup", "WujiHand_Tracking_CupMulti_CGSmooth_Contact", "cup"),
  ("apple", "WujiHand_Tracking_AppleMulti_CGSmooth_Contact", "apple"),
]


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


print(f"[{LABEL}] ckpt={CKPT} data={DATA_VER} thresh={THRESH}m", flush=True)
allrows, tot_ok, tot_n = [], 0, 0
for oname, task, objkey in OBJS:
  seqs = _object_sequences(objkey)
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
  # success judged at the frame where the REFERENCE object is displaced most from
  # its start (lift peak / farthest-move moment) — the moment that most demands the
  # object be manipulated. mean-error is fooled (diluted by resting frames).
  ref_traj = cmd._ref_obj_pos[lock_idx]                         # [n, T, 3]
  ref_disp = torch.linalg.norm(ref_traj - ref_traj[:, :1], dim=-1)  # [n, T]
  peak_t = ref_disp.argmax(dim=1)                                # [n]
  base.reset()
  obs = env.get_observations()
  err_peak = torch.zeros(len(seqs), device="cuda:0")
  for t in range(300):
    obs, _, _, _ = env.step(pol(obs))
    e = torch.linalg.norm(cmd.obj_pos - cmd.ref_obj_pos, dim=-1)
    err_peak = torch.where(peak_t == t, e, err_peak)
  err_peak = err_peak.cpu()
  ok = err_peak < THRESH
  for nm, e, o in zip(seqs, err_peak.tolist(), ok.tolist()):
    act = nm.split(objkey + "_")[-1].rsplit("_", 1)[0] if objkey + "_" in nm else "?"
    allrows.append((oname, act, nm, e, o))
  n_ok = int(ok.sum()); tot_ok += n_ok; tot_n += len(seqs)
  print(f"  {oname}: {n_ok}/{len(seqs)}  (peakErr med={err_peak.median():.3f})", flush=True)
  del env, base, r
  torch.cuda.empty_cache()

# per-action-type breakdown
print(f"\n===== {LABEL}  全序列 obj-track-err  (成功=<{THRESH}m) =====")
from collections import defaultdict
by_act = defaultdict(lambda: [0, 0])
for oname, act, nm, e, o in allrows:
  key = f"{oname}/{act}"
  by_act[key][0] += int(o); by_act[key][1] += 1
for key in sorted(by_act):
  a, b = by_act[key]
  print(f"  {key:<20} {a}/{b}")
print("-" * 40)
print(f"{LABEL}: {tot_ok}/{tot_n} 全序列成功")
# 多阈值
for th in (0.02, 0.05, 0.10):
  n = sum(1 for _, _, _, e, _ in allrows if e < th)
  print(f"  阈值<{th}m: {n}/{tot_n}")

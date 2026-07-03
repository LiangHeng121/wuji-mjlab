"""Q1 eval: per apple-lift-sequence offline rollout, record max_z vs ref_peak.

Success = max_z >= ref_peak - 0.05. Evaluates 3 ckpts:
  1. apple specialist 8000 env (exp1, new)   -> single-object apple env
  2. apple specialist 22000 env (old, 0/8)   -> single-object apple env
  3. 3obj park generalist (apple slice, 3/8) -> 3obj env, LOCK apple seq + PARK
     the non-active cube/cup (else they pollute the grasp).

Answers Q1: is apple-specialist failure (0/8) about env count (22000) or missing
multi-object transfer? Run on an idle GPU.  cwd = wuji-mjlab.
"""
from __future__ import annotations
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
  ("apple8000",  "WujiHand_Tracking_AppleMulti_CGSmooth_Contact",
   f"{CK}/2026-06-24_23-51-19_Tracking_AppleMulti_CGSmooth_Contact_Env8000/model_9999.pt", False),
  ("apple22000", "WujiHand_Tracking_AppleMulti_CGSmooth_Contact",
   f"{CK}/2026-06-21_06-50-25_Tracking_AppleMulti_CGSmooth_Contact/model_4500.pt", False),
  ("3obj_park",  "WujiHand_Tracking_3Obj_CGSmooth_Contact",
   f"{CK}/2026-06-21_06-50-25_Tracking_3Obj_CGSmooth_Contact/model_9999.pt", True),
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
    if is_3obj:  # active -> ref, others -> parked far (avoid pollution)
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

print("\n========== apple-lift max_z (✓ = max_z >= ref_peak - 0.05) ==========")
hdr = f"{'sequence':<22}"
for label, _, _, _ in RUNS:
  hdr += f"{label:>20}"
print(hdr + f"{'ref_peak':>10}")
for nm in APPLE_LIFT:
  rp = results[(RUNS[0][0], nm)][1]
  row = f"{nm:<22}"
  for label, _, _, _ in RUNS:
    mz, _, ok = results[(label, nm)]
    row += f"{mz:>14.3f} {'✓' if ok else '·':>4}"
  print(row + f"{rp:>10.3f}")
print("-" * (22 + 20 * len(RUNS) + 10))
foot = f"{'n/8 lifted':<22}"
for label, _, _, _ in RUNS:
  n = sum(1 for nm in APPLE_LIFT if results[(label, nm)][2])
  foot += f"{str(n) + '/8':>20}"
print(foot)

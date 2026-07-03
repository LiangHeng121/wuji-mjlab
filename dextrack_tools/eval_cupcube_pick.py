"""Pick sequences for cup/cube cmp videos: per cup-lift & cube-lift seq, offline
rollout max_z for the single-object specialist AND the 3obj generalist (park).
Prints a table so we can choose (a) a cup-lift seq where specialist FAILS but 3obj
LIFTS, (b) a cube-lift lift representative.  cwd = wuji-mjlab, idle GPU.
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

import glob
CK = "logs/rsl_rl/wuji_tracking"


def latest_model(dir_glob):
  dirs = sorted(glob.glob(dir_glob))
  d = dirs[-1]
  models = glob.glob(f"{d}/model_*.pt")
  return max(models, key=lambda p: int(p.split("model_")[-1].split(".pt")[0]))


CUP = [f"ori_grab_s{i}_cup_lift" for i in (1, 3, 4, 5, 6, 8, 9, 10)]
CUBE = [f"ori_grab_s{i}_cubesmall_lift" for i in (1, 2, 5, 6, 8, 10)]

# label, task, ckpt, is_3obj, seqs
RUNS = [
  ("cup_spec", "WujiHand_Tracking_CupMulti_CGSmooth_Contact",
   latest_model(f"{CK}/*CupMulti_CGSmooth_Contact"), False, CUP),
  ("cube_spec", "WujiHand_Tracking_CubesmallMulti_CGSmooth_Contact",
   latest_model(f"{CK}/*CubesmallMulti_CGSmooth_Contact"), False, CUBE),
  ("3obj", "WujiHand_Tracking_3Obj_CGSmooth_Contact",
   latest_model(f"{CK}/2026-06-21_06-50-25_Tracking_3Obj_CGSmooth_Contact"), True, CUP + CUBE),
]


def make_locked(idx, is_3obj):
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


res = {}
for label, task, ckpt, is_3obj, seqs in RUNS:
  print(f"\n=== {label}: {ckpt}", flush=True)
  env_cfg, agent_cfg = prepare_task_cfgs(task, [], play=True)
  env_cfg.scene.num_envs = 1
  base = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
  r = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cuda:0")
  r.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  pol = r.get_inference_policy(device="cuda:0")
  cmd = base.command_manager.get_term("motion")
  mf = list(cmd.cfg.motion_files)
  for nm in seqs:
    idx = next(i for i, m in enumerate(mf) if nm in m)
    cmd._resample_command = types.MethodType(make_locked(idx, is_3obj), cmd)
    base.reset()
    obs = env.get_observations()
    oz = []
    for _ in range(300):
      obs, _, _, _ = env.step(pol(obs))
      oz.append(float(cmd.obj_pos[0, 2]))
    mz = max(oz)
    rp = float(cmd._ref_obj_pos[idx, :, 2].max())
    res[(label, nm)] = (mz, rp)
    print(f"  {nm:<26} max_z={mz:.3f} ref_peak={rp:.3f} "
          f"{'LIFT' if mz >= rp - 0.05 else '.'}", flush=True)
  del env, base, r
  torch.cuda.empty_cache()

print("\n===== CUP: spec vs 3obj =====")
print(f"{'seq':<26}{'spec':>8}{'3obj':>8}{'ref':>8}  pick")
for nm in CUP:
  sp = res[("cup_spec", nm)]
  ob = res[("3obj", nm)]
  spok = sp[0] >= sp[1] - 0.05
  obok = ob[0] >= ob[1] - 0.05
  tag = "<== DIFF" if (obok and not spok) else ""
  print(f"{nm:<26}{sp[0]:>8.3f}{ob[0]:>8.3f}{sp[1]:>8.3f}  {tag}")
print("\n===== CUBE: spec vs 3obj =====")
print(f"{'seq':<26}{'spec':>8}{'3obj':>8}{'ref':>8}")
for nm in CUBE:
  sp = res[("cube_spec", nm)]
  ob = res[("3obj", nm)]
  print(f"{nm:<26}{sp[0]:>8.3f}{ob[0]:>8.3f}{sp[1]:>8.3f}")

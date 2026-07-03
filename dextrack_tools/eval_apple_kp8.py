"""Q2 eval: apple kp8 specialist (22000 env) per apple-lift max_z vs ref_peak.
Control = apple kp1 22000 = 0/8 (measured in Q1). Same env (22000), only variable
is finger kp (8 vs 1). Does kp8 let apple lift?  cwd = wuji-mjlab."""
from __future__ import annotations
import types
from dataclasses import asdict

import torch

import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

TASK = "WujiHand_Tracking_AppleMulti_CGSmooth_Contact_Kp8"
CKPT = ("logs/rsl_rl/wuji_tracking/"
        "2026-06-24_23-51-22_Tracking_AppleMulti_CGSmooth_Contact_Kp8_Env22000/model_5000.pt")
APPLE_LIFT = [f"ori_grab_s{i}_apple_lift" for i in (1, 2, 3, 4, 6, 7, 8, 9)]


def make_locked(idx):
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


env_cfg, agent_cfg = prepare_task_cfgs(TASK, [], play=True)
env_cfg.scene.num_envs = 1
base = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
r = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cuda:0")
r.load(CKPT, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
pol = r.get_inference_policy(device="cuda:0")
cmd = base.command_manager.get_term("motion")
mf = list(cmd.cfg.motion_files)

res = {}
for nm in APPLE_LIFT:
  idx = next(i for i, m in enumerate(mf) if nm in m)
  cmd._resample_command = types.MethodType(make_locked(idx), cmd)
  base.reset()
  obs = env.get_observations()
  oz = [float(cmd.obj_pos[0, 2]) for _ in range(0)]
  for _ in range(300):
    obs, _, _, _ = env.step(pol(obs))
    oz.append(float(cmd.obj_pos[0, 2]))
  res[nm] = (max(oz), float(cmd._ref_obj_pos[idx, :, 2].max()))

print("\n===== apple kp8 (22000 env) max_z (✓ = mz >= ref-0.05) =====")
print(f"{'sequence':<22}{'kp8 max_z':>12}{'ref_peak':>10}{'':>4}")
n = 0
for nm in APPLE_LIFT:
  mz, rp = res[nm]
  ok = mz >= rp - 0.05
  n += ok
  print(f"{nm:<22}{mz:>12.3f}{rp:>10.3f}  {'✓' if ok else '·'}")
print(f"{'n/8 lifted':<22}{str(n) + '/8':>12}")

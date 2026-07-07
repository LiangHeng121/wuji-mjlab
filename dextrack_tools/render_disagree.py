"""Render the 一起-TOPO generalist on the max_z-vs-track DISAGREEMENT apple seqs
(s1/s8/s9: track-pass but max_z-fail) + s4 (both-pass, real lift) as reference.
Renders BOTH the policy rollout and overlays the object height vs reference so we
can see whether the object is really being tracked or just sitting on the table.
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl.
"""
from __future__ import annotations
import os
os.environ["WUJI_DATA_VER"] = "GRAB_Tracking_PK_WUJI_TOPO_v1"

import glob, types
from dataclasses import asdict
import numpy as np, torch, imageio

import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

TASK = "WujiHand_Tracking_AppleMulti_CGSmooth_Contact"
CKPT = sorted(glob.glob("logs/rsl_rl/wuji_tracking/*Gen3obj_MassCur_coef0_TOPO_Env8000/model_*.pt"),
              key=lambda x: int(x.split("_")[-1][:-3]))[-1]
SEQS = ["ori_grab_s1_apple_lift", "ori_grab_s8_apple_lift",
        "ori_grab_s9_apple_lift", "ori_grab_s4_apple_lift"]  # last = both-pass ref
OUTDIR = "/data/home/liangheng/DexTrack/videos/disagree"
os.makedirs(OUTDIR, exist_ok=True)
print("CKPT", CKPT)


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
base = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array")
env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
r = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cuda:0")
r.load(CKPT, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
pol = r.get_inference_policy(device="cuda:0")
cmd = base.command_manager.get_term("motion")
mf = list(cmd.cfg.motion_files)

for nm in SEQS:
  idx = next(i for i, m in enumerate(mf) if nm in m)
  cmd._resample_command = types.MethodType(make_locked(idx), cmd)
  base.reset()
  obs = env.get_observations()
  frames, oz, rz, err = [], [], [], []
  for _ in range(300):
    obs, _, _, _ = env.step(pol(obs))
    img = base.render()
    if img is not None: frames.append(np.asarray(img))
    oz.append(float(cmd.obj_pos[0, 2]))
    rz.append(float(cmd.ref_obj_pos[0, 2]))
    err.append(float(torch.linalg.norm(cmd.obj_pos[0] - cmd.ref_obj_pos[0])))
  out = f"{OUTDIR}/{nm}.mp4"
  imageio.mimwrite(out, frames, fps=30, quality=8)
  print(f"WROTE {out}  obj_z[max={max(oz):.3f}]  ref_z[max={max(rz):.3f}]  "
        f"track_err[mean={np.mean(err):.3f} max={max(err):.3f}]")

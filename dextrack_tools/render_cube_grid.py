"""Render mjlab cube specialist on candidate cube seqs (across actions) at high res,
report peak-displacement track error, save per-seq mp4 + frames. Successful ones
(peakErr<5cm) are later stitched into a grid GIF. MUJOCO_GL=egl."""
from __future__ import annotations
import os
os.environ["WUJI_DATA_VER"] = "GRAB_Tracking_PK_WUJI_FPOS_v1"

import glob, types
from dataclasses import asdict
import numpy as np, torch, imageio

import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

TASK = "WujiHand_Tracking_CubesmallMulti_CGSmooth_Contact"
CKPT = sorted(glob.glob("logs/rsl_rl/wuji_tracking/2026-06-21_06-50-25_Tracking_CubesmallMulti_CGSmooth_Contact/model_*.pt"),
              key=lambda x: int(x.split("_")[-1][:-3]))[-1]
W, H = 512, 384
CANDS = [
  ("lift", "ori_grab_s2_cubesmall_lift"), ("lift", "ori_grab_s6_cubesmall_lift"),
  ("pass", "ori_grab_s5_cubesmall_pass_1"), ("pass", "ori_grab_s10_cubesmall_pass_1"),
  ("inspect", "ori_grab_s1_cubesmall_inspect_1"), ("inspect", "ori_grab_s10_cubesmall_inspect_1"),
  ("offhand", "ori_grab_s6_cubesmall_offhand_1"), ("offhand", "ori_grab_s8_cubesmall_offhand_1"),
]
OUTDIR = "/data/home/liangheng/DexTrack/videos/cube_grid"
os.makedirs(OUTDIR, exist_ok=True)
print("CKPT", CKPT, "res", W, H)


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
    self.obj.write_root_state_to_sim(torch.cat([ref_pos, ref_quat, vel], dim=-1), env_ids=env_ids)
    self.obj.reset(env_ids=env_ids)
  return _resample


env_cfg, agent_cfg = prepare_task_cfgs(TASK, [], play=True)
env_cfg.scene.num_envs = 1
env_cfg.viewer.width = W
env_cfg.viewer.height = H
base = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array")
env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
r = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cuda:0")
r.load(CKPT, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
pol = r.get_inference_policy(device="cuda:0")
cmd = base.command_manager.get_term("motion")
mf = list(cmd.cfg.motion_files)

for act, nm in CANDS:
  try:
    idx = next(i for i, m in enumerate(mf) if nm in m)
  except StopIteration:
    print(f"SKIP {nm} (not found)"); continue
  ref_traj = cmd._ref_obj_pos[idx]
  disp = torch.linalg.norm(ref_traj - ref_traj[:1], dim=-1)
  peak_t = int(disp.argmax().item())
  cmd._resample_command = types.MethodType(make_locked(idx), cmd)
  base.reset()
  obs = env.get_observations()
  frames, err_peak = [], None
  for t in range(300):
    obs, _, _, _ = env.step(pol(obs))
    img = base.render()
    if img is not None: frames.append(np.asarray(img))
    if t == peak_t:
      err_peak = float(torch.linalg.norm(cmd.obj_pos[0] - cmd.ref_obj_pos[0]))
  out = f"{OUTDIR}/{act}__{nm}.mp4"
  imageio.mimwrite(out, frames, fps=30, quality=9)
  ok = err_peak < 0.05
  print(f"WROTE {act:<8} {nm:<32} peakErr={err_peak:.3f} {'OK' if ok else 'FAIL'}")

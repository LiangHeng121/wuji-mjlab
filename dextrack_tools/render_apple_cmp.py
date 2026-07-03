"""Side-by-side rollout video: apple specialist (hovers at ground) vs 3obj
generalist (lifts the apple), same locked apple-lift sequence.

Left  = apple specialist 8000env kp1  (single-object apple env)
Right = 3obj generalist (park)         (active=apple ref, cube/cup parked far)

Both panels lock the SAME sequence via a monkeypatched _resample_command; the
3obj panel PARKS the non-active objects (cmd._park[j]) so they don't pollute the
grasp. Object z is overlaid each frame. Usage:
  render_apple_cmp.py <out_prefix> [seq1 seq2 ...]
cwd = wuji-mjlab.  Env: MUJOCO_GL=egl PYOPENGL_PLATFORM=egl, idle GPU.
"""
from __future__ import annotations
import glob
import sys
import types
from dataclasses import asdict

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

OUT_PREFIX = sys.argv[1] if len(sys.argv) > 1 else "/data/home/liangheng/DexTrack/mjlab_apple_cmp"
SEQ_NAMES = sys.argv[2:] or ["ori_grab_s2_apple_lift", "ori_grab_s4_apple_lift"]
CK = "logs/rsl_rl/wuji_tracking"


def latest_model(dir_glob: str) -> str:
  dirs = sorted(glob.glob(dir_glob))
  assert dirs, f"no dir matches {dir_glob}"
  d = dirs[-1]
  models = glob.glob(f"{d}/model_*.pt")
  assert models, f"no model in {d}"
  return max(models, key=lambda p: int(p.split("model_")[-1].split(".pt")[0]))


# label, task, ckpt, is_3obj  (pin the same ckpts used for the figures)
PANELS = [
  ("apple specialist (8000env kp1)", "WujiHand_Tracking_AppleMulti_CGSmooth_Contact",
   latest_model(f"{CK}/*AppleMulti_CGSmooth_Contact_Env8000*"), False),
  ("3obj generalist (park)", "WujiHand_Tracking_3Obj_CGSmooth_Contact",
   latest_model(f"{CK}/2026-06-21_06-50-25_Tracking_3Obj_CGSmooth_Contact"), True),
]
NSTEP = 300


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


def label(frame, lines):
  img = Image.fromarray(np.ascontiguousarray(frame))
  d = ImageDraw.Draw(img)
  d.rectangle([0, 0, img.width, 40], fill=(0, 0, 0))
  d.text((6, 3), lines[0], fill=(255, 255, 255))
  d.text((6, 21), lines[1], fill=(120, 230, 120))
  return np.asarray(img)


# frames[panel_label][seq] = [frame,...]
frames = {lbl: {} for lbl, *_ in PANELS}
for lbl, task, ckpt, is_3obj in PANELS:
  print(f"[{lbl}] {ckpt}", flush=True)
  env_cfg, agent_cfg = prepare_task_cfgs(task, [], play=True)
  env_cfg.scene.num_envs = 1
  base = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array")
  env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
  r = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cuda:0")
  r.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  pol = r.get_inference_policy(device="cuda:0")
  cmd = base.command_manager.get_term("motion")
  mf = list(cmd.cfg.motion_files)
  for nm in SEQ_NAMES:
    idx = next(i for i, m in enumerate(mf) if nm in m)
    cmd._resample_command = types.MethodType(make_locked(idx, is_3obj), cmd)
    base.reset()
    obs = env.get_observations()
    fr = []
    for _ in range(NSTEP):
      obs, _, _, _ = env.step(pol(obs))
      img = base.render()
      z = float(cmd.obj_pos[0, 2])
      if img is not None:
        fr.append(label(np.asarray(img), [lbl, f"apple z = {z:.2f} m"]))
    frames[lbl][nm] = fr
    print(f"  {nm}: {len(fr)} frames rendered", flush=True)
  del env, base, r
  torch.cuda.empty_cache()

lbls = [p[0] for p in PANELS]
for nm in SEQ_NAMES:
  a, b = frames[lbls[0]][nm], frames[lbls[1]][nm]
  n = min(len(a), len(b))
  h = max(a[0].shape[0], b[0].shape[0])

  def padh(f):
    if f.shape[0] == h:
      return f
    pad = np.zeros((h - f.shape[0], f.shape[1], f.shape[2]), dtype=f.dtype)
    return np.concatenate([f, pad], axis=0)

  sep = np.full((h, 4, 3), 255, dtype=np.uint8)
  combined = [np.concatenate([padh(a[i]), sep, padh(b[i])], axis=1) for i in range(n)]
  short = nm.replace("ori_grab_", "").replace("_apple_lift", "")
  out = f"{OUT_PREFIX}_{short}.mp4"
  imageio.mimwrite(out, combined, fps=30, quality=8)
  print(f"WROTE {out}  ({n} frames, left=specialist | right=3obj)", flush=True)

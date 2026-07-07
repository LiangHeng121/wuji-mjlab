"""一起-TOPO generalist: per object pick the best seq of 3 DIFFERENT actions, render
at high res, stitch into 3x3 grid (rows=cube/cup/apple, cols=3 actions) -> GIF+mp4.
MUJOCO_GL=egl."""
from __future__ import annotations
import os
os.environ["WUJI_DATA_VER"] = "GRAB_Tracking_PK_WUJI_TOPO_v1"

import glob, re, types
from dataclasses import asdict
import numpy as np, torch, imageio
try:
  from PIL import Image, ImageDraw, ImageFont
  HAVE_PIL = True
except Exception:
  HAVE_PIL = False

import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from wuji_mjlab.tasks.tracking.config.wuji_hand.env_cfgs import _object_sequences
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

CKPT = sorted(glob.glob("logs/rsl_rl/wuji_tracking/*Gen3obj_MassCur_coef0_TOPO_Env8000/model_*.pt"),
              key=lambda x: int(x.split("_")[-1][:-3]))[-1]
# rows = objects, each with 3 chosen action types (cols)
OBJS = [("CUBE", "WujiHand_Tracking_CubesmallMulti_CGSmooth_Contact", "cubesmall", ["lift", "pass", "inspect"]),
        ("CUP", "WujiHand_Tracking_CupMulti_CGSmooth_Contact", "cup", ["lift", "pour", "drink"]),
        ("APPLE", "WujiHand_Tracking_AppleMulti_CGSmooth_Contact", "apple", ["lift", "pass", "eat"])]
W, H = 480, 360
OUTDIR = "/data/home/liangheng/DexTrack/videos/topo_grid"
os.makedirs(OUTDIR, exist_ok=True)
print("CKPT", CKPT, flush=True)


def locked(li, single):
  def _r(self, ids):
    self.time_steps[ids] = 0
    self.env_seq[ids] = li if single else li[ids]
    self.env_obj[ids] = self.seq_obj[self.env_seq[ids]]
    s, t = self.env_seq[ids], self.time_steps[ids]
    soft = self.robot.data.soft_joint_pos_limits[ids]
    jp = torch.clip(self._ref_qpos[s, t].clone(), soft[:, :, 0], soft[:, :, 1])
    self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids); self.robot.reset(env_ids=ids)
    rp, rq = self._ref_obj_pos[s, t], self._ref_obj_quat[s, t]
    vel = torch.zeros(len(ids), 6, device=self.device)
    self.obj.write_root_state_to_sim(torch.cat([rp, rq, vel], dim=-1), env_ids=ids); self.obj.reset(env_ids=ids)
  return _r


def build(task, n, render):
  cfg, ag = prepare_task_cfgs(task, [], play=True)
  cfg.scene.num_envs = n
  if render: cfg.viewer.width, cfg.viewer.height = W, H
  base = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array" if render else None)
  env = RslRlVecEnvWrapper(base, clip_actions=ag.clip_actions)
  r = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(ag), device="cuda:0")
  r.load(CKPT, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  return base, env, r, r.get_inference_policy(device="cuda:0")


tiles = {}  # (label, col) -> frames
for label, task, objkey, actions in OBJS:
  seqs = _object_sequences(objkey)
  base, env, r, pol = build(task, len(seqs), render=False)
  cmd = base.command_manager.get_term("motion"); mf = list(cmd.cfg.motion_files)
  idx = torch.tensor([next(i for i, m in enumerate(mf) if nm in m) for nm in seqs], device="cuda:0")
  rt = cmd._ref_obj_pos[idx]; peak_t = torch.linalg.norm(rt - rt[:, :1], dim=-1).argmax(dim=1)
  cmd._resample_command = types.MethodType(locked(idx, False), cmd)
  base.reset(); obs = env.get_observations(); errp = torch.zeros(len(seqs), device="cuda:0")
  for t in range(300):
    obs, _, _, _ = env.step(pol(obs))
    e = torch.linalg.norm(cmd.obj_pos - cmd.ref_obj_pos, dim=-1); errp = torch.where(peak_t == t, e, errp)
  # best seq per requested action
  chosen = []
  for act in actions:
    cand = [(seqs[i], float(errp[i])) for i in range(len(seqs)) if re.search(objkey + f"_{act}(_|$)", seqs[i])]
    cand.sort(key=lambda x: x[1])
    chosen.append((act, cand[0][0], cand[0][1]) if cand else (act, None, 9))
  print(f"{label}: {[(a, s, round(e,3)) for a,s,e in chosen]}", flush=True)
  del env, base, r; torch.cuda.empty_cache()
  base, env, r, pol = build(task, 1, render=True)
  cmd = base.command_manager.get_term("motion"); mf = list(cmd.cfg.motion_files)
  for col, (act, seq, err) in enumerate(chosen):
    if seq is None: tiles[(label, col)] = ([np.zeros((H, W, 3), np.uint8)] * 300, act); continue
    i = next(k for k, m in enumerate(mf) if seq in m)
    cmd._resample_command = types.MethodType(locked(i, True), cmd)
    base.reset(); obs = env.get_observations(); fr = []
    for _ in range(300):
      obs, _, _, _ = env.step(pol(obs)); img = base.render()
      if img is not None: fr.append(np.asarray(img)[:, :, :3])
    tiles[(label, col)] = (fr, act)
    imageio.mimwrite(f"{OUTDIR}/{label}_{act}_{seq}.mp4", fr, fps=30, quality=9)
    print(f"  rendered {label} {act} {seq} err={err:.3f}", flush=True)
  del env, base, r; torch.cuda.empty_cache()


def lab(img, txt):
  if not HAVE_PIL: return img
  im = Image.fromarray(img).convert("RGB"); d = ImageDraw.Draw(im)
  try: f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
  except Exception: f = ImageFont.load_default()
  d.rectangle([0, 0, 200, 30], fill=(0, 0, 0)); d.text((6, 3), txt, fill=(255, 255, 255), font=f)
  return np.asarray(im)


n = min(len(v[0]) for v in tiles.values())
out = []
for t in range(0, n, 2):
  rows = []
  for label, _, _, _ in OBJS:
    row = [lab(tiles[(label, c)][0][t].copy(), f"{label} {tiles[(label, c)][1]}") for c in range(3)]
    rows.append(np.concatenate(row, axis=1))
  out.append(np.concatenate(rows, axis=0))
gif = f"{OUTDIR}/topo_success_grid.gif"
imageio.mimwrite(gif, out, fps=20, loop=0)
imageio.mimwrite(f"{OUTDIR}/topo_success_grid.mp4", out, fps=20, quality=9)
print(f"WROTE {gif}  {out[0].shape[1]}x{out[0].shape[0]}  {len(out)}帧  {os.path.getsize(gif)/1e6:.1f}MB", flush=True)

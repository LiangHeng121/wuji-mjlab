"""一起-FPOS generalist: per object pick 2 best (lowest peak-displacement err) seqs,
render at high res, stitch into 3-col (cube/cup/apple) x 2-row grid -> GIF + mp4.
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl."""
from __future__ import annotations
import os
os.environ["WUJI_DATA_VER"] = "GRAB_Tracking_PK_WUJI_FPOS_v1"

import glob, types
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
from wuji_mjlab.tasks.tracking.mdp.rewards import n_finger_contacts
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

CKPT = sorted(glob.glob("logs/rsl_rl/wuji_tracking/*Gen3obj_MassCur_coef0_FPOS_Env8000/model_*.pt"),
              key=lambda x: int(x.split("_")[-1][:-3]))[-1]
OBJS = [("CUBE", "WujiHand_Tracking_CubesmallMulti_CGSmooth_Contact", "cubesmall"),
        ("CUP", "WujiHand_Tracking_CupMulti_CGSmooth_Contact", "cup"),
        ("APPLE", "WujiHand_Tracking_AppleMulti_CGSmooth_Contact", "apple")]
# 定稿判据(和 eval_model.py 一致)
POS_TH, FRAC_TH, MIN_C = 0.05, 0.90, 2
LIFT_RATIO, FLY_MARGIN = 0.5, 0.30
EXCLUDE = {"ori_grab_s8_cubesmall_pass_1", "ori_grab_s4_cup_drink_2", "ori_grab_s8_cup_drink_1",
           "ori_grab_s4_cup_pour_1", "ori_grab_s7_cup_pour_1", "ori_grab_s2_apple_eat_1"}
W, H = 512, 384
OUTDIR = "/data/home/liangheng/DexTrack/videos/fpos_grid"
os.makedirs(OUTDIR, exist_ok=True)
print("CKPT", CKPT, flush=True)


def quat_deg(a, b):
  return torch.rad2deg(2 * torch.arccos((a * b).sum(-1).abs().clamp(max=1.0)))


def locked(lock_idx, single):
  def _resample(self, env_ids):
    self.time_steps[env_ids] = 0
    self.env_seq[env_ids] = lock_idx if single else lock_idx[env_ids]
    self.env_obj[env_ids] = self.seq_obj[self.env_seq[env_ids]]
    s, t = self.env_seq[env_ids], self.time_steps[env_ids]
    soft = self.robot.data.soft_joint_pos_limits[env_ids]
    jp = torch.clip(self._ref_qpos[s, t].clone(), soft[:, :, 0], soft[:, :, 1])
    self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)
    rp, rq = self._ref_obj_pos[s, t], self._ref_obj_quat[s, t]
    vel = torch.zeros(len(env_ids), 6, device=self.device)
    self.obj.write_root_state_to_sim(torch.cat([rp, rq, vel], dim=-1), env_ids=env_ids)
    self.obj.reset(env_ids=env_ids)
  return _resample


def build(task, num_envs, render):
  cfg, agent = prepare_task_cfgs(task, [], play=True)
  cfg.scene.num_envs = num_envs
  if render:
    cfg.viewer.width, cfg.viewer.height = W, H
  base = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array" if render else None)
  env = RslRlVecEnvWrapper(base, clip_actions=agent.clip_actions)
  r = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent), device="cuda:0")
  r.load(CKPT, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  return base, env, r, r.get_inference_policy(device="cuda:0")


tiles = {}   # (label, rank) -> (frames, seq, act, err)
for label, task, objkey in OBJS:
  seqs = [s for s in _object_sequences(objkey) if s not in EXCLUDE]
  # --- pass 1: finalized metric (frac>=0.90 + 举高), same as eval_model.py ---
  base, env, r, pol = build(task, len(seqs), render=False)
  cmd = base.command_manager.get_term("motion")
  mf = list(cmd.cfg.motion_files)
  idx = torch.tensor([next(i for i, m in enumerate(mf) if nm in m) for nm in seqs], device="cuda:0")
  N = len(seqs)
  rtp = cmd._ref_obj_pos[idx]; rtq = cmd._ref_obj_quat[idx]
  disp = torch.linalg.norm(rtp - rtp[:, :1], dim=-1)   # [n, T]
  rotd = quat_deg(rtq.reshape(-1, 4), rtq[:, :1].expand(-1, rtq.shape[1], -1).reshape(-1, 4)).reshape(N, -1)
  active = (disp > 0.02) | (rotd > 15.0)
  disp_max = disp.max(dim=1).values                    # per-seq max object travel
  ref_z = (rtp[:, :, 2] - rtp[:, :1, 2]).max(dim=1).values
  cmd._resample_command = types.MethodType(locked(idx, False), cmd)
  base.reset(); obs = env.get_observations()
  ok_cnt = torch.zeros(N, device="cuda:0"); obj0 = cmd.obj_pos.clone(); act_z = torch.zeros(N, device="cuda:0")
  for t in range(300):
    obs, _, _, _ = env.step(pol(obs))
    pe = torch.linalg.norm(cmd.obj_pos - cmd.ref_obj_pos, dim=-1)
    ok_cnt += ((pe < POS_TH) & (n_finger_contacts(base) >= MIN_C) & active[:, t]).float()
    act_z = torch.maximum(act_z, cmd.obj_pos[:, 2] - obj0[:, 2])
  n_act = active[:, :300].sum(1).clamp(min=1)
  frac = ok_cnt / n_act.float()
  followed = (act_z >= LIFT_RATIO * ref_z) & (act_z <= ref_z + FLY_MARGIN)
  succ = (frac >= FRAC_TH) & followed & (active[:, :300].sum(1) >= 10)
  # among genuine successes, pick the 2 with the LARGEST object travel (impressive).
  cand = [(i, float(disp_max[i])) for i in range(N) if bool(succ[i])]
  cand.sort(key=lambda x: -x[1])
  best2 = [(seqs[i], float(frac[i])) for i, _ in cand[:2]]
  print(f"{label} 成功{int(succ.sum())}/{N}  best2(by travel): {[(s, round(e,2)) for s,e in best2]}", flush=True)
  del env, base, r; torch.cuda.empty_cache()
  # --- pass 2: render best 2 at high res ---
  base, env, r, pol = build(task, 1, render=True)
  cmd = base.command_manager.get_term("motion")
  mf = list(cmd.cfg.motion_files)
  for rank, (seq, err) in enumerate(best2):
    i = next(k for k, m in enumerate(mf) if seq in m)
    cmd._resample_command = types.MethodType(locked(i, True), cmd)
    base.reset(); obs = env.get_observations()
    frames = []
    for _ in range(300):
      obs, _, _, _ = env.step(pol(obs))
      img = base.render()
      if img is not None: frames.append(np.asarray(img)[:, :, :3])
    act = seq.split(objkey + "_")[-1].rsplit("_", 1)[0]
    tiles[(label, rank)] = (frames, seq, act, err)
    imageio.mimwrite(f"{OUTDIR}/{label}_{rank}_{seq}.mp4", frames, fps=30, quality=9)
    print(f"  rendered {label} r{rank} {seq} err={err:.3f}", flush=True)
  del env, base, r; torch.cuda.empty_cache()


def lab(img, txt):
  if not HAVE_PIL: return img
  im = Image.fromarray(img).convert("RGB"); d = ImageDraw.Draw(im)
  try: f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
  except Exception: f = ImageFont.load_default()
  d.rectangle([0, 0, 190, 32], fill=(0, 0, 0)); d.text((8, 4), txt, fill=(255, 255, 255), font=f)
  return np.asarray(im)


n = min(len(v[0]) for v in tiles.values())
out = []
for t in range(0, n, 2):
  rows = []
  for rank in (0, 1):
    row = []
    for label, _, _ in OBJS:
      fr, seq, act, err = tiles[(label, rank)]
      row.append(lab(fr[t].copy(), f"{label} {act}"))
    rows.append(np.concatenate(row, axis=1))
  out.append(np.concatenate(rows, axis=0))
gif = f"{OUTDIR}/fpos_success_grid.gif"
imageio.mimwrite(gif, out, fps=20, loop=0)
imageio.mimwrite(f"{OUTDIR}/fpos_success_grid.mp4", out, fps=20, quality=9)
print(f"WROTE {gif}  {out[0].shape[1]}x{out[0].shape[0]}  {len(out)}帧  {os.path.getsize(gif)/1e6:.1f}MB", flush=True)

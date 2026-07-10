"""9obj TOPO generalist: per object pick 1 best success (frac>=0.90, largest object
travel), render high-res, stitch 3x3 grid. MUJOCO_GL=egl."""
from __future__ import annotations
import os, sys, glob, types, math
os.environ["WUJI_DATA_VER"] = "GRAB_Tracking_PK_WUJI_TOPO_v1"
from dataclasses import asdict
import numpy as np, torch, imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import wuji_mjlab.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from wuji_mjlab.tasks.tracking.config.wuji_hand.env_cfgs import _object_sequences
from wuji_mjlab.tasks.tracking.mdp.rewards import n_finger_contacts
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

CKPT = sorted(glob.glob("logs/rsl_rl/wuji_tracking/*Gen9obj_coef0_TOPO_Env2700/model_*.pt"),
              key=lambda x: int(x.split("_")[-1][:-3]))[-1]
OBJS = [("cube", "WujiHand_Tracking_CubesmallMulti_CGSmooth_Contact", "cubesmall"),
        ("cup", "WujiHand_Tracking_CupMulti_CGSmooth_Contact", "cup"),
        ("duck", "WujiHand_Tracking_DuckMulti_CGSmooth_Contact", "duck"),
        ("elephant", "WujiHand_Tracking_ElephantMulti_CGSmooth_Contact", "elephant"),
        ("mouse", "WujiHand_Tracking_MouseMulti_CGSmooth_Contact", "mouse"),
        ("phone", "WujiHand_Tracking_PhoneMulti_CGSmooth_Contact", "phone"),
        ("train", "WujiHand_Tracking_TrainMulti_CGSmooth_Contact", "train"),
        ("alarmclock", "WujiHand_Tracking_AlarmclockMulti_CGSmooth_Contact", "alarmclock"),
        ("flute", "WujiHand_Tracking_FluteMulti_CGSmooth_Contact", "flute")]
POS_TH, FRAC_TH, MIN_C, LIFT_RATIO, FLY_MARGIN = 0.05, 0.90, 2, 0.5, 0.30
W, H = 480, 360
OUTDIR = "/data/home/liangheng/DexTrack/videos/0709"
os.makedirs(OUTDIR, exist_ok=True)
print("CKPT", CKPT, flush=True)


def quat_deg(a, b): return torch.rad2deg(2 * torch.arccos((a * b).sum(-1).abs().clamp(max=1.0)))


def locked(li, single):
  def _r(self, ids):
    self.time_steps[ids] = 0
    self.env_seq[ids] = li if single else li[ids]; self.env_obj[ids] = self.seq_obj[self.env_seq[ids]]
    s, t = self.env_seq[ids], self.time_steps[ids]; soft = self.robot.data.soft_joint_pos_limits[ids]
    jp = torch.clip(self._ref_qpos[s, t].clone(), soft[:, :, 0], soft[:, :, 1])
    self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids); self.robot.reset(env_ids=ids)
    rp, rq = self._ref_obj_pos[s, t], self._ref_obj_quat[s, t]; vel = torch.zeros(len(ids), 6, device=self.device)
    self.obj.write_root_state_to_sim(torch.cat([rp, rq, vel], dim=-1), env_ids=ids); self.obj.reset(env_ids=ids)
  return _r


def build(task, n, render):
  cfg, ag = prepare_task_cfgs(task, [], play=True); cfg.scene.num_envs = n
  if render: cfg.viewer.width, cfg.viewer.height = W, H
  base = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array" if render else None)
  env = RslRlVecEnvWrapper(base, clip_actions=ag.clip_actions)
  r = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(ag), device="cuda:0")
  r.load(CKPT, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  return base, env, r, r.get_inference_policy(device="cuda:0")


def fnt(s):
  try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", s)
  except Exception: return ImageFont.load_default()


tiles = []
for oname, task, objkey in OBJS:
  seqs = _object_sequences(objkey); N = len(seqs)
  base, env, r, pol = build(task, N, False)
  cmd = base.command_manager.get_term("motion"); mf = list(cmd.cfg.motion_files)
  idx = torch.tensor([next(i for i, m in enumerate(mf) if nm in m) for nm in seqs], device="cuda:0")
  rtp = cmd._ref_obj_pos[idx]; rtq = cmd._ref_obj_quat[idx]
  disp = torch.linalg.norm(rtp - rtp[:, :1], dim=-1); disp_max = disp.max(dim=1).values
  rotd = quat_deg(rtq.reshape(-1, 4), rtq[:, :1].expand(-1, rtq.shape[1], -1).reshape(-1, 4)).reshape(N, -1)
  active = (disp > 0.02) | (rotd > 15.0)
  ref_z = (rtp[:, :, 2] - rtp[:, :1, 2]).max(dim=1).values
  cmd._resample_command = types.MethodType(locked(idx, False), cmd)
  base.reset(); obs = env.get_observations()
  ok = torch.zeros(N, device="cuda:0"); obj0 = cmd.obj_pos.clone(); az = torch.zeros(N, device="cuda:0")
  for t in range(300):
    obs, _, _, _ = env.step(pol(obs))
    pe = torch.linalg.norm(cmd.obj_pos - cmd.ref_obj_pos, dim=-1)
    ok += ((pe < POS_TH) & (n_finger_contacts(base) >= MIN_C) & active[:, t]).float()
    az = torch.maximum(az, cmd.obj_pos[:, 2] - obj0[:, 2])
  frac = ok / active[:, :300].sum(1).clamp(min=1).float()
  followed = (az >= LIFT_RATIO * ref_z) & (az <= ref_z + FLY_MARGIN)
  succ = (frac >= FRAC_TH) & followed & (active[:, :300].sum(1) >= 10)
  cand = [(i, float(disp_max[i])) for i in range(N) if bool(succ[i])]
  cand.sort(key=lambda x: -x[1])
  pick = seqs[cand[0][0]] if cand else seqs[int(disp_max.argmax())]  # fallback: largest travel
  print(f"{oname}: pick {pick} (成功{int(succ.sum())}/{N})", flush=True)
  del env, base, r; torch.cuda.empty_cache()
  # render the pick high-res
  base, env, r, pol = build(task, 1, True)
  cmd = base.command_manager.get_term("motion"); mf = list(cmd.cfg.motion_files)
  i = next(k for k, m in enumerate(mf) if pick in m)
  cmd._resample_command = types.MethodType(locked(i, True), cmd)
  base.reset(); obs = env.get_observations(); fr = []
  for _ in range(300):
    obs, _, _, _ = env.step(pol(obs)); img = base.render()
    if img is not None:
      a = np.asarray(img)[:, :, :3].copy(); pim = Image.fromarray(a); d = ImageDraw.Draw(pim)
      d.rectangle([0, 0, W, 24], fill=(0, 0, 0)); d.text((5, 3), oname, fill=(150, 255, 150), font=fnt(18))
      fr.append(np.asarray(pim))
  tiles.append(fr)
  del env, base, r; torch.cuda.empty_cache()

n = min(len(t) for t in tiles); cols = 3; rows = 3
blank = np.zeros((H, W, 3), np.uint8); out = []
for t in range(n):
  grid = []
  for rr in range(rows):
    row = [tiles[rr * cols + cc][t] if rr * cols + cc < len(tiles) else blank for cc in range(cols)]
    grid.append(np.concatenate(row, axis=1))
  out.append(np.concatenate(grid, axis=0))
imageio.mimwrite(f"{OUTDIR}/9obj_success_grid.mp4", out, fps=30, quality=8)
imageio.mimwrite(f"{OUTDIR}/9obj_success_grid.gif", out[::2], fps=15, loop=0)
print(f"WROTE {OUTDIR}/9obj_success_grid.mp4 {out[0].shape[1]}x{out[0].shape[0]} {len(out)}f", flush=True)
os._exit(0)

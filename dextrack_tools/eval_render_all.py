"""一起-TOPO: eval ALL seqs of one object with the FINAL metric (frac>=0.50 AND
height-gain>=0.15), then render every seq labeled PASS/FAIL f=.. h=.. into one grid.
Usage: eval_render_all.py <object>   (object=cubesmall|cup|apple). MUJOCO_GL=egl."""
from __future__ import annotations
import os, sys, math, glob, types
OBJ = sys.argv[1]
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

TASKS = {"cubesmall": "WujiHand_Tracking_CubesmallMulti_CGSmooth_Contact",
         "cup": "WujiHand_Tracking_CupMulti_CGSmooth_Contact",
         "apple": "WujiHand_Tracking_AppleMulti_CGSmooth_Contact"}
CKPT = sorted(glob.glob("logs/rsl_rl/wuji_tracking/*Gen3obj_MassCur_coef0_TOPO_Env8000/model_*.pt"),
              key=lambda x: int(x.split("_")[-1][:-3]))[-1]
POS_TH, FRAC_TH, MIN_C = 0.05, 0.90, 2
LIFT_RATIO, FLY_MARGIN = 0.5, 0.30
EXCLUDE = {"ori_grab_s8_cubesmall_pass_1", "ori_grab_s4_cup_drink_2", "ori_grab_s8_cup_drink_1",
           "ori_grab_s4_cup_pour_1", "ori_grab_s7_cup_pour_1", "ori_grab_s2_apple_eat_1"}
W, H = 320, 240
task = TASKS[OBJ]; seqs = [s for s in _object_sequences(OBJ) if s not in EXCLUDE]; N = len(seqs)
OUTDIR = "/data/home/liangheng/DexTrack/videos/metric_all"; os.makedirs(OUTDIR, exist_ok=True)


def quat_deg(a, b):
  return torch.rad2deg(2 * torch.arccos((a * b).sum(-1).abs().clamp(max=1.0)))


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


def build(n, render):
  cfg, ag = prepare_task_cfgs(task, [], play=True); cfg.scene.num_envs = n
  if render: cfg.viewer.width, cfg.viewer.height = W, H
  base = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array" if render else None)
  env = RslRlVecEnvWrapper(base, clip_actions=ag.clip_actions)
  r = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(ag), device="cuda:0")
  r.load(CKPT, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  return base, env, r, r.get_inference_policy(device="cuda:0")


# ---- Phase 1: eval all seqs (batched) ----
base, env, r, pol = build(N, False)
cmd = base.command_manager.get_term("motion"); mf = list(cmd.cfg.motion_files)
idx = torch.tensor([next(i for i, m in enumerate(mf) if nm in m) for nm in seqs], device="cuda:0")
rtp = cmd._ref_obj_pos[idx]; rtq = cmd._ref_obj_quat[idx]
disp = torch.linalg.norm(rtp - rtp[:, :1], dim=-1)
rotd = quat_deg(rtq.reshape(-1, 4), rtq[:, :1].expand(-1, rtq.shape[1], -1).reshape(-1, 4)).reshape(N, -1)
active = (disp > 0.02) | (rotd > 15.0)
ref_z = (rtp[:, :, 2] - rtp[:, :1, 2]).max(dim=1).values.cpu()
cmd._resample_command = types.MethodType(locked(idx, False), cmd)
base.reset(); obs = env.get_observations()
ok_cnt = torch.zeros(N, device="cuda:0"); obj0 = cmd.obj_pos.clone(); act_z = torch.zeros(N, device="cuda:0")
for t in range(300):
  obs, _, _, _ = env.step(pol(obs))
  pe = torch.linalg.norm(cmd.obj_pos - cmd.ref_obj_pos, dim=-1)
  ok_cnt += ((pe < POS_TH) & (n_finger_contacts(base) >= MIN_C) & active[:, t]).float()
  act_z = torch.maximum(act_z, cmd.obj_pos[:, 2] - obj0[:, 2])
n_act = active[:, :300].sum(1).clamp(min=1)
frac = (ok_cnt / n_act.float()).cpu(); az = act_z.cpu()
followed = (az >= LIFT_RATIO * ref_z) & (az <= ref_z + FLY_MARGIN)
succ = (frac >= FRAC_TH) & followed & (active[:, :300].sum(1).cpu() >= 10)
nsucc = int(succ.sum())
print(f"[{OBJ}] {nsucc}/{N}", flush=True)
del env, base, r; torch.cuda.empty_cache()

# ---- Phase 2: render each seq labeled ----
base, env, r, pol = build(1, True)
cmd = base.command_manager.get_term("motion"); mf = list(cmd.cfg.motion_files)


def lab(img, txt, ok):
  im = Image.fromarray(np.ascontiguousarray(img[:, :, :3])).convert("RGB"); d = ImageDraw.Draw(im)
  try: f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
  except Exception: f = ImageFont.load_default()
  col = (150, 255, 150) if ok else (255, 130, 130)
  d.rectangle([0, 0, W, 18], fill=(0, 0, 0)); d.text((3, 2), txt, fill=col, font=f)
  return np.asarray(im)


tiles = []
for k, seq in enumerate(seqs):
  i = next(j for j, m in enumerate(mf) if seq in m)
  cmd._resample_command = types.MethodType(locked(i, True), cmd)
  base.reset(); obs = env.get_observations(); fr = []
  short = seq.replace("ori_grab_", "").replace(f"_{OBJ}", "").replace("cubesmall", "")
  txt = f"{short} {'P' if bool(succ[k]) else 'F'} f{float(frac[k]):.2f} h{float(az[k]):.2f}"
  for _ in range(300):
    obs, _, _, _ = env.step(pol(obs)); img = base.render()
    if img is not None: fr.append(lab(np.asarray(img), txt, bool(succ[k])))
  tiles.append(fr); print(f"  rendered {seq} {'P' if bool(succ[k]) else 'F'}", flush=True)

cols = math.ceil(math.sqrt(N)); rows = math.ceil(N / cols)
nfr = min(len(t) for t in tiles)
blank = np.zeros((H, W, 3), np.uint8)
out = []
for t in range(0, nfr, 2):
  grid = []
  for rr in range(rows):
    rowimg = [tiles[rr * cols + cc][t] if rr * cols + cc < N else blank for cc in range(cols)]
    grid.append(np.concatenate(rowimg, axis=1))
  out.append(np.concatenate(grid, axis=0))
gif = f"{OUTDIR}/{OBJ}_clean_grid.gif"
imageio.mimwrite(gif, out, fps=20, loop=0)
imageio.mimwrite(f"{OUTDIR}/{OBJ}_clean_grid.mp4", out, fps=20, quality=8)
print(f"WROTE {gif} {out[0].shape[1]}x{out[0].shape[0]} {len(out)}f {os.path.getsize(gif)/1e6:.1f}MB  {nsucc}/{N}", flush=True)
os._exit(0)

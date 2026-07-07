"""Retarget comparison 6-grid: kinematic replay of FPOS_v1 vs TopoRetarget for 3
OBJECTS (cube/cup/apple, one lift seq each). Grid: 3 cols (objects) x 2 rows (FPOS
top / TOPO bottom). Teleport hand+object to reference each frame (no physics).
Keeps envs alive + os._exit to dodge the EGL destroy bug. -> GIF + mp4. MUJOCO_GL=egl."""
from __future__ import annotations
import os
import numpy as np, torch, imageio
try:
  from PIL import Image, ImageDraw, ImageFont
  HAVE_PIL = True
except Exception:
  HAVE_PIL = False

import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

DEX = "/data/home/liangheng/DexTrack"
FPOS_DIR = f"{DEX}/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_FPOS_v1/data"
TOPO_DIR = f"{DEX}/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_TOPO_v1/data"
CONTACT_DIR = f"{DEX}/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_FPOS_v1/contact_grab2"
# (col-label, task, seq)  one lift seq per object
OBJS = [("CUBE", "WujiHand_Tracking_CubesmallMulti_CGSmooth_Contact", "ori_grab_s2_cubesmall_lift"),
        ("CUP", "WujiHand_Tracking_CupMulti_CGSmooth_Contact", "ori_grab_s1_cup_lift"),
        ("APPLE", "WujiHand_Tracking_AppleMulti_CGSmooth_Contact", "ori_grab_s2_apple_lift")]
W, H, NSTEP, DEV = 512, 384, 300, "cuda:0"
OUTDIR = f"{DEX}/videos/retarget_grid"
os.makedirs(OUTDIR, exist_ok=True)


def lab(frame, txt, color):
  if not HAVE_PIL: return frame
  img = Image.fromarray(np.ascontiguousarray(frame)); d = ImageDraw.Draw(img)
  try: f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
  except Exception: f = ImageFont.load_default()
  d.rectangle([0, 0, 250, 32], fill=(0, 0, 0)); d.text((6, 3), txt, fill=color, font=f)
  return np.asarray(img)


envs = []           # keep alive (don't del -> avoid EGL destroy bug)
tiles = {}          # (row, col) -> frames ; row0=FPOS row1=TOPO
for col, (label, task, seq) in enumerate(OBJS):
  env_cfg, _ = prepare_task_cfgs(task, [], play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.viewer.width, env_cfg.viewer.height = W, H
  m = env_cfg.commands["motion"]
  m.motion_files = (f"{FPOS_DIR}/wuji_passive_active_info_{seq}_nf_300.npy",
                    f"{TOPO_DIR}/wuji_passive_active_info_{seq}_nf_300.npy")
  c = f"{CONTACT_DIR}/{seq}_contact.npy"
  m.contact_files = (c, c)
  m.seq_object_idx = (0, 0)
  base = ManagerBasedRlEnv(cfg=env_cfg, device=DEV, render_mode="rgb_array")
  envs.append(base)
  cmd = base.command_manager.get_term("motion")
  robot, obj = cmd.robot, cmd.obj
  ids = torch.tensor([0], device=DEV); Tref = cmd._ref_obj_pos.shape[1]
  base.reset()
  for row, idx in enumerate((0, 1)):
    name = "FPOS (old)" if idx == 0 else "TopoRetarget"
    color = (255, 150, 150) if idx == 0 else (120, 255, 140)
    fr = []
    for t in range(NSTEP):
      tt = min(t, Tref - 1)
      jp = cmd._ref_qpos[idx, tt].unsqueeze(0)
      robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
      root = torch.cat([cmd._ref_obj_pos[idx, tt], cmd._ref_obj_quat[idx, tt],
                        torch.zeros(6, device=DEV)]).unsqueeze(0)
      obj.write_root_state_to_sim(root, env_ids=ids)
      base.sim.forward()
      img = base.render()
      if img is not None:
        fr.append(lab(np.asarray(img)[:, :, :3], f"{label}  {name}", color))
    tiles[(row, col)] = fr
    print(f"  {label} {name}: {len(fr)} frames", flush=True)

n = min(len(v) for v in tiles.values())
out = []
for t in range(0, n, 2):
  rows = [np.concatenate([tiles[(row, c)][t] for c in range(len(OBJS))], axis=1) for row in (0, 1)]
  out.append(np.concatenate(rows, axis=0))
gif = f"{OUTDIR}/retarget_fpos_vs_topo_grid.gif"
imageio.mimwrite(gif, out, fps=20, loop=0)
imageio.mimwrite(f"{OUTDIR}/retarget_fpos_vs_topo_grid.mp4", out, fps=20, quality=9)
print(f"WROTE {gif}  {out[0].shape[1]}x{out[0].shape[0]}  {len(out)}帧  {os.path.getsize(gif)/1e6:.1f}MB", flush=True)
os._exit(0)   # skip interpreter cleanup -> avoid eglDestroyContext EGLError

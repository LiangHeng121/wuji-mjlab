"""Validate no-orient metric on cube+cup: render disagreement/borderline cases,
3x2 grid (row1 cube, row2 cup), labeled with NEW verdict+frac. MUJOCO_GL=egl."""
from __future__ import annotations
import os
os.environ["WUJI_DATA_VER"] = "GRAB_Tracking_PK_WUJI_FPOS_v1"
import glob, types
from dataclasses import asdict
import numpy as np, torch, imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import wuji_mjlab.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

CKPT = sorted(glob.glob("logs/rsl_rl/wuji_tracking/*Gen3obj_MassCur_coef0_FPOS_Env8000/model_*.pt"),
              key=lambda x: int(x.split("_")[-1][:-3]))[-1]
W, H = 512, 384
# (task, seq, label, color)
ROWS = [
  [("WujiHand_Tracking_CubesmallMulti_CGSmooth_Contact", "ori_grab_s2_cubesmall_inspect_1", "cube inspect NEW-FAIL 0.72", (255, 140, 140)),
   ("WujiHand_Tracking_CubesmallMulti_CGSmooth_Contact", "ori_grab_s8_cubesmall_pass_1", "cube pass BORDER", (255, 220, 120)),
   ("WujiHand_Tracking_CubesmallMulti_CGSmooth_Contact", "ori_grab_s9_cubesmall_pass_1", "cube pass BORDER", (255, 220, 120))],
  [("WujiHand_Tracking_CupMulti_CGSmooth_Contact", "ori_grab_s8_cup_lift", "cup lift NEW-FAIL 0.24", (255, 140, 140)),
   ("WujiHand_Tracking_CupMulti_CGSmooth_Contact", "ori_grab_s10_cup_lift", "cup lift BORDER", (255, 220, 120)),
   ("WujiHand_Tracking_CupMulti_CGSmooth_Contact", "ori_grab_s4_cup_pour_1", "cup pour BORDER", (255, 220, 120))],
]
OUTDIR = "/data/home/liangheng/DexTrack/videos/metric_check"
os.makedirs(OUTDIR, exist_ok=True)


def locked(i):
  def _r(self, ids):
    self.time_steps[ids] = 0; self.env_seq[ids] = i; self.env_obj[ids] = self.seq_obj[self.env_seq[ids]]
    s, t = self.env_seq[ids], self.time_steps[ids]; soft = self.robot.data.soft_joint_pos_limits[ids]
    jp = torch.clip(self._ref_qpos[s, t].clone(), soft[:, :, 0], soft[:, :, 1])
    self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids); self.robot.reset(env_ids=ids)
    rp, rq = self._ref_obj_pos[s, t], self._ref_obj_quat[s, t]; vel = torch.zeros(len(ids), 6, device=self.device)
    self.obj.write_root_state_to_sim(torch.cat([rp, rq, vel], dim=-1), env_ids=ids); self.obj.reset(env_ids=ids)
  return _r


def lab(img, txt, color):
  im = Image.fromarray(np.ascontiguousarray(img[:, :, :3])).convert("RGB"); d = ImageDraw.Draw(im)
  try: f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
  except Exception: f = ImageFont.load_default()
  d.rectangle([0, 0, 330, 28], fill=(0, 0, 0)); d.text((6, 3), txt, fill=color, font=f)
  return np.asarray(im)


def render_task(task, cells):
  cfg, ag = prepare_task_cfgs(task, [], play=True); cfg.scene.num_envs = 1
  cfg.viewer.width, cfg.viewer.height = W, H
  base = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
  env = RslRlVecEnvWrapper(base, clip_actions=ag.clip_actions)
  r = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(ag), device="cuda:0")
  r.load(CKPT, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  pol = r.get_inference_policy(device="cuda:0")
  cmd = base.command_manager.get_term("motion"); mf = list(cmd.cfg.motion_files)
  outs = []
  for _, seq, label, color in cells:
    i = next(k for k, m in enumerate(mf) if seq in m)
    cmd._resample_command = types.MethodType(locked(i), cmd)
    base.reset(); obs = env.get_observations(); fr = []
    for _ in range(300):
      obs, _, _, _ = env.step(pol(obs)); img = base.render()
      if img is not None: fr.append(lab(np.asarray(img), label, color))
    outs.append(fr); print(f"rendered {seq}", flush=True)
  return outs


tiles = [render_task(ROWS[0][0][0], ROWS[0]), render_task(ROWS[1][0][0], ROWS[1])]
n = min(len(f) for row in tiles for f in row)
out = []
for t in range(0, n, 2):
  rows = [np.concatenate([tiles[ri][ci][t] for ci in range(3)], axis=1) for ri in range(2)]
  out.append(np.concatenate(rows, axis=0))
gif = f"{OUTDIR}/metric_check2_grid.gif"
imageio.mimwrite(gif, out, fps=20, loop=0)
imageio.mimwrite(f"{OUTDIR}/metric_check2_grid.mp4", out, fps=20, quality=9)
print(f"WROTE {gif} {out[0].shape[1]}x{out[0].shape[0]} {len(out)}f {os.path.getsize(gif)/1e6:.1f}MB", flush=True)
os._exit(0)

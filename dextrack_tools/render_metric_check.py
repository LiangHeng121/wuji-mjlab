"""Render 一起-FPOS on 6 apple seqs spanning new-metric verdicts, 3x2 grid, labeled
with the NEW verdict + active-frame frac, so a human can validate the metric.
MUJOCO_GL=egl."""
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
TASK = "WujiHand_Tracking_AppleMulti_CGSmooth_Contact"
# (seq, label)  row1 = NEW-fail (should look failed), row2 = NEW-pass + borderline
CELLS = [
  ("ori_grab_s1_apple_eat_1", "eat  NEW-FAIL 0.02", (255, 140, 140)),
  ("ori_grab_s2_apple_pass_1", "pass NEW-FAIL 0.29", (255, 140, 140)),
  ("ori_grab_s6_apple_pass_1", "pass NEW-FAIL 0.03", (255, 140, 140)),
  ("ori_grab_s3_apple_pass_1", "pass NEW-OK 0.80", (140, 255, 140)),
  ("ori_grab_s5_apple_pass_1", "pass NEW-OK 0.89", (140, 255, 140)),
  ("ori_grab_s9_apple_lift", "lift BORDER 0.62", (255, 220, 120)),
]
W, H = 512, 384
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


env_cfg, agent_cfg = prepare_task_cfgs(TASK, [], play=True)
env_cfg.scene.num_envs = 1; env_cfg.viewer.width, env_cfg.viewer.height = W, H
base = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array")
env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
r = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cuda:0")
r.load(CKPT, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
pol = r.get_inference_policy(device="cuda:0")
cmd = base.command_manager.get_term("motion"); mf = list(cmd.cfg.motion_files)


def lab(img, txt, color):
  im = Image.fromarray(np.ascontiguousarray(img[:, :, :3])).convert("RGB"); d = ImageDraw.Draw(im)
  try: f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
  except Exception: f = ImageFont.load_default()
  d.rectangle([0, 0, 300, 30], fill=(0, 0, 0)); d.text((6, 3), txt, fill=color, font=f)
  return np.asarray(im)


tiles = []
for seq, label, color in CELLS:
  i = next(k for k, m in enumerate(mf) if seq in m)
  cmd._resample_command = types.MethodType(locked(i), cmd)
  base.reset(); obs = env.get_observations(); fr = []
  for _ in range(300):
    obs, _, _, _ = env.step(pol(obs)); img = base.render()
    if img is not None: fr.append(lab(np.asarray(img), label, color))
  tiles.append(fr); print(f"rendered {seq}", flush=True)

n = min(len(t) for t in tiles)
out = []
for t in range(0, n, 2):
  r1 = np.concatenate([tiles[0][t], tiles[1][t], tiles[2][t]], axis=1)
  r2 = np.concatenate([tiles[3][t], tiles[4][t], tiles[5][t]], axis=1)
  out.append(np.concatenate([r1, r2], axis=0))
gif = f"{OUTDIR}/metric_check_grid.gif"
imageio.mimwrite(gif, out, fps=20, loop=0)
imageio.mimwrite(f"{OUTDIR}/metric_check_grid.mp4", out, fps=20, quality=9)
print(f"WROTE {gif} {out[0].shape[1]}x{out[0].shape[0]} {len(out)}f {os.path.getsize(gif)/1e6:.1f}MB", flush=True)
os._exit(0)

"""Re-render CUP & APPLE lift tiles with the HIGHEST-lifted seq, re-stitch TOPO 9-grid."""
from __future__ import annotations
import os
os.environ["WUJI_DATA_VER"] = "GRAB_Tracking_PK_WUJI_TOPO_v1"
import glob, re, types
from dataclasses import asdict
import numpy as np, torch, imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import wuji_mjlab.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from wuji_mjlab.tasks.tracking.config.wuji_hand.env_cfgs import _object_sequences
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

CKPT = sorted(glob.glob("logs/rsl_rl/wuji_tracking/*Gen3obj_MassCur_coef0_TOPO_Env8000/model_*.pt"),
              key=lambda x: int(x.split("_")[-1][:-3]))[-1]
W, H = 480, 360
D = "/data/home/liangheng/DexTrack/videos/topo_grid"


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
  cfg, ag = prepare_task_cfgs(task, [], play=True); cfg.scene.num_envs = n
  if render: cfg.viewer.width, cfg.viewer.height = W, H
  base = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array" if render else None)
  env = RslRlVecEnvWrapper(base, clip_actions=ag.clip_actions)
  r = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(ag), device="cuda:0")
  r.load(CKPT, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  return base, env, r, r.get_inference_policy(device="cuda:0")


new = {}
for label, task, objkey in [("CUP", "WujiHand_Tracking_CupMulti_CGSmooth_Contact", "cup"),
                            ("APPLE", "WujiHand_Tracking_AppleMulti_CGSmooth_Contact", "apple")]:
  seqs = [s for s in _object_sequences(objkey) if re.search(objkey + r"_lift(_|$)", s)]
  base, env, r, pol = build(task, len(seqs), False)
  cmd = base.command_manager.get_term("motion"); mf = list(cmd.cfg.motion_files)
  idx = torch.tensor([next(i for i, m in enumerate(mf) if nm in m) for nm in seqs], device="cuda:0")
  rt = cmd._ref_obj_pos[idx]; peak_t = torch.linalg.norm(rt - rt[:, :1], dim=-1).argmax(dim=1)
  cmd._resample_command = types.MethodType(locked(idx, False), cmd)
  base.reset(); obs = env.get_observations()
  errp = torch.zeros(len(seqs), device="cuda:0"); maxz = torch.full((len(seqs),), -9., device="cuda:0")
  for t in range(300):
    obs, _, _, _ = env.step(pol(obs))
    e = torch.linalg.norm(cmd.obj_pos - cmd.ref_obj_pos, dim=-1); errp = torch.where(peak_t == t, e, errp)
    maxz = torch.maximum(maxz, cmd.obj_pos[:, 2])
  cand = [(i, float(maxz[i])) for i in range(len(seqs)) if float(errp[i]) < 0.05]
  cand.sort(key=lambda x: -x[1]); pick = seqs[cand[0][0]]
  print(f"{label} lift -> {pick} maxz={cand[0][1]:.3f}", flush=True)
  del env, base, r; torch.cuda.empty_cache()
  base, env, r, pol = build(task, 1, True)
  cmd = base.command_manager.get_term("motion"); mf = list(cmd.cfg.motion_files)
  i = next(k for k, m in enumerate(mf) if pick in m)
  cmd._resample_command = types.MethodType(locked(i, True), cmd)
  base.reset(); obs = env.get_observations(); fr = []
  for _ in range(300):
    obs, _, _, _ = env.step(pol(obs)); img = base.render()
    if img is not None: fr.append(np.asarray(img)[:, :, :3])
  new[label] = fr
  del env, base, r; torch.cuda.empty_cache()

# assemble grid: rows objects, cols actions
GRID = [
  [("CUBE lift", glob.glob(f"{D}/CUBE_lift_*.mp4")[0]), ("CUBE pass", glob.glob(f"{D}/CUBE_pass_*.mp4")[0]), ("CUBE inspect", glob.glob(f"{D}/CUBE_inspect_*.mp4")[0])],
  [("CUP lift", None), ("CUP pour", glob.glob(f"{D}/CUP_pour_*.mp4")[0]), ("CUP drink", glob.glob(f"{D}/CUP_drink_*.mp4")[0])],
  [("APPLE lift", None), ("APPLE pass", glob.glob(f"{D}/APPLE_pass_*.mp4")[0]), ("APPLE eat", glob.glob(f"{D}/APPLE_eat_*.mp4")[0])],
]
NEWMAP = {"CUP lift": new["CUP"], "APPLE lift": new["APPLE"]}


def frames_of(txt, path):
  return NEWMAP[txt] if path is None else imageio.mimread(path, memtest=False)


def lab(img, txt):
  im = Image.fromarray(np.ascontiguousarray(img[:, :, :3])).convert("RGB"); d = ImageDraw.Draw(im)
  try: f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
  except Exception: f = ImageFont.load_default()
  d.rectangle([0, 0, 200, 30], fill=(0, 0, 0)); d.text((6, 3), txt, fill=(255, 255, 255), font=f)
  return np.asarray(im)


rows_frames = [[frames_of(t, p) for t, p in row] for row in GRID]
labels = [[t for t, _ in row] for row in GRID]
n = min(len(c) for r in rows_frames for c in r)
out = []
for t in range(0, n, 2):
  rows = []
  for ri, row in enumerate(rows_frames):
    rows.append(np.concatenate([lab(row[ci][t], labels[ri][ci]) for ci in range(3)], axis=1))
  out.append(np.concatenate(rows, axis=0))
gif = f"{D}/topo_success_grid.gif"
imageio.mimwrite(gif, out, fps=20, loop=0)
imageio.mimwrite(f"{D}/topo_success_grid.mp4", out, fps=20, quality=9)
print(f"WROTE {gif} {out[0].shape[1]}x{out[0].shape[0]} {len(out)}f {os.path.getsize(gif)/1e6:.1f}MB", flush=True)
os._exit(0)

"""Actual-vs-reference comparison render for cube s8_pass. Actual (solid) blended
with the reference pose (ghost); frame border RED when object pos error > 5cm."""
from __future__ import annotations
import os
os.environ["WUJI_DATA_VER"] = "GRAB_Tracking_PK_WUJI_TOPO_v1"
import glob, types
from dataclasses import asdict
import numpy as np, torch, imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import wuji_mjlab.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

CKPT = sorted(glob.glob("logs/rsl_rl/wuji_tracking/*Gen3obj_MassCur_coef0_TOPO_Env8000/model_*.pt"),
              key=lambda x: int(x.split("_")[-1][:-3]))[-1]
TASK = os.environ.get("CMPTASK","WujiHand_Tracking_CubesmallMulti_CGSmooth_Contact")
SEQ = os.environ.get("CMPSEQ","ori_grab_s8_cubesmall_pass_1")
N = 300


def locked(i):
  def _r(self, ids):
    self.time_steps[ids] = 0
    self.env_seq[ids] = i; self.env_obj[ids] = self.seq_obj[self.env_seq[ids]]
    s, t = self.env_seq[ids], self.time_steps[ids]; soft = self.robot.data.soft_joint_pos_limits[ids]
    jp = torch.clip(self._ref_qpos[s, t].clone(), soft[:, :, 0], soft[:, :, 1])
    self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids); self.robot.reset(env_ids=ids)
    rp, rq = self._ref_obj_pos[s, t], self._ref_obj_quat[s, t]; vel = torch.zeros(len(ids), 6, device=self.device)
    self.obj.write_root_state_to_sim(torch.cat([rp, rq, vel], dim=-1), env_ids=ids); self.obj.reset(env_ids=ids)
  return _r


cfg, ag = prepare_task_cfgs(TASK, [], play=True)
cfg.scene.num_envs = 1; cfg.viewer.width, cfg.viewer.height = 560, 420
base = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
env = RslRlVecEnvWrapper(base, clip_actions=ag.clip_actions)
r = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(ag), device="cuda:0")
r.load(CKPT, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
pol = r.get_inference_policy(device="cuda:0")
cmd = base.command_manager.get_term("motion"); mf = list(cmd.cfg.motion_files)
i = next(k for k, m in enumerate(mf) if SEQ in m)
cmd._resample_command = types.MethodType(locked(i), cmd)
rob = base.scene["robot"]; obj = base.scene["object"]

# Pass 1: actual policy rollout
base.reset(); obs = env.get_observations(); A = []; ERR = []
for t in range(N):
  a = pol(obs); obs, _, _, _ = env.step(a)
  ERR.append(float(torch.linalg.norm(cmd.obj_pos - cmd.ref_obj_pos, dim=-1)[0]))
  im = base.render(); A.append(np.asarray(im)[:, :, :3].copy() if im is not None else A[-1])

# Pass 2: reference pose (teleport hand+object, no physics)
base.reset(); R = []
for t in range(N):
  s = int(cmd.env_seq[0])
  soft = rob.data.soft_joint_pos_limits
  jp = torch.clip(cmd._ref_qpos[s, t].clone().unsqueeze(0), soft[:, :, 0], soft[:, :, 1])
  rob.write_joint_state_to_sim(jp, torch.zeros_like(jp))
  st = torch.cat([cmd._ref_obj_pos[s, t].unsqueeze(0), cmd._ref_obj_quat[s, t].unsqueeze(0),
                  torch.zeros(1, 6, device=base.device)], dim=-1)
  obj.write_root_state_to_sim(st); base.sim.forward(); cmd.time_steps[0] = t
  im = base.render(); R.append(np.asarray(im)[:, :, :3].copy() if im is not None else R[-1])


def font(s):
  try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", s)
  except Exception: return ImageFont.load_default()


def panel(arr, title, col):
  im = Image.fromarray(arr.copy()); d = ImageDraw.Draw(im)
  d.rectangle([0, 0, im.width - 1, im.height - 1], outline=col, width=6)
  d.rectangle([0, 0, im.width, 26], fill=(0, 0, 0))
  d.text((6, 3), title, fill=col, font=font(18))
  return np.asarray(im)


out = []
for t in range(N):
  red = ERR[t] > 0.05
  col = (255, 60, 60) if red else (120, 255, 120)
  left = panel(A[t], f"ACTUAL (physics) t{t} err{ERR[t]:.03f}m", col)
  right = panel(R[t], "REFERENCE (no physics, ideal)", col)
  sep = np.full((left.shape[0], 4, 3), 40, np.uint8)
  out.append(np.concatenate([left, sep, right], axis=1))
O = "/data/home/liangheng/DexTrack/videos/metric_check"
imageio.mimwrite(f"{O}/cmp_"+os.environ.get("TAG","x")+".mp4", out, fps=30, quality=9)
imageio.mimwrite(f"{O}/cmp_"+os.environ.get("TAG","x")+".gif", out[::2], fps=15, loop=0)
print(f"WROTE 红帧={sum(1 for e in ERR if e > 0.05)}/{N} (solid=actual, ghost=reference)", flush=True)
os._exit(0)

"""Offline 3-way reward comparison: roll out each trained policy and score it on
ONE common metric suite, including the config-independent DexTrack "fair" reward
(canonical base coefs: hand_pose 0.6/0.1/0.1, grip 0.22, palm_dist_rew_w 2.0,
4-finger sum, NO pinall3 / NO patches -- mirrors task.py fair@0.22).

The three policies train on different rewards but the physics (kp x1, wdelta,
cubesmall) is identical, so rolling each out in its own task and scoring with the
SAME functions is apples-to-apples. Metrics: fair reward, object lift/track,
jitter ||dq_finger||, finger-pose error (naturalness), palm error, contact dist.
"""
from __future__ import annotations

import glob
from dataclasses import asdict

import numpy as np
import torch

import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.lab_api.math import quat_apply
from wuji_mjlab.tasks.tracking.mdp import rewards as R
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

CONFIGS = [
  ("original", "WujiHand_Tracking_Cubesmall_Cmp_Original"),
  ("pinall3", "WujiHand_Tracking_Cubesmall_Cmp_Pinall3"),
  ("cgsmooth_b2_softclip", "WujiHand_Tracking_Cubesmall_Cmp_CGSmooth"),
]
N_ENVS = 64
N_STEPS = 300


def _fair_reward(env) -> torch.Tensor:
  """Canonical base reward (config-independent), per env, this step."""
  hp = R.hand_pose_tracking(env, "motion", 0.6, 0.1, 0.1)
  fo = R.finger_object_distance(env, "motion", palm_dist_rew_w=2.0,
                                grip_thres=0.22, n_finger_sum=4)
  op = R.object_pos_tracking(env, "motion", grip_thres=0.22, n_finger_sum=4)
  ib = R.object_inplace_bonus(env, "motion", grip_thres=0.22, n_finger_sum=4)
  return 0.5 * hp + 0.3 * fo + 1.0 * op + 1.0 * ib


def _latest_ckpt(task: str) -> str:
  run = task.split("_")[-1]
  pts = glob.glob(f"logs/rsl_rl/wuji_tracking/*Cmp_{run}/model_*.pt")
  if not pts:
    raise FileNotFoundError(f"no checkpoint for {task}")
  return sorted(pts, key=lambda x: int(x.split("_")[-1][:-3]))[-1]


def eval_one(label: str, task: str) -> dict:
  ckpt = _latest_ckpt(task)
  env_cfg, agent_cfg = prepare_task_cfgs(task, [], play=True)
  env_cfg.scene.num_envs = N_ENVS
  base = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
  r = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cuda:0")
  r.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  pol = r.get_inference_policy(device="cuda:0")

  cmd = base.command_manager.get_term("motion")
  robot = base.scene["robot"]
  obj = base.scene["object"]
  # contact data for the contact-distance metric (load once, any config).
  cf = np.load(
    "/home/liangh/DexTrack/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_FPOS_v1/"
    "contact_grab2/ori_grab_s2_cubesmall_inspect_1_contact.npy", allow_pickle=True
  ).item()
  c_flag = torch.tensor(cf["contact_flag"], device="cuda:0")  # (T,5)
  c_local = torch.tensor(cf["contact_pos_local"], device="cuda:0")  # (T,5,3)
  tip_idx = torch.as_tensor(
    robot.find_sites([f"right_finger{i}_tip" for i in range(1, 6)], preserve_order=True)[0],
    device="cuda:0", dtype=torch.long,
  )

  obs = env.get_observations()
  fair = torch.zeros(N_ENVS, device="cuda:0")
  prev_q = None
  jit, ferr, perr, otrk, cdist, ozs, rzs = [], [], [], [], [], [], []
  for i in range(N_STEPS):
    obs, _, _, _ = env.step(pol(obs))
    fair += _fair_reward(base)
    q = robot.data.joint_pos
    if prev_q is not None:
      jit.append(torch.norm(q[:, 6:] - prev_q[:, 6:], dim=-1).mean().item())
    prev_q = q.clone()
    ferr.append(torch.norm(cmd.ref_qpos[:, 6:] - q[:, 6:], dim=-1).mean().item())
    sim_palm = robot.data.body_link_pos_w[:, robot.find_bodies(["right_palm_link"])[0][0]]
    perr.append(torch.norm(sim_palm - cmd.ref_palm_pos, dim=-1).mean().item())
    otrk.append(torch.norm(obj.data.root_link_pos_w - cmd.ref_obj_pos, dim=-1).mean().item())
    ozs.append(obj.data.root_link_pos_w[:, 2].mean().item())
    rzs.append(cmd.ref_obj_pos[:, 2].mean().item())
    # contact distance for the 3 contact fingers at this frame.
    t = int(cmd.time_steps[0])
    fl, lo = c_flag[t], c_local[t]  # (5,), (5,3)
    world = quat_apply(obj.data.root_link_quat_w.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 4),
                       lo.unsqueeze(0).expand(N_ENVS, -1, -1).reshape(-1, 3)
                       ).reshape(N_ENVS, 5, 3) + obj.data.root_link_pos_w.unsqueeze(1)
    d = torch.norm(robot.data.site_pos_w[:, tip_idx] - world, dim=-1)  # (E,5)
    if fl.sum() > 0:
      # per-env mean over contact fingers, then mean over envs (NOT a global sum
      # over envs/fingers divided by #fingers -- that inflates by #envs).
      per_env = (d * fl).sum(dim=-1) / (fl.sum() + 1e-6)  # (E,)
      cdist.append(per_env.mean().item())

  ozs, rzs = np.array(ozs), np.array(rzs)
  res = {
    "label": label, "ckpt": ckpt.split("/")[-1],
    "fair": fair.mean().item(),
    "max_z": float(ozs.max()), "ref_max_z": float(rzs.max()),
    "lift_ok": bool(ozs.max() > 0.25),
    "obj_trk_mm": float(np.mean(otrk) * 1000),
    "jitter": float(np.mean(jit)),
    "finger_err_rad": float(np.mean(ferr)),
    "palm_err_cm": float(np.mean(perr) * 100),
    "contact_cm": float(np.mean(cdist) * 100) if cdist else float("nan"),
  }
  del env, base
  torch.cuda.empty_cache()
  return res


def main():
  rows = [eval_one(lbl, t) for lbl, t in CONFIGS]
  hdr = ["label", "ckpt", "fair", "max_z", "obj_trk_mm", "jitter",
         "finger_err_rad", "palm_err_cm", "contact_cm", "lift_ok"]
  print("\n===== 3-WAY REWARD COMPARISON (kp x1, wdelta) =====")
  print(" | ".join(f"{h:>14s}" for h in hdr))
  for r in rows:
    print(" | ".join(
      f"{r[h]:>14.3f}" if isinstance(r[h], float) else f"{str(r[h]):>14s}"
      for h in hdr))


if __name__ == "__main__":
  main()

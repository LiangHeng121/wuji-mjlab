"""Final metric, all 3 objects: success = pos<5cm AND >=2 finger contacts, for
>=50% of active frames (frames where ref pose moved from start). No orientation.
Usage: eval_newmetric_full.py <ckpt> <data_ver> <label>"""
from __future__ import annotations
import os, sys
CKPT, DATA_VER, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ["WUJI_DATA_VER"] = DATA_VER

import types
from dataclasses import asdict
import torch

import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from wuji_mjlab.tasks.tracking.config.wuji_hand.env_cfgs import _object_sequences
from wuji_mjlab.tasks.tracking.mdp.rewards import n_finger_contacts
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

OBJS = [("cube", "WujiHand_Tracking_CubesmallMulti_CGSmooth_Contact", "cubesmall"),
        ("cup", "WujiHand_Tracking_CupMulti_CGSmooth_Contact", "cup"),
        ("apple", "WujiHand_Tracking_AppleMulti_CGSmooth_Contact", "apple")]
POS_TH, FRAC_TH, MIN_C = 0.05, 0.50, 2
LIFT_RATIO, FLY_MARGIN = 0.5, 0.30  # act_z in [ratio*ref_z, ref_z+margin]: followed lift, no fly-off


def quat_deg(a, b):
  return torch.rad2deg(2 * torch.arccos((a * b).sum(-1).abs().clamp(max=1.0)))


def make_locked(li):
  def _r(self, ids):
    self.time_steps[ids] = 0
    self.env_seq[ids] = li[ids]; self.env_obj[ids] = self.seq_obj[self.env_seq[ids]]
    s, t = self.env_seq[ids], self.time_steps[ids]
    soft = self.robot.data.soft_joint_pos_limits[ids]
    jp = torch.clip(self._ref_qpos[s, t].clone(), soft[:, :, 0], soft[:, :, 1])
    self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids); self.robot.reset(env_ids=ids)
    rp, rq = self._ref_obj_pos[s, t], self._ref_obj_quat[s, t]
    vel = torch.zeros(len(ids), 6, device=self.device)
    self.obj.write_root_state_to_sim(torch.cat([rp, rq, vel], dim=-1), env_ids=ids); self.obj.reset(env_ids=ids)
  return _r


print(f"[{LABEL}] {CKPT}  data={DATA_VER}", flush=True)
tot_ok, tot_n = 0, 0
for oname, task, objkey in OBJS:
  seqs = _object_sequences(objkey)
  env_cfg, agent_cfg = prepare_task_cfgs(task, [], play=True)
  env_cfg.scene.num_envs = len(seqs)
  base = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
  r = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cuda:0")
  r.load(CKPT, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  pol = r.get_inference_policy(device="cuda:0")
  cmd = base.command_manager.get_term("motion")
  mf = list(cmd.cfg.motion_files)
  idx = torch.tensor([next(i for i, m in enumerate(mf) if nm in m) for nm in seqs], device="cuda:0")
  N = len(seqs)
  rtp = cmd._ref_obj_pos[idx]; rtq = cmd._ref_obj_quat[idx]
  disp = torch.linalg.norm(rtp - rtp[:, :1], dim=-1)
  rotd = quat_deg(rtq.reshape(-1, 4), rtq[:, :1].expand(-1, rtq.shape[1], -1).reshape(-1, 4)).reshape(N, -1)
  active = (disp > 0.02) | (rotd > 15.0)
  ref_z = (rtp[:, :, 2] - rtp[:, :1, 2]).max(dim=1).values   # reference max height gain
  cmd._resample_command = types.MethodType(make_locked(idx), cmd)
  base.reset(); obs = env.get_observations()
  ok_cnt = torch.zeros(N, device="cuda:0")
  obj0 = cmd.obj_pos.clone(); act_z = torch.zeros(N, device="cuda:0")
  for t in range(300):
    obs, _, _, _ = env.step(pol(obs))
    pe = torch.linalg.norm(cmd.obj_pos - cmd.ref_obj_pos, dim=-1)
    fok = (pe < POS_TH) & (n_finger_contacts(base) >= MIN_C) & active[:, t]
    ok_cnt += fok.float()
    act_z = torch.maximum(act_z, cmd.obj_pos[:, 2] - obj0[:, 2])
  n_act = active[:, :300].sum(1).clamp(min=1)
  frac = ok_cnt / n_act.float()
  followed = (act_z >= LIFT_RATIO * ref_z) & (act_z <= ref_z + FLY_MARGIN)  # reached ref lift, no fly-off
  succ = (frac >= FRAC_TH) & followed & (active[:, :300].sum(1) >= 10)
  n_ok = int(succ.sum()); tot_ok += n_ok; tot_n += N
  print(f"  {oname}: {n_ok}/{N}", flush=True)
  del env, base, r; torch.cuda.empty_cache()

print(f"{LABEL}: {tot_ok}/{tot_n}", flush=True)

"""一起-TOPO eval excluding the 6 range-bug sequences, frac threshold from env."""
from __future__ import annotations
import os, glob, types
import os as _o; _o.environ["WUJI_DATA_VER"] = __import__("sys").argv[2]
from dataclasses import asdict
import torch
import wuji_mjlab.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from wuji_mjlab.tasks.tracking.config.wuji_hand.env_cfgs import _object_sequences
from wuji_mjlab.tasks.tracking.mdp.rewards import n_finger_contacts
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

import sys
CKPT_ARG, DATA_VER_ARG, LABEL_ARG = sys.argv[1], sys.argv[2], sys.argv[3]
OBJS_ARG = sys.argv[4].split(",") if len(sys.argv)>4 else ["cube","cup","apple"]
FRAC_TH = float(os.environ.get("WUJI_FRAC", "0.90"))
# range-bug 6条(±π wrap / 平移限位)已由 commands.py unwrap + MJCF放宽修复, 重训模型上
# 可达(9obj实测4/5成功, cup_s4_drink_2 是正常失败非bug), 不再排除。
# 穿地板5条由 _object_sequences._UNDERGROUND_SEQS 处理(物理不可达)。故 EXCLUDE 置空。
EXCLUDE = set()
POS_TH, MIN_C, LIFT_RATIO, FLY_MARGIN = 0.05, 2, 0.5, 0.30
ALL_OBJS = [("cube", "WujiHand_Tracking_CubesmallMulti_CGSmooth_Contact", "cubesmall"),
        ("cup", "WujiHand_Tracking_CupMulti_CGSmooth_Contact", "cup"),
        ("apple", "WujiHand_Tracking_AppleMulti_CGSmooth_Contact", "apple"),
        ("duck", "WujiHand_Tracking_DuckMulti_CGSmooth_Contact", "duck"),
        ("elephant", "WujiHand_Tracking_ElephantMulti_CGSmooth_Contact", "elephant"),
        ("mouse", "WujiHand_Tracking_MouseMulti_CGSmooth_Contact", "mouse"),
        ("phone", "WujiHand_Tracking_PhoneMulti_CGSmooth_Contact", "phone"),
        ("train", "WujiHand_Tracking_TrainMulti_CGSmooth_Contact", "train"),
        ("alarmclock", "WujiHand_Tracking_AlarmclockMulti_CGSmooth_Contact", "alarmclock"),
        ("flute", "WujiHand_Tracking_FluteMulti_CGSmooth_Contact", "flute")]
OBJS = [o for o in ALL_OBJS if o[0] in OBJS_ARG]
CKPT = CKPT_ARG


def quat_deg(a, b): return torch.rad2deg(2 * torch.arccos((a * b).sum(-1).abs().clamp(max=1.0)))


def make_locked(li):
  def _r(self, ids):
    self.time_steps[ids] = 0
    self.env_seq[ids] = li[ids]; self.env_obj[ids] = self.seq_obj[self.env_seq[ids]]
    s, t = self.env_seq[ids], self.time_steps[ids]; soft = self.robot.data.soft_joint_pos_limits[ids]
    jp = torch.clip(self._ref_qpos[s, t].clone(), soft[:, :, 0], soft[:, :, 1])
    self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids); self.robot.reset(env_ids=ids)
    rp, rq = self._ref_obj_pos[s, t], self._ref_obj_quat[s, t]; vel = torch.zeros(len(ids), 6, device=self.device)
    self.obj.write_root_state_to_sim(torch.cat([rp, rq, vel], dim=-1), env_ids=ids); self.obj.reset(env_ids=ids)
  return _r


print(f"[{LABEL_ARG}] frac>={FRAC_TH} 排除{len(EXCLUDE)}条 objs={OBJS_ARG}", flush=True)
tot_ok, tot_n = 0, 0
for oname, task, objkey in OBJS:
  seqs = [s for s in _object_sequences(objkey) if s not in EXCLUDE]
  cfg, ag = prepare_task_cfgs(task, [], play=True); cfg.scene.num_envs = len(seqs)
  base = ManagerBasedRlEnv(cfg=cfg, device="cuda:0"); env = RslRlVecEnvWrapper(base, clip_actions=ag.clip_actions)
  r = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(ag), device="cuda:0")
  r.load(CKPT, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  pol = r.get_inference_policy(device="cuda:0")
  cmd = base.command_manager.get_term("motion"); mf = list(cmd.cfg.motion_files)
  idx = torch.tensor([next(i for i, m in enumerate(mf) if nm in m) for nm in seqs], device="cuda:0")
  N = len(seqs)
  rtp = cmd._ref_obj_pos[idx]; rtq = cmd._ref_obj_quat[idx]
  disp = torch.linalg.norm(rtp - rtp[:, :1], dim=-1)
  rotd = quat_deg(rtq.reshape(-1, 4), rtq[:, :1].expand(-1, rtq.shape[1], -1).reshape(-1, 4)).reshape(N, -1)
  active = (disp > 0.02) | (rotd > 15.0)
  ref_z = (rtp[:, :, 2] - rtp[:, :1, 2]).max(dim=1).values
  cmd._resample_command = types.MethodType(make_locked(idx), cmd)
  base.reset(); obs = env.get_observations()
  ok = torch.zeros(N, device="cuda:0"); obj0 = cmd.obj_pos.clone(); act_z = torch.zeros(N, device="cuda:0")
  for t in range(300):
    obs, _, _, _ = env.step(pol(obs))
    pe = torch.linalg.norm(cmd.obj_pos - cmd.ref_obj_pos, dim=-1)
    ok += ((pe < POS_TH) & (n_finger_contacts(base) >= MIN_C) & active[:, t]).float()
    act_z = torch.maximum(act_z, cmd.obj_pos[:, 2] - obj0[:, 2])
  frac = ok / active[:, :300].sum(1).clamp(min=1).float()
  followed = (act_z >= LIFT_RATIO * ref_z) & (act_z <= ref_z + FLY_MARGIN)
  succ = (frac >= FRAC_TH) & followed & (active[:, :300].sum(1) >= 10)
  no = int(succ.sum()); tot_ok += no; tot_n += N
  print(f"  {oname}: {no}/{N}", flush=True)
  del env, base, r; torch.cuda.empty_cache()
print(f"{LABEL_ARG} TOTAL: {tot_ok}/{tot_n}", flush=True)

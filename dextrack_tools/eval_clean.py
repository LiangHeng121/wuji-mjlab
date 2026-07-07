"""一起-TOPO eval excluding the 6 range-bug sequences, frac threshold from env."""
from __future__ import annotations
import os, glob, types
os.environ["WUJI_DATA_VER"] = "GRAB_Tracking_PK_WUJI_TOPO_v1"
from dataclasses import asdict
import torch
import wuji_mjlab.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from wuji_mjlab.tasks.tracking.config.wuji_hand.env_cfgs import _object_sequences
from wuji_mjlab.tasks.tracking.mdp.rewards import n_finger_contacts
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

FRAC_TH = float(os.environ.get("WUJI_FRAC", "0.90"))
EXCLUDE = {"ori_grab_s8_cubesmall_pass_1", "ori_grab_s4_cup_drink_2", "ori_grab_s8_cup_drink_1",
           "ori_grab_s4_cup_pour_1", "ori_grab_s7_cup_pour_1", "ori_grab_s2_apple_eat_1"}
POS_TH, MIN_C, LIFT_RATIO, FLY_MARGIN = 0.05, 2, 0.5, 0.30
OBJS = [("cube", "WujiHand_Tracking_CubesmallMulti_CGSmooth_Contact", "cubesmall"),
        ("cup", "WujiHand_Tracking_CupMulti_CGSmooth_Contact", "cup"),
        ("apple", "WujiHand_Tracking_AppleMulti_CGSmooth_Contact", "apple")]
CKPT = sorted(glob.glob("logs/rsl_rl/wuji_tracking/*Gen3obj_MassCur_coef0_TOPO_Env8000/model_*.pt"),
              key=lambda x: int(x.split("_")[-1][:-3]))[-1]


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


print(f"[一起-TOPO clean]  frac>={FRAC_TH}  排除{len(EXCLUDE)}条", flush=True)
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
print(f"一起-TOPO clean: {tot_ok}/{tot_n}", flush=True)

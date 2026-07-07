"""New success metric vs old, per seq, to find borderline / disagreement cases.
OLD: peak-displacement-frame pos err < 5cm.
NEW: over the ACTIVE phase (frames where ref pose moved from start), require
     pos<5cm AND orient<30deg AND >=2 finger contacts, for >=80% of active frames.
Usage: eval_new_metric.py <ckpt> <data_ver> <object>  (object=cubesmall|cup|apple)
"""
from __future__ import annotations
import os, sys
CKPT, DATA_VER, OBJ = sys.argv[1], sys.argv[2], sys.argv[3]
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

TASKS = {"cubesmall": "WujiHand_Tracking_CubesmallMulti_CGSmooth_Contact",
         "cup": "WujiHand_Tracking_CupMulti_CGSmooth_Contact",
         "apple": "WujiHand_Tracking_AppleMulti_CGSmooth_Contact"}
POS_TH, ORI_TH, FRAC_TH, MIN_C = 0.05, 30.0, 0.50, 2


def quat_deg(a, b):  # geodesic angle (deg) between quats [N,4]
  d = (a * b).sum(-1).abs().clamp(max=1.0)
  return torch.rad2deg(2 * torch.arccos(d))


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


task = TASKS[OBJ]
seqs = _object_sequences(OBJ)
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
rtp = cmd._ref_obj_pos[idx]; rtq = cmd._ref_obj_quat[idx]        # [N,T,3],[N,T,4]
disp = torch.linalg.norm(rtp - rtp[:, :1], dim=-1)              # [N,T]
rotd = quat_deg(rtq.reshape(-1, 4), rtq[:, :1].expand(-1, rtq.shape[1], -1).reshape(-1, 4)).reshape(N, -1)
active_ref = (disp > 0.02) | (rotd > 15.0)                      # [N,T] ref-demands-motion frames
peak_t = disp.argmax(dim=1)
cmd._resample_command = types.MethodType(make_locked(idx), cmd)
base.reset(); obs = env.get_observations()
pos_err = torch.zeros(N, 300, device="cuda:0"); ori_err = torch.zeros(N, 300, device="cuda:0")
contacts = torch.zeros(N, 300, device="cuda:0"); err_peak = torch.zeros(N, device="cuda:0")
obj0 = cmd.obj_pos.clone()                       # actual object start pos
act_disp = torch.zeros(N, device="cuda:0"); act_z = torch.zeros(N, device="cuda:0")
final_err = torch.zeros(N, device="cuda:0")
for t in range(300):
  obs, _, _, _ = env.step(pol(obs))
  pe = torch.linalg.norm(cmd.obj_pos - cmd.ref_obj_pos, dim=-1)
  pos_err[:, t] = pe; ori_err[:, t] = quat_deg(cmd.obj_quat, cmd.ref_obj_quat)
  contacts[:, t] = n_finger_contacts(base)
  err_peak = torch.where(peak_t == t, pe, err_peak)
  act_disp = torch.maximum(act_disp, torch.linalg.norm(cmd.obj_pos - obj0, dim=-1))
  act_z = torch.maximum(act_z, cmd.obj_pos[:, 2] - obj0[:, 2])
  final_err = pe                                 # last frame tracking error
ref_disp_max = disp.max(dim=1).values            # reference peak displacement
ref_z = (rtp[:, :, 2] - rtp[:, :1, 2]).max(dim=1).values   # reference max height gain

act = active_ref[:, :300]
n_act = act.sum(1).clamp(min=1)
# NEW metric (all objects): position<5cm AND >=2 finger contacts, no orientation.
frame_ok = (pos_err < POS_TH) & (contacts >= MIN_C)
frac = (frame_ok & act).sum(1).float() / n_act.float()
new_ok = (frac >= FRAC_TH) & (act.sum(1) >= 10)
old_ok = err_peak < POS_TH

print(f"\n==== {OBJ}  占比/实举高/参举高/末误差/peakErr ====")
print(f"{'seq':<26}{'占比':>6}{'实举高':>7}{'参举高':>7}{'末误差':>7}{'peak':>7}")
for i in range(N):
  print(f"{seqs[i]:<26}{float(frac[i]):>6.2f}{float(act_z[i]):>7.3f}{float(ref_z[i]):>7.3f}"
        f"{float(final_err[i]):>7.3f}{float(err_peak[i]):>7.3f}")
print(f"\nOLD: {int(old_ok.sum())}/{N}   NEW: {int(new_ok.sum())}/{N}")
dis = [s for s, _, o, _, nw, _ in rows if o and not nw]
bord = [s for s, _, _, fr, _, _ in rows if 0.6 <= fr < 0.9]
print("DISAGREE(旧✓新✗)=", dis)
print("边界(新占比0.6~0.9)=", bord)

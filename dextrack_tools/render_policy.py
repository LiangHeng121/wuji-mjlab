"""Render a trained policy rollout (default fixed camera; full physics env)."""
import sys, glob, numpy as np, torch, imageio
from dataclasses import asdict
import wuji_mjlab.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

TASK = sys.argv[1]
PAT  = sys.argv[2]                      # glob to pick checkpoint dir
OUT  = sys.argv[3]
wd = sorted(glob.glob(PAT), key=lambda x: int(x.split("_")[-1][:-3]))[-1]
print("CKPT", wd)
env_cfg, agent_cfg = prepare_task_cfgs(TASK, [], play=True)
env_cfg.scene.num_envs = 1
base = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array")
env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
r = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device="cuda:0")
r.load(wd, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
pol = r.get_inference_policy(device="cuda:0")
obj = base.scene["object"]; cmd = base.command_manager.get_term("motion")
obs = env.get_observations()
frames, oz, errs = [], [], []
for i in range(300):
    obs, _, _, _ = env.step(pol(obs))
    img = base.render()
    if img is not None: frames.append(np.asarray(img))
    z = float(obj.data.root_link_pos_w[0,2]); rz = float(cmd.ref_obj_pos[0,2])
    oz.append(z); errs.append(abs(z-rz))
imageio.mimwrite(OUT, frames, fps=30, quality=8)
oz = np.array(oz)
print("WROTE", OUT, "frames", len(frames), "| max z", round(oz.max(),3),
      "| z err mean", round(float(np.mean(errs)),3))

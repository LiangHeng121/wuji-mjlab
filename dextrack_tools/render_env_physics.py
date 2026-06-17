"""Physics open-loop render via the REAL env (mujoco-warp), any object+sequence."""
import sys, numpy as np, torch, imageio
from mjlab.envs import ManagerBasedRlEnv
from wuji_mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from wuji_mjlab.tasks.tracking.config.wuji_hand.robot import get_wuji_fly_hand_cfg
from wuji_mjlab.tasks.tracking.grab_object_cfg import get_grab_object_cfg

OBJ = sys.argv[1] if len(sys.argv) > 1 else "duck"
SEQ = sys.argv[2] if len(sys.argv) > 2 else "ori_grab_s1_duck_lift"
RGBA = (0.95, 0.78, 0.1, 1) if OBJ == "duck" else (0.9, 0.15, 0.05, 1)
REF = ("/home/liangh/DexTrack/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_FPOS_v1/"
       f"data/wuji_passive_active_info_{SEQ}_nf_300.npy")
OUT = f"/home/liangh/DexTrack/mjlab_{OBJ}_PHYSICS_env.mp4"

cfg = make_tracking_env_cfg(num_envs=1)
cfg.scene.entities = {"robot": get_wuji_fly_hand_cfg(),
                      "object": get_grab_object_cfg(OBJ, rgba=RGBA)}
cfg.commands["motion"].motion_file = REF
cfg.viewer.body_name = "right_palm_link"
cfg.observations["policy"].enable_corruption = False

env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
adim = env.action_manager.total_action_dim
obs, _ = env.reset()
obj = env.scene["object"]

frames, oz = [], []
for i in range(300):
    env.step(torch.zeros((env.num_envs, adim), device=env.device))
    img = env.render()
    if img is not None:
        frames.append(np.asarray(img))
    oz.append(float(obj.data.root_link_pos_w[0, 2]))
if frames:
    imageio.mimwrite(OUT, frames, fps=30, quality=8)
oz = np.array(oz)
print("WROTE", OUT, "frames", len(frames), "| obj z max", round(oz.max(), 3),
      "->", "STABLE" if oz.max() < 1.0 else "EXPLODED")

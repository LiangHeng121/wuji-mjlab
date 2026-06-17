import torch
from mjlab.envs import ManagerBasedRlEnv
from wuji_mjlab.tasks.tracking.config.wuji_hand.env_cfgs import (
    wuji_hand_cubesmall_tracking_env_cfg,
)

cfg = wuji_hand_cubesmall_tracking_env_cfg(play=True)
cfg.scene.num_envs = 8
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
adim = env.action_manager.total_action_dim
print("action dim:", adim, "num_envs:", env.num_envs)

obs, extras = env.reset()
for k, v in obs.items():
    print("obs group", k, "->", v.shape)

m = env.command_manager.get_term("motion")
print("ref frames:", m.time_step_total)

for i in range(305):
    if i < 30:
        act = torch.zeros((env.num_envs, adim), device=env.device)
    else:
        act = torch.rand((env.num_envs, adim), device=env.device) * 2 - 1
    obs, rew, term, trunc, info = env.step(act)
    assert torch.isfinite(rew).all(), f"non-finite reward at step {i}"
    if i % 60 == 0 or i > 300:
        print(
            f"step {i:3d}: rew {rew.mean().item():+.3f} | "
            f"err_joint {m.metrics['error_joint_pos'].mean().item():.3f} | "
            f"err_obj {m.metrics['error_obj_pos'].mean().item():.3f} | "
            f"term {int(term.sum())} trunc {int(trunc.sum())} | ts {int(m.time_steps[0])}"
        )
print("SMOKE OK")

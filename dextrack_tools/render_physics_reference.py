"""Physics-driven reference render.

Unlike the kinematic replay (which teleports qpos), this drives the hand's
position actuators toward the reference q26[t] (zero policy residual = pure
kinematics bias) and steps REAL MuJoCo physics: gravity on, the object is a
free body responding to contact. Shows whether following the retargeted
reference actually produces a working grasp under MuJoCo physics + the current
(placeholder) actuator gains -- the decisive pre-training physics sanity check.
"""

import pathlib
import numpy as np
import mujoco
import imageio

HAND = "src/wuji_mjlab/assets/robots/wuji_hand/mjcf/right_fly_mjlab.xml"
REF = (
    "/home/liangh/DexTrack/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_FPOS_v1/"
    "data/wuji_passive_active_info_ori_grab_s2_cubesmall_inspect_1_nf_300.npy"
)
OUT = "/home/liangh/DexTrack/mjlab_fly_cubesmall_PHYSICS_reference.mp4"

D = np.load(REF, allow_pickle=True).item()
q = D["robot_delta_states_weights_np"]      # (300,26) q26 order == actuator order
otrans = D["object_transl"]
oquat_xyzw = D["object_rot_quat"]

spec = mujoco.MjSpec.from_file(HAND)
mesh_root = (pathlib.Path(HAND).parent / spec.meshdir).resolve()
spec.assets = {f"{spec.meshdir}/{f.name}": f.read_bytes() for f in mesh_root.glob("*.STL")}

wb = spec.worldbody
wb.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1])
wb.add_light(pos=[0.5, 0.5, 1.0], dir=[-1, -1, -2])
wb.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], pos=[0, 0, 0],
            rgba=[0.4, 0.4, 0.4, 1])
cube = wb.add_body(name="obj", pos=[0, 0, 0])
cube.add_freejoint()
cg = cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025, 0.025, 0.025],
                   rgba=[0.9, 0.15, 0.05, 1], mass=0.085)
cg.group = 2

model = spec.compile()
model.opt.timestep = 0.0083  # matches the env (decimation 2 -> control dt 0.0166)
SUBSTEPS = 2
data = mujoco.MjData(model)

oa = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obj")]

# Initialize to reference frame 0 (hand joints + object pose).
data.qpos[:26] = q[0]
x, y, z = otrans[0]
qx, qy, qz, qw = oquat_xyzw[0]
data.qpos[oa:oa + 7] = [x, y, z, qw, qx, qy, qz]
data.ctrl[:26] = q[0]
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, 480, 640)
cam = mujoco.MjvCamera()
cam.lookat[:] = [-0.04, 0.0, 0.22]
cam.distance = 0.85
cam.azimuth = 130
cam.elevation = -12

frames = []
obj_z = []
for t in range(300):
    data.ctrl[:26] = q[t]                 # actuator targets = reference q26[t]
    for _ in range(SUBSTEPS):
        mujoco.mj_step(model, data)
    renderer.update_scene(data, camera=cam)
    frames.append(renderer.render())
    obj_z.append(float(data.qpos[oa + 2]))

imageio.mimwrite(OUT, frames, fps=30, quality=8)
oz = np.array(obj_z)
print("WROTE", OUT, "frames", len(frames))
print(f"object z: start {oz[0]:.3f}  max {oz.max():.3f}  end {oz[-1]:.3f}  ref_max {otrans[:,2].max():.3f}")
print("(object lifted)" if oz.max() > 0.1 else "(object stayed low -- grasp may have failed under physics)")

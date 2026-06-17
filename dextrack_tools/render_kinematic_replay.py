"""Kinematic replay: fly wuji hand + cube through 300 cubesmall reference frames.
Sets qpos = q26 and cube freejoint = ref object pose (xyzw->wxyz), renders mp4.
"""
import os, pathlib, numpy as np, mujoco, imageio

HAND = "src/wuji_mjlab/assets/robots/wuji_hand/mjcf/right_fly_mjlab.xml"
REF = "/home/liangh/DexTrack/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_FPOS_v1/data/wuji_passive_active_info_ori_grab_s2_cubesmall_inspect_1_nf_300.npy"
OUT = "/home/liangh/DexTrack/mjlab_fly_cubesmall_replay.mp4"

D = np.load(REF, allow_pickle=True).item()
q = D["robot_delta_states_weights_np"]          # (300,26)
otrans = D["object_transl"]                      # (300,3)
oquat_xyzw = D["object_rot_quat"]                # (300,4) xyzw

# --- build scene: hand spec + cube + floor + light + camera ---
spec = mujoco.MjSpec.from_file(HAND)
# inline mesh assets (meshdir is ../meshes/right relative to the xml)
mesh_root = (pathlib.Path(HAND).parent / spec.meshdir).resolve()
assets = {}
for f in mesh_root.glob("*.STL"):
    assets[f"{spec.meshdir}/{f.name}"] = f.read_bytes()
spec.assets = assets

wb = spec.worldbody
wb.add_light(pos=[0.0, 0.0, 1.5], dir=[0, 0, -1])
wb.add_light(pos=[0.5, 0.5, 1.0], dir=[-1, -1, -2])
# floor far below so it doesn't occlude
wb.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1],
            pos=[0, 0, -0.5], rgba=[0.5, 0.5, 0.5, 1])
cube = wb.add_body(name="obj", pos=[0, 0, 0])
cube.add_freejoint()
cube_geom = cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025, 0.025, 0.025],
                          rgba=[0.9, 0.15, 0.05, 1], mass=0.05)
# IMPORTANT: the wuji MJCF sets a global <default><geom group="3"> (collision group,
# hidden by the renderer). add_geom inherits it, so without an explicit visible group
# the cube never draws. Force group=2 (visual).
cube_geom.group = 2

model = spec.compile()
data = mujoco.MjData(model)

obj_qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obj")]
print("nq", model.nq, "obj qpos addr", obj_qadr)

W, H = 640, 480
renderer = mujoco.Renderer(model, H, W)
cam = mujoco.MjvCamera()
cam.lookat[:] = [-0.04, 0.0, 0.22]
cam.distance = 0.85
cam.azimuth = 130
cam.elevation = -12

frames = []
for t in range(300):
    data.qpos[:26] = q[t]
    x, y, z = otrans[t]
    qx, qy, qz, qw = oquat_xyzw[t]
    data.qpos[obj_qadr:obj_qadr+7] = [x, y, z, qw, qx, qy, qz]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    frames.append(renderer.render())

imageio.mimwrite(OUT, frames, fps=30, quality=8)
print("WROTE", OUT, "frames", len(frames))

"""Stitch 6 cube success renders into a 3x2 grid (columns = action type), label
each tile, and export a high-res GIF + mp4."""
import numpy as np, imageio.v2 as imageio, os
try:
  from PIL import Image, ImageDraw, ImageFont
  HAVE_PIL = True
except Exception:
  HAVE_PIL = False

D = "/data/home/liangheng/DexTrack/videos/cube_grid"
# grid: rows x cols = 2 x 3, each column an action
LAYOUT = [
  ["lift__ori_grab_s2_cubesmall_lift", "pass__ori_grab_s5_cubesmall_pass_1", "inspect__ori_grab_s1_cubesmall_inspect_1"],
  ["lift__ori_grab_s6_cubesmall_lift", "pass__ori_grab_s10_cubesmall_pass_1", "inspect__ori_grab_s10_cubesmall_inspect_1"],
]
LABELS = [["LIFT", "PASS", "INSPECT"], ["LIFT", "PASS", "INSPECT"]]

# load
vids = {}
for row in LAYOUT:
  for nm in row:
    vids[nm] = imageio.mimread(f"{D}/{nm}.mp4", memtest=False)
n = min(len(v) for v in vids.values())
h, w = vids[LAYOUT[0][0]][0].shape[:2]
print(f"tiles {w}x{h}, frames {n}, grid {len(LAYOUT[0])}x{len(LAYOUT)} -> {w*3}x{h*2}")


def label(img, txt):
  if not HAVE_PIL:
    return img
  im = Image.fromarray(img).convert("RGB")
  d = ImageDraw.Draw(im)
  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
  except Exception:
    font = ImageFont.load_default()
  d.rectangle([0, 0, 118, 34], fill=(0, 0, 0))
  d.text((8, 4), txt, fill=(255, 255, 255), font=font)
  return np.asarray(im)


frames = []
for t in range(0, n, 2):  # subsample every 2nd frame to keep GIF size sane
  rows = []
  for ri, row in enumerate(LAYOUT):
    tiles = [label(np.asarray(vids[nm][t])[:, :, :3], LABELS[ri][ci]) for ci, nm in enumerate(row)]
    rows.append(np.concatenate(tiles, axis=1))
  frames.append(np.concatenate(rows, axis=0))

gif = f"{D}/cube_success_grid.gif"
mp4 = f"{D}/cube_success_grid.mp4"
imageio.mimwrite(gif, frames, fps=20, loop=0)
imageio.mimwrite(mp4, frames, fps=20, quality=9)
sz = os.path.getsize(gif) / 1e6
print(f"WROTE {gif}  {frames[0].shape[1]}x{frames[0].shape[0]}  {len(frames)}帧  {sz:.1f}MB")
print(f"WROTE {mp4}")

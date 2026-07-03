"""Make the 3 figures from dextrack_tools/figs/apple_ztraj.npz.

fig1  per-seq grouped bar of max_z (4 configs) + ref_peak dashed marker
fig2  object-z trajectory: apple specialists (flat ~0.06) vs 3obj (climbs) + ref
fig3  success-rate summary (n/8) per config
"""
from __future__ import annotations
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "dextrack_tools/figs"
d = np.load(f"{FIG}/apple_ztraj.npz", allow_pickle=True)
labels = [str(x) for x in d["labels"]]
seqs = [str(x) for x in d["seqs"]]
PZ, RZ, MZ, LF, RP = d["phys_z"], d["ref_z"], d["max_z"], d["lifted"], d["ref_peak"]
short = [s.replace("ori_grab_", "").replace("_apple_lift", "") for s in seqs]

pretty = {"apple8000kp1": "apple spec. 8000env kp1",
          "apple22000kp1": "apple spec. 22000env kp1",
          "apple22000kp8": "apple spec. 22000env kp8",
          "3obj": "3obj generalist (park)"}
colors = {"apple8000kp1": "#9ecae1", "apple22000kp1": "#3182bd",
          "apple22000kp8": "#6a51a3", "3obj": "#e6550d"}
lstyle = {"apple8000kp1": "-", "apple22000kp1": "-",
          "apple22000kp8": "--", "3obj": "-"}
GROUND = 0.063

# ---------- fig1: grouped bars ----------
fig, ax = plt.subplots(figsize=(13, 6))
S, L = len(seqs), len(labels)
x = np.arange(S)
w = 0.2
for j, lab in enumerate(labels):
  bars = ax.bar(x + (j - (L - 1) / 2) * w, MZ[j], w, label=pretty[lab],
                color=colors[lab])
  for k, b in enumerate(bars):
    if LF[j, k]:
      ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, "✓",
              ha="center", va="bottom", color="green", fontsize=12, fontweight="bold")
# ref_peak markers (dashed horizontal segment per seq group)
for k in range(S):
  ax.hlines(RP[k], x[k] - 0.5, x[k] + 0.5, colors="k", linestyles="--", lw=1.6,
            label="ref_peak (target)" if k == 0 else None)
ax.axhline(GROUND, color="gray", lw=1, ls=":")
ax.text(S - 0.5, GROUND + 0.004, "ground / rest height (~0.063)", ha="right",
        va="bottom", color="gray", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(short)
ax.set_ylabel("object max height  max_z  [m]")
ax.set_title("Apple-lift: max object height per sequence — only 3obj generalist reaches the target")
ax.legend(loc="upper left", fontsize=9, ncol=2)
ax.set_ylim(0, max(RP.max(), MZ.max()) + 0.06)
fig.tight_layout()
fig.savefig(f"{FIG}/fig1_maxz_bars.png", dpi=140)
plt.close(fig)

# ---------- fig2: z trajectories for representative seqs ----------
rep = [s for s in ["s2", "s4", "s9"] if s in short]
rep_idx = [short.index(s) for s in rep]
fig, axes = plt.subplots(1, len(rep_idx), figsize=(6 * len(rep_idx), 5.2), sharey=True)
if len(rep_idx) == 1:
  axes = [axes]
steps = np.arange(PZ.shape[2])
for ax, si in zip(axes, rep_idx):
  # reference (use 3obj's aligned ref; identical target across configs)
  li3 = labels.index("3obj")
  ax.plot(steps, RZ[li3, si], color="k", lw=2.4, ls="--", label="reference (target)")
  # draw the flat specialists first, then lifters (3obj, then kp8 dashed) on top
  # so overlapping lifted curves both stay visible
  order = ["apple8000kp1", "apple22000kp1", "3obj", "apple22000kp8"]
  for lab in order:
    j = labels.index(lab)
    lifted = "  ✓lift" if LF[j, si] else ""
    ax.plot(steps, PZ[j, si], color=colors[lab], lw=2.0, ls=lstyle[lab],
            alpha=0.85, label=pretty[lab] + lifted)
  ax.axhline(GROUND, color="gray", lw=1, ls=":")
  ax.set_title(f"{short[si]}   (ref_peak={RP[si]:.2f} m)")
  ax.set_xlabel("sim step")
  ax.grid(alpha=0.25)
axes[0].set_ylabel("object height z [m]")
axes[0].legend(loc="upper left", fontsize=8)
fig.suptitle("Object-z over rollout: apple specialists hover near the ground (~0.06 m); "
             "only the 3obj generalist climbs to the reference", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(f"{FIG}/fig2_ztraj.png", dpi=140)
plt.close(fig)

# ---------- fig3: success rate summary ----------
fig, ax = plt.subplots(figsize=(7, 5))
n = LF.sum(axis=1)
bars = ax.bar([pretty[l] for l in labels], n, color=[colors[l] for l in labels])
for b, v in zip(bars, n):
  ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{int(v)}/8",
          ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylabel("# apple-lift seqs lifted (max_z ≥ ref_peak − 0.05)")
ax.set_ylim(0, 8)
ax.set_title("Apple-lift success count (of 8) per config")
plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG}/fig3_success.png", dpi=140)
plt.close(fig)

print("wrote:")
for f in ["fig1_maxz_bars.png", "fig2_ztraj.png", "fig3_success.png"]:
  print(" ", os.path.abspath(f"{FIG}/{f}"))

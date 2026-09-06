"""Rebuild the Chapter 2 cost-volume explanation with readable Times text."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent.parent
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "pdf.fonttype": 42,
})

rng = np.random.default_rng(7)
h, w, nd = 24, 36, 18
truth = np.full((h, w), 5, dtype=int)
truth[5:19, 7:18] = 10
truth[3:10, 25:33] = 14
d = np.arange(nd)[:, None, None]
volume = 0.13 + 0.045 * np.abs(d - truth[None, :, :])
volume += rng.normal(0, 0.025, volume.shape)
volume = np.clip(volume, 0, 1)

fig, ax = plt.subplots(2, 2, figsize=(6.35, 4.35), constrained_layout=True)

a = ax[0, 0]
for i, (dx, dy, color) in enumerate(((0.00, 0.00, "#d9ecff"),
                                     (0.08, 0.07, "#acd4f5"),
                                     (0.16, 0.14, "#72b6e1"))):
    a.add_patch(Rectangle((0.13 + dx, 0.14 + dy), 0.56, 0.58,
                          facecolor=color, edgecolor="#2f5d7c", alpha=.78))
    a.text(0.43 + dx, 0.43 + dy, f"d = {i + 1}", ha="center", va="center")
a.annotate("disparity axis", xy=(0.82, .78), xytext=(.60, .93),
           arrowprops=dict(arrowstyle="->", lw=1))
a.set(xlim=(0, 1), ylim=(0, 1), title="(a) Cost volume C(x, y, d)")
a.axis("off")

a = ax[0, 1]
im = a.imshow(volume[10], cmap="RdYlGn_r", vmin=0, vmax=1)
a.set_title("(b) Fixed-disparity slice, d = 10")
a.set_xlabel("x (pixel)")
a.set_ylabel("y (pixel)")
cb = fig.colorbar(im, ax=a, fraction=.048, pad=.03)
cb.set_label("matching cost")

a = ax[1, 0]
profile = volume[:, 10, 12]
a.plot(np.arange(nd), profile, marker="o", ms=3, color="#0072b2")
winner = int(np.argmin(profile))
a.axvline(winner, color="#d55e00", ls="--", lw=1.2)
a.annotate(f"minimum: d = {winner}", (winner, profile[winner]),
           xytext=(winner + 2, profile[winner] + .2),
           arrowprops=dict(arrowstyle="->", lw=1))
a.set_title("(c) Cost profile for one pixel")
a.set_xlabel("candidate disparity d")
a.set_ylabel("matching cost")
a.grid(alpha=.22)

a = ax[1, 1]
pred = np.argmin(volume, axis=0)
im = a.imshow(pred, cmap="viridis", vmin=0, vmax=nd - 1)
a.set_title("(d) Disparity from argmin over d")
a.set_xlabel("x (pixel)")
a.set_ylabel("y (pixel)")
cb = fig.colorbar(im, ax=a, fraction=.048, pad=.03)
cb.set_label("disparity (px)")

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_2_costvolume_times.{ext}", dpi=300,
                bbox_inches="tight", facecolor="white")
plt.close(fig)

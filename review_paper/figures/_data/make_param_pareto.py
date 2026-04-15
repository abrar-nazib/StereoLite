"""Param vs accuracy plot: same design as make_pareto (shape per family,
colour per paper, tiered legend below)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from method_data import METHODS, FAMILIES, FAMILY_MARKER, per_method_colors

import matplotlib.pyplot as plt, matplotlib as mpl

mpl.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 9, "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":",
    "axes.axisbelow": True, "figure.dpi": 150,
})

colors = per_method_colors()

fig = plt.figure(figsize=(11.0, 8.8))
gs = fig.add_gridspec(2, 1, height_ratios=[2.1, 1.0], hspace=0.30)
ax = fig.add_subplot(gs[0, 0])
lax = fig.add_subplot(gs[1, 0]); lax.axis('off')

for n, m in METHODS.items():
    if m["params_m"] is None or m["kitti_d1"] is None: continue
    marker = FAMILY_MARKER[m["family"]]
    s = 180 if marker == '*' else (140 if marker == 'D' else 120)
    ax.scatter(m["params_m"], m["kitti_d1"], c=colors[n], marker=marker,
               s=s, edgecolor='black', linewidth=0.8, alpha=0.95, zorder=3)

ax.set_xscale('log')
ax.set_xlabel('Trainable parameters (M, log scale)')
ax.set_ylabel('KITTI 2015 D1-all (\\%, lower is better)')
ax.set_xlim(0.3, 600)
ax.set_ylim(1.2, 7.0)
ax.set_title('Model size vs. accuracy across deep stereo methods (2017--2026)',
             fontsize=11.5, pad=8)

family_order = ["foundation", "iterative", "3dcv", "efficient", "efficient_iter"]
lax.set_xlim(0, 1); lax.set_ylim(0, 1)
n_cols = len(family_order)
COL_WIDTH = 1.0 / n_cols
HEADER_Y = 0.97
LINE_H = 0.078

for ci, fam in enumerate(family_order):
    members = [n for n in METHODS if METHODS[n]["family"] == fam
               and METHODS[n].get("params_m") is not None
               and METHODS[n].get("kitti_d1") is not None]
    if not members: continue
    x_left = ci * COL_WIDTH + 0.005
    fam_color = FAMILIES[fam]["color"]
    marker = FAMILY_MARKER[fam]
    lax.scatter(x_left + 0.015, HEADER_Y - 0.02,
                s=(120 if marker == '*' else 70),
                c=fam_color, marker=marker,
                edgecolor='black', linewidth=0.6, clip_on=False, zorder=5)
    lax.text(x_left + 0.04, HEADER_Y - 0.02,
             FAMILIES[fam]["label"],
             fontsize=8.8, fontweight='bold', color=fam_color, va='center')
    for j, n in enumerate(members):
        y = HEADER_Y - 0.11 - j * LINE_H
        lax.scatter(x_left + 0.018, y - 0.005,
                    s=(90 if marker == '*' else (70 if marker == 'D' else 60)),
                    c=colors[n], marker=marker,
                    edgecolor='black', linewidth=0.5, clip_on=False, zorder=5)
        lax.text(x_left + 0.048, y - 0.005, n, fontsize=7.6, va='center')

for ci in range(1, n_cols):
    lax.axvline(ci * COL_WIDTH, color='gray', alpha=0.25, linewidth=0.5)

plt.tight_layout()
out_pdf = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       'fig_param_pareto.pdf')
plt.savefig(out_pdf, bbox_inches='tight', pad_inches=0.15)
plt.savefig(out_pdf.replace('.pdf', '.png'), bbox_inches='tight', pad_inches=0.15, dpi=200)
print("saved", out_pdf)

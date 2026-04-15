"""Family-level contribution analysis: per-family distributions of latency,
accuracy, and parameters across the methods in our corpus.

Three side-by-side box plots, one per metric, colored by family.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from method_data import METHODS, FAMILIES

import matplotlib.pyplot as plt, matplotlib as mpl

mpl.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":",
    "axes.axisbelow": True, "figure.dpi": 150,
})

# Order families by typical latency target (low to high)
FAM_ORDER = ["efficient_iter", "efficient", "iterative", "3dcv", "foundation"]
FAM_LABELS = {
    "efficient_iter":  "Efficient\n+Found.\n(2024+)",
    "efficient":       "Efficient\n(pre-2024)",
    "iterative":       "Iterative\n(post-RAFT)",
    "3dcv":            "3D Cost\nVolume",
    "foundation":      "Foundation\nModels",
}

# Collect per-family metric arrays
def collect(metric):
    by_fam = {f: [] for f in FAM_ORDER}
    for n, m in METHODS.items():
        if m["family"] not in FAM_ORDER: continue
        v = m.get(metric)
        if v is not None:
            by_fam[m["family"]].append(v)
    return by_fam

lat = collect("latency_ms")
acc = collect("kitti_d1")
params = collect("params_m")

fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))

def boxplot(ax, data_dict, ylabel, title, log=False):
    positions = list(range(1, len(FAM_ORDER) + 1))
    data = [data_dict[f] for f in FAM_ORDER]
    colors = [FAMILIES[f]["color"] for f in FAM_ORDER]
    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.2),
                    boxprops=dict(linewidth=0.7),
                    whiskerprops=dict(linewidth=0.7),
                    capprops=dict(linewidth=0.7),
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    # Overlay individual points
    for pos, vals, color in zip(positions, data, colors):
        jitter = np.random.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter([pos + j for j in jitter], vals, c=color, s=22,
                   edgecolor='black', linewidth=0.4, alpha=0.85, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([FAM_LABELS[f] for f in FAM_ORDER], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    if log:
        ax.set_yscale('log')

boxplot(axes[0], lat, "Inference latency (ms)",
        "Latency by compression family", log=True)
boxplot(axes[1], acc, "KITTI 2015 D1-all (\\%)",
        "Accuracy by compression family", log=False)
boxplot(axes[2], params, "Parameters (M)",
        "Model size by compression family", log=True)

# Subtle annotation: arrows indicating preferred direction
for ax, txt in zip(axes, ["lower is better", "lower is better", "smaller is more deployable"]):
    ax.text(0.98, 0.97, txt, transform=ax.transAxes,
            ha='right', va='top', fontsize=7.5, style='italic', color='gray',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

fig.suptitle("Compression-family contribution analysis: distribution of metrics across the surveyed corpus",
             fontsize=12, y=1.02)
plt.tight_layout()
out_pdf = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fig_family_contribution.pdf')
plt.savefig(out_pdf, bbox_inches='tight', pad_inches=0.2)
plt.savefig(out_pdf.replace('.pdf', '.png'), bbox_inches='tight', pad_inches=0.2, dpi=200)
print("saved", out_pdf)

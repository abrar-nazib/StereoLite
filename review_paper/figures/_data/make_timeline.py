"""Evolution timeline.
- Categorical x-axis (equally spaced years).
- Lane labels INSIDE the colored band.
- Multi-column layout within each lane to keep figure compact.
- Each method placed at its actual year on the x-axis.
"""
import os, matplotlib.pyplot as plt, matplotlib as mpl
from collections import defaultdict

mpl.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "figure.dpi": 150,
})

# Each paradigm: (label, color, list of (year, method))
LANES = [
    ("Modern Edge (2024-2026)", "#8c564b",
     [(2024, "DTP"), (2025, "BANet"), (2025, "LightStereo"),
      (2025, "LiteAnyStereo"), (2026, "Pip-Stereo"),
      (2026, "Fast-FoundationStereo"), (2026, "GGEV")]),
    ("Foundation Era (2025)", "#d62728",
     [(2025, "DEFOM-Stereo"), (2025, "FoundationStereo"),
      (2025, "MonSter"), (2025, "Stereo-Anywhere"), (2025, "IGEV++")]),
    ("Iterative (2021-2024)", "#ff7f0e",
     [(2021, "RAFT-Stereo"), (2022, "CREStereo"),
      (2023, "IGEV-Stereo"), (2024, "Selective-Stereo")]),
    ("Early Efficient (2018-2021)", "#9467bd",
     [(2018, "StereoNet"), (2018, "MADNet"),
      (2019, "AnyNet"), (2019, "DeepPruner"),
      (2020, "AANet"), (2020, "Cascade-CV"),
      (2021, "HITNet"), (2021, "BGNet"), (2021, "CoEx")]),
    ("3D Cost Volume (2017-2022)", "#2ca02c",
     [(2017, "GC-Net"), (2018, "PSMNet"),
      (2019, "GA-Net"), (2019, "GwcNet"),
      (2020, "AANet"), (2022, "ACVNet")]),
    ("Early Deep (2016)", "#1f77b4",
     [(2016, "MC-CNN"), (2016, "DispNetC")]),
    ("Classical (pre-2016)", "#7f7f7f",
     [(2002, "Scharstein & Szeliski"), (2008, "SGM (Hirschmuller)")]),
]

# Categorical x positions
all_years = sorted({yr for _, _, evs in LANES for (yr, _) in evs})
year_to_x = {y: i for i, y in enumerate(all_years)}

# Inside each lane, group events by year. Stack them VERTICALLY at the same
# x position (with small offsets to disambiguate markers visually).
ROW_HEIGHT = 0.42
GAP = 0.20
LANE_TITLE_H = 0.28  # top space inside lane reserved for the lane title

# Determine row count per lane = max number of events in any single year
lane_geom = []
y_cursor = 0
for label, color, events in reversed(LANES):
    by_year = defaultdict(list)
    for yr, name in events:
        by_year[yr].append(name)
    max_rows = max(len(v) for v in by_year.values())
    h = LANE_TITLE_H + max_rows * ROW_HEIGHT
    lane_geom.append((label, color, by_year, max_rows, y_cursor, y_cursor + h))
    y_cursor += h + GAP
total_h = y_cursor

fig_w = 11.5
fig_h = max(4.5, 0.4 + total_h * 0.95)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

MARKER_SIZE = 60

for label, color, by_year, max_rows, y_bot, y_top in lane_geom:
    # Lane band
    ax.axhspan(y_bot, y_top, color=color, alpha=0.13, zorder=0)
    # Lane label INSIDE the band, top-left, on its own reserved strip
    ax.text(-0.45, y_top - LANE_TITLE_H * 0.55, label,
            ha='left', va='center',
            fontsize=10, color=color, fontweight='bold')
    # Plot events: each year stacks vertically below the title strip
    plot_top = y_top - LANE_TITLE_H
    for yr, names in by_year.items():
        x = year_to_x[yr]
        for slot, name in enumerate(names):
            y = plot_top - (slot + 0.5) * ROW_HEIGHT
            ax.scatter(x, y, s=MARKER_SIZE, c=color, edgecolor='black',
                       linewidth=0.6, zorder=4)
            ax.text(x + 0.13, y, name, ha='left', va='center',
                    fontsize=7.8, color='black',
                    bbox=dict(facecolor='white', alpha=0.95,
                              edgecolor=color, linewidth=0.5,
                              boxstyle='round,pad=0.16'),
                    zorder=5)

n_years = len(all_years)
ax.set_xlim(-0.7, n_years - 0.3)
ax.set_ylim(-0.25, total_h + 0.05)
ax.set_xticks(range(n_years))
ax.set_xticklabels([str(y) for y in all_years], fontsize=9)
ax.set_yticks([])
ax.tick_params(axis='x', length=4)
ax.plot([-0.7, n_years - 0.3], [-0.05, -0.05], color='black',
        linewidth=0.5, zorder=2)
ax.set_xlabel('Year', fontsize=10)
ax.set_title('Evolution of deep stereo matching: paradigm timeline (2002--2026)',
             fontsize=11.5, pad=8)

# Light vertical gridlines
for i in range(n_years):
    ax.axvline(i, color='gray', alpha=0.18, linewidth=0.4, zorder=1)

plt.tight_layout()
out_pdf = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fig_timeline.pdf')
plt.savefig(out_pdf, bbox_inches='tight', pad_inches=0.18)
plt.savefig(out_pdf.replace('.pdf', '.png'), bbox_inches='tight', pad_inches=0.18, dpi=200)
print("saved", out_pdf, "h=", fig_h)

"""Combined thesis Gantt chart (4th year odd + even semesters, 11 tasks).

Corrected version based on the user-supplied schedule
(/home/abrar/Downloads/Telegram Desktop/test_code.py). Only the output
paths/format, the x-axis label text, and the title text were changed;
colors, bar height, layout, and the month locator are unchanged.
"""
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

OUT = Path("/home/abrar/Research/stero_research_claude/thesis/book/figures")

# -----------------------------
# Combined Thesis Gantt Chart Data
# -----------------------------
tasks = [
    # 4th Year Odd Semester
    ("Topic Selection", "2025-07-22", "2025-07-31"),
    ("Literature Review", "2025-08-01", "2025-09-15"),
    ("Thesis Proposal", "2025-09-15", "2025-10-15"),
    ("Dataset Collection & Processing", "2025-10-15", "2025-11-28"),
    # 4th Year Even Semester
    ("Model selection and setup", "2025-12-01", "2026-01-20"),
    ("Pre-training and fine-tuning", "2025-12-15", "2026-03-15"),
    ("Real-data capture", "2026-01-15", "2026-03-25"),
    ("Ablation studies", "2026-02-15", "2026-04-30"),
    ("Benchmarking and 3D reconstruction", "2026-03-15", "2026-05-20"),
    ("Result analysis and evaluation", "2026-04-20", "2026-06-15"),
    ("Thesis writing and documentation", "2026-05-01", "2026-07-05"),
]

colors = [
    "#8DA0CB",  # Muted slate blue
    "#66C2A5",  # Mint green
    "#FC8D62",  # Salmon orange
    "#0B9E77",  # Deep Green
    "#0B7FAB",  # deep blue
    "#56B1DD",  # light blue
    "#0B9E77",  # green
    "#66C2A5",  # mint
    "#E69F00",  # orange
    "#D95F02",  # dark orange
    "#CC79A7",  # pink
]

# -----------------------------
# Convert dates
# -----------------------------
starts = [datetime.strptime(t[1], "%Y-%m-%d") for t in tasks]
ends = [datetime.strptime(t[2], "%Y-%m-%d") for t in tasks]
durations = [(end - start).days for start, end in zip(starts, ends)]
labels = [t[0] for t in tasks]

# -----------------------------
# Plot Setup
# -----------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 11

# DESIGN CHOICE: Reduced figsize height (from 7.0 to 5.5) to compact the vertical layout
fig, ax = plt.subplots(figsize=(12, 5.5))

y_pos = range(len(tasks))

for i, (start, duration) in enumerate(zip(starts, durations)):
    ax.barh(
        i,
        duration,
        left=start,
        height=0.75,  # DESIGN CHOICE: Increased bar thickness from 0.55 to 0.75 to close gaps
        color=colors[i],
        edgecolor="#333333",
        linewidth=0.7,
    )

# -----------------------------
# Axis formatting
# -----------------------------
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()

ax.set_xlim(datetime(2025, 7, 1), datetime(2026, 7, 20))

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

ax.grid(axis="x", color="#e6e6e6", linewidth=1)
ax.set_axisbelow(True)

ax.set_xlabel("Timeline (2025 to 2026)", fontsize=12, labelpad=12)

# Remove unnecessary borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", length=6, width=1)

# -----------------------------
# Title style
# -----------------------------
ax.set_title(
    "Complete Thesis Timeline (4th Year Odd and Even Semesters)",
    fontsize=14,
    fontweight="bold",
    pad=20,
)

# Readjust margins tightly around the newly condensed structure
plt.subplots_adjust(left=0.28, right=0.96, top=0.88, bottom=0.18)

# -----------------------------
# Export
# -----------------------------
plt.savefig(OUT / "fig_b_1_gantt_fixed.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig_b_1_gantt_fixed.png", dpi=300, bbox_inches="tight")

print("saved fig_b_1_gantt_fixed (.pdf, .png)")

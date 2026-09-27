"""Combined thesis Gantt chart for the seventh and eighth semesters."""
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[4] / "thesis/book/figures"

# -----------------------------
# Combined Thesis Gantt Chart Data
# -----------------------------
tasks = [
    # Fourth-year odd semester (seventh semester)
    ("Topic Selection", "2025-07-22", "2025-07-31"),
    ("Literature Review", "2025-08-01", "2025-09-15"),
    ("Thesis Proposal", "2025-09-15", "2025-10-15"),
    ("Dataset Collection & Processing", "2025-10-15", "2025-11-28"),
    # Main experimental period: December 2025 to July 2026
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
    "#0B9E77",  # Deep green
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
# Plot Setup
# -----------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "TeX Gyre Termes", "DejaVu Serif"]
plt.rcParams["font.size"] = 10.5

# Draw at the thesis text width so the type is not reduced again when
# LaTeX includes the figure at \textwidth.
fig, axes = plt.subplots(
    2, 1, figsize=(6.3, 6.0),
    gridspec_kw={"height_ratios": [0.82, 1.18], "hspace": 0.52},
)


def draw_panel(ax, panel_tasks, panel_colors, start, end, title):
    starts = [datetime.strptime(t[1], "%Y-%m-%d") for t in panel_tasks]
    ends = [datetime.strptime(t[2], "%Y-%m-%d") for t in panel_tasks]
    durations = [(finish - begin).days for begin, finish in zip(starts, ends)]

    for i, (begin, duration) in enumerate(zip(starts, durations)):
        ax.barh(i, duration, left=begin, height=0.62,
                color=panel_colors[i], edgecolor="#333333", linewidth=0.7)

    ax.set_yticks(range(len(panel_tasks)))
    ax.set_yticklabels([t[0] for t in panel_tasks], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e6e6e6", linewidth=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=9)


draw_panel(
    axes[0], tasks[:4], colors[:4],
    datetime(2025, 7, 1), datetime(2025, 12, 1),
    "Seventh semester: preparatory work (July–November 2025)",
)
draw_panel(
    axes[1], tasks[4:], colors[4:],
    datetime(2025, 12, 1), datetime(2026, 8, 1),
    "Main experimental period (December 2025–July 2026)",
)

fig.suptitle("Complete Thesis Timeline", fontsize=12, fontweight="bold", y=0.985)
fig.subplots_adjust(left=0.40, right=0.98, top=0.91, bottom=0.07)

# -----------------------------
# Export
# -----------------------------
plt.savefig(OUT / "fig_b_1_gantt_fixed.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig_b_1_gantt_fixed.png", dpi=300, bbox_inches="tight")

print("saved fig_b_1_gantt_fixed (.pdf, .png)")

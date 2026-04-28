"""Per-stage architecture diagrams for slides 7.2 / 7.3 / 7.4.

Wider canvas (5.5 in) and short, ASCII-only text — text was overflowing
on the previous, narrower 4.4 in version.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Wedge

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

CREAM   = "#F4EFE6"
PANEL   = "#FBF7EE"
INK     = "#1A1A1F"
SUBINK  = "#5A5550"
ACCENT  = "#C24A1C"
SOFT    = "#D9826A"
DOT_BLUE  = "#3B6FB6"
DOT_GREEN = "#3F7A48"
DOT_TEAL  = "#3F8C8E"
DOT_AMBER = "#A07A1F"

mpl.rcParams.update({
    "font.family": "DejaVu Serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "figure.dpi": 200,
})

# Canvas: 12 graph units wide x 7 tall — gives more room for labels.
W = 12.0
H = 7.0
FIGSIZE = (5.0, 2.92)         # ~ slide-half size when embedded


def _box(ax, x, y, w, h, *, fc, ec=INK, lw=1.0, radius=0.05):
    p = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)


def _txt(ax, x, y, s, *, size=9, color=INK, weight="normal",
         family="DejaVu Serif", ha="center", va="center",
         italic=False):
    style = "italic" if italic else "normal"
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
             ha=ha, va=va, fontfamily=family, fontstyle=style, zorder=4)


def _arrow(ax, x0, y0, x1, y1, *, color=ACCENT, lw=1.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=12, color=color,
        linewidth=lw, zorder=3))


def _frame():
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=CREAM)
    ax.set_facecolor(CREAM)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    return fig, ax


def _save(fig, name):
    out = OUT / name
    fig.savefig(out, bbox_inches="tight", pad_inches=0.10,
                 facecolor=CREAM, dpi=220)
    plt.close(fig)
    print(f"  -> {out}")


# --------------------------------------------------------------------------
# STAGE 2  ·  TILE HYPOTHESIS INIT
# --------------------------------------------------------------------------

def stage2_diagram():
    fig, ax = _frame()
#     _txt(ax, 6.0, 6.45, "STAGE 2  ·  TILE HYPOTHESIS INIT",
#           size=10, weight="bold", color=ACCENT,
#           family="DejaVu Sans Mono")

    # Two feature inputs on the left (boxes widened to fit captions)
    _box(ax, 0.40, 5.1, 2.10, 1.0, fc=DOT_BLUE, ec=INK, lw=0.8)
    _txt(ax, 1.45, 5.6, "left 1/16", size=8, color="white",
          weight="bold")
    _box(ax, 0.40, 3.0, 2.10, 1.0, fc=DOT_BLUE, ec=INK, lw=0.8)
    _txt(ax, 1.45, 3.5, "right 1/16", size=8, color="white",
          weight="bold")

    # Group correlation
    _arrow(ax, 2.55, 5.6, 3.2, 5.0)
    _arrow(ax, 2.55, 3.5, 3.2, 4.2)
    _box(ax, 3.2, 4.0, 2.3, 1.7, fc="#F0A56A", ec=INK, lw=0.8)
    _txt(ax, 4.35, 5.30, "Cost vol.", size=9.5, weight="bold")
    _txt(ax, 4.35, 4.80, "8 groups", size=8.5, color=SUBINK)
    _txt(ax, 4.35, 4.40, "24 disp", size=8.5, color=SUBINK)

    # 3D aggregator
    _arrow(ax, 5.55, 4.85, 6.20, 4.85)
    _box(ax, 6.20, 4.0, 2.0, 1.7, fc=DOT_GREEN, ec=INK, lw=0.8)
    _txt(ax, 7.20, 5.30, "3D agg", size=9, weight="bold", color="white")
    _txt(ax, 7.20, 4.80, "Conv3D", size=7, color="white")
    _txt(ax, 7.20, 4.40, "GN + SiLU", size=7, color="white")

    # Soft argmin
    _arrow(ax, 8.25, 4.85, 8.90, 4.85)
    _box(ax, 8.90, 4.0, 2.7, 1.7, fc=SOFT, ec=INK, lw=0.8)
    _txt(ax, 10.25, 5.30, "soft argmin", size=8, weight="bold",
          color="white")
    _txt(ax, 10.25, 4.80, "init d", size=9, color="white")
    _txt(ax, 10.25, 4.40, "+ confidence", size=8.5, color="white")

    # Tile state row at bottom
    _txt(ax, 6.0, 2.55, "PER TILE STATE  Tk",
          size=9, weight="bold", color=ACCENT,
          family="DejaVu Sans Mono")
    syms = [("d", "disparity"),
             ("sx", "slope x"),
             ("sy", "slope y"),
             ("f", "16 ch"),
             ("c", "conf")]
    sx0 = 0.8
    sw = 2.10
    sgap = 0.10
    for i, (sym, lbl) in enumerate(syms):
        x = sx0 + i * (sw + sgap)
        _box(ax, x, 0.65, sw, 1.50, fc=PANEL, ec=SUBINK, lw=0.6)
        _txt(ax, x + sw / 2, 1.75, sym, size=14, weight="bold",
              color=ACCENT)
        _txt(ax, x + sw / 2, 1.10, lbl, size=8.5, color=SUBINK)

    _save(fig, "stage2_init.png")


# --------------------------------------------------------------------------
# STAGE 3  ·  ITERATIVE REFINEMENT
# --------------------------------------------------------------------------

def stage3_diagram():
    fig, ax = _frame()
#     _txt(ax, 6.0, 6.45, "STAGE 3  ·  ITERATIVE REFINEMENT",
#           size=10, weight="bold", color=ACCENT,
#           family="DejaVu Sans Mono")

    # Three scale columns
    scales = [("1/16", 2, "0.13 M", DOT_BLUE),
               ("1/8",  3, "0.08 M", DOT_TEAL),
               ("1/4",  3, "0.07 M", DOT_GREEN)]
    cw = 2.37
    cx = 0.55
    cgap = 1.90
    for i, (scale, iters, params, color) in enumerate(scales):
        x = cx + i * (cw + cgap)
        _box(ax, x, 2.85, cw, 2.85, fc=PANEL, ec=color, lw=1.6)
        _txt(ax, x + cw / 2, 5.20, f"refine {scale}",
              size=8, weight="bold", color=INK)
        _txt(ax, x + cw / 2, 4.45, f"x {iters}",
              size=24, weight="bold", color=color)
        _txt(ax, x + cw / 2, 3.55, params,
              size=10, color=SUBINK, family="DejaVu Sans Mono")
        _txt(ax, x + cw / 2, 3.10, "trainable",
              size=8, color=SUBINK, italic=True)
        if i < 2:
            ax_x = x + cw + 0.10
            _arrow(ax, ax_x, 4.27, ax_x + cgap - 0.20, 4.27,
                    color=ACCENT, lw=1.8)
            _txt(ax, ax_x + (cgap - 0.20) / 2, 4.65,
                  "plane up", size=8.5, color=ACCENT, italic=True)

    # Bottom: per-iteration mechanism, 4 short steps
    _txt(ax, 6.0, 2.30, "EACH ITERATION",
          size=9, weight="bold", color=ACCENT,
          family="DejaVu Sans Mono")
    steps = ["warp R", "concat", "trunk", "predict"]
    sw = 1.89
    sx0 = 0.65
    sgap = 1.10
    for i, st in enumerate(steps):
        x = sx0 + i * (sw + sgap)
        _box(ax, x, 0.55, sw, 1.20, fc=CREAM, ec=SUBINK, lw=0.5)
        _txt(ax, x + sw / 2, 1.15, st, size=10, color=INK)
        if i < len(steps) - 1:
            ax_x = x + sw + 0.04
            _arrow(ax, ax_x, 1.15, ax_x + sgap - 0.08, 1.15,
                    color=SUBINK, lw=1.1)

    _save(fig, "stage3_refine.png")


# --------------------------------------------------------------------------
# STAGE 4  ·  CONVEX UPSAMPLE
# --------------------------------------------------------------------------

def stage4_diagram():
    fig, ax = _frame()
    _txt(ax, 6.0, 6.45, "9-NEIGHBOUR WEIGHTED AVERAGE",
          size=10, weight="bold", color=ACCENT,
          family="DejaVu Sans Mono")

    # 3x3 grid
    cell = 1.05
    grid_left = 6.0 - 1.5 * cell
    grid_bot = 2.7
    weights = [["w1", "w2", "w3"],
                ["w4", " * ", "w6"],
                ["w7", "w8", "w9"]]
    for i in range(3):
        for j in range(3):
            x = grid_left + j * cell
            y = grid_bot + (2 - i) * cell
            is_centre = (i == 1 and j == 1)
            fc = ACCENT if is_centre else PANEL
            _box(ax, x, y, cell - 0.06, cell - 0.06, fc=fc,
                  ec=INK, lw=0.7)
            sym = weights[i][j]
            _txt(ax, x + (cell - 0.06) / 2, y + (cell - 0.06) / 2, sym,
                  size=11.5, weight="bold",
                  color="white" if is_centre else INK)

    _txt(ax, 6.0, 2.20, "fine_d  =  sum( wi * d_coarse(i) )",
          size=11, weight="bold", color=INK,
          family="DejaVu Sans Mono")
    _txt(ax, 6.0, 1.65, "sum(wi) = 1   (convex combination)",
          size=9, color=SUBINK, italic=True,
          family="DejaVu Sans Mono")
    _txt(ax, 6.0, 0.85,
          "applied twice:  1/4 -> 1/2  ->  full",
          size=10, color=ACCENT, weight="bold",
          family="DejaVu Sans Mono")

    _save(fig, "stage4_upsample.png")


# --------------------------------------------------------------------------
# SUPERVISION
# --------------------------------------------------------------------------

def supervision_diagram():
    fig, ax = _frame()
    _txt(ax, 6.0, 6.45, "MULTI SCALE LOSS  ·  THREE TERMS",
          size=10, weight="bold", color=ACCENT,
          family="DejaVu Sans Mono")

    # Equation in dark band — single line at smaller size so it fits.
    _box(ax, 0.4, 5.30, 11.2, 0.75, fc=INK, ec=INK, lw=0.5,
          radius=0.04)
    _txt(ax, 6.0, 5.67,
          "L  =  sum_k wk [ L1 + a*Lgrad + b*Lhinge ]  +  c*Lsmooth",
          size=7, weight="bold", color="white",
          family="DejaVu Sans Mono")

    # Three term cards
    cards = [
        ("L1",     "PIXEL ERROR",
         "Mean abs error\non valid GT.",
         DOT_BLUE),
        ("Lgrad",  "SOBEL",
         "L1 on dx, dy of d.\nKeeps edges crisp.",
         DOT_TEAL),
        ("Lhinge", "BAD-1 HINGE",
         "Penalty when\n|err| > 1 px.",
         DOT_AMBER),
    ]
    cw = 3.30
    gap = 0.45
    cx0 = (W - 3 * cw - 2 * gap) / 2
    for i, (sym, title, body, color) in enumerate(cards):
        x = cx0 + i * (cw + gap)
        _box(ax, x, 1.65, cw, 2.95, fc=PANEL, ec=color, lw=1.4)
        _txt(ax, x + cw / 2, 4.00, sym, size=15, weight="bold",
              color=color)
        _txt(ax, x + cw / 2, 3.20, title, size=8.5, weight="bold",
              color=ACCENT, family="DejaVu Sans Mono")
        _txt(ax, x + cw / 2, 2.30, body, size=7, color=INK)

    # Defaults strip
    _txt(ax, 6.0, 1.10,
          "a = 0.5    b = 0.3    c = 0.02",
          size=9.5, color=SUBINK, family="DejaVu Sans Mono")
    _txt(ax, 6.0, 0.55,
          "wk = (0.2, 1.0, 0.3, 0.5, 0.7, 1.0)",
          size=8.5, color=SUBINK, family="DejaVu Sans Mono")

    _save(fig, "supervision_loss.png")


# --------------------------------------------------------------------------
# PARAMETER BUDGET
# --------------------------------------------------------------------------

def budget_diagram():
    fig, ax = _frame()
    _txt(ax, 6.0, 6.50, "PARAMETER BUDGET  ·  0.87 M TOTAL",
          size=10, weight="bold", color=ACCENT,
          family="DejaVu Sans Mono")

    parts = [
        ("Encoder",      0.54, DOT_BLUE),
        ("Refine 1/16",  0.13, DOT_TEAL),
        ("Refine 1/8",   0.08, DOT_GREEN),
        ("Refine 1/4",   0.07, DOT_AMBER),
        ("Tile init",    0.03, SOFT),
        ("Upsample",     0.02, ACCENT),
    ]
    total = sum(p[1] for p in parts)
    cx, cy = 2.6, 3.4
    r_outer = 1.85
    r_inner = 1.05
    start = 90.0
    for label, val, color in parts:
        sweep = 360.0 * val / total
        wedge = Wedge((cx, cy), r_outer,
                       theta1=start - sweep, theta2=start,
                       facecolor=color, edgecolor=CREAM, linewidth=1.5,
                       width=r_outer - r_inner, zorder=3)
        ax.add_patch(wedge)
        start -= sweep
    _txt(ax, cx, cy + 0.20, "0.87 M",
          size=15, weight="bold", color=INK)
    _txt(ax, cx, cy - 0.35, "trainable",
          size=8.5, color=SUBINK)

    # Right-side legend with three columns
    swatch_x = 5.20
    label_x  = 5.65
    value_x  = 9.95
    pct_x    = 11.50
    ly = 5.60
    row_h = 0.62
    for label, val, color in parts:
        ax.add_patch(Rectangle((swatch_x, ly - 0.18), 0.35, 0.36,
                                  facecolor=color, edgecolor=INK,
                                  linewidth=0.5, zorder=3))
        _txt(ax, label_x, ly, label,
              size=10, color=INK, ha="left")
        _txt(ax, value_x, ly,
              f"{val:.2f} M",
              size=10, color=SUBINK, ha="right",
              family="DejaVu Sans Mono")
        pct = 100 * val / total
        _txt(ax, pct_x, ly,
              f"{pct:>3.0f}%",
              size=10, color=SUBINK, ha="right",
              family="DejaVu Sans Mono")
        ly -= row_h

    _txt(ax, 6.0, 0.55,
          "Checkpoint on disk: 8.7 MB (fp32)",
          size=9.5, color=ACCENT, italic=True)

    _save(fig, "param_budget.png")


# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# DISCUSSION  ·  EPE TRAJECTORY (slide 10)
# --------------------------------------------------------------------------

def epe_trajectory_diagram():
    """Three milestone cards joined by arrows, showing the path
    synthetic baseline -> indoor fine tune -> projected full pretrain.
    Used on the discussion slide to visualise potential."""
    fig, ax = _frame()
    _txt(ax, 6.0, 6.45, "EPE TRAJECTORY  ·  SCENE FLOW DRIVING 200 VAL",
          size=10, weight="bold", color=ACCENT,
          family="DejaVu Sans Mono")

    cards = [
        ("CURRENT", "1.54", "px",
         "Synthetic only.\n12% of SF corpus.",
         DOT_BLUE, False),
        ("FINE TUNED", "0.515", "px",
         "Real indoor.\n997 pseudo GT.",
         DOT_GREEN, False),
        ("PROJECTED", "~0.71", "px",
         "Full SF corpus.\nA100 class GPU.",
         ACCENT, True),
    ]

    cw = 3.30
    cgap = 0.50
    cx0 = (W - 3 * cw - 2 * cgap) / 2
    cy = 1.50
    ch = 4.20

    for i, (title, big, unit, sub, color, dashed) in enumerate(cards):
        x = cx0 + i * (cw + cgap)
        # Card border (dashed for projected)
        if dashed:
            rect = Rectangle((x, cy), cw, ch,
                              facecolor=PANEL, edgecolor=color,
                              linewidth=1.8, linestyle="--",
                              zorder=2)
            ax.add_patch(rect)
        else:
            _box(ax, x, cy, cw, ch, fc=PANEL, ec=color, lw=1.8)
        _txt(ax, x + cw / 2, cy + ch - 0.55, title,
              size=10, weight="bold", color=ACCENT,
              family="DejaVu Sans Mono")
        _txt(ax, x + cw / 2, cy + ch - 1.85, big,
              size=22, weight="bold", color=color)
        _txt(ax, x + cw / 2, cy + ch - 2.65, unit,
              size=11, color=color)
        _txt(ax, x + cw / 2, cy + 0.95, sub,
              size=9, color=SUBINK)

        if i < 2:
            ax_x = x + cw + 0.05
            _arrow(ax, ax_x, cy + ch / 2,
                    ax_x + cgap - 0.10, cy + ch / 2,
                    color=ACCENT, lw=2.0)

    _txt(ax, 6.0, 0.85,
          "0.515 px on real indoor data already beats the 1.54 px "
          "synthetic baseline by 3x.",
          size=10, color=ACCENT, weight="bold", italic=True)

    _save(fig, "epe_trajectory.png")


# --------------------------------------------------------------------------
# STAGE 1  ·  ENCODER (very simple: stereo input -> shared encoder -> 4 scales)
# --------------------------------------------------------------------------

def stage1_encoder_diagram():
    fig, ax = _frame()
#     _txt(ax, 6.0, 6.45, "STAGE 1  ·  SHARED ENCODER",
#           size=10, weight="bold", color=ACCENT,
#           family="DejaVu Sans Mono")

    # Two small input boxes on the left
    _box(ax, 0.50, 4.40, 1.50, 1.10, fc=DOT_BLUE, ec=INK, lw=0.8)
    _txt(ax, 1.25, 4.95, "I_L", size=14, weight="bold", color="white")
    _box(ax, 0.50, 2.50, 1.50, 1.10, fc=DOT_BLUE, ec=INK, lw=0.8)
    _txt(ax, 1.25, 3.05, "I_R", size=14, weight="bold", color="white")

    # Single shared encoder block — short label, big box
    _arrow(ax, 2.05, 4.95, 3.30, 4.40)
    _arrow(ax, 2.05, 3.05, 3.30, 3.60)
    _box(ax, 3.30, 2.85, 2.85, 2.30, fc="#F0A56A", ec=INK, lw=0.8)
    _txt(ax, 4.725, 4.55, "MobileNetV2", size=9, weight="bold")
    _txt(ax, 4.725, 4.05, "(truncated)",
          size=9, color=SUBINK, italic=True)
    _txt(ax, 4.725, 3.55, "shared L+R",
          size=9, color=SUBINK, italic=True)
    _txt(ax, 4.725, 3.05, "0.54 M",
          size=11, color=ACCENT, weight="bold",
          family="DejaVu Sans Mono")

    # Output: 4 feature maps in a row, pushed right to leave room for
    # a clearly visible arrow from the encoder block.
    scales = [("1/2", "16 ch"), ("1/4", "24 ch"),
               ("1/8", "32 ch"), ("1/16", "96 ch")]
    fw = 1.15
    fh = 1.10
    fgap = 0.10
    fx0 = 7.00
    fy = 3.45
    _txt(ax, fx0 + (4 * fw + 3 * fgap) / 2, 5.30,
          "feature pyramid",
          size=9, color=SUBINK, italic=True)
    for i, (sc, ch) in enumerate(scales):
        x = fx0 + i * (fw + fgap)
        _box(ax, x, fy, fw, fh, fc=DOT_BLUE, ec=INK, lw=0.7)
        _txt(ax, x + fw / 2, fy + fh / 2 + 0.18, sc,
              size=11, weight="bold", color="white")
        _txt(ax, x + fw / 2, fy + fh / 2 - 0.22, ch,
              size=8, color="white",
              family="DejaVu Sans Mono")

    # Long visible arrow from encoder (ends at x=6.15) to first feature
    # map (starts at fx0=7.00).
    _arrow(ax, 6.20, 4.00, 6.95, 4.00, color=ACCENT, lw=1.6)

    # Bottom commentary
    _txt(ax, 6.0, 2.05,
          "Truncated after stage four (saves 1 M params).",
          size=10.5, color=INK, italic=True)
    _txt(ax, 6.0, 1.55,
          "Features at 1/2, 1/4, 1/8, 1/16 feed the four refinement "
          "stages.",
          size=10, color=SUBINK, italic=True)

    _save(fig, "stage1_encoder.png")


def methodology_diagram():
    """Two-track flowchart: training pipeline (top) + inference
    pipeline (bottom), linked by a 'load weights' connector.
    Used on slide 11 (Methodology).  Wide canvas (5.5 in) so the
    figure fills the slide-width area when embedded."""
    # Wider canvas, taller to fit two rows + connector + footnote
    fig, ax = plt.subplots(figsize=(11.0, 5.4), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    W2, H2 = 22.0, 11.0
    ax.set_xlim(0, W2); ax.set_ylim(0, H2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

    # ---- Title strip ----
#     _txt(ax, W2 / 2, H2 - 0.55, "TRAINING + INFERENCE PIPELINES",
#           size=11, weight="bold", color=ACCENT,
#           family="DejaVu Sans Mono")

    # ---- Geometry ----
    n = 5
    box_w = 3.2
    box_h = 1.80
    gap = (W2 - 2.0 - n * box_w) / (n - 1)
    row_x0 = 1.0

    NAVY_FILL = "#1F2C4E"
    NAVY_LITE = "#2E467A"
    INF_BORDER = ACCENT
    INF_FILL = "#FBF1E5"
    BAND_TRAIN = "#ECEDF2"
    BAND_INF = "#FBF1E5"

    # ---- Training row ----
    train_y = H2 - 4.2          # box top y in graph units (note ax y axis up)
    train_band_y = train_y - 0.40
    train_band_h = box_h + 0.85

    # Band background
    ax.add_patch(FancyBboxPatch((row_x0 - 0.40, train_band_y),
        W2 - 2.0 + 0.80, train_band_h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=0.6, edgecolor="#D7D9DD",
        facecolor=BAND_TRAIN, zorder=1))
    # Row label (left, vertical)
#     _txt(ax, row_x0 - 0.10, train_y + box_h / 2, "TRAINING",
#           size=10, weight="bold", color="#1F2C4E", ha="right",
#           family="DejaVu Sans Mono")

    train_steps = [
        ("Scene Flow", "synthetic\nstereo dataset"),
        ("Pretrain",   "30 epochs,\n2 × T4 (Kaggle)"),
        ("Foundation\nteacher",
                       "FoundationStereo\npseudo-disparity"),
        ("Finetune",   "indoor pseudo-GT,\nmulti-scale loss"),
        ("Checkpoint", "0.87 M params,\n8.7 MB"),
    ]
    train_xs = []
    for i, (title, sub) in enumerate(train_steps):
        x = row_x0 + i * (box_w + gap)
        train_xs.append(x)
        ax.add_patch(FancyBboxPatch((x, train_y), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.0, edgecolor=NAVY_FILL,
            facecolor=NAVY_FILL, zorder=2))
        _txt(ax, x + box_w / 2, train_y + box_h - 0.45, title,
              size=11, weight="bold", color="white",
              family="DejaVu Serif")
        _txt(ax, x + box_w / 2, train_y + box_h / 2 - 0.25, sub,
              size=7, color="#CFD3DD",
              family="DejaVu Serif")
        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (x + box_w + 0.05, train_y + box_h / 2),
                (x + box_w + gap - 0.10, train_y + box_h / 2),
                arrowstyle="-|>", mutation_scale=14,
                color="#3B5078", linewidth=2.0, zorder=3))

    # ---- Inference row ----
    infer_y = 1.5
    infer_band_y = infer_y - 0.40
    infer_band_h = box_h + 0.85

    ax.add_patch(FancyBboxPatch((row_x0 - 0.40, infer_band_y),
        W2 - 2.0 + 0.80, infer_band_h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=0.6, edgecolor="#E5D7C5",
        facecolor=BAND_INF, zorder=1))
#     _txt(ax, row_x0 - 0.10, infer_y + box_h / 2, "INFERENCE",
#           size=10, weight="bold", color=ACCENT, ha="right",
#           family="DejaVu Sans Mono")

    infer_steps = [
        ("Stereo\ncamera",  "AR0144,\nrectified pair"),
        ("Pre-process",     "BGR → tensor,\ncrop / pad"),
        ("StereoLite",      "encoder → tile init,\nrefine × 8, upsample"),
        ("Disparity",       "px-resolution\nleft-frame map"),
        ("Depth + 3D",      "triangulate →\nOpen3D point cloud"),
    ]
    infer_xs = []
    for i, (title, sub) in enumerate(infer_steps):
        x = row_x0 + i * (box_w + gap)
        infer_xs.append(x)
        ax.add_patch(FancyBboxPatch((x, infer_y), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.4, edgecolor=INF_BORDER,
            facecolor=INF_FILL, zorder=2))
        _txt(ax, x + box_w / 2, infer_y + box_h - 0.45, title,
              size=11, weight="bold", color=INK,
              family="DejaVu Serif")
        _txt(ax, x + box_w / 2, infer_y + box_h / 2 - 0.25, sub,
              size=7, color=SUBINK,
              family="DejaVu Serif")
        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (x + box_w + 0.05, infer_y + box_h / 2),
                (x + box_w + gap - 0.10, infer_y + box_h / 2),
                arrowstyle="-|>", mutation_scale=14,
                color=ACCENT, linewidth=2.0, zorder=3))

    # ---- Vertical "load weights" connector ----
    cp_cx = train_xs[-1] + box_w / 2
    sl_cx = infer_xs[2] + box_w / 2
    cp_bottom = train_y
    sl_top = infer_y + box_h
    mid_y = (cp_bottom + sl_top) / 2

    # 3-segment polyline  (down, left, down with arrowhead)
    ax.plot([cp_cx, cp_cx], [cp_bottom, mid_y],
             color=ACCENT, linewidth=2.0, zorder=3)
    ax.plot([cp_cx, sl_cx], [mid_y, mid_y],
             color=ACCENT, linewidth=2.0, zorder=3)
    ax.add_patch(FancyArrowPatch((sl_cx, mid_y), (sl_cx, sl_top + 0.05),
        arrowstyle="-|>", mutation_scale=14,
        color=ACCENT, linewidth=2.0, zorder=3))
    # Annotation
    _txt(ax, (cp_cx + sl_cx) / 2, mid_y + 0.30, "load weights",
          size=9, color=ACCENT, italic=True,
          family="DejaVu Serif")

    # ---- Footnote ----
#     _txt(ax, W2 / 2, 0.35,
#           "Training is offline; the trained checkpoint is loaded once and reused at inference.",
#           size=9.5, italic=True, color=SUBINK,
#           family="DejaVu Serif")

    out = OUT / "methodology_pipeline.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.10,
                 facecolor=CREAM, dpi=220)
    plt.close(fig)
    print(f"  -> {out}")


if __name__ == "__main__":
    stage1_encoder_diagram()
    stage2_diagram()
    stage3_diagram()
    stage4_diagram()
    supervision_diagram()
    budget_diagram()
    epe_trajectory_diagram()
    methodology_diagram()

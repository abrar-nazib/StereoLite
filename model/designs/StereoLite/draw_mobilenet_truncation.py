"""Render a 3-panel diagram explaining the MobileNetV2 truncation fix.

Top:    What we asked timm for (4 feature outputs at 1/2..1/16)
Middle: What timm actually built (full MobileNetV2 with 2 dead blocks)
Bottom: After the fix (explicit truncation, all blocks alive)

Run:
    python3 model/designs/d1_tile/draw_mobilenet_truncation.py
Output:
    model/designs/d1_tile/mobilenet_truncation.png
    model/designs/d1_tile/mobilenet_truncation.pdf
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


C_INPUT = "#dee2e6"
C_ALIVE = "#a8e6cf"     # green - blocks doing useful work
C_DEAD = "#ffadad"      # red - blocks computed but ignored
C_HOOK = "#ffd166"      # yellow - feature hooks
C_OUT = "#bdb2ff"       # purple - what we use downstream
C_ANNOT = "#1a1a2e"


def block(ax, x, y, w, h, text, color, ec="#333", lw=1.2, fs=8.0,
          alpha=1.0, ls="solid"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=lw, edgecolor=ec, facecolor=color, alpha=alpha,
        linestyle=ls)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=C_ANNOT, zorder=5)


def arrow(ax, x0, y0, x1, y1, color="#333", lw=1.0, style="-|>"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                         mutation_scale=10, linewidth=lw, color=color, zorder=4)
    ax.add_patch(a)


def label(ax, x, y, text, fs=9, color=C_ANNOT, ha="center", va="center",
          weight="normal", style="normal"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color,
            weight=weight, style=style)


# Block widths/positions used in all three panels for visual alignment
BLOCK_W = 1.05
BLOCK_H = 0.55
BLOCK_GAP = 0.20
LEFT_X = 0.4
LEFT_Y_BOTTOM = 0.0  # placeholder, set per panel

# MobileNetV2-100 block sequence (block index -> output stride)
BLOCKS = [
    (0, "block 0\nstride /1\n16ch"),
    (1, "block 1\nstride /2\n24ch"),
    (2, "block 2\nstride /4\n32ch"),
    (3, "block 3\nstride /8\n64ch"),
    (4, "block 4\nstride /16\n96ch"),
    (5, "block 5\nstride /32\n160ch"),
    (6, "block 6\nstride /32\n320ch"),
]
# Stage feature indices (out_indices=(0,1,2,3) hooks these)
STAGE_HOOK_BLOCK = {0: 0, 1: 1, 2: 2, 3: 4}
# Stage labels
STAGE_LABELS = {
    0: "f2 (1/2)\n16 ch",
    1: "f4 (1/4)\n24 ch",
    2: "f8 (1/8)\n32 ch",
    3: "f16 (1/16)\n96 ch",
}


def draw_blocks(ax, y, alive_mask, title, subtitle, show_dead=False):
    """alive_mask: list of bool, one per block. True = used downstream."""
    label(ax, 0.4, y + 1.5, title, fs=12, weight="bold", ha="left")
    if subtitle:
        label(ax, 0.4, y + 1.18, subtitle, fs=9, color="#666", ha="left",
              style="italic")

    # Input image
    block(ax, 0.4, y - 0.2, 0.95, BLOCK_H, "left img\n(3, H, W)", C_INPUT,
          fs=8)
    last_x = 0.4 + 0.95

    block_centers = []
    for i, (idx, txt) in enumerate(BLOCKS):
        x = last_x + BLOCK_GAP + i * (BLOCK_W + BLOCK_GAP)
        if alive_mask[i]:
            color = C_ALIVE
            ec = "#2d6a4f"
            lw = 1.4
        else:
            color = C_DEAD
            ec = "#9d0208"
            lw = 1.6
        block(ax, x, y - 0.2, BLOCK_W, BLOCK_H, txt, color, ec=ec, lw=lw,
              fs=7.5)
        block_centers.append(x + BLOCK_W / 2)
        # Arrow from previous to this block
        if i == 0:
            arrow(ax, last_x, y + 0.05, x, y + 0.05)
        else:
            prev_x = x - BLOCK_GAP
            arrow(ax, prev_x, y + 0.05, x, y + 0.05)

    # Hook callouts: for each stage in STAGE_HOOK_BLOCK, draw a yellow
    # hook box above the corresponding block
    for stage, src_block in STAGE_HOOK_BLOCK.items():
        if not alive_mask[src_block]:
            continue
        cx = block_centers[src_block]
        block(ax, cx - 0.55, y + 0.85, 1.1, 0.45, STAGE_LABELS[stage],
              C_HOOK, fs=7, ec="#b08900")
        arrow(ax, cx, y + 0.35, cx, y + 0.85, lw=0.9, color="#b08900")

    # Mark dead blocks with a callout
    if show_dead:
        dead_centers = [block_centers[i] for i, a in enumerate(alive_mask)
                         if not a]
        if dead_centers:
            mid = (dead_centers[0] + dead_centers[-1]) / 2
            block(ax, mid - 1.6, y - 1.25, 3.2, 0.55,
                  "still computed every forward pass\n"
                  "but output goes nowhere -> 0 gradient",
                  "#ffd6d6", ec="#9d0208", lw=1.0, fs=7.5)
            arrow(ax, mid, y - 0.7, mid, y - 0.25, color="#9d0208", lw=1.0)

    return block_centers


def main():
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 12.5)
    ax.axis("off")

    # PANEL 1 (top): What we asked for
    y1 = 10
    draw_blocks(
        ax, y1,
        alive_mask=[True] * 5 + [False, False],
        title="(1) What we asked timm for",
        subtitle="features_only=True, out_indices=(0, 1, 2, 3) -> "
                 "give me 4 feature maps at 1/2, 1/4, 1/8, 1/16",
        show_dead=False)
    # Re-render with all alive (mental model)
    label(ax, 12.5, y1 - 0.2 + BLOCK_H / 2,
          "We need\nblocks 0-4\nonly",
          fs=8, ha="left", color="#2d6a4f", weight="bold")

    # PANEL 2 (middle): What timm actually built
    y2 = 6.5
    draw_blocks(
        ax, y2,
        alive_mask=[True] * 5 + [False, False],
        title="(2) What timm actually built",
        subtitle="features_only=True is a HOOK wrapper. The full backbone "
                 "is built and runs. Hooks just steal outputs at the right depths.",
        show_dead=True)

    # PANEL 3 (bottom): After our fix
    y3 = 2.5
    draw_blocks(
        ax, y3,
        alive_mask=[True] * 5 + [None, None],  # last 2 not drawn
        title="(3) After our fix: explicit truncation",
        subtitle="self.backbone.blocks = self.backbone.blocks[:5]    -> "
                 "blocks 5 & 6 are deleted, not just unused.",
        show_dead=False)

    # Bottom panel: re-do drawing because alive_mask=None doesn't draw
    # Easier: just overlay grey ghost outlines for dropped blocks
    for i in (5, 6):
        idx, txt = BLOCKS[i]
        last_x = 0.4 + 0.95
        x = last_x + BLOCK_GAP + i * (BLOCK_W + BLOCK_GAP)
        block(ax, x, y3 - 0.2, BLOCK_W, BLOCK_H,
              "deleted",
              "#f8f9fa", ec="#adb5bd", lw=1.0, fs=8.5, ls="dashed")

    # Right-side annotation
    label(ax, 12.5, y3 - 0.2 + BLOCK_H / 2,
          "0.874 M\ntrainable\n(was 2.14 M)",
          fs=8.5, ha="left", color="#2d6a4f", weight="bold")

    # Big title
    label(ax, 6.5, 12.0,
          "MobileNetV2 inside StereoLite v8 -> v9   (the dead-block fix)",
          fs=14, weight="bold")

    # Bottom legend
    leg_y = 0.6
    items = [
        ("alive (used in forward + backward)", C_ALIVE, "#2d6a4f"),
        ("dead (forward only, no gradient)", C_DEAD, "#9d0208"),
        ("hook (output we actually use)", C_HOOK, "#b08900"),
        ("input image", C_INPUT, "#666"),
    ]
    x = 0.4
    for name, col, ec in items:
        block(ax, x, leg_y - 0.18, 0.32, 0.36, "", col, ec=ec, lw=1.0)
        label(ax, x + 0.4, leg_y, name, fs=8.5, ha="left")
        x += 3.2

    out_dir = os.path.dirname(os.path.abspath(__file__))
    png = os.path.join(out_dir, "mobilenet_truncation.png")
    pdf = os.path.join(out_dir, "mobilenet_truncation.pdf")
    fig.savefig(png, dpi=170, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.15)
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()

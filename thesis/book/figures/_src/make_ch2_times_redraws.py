"""Create editable Times New Roman redraws for Chapter 2 diagrams.

The architecture figures are intentionally concise adaptations of the cited
papers, rather than screenshots.  They retain the mechanisms discussed in the
chapter while keeping labels legible at thesis-page width.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from drawio_lib import (  # noqa: E402
    BLUE_F, BLUE_S, EDGE, GREEN_F, GREEN_S, GREY_F, GREY_S,
    LAV_F, LAV_S, ORAN_F, ORAN_S, RED_F, RED_S, YEL_F, YEL_S, D,
)

OUT = SRC.parent


def census() -> None:
    d = D("census-times", 940, 330)
    d.text(20, 8, 900, 30, "Census transform and Hamming matching cost",
           fs=27, bold=True)

    def grid(x: int, title: str, values: list[int], fill: str) -> str:
        d.text(x, 48, 240, 24, title, fs=25, bold=True)
        cell = 58
        ids = []
        for row in range(3):
            for col in range(3):
                i = row * 3 + col
                ids.append(d.box(x + col * cell, 82 + row * cell,
                                 cell, cell, str(values[i]),
                                 fill=fill if i == 4 else "#ffffff",
                                 stroke=EDGE, rounded=0, fs=23))
        return ids[4]

    grid(42, "Left patch", [124, 74, 32, 88, 90, 15, 210, 55, 120], YEL_F)
    grid(724, "Right patch", [130, 70, 40, 92, 90, 10, 200, 60, 118], GREEN_F)
    d.box(350, 92, 240, 82,
          "Compare each neighbour<br>with the centre pixel",
          fill=BLUE_F, stroke=BLUE_S, fs=25)
    d.text(350, 180, 240, 48,
           "1 if neighbour ≥ centre<br>0 otherwise", fs=22)
    d.text(35, 274, 250, 28, "Census string: 101100100",
           fs=23, color="#0072b2")
    d.text(690, 274, 250, 28, "Census string: 101101100",
           fs=23, color="#0072b2")
    d.text(330, 258, 280, 54,
           "Hamming distance = 1<br>(matching cost)", fs=25, bold=True,
           color="#b22222")
    d.save(OUT / "fig_2_census_times.drawio")


def psmnet() -> None:
    d = D("psmnet-adapted", 1080, 270)
    left = d.box(25, 45, 130, 64, "Left image", fill=GREY_F,
                 stroke=GREY_S, fs=25)
    right = d.box(25, 160, 130, 64, "Right image", fill=GREY_F,
                  stroke=GREY_S, fs=25)
    enc = d.box(205, 86, 180, 96,
                "Shared CNN<br>feature extraction", fill=BLUE_F,
                stroke=BLUE_S, fs=25)
    spp = d.box(430, 86, 170, 96,
                "Spatial pyramid<br>pooling (SPP)", fill=GREEN_F,
                stroke=GREEN_S, fs=25)
    vol = d.box(645, 86, 150, 96,
                "Concatenation<br>cost volume", fill=ORAN_F,
                stroke=ORAN_S, fs=25)
    hour = d.box(840, 72, 190, 124,
                 "Stacked 3-D<br>hourglass modules<br>+ supervision",
                 fill=LAV_F, stroke=LAV_S, fs=25)
    for src in (left, right):
        d.edge(src, enc)
    d.edge(enc, spp)
    d.edge(spp, vol)
    d.edge(vol, hour)
    d.text(858, 212, 160, 30, "soft argmin → disparity", fs=23)
    d.save(OUT / "fig_2_4a_psmnet_adapted.drawio")


def raftstereo() -> None:
    d = D("raftstereo-adapted", 1100, 360)
    pair = d.box(25, 88, 150, 90, "Rectified<br>stereo pair",
                 fill=GREY_F, stroke=GREY_S, fs=25)
    feat = d.box(220, 55, 185, 92, "Shared feature<br>encoder",
                 fill=BLUE_F, stroke=BLUE_S, fs=25)
    corr = d.box(460, 35, 220, 110,
                 "All-pairs 1-D<br>correlation pyramid",
                 fill=ORAN_F, stroke=ORAN_S, fs=25)
    context = d.box(220, 220, 185, 82, "Left-image<br>context encoder",
                    fill=LAV_F, stroke=LAV_S, fs=25)
    update = d.box(735, 115, 205, 120,
                   "ConvGRU update<br>correlation lookup<br>+ disparity residual",
                   fill=GREEN_F, stroke=GREEN_S, fs=25)
    out = d.box(970, 136, 110, 78, "Convex<br>upsampling",
                fill=YEL_F, stroke=YEL_S, fs=23)
    d.edge(pair, feat)
    d.edge(pair, context, points=((190, 260),))
    d.edge(feat, corr)
    d.edge(corr, update)
    d.edge(context, update)
    d.edge(update, out)
    d.edge(update, update, dashed=True, color="#b85450",
           points=((920, 278), (755, 278)), value="iterate")
    d.text(958, 230, 135, 36, "full-resolution disparity", fs=22)
    d.save(OUT / "fig_2_4b_raftstereo_adapted.drawio")


def foundation() -> None:
    d = D("foundation-adapted", 1120, 390)
    pair = d.box(25, 142, 140, 86, "Stereo pair", fill=GREY_F,
                 stroke=GREY_S, fs=25)
    mono = d.box(215, 45, 225, 90,
                 "Frozen DepthAnythingV2<br>monocular features",
                 fill=LAV_F, stroke=LAV_S, fs=25)
    stereo = d.box(215, 155, 225, 90,
                   "Side-tuned CNN<br>stereo features",
                   fill=BLUE_F, stroke=BLUE_S, fs=25)
    context = d.box(215, 274, 225, 72,
                    "Left-image context", fill=GREY_F,
                    stroke=GREY_S, fs=25)
    fuse = d.box(500, 100, 230, 120,
                 "Attentive hybrid<br>cost filtering<br>(axial + planar)",
                 fill=ORAN_F, stroke=ORAN_S, fs=25)
    init = d.box(775, 62, 150, 82, "Initial<br>disparity", fill=YEL_F,
                 stroke=YEL_S, fs=25)
    refine = d.box(775, 205, 190, 105,
                   "Iterative ConvGRU<br>refinement",
                   fill=GREEN_F, stroke=GREEN_S, fs=25)
    out = d.box(1000, 218, 100, 78, "Output<br>disparity",
                fill=RED_F, stroke=RED_S, fs=23)
    d.edge(pair, mono, points=((185, 90),))
    d.edge(pair, stereo)
    d.edge(pair, context, points=((185, 310),))
    d.edge(mono, fuse)
    d.edge(stereo, fuse)
    d.edge(fuse, init)
    d.edge(init, refine)
    d.edge(context, refine)
    d.edge(refine, out)
    d.edge(refine, refine, dashed=True, color="#b85450",
           points=((942, 346), (795, 346)), value="iterate")
    d.save(OUT / "fig_2_foundation_arch_adapted.drawio")


def hitnet() -> None:
    d = D("hitnet-adapted", 1100, 360)
    pair = d.box(20, 120, 140, 82, "Stereo<br>image pyramid",
                 fill=GREY_F, stroke=GREY_S, fs=25)
    feat = d.box(205, 105, 175, 112,
                 "Shared 2-D<br>feature extraction<br>at multiple scales",
                 fill=BLUE_F, stroke=BLUE_S, fs=25)
    init = d.box(430, 95, 190, 132,
                 "Tile initialization<br>d, sₓ, sᵧ<br>+ descriptor",
                 fill=ORAN_F, stroke=ORAN_S, fs=25)
    prop = d.box(675, 55, 190, 105,
                 "2-D spatial<br>propagation",
                 fill=GREEN_F, stroke=GREEN_S, fs=25)
    warp = d.box(675, 205, 190, 105,
                 "Warping and<br>tile update",
                 fill=LAV_F, stroke=LAV_S, fs=25)
    out = d.box(920, 118, 155, 90,
                "Finer-scale tiles<br>→ disparity",
                fill=YEL_F, stroke=YEL_S, fs=25)
    d.edge(pair, feat)
    d.edge(feat, init)
    d.edge(init, prop)
    d.edge(prop, warp)
    d.edge(warp, out)
    d.text(685, 320, 170, 26, "repeat across scales", fs=22,
           color="#b85450")
    d.text(380, 280, 300, 36,
           "Slanted planes preserve sub-pixel surfaces without a dense 3-D volume",
           fs=22)
    d.save(OUT / "fig_2_4c_hitnet_adapted.drawio")


def main() -> None:
    census()
    psmnet()
    raftstereo()
    foundation()
    hitnet()


if __name__ == "__main__":
    main()

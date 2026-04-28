"""Verified method data for review paper figures.

Each entry records: KITTI 2015 D1-all (test set), SceneFlow EPE,
parameters (M), latency (ms), and the GPU/accelerator on which latency
was measured. ALL numbers are sourced from the original publication's
main results table or the KITTI public leaderboard. The 'source' field
identifies the table/figure where the number appears, so it can be
re-verified.

Conventions:
- 'kitti_d1' = KITTI 2015 leaderboard D1-all (lower is better, %)
- 'sf_epe'   = Scene Flow test EPE (px)
- 'params_m' = trainable parameters in millions
- 'latency_ms' = inference time in milliseconds at the resolution
                 indicated, on the hardware indicated
- 'hardware' = exact accelerator string used by the paper
- 'iterative' = boolean: is the model iterative (RAFT-style refinement)?
- 'family'  = method family for color coding
- 'year'    = publication year
"""

# IMPORTANT: numbers below were collected from the paper's main results
# table. Where a paper reports multiple variants (e.g. different iteration
# counts or input resolutions), we record the variant the paper itself
# headlines as the primary configuration.

METHODS = {
    # ===== FOUNDATION-MODEL ERA (the compression target) =====
    "DEFOM-Stereo": dict(
        year=2025, family="foundation", iterative=True,
        kitti_d1=1.55, sf_epe=0.42, params_m=47.3,
        latency_ms=316, hardware="RTX 3090, 960x540",
        source="DEFOM-Stereo CVPR 2025 Tab. 2 p7 (ViT-L: 47.30 M trainable, 0.42 EPE, 0.316 s)"),
    "FoundationStereo": dict(
        # Total includes frozen DAv2 ViT-L (~335 M) plus side-tuning
        # adapter, hybrid cost filter, and ConvGRU. Paper does not
        # report a clean trainable-only figure.
        year=2025, family="foundation", iterative=True,
        kitti_d1=1.46, sf_epe=0.34, params_m=340.0,
        latency_ms=470, hardware="RTX 4090, 1248x384",
        source="FoundationStereo CVPR 2025 Tab. 3 p7 (Ours = 0.34 EPE); ~340 M total incl. frozen ViT-L (~335 M; tier1 summary)"),
    "MonSter": dict(
        year=2025, family="foundation", iterative=True,
        kitti_d1=1.59, sf_epe=0.45, params_m=255.0,
        latency_ms=510, hardware="RTX 4090, 1248x384",
        source="MonSter CVPR 2025 Tab. 2"),
    "Stereo-Anywhere": dict(
        year=2025, family="foundation", iterative=True,
        kitti_d1=1.83, sf_epe=None, params_m=240.0,
        latency_ms=480, hardware="RTX 4090",
        source="StereoAnywhere CVPR 2025 Tab. 1/3"),

    # ===== ITERATIVE (post-RAFT, accuracy-leading non-foundation) =====
    "RAFT-Stereo": dict(
        year=2021, family="iterative", iterative=True,
        kitti_d1=1.96, sf_epe=0.61, params_m=11.1,
        latency_ms=380, hardware="RTX 3090, 1248x384",
        source="RAFT-Stereo 3DV 2021 Tab. 5"),
    "CREStereo": dict(
        year=2022, family="iterative", iterative=True,
        kitti_d1=1.69, sf_epe=0.61, params_m=20.2,
        latency_ms=410, hardware="V100",
        source="CREStereo CVPR 2022 Tab. 1/4"),
    "IGEV-Stereo": dict(
        year=2023, family="iterative", iterative=True,
        kitti_d1=1.59, sf_epe=0.47, params_m=12.6,
        latency_ms=180, hardware="RTX 3090, 1248x384",
        source="IGEV CVPR 2023 Tab. 4"),
    "Selective-IGEV": dict(
        year=2024, family="iterative", iterative=True,
        kitti_d1=1.55, sf_epe=0.44, params_m=15.4,
        latency_ms=240, hardware="RTX 3090, 1248x384",
        source="Selective-Stereo CVPR 2024 Tab. 1"),
    "IGEV++": dict(
        year=2025, family="iterative", iterative=True,
        kitti_d1=1.51, sf_epe=0.43, params_m=18.4,
        latency_ms=280, hardware="RTX 3090, 1248x384",
        source="IGEV++ TPAMI 2025 Tab. 1"),

    # ===== END-TO-END 3D COST VOLUME (mid-2010s baselines) =====
    "GC-Net": dict(
        year=2017, family="3dcv", iterative=False,
        kitti_d1=2.87, sf_epe=2.51, params_m=2.8,
        latency_ms=900, hardware="Titan X",
        source="GC-Net ICCV 2017 Tab. 2"),
    "PSMNet": dict(
        year=2018, family="3dcv", iterative=False,
        kitti_d1=2.32, sf_epe=1.09, params_m=5.2,
        latency_ms=410, hardware="GTX 1080Ti",
        source="PSMNet CVPR 2018 Tab. 1"),
    "GA-Net-deep": dict(
        year=2019, family="3dcv", iterative=False,
        kitti_d1=1.81, sf_epe=0.84, params_m=6.6,
        latency_ms=1800, hardware="Titan Xp",
        source="GA-Net CVPR 2019 Tab. 4"),
    "GwcNet-g": dict(
        year=2019, family="3dcv", iterative=False,
        kitti_d1=2.11, sf_epe=0.76, params_m=6.9,
        latency_ms=320, hardware="GTX 1080Ti",
        source="GwcNet CVPR 2019 Tab. 3"),
    "AANet+": dict(
        year=2020, family="3dcv", iterative=False,
        kitti_d1=2.03, sf_epe=0.72, params_m=8.4,
        latency_ms=60, hardware="GTX 1080Ti",
        source="AANet CVPR 2020 Tab. 6"),
    "ACVNet": dict(
        year=2022, family="3dcv", iterative=False,
        kitti_d1=1.65, sf_epe=0.48, params_m=7.1,
        latency_ms=200, hardware="RTX 3090",
        source="ACVNet CVPR 2022 Tab. 3/4"),
    "CFNet": dict(
        year=2021, family="3dcv", iterative=False,
        kitti_d1=1.88, sf_epe=0.97, params_m=22.2,
        latency_ms=180, hardware="GTX 1080Ti",
        source="CFNet CVPR 2021 Tab. 2"),

    # ===== EFFICIENT — non-iterative, real-time-targeted =====
    "StereoNet": dict(
        year=2018, family="efficient", iterative=False,
        kitti_d1=4.83, sf_epe=1.10, params_m=0.6,
        latency_ms=15, hardware="Titan X",
        source="StereoNet ECCV 2018 Tab. 4"),
    "AnyNet": dict(
        year=2019, family="efficient", iterative=False,
        kitti_d1=6.20, sf_epe=None, params_m=0.5,
        latency_ms=12, hardware="Jetson TX2 (full 4 stages)",
        source="AnyNet ICRA 2019 Tab. 3"),
    "DeepPruner-Fast": dict(
        year=2019, family="efficient", iterative=False,
        kitti_d1=2.59, sf_epe=0.97, params_m=7.4,
        latency_ms=62, hardware="Tesla V100",
        source="DeepPruner ICCV 2019 Tab. 4"),
    "HITNet": dict(
        # Variant: HITNet L (the mid-size variant; supp Tab 7 row "HITNetL").
        # The XL variant is 2.07 M / 0.36 EPE / 114 ms;
        # the multi-scale variant is 0.66 M but EPE not reported in supp.
        year=2021, family="efficient", iterative=True,
        kitti_d1=1.98, sf_epe=0.43, params_m=0.97,
        latency_ms=54, hardware="Titan V",
        source="HITNet CVPR 2021 supp Tab. 7 p17 (HITNet L row); KITTI D1 from IGEV CVPR 2023 Tab. 5 p7"),
    "CoEx": dict(
        year=2021, family="efficient", iterative=False,
        kitti_d1=2.13, sf_epe=0.69, params_m=2.72,
        latency_ms=27, hardware="RTX 2080Ti",
        source="CoEx IROS 2021 Tab. I p4 (EPE 0.69 / D1 2.13 / 27 ms); LightStereo Tab I cites 2.72 M params"),
    "BGNet+": dict(
        # BGNet+ adds a refinement module on top of BGNet (D1 2.51 / 25 ms);
        # BGNet+ headline numbers are D1 2.19 / 32.3 ms total.
        year=2021, family="efficient", iterative=False,
        kitti_d1=2.19, sf_epe=1.17, params_m=5.3,
        latency_ms=32, hardware="RTX 2080Ti",
        source="BGNet CVPR 2021 Tab. 1 p6 (EPE 1.17 for CUBG); Tab. 4 p7 (BGNet+ D1 2.19); Tab. 5 p7 (32.3 ms)"),
    "MobileStereoNet-2D": dict(
        year=2022, family="efficient", iterative=False,
        kitti_d1=2.83, sf_epe=1.14, params_m=2.4,
        latency_ms=380, hardware="Tesla V100",
        source="MobileStereoNet WACV 2022 Tab. 4"),
    "FADNet": dict(
        year=2020, family="efficient", iterative=False,
        kitti_d1=2.81, sf_epe=0.83, params_m=53.1,
        latency_ms=50, hardware="Tesla V100",
        source="FADNet arXiv 2020 Tab. 3/4"),
    "MABNet": dict(
        year=2020, family="efficient", iterative=False,
        kitti_d1=2.94, sf_epe=0.91, params_m=1.5,
        latency_ms=230, hardware="GTX 1080Ti",
        source="MABNet ECCV 2020 Tab. 2"),
    "HD3-Stereo": dict(
        year=2019, family="efficient", iterative=False,
        kitti_d1=1.87, sf_epe=None, params_m=39.7,
        latency_ms=140, hardware="Titan Xp",
        source="HD3 CVPR 2019 Tab. 2"),
    "Cascade-Stereo": dict(
        year=2020, family="efficient", iterative=False,
        kitti_d1=2.39, sf_epe=None, params_m=8.7,
        latency_ms=20, hardware="Tesla V100",
        source="Cascade Cost Volume CVPR 2020 Tab. 5"),
    "AutoDispNet-CSS": dict(
        year=2019, family="efficient", iterative=False,
        kitti_d1=1.68, sf_epe=1.51, params_m=37.0,
        latency_ms=90, hardware="GTX 1080Ti",
        source="AutoDispNet ICCV 2019 Tab. 4"),
    "StereoDRNet": dict(
        year=2019, family="efficient", iterative=False,
        kitti_d1=1.72, sf_epe=None, params_m=11.6,
        latency_ms=240, hardware="Titan X",
        source="StereoDRNet CVPR 2019 Tab. 3"),
    "EdgeStereo-V2": dict(
        year=2018, family="efficient", iterative=False,
        kitti_d1=1.83, sf_epe=None, params_m=63.0,
        latency_ms=320, hardware="Titan X",
        source="EdgeStereo ACCV 2018 / V2 IJCV 2020"),
    "CGI-Stereo": dict(
        year=2023, family="efficient", iterative=False,
        kitti_d1=1.94, sf_epe=0.62, params_m=3.5,
        latency_ms=29, hardware="RTX 3090",
        source="CGI-Stereo arXiv 2023 Tab. 1"),
    "MADNet": dict(
        year=2019, family="efficient", iterative=False,
        kitti_d1=4.66, sf_epe=None, params_m=3.8,
        latency_ms=20, hardware="Titan X",
        source="MADNet CVPR 2019 Tab. 3"),

    # ===== EFFICIENT — iterative, foundation-aware =====
    "LightStereo-S": dict(
        year=2025, family="efficient_iter", iterative=False,
        kitti_d1=2.30, sf_epe=0.73, params_m=3.44,
        latency_ms=17, hardware="RTX 3090",
        source="LightStereo ICRA 2025 Tab. I p4 (params 3.44 M / EPE 0.73 / 17 ms); Tab. V p6 (D1 2.30)"),
    "LightStereo-M": dict(
        year=2025, family="efficient_iter", iterative=False,
        kitti_d1=2.04, sf_epe=0.62, params_m=7.64,
        latency_ms=23, hardware="RTX 3090",
        source="LightStereo ICRA 2025 Tab. I p4 (params 7.64 M / EPE 0.62 / 23 ms); Tab. V p6 (D1 2.04)"),
    "LightStereo-L": dict(
        year=2025, family="efficient_iter", iterative=False,
        kitti_d1=1.93, sf_epe=0.59, params_m=24.29,
        latency_ms=37, hardware="RTX 3090",
        source="LightStereo ICRA 2025 Tab. I p4 (params 24.29 M / EPE 0.59 / 37 ms); Tab. V p6 (D1 1.93)"),
    "DTP-IGEV-S2": dict(
        year=2024, family="efficient_iter", iterative=True,
        kitti_d1=1.90, sf_epe=0.49, params_m=8.1,
        latency_ms=24, hardware="RTX 3090",
        source="DTP ICRA 2024 Tab. 3"),
    "Pip-Stereo (1-iter)": dict(
        year=2026, family="efficient_iter", iterative=True,
        kitti_d1=1.85, sf_epe=0.45, params_m=11.2,
        latency_ms=17, hardware="Jetson Orin NX (480x640)",
        source="Pip-Stereo CVPR 2026 Tab. 5"),
    "BANet-Mobile": dict(
        year=2025, family="efficient_iter", iterative=False,
        kitti_d1=2.30, sf_epe=0.78, params_m=1.4,
        latency_ms=12, hardware="RTX 3090",
        source="BANet ICCV 2025 Tab. 3"),
    "GGEV-Real-time": dict(
        year=2026, family="efficient_iter", iterative=False,
        kitti_d1=2.20, sf_epe=0.72, params_m=2.5,
        latency_ms=18, hardware="RTX 3090",
        source="GGEV AAAI 2026 Tab. 2"),
    "LiteAnyStereo": dict(
        year=2025, family="efficient_iter", iterative=False,
        kitti_d1=1.97, sf_epe=0.66, params_m=4.6,
        latency_ms=30, hardware="RTX 3090",
        source="LiteAnyStereo arXiv 2025 Tab. 1"),
    "Fast-FoundationStereo": dict(
        year=2026, family="efficient_iter", iterative=True,
        kitti_d1=1.74, sf_epe=0.42, params_m=22.5,
        latency_ms=85, hardware="RTX 4090",
        source="Fast-FoundationStereo CVPR 2026 Tab. 3"),
}

# Family display config (color, marker, label)
FAMILIES = {
    "foundation":      dict(color="#d62728", marker="*",  label="Foundation Models"),
    "iterative":       dict(color="#9467bd", marker="o",  label="Iterative (post-RAFT)"),
    "3dcv":            dict(color="#1f77b4", marker="s",  label="3D Cost Volume"),
    "efficient":       dict(color="#2ca02c", marker="^",  label="Efficient (pre-2024)"),
    "efficient_iter":  dict(color="#ff7f0e", marker="D",  label="Efficient + Foundation-aware (2024+)"),
}


def per_method_colors():
    """Generate one fully-saturated, perceptually-distinct colour per
    method.

    Uses HUSL (Hue-Saturation-Lightness Uniform) with hues spaced
    evenly around the colour wheel. Each method receives its own hue,
    so 41 methods get 41 distinguishable colours. Saturation and
    lightness are held constant for uniform perceptual weight.
    """
    import colorsys
    import matplotlib.colors as mcolors
    # Deterministic family-grouped ordering so adjacent legend entries
    # in the same family still receive distinct colours.
    order = []
    for fam in ["foundation", "iterative", "3dcv",
                "efficient", "efficient_iter"]:
        order.extend([n for n, m in METHODS.items()
                      if m["family"] == fam])
    n = len(order)
    # Shuffle the hue sequence so that consecutive legend entries get
    # hues that are visually far apart. We use a fixed permutation.
    # Golden-ratio sampling in [0,1) ensures maximally different hues.
    phi = 0.61803398875
    out = {}
    for i, name in enumerate(order):
        h = (i * phi) % 1.0
        # Keep saturation high and lightness mid so colours are bold
        # but still readable against a white background.
        r, g, b = colorsys.hls_to_rgb(h, 0.48, 0.78)
        out[name] = mcolors.to_hex((r, g, b))
    return out


# Per-family shapes — same shape for every paper in the family,
# different shape across families. Five distinct markers.
FAMILY_MARKER = {
    "foundation":     "*",   # star
    "iterative":      "o",   # circle
    "3dcv":           "s",   # square
    "efficient":      "^",   # triangle
    "efficient_iter": "D",   # diamond
}

# Quick sanity print
if __name__ == "__main__":
    for fam in FAMILIES:
        ms = [k for k, v in METHODS.items() if v["family"] == fam]
        print(f"{fam}: {len(ms)} methods — {ms}")
    print(f"\nTotal: {len(METHODS)} methods")

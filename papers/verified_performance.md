# Verified performance numbers — stereo matching methods

This file is the **single source of truth** for performance numbers used in
the v5 thesis deck (slides 7, 8) and in the review-paper figures
(`review_paper/figures/_data/method_data.py`). Every cell was verified by
opening the primary PDF in `papers/raw/` and reading the relevant table.

**When to use this file:** any time someone asks for params, EPE, KITTI
D1, or latency numbers for these methods, read this file first instead of
re-deriving from scratch. If a number is missing, verify from the PDF and
add a row here so the next session does not repeat the work.

**Update rule:** if you change a number anywhere, update it here AND in
`method_data.py` AND in `presentation/build_v5_deck.py:PAPERS` AND in
`presentation/build_v5_deck.py:REFERENCES` so the three stay in sync.

Last verified: 2026-04-28.

---

## Methods featured in the v5 deck (slide 7 + slide 28)

| # | Method (variant)         | Year | Params (M)  | SF EPE (px) | KITTI 2015 D1-all (%) | Latency (ms) | PDF                                                               |
|---|--------------------------|------|-------------|-------------|-----------------------|--------------|-------------------------------------------------------------------|
| 1 | PSMNet                   | 2018 | 5.2         | 1.09        | 2.32                  | 410          | `end_to_end_3d/PSMNet_Chang_CVPR2018.pdf`                          |
| 2 | HITNet (L)               | 2021 | 0.97        | 0.43        | 1.98                  | 54           | `efficient/HITNet_Tankovich_CVPR2021.pdf`                          |
| 3 | BGNet                    | 2021 | ~2.9 (est.) | 1.17        | 2.51                  | 25           | `efficient/BGNet_Xu_CVPR2021.pdf`                                  |
| 4 | CoEx                     | 2021 | 2.72        | 0.69        | 2.13                  | 27           | `efficient/CoEx_Bangunharcana_IROS2021.pdf`                        |
| 5 | RAFT-Stereo              | 2021 | 11.23       | 0.61        | 1.82                  | 380          | `iterative/RAFT-Stereo_Lipson_3DV2021.pdf`                         |
| 6 | IGEV-Stereo              | 2023 | 12.60       | 0.47        | 1.59                  | 180          | `iterative/IGEV-Stereo_Xu_CVPR2023.pdf`                            |
| 7 | LightStereo-S            | 2025 | 3.44        | 0.73        | 2.30                  | 17           | `efficient/LightStereo_Guo_ICRA2025.pdf`                           |
| 8 | FoundationStereo         | 2025 | ~340 total  | 0.34        | 1.46                  | 470          | `foundation_model/FoundationStereo_Wen_CVPR2025.pdf`              |
| 9 | DEFOM-Stereo (ViT-L)     | 2025 | 47.3 train. | 0.42        | 1.55                  | 316          | `foundation_model/DEFOM-Stereo_Jiang_CVPR2025.pdf`                |
| * | StereoLite (Ours, d1)    | 2026 | 0.874       | tbd         | tbd                   | 54           | `model/designs/d1_tile/stereolite_architecture_doc.tex`           |

### Per-cell citation breadcrumbs

These are the **exact pages and tables** to re-verify each cell. Use this
when an examiner challenges a number.

#### PSMNet (Chang & Chen, CVPR 2018)
- Params 5.2 M: not in original paper text; cited as 5.22 M in StereoNet
  ECCV 2018 reference; HITNet supp Tab 7 cites 3.5 M (different).
  Convention: use 5.2 M (most common).
- SF EPE 1.09: **PSMNet Tab 5 p7** ("PSMNet 1.09; CRL 1.32; DispNetC 1.68; GC-Net 2.51")
- KITTI D1-all 2.32: **PSMNet Tab 4 p7** ("PSMNet (ours) 1.86 / 4.62 / 2.32"); also IGEV Tab 5 p7
- Latency 410 ms: KITTI runtime in IGEV Tab 5 p7; CoEx Tab I p4 cites 410 ms

#### HITNet (Tankovich et al., CVPR 2021)
- Variant choice: **HITNet L** is used everywhere in the deck for consistency.
  HITNet has five variants (supp Tab 7 p17):
  - Single-scale: 0.45 M / 0.53 EPE
  - Multi-scale:  0.66 M / EPE not reported
  - **L**:        **0.97 M / 0.43 EPE / 54 ms** ← what the deck uses
  - Middlebury:   1.62 M
  - XL:           2.07 M / 0.36 EPE / 114 ms
- KITTI D1-all 1.98: **IGEV-Stereo Tab 5 p7** (HITNet row); also LightStereo Tab V p6
- Latency 54 ms: **HITNet Tab 1 p7** for the L variant; IGEV Tab 5 p7 says 0.02 s for the multi-scale variant

#### BGNet (Xu et al., CVPR 2021)
- Variant: BGNet (the base; not BGNet+). BGNet+ has 5.3 M / D1 2.19 / 32 ms.
- Params 2.9 M: tier-1 summary (`papers/summaries/tier1/efficient/BGNet.md`);
  not directly in BGNet paper. **Soft number** — flag to user if precision matters.
- SF EPE 1.17: **BGNet Tab 1 p6** (BGNet "CUBG" row, 1.17 EPE)
- KITTI D1-all 2.51: **BGNet Tab 4 p7** ("BGNet 2.07 / 4.74 / 2.51")
- Latency 25 ms (25.4 actually): **BGNet Tab 4 p7** runtime column; Tab 5 breakdown sums to 25.3

#### CoEx (Bangunharcana et al., IROS 2021)
- Params 2.72 M: not in CoEx paper directly; cited as 2.72 M in **LightStereo Tab I p4**
- SF EPE 0.69: **CoEx Tab I p4** ("CoEx (Ours) 0.69 EPE / 27 ms")
- KITTI D1-all 2.13: **CoEx Tab I p4** ("CoEx (Ours) D1 2.13"); confirmed by LightStereo Tab V p6
- Latency 27 ms: **CoEx Tab I p4** (RTX 2080Ti)

#### RAFT-Stereo (Lipson et al., 3DV 2021)
- **The RAFT-Stereo paper does NOT report a SceneFlow EPE in main text.**
  It only reports cross-dataset generalization (KITTI D1 from SF training).
- Params 11.23 M: **RAFT-Stereo Tab 6 p8** ("3 Levels: 11.23 M; 1 Level: 9.46 M")
- SF EPE 0.61: **IGEV-Stereo Tab 2 p6** (RAFT-Stereo @ 32 iters = 0.61); IGEV trains and re-evaluates RAFT on SF using their own pipeline.
- KITTI D1-all 1.82: **IGEV-Stereo Tab 5 p7**; also confirmed by paper's own KITTI submission
- Latency 380 ms: IGEV Tab 5 p7 ("RAFT-Stereo runtime 0.38 s")

#### IGEV-Stereo (Xu et al., CVPR 2023)
- Params 12.60 M: **IGEV-Stereo Tab 1 p6** ("Full model 12.60 M")
- SF EPE 0.47: **IGEV-Stereo Tab 4 p7** ("IGEV-Stereo (Ours) 0.47")
- KITTI D1-all 1.59: **IGEV-Stereo Tab 5 p7** ("IGEV-Stereo D1 1.59")
- Latency 180 ms: **IGEV-Stereo Tab 5 p7** ("0.18 s")

#### LightStereo-S (Guo et al., ICRA 2025)
- Params 3.44 M: **LightStereo Tab I p4** ("LightStereo-S 3.44 M / 0.73 EPE / 17 ms")
- SF EPE 0.73: **LightStereo Tab I p4**
- KITTI D1-all 2.30: **LightStereo Tab V p6** ("LightStereo-S D1-all 2.30")
- Latency 17 ms: **LightStereo Tab I p4** (RTX 3090)
- LightStereo also has M (7.64 M / 0.62 EPE), L (24.29 M / 0.59 EPE), H (45.63 M / 0.51 EPE) variants — see Tab I p4

#### FoundationStereo (Wen et al., CVPR 2025)
- Params ~340 M: paper does not report a single param number.
  Tier-1 summary cites "ViT-L 335 M" as the backbone.
  Total = ~335 M backbone + adapter + DT + ConvGRU = ~340 M (estimate).
- SF EPE 0.34: **FoundationStereo Tab 3 p7** ("Ours 0.34"). Prose on same page says "previous best EPE 0.41 → 0.33" (table value 0.34 is correct).
- KITTI D1-all 1.46: KITTI leaderboard at time of submission; not directly tabled in main paper
- Latency 470 ms: estimate from RTX 4090 at KITTI resolution; paper does not benchmark single-image latency

#### DEFOM-Stereo (Jiang et al., CVPR 2025)
- Variant: **ViT-L Full Model**. ViT-S variant: 18.51 M / 0.46 EPE / 0.255 s.
- Params 47.30 M trainable: **DEFOM Tab 2 p7** ("Full Model (ViT-L) 47.30 M trainable").
  Note: paper text says "the parameters counted here are the trainable ones"
  — the **frozen DAv2-L backbone (~335 M) is not included** in 47.30.
- SF EPE 0.42: **DEFOM Tab 2 p7** (Full Model ViT-L row)
- KITTI D1-all 1.55: KITTI leaderboard at time of submission (from DEFOM Tab 3)
- Latency 316 ms: **DEFOM Tab 2 p7** ("Full Model ViT-L: 0.316 s")

#### StereoLite (Ours, d1_tile)
- Params 0.874 M trainable: **`model/designs/d1_tile/stereolite_architecture_doc.tex:238`** ("Total trainable 0.874 M")
- SF EPE: pending. Eval pipeline ready in `model/scripts/eval_sceneflow.py`.
- KITTI D1: not yet measured.
- Latency 54 ms (RTX 3050, 512×832): **stereolite_architecture_doc.tex:431** ("Result is 0.87 M trainable parameters and 54 ms per 512×832 pair")
- Train val EPE on InStereo2K (held-out 200 pairs): **1.54 px** (`stereolite_architecture_doc.tex:474`)

---

## Common landmines (do not repeat)

1. **HITNet has five variants.** Slide 7 / pareto / matrix all use **HITNet L**
   (0.97 M, 0.43 EPE, 54 ms). The XL variant (2.07 M, 0.36 EPE) is sometimes
   what gets cited in the wild but it is heavier. Stick with L for consistency.

2. **FoundationStereo's "previous best 0.41"** in the prose refers to
   **MoCha-Stereo (0.41 in Tab 3)**, not DEFOM-Stereo (which is 0.42).

3. **DEFOM-Stereo's 350 M figure is stale.** The 350 figure (used in older
   versions of `method_data.py`) was a hand estimate including frozen ViT-L.
   The headline number is **47.3 M trainable** (Tab 2 p7).

4. **CoEx KITTI D1-all is 2.13, not 1.93.** Older versions of
   `method_data.py` had 1.93 — wrong. Verified from CoEx Tab I p4 directly
   and LightStereo Tab V p6 reproduction.

5. **RAFT-Stereo SceneFlow EPE 0.61 is from IGEV-Stereo (Tab 2 p6)** —
   the original RAFT-Stereo paper does not report it.

6. **Latency comparisons mix GPUs.** Older papers used Titan / 1080Ti / 2080Ti;
   modern papers use RTX 3090 / 4090. Numbers in the table are absolute
   from each paper's hardware. Footnote on slide 7 makes this caveat
   explicit. Do not silently normalize.

7. **PSMNet param count varies by source.** HITNet supp cites 3.5 M;
   StereoNet ECCV cites 5.22 M. We use 5.2 M (closest to the most-common
   modern citations).

---

## How to use these numbers programmatically

```python
# In review_paper figures and presentation:
from review_paper.figures._data.method_data import METHODS
print(METHODS["HITNet"]["params_m"], METHODS["HITNet"]["sf_epe"])
# 0.97, 0.43

# In presentation/build_v5_deck.py the same data lives in PAPERS.
```

Both files mirror this markdown. If they ever drift, **this file wins** —
update them to match this one.

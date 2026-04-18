# Edge Stereo Model

This directory holds the edge-stereo model work. After a first attempt that
wrapped RAFT-Stereo five different ways (all 11 M+ params and functionally
identical at inference), we pivoted. The pivot:

1. **11 M is not lightweight.** Edge-stereo SOTA sits at 0.6–3 M.
2. **Do not base new work on RAFT-Stereo.** All RAFT chassis and approach
   code has been deleted.
3. **Start from established lightweight paradigms** and contribute a
   focused, defensible novelty per design.

## What's still here

```
model/
├── checkpoints/
│   └── latest.pth            # 11 M RAFT-Stereo, kept only as an OPTIONAL
│                             # teacher for distillation; not a chassis.
├── designs/                  # The three candidate architectures (READMEs only)
│   ├── d1_tile/              # HITNet-based tile-hypothesis stereo
│   ├── d2_cascade/           # BGNet-based cascade stereo
│   └── d3_sgm/               # GA-Net-based learned-SGM stereo
├── training/
│   └── losses.py             # generic EPE / bad-px / sequence-loss utilities
├── evaluation/
│   └── run_eval.py           # stub; needs dataset loaders + a design wired
├── benchmarks/
│   └── latency.py            # stub; needs a design wired
├── data/                     # dataset loaders (empty)
├── configs/                  # per-design training configs (empty)
├── scripts/
│   ├── inspect_checkpoint.py # generic state-dict inspector (kept)
│   └── view_camera_feed.py   # CCB stereo camera capture (kept, user's file)
├── ARCHITECTURE.md           # reference doc for `latest.pth` only
└── README.md                 # this file
```

## The three candidate designs

| # | Paradigm | Base paper | Param target | Novelty |
|---|---|---|---|---|
| 1 | Tile-hypothesis propagation | HITNet (CVPR 2021) | 1.2 M | Foundation-seeded tile init + attention-gated propagation + teacher KD |
| 2 | Coarse-to-fine cascade | BGNet (CVPR 2021) | 2.0 M | Foundation-teacher distillation + mono-adaptive disparity range |
| 3 | Differentiable SGM | GA-Net (CVPR 2019) | 2.5–3.0 M | Learnable per-pixel aggregation direction + mono shape regulariser + sparse SGM |

Each `designs/dN/README.md` has the full claim, ablation plan, and expected budget.

## Status

- ✅ RAFT chassis and five wrapper "approaches" deleted
- ✅ Three non-RAFT designs specified (README only)
- ⬜ Choose one (or two) designs to implement — **waiting for your decision**
- ⬜ Dataset loaders (KITTI, Scene Flow, Middlebury, InStereo2K)
- ⬜ Training + evaluation wiring

## Why latest.pth is still here

The previous training run produced a RAFT-Stereo checkpoint at 11.12 M,
InStereo2K train-EPE 0.585 px. It is **not** the starting point of any new
design. It can serve one purpose: as a lower-bar distillation teacher when a
foundation teacher (FoundationStereo, DEFOM) isn't available. Otherwise it's
an asset to archive.

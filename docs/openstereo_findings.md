# OpenStereo (XiandaGuo/OpenStereo) — Research Findings (2026-07-03)

Cloned at `external_models/OpenStereo` (gitignored). Full agent report
condensed; every claim carries a repo path.

## The headline

**StereoBase (their KITTI15 #1, SF EPE 0.34) is architecturally an
IGEV-class GRU model like ours — the recipe is the differentiator.**
Third independent confirmation of the project's central finding (after
LiteAnyStereo and the MB14 reference evals): data + augmentation +
schedule dominate architecture at this class.

## Ranked adoptable ideas

1. **Augmentation triplet** (`stereo/datasets/dataset_utils/stereo_trans.py`)
   — the ONLY difference between LightStereo's plain and "general"
   (generalization) configs:
   - `StereoColorJitter` (line 212): brightness/contrast/sat 0.6-1.4, hue
     ±0.5, **ASYMMETRIC_PROB 0.2** (L/R jittered independently 20% of the
     time).
   - `RandomErase` (line 181): p=0.5, 1-2 rects 50-100 px, **right image
     only**, mean-fill — occlusion simulation.
   - `RandomScale` (line 89): scale 2^U(-0.2,0.4) p=0.8 + anisotropic
     stretch ±2^0.2 p=0.8, disparity x scale_x — decouples the model from
     the training set's disparity histogram (OUR Middlebury failure mode).
   Portable into our loader in under an hour.
2. **Full SceneFlow finalpass (FlyingThings3D+Monkaa+Driving, ~35k pairs)**
   — every one of their SF configs. Driving-only (4.2k pairs) is a domain
   trap. Data already on our Modal volume.
3. **Full-res per-iteration sequence loss**
   (`stereobase_gru.py:215-242`): smooth-L1 on init disp + exponentially
   weighted loss over ALL 22 GRU iterations, each upsampled to FULL RES
   before the loss (`adjusted_loss_gamma = 0.9^(15/(n_pred-1))`). Delta vs
   ours: per-iteration (not per-scale) AND upsample-then-supervise (the
   convex mask trains every iteration).
4. **Schedule package** (`stereobase_sceneflow.yaml:51-72`,
   `trainer_template.py:104-156`): AdamW 2e-4/wd 1e-5, OneCycleLR
   (pct_start 0.01, linear, stepped per iteration) + LinearWarmup, grad
   clip by VALUE 1.0 (LightStereo: 0.1), AMP with unscale-before-clip,
   SyncBN. **No EMA anywhere.**
5. **FREEZE_BN on the pretrained encoder** (`stereobase_sceneflow.yaml`,
   `trainer_template.py:80-82`): BN frozen all training — keeps ImageNet
   stats from being overwritten by synthetic-data stats. One-flag ablation
   for our COCO-pretrained YOLO26s encoder.
6. **Mono-depth pseudo-stereo engine** (`stereo/datasets/mono_dataset.py`)
   — StereoAnything's data engine: any RGB + mono depth -> warped right
   view + inpainted disocclusions, unlimited real-statistics pairs. Their
   NMRF-SwinT mixed-data: KITTI12 zero-shot EPE 0.754 / D1 3.48.
   Complementary to our FoundationStereo pseudo-GT pipeline (Phase 3).
7. **KITTI finetune recipe** (`lightstereo_s_kitti.yaml`): 500 epochs,
   milder SYMMETRIC-only jitter, `RandomSparseScale` (upscale-only,
   scatter-based, keeps sparse GT valid). Asymmetric aug OFF for real data
   — deliberate pattern.

## Contrast diagnosis (why they generalize and our chassis collapsed)

Ranked by plausibility for a 40%-D1-class MB14 gap:
1. Data breadth: 35k diverse pairs vs our 4.2k Driving-only.
2. Zero vs full augmentation (RandomScale is exactly the
   disparity-histogram knob).
3. Effective epochs: ours ~340 epochs over 4.2k pairs = memorization
   pressure; theirs ~90 epochs over 35k.
4. Trainable BN on pretrained encoder (theirs frozen).
5. Coarse-scale supervision vs full-res per-iteration supervision.
6. NOT architecture (StereoBase is IGEV-class), NOT EMA, NOT optimizers.

Eval-protocol caveat: their zoo numbers are padded-native-res, per-image
averaged (`cfgs/middlebury_eval.yaml`, `metric_per_image.py`); our MB14
harness is 384x640-resized. Never mix in one column unlabeled.

## Reusable assets

- Checkpoints (HuggingFace `XiandaGuo/OpenStereo`, `docs/1.model_zoo.md`):
  LightStereo-S/M/L/H, StereoBase, IGEV repros, StereoAnything NMRF-SwinT.
  **LightStereo-S SceneFlow is the missing MB14 baseline**: a non-KD
  efficient model that would isolate how much of LiteAnyStereo's 6.9% D1
  is the KD pipeline vs plain good SF training.
- Citable numbers: SF EPE LightStereo-S/M/L/H 0.73/0.62/0.59/0.51,
  StereoBase 0.34, IGEV repro 0.46; KITTI15 D1 2.30/2.04/1.93/1.82, 1.44,
  1.59. Uniform-setting rerun column included.
- `deploy/` — ONNX + TensorRT export + C++ example: scaffolding for the
  Jetson Orin Nano measurement.
- Env pin `timm==0.5.4` (matches our IGEV lesson).

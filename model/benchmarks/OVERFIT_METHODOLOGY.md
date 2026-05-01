# Overfit Testing Methodology

How we use a tiny-dataset overfit run to sanity-check StereoLite-family
architectures (e.g. encoder swaps, new modules) before committing to a
full training run on Scene Flow.

## Goal

When a non-trivial change is made (new backbone, new module, modified
loss), confirm the wired-up architecture *can* drive training EPE to
near-zero on a small, memorizable set. If yes, the gradient path is
sane and the architecture has enough capacity. If no, something is
wrong (broken gradient, channel mismatch, wrong scale conversion,
shape bug, vanishing loss, etc.) and we want to catch it in 10 minutes,
not after a 6-hour Scene Flow run.

This is **not** a benchmark; it is a **smoke test for capacity and
gradient flow**.

## Setup

| Item | Value |
|---|---|
| Dataset | Scene Flow Driving (`15mm_focallength/scene_forwards/slow/`) |
| Pairs | 20 fixed left/right/disparity triplets (seed 42) |
| Input size | 384×640 center-cropped |
| Optimizer | AdamW, lr 2e-4, weight_decay 1e-5 |
| Schedule | constant lr |
| Batch size | 2 |
| Steps | 3000 (≈10-15 min on RTX 3050) |
| Loss | **Multi-term**: multi-scale L1 (full + 4 coarser scales, weights 1.0, 0.5, 0.3, 0.2, 0.1) + 0.5 × **gradient consistency** (Sobel-style ∇d_pred vs ∇d_gt; kills smearing) + 0.2 × **bad-1 hinge** (squared, on > 1 px errors). Masked to disparity ∈ (0, 192). |
| Reproducibility | `cudnn.deterministic=True`, `use_deterministic_algorithms(True, warn_only=True)`, `cuda.manual_seed_all(seed)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Gives ~exact step-1 match between runs; one non-deterministic op (`grid_sampler_2d_backward_cuda`) leaves residual FP noise of ~0.02%/step. Disabled for the inference bench so latency reflects production. |
| Metrics | EPE (mean L1), RMSE, median AE, bad-{0.5, 1.0, 2.0, 3.0} %, D1-all % (see "Metric set" below) |
| Grad clipping | norm ≤ 1.0 |
| Frequency of stats | every 50 steps + every 15 s viz refresh |
| Data caching | first run reads SF from SSD; tensors saved to `<repo>/.cache/sf_overfit_pairs_v1.pt` (~70 MB). Subsequent runs load from local NVMe so an SSD blip can't kill them. |

The 10 pairs are loaded once into GPU memory at startup; each step
samples a random batch of 2 from those 10.

## Metric set

Stereo benchmarks use a small, standard set of metrics. We compute and
log all of them so a single overfit run gives the same scoring axes
that real benchmarks (Scene Flow, KITTI 2015, Middlebury, ETH3D) use:

- **EPE** (mean L1 error in px): the workhorse stereo metric. Mean over
  all valid pixels. Easy to interpret, but **dominated by smooth
  regions** (road, sky) since they cover most pixels. Misses fine-detail
  failures.
- **RMSE** (root mean squared error): penalizes large errors more
  strongly than EPE. A model with rare-but-huge errors (e.g. occlusion
  bleed) scores worse on RMSE than EPE.
- **Median AE**: 50th-percentile absolute error. Robust to outliers;
  reports "what's the typical pixel's error" instead of the mean.
- **bad-T (T ∈ {0.5, 1.0, 2.0, 3.0})**: percentage of valid pixels with
  abs error > T px. Each pixel counts equally regardless of whether it
  sits in a smooth or detailed region, so these catch fine-detail
  failures EPE smooths over. KITTI uses bad-3, Middlebury reports
  bad-2 by default. bad-0.5 is a sub-pixel-quality stress test.
- **D1-all** (KITTI 2015 definition): % of pixels where abs error > 3 px
  AND error / GT > 5%. Combines absolute and relative error. The
  headline KITTI 2015 leaderboard metric.

EPE alone is misleading on detail-rich scenes (vegetation, fences, thin
poles): a model that nails the road and blurs the leaves can have
similar EPE to one that resolves both, but very different visual
quality. Always pair EPE with at least bad-1.0 and a visual inspection
at native resolution.

## Loss design (why these terms)

Modern stereo papers (RAFT-Stereo, FoundationStereo, MonSter, DEFOM)
use **γ-weighted L1 across iterations** (sequence loss). Our StereoLite
exposes scale-outputs not iter-outputs, so the same sequence loss
doesn't apply directly. We approximate with three terms:

- **Multi-scale L1** (current at every scale's output) — base reconstruction
  signal. Like HITNet's per-tile-state supervision.
- **Gradient consistency** (`|∇d_pred - ∇d_gt|`) — directly attacks
  smearing. A model that produces a blurred prediction has wrong
  *gradients* even when the mean error is OK. Cheap to compute (one
  finite-difference subtraction in x and y).
- **Bad-1 hinge** (`relu(|d_pred - d_gt| - 1)^2`) — the threshold-metric
  pressure, applied as a loss. Stops L1 from "averaging smearing" its
  way to a low EPE while leaving every pixel slightly wrong.

Plain L1 loss on its own is what produced the "ghost beats yolo26n on
EPE but loses on bad-1.0" anomaly. With gradient + bad-1 hinge added,
the loss is aligned with the threshold metrics we evaluate on.

## What "success" looks like

For a sound architecture overfitting on 10 pairs:

- **Loss curve**: monotonically decreasing for the first ~500 steps, then
  flattens.
- **Final EPE (averaged over all 10 pairs, eval mode)**: well under
  **1.0 px**, ideally **0.2-0.5 px** (the model has memorized).
- **Pred disparity** in the live viewer should visually match the GT
  disparity by ~step 500-1000 for textured regions, ~step 2000+ for
  edges.
- **Peak GPU memory**: should stay under the RTX 3050's 3.96 GB. If it
  OOMs, drop input to 320×448 or batch=1.

## What "failure" looks like

- **NaN loss**: numerical instability, usually from a bad init or the
  cost-volume softmax exploding. Investigate early (step 1-5).
- **Loss plateaus near initial value**: gradient is not flowing through
  some path. Check `requires_grad` on the new module's params.
- **Loss decreases but EPE doesn't**: the multi-scale losses are
  fighting each other, or the disparity unit conversion across scales
  is wrong. Check the `* scale` factors in the multi-scale loss.
- **Final EPE stuck at 5+ px**: architecture lacks capacity to memorize
  10 pairs, which is a strong signal something is broken (a healthy
  network of any size should be able to memorize 10 images).

## Outputs (per run)

Each invocation creates a timestamped directory under
`model/benchmarks/yolo_ablation_<TS>/<backbone>/`:

| File | Contents |
|---|---|
| `meta.json` | full config + GPU + git-state-equivalent + final EPE |
| `train.csv` | per-50-step `step, loss, epe_full, lr, elapsed_s` |
| `curve.png` | loss + EPE over steps (twin axes) |
| `viz/step_<N>.png` | live viewer snapshot every ~15 s + every 100 steps: 2×2 panel of (left | GT disp | pred disp | stats) |
| `README.md` | self-contained run summary |

`meta.json` also contains an `inference_bench` block with mean / median /
p95 / std forward-pass latency at batch=1, plus FPS, recorded after the
training loop completes (10 warmup + 100 timed runs, with
`torch.cuda.synchronize()` around each call). This is the *clean*
inference number — no data loading, no loss computation — so it's
comparable across architecture variants.

## Live visualization

`overfit_yolo_ablation.py --show 1` (default) opens an OpenCV window
that refreshes every ~15 s with the current state of pair 0:

```
+---------------------+---------------------+
|     left image      |    GT disparity     |
+---------------------+---------------------+
|   pred disparity    |    stats overlay    |
+---------------------+---------------------+
```

The same panel is saved to `viz/step_<N>.png` so the trajectory is
preserved even after the window is closed. Press `q` in the window to
abort early.

## Running

```bash
# Single backbone
python3 model/scripts/overfit_yolo_ablation.py --backbone yolo26n

# Both back-to-back into one timestamped directory
TS=$(date +%Y%m%d-%H%M%S)
OUT=model/benchmarks/yolo_ablation_$TS
python3 model/scripts/overfit_yolo_ablation.py --backbone yolo26n --out_root $OUT
python3 model/scripts/overfit_yolo_ablation.py --backbone yolo26s --out_root $OUT
```

## Caveats

- **Overfit success ≠ generalization.** A model that overfits 10 pairs
  may still fail on the full 39K-pair Scene Flow. This test only
  validates wiring + capacity, not learning quality.
- **The 10 pairs are from one camera setting** (15 mm focal length,
  scene_forwards/slow). They share lighting, texture statistics, and
  baseline. Real Scene Flow training uses all 4 settings × 2 motion
  speeds for diversity.
- **No augmentation.** Crops, flips, color jitter etc. are turned off
  precisely to allow memorization. Real training uses heavy augmentation.
- **L1 multi-scale loss only.** Real training adds gradient, edge-aware
  smoothness, and bad-1 hinge terms. We strip those for the smoke test
  to isolate the matching signal.

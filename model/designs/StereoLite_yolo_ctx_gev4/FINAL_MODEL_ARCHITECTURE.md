# StereoLite-YOLO-CTX-GEV4: Final Model Architecture and Experimental Record

## 1. Document scope

This document records the finalized `StereoLite_yolo_ctx_gev4` stereo-disparity
architecture, its implementation, training objective, controlled overfit
experiment, broader architectural ablations, and the final SceneFlow checkpoint
run. It is intended to be a factual source for the thesis methodology and
experimental chapters.

The principal evidence is:

- implementation: `model/designs/StereoLite_yolo_ctx_gev4/`;
- overfit experiment: `model/benchmarks/arch_ablation_20260624-181105/yolo_ctx_gev4_yolo26s/`;
- full SceneFlow run: `model/checkpoints/yolo_ctx_gev4_full_retry_es/`;
- experiment code: `model/scripts/overfit_arch_ablation.py` and
  `model/scripts/train_arch_sceneflow.py`.

Two results must not be conflated:

1. **Controlled overfit result: 0.2610 px EPE.** The same 20 image pairs were
   optimized and evaluated. This measures model capacity and optimization, not
   unseen-data generalization.
2. **Held-out SceneFlow result: 0.9635 px EPE at step 89,000.** This was measured
   on 200 held-out resized image pairs and is the appropriate generalization
   result from the recorded full run.

## 2. Model summary

StereoLite-YOLO-CTX-GEV4 is a lightweight, coarse-to-fine stereo model combining:

- a shared, truncated YOLO26s image encoder for stereo matching;
- a separate left-image context encoder inspired by recurrent stereo models;
- a HITNet-like tile state containing disparity, local plane slopes, confidence,
  and a persistent hidden feature;
- recurrent, local-correlation ConvGRU refinement at 1/16, 1/8, and 1/4 scale;
- an IGEV-style regularized geometry encoding volume at 1/4 scale;
- a learned fail-soft gate that blends geometry-volume disparity into the
  coarse-to-fine tile estimate; and
- two learned convex upsampling stages to recover full resolution.

The final configuration has **2.9623 million trainable parameters**. At
384 x 640, batch-one inference on the recorded RTX 4070 benchmark took
**24.591 ms on average**, equivalent to **40.67 FPS**.

## 3. End-to-end data flow

```text
Left image L -----------------------+------------------------+
                                   |                        |
Right image R ----+                 |                        |
                  v                 v                        |
       shared truncated YOLO26s   left-only context encoder |
       fL/fR at /2,/4,/8,/16          context at /4          |
                  |                 /       |       \         |
                  v              ctx16    ctx8     ctx4       |
       /16 group-wise cost volume   |       |       |         |
       + 3-D regularization         |       |       |         |
                  v                 v       |       |         |
       initial tile state + context hidden state             |
                  |                                          |
       2 x /16 recurrent refinement                          |
                  |                                          |
       plane-aware tile upsample /16 -> /8                    |
                  |                                          |
       3 x /8 recurrent refinement <--- ctx8                  |
                  |                                          |
       plane-aware tile upsample /8 -> /4                     |
                  |                                          |
                  +<--- regularized /4 GEV from fL4 and fR4   |
                  |     fail-soft learned fusion              |
                  v                                          |
       3 x /4 recurrent refinement <--- ctx4                  |
                  |                                          |
       convex upsample /4 -> /2 using fL4                     |
                  |                                          |
       convex upsample /2 -> full using fL2 ------------------+
                  |
            full-resolution disparity
```

For an input tensor `(B, 3, H, W)`, the default 384 x 640 spatial sizes are:

| Scale | Spatial size | YOLO26s channels | Main operation |
|---|---:|---:|---|
| Full | 384 x 640 | 3 | Input/output |
| 1/2 | 192 x 320 | 32 | Convex-upsample guidance |
| 1/4 | 96 x 160 | 128 | GEV fusion and 3 refinements |
| 1/8 | 48 x 80 | 256 | 3 refinements |
| 1/16 | 24 x 40 | 256 | Cost initialization and 2 refinements |

## 4. Shared stereo feature encoder

The matching network is `YoloTruncatedEncoder("yolo26s")`. The same trainable
weights process both images. Left and right images are concatenated along the
batch dimension, passed through one encoder invocation, and split afterward.
This is computationally convenient and guarantees a Siamese shared-weight
encoder.

Only YOLO26s backbone layers 0 through 6 are retained:

| Output | Backbone location | Stride | Channels |
|---|---|---:|---:|
| `f2` | stem convolution | 2 | 32 |
| `f4` | first C3k2 stage | 4 | 128 |
| `f8` | second C3k2 stage | 8 | 256 |
| `f16` | third C3k2 stage | 16 | 256 |

The 1/32 stage, feature pyramid, PAN neck, and object-detection head are removed.
The encoder is initialized from YOLO26s pretrained weights and then trained
end-to-end. Its role is feature extraction only; no YOLO detection output or
detection loss is used.

## 5. Left-image context encoder

Stereo matching evidence and recurrent context are intentionally separated.
The matching encoder sees both views, while the context encoder sees only the
left image and supplies scene structure to the recurrent hidden state.

The context encoder performs:

1. input normalization from `[0,255]` to `[0,1]`;
2. a 7 x 7, stride-2 convolution with 24 channels;
3. a stride-2 GhostConv from 24 to 48 channels;
4. squeeze-and-excitation;
5. a stride-1 GhostConv with 48 channels;
6. squeeze-and-excitation; and
7. a 1 x 1 projection to a 32-channel 1/4-resolution context map.

Group normalization and SiLU activations are used. The 1/4 context is resized
to the 1/8 and 1/16 feature resolutions. Importantly, these operations are
spatial downsampling from `ctx4`; they do not recover detail from a coarse map.

At 1/16, the resized context directly replaces the zero hidden state emitted by
the initializer. This gives the ConvGRU a meaningful persistent state before
its first update.

## 6. Tile state and coarse initialization

The recurrent state at every pixel is

```text
T = (d, sx, sy, h, c)
```

where:

- `d` is disparity in pixels of the current resolution;
- `sx` and `sy` are local disparity-plane slopes;
- `h` is a 32-channel persistent ConvGRU hidden state; and
- `c` is a scalar confidence estimate.

At 1/16 resolution, `TileInit` builds a group-wise correlation volume with
8 groups and 24 disparity hypotheses. For group `g` and integer disparity `d`,
the matching score is the mean channel product between left features and the
horizontally shifted right features:

```text
C(g,d,y,x) = mean_c [ fL(g,c,y,x) * fR(g,c,y,x-d) ].
```

A small 3-D CNN regularizes this volume (`8 -> 16 -> 16 -> 1`). Softmax along
the disparity dimension produces `p(d)`. Initial disparity and confidence are:

```text
d0 = sum_d p(d) d
c0 = max_d p(d).
```

Initial slopes are zero. The emitted hidden feature is replaced by `ctx16`.
The 24 hypotheses at 1/16 correspond nominally to disparities 0 through 23 at
that scale, or approximately 0 through 368 full-resolution pixels.

## 7. Recurrent tile refinement

Separate `TileRefineCtx` modules are used at 1/16, 1/8, and 1/4 because the
matching-feature channel counts differ. Their weights are not shared.

At each iteration, the right feature map is warped toward the left view using
the current disparity. A differentiable local correlation lookup evaluates
five offsets around the current estimate (`half_range=2`). The recurrent input
contains:

```text
x = [fL, warped(fR,d), d, sx, sy, c, local correlation, context].
```

The persistent hidden state is updated by a convolutional GRU:

```text
z = sigmoid(Conv([h,x]))
r = sigmoid(Conv([h,x]))
q = tanh(Conv([r*h,x]))
h' = (1-z)*h + z*q.
```

A two-layer 48-channel head predicts residuals for disparity, both slopes, and
confidence. A second learned gate examines the updated hidden state, context,
current confidence, local cost peak, local cost entropy, and predicted update
magnitudes. Four sigmoid gates independently control the disparity, x-slope,
y-slope, and confidence updates. Disparity is constrained non-negative with
`softplus`.

The default recurrence schedule is:

| Resolution | Iterations | Local search offsets |
|---|---:|---:|
| 1/16 | 2 | -2, -1, 0, +1, +2 |
| 1/8 | 3 | -2, -1, 0, +1, +2 |
| 1/4 | 3 | -2, -1, 0, +1, +2 |

This gives eight learned recurrent updates in total.

## 8. Plane-aware cross-scale propagation

Between 1/16 and 1/8, and between 1/8 and 1/4, `TileUpsample` bilinearly
resizes disparity, slopes, hidden features, and confidence. Disparity is
multiplied by two because its pixel unit changes with resolution.

Unlike plain bilinear interpolation, the propagated disparity also uses the
local plane slopes to adjust the four child pixels around each parent tile:

```text
d_child = 2 * bilinear(d_parent) + 2*sx*dx + 2*sy*dy,
```

with child offsets `dx,dy` in `{-0.25,+0.25}`. This is the implementation's
main HITNet-like tile-plane component. It preserves a local slanted-surface
hypothesis while moving to a finer grid.

## 9. Quarter-resolution geometry encoding volume

GEV4 adds an independent global matching proposal at 1/4 resolution. A
group-wise correlation volume is formed from `fL4` and `fR4` using 8 groups
and 64 integer disparity hypotheses. Its nominal full-resolution range is
0 through 252 pixels because each 1/4-scale disparity pixel represents four
full-resolution pixels.

The volume is regularized by three 3 x 3 x 3 convolutions with 16 hidden
channels. A final 3-D convolution produces disparity logits. From the softmax
distribution, the branch computes:

```text
d_gev  = sum_d p(d) d
c_gev  = max_d p(d)
g_gev  = projection(sum_d G(d) p(d)),
```

where `G` is the regularized 3-D geometry feature and `g_gev` is a 16-channel
2-D expected geometry feature.

### Fail-soft fusion

The GEV proposal is not substituted unconditionally. A learned fusion network
receives context, expected geometry, tile confidence, GEV confidence, absolute
disagreement, and the existing tile disparity:

```text
w = sigmoid(F([ctx4, g_gev, c_tile, c_gev,
               abs(d_gev-d_tile), d_tile])).
d_fused = softplus(d_tile + w*(d_gev-d_tile)).
```

The final fusion-layer bias is initialized to `-4`, so its initial sigmoid
weight is approximately 0.018. Training therefore starts close to the stable
context-gated baseline, and the network must learn where GEV information is
useful. Slopes are attenuated by `(1-w)`, confidence becomes the maximum of the
two estimates, and the hidden state receives a small projected geometry
residual (`ctx4 + 0.1*projection(g_gev)`).

## 10. Learned convex upsampling

The final 1/4 disparity is upsampled in two 2x stages:

1. 1/4 to 1/2, with masks predicted from `fL4`;
2. 1/2 to full resolution, with masks predicted from `fL2`.

For every output subpixel, each mask predicts nine weights over a 3 x 3
neighborhood. A softmax makes these weights non-negative and sum to one. The
upsampled disparity is therefore a learned convex combination of nearby
disparities, with the proper factor-of-two disparity scaling at each stage.

This is preferable to fixed bilinear interpolation because image features can
select different source neighbors near boundaries. However, it cannot invent
high-frequency disparity structure absent from the 1/4 prediction; thin and
distant structures remain a known limitation.

## 11. Default architecture configuration

| Hyperparameter | Value |
|---|---:|
| Context base channels | 24 |
| Context output channels | 32 |
| Tile hidden channels | 32 |
| Refinement-head hidden channels | 48 |
| Initialization groups | 8 |
| 1/16 initialization disparities | 24 |
| GEV4 disparities | 64 |
| GEV4 geometry channels | 16 |
| GEV4 3-D hidden channels | 16 |
| Local cost half-range | 2 |
| Refinements at 1/16, 1/8, 1/4 | 2, 3, 3 |
| Backbone | YOLO26s |
| Backbone initialization | pretrained |
| Total/trainable parameters | 2.9623 M / 2.9623 M |

## 12. Training objective

The full training run used a multi-scale supervised loss. Ground-truth pixels
were valid when disparity was finite and in `(0,320)` pixels. Each lower-scale
prediction was resized to full resolution and multiplied by its scale factor
before comparison.

```text
L = 1.00 L1(d_full)
  + 0.50 L1(d_half)
  + 0.30 L1(d_4)
  + 0.20 L1(d_8)
  + 0.10 L1(d_16)
  + 0.15 L1(d_GEV4)
  + 0.50 L_gradient
  + 0.20 L_threshold
  + 0.20 L_D1
  + 0.02 L_edge-smoothness.
```

`L_gradient` penalizes horizontal and vertical disparity-gradient error.
`L_threshold` uses squared hinge penalties above 0.5, 1, 2, and 3 pixels.
`L_D1` applies a squared hinge above 3 pixels where relative error also exceeds
5%. Edge-aware smoothness penalizes disparity variation less strongly where
the left image has a strong intensity edge.

The overfit benchmark used the closely related baseline loss but did not yet
include the full-run threshold/D1/smoothness combination in exactly the same
form. Consequently, its loss magnitude should not be numerically compared to
the full-run loss magnitude.

## 13. Controlled overfit experiment

### Protocol

| Item | Value |
|---|---|
| Dataset subset | SceneFlow Driving, first 20 fixed pairs |
| Train/evaluation relationship | same 20 pairs |
| Input size | 384 x 640 |
| Steps | 7,000 |
| Batch size | 4 |
| Learning rate | 0.0002 |
| Seed | 42 |
| Device | NVIDIA GeForce RTX 4070 |
| PyTorch | 2.12.0+cu132 |
| Peak allocated GPU memory | 7.595 GB |
| Wall time | 2,201.12 s (36.69 min) |

### Final all-pair result

| Metric | Result |
|---|---:|
| EPE | **0.2610 px** |
| RMSE | 0.8853 px |
| Median absolute error | 0.0589 px |
| Bad-0.5 | 9.9692% |
| Bad-1.0 | 5.2450% |
| Bad-2.0 | 2.5604% |
| Bad-3.0 | 1.5651% |
| D1-all | 1.5617% |

The best logged mini-batch point was step 6,750: EPE 0.23162, RMSE 0.84619,
bad-1 4.641%, and D1-all 1.423%. This is a sampled training-batch measurement,
whereas 0.2610 is the final evaluation over all 20 pairs and should be used as
the experiment headline.

### Inference benchmark

The benchmark used batch size 1, 384 x 640 input, 10 warm-up passes, and 100
timed passes:

| Statistic | Result |
|---|---:|
| Mean latency | 24.591 ms |
| Standard deviation | 0.595 ms |
| Median latency | 24.447 ms |
| 95th percentile latency | 25.654 ms |
| Throughput from mean | 40.67 FPS |

These numbers are hardware- and software-specific and should not be presented
as platform-independent model speed.

## 14. Architecture ablation record

The following table aggregates completed architecture runs found under
`model/benchmarks/`. All use the same 20-pair overfit concept and 384 x 640
resolution, but some differ in steps and batch size. It is therefore an
engineering ablation record, not a perfectly controlled publication table.

| Architecture | Steps | Batch | Params (M) | EPE | RMSE | Median AE | Bad-1 (%) | D1 (%) | Mean ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO context | 3,000 | 4 | 2.8308 | 0.4238 | 1.1036 | 0.1670 | 7.6168 | 2.2291 | 10.730 |
| YOLO context, batch 8 | 3,000 | 8 | 2.8308 | 0.5418 | 1.1612 | 0.2775 | 11.4444 | 2.3895 | 10.494 |
| Geometry context | 3,000 | 2 | 2.9193 | 0.4585 | 1.1101 | 0.2180 | 8.1558 | 2.2219 | 36.709 |
| Context gate | 3,000 | 4 | 2.9211 | 0.4726 | 1.0962 | 0.2460 | 7.7474 | 2.1186 | 12.242 |
| Context gate, YOLO26n | 8,000 | 8 | 1.4101 | 0.3873 | 1.0255 | 0.1821 | 5.9835 | 1.8912 | 10.861 |
| Context gate, YOLO26s | 8,000 | 8 | 2.9211 | 0.3187 | 0.8941 | 0.1267 | 5.2374 | **1.5363** | 11.857 |
| Sharp tail | 7,000 | 4 | 2.9501 | 0.3217 | **0.8778** | 0.1361 | **5.2095** | 1.5492 | 14.632 |
| High-resolution refine | 7,000 | 4 | 3.1063 | 0.3512 | 0.8862 | 0.1856 | 5.1754 | 1.5394 | 16.144 |
| SRU | 7,000 | 4 | 3.3976 | 0.4738 | 1.0290 | 0.2548 | 8.2023 | 1.8136 | 14.801 |
| Raw 1/4 initialization | 6,000 | 4 | 2.9476 | 0.4553 | 1.0518 | 0.1948 | 10.6373 | 1.9127 | 20.733 |
| **Context gate + GEV4** | **7,000** | **4** | **2.9623** | **0.2610** | 0.8853 | **0.0589** | 5.2450 | 1.5617 | 24.591 |

GEV4 achieved the lowest recorded EPE and median error, indicating a strong
improvement in typical subpixel accuracy. Its bad-1 and D1 results were close
to, but not strictly better than, every competing run. It also increased
latency substantially relative to the plain context-gate model because the
64-plane 1/4-resolution 3-D volume is expensive. Thus the evidence supports an
accuracy/capacity improvement, not an across-the-board dominance claim.

For a publication-grade causal ablation, all variants should be rerun with the
same seed set, batch size, number of optimizer steps or sample exposures,
augmentation, validation split, and timing environment. Mean and standard
deviation over at least three seeds should be reported.

## 15. Full SceneFlow training run

### Configuration

| Item | Value |
|---|---|
| Run name | `yolo_ctx_gev4_full_retry_es` |
| Training pairs | 4,200 |
| Validation pairs | 200 held-out pairs |
| Training mode | full-frame resize |
| Input size | 384 x 640 |
| Maximum valid disparity | 320 px |
| Requested maximum steps | 100,000 |
| Actual final step | 89,000, early-stopped |
| Batch size | 4 |
| Optimizer | AdamW |
| Initial learning rate | 0.0002 |
| Weight decay | 0.00001 |
| Scheduler | cosine annealing to 5% of initial LR |
| Mixed precision | enabled |
| Gradient clipping | global norm 1.0 |
| Validation interval | 1,000 steps |
| Checkpoint interval | 5,000 steps |
| Early-stop metric | resized-validation EPE |
| Early-stop patience | 8 validations |
| Minimum improvement | 0.003 px |
| Minimum steps | 12,000 |
| Seed | 42 |
| Device | NVIDIA GeForce RTX 4070 |
| Peak allocated GPU memory | 9.469 GB |

Because the run used `train_mode=resize`, the scale/color/erasing settings
stored in metadata were not the active native-crop augmentation path. Each
source frame was resized directly to 384 x 640, and horizontal disparity was
scaled by the width ratio. This explains why resized validation is much better
than native center-crop validation.

### Held-out convergence milestones

| Step | Resize EPE | Bad-1 (%) | Bad-2 (%) | D1 (%) | Crop EPE |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 2.8574 | 60.545 | 38.433 | 21.818 | 5.1717 |
| 10,000 | 1.4026 | 29.238 | 17.164 | 10.992 | **3.2206** |
| 20,000 | 1.2486 | 25.116 | 15.259 | 10.075 | 3.9123 |
| 30,000 | 1.1155 | 22.731 | 13.728 | 8.899 | 3.5774 |
| 40,000 | 1.0581 | 21.426 | 12.888 | 8.403 | 3.6591 |
| 50,000 | 1.0505 | 21.467 | 12.980 | 8.396 | 4.1752 |
| 60,000 | 1.0046 | 20.462 | 12.337 | 8.083 | 4.2026 |
| 70,000 | 0.9848 | 20.039 | 12.247 | 7.976 | 4.2414 |
| 80,000 | 0.9787 | 20.036 | 12.227 | 7.988 | 4.3009 |
| 81,000 | 0.9663 | **19.676** | **11.982** | **7.864** | 4.3043 |
| 89,000 | **0.9635** | 19.676 | 12.042 | 7.876 | 4.3695 |

All milestone values above are transcribed from the validation rows in
`train.csv`; the CSV remains the authoritative per-step record.

### Final and best checkpoint semantics

The run contains two important checkpoints:

- `best.pth`: step **81,000**, stored best EPE **0.9663327**, bad-1
  **19.67557%**, bad-2 **11.98233%**, D1 **7.86408%**.
- `latest.pth`: step **89,000**, raw EPE **0.9634784**, bad-1 **19.67595%**,
  bad-2 **12.04191%**, D1 **7.87579%**.

Although step 89,000 has a numerically lower EPE, it improved over the stored
best by only 0.0028543 px, less than the configured 0.003 minimum delta.
Therefore it did not replace `best.pth`; it became the eighth consecutive
non-qualifying validation and triggered early stopping.

For reproducible deployment, use `best.pth`, because that is the checkpoint
selected by the declared early-stopping rule. For analysis of the absolute
lowest sampled resize EPE, report step 89,000 as the final/latest result and
explicitly state this distinction.

The final logged 50-step training-window statistics at step 89,000 were loss
8.26446, train EPE 0.95564, and train bad-1 19.6071%. The minimum logged
training-window EPE was 0.87663 at step 80,400. Small train/resize-validation
separation near the end does not indicate severe conventional overfitting on
the resized domain. In contrast, crop EPE degraded from 3.2206 at step 10,000
to 4.3695 at step 89,000, demonstrating poor transfer from resized full frames
to native-resolution local crops.

The training run lasted approximately 26,335 seconds in the CSV timer, or
7.32 hours.

## 16. Metric definitions

- **EPE:** mean absolute disparity error over valid pixels.
- **RMSE:** square root of mean squared disparity error.
- **Median AE:** median absolute disparity error, robust to large outliers.
- **Bad-t:** percentage of valid pixels with absolute error greater than
  threshold `t` pixels.
- **D1-all:** percentage with absolute error greater than 3 pixels and relative
  error greater than 5% of ground-truth disparity.

The full-run evaluator averages each image's metric over 200 images rather than
first pooling every valid pixel globally. This image-mean convention should be
kept consistent when comparing future runs.

## 17. Interpretation of visual quality

The model's strong median error and EPE show that most broad surfaces and many
object regions are reconstructed accurately. The remaining visual softness is
consistent with the architecture:

- the last recurrent prediction is only 1/4 resolution;
- convex upsampling selects mixtures from a 3 x 3 disparity neighborhood but
  cannot create a missing foreground/background hypothesis;
- the GEV branch is also 1/4 resolution and capped at 64 hypotheses;
- training by global pixel losses is dominated by large smooth regions;
- resize-only training reduces native high-frequency structure; and
- occlusion boundaries, foliage, poles, and thin distant objects occupy a
  small fraction of pixels but dominate visual judgment.

Therefore low EPE and perfect boundary appearance are related but not
equivalent objectives. The crop-validation degradation is direct evidence
that the final run learned the resized training domain better than native
fine-detail geometry.

## 18. Strengths, limitations, and thesis-safe claims

### Supported strengths

- The network is compact at 2.9623 M parameters.
- It runs in real time on the measured RTX 4070 setup at 40.67 FPS for
  384 x 640 batch-one input.
- It can fit a 20-pair subset to 0.2610 EPE, demonstrating sufficient model
  capacity for accurate subpixel disparity on that subset.
- It reaches below 1.0 EPE on a held-out resized SceneFlow split.
- Fail-soft GEV fusion produced the best EPE among the recorded architecture
  experiments while adding only about 0.0412 M parameters over context gate.
- The design combines explicit stereo correspondence, recurrent refinement,
  local tile planes, semantic context, and learned edge-aware upsampling.

### Important limitations

- The 0.2610 result is an overfit result and must never be presented as test
  or validation performance.
- The broader ablation table is not fully controlled because steps and batches
  differ.
- The final checkpoint was evaluated only on the repository's held-out
  SceneFlow split; no official SceneFlow test, KITTI, Middlebury, or ETH3D
  benchmark result is recorded here.
- Resize-only training generalizes poorly to native crops.
- The model has no explicit left-right consistency or occlusion head.
- Final recurrent reasoning stops at 1/4 resolution, limiting thin-boundary
  recovery.
- GEV4's nominal 252-pixel full-resolution range is lower than the 320-pixel
  valid-training cutoff, although the coarse 1/16 branch and recurrent updates
  can represent larger disparities.
- Runtime was measured on one GPU/software setup without cross-model memory or
  FLOP profiling under a standardized benchmark.
- A single seed does not establish statistical significance.

### Recommended thesis wording

Use wording such as:

> StereoLite-YOLO-CTX-GEV4 is a 2.96 M-parameter hybrid stereo architecture
> that combines a truncated YOLO feature encoder, left-image recurrent context,
> plane-aware coarse-to-fine tile refinement, and a fail-soft quarter-resolution
> geometry encoding volume. On our 200-pair held-out resized SceneFlow split,
> the final run achieved 0.9635 px EPE; the early-stopping-selected checkpoint
> achieved 0.9663 px EPE. A separate 20-pair capacity experiment reached
> 0.2610 px EPE on the optimized samples.

Do not call the model state of the art until it is evaluated under the official
protocol of the target benchmark and compared against published methods using
the same resolution, training data, metric definition, and hardware conditions.

## 19. Reproduction and artifact map

Important artifacts:

| Artifact | Purpose |
|---|---|
| `model.py` | End-to-end GEV4 model |
| `tile_propagate.py` | Tile state, cost lookup, ConvGRU, gated update, plane upsampling |
| benchmark `meta.json` | Exact overfit setup and final metrics |
| benchmark `train.csv` | Per-50-step overfit trajectory |
| benchmark `curve.png` | Overfit learning curves |
| benchmark `viz/` | Rotating visual samples during optimization |
| checkpoint `meta.json` | Full-run setup and architecture configuration |
| checkpoint `train.csv` | Full train/validation trajectory |
| checkpoint `best.pth` | Rule-selected step-81,000 checkpoint |
| checkpoint `latest.pth` | Early-stopped step-89,000 checkpoint |
| checkpoint `step_*.pth` | Periodic snapshots every 5,000 steps |
| checkpoint `step_*_resize.png` | Recorded resized-domain visual panels |

The checkpoint files contain model state, optimizer state, scheduler state,
mixed-precision scaler state, step, stored best EPE, command arguments, and the
validation metrics available at save time. This supports exact continuation,
provided the same model code and dependency versions are retained.

## 20. Final reported numbers

For quick reference:

| Category | Result |
|---|---:|
| Parameters | 2.9623 M |
| Controlled 20-pair overfit EPE | 0.2610 px |
| Controlled overfit bad-1 | 5.2450% |
| Controlled overfit D1-all | 1.5617% |
| Held-out resize EPE, selected `best.pth`, step 81k | 0.9663 px |
| Held-out resize EPE, final `latest.pth`, step 89k | **0.9635 px** |
| Held-out resize bad-1, step 89k | 19.6760% |
| Held-out resize bad-2, step 89k | 12.0419% |
| Held-out resize D1, step 89k | 7.8758% |
| Native center-crop EPE, step 89k | 4.3695 px |
| RTX 4070 mean latency at 384 x 640, batch 1 | 24.591 ms |
| RTX 4070 throughput | 40.67 FPS |
| Peak memory, overfit batch 4 | 7.595 GB |
| Peak memory, full training batch 4 | 9.469 GB |

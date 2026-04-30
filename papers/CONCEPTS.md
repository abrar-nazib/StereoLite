# Concepts

Short, plain-language definitions of stereo matching concepts collected during
study.

**Format rules** (apply when adding new entries):
- Concepts in **alphabetical order**.
- Each entry has two parts: a tight **definition** (2-4 sentences) and a
  concrete **Example** that grounds it.
- No tables, no lengthy trade-off lists; ask for depth in chat instead.

## Channel

A normal photo has 3 channels: R, G, B (three numbers per pixel saying how
much red, green, blue is present). A "256-channel feature map" is the same
idea, just with 256 numbers per pixel instead of 3. Each number is the
response of one learned filter, firing wherever its specific pattern
(vertical edge, fur-like texture, etc.) appears in the image.

**Example:** a feature map of shape (B, C=128, H=32, W=52) means each of
the 32×52 spatial positions carries a 128-dimensional feature vector. At
coarser scales the channels typically encode increasingly abstract
patterns (early: edges, mid: textures, late: object parts).

## Coarse Disparity

A disparity map computed at a downsampled resolution (typically 1/8 or 1/16
of input), giving one disparity value per patch instead of one per pixel.
Used as a seed in coarse-to-fine pipelines: cheap to compute, each cell sees
wider context (helpful in textureless regions), but it loses sub-pixel
accuracy and blurs object edges; refinement stages downstream bring the
detail back.

**Example:** for a 512×832 input at 1/16, the coarse disparity map is
32×52 = 1,664 values, each describing the average shift for a 16×16 patch.
A 24-bin search at that scale still covers 24 × 16 = 384 native pixels of
disparity range.

## Context Encoder

A second encoder used in iterative-refinement stereo networks (RAFT-Stereo
and its descendants). Runs only on the **left** image and produces features
whose job is to *condition* the refinement step, not to match. While the
matching cost is built from a separate "feature encoder" that runs on both
L and R, the context encoder feeds the refiner additional info about
*what's at this pixel* (semantics, edges, scene structure) so it knows *how*
to update the disparity.

**Example:** in RAFT-Stereo the context encoder's output is split in two:
one half initializes the GRU's hidden state, the other is concatenated to
the GRU input on every iteration. So at each step the refiner sees "what
does the cost volume say + what kind of region am I in?". A pixel on a
flat road is nudged differently than a pixel on a thin pole even when
their cost-volume readings look similar.

## Feature Encoder

A neural network module (almost always a CNN) that turns a raw image into
a tensor of abstract features. Each spatial position in the output
describes patterns at the corresponding region of the input (edges,
textures, parts) instead of raw RGB. In stereo, the same encoder runs on
both left and right images so the two views are described in the same
"language" before matching.

**Example:** StereoLite's encoder maps a (3, 384, 640) RGB image to four
feature maps at strides (2, 4, 8, 16) with channel counts (24, 48, 72, 96)
for the GhostConv default. Spatial resolution shrinks while channel count
grows, a pattern shared across ResNet, MobileNet, EfficientNet, and YOLO
backbones.

## Optical Flow

The per-pixel 2D motion vector (u, v) between two consecutive video frames.
For each pixel at time t, flow tells you how many pixels it moved in x and
y by time t+1. Stereo matching is a special case where motion is purely
horizontal (v ≈ 0), which is why stereo and optical-flow networks share so
much architecture (cost volumes, iterative refinement, etc.).

**Example:** a person walking right-to-left across a static scene produces
flow vectors pointing left at their silhouette and near-zero vectors on the
background. RAFT (Teed & Deng, ECCV 2020) was the iterative-refinement
breakthrough on flow that stereo later borrowed as RAFT-Stereo (2021).

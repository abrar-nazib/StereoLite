# HybridNets: End-to-End Perception Network

**Authors:** Dat Vu, Bao Ngo, Hung Phan (FPT University, Hanoi)
**Venue:** Pattern Recognition and Image Analysis 2022, arXiv:2203.09035
**Tier:** 3 (BiFPN-based multi-task driving perception, direct YOLOP follow-on)

---

## Core Idea
HybridNets is a deliberate upgrade of YOLOP's "shared encoder, multiple decoders" recipe with three specific changes: replace the CSPDarknet backbone with **ImageNet-pretrained EfficientNet-B3**, replace the FPN+PAN neck with a weighted **BiFPN** (from EfficientDet), and **fold YOLOP's two binary segmentation heads into a single 3-class head** (background, drivable area, lane line). The first two boost feature quality; the third reduces decoder duplication. Together with automatically-tuned k-means anchors and a Tversky+Focal segmentation loss, the result is fewer FLOPs than YOLOP despite more parameters.

## Architecture
- **Backbone:** EfficientNet-B3, ImageNet-pretrained; produces feature levels P1 to P5 (1/2 to 1/32). Page 5 to 6.
- **Neck:** BiFPN (bidirectional FPN with learned per-feature weights). P6 and P7 levels created by downsampling P5. Cross-scale connections in both top-down and bottom-up paths.
- **Detection head:** anchor-based, **9 anchors per grid cell** (3 scales x 3 aspect ratios) chosen automatically by k-means on BDD100K. Smooth L1 box loss + focal class/obj loss. IoU match threshold 0.5 for boxes larger than 100 px, 0.25 for smaller. Page 6 to 9.
- **Segmentation head:** consumes 5 levels P3 to P7 from neck, upsamples each to (W/4, H/4, 64), sums them, restores to (W, H, 3) for the 3 classes. Also fuses raw backbone P2 to inject low-level features. Page 7.
- **Input:** 640x384, chosen for aspect-ratio preservation and divisibility by 128 (BiFPN requirement).
- **Total:** **12.83 M params, 15.6 BFLOPs**. AdamW optimizer, mosaic augmentation during detection-only stage, 200 epochs on RTX 3090, batch 16.

## Main Innovation
**Single 3-class segmentation head instead of YOLOP's two binary heads.** YOLOP runs two separate decoders for drivable-area and lane-line; HybridNets argues this is wasteful because the two classes are mutually exclusive at the pixel level (lane pixels are not drivable, drivable pixels are not lane), so a single softmax head models the relationship directly. Combined with BiFPN's learned cross-scale weighting and a Tversky+Focal loss that handles the lane-line class imbalance, this enables detecting "incredibly small objects ranging from 3 to 10 pixels" (page 14) that the older FPN-only YOLOP misses.

## Key Benchmark Numbers
- **Total params:** 12.83 M (page 13, Table 1; abstract). **FLOPs:** 15.6 BFLOPs (lower than YOLOP's 18.6 BFLOPs despite more parameters, via depthwise separable convolutions in BiFPN).
- **Latency:** **37 ms on V100 FP16** at 640x384 batch 1, **1.4x faster than YOLOP's 52 ms** on the same setup (Table 1, page 13). No Jetson / edge device benchmark provided.
- **Training data:** BDD100K (same protocol as YOLOP); 70K train / 10K val. {car, truck, bus, train} merged into one "vehicle" class; {direct, alternative} drivable merged.
- **Vehicle detection (Table 2, page 14):** **Recall 92.8%, mAP50 77.3%**; 3.6 pt recall over YOLOP (89.2), best in the table; beats DLT-Net (89.4 / 68.4), MultiNet (81.3 / 60.2), Faster R-CNN (77.2 / 55.6).
- **Drivable area (Table 3, page 16):** **mIoU 90.5%**, slightly **below YOLOP's 91.5%** (-1.0 pt). Authors attribute this to using a shared 3-class head vs YOLOP's dedicated 2-class head.
- **Lane detection (Table 4, page 18):** **Accuracy 85.4%, IoU 31.6%**; beats YOLOP (70.5 / 26.2) by +14.9 pt accuracy and +5.4 pt IoU; large jump driven by BiFPN multi-scale fusion and Tversky loss for the imbalanced lane class.

## Multi-Task Coupling: Load-Bearing or Decorative?
**Not directly testable from this paper; the authors do not provide a joint-vs-single-task ablation.** No equivalent of YOLOP Table 5 appears anywhere in HybridNets. The paper's empirical comparisons are exclusively *between HybridNets and YOLOP / MultiNet / DLT-Net at the multi-task level*, never against a HybridNets-encoder-only-with-one-head baseline. So the question "would HybridNets do better on detection if it didn't have to share the encoder with segmentation?" is unanswered. Indirectly, the **drivable mIoU drops from 91.5% (YOLOP, two heads) to 90.5% (HybridNets, fused 3-class head)** suggests task coupling has a small negative effect on the dominant-class metric, while **lane IoU jumps from 26.2% to 31.6%** suggests the same coupling helps the under-represented class; net effect: **marginal task interaction, dominated by architecture changes (BiFPN, EfficientNet, anchor tuning), not by joint-task gradient sharing.**

## Relevance to Our Project
- **BiFPN is the missing piece in our YOLO26 stack.** YOLOv26 uses a YOLO-style PAN+FPN neck; if we add segmentation or drivable-area heads, HybridNets shows BiFPN's weighted cross-scale fusion gives +14.9 pt lane accuracy over plain FPN+PAN at comparable cost. Worth prototyping on top of `model/designs/StereoLite_yolo/`.
- **Edge latency unproven for HybridNets.** All quoted speeds are V100 FP16 (37 ms / ~27 FPS); the paper never benchmarks Jetson TX2 / Xavier / Orin Nano. Given 12.83 M params and BiFPN's memory traffic, real-time on Jetson Orin Nano is plausible but not demonstrated. Our 2.06 M YOLO26s mid-tier is ~6x smaller.
- **Single softmax head pattern transfers to stereo.** If StereoLite ever produces a "free space + obstacle + lane" attention prior for drone navigation, the HybridNets unified-decoder pattern (5 BiFPN levels summed at 1/4 resolution, restored to full res) is simpler than YOLOP's per-class deconv stacks.
- **Anchor auto-tuning is a free win.** Our YOLO26 inherits COCO anchors; HybridNets shows k-means on the deployment dataset alone is enough to detect 3 to 10 px objects. Useful when re-aiming the mid-tier for drone-on-drone perception.

## Limitations / What This Paper Doesn't Solve
- **No isolated-task baseline.** Cannot tell whether the joint training helps or hurts each individual task on this architecture.
- **No edge benchmark.** V100 FP16 is not a deployment target; the deployment story for Jetson-class hardware is left implicit. Cannot directly compare against YOLOP's 23 FPS on TX2.
- **Authors admit crossroad failures (page 19):** lane detection breaks, and drivable area can be misjudged onto the opposite side of the road. The architecture has no explicit topological prior.
- **No depth or stereo.** Same gap as YOLOP; the architecture is photometric-only and 2D.

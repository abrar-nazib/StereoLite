# YOLOP: You Only Look Once for Panoptic Driving Perception

**Authors:** Dong Wu, Manwen Liao, Weitian Zhang, Xinggang Wang, Xiang Bai, Wenqing Cheng, Wenyu Liu (Huazhong University of Science and Technology)
**Venue:** Machine Intelligence Research (MIR) 2022, arXiv:2108.11250
**Tier:** 3 (canonical real-time multi-task driving network, first to hit Jetson TX2 real-time on three tasks)

---

## Core Idea
Treat panoptic driving perception as a single forward pass: one CSPDarknet/SPP/FPN encoder feeds three decoders (object detection, drivable area segmentation, lane segmentation) that all consume the same shared feature pyramid. The authors argue the bottleneck for embedded perception is not any single task's accuracy but the latency of running three separate networks sequentially, so even an architecturally simple shared-encoder design is a win if it stays close to per-task SOTA. The paper's central empirical claim is that **end-to-end joint training matches or beats step-by-step alternating training**, so the three driving tasks are mutually compatible under one CNN.

## Architecture
- **Backbone:** CSPDarknet (from YOLOv4); produces multi-scale features (1/8, 1/16, 1/32). Page 3.
- **Neck:** SPP module (spatial pyramid pooling at 1/32) plus FPN top-down path. Features fused by concatenation. Page 3.
- **Detection head:** PAN bottom-up path on top of FPN, anchor-based at three scales, k-means anchor priors, three prior anchors per grid cell. CIoU box loss + focal class/obj loss. Page 4.
- **Drivable area head:** feeds from the *bottom* of FPN at (W/8, H/8, 256), three nearest-neighbour upsamples back to (W, H, 2), no extra SPP. Page 4.
- **Lane line head:** identical structure to drivable area, output (W, H, 2), but loss adds an IoU term on top of cross-entropy because lanes are a sparse class. Page 4 to 5.
- Input resized to 640x384 from BDD100K's native 1280x720. Adam optimizer, warm-up + cosine annealing.

## Main Innovation
Two coupled claims, validated by ablation: (1) **grid-based detection heads are more compatible with semantic segmentation heads than region-based detection heads** because both make pixel-wise dense predictions on the same feature map; region proposals introduce a representational mismatch (Table 6 ablation, page 12). (2) **End-to-end joint training is sufficient**; the four-step alternating optimization protocol that worked for Faster R-CNN is unnecessary here (Table 4, page 12).

## Key Benchmark Numbers
- **Total params and latency:** ~7.9 M (per HybridNets paper Table 1, page 13, which retrains YOLOP for fair comparison); 18.6 BFLOPs; **41 FPS on NVIDIA TITAN XP** at 640x384, **23 FPS on Jetson TX2** (abstract, page 1). YOLOP paper itself does not list a single param-count number explicitly.
- **Training data:** BDD100K, 70K train / 10K val / 20K test (test labels withheld; eval on val).
- **Vehicle detection (Table 1, page 6):** Recall 89.2%, mAP50 76.5%, 41 FPS. Beats MultiNet (81.3 / 60.2 / 8.6 FPS), DLT-Net (89.4 / 68.4 / 9.3 FPS), Faster R-CNN (81.2 / 64.9 / 8.8 FPS); slightly below YOLOv5s (86.8 / 77.2 / 82 FPS) which has no segmentation heads.
- **Drivable area segmentation (Table 2, page 6):** **mIoU 91.5%**, 41 FPS. Beats MultiNet (71.6%), DLT-Net (71.3%), PSPNet (89.6% at 11.1 FPS).
- **Lane detection (Table 3, page 10):** **Accuracy 70.5%, IoU 26.2%**, 41 FPS. Beats ENet (34.12 / 14.64 / 100 FPS), SCNN (35.79 / 15.84 / 19.8 FPS), ENet-SAD (36.56 / 16.02 / 50.6 FPS) by a wide margin.

## Multi-Task Coupling: Load-Bearing or Decorative?
**Marginal / decorative.** Table 5 (page 12) compares joint multi-task training against the same network trained on each task in isolation:

| Setting | Det Recall / mAP | Drivable mIoU | Lane Accuracy / IoU | Speed (ms/frame) |
|---|---|---|---|---|
| Det only | 88.2 / 76.9 | -- | -- | 15.7 |
| Da-Seg only | -- | 92.0 | -- | 14.8 |
| Ll-Seg only | -- | -- | 79.6 / 27.9 | 14.8 |
| Multitask | 89.2 / 76.5 | 91.5 | 70.5 / 26.2 | 24.4 |

Joint training **gains 1.0 pt recall on detection** but **loses 0.5 pt drivable mIoU and 9.1 pt lane accuracy / 1.7 pt lane IoU** versus single-task. The honest read: multi-tasking does not improve performance; the value is purely the **2.0x latency saving** (24.4 ms versus running three nets sequentially at ~45 ms). The shared encoder is a compute amortizer, not a representation-improver. Table 6 (page 12) does show that the region-based alternative (R-CNNP) suffers a much bigger multi-task penalty (Det only 79.0 / 67.3 to Multitask 77.2 / 62.6, a 4-7 pt drop), so the YOLOP architecture is at least multi-task-stable, which the region-based one is not.

## Relevance to Our Project
- **Direct template for a YOLO26n-based perception head.** If StereoLite's mid-tier YOLO26s encoder is later asked to also emit a drivable area or obstacle mask for drone navigation, YOLOP's "one neck, two simple deconv-upsample heads" recipe is the minimal-disruption way to add it; the per-task latency budget is only ~10 ms on top of detection on Pascal-class hardware.
- **Edge latency in our band:** 23 FPS on Jetson TX2 at 640x384 corresponds to ~43 ms; comparable to our 25 to 54 ms StereoLite envelope on RTX 3050 / Jetson Orin Nano. YOLOP confirms that ~8 M params + FPN + 3 heads is roughly the right scale for sub-50-ms real-time.
- **Cautionary signal on joint training:** Table 5 shows the joint pass loses lane-segmentation accuracy versus single-task. If we ever add stereo + segmentation heads on a shared encoder, expect each task to give up 1 to 9 percentage points compared to its dedicated baseline; budget for that.
- **Grid-versus-region lesson:** for stereo + det, both tasks are dense pixel predictions, so the YOLOP-style coupling should transfer cleanly. Avoid Faster-R-CNN-style two-stage heads on a shared stereo backbone.

## Limitations / What This Paper Doesn't Solve
- **No depth or stereo head.** The conclusion (page 13) explicitly flags depth estimation as future work; this paper offers no architectural template for fusing stereo cost volumes with detection/segmentation features on the same encoder.
- **Lane detection accuracy still poor in absolute terms:** 70.5% pixel accuracy + 26.2% IoU is "best in the table" but the lanes are visibly noisy in Figure 8. Authors admit the lane IoU is low because of the 8-pixel dilation pre-processing trick rather than a true model improvement.
- **Single dataset (BDD100K).** No cross-domain generalization study; cannot infer whether the multi-task gains hold on KITTI / Cityscapes / drone footage.
- **Joint training does not improve any task** versus its single-task counterpart (Table 5). The contribution is latency amortization, not representation sharing. If we want representation gains, this paper is not the right template.

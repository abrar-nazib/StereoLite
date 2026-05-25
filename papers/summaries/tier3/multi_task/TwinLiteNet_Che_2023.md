# TwinLiteNet: An Efficient and Lightweight Model for Driveable Area and Lane Segmentation in Self-Driving Cars

**Authors:** Quang-Huy Che, Dinh-Phuc Nguyen, Minh-Quan Pham, Duc-Khai Lam (UIT, VNU-HCM)
**Venue:** Multi-disciplinary Conference on Mechanical Engineering and Mechatronics 2023, arXiv:2307.10705
**Tier:** 3 (the extreme-lightweight YOLOP/HybridNets follow-on, 0.4 M parameters, Jetson Xavier real-time)

---

## Core Idea
Drop the object-detection head entirely, keep the two segmentation heads (drivable area + lane line), and shrink the encoder until you can run on Jetson TX2-class hardware at real-time. The authors observe that for many ADAS pipelines, lane and drivable-area segmentation are the load-bearing perception signals and detection can come from a separate stack; so a 0.4 M-parameter, segmentation-only twin-decoder net is the right operating point. The model is ESPNet-C (dilated-conv encoder) + Dual Attention Module + two ConvTranspose decoder heads.

## Architecture
- **Backbone (encoder):** ESPNet-C, a stack of Efficient Spatial Pyramid (ESP) modules with dilated convolutions. Output feature map A is shape (32, H/8, W/8); only **32 channels** at 1/8 resolution. Page 2 to 3.
- **Neck:** **Dual Attention Module** (PAM + CAM from "Dual Attention Network for Scene Segmentation", Fu et al. 2018). Position Attention Module captures spatial dependencies; Channel Attention Module captures inter-channel dependencies. Outputs summed element-wise to give feature B at (32, H/8, W/8). Page 3.
- **Two decoders:** identical structure, one per task. Each is a stack of ConvTranspose + BatchNorm + pReLU; restores the 32-channel 1/8 feature to (W, H, 1) per task. Page 3.
- **Loss per head:** Focal loss + Tversky loss, summed. Identical to HybridNets' segmentation loss. Page 3.
- **Input:** 640x360 BDD100K. Adam optimizer, 100 epochs, batch 32, RTX A5000. Convolution+BN merging via re-parameterization at inference time. Page 3.
- **Total:** **0.4 M params** (439,339 at the fully-loaded config per Table IV).

## Main Innovation
**Multi-output (two heads) rather than multi-class (one head) for two segmentation tasks**; the deliberate opposite of HybridNets' design choice. The authors argue that drivable area and lane line are conceptually different (region versus line) and benefit from independent optimization, so they keep YOLOP's two-decoder layout. Table IV makes the trade-off explicit: adding the second head improves drivable mIoU by 2.0 pt and lane IoU by 5.48 pt at the cost of only ~2.7 K extra parameters and 25 FPS. The combination of (ESPNet dilated convs) + (Dual Attention) + (two simple ConvTranspose heads) + (re-parameterization at inference) is the assembled recipe.

## Key Benchmark Numbers
- **Params:** **0.4 M** (Table I, page 3); **20x smaller than YOLOP (7.9 M), 32x smaller than HybridNets (12.83 M), 97x smaller than YOLOPv2 (38.9 M)**.
- **Speed:** **415 FPS on RTX A5000** at 640x360 (Table I). YOLOP 93 FPS / YOLOPv2 95 FPS / HybridNets 25 FPS on the same device.
- **Edge devices (Section IV-E):** **Jetson Xavier NX: 60 FPS**, **Jetson TX2: 25 FPS** with TensorRT. Real-time on both. The only paper in this group that benchmarks Jetson explicitly.
- **Training data:** BDD100K; 70K train / 10K val, 1280x720 resized to 640x360.
- **Drivable area (Table II, page 3):** **mIoU 91.3%**; beats MultiNet (71.6), DLT-Net (71.3), PSPNet (89.6), HybridNets (90.5); -0.2 pt vs YOLOP (91.5), -1.9 pt vs YOLOPv2 (93.2).
- **Lane detection (Table III, page 3):** **IoU 31.08%**; beats ENet (14.64), SCNN (15.84), R-101-SAD (15.96), ENet-SAD (16.02), YOLOP (26.20), YOLOPv2 (27.25); -0.52 pt vs HybridNets (31.6).
- **Power and thermal:** monitored on TX2 / Xavier NX with figures (Figure 5) but no exact numbers in the body; the model is positioned as battery-friendly.

## Multi-Task Coupling: Load-Bearing or Decorative?
**Load-bearing for lane detection, marginal for drivable area.** Table IV (page 6) ablates Dual Attention + Multiple Head + Re-parameterization:

| Dual-Attn | Multi-Head | Re-param | Drivable mIoU | Lane IoU | FPS | Params |
|---|---|---|---|---|---|---|
| -- | -- | -- | 88.7 | 24.7 | 530 | 417 K |
| yes | -- | -- | 89.3 | 25.6 | 425 | 437 K |
| yes | yes | -- | 91.3 | 31.08 | 400 | 440 K |
| yes | yes | yes | 91.3 | 31.08 | 415 | 439 K |

Going from one shared head to two task-specific decoders gives **+2.0 pt drivable mIoU and +5.48 pt lane IoU** at almost zero parameter cost. The lane gain is dramatic and confirms the "lane line is not just background-of-drivable" intuition. The Dual Attention Module separately adds +0.6 / +0.9 pt; small but cheap. Conclusion: **multi-head decoupling is load-bearing (delta > 5% on lane IoU)**, attention is marginal, re-param is purely inference acceleration.

## Relevance to Our Project
- **Strongest precedent for edge-tier multi-task on Jetson.** TwinLiteNet's 25 FPS on TX2 (with TensorRT, 0.4 M params) sits inside our edge-tier envelope. If we ever add drivable-mask + lane head to StereoLite's 0.87 M edge tier, this is the architectural and parameter-count template; total budget could stay under 1.5 M.
- **Multi-head not multi-class is the right pattern.** TwinLiteNet's ablation directly contradicts HybridNets' "fuse into one 3-class head" choice and shows +5 pt IoU on the smaller class from keeping heads separate. For us this matters if we ever fuse stereo disparity + obstacle mask: keep separate heads off the shared encoder.
- **ESPNet + Dual Attention combination is cheap and effective.** A possible mid-tier alternative neck for StereoLite if BiFPN is too heavy.
- **Re-parameterization trick is free latency.** Folding Conv+BN at inference is what enables 415 FPS without retraining; trivially applicable to our YOLO26n encoder.

## Limitations / What This Paper Doesn't Solve
- **No object detection head, no stereo, no depth.** It is a two-task segmentation-only network. To match YOLOP/HybridNets functionality you still need a separate detector running in parallel; which may erase the latency win.
- **Single dataset (BDD100K).** No KITTI, Cityscapes, or cross-domain results. The 91.3 / 31.08 numbers may not transfer to indoor or drone footage.
- **Lane IoU is 0.52 pt below HybridNets.** At 30x fewer parameters this is impressive, but the absolute IoU (31%) is still low; these lanes are not production-ready without post-processing (curve fitting, temporal smoothing).
- **No explicit task-conflict analysis.** Table IV adds heads but does not isolate single-task baselines, so we cannot tell whether each task by itself would benefit from the full 0.4 M parameters routed to one head.

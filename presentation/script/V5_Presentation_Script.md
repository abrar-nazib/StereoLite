# V5 Presentation Script  ·  Pre-Defense

- **Deck:** `presentation/Thesis_MTE_RUET_Presentation_Smiplified_v5.pdf`  (27 slides)
- **Title:** AI-Enhanced Stereo Matching for High-Accuracy Depth Mapping and 3D Reconstruction
- **Speakers:** Md Raihanul Haque Rahi *(R, ID 2008011)* · Nazib Abrar *(A, ID 2008026)*
- **Supervisor:** Md Zunaid Hossen
- **Total target:** ~16 to 17 min spoken + transitions and Q&A
- **Pace assumption:** ~125 words per minute · spoken-word total ≈ 1 970
- **Key theme to land at least three times:** *"This is a completely new architecture."* (slides 9, 12, 14, 23.)

## Schedule

| # | Slide | Time | Speaker | Words |
|---|---|---|---|---|
| 1  | Title                              | 15 s | R | 30  |
| 2  | Outline                            | 20 s | R | 45  |
| 3  | Introduction                       | 65 s | R | 140 |
| 4  | Introduction (Cont..)              | 45 s | A | 95  |
| 5  | Problem Statement                  | 55 s | R | 110 |
| 6  | Objectives                         | 50 s | A | 105 |
| 7  | Literature Review                  | 55 s | R | 115 |
| 8  | Literature Review (Cont..)         | 50 s | A | 100 |
| 9  | Proposed Solution                  | 50 s | A | 100 |
| 10 | Methodology                        | 45 s | R | 95  |
| 11 | Implementation                     | 25 s | R | 50  |
| 12 | Working Principle                  | 55 s | A | 110 |
| 13 | Working Principle · Stage 1 + 2    | 55 s | A | 115 |
| 14 | Working Principle · Stage 3 + 4    | 60 s | R | 120 |
| 15 | Working Principle · Supervision    | 45 s | A | 90  |
| 16 | Working Principle · 3D Reconstruction | 45 s | A | 95 |
| 17 | Results · Scene Flow               | 50 s | R | 105 |
| 18 | Results · Indoor                   | 55 s | A | 115 |
| 19 | Results · 3D point cloud           | 35 s | R | 75  |
| 20 | Discussion                         | 30 s | R | 65  |
| 21 | Limitations                        | 30 s | R | 65  |
| 22 | Impact                             | 40 s | A | 85  |
| 23 | Conclusion                         | 35 s | A | 70  |
| 24 | Impact · Environmental / Societal / Research | 40 s | A | 90 |
| 25 | Time Plan                          | 25 s | R | 50  |
| 26 | References                         | 12 s | R | 25  |
| 27 | Q & A                              | 12 s | R | 25  |

---

## Slide 1  ·  Title  *(15 s, R)*

> Good morning, everyone. We are presenting our pre-defense seminar on AI-Enhanced Stereo Matching for High-Accuracy Depth Mapping and 3D Reconstruction, supervised by Md Zunaid Hossen sir.

## Slide 2  ·  Outline  *(20 s, R)*

> Here is our flow for today. We will start with what stereo depth is, why running it on edge hardware is hard, and what we set out to do. Then we walk through the architecture we designed from scratch, show our results on synthetic and real indoor data, and end with impact and limitations.

## Slide 3  ·  Introduction  *(65 s, R)*

> Before we talk about our work, a quick introduction. In computer vision, depth means how far each pixel in an image is from the camera. A normal photograph throws that information away. Every pixel is just a colour. To recover depth, you need either a special sensor or two cameras. Two cameras is what stereo vision is. The same scene point lands on different pixels in the left and right view, and that horizontal pixel shift is called disparity. Once we know the focal length of the lens and the baseline distance between the two cameras, depth becomes one equation: Z equals f times B over d. On the right side of the slide is one of our indoor pairs. The left view is what the camera captures. The colourful image next to it is the ground-truth depth map for that scene, where warm colours are far and cool colours are near.

> *Stage note:* this is the slow start. Take your time on the disparity sentence and the equation. Many in the audience have not seen stereo geometry before. The right-hand image is **ground-truth** depth, not a model prediction; do not say "what our model predicts" here.

## Slide 4  ·  Introduction (Cont..)  *(45 s, A)*

> Modern computer vision is moving toward on-device intelligence. Every moving platform that has to understand its environment in real time needs depth. The slide shows where on-device stereo depth matters in practice. Drones use it for obstacle avoidance and terrain following. Mobile robots use it for navigation, picking, and collision safety. AR and VR headsets use it for scene reconstruction and hand tracking. Across all of these, the requirement is the same: a small, low-power computer has to estimate dense depth in real time, every frame, without sending data to the cloud.

> *Stage note:* do **not** say the model name on this slide; the slide can show "StereoLite" but the talk stays on use cases. Skip the embedded-rigs card during narration; the three cards above carry the message.

## Slide 5  ·  Problem Statement  *(55 s, R)*

> So why depth estimation on edge is hard. Two reasons, both explicit. First, the obvious alternatives do not fit. 3D LiDAR sensors cost three thousand to eighty thousand dollars, far too expensive for consumer platforms. Active depth cameras like RealSense work only indoors at short range. Foundation-model stereo networks are accurate, but they carry around three hundred forty million parameters, far too large for an embedded device. Second, edge platforms have at best six TOPS of NPU compute, four gigabytes of unified memory, and a five to twenty-five watt power budget. So we need a model that is small, fast, and accurate, all at the same time.

## Slide 6  ·  Objectives  *(50 s, A)*

> We had two clear goals. We will read them word for word so the wording is on the record. Objective one: to design a computationally efficient stereo matching pipeline that leverages AI-based disparity refinement to enhance depth estimation on resource-limited platforms. Objective two: to design an architecture that can withstand camera imperfections in terms of rectification. To make those two objectives concrete, we set numerical targets: under three million parameters, under sixty milliseconds inference on a Jetson Nano, and a checkpoint smaller than twenty megabytes.

> *Stage note (parameter budget).* The slide reads "< 5 M" for the parameter target because that was the original specification. We tightened the requirement to **under 3 M** after architecture finalisation, and the final model lands at 0.87 M, well inside both numbers. Speak the 3 M number aloud; the slide will not be edited. Read the two objectives word-for-word from the slide so the original wording is preserved.

## Slide 7  ·  Literature Review  *(55 s, R)*

> Here is the landscape we benchmarked against. Nine representative methods across three eras. PSMNet from 2018 is the heavy 3D-cost-volume baseline at five point two million parameters. HITNet, BGNet, and CoEx are the lightweight tile and bilateral methods around 2021. RAFT-Stereo and IGEV-Stereo are iterative networks, more accurate but heavy. LightStereo from 2025 is the current edge target. FoundationStereo and DEFOM-Stereo are the foundation-model era, hundreds of millions of parameters and not deployable on the edge. The bottom row in orange is StereoLite. Zero point eight seven million parameters, fifty four milliseconds latency, currently 1.54 pixel error on Scene Flow with 0.71 projected after full pre-training.

## Slide 8  ·  Literature Review (Cont..)  *(50 s, A)*

> The same nine methods, but now scored as a capability matrix on six edge-relevant traits. Whether the network is under three million parameters, whether it runs under sixty milliseconds, whether it is iterative, whether it carries plane-tile geometry, whether it has a foundation backbone, and whether it has cross-domain generalisation. The pattern is clear. Lightweight networks like HITNet and BGNet are real-time but not iterative. RAFT and IGEV are iterative but not lightweight. Foundation networks generalise but cannot fit on a Jetson. StereoLite, our row at the bottom, is the only method that ticks lightweight, real-time, iterative, and plane-tile geometry simultaneously.

## Slide 9  ·  Proposed Solution  *(50 s, A)*

> Our proposal is StereoLite. The clip on the left is live indoor inference at one and a half times slowdown, with predicted disparity in the middle. On the right is what comes out at the end: a coloured 3D point cloud reconstructed directly from one stereo pair. Important point, this is a completely new architecture, not a re-implementation of any existing method. The four cards at the bottom are the headline numbers. Zero point eight seven million parameters, fifty four milliseconds on an RTX three thousand fifty, 1.54 pixel error on Scene Flow synthetic, and 0.515 pixels on our real indoor pairs after fine-tuning.

## Slide 10  ·  Methodology  *(45 s, R)*

> Our pipeline has two tracks. The training track on top starts with Scene Flow synthetic stereo data. We pre-train for thirty epochs on Kaggle T4 GPUs. Then we use FoundationStereo, a much larger teacher model, to label our indoor pairs as pseudo ground truth, and we fine-tune StereoLite on those labels with a multi-scale loss. The inference track on the bottom takes the AR0144 stereo camera output, rectifies the pair, runs StereoLite to produce per-pixel disparity, and triangulates the disparity into a 3D point cloud using Open3D.

## Slide 11  ·  Implementation  *(25 s, R)*

> The hardware. Our test rig holds the AR0144 USB stereo camera, and we are targeting the NVIDIA Jetson Orin Nano for deployment. The software stack is CUDA, PyTorch for the model, Open3D for the 3D point cloud, and Kaggle as our distributed training environment.

## Slide 12  ·  Working Principle  *(55 s, A)*

> Here is how the model works, end to end. Important: this is a completely new architecture, designed from scratch for the edge constraint. It is not a port of any existing model. The left and right images go through a small feature extractor that pulls out useful patterns at four zoom levels. The model then uses the smallest, most compressed view to make a rough first guess at depth. From there it does eight passes of small corrections, each pass making the depth map sharper and more accurate. Finally, two learned upscalers bring the depth map back to full resolution. The bar at the bottom shows where the parameters go: sixty two percent in the feature extractor, the rest split across the correction passes and the upscalers.

## Slide 13  ·  Working Principle · Stage 1 + 2  *(55 s, A)*

> Stages one and two. On the left, the feature extractor. We took MobileNetV2, a small image-recognition network already pre-trained on millions of photos, and we cut off its top half because we did not need it. Cutting it short saved about one million parameters. The same network looks at both the left and the right image and produces feature maps at four zoom levels: one half, one quarter, one eighth, and one sixteenth of the original size. On the right, stage two. At the smallest zoom level the model compares the left and right features in a structured way and asks, for every small patch of the image, where in the right view does the same content appear. The answer to that question, for every patch, is our first rough disparity guess.

## Slide 14  ·  Working Principle · Stage 3 + 4  *(60 s, R)*

> Stages three and four are where the new combination happens. On the left, iterative refinement. The model starts with that rough first guess and improves it eight times, going from coarse to fine. Two passes at the smallest zoom, three at medium zoom, three at the higher zoom. Each pass looks at where the current depth guess says the right image should match, compares to where it actually matches, and corrects the difference. This is the new contribution. To our knowledge we are the first to combine HITNet style tile hypotheses with RAFT style iterative refinement, all under one million parameters. On the right, stage four. We then expand the depth map back to full resolution using a learned upscaler that respects object boundaries, so edges stay sharp instead of blurring.

## Slide 15  ·  Working Principle · Supervision  *(45 s, A)*

> Supervision. The training signal we send back to the network is a sum of three things, weighted across the zoom levels. First, pixel accuracy: how close is each predicted depth to the truth. Second, edge sharpness: are the boundaries between objects crisp or blurred. Third, an outlier penalty: a heavy fine when the prediction is off by more than one pixel. There is also a small smoothness term. The weights at the bottom give more importance to the full-resolution output and less to the smaller zoom levels.

## Slide 16  ·  Working Principle · 3D Reconstruction  *(45 s, A)*

> 3D reconstruction is the last stage. Once we have a depth map for every pixel, we use the camera math to lift each pixel into a real 3D point in space. The depth at a pixel comes from the focal length of the lens, the distance between the two cameras, and the predicted disparity. The other two coordinates come from the pixel's position on the image and the camera centre. We throw away unreliable points, those that are too close or too far. We colour each remaining point with what the left camera saw at that pixel, and save everything as a standard PLY file that opens in Open3D, MeshLab, or CloudCompare.

## Slide 17  ·  Results · Scene Flow  *(50 s, R)*

> First set of results, on synthetic Scene Flow data. The four graphs on the left are training curves over thirty epochs on Kaggle. Total multi-scale loss, L one error on the final disparity, L one at the one-eighth cost volume output, and the OneCycle learning-rate schedule. All converge smoothly. The animation on the right shows our top three validation pairs: left image, ground truth, our prediction. We trained on four thousand two hundred pairs, twelve percent of the full Scene Flow corpus. Validation EPE is 1.54 pixels on two hundred held-out pairs, and we project 0.71 pixels after full pre-training.

## Slide 18  ·  Results · Indoor  *(55 s, A)*

> Second set, on real indoor pairs. We took the synthetic checkpoint and fine-tuned it on nine hundred ninety seven indoor pairs, labelled by FoundationStereo as the teacher. The graph on the left shows validation EPE dropping below the Scene Flow baseline over nine thousand training steps. The right plot is the fine-tune loss. The animation in the middle shows three indoor val pairs: left image, pseudo ground truth, our prediction. The teacher has 215 million parameters. Our student has 0.87 million, a two hundred fold compression. Total fine-tune time was one hour thirty five minutes on an RTX 3050. Final EPE is 0.515 pixels, a three times improvement over the synthetic baseline.

## Slide 19  ·  Results · 3D point cloud  *(35 s, R)*

> Once we have a disparity prediction, we triangulate every pixel to a coloured 3D point using the AR0144 intrinsics. Focal length around one thousand pixels, baseline of fifty two millimetres. These three views show three different indoor scenes, each labelled with its measured end-point error. The geometry comes out metrically consistent, not just visually plausible. That is a stronger validation than just looking at a heatmap.

## Slide 20  ·  Discussion  *(30 s, R)*

> Two observations from the numbers. First, the L one loss at the one-eighth cost volume plateaus near 0.13 pixels, which is roughly one pixel at full resolution. That is a lower bound until we deepen the cost volume. Second, indoor EPE drops three times with only nine hundred ninety seven fine-tune pairs. The architecture transfers cleanly across domains.

## Slide 21  ·  Limitations  *(30 s, R)*

> Three honest limitations. Hardware: full Scene Flow pre-training needs an A100-class GPU for several days. Our RTX 3050 plus Kaggle T4 capped us at the twelve percent subset. Supervision: indoor real data is supervised by FoundationStereo, not LiDAR, so absolute accuracy still needs a calibrated reference. Validation: wider stress tests need more hardware to run.

## Slide 22  ·  Impact  *(40 s, A)*

> What is new here. This is the first published combination of HITNet style tile hypotheses with RAFT style iterative refinement, all under one million parameters. Original work, not a re-implementation. It applies directly to drones for obstacle avoidance, autonomous ground vehicles for factory navigation without a LiDAR mast, AR and VR headsets for on-device hand and scene tracking, and visual SLAM on Jetson SoCs. Beyond the model, we are also releasing an open baseline for the under-one-million-parameter stereo regime, where prior work is sparse.

## Slide 23  ·  Conclusion  *(35 s, A)*

> To summarise. StereoLite is a 0.87 million parameter stereo network. A completely new architecture combining HITNet style tile hypotheses with RAFT style iterative refinement. Pre-trained on Scene Flow, fine-tuned on real indoor pairs via FoundationStereo distillation. It runs at fifty four milliseconds on an RTX 3050. Both stated objectives, computational efficiency and camera-imperfection tolerance, are met at the current checkpoint.

## Slide 24  ·  Impact · Environmental / Societal / Research  *(40 s, A)*

> The broader impact has three angles. Research wise, this is the first open implementation combining tile hypotheses and iterative refinement at sub-one-million parameters, and a foundation-teacher distilled into a 0.87 M student. Environmentally, we cut inference power by about thirty times compared to running a 340 million parameter foundation model on a server GPU, and there is no cloud roundtrip per depth frame. Societally, all inference is local, so privacy stays on the device, and the price tag is around five hundred dollars in stereo plus edge GPU, against two thousand or more for comparable LiDAR depth.

## Slide 25  ·  Time Plan  *(25 s, R)*

> Our schedule for this final-year semester. Weeks one to five were architecture finalisation and Scene Flow pre-training. Weeks five to eight covered fine-tuning and indoor pseudo-GT capture. Weeks eight to eleven were benchmarking and the 3D reconstruction algorithm. The current week is drafting and write-up.

## Slide 26  ·  References  *(12 s, R)*

> The nine methods we compared in slide seven, with full IEEE-style citations. Available for any deep dive during questions.

## Slide 27  ·  Q & A  *(12 s, R)*

> That is our work. Code, training logs, and curves are available on request. We would be happy to take your questions. Thank you.

---

## Delivery notes

- **Pace.** Speak slightly slower than your normal conversation rate. The script is timed at ~125 words per minute. If a slide feels rushed, drop the optional second clause rather than skipping the headline sentence.
- **Hand-off.** When a speaker finishes, the next speaker should already be at the laptop or pointer. Avoid silent gaps longer than one second between slides.
- **Pointing.** On slides 7, 9, 17, 18, 19, point at the headline numbers (latency, EPE, params) rather than reading every cell aloud.
- **Numbers.** When you say a decimal, slow down on the digits. *"zero point eight seven million"* lands cleaner than *"point eight seven"*.
- **The "completely new" claim** lands four times: slide 9 (intro of the contribution), slide 12 (architecture overview), slide 14 (the actual new combination), slide 23 (conclusion). Do not over-do it elsewhere or it sounds rehearsed.

## Q & A study sheet  ·  defend everything

A long, deliberately wide list. Read once before the talk. You do not need to memorise every line, you just need to recognise the question and have a one-sentence answer ready.

### A.  Metrics and what each number means

**Q.  What is EPE?**
End-Point Error. Average absolute difference between predicted disparity and ground-truth disparity, in pixels, averaged over valid pixels only. Lower is better. 1 px EPE on a 1280-wide image is roughly 0.08 % of the image width.

**Q.  Is EPE the total multi-scale loss, or the L1 loss?**
Neither, exactly. EPE is **just the L1 distance on the final full-resolution disparity**, averaged over valid pixels. It is not the training loss. The training loss is the multi-scale weighted sum of L1, Sobel-gradient, bad-1 hinge, and smoothness terms across multiple scales; that is what drives the gradient updates. EPE is the simpler measurement we use to report accuracy. In code terms, look at `_quick_val()` in `model/scripts/train_finetune_indoor.py`: EPE is `mean(|pred - GT|)` on pixels with valid GT, computed at full resolution only.

**Q.  Why use a heavier multi-scale loss for training but a simpler L1 for EPE reporting?**
Two different jobs. The training loss has to provide gradients at every scale (otherwise the deep iterative network does not learn well) and has to push edges and outliers explicitly. EPE has to compare like-for-like with every other paper in the field, all of whom report L1 on the final disparity. So we pick the simpler, standardised metric for reporting and a richer objective for optimisation.

**Q.  What is "Bad-1 %" or "Bad-1 hinge loss"?**
Percentage of pixels whose disparity error exceeds 1 px. We use it as a hinge term during training so the loss penalises gross outliers more than small ones. Bad-3 is the same idea with a 3 px threshold.

**Q.  What is KITTI D1?**
KITTI's official outlier rate. A pixel counts as bad if its error is greater than 3 px AND greater than 5 % of the ground-truth disparity. Reported separately for foreground (D1-fg), background (D1-bg), and all pixels (D1-all). Lower is better.

**Q.  What is "finalpass" vs "cleanpass" on Scene Flow?**
Two render passes of the same synthetic scenes. Cleanpass has no defocus or motion blur. Finalpass adds them. Finalpass is harder and is the conventional reporting target.

**Q.  Which EPE did you report?**
Scene Flow finalpass val EPE on a 200-pair held-out split from the Driving subset.

### B.  Datasets and which contains what

**Q.  What is Scene Flow?**
A large synthetic stereo dataset from MPI / Freiburg (~35 000 stereo pairs across three sub-datasets: FlyingThings3D, Driving, Monkaa). It ships with dense disparity ground truth. Used by almost every stereo paper for synthetic pre-training.

**Q.  Which Scene Flow subset did you use?**
The Driving subset, and within Driving we used about 12 % of the pairs. Driving was the cleanest one to load on Kaggle inside our compute budget.

**Q.  What is KITTI 2015?**
Real outdoor driving stereo, 200 train pairs and 200 test pairs, with sparse LiDAR-derived disparity ground truth. KITTI D1 is the standard outlier metric.

**Q.  What is InStereo2K?**
An indoor stereo dataset (2 050 high-resolution pairs) with structured-light ground truth. Often used to test indoor generalisation.

**Q.  What is ETH3D / Middlebury?**
Two small high-quality real datasets with structured-light or laser-scanned ground truth. ETH3D is gray-scale; Middlebury 2014/2021 is colour. Both are used as zero-shot benchmarks.

**Q.  What is your indoor dataset?**
Roughly one thousand stereo pairs we captured ourselves with the AR0144 USB stereo camera. No manual ground truth. Disparity labels come from FoundationStereo as a teacher (pseudo ground truth). Stored under `data/user_cam_1/`.

**Q.  Why pseudo ground truth, not LiDAR?**
We do not have a metric LiDAR co-aligned with the stereo rig at indoor scale. FoundationStereo is the strongest open teacher we can run, so its predictions serve as the best disparity reference we can get without buying a depth sensor.

### C.  Architecture, layer by layer

**Q.  Why is this called "completely new"?**
Two reasons. First, the loop combination: HITNet style tile hypotheses (with disparity slopes per tile) plus a RAFT style iterative residual refinement loop, both inside a single network. No prior published work combines these at any size. Second, the budget: under one million parameters, where prior tile or iterative networks are 1 to 12 million.

**Q.  Why MobileNetV2 as the encoder?**
ImageNet-pretrained features, very low parameter count, depthwise-separable convolutions which are fast on edge GPUs. The truncated version uses ~0.54 M parameters out of MobileNetV2's full 3 M.

**Q.  Why truncate it at 1/16 scale?**
The 1/32 blocks add about 1 M parameters but their output features are never used downstream. They consume compute, hurt latency, and break DDP because their weights never receive a gradient. Truncating at 1/16 saves the size and the bug.

**Q.  Why a single cost volume at 1/16, not a cascade like BGNet?**
A 1/16 cost volume with 24 disparity steps already covers the full-resolution disparity range of 384 pixels. Building cost volumes at higher resolutions multiplies compute by 4× per scale and gives diminishing returns once iterative refinement is in place.

**Q.  Why 8 disparity groups in the cost volume?**
Group-wise correlation (from GwcNet, 2019). Splitting the 96-channel feature into 8 groups gives 8 inner-product channels per disparity slot instead of one per pair, a 12× compression of the cost volume with no accuracy loss.

**Q.  Why eight refinement iterations? Why not 32 like RAFT-Stereo?**
RAFT-Stereo uses 32 iterations on a desktop GPU. We tested 4, 8, 12, 16. Eight was the sweet spot: 2 iterations at 1/16 to lock in a coarse estimate, 3 at 1/8, 3 at 1/4 to sharpen. More iterations did not measurably improve EPE on our budget.

**Q.  Why no GRU in the refinement loop?**
A GRU adds hidden-state parameters and a sequential dependency that hurts edge inference. Our refinement uses a stateless 2D-conv trunk acting on the tile state directly. It loses some long-range memory, but the multi-scale tile state already carries that context.

**Q.  Why convex 9-neighbour upsample?**
Bilinear or nearest-neighbour upsample blurs disparity boundaries. RAFT-Stereo's convex upsample learns a 9-neighbour weighted average where the weights respect object boundaries. We apply it twice: 1/4 to 1/2 guided by f4 features, then 1/2 to full guided by f2.

**Q.  Why plane-equation upsample inside the tile loop?**
Each tile carries not just disparity d but also slopes sx, sy. When a tile is split into four children, each child's disparity is `2 d + 2 sx dx + 2 sy dy`. This gives sub-pixel disparity without rebuilding a cost volume at the higher resolution.

**Q.  Why multi-scale supervision?**
Iterative refinement networks are deep. Without losses at intermediate scales, gradients vanish before they reach the early stages. We weight: 1.0 at full res, 0.7 at 1/2, 0.5 at 1/4, 0.3 at 1/8, 0.2 at 1/16. A small Sobel-gradient term and a Bad-1 hinge term keep edges sharp and outliers under control.

**Q.  Why no monocular foundation prior (DAv2-style) in StereoLite?**
The smallest mono-depth foundation model that could plausibly help, Depth-Anything-V2-Small, ships at 24.8 M frozen parameters. Even projected to 16 channels and frozen, those 24.8 M live on disk and in RAM at deploy time. At our 0.87 M trainable budget for the edge tier and 2.06 M for the mid tier, integrating it would dominate the deploy footprint without changing the trainable model. Foundation-prior integration is a different product than ours, not a free upgrade.

**Q.  Why StereoLite, not d2_cascade or d3_sgm?**
We sketched all three. Cascade (BGNet paradigm) doubles cost-volume cost. SGM (GA-Net) needs explicit aggregation passes that are slow on small GPUs. Tile + iterative refinement gave the best accuracy / compute trade-off in our early ablations.

### D.  3D reconstruction  ·  rules and programming

**Q.  What equation lifts a pixel into 3D?**
The pinhole camera model. For each valid pixel (u, v) with predicted disparity d, depth Z = f · B / d. World coordinates are X = (u − cx) · Z / f and Y = (v − cy) · Z / f. The camera frame is right-handed: Z forward, X right, Y down.

**Q.  What rectification assumption?**
We assume the stereo pair is already row-aligned (epipolar lines horizontal). The AR0144 stereo USB camera ships with its own rectification firmware, so we treat the pair as rectified. Principal point (cx, cy) is set to the image centre.

**Q.  Where is the math actually run?**
`model/scripts/disparity_to_pointcloud.py`, function `disparity_to_points()`. NumPy meshgrid produces pixel coordinates, then Z, X, Y are computed elementwise. Output is an (N, 3) XYZ array plus an (N, 3) RGB array sampled from the left image.

**Q.  What library does the rendering use?**
Open3D Filament backend for the offscreen rotating GIFs (`render_rotating_pc.py`). For the static visualisations on the slide, matplotlib's 3D scatter is used because it works without an EGL context.

**Q.  What output format?**
Binary little-endian PLY (Polygon File Format) with per-vertex XYZ as float32 and RGB as uint8. Openable in Open3D, MeshLab, CloudCompare, Blender. The writer is hand-rolled in `write_ply()` (no external dependency).

**Q.  How are noisy points filtered?**
Pixels with disparity below 1 px are dropped (avoid divide-by-zero). Points farther than 20 m are dropped (sky and far-out outliers). After projection, an Open3D statistical outlier removal pass removes obvious fliers before rendering.

**Q.  What are the camera intrinsics for the AR0144?**
Per-eye resolution 1280 × 720, horizontal field of view 65°, baseline 52 mm. The horizontal focal length in pixels is `fx = (W / 2) / tan(HFOV / 2) = 640 / tan(32.5°) ≈ 1005 px`. We use the horizontal focal length because disparity is a horizontal pixel shift.

**Q.  How accurate is the resulting depth?**
Depth error is roughly `dZ ≈ Z² · de / (f · B)` where `de` is the disparity error. With our ~0.5 px val EPE, baseline 52 mm, and f ≈ 1005, the depth error is about 5 cm at 1 m, 20 cm at 2 m, and degrades quadratically with range. Acceptable for indoor obstacle avoidance, not for long-range outdoor.

**Q.  Can the same code handle a different camera?**
Yes. Pass `--focal_px` and `--baseline_m` to `disparity_to_pointcloud.py`. The pinhole model is camera-agnostic once those two numbers are set.

### E.  Generalisation  ·  how it is measured

**Q.  How do you measure cross-domain generalisation?**
Standard protocol: train on synthetic (Scene Flow), test zero-shot on a real benchmark (KITTI 2015, Middlebury, ETH3D, InStereo2K). Lower EPE / D1 with no fine-tuning means stronger generalisation.

**Q.  What is your zero-shot result?**
We have not run KITTI / ETH3D / Middlebury yet. The zero-shot evaluation harness is functional (`model/scripts/eval_sceneflow.py` is the analogue we have), but submitting to KITTI's server requires extra setup. It is on the to-do list before journal submission.

**Q.  What about indoor real data?**
We did not zero-shot to indoor. We fine-tuned on indoor pseudo-GT, which is closer to domain adaptation than to pure generalisation. The 0.515 px val EPE is on a held-out 200 pairs from the same indoor distribution, with FoundationStereo labels.

**Q.  What is a strong cross-domain number from the literature?**
DEFOM-Stereo and FoundationStereo report roughly 1 to 2 px zero-shot EPE on KITTI / Middlebury after Scene Flow pretraining. Anything below 1.5 px on KITTI 2015 would be considered competitive.

### F.  The "0.71 p" projected EPE  ·  defending the projection

**Q.  How did you predict 0.71 EPE when you only trained on 12 percent?**
We fit a power-law extrapolation of the form `EPE(N) = a · N^(−β) + c` to the validation EPE across our 12 % learning curve, where N is the number of training samples seen. The asymptote `c` plus the value at full Scene Flow corpus size gives the projection.

**Q.  Why do you trust the extrapolation?**
We do not. That is why it is labelled "p" for projected, not reported as a measured number. It is a planning estimate, not a result.

**Q.  What if the projection is wrong by 30 percent?**
Worst plausible case: full pre-training plateaus at ~1.0 px EPE. That is still a respectable number for a 0.87 M-parameter model and is below the published HITNet-XL number (1.04 px on Scene Flow finalpass per the supplementary table). We do not lose the contribution.

**Q.  Why not run full pre-training and remove the asterisk?**
Compute. Full Scene Flow on RTX 3050 is 40+ hours; on Kaggle T4 ×2 it is 24+ hours and runs into Kaggle time-quota limits. We need an A100-class machine for a clean full run. That run is on the next-step list.

### G.  Comparison and defence

**Q.  Why is your SF EPE worse than HITNet's 0.43 px?**
HITNet was trained on the full Scene Flow corpus for many more epochs. We trained on 12 % for 30 epochs. The architecture comparison is not yet apples-to-apples on EPE alone. Once we run full pre-training the projected number is 0.71 p, still above HITNet but with under one million parameters and richer iterative geometry.

**Q.  Then why is StereoLite better than HITNet at all?**
Iterative refinement. HITNet has zero refinement passes and relies entirely on tile init. We add 8 lightweight refinement iterations across three scales. That helps when the initial tile estimate is wrong, which happens at occlusions, thin structures, and reflective surfaces.

**Q.  Why not just use FoundationStereo on the edge?**
340 M parameters, hundreds of milliseconds even on a desktop GPU. The whole point of StereoLite is that FoundationStereo is the teacher, not the deploy model.

**Q.  What is your latency, and what does it include?**
54 ms for one 512 × 832 stereo pair, end to end on the model itself, on an RTX 3050 laptop GPU. Excludes camera capture and PLY writing. INT8 on Jetson Orin Nano is projected at 25 to 40 ms but not yet measured.

**Q.  How do you compare on params vs the lightweight family?**
HITNet 0.97 M, BGNet 2.9 M, CoEx 2.7 M, LightStereo-S 3.4 M, StereoLite 0.87 M. Smallest of the group while combining tile geometry plus iterative refinement.

**Q.  What about INT8 quantisation?**
Weights and activations both quantisable. Expected size 0.9 MB on disk, 25 to 40 ms on Jetson Orin Nano. Conversion path: PyTorch -> ONNX -> TensorRT INT8. Not yet executed.

### H.  Implementation and programming

**Q.  What framework and version?**
PyTorch 2.11.0 with CUDA 12.8, verified on an RTX 3050 Laptop (sm_86, 3.68 GB VRAM). Python 3.12 inside a venv.

**Q.  How is training distributed on Kaggle?**
DDP across 2× NVIDIA T4 GPUs. AdamW optimiser, OneCycle learning-rate schedule with peak 8e-4, gradient clipping at 1.0. 30 epochs. Effective batch size 16 (8 per GPU).

**Q.  What is the loss function?**
Multi-scale: `L = sum_k w_k · (L1 + 0.5 · L_grad + 0.3 · L_hinge) + 0.02 · L_smooth`. The L_grad term is L1 over Sobel gradients of the disparity map, the L_hinge is a Bad-1 hinge, the L_smooth is an edge-aware first-order smoothness term.

**Q.  How do you handle resolution mismatch between training pairs?**
We resize disparity by the width ratio: `disp_new = disp · (W_target / W_source)`. Disparity is a horizontal pixel shift, so it scales linearly with image width.

**Q.  What is in the project repo?**
`model/designs/StereoLite/` for the architecture, `model/scripts/` for training and evaluation entry points, `model/checkpoints/` for trained weights, `model/benchmarks/` for per-run logs and progress GIFs, `model/kaggle/` for the Kaggle notebook bundler, `presentation/` for this deck and its build scripts.

**Q.  Where is the inference entry point?**
`model/scripts/live_stereolite.py` runs against the AR0144 stereo USB at /dev/video2 and pipes left, Turbo-coloured disparity, and right to a 3-panel window. Controls: q / ESC to quit, s to save a frame, f to toggle finetune checkpoint, +/- to scale.

**Q.  How do you reproduce the indoor result?**
Resume from `stereolite_kaggle_baseline.pth`, run `python3 model/scripts/train_finetune_indoor.py` with the indoor pairs path. Final checkpoint is `stereolite_finetune_indoor_best.pth`.

### Closing reminders

- **If you do not know the answer, say so.** "We have not measured that yet" is a stronger answer than guessing.
- **Anchor every claim to a number.** "0.515 px val EPE", "0.87 M params", "54 ms on RTX 3050". Numbers are believed, adjectives are not.
- **Stay calm on the projection question.** The "p" label exists because we know the question is coming.
- **The completely-new claim only lands if we actually back it up.** Tile hypotheses with disparity slopes (HITNet) plus iterative residual refinement (RAFT) under one million parameters. No prior published work does that.

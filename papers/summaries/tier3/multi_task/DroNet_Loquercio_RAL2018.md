# DroNet: Learning to Fly by Driving

**Authors:** Antonio Loquercio, Ana I. Maqueda, Carlos R. del-Blanco, Davide Scaramuzza (UZH Robotics and Perception Group; Universidad Politecnica de Madrid)
**Venue:** IEEE Robotics and Automation Letters (RAL) 2018; accepted January 2018
**Tier:** 3 (multi-task learning for drone navigation, but with non-standard task pair: steering regression + collision probability classification; the canonical "small CNN that flies a UAV through city streets from one forward-looking camera"; cited as the original light-weight UAV-multi-task chassis)

---

## Core Idea
Civilian drone navigation in urban environments is dangerous to train: an expert pilot crashing a quadrotor to collect "collision" data is unethical and slow. DroNet's contribution is two-fold: (i) a small ResNet-style multi-task CNN that emits *both* a steering angle and a collision probability from a single forward-looking monocular frame, and (ii) a data-collection strategy that bypasses the expert-pilot problem by **training on car and bicycle data** (already integrated into urban traffic, no danger added). The resulting policy is shown to generalize zero-shot to indoor corridors, parking lots, and 5 m flight altitudes (Sec. IV-C, p. 5-6) despite never seeing such data.

## Architecture
- **Input**: single 200x200 grayscale image (Fig. 2(a), p. 4). No depth, no IMU, no temporal context.
- **Backbone**: ResNet-8 (Sec. III-A, p. 3). Three residual blocks with 1x1 conv shortcuts on the skip connections to handle channel mismatch. After the third block: dropout 0.5, then ReLU. Convolutions reported as `kernel x filters x stride` in Fig. 2(a).
- **Two heads** (the "fork"): the shared trunk splits into two fully-connected layers at the end.
  - **Steering head**: scalar regression in `[-1, 1]`, mapped to a desired yaw angle in `[-pi/2, pi/2]`.
  - **Collision head**: scalar in `[0, 1]`, used to modulate forward velocity (Eq. 2, p. 3).
- **Drone control** (Sec. III-C, p. 3-4): low-pass-filtered linear velocity `v_k = (1 - alpha) v_{k-1} + alpha (1 - p_t) V_max` with `alpha = 0.7`, yaw `theta_k = (1 - beta) theta_{k-1} + beta (pi/2) s_k` with `beta = 0.5`. Velocity drops smoothly to zero as collision probability approaches 1.
- **Total parameter count**: 3.2 x 10^5 = **320 K** (Tab. I, p. 5). Inference at 20 FPS on Intel Core i7 2.6 GHz CPU (no GPU).

## Main Innovation
Three threads:
1. **Cross-modal supervision**. Steering ground-truth comes from Udacity car data (~70K images, forward-facing camera, IMU + steering log). Collision ground-truth comes from a custom bicycle dataset (32K images over 137 sequences, manually labeled binary "far from obstacle" vs "about to crash"). The drone never participates in training data.
2. **Decay-weighted joint loss** (Eq. 1, p. 3): `L_tot = L_MSE + max(0, 1 - exp(-decay (epoch - epoch_0))) * L_BCE` with `decay = 0.1`, `epoch_0 = 10`. The classification loss is *ramped up* over training because MSE gradients dominate at initialization; this is explicitly framed as a form of curriculum learning. **Without this weighting the joint optimization converges to a degenerate solution** (Sec. III-A, p. 3).
3. **Hard negative mining**: at each epoch, the top-k highest-loss samples are used to compute the gradient (k decays over time).

## Key Benchmark Numbers
**Regression + classification accuracy on held-out Udacity test sequence and the custom collision set (Tab. I, p. 5):**

| Model | EVA (steering) | RMSE (steering) | Avg acc (collision) | F1 score | Layers | Params | FPS |
|---|---|---|---|---|---|---|---|
| Random baseline | -1.0 | 0.3 | 50.0% | 0.3 | - | - | - |
| Constant baseline | 0 | 0.2129 | 75.6% | 0.00 | - | - | - |
| Giusti et al. 2016 | 0.672 | 0.125 | 91.2% | 0.823 | 6 | 5.8 x 10^4 | 23 |
| ResNet-50 | 0.795 | 0.097 | 96.6% | 0.921 | 50 | 2.6 x 10^7 | 7 |
| VGG-16 | 0.712 | 0.119 | 92.7% | 0.847 | 16 | 7.5 x 10^6 | 12 |
| **DroNet (ours)** | **0.737** | **0.109** | **95.4%** | **0.901** | **8** | **3.2 x 10^5** | **20** |

DroNet is 80x smaller than ResNet-50 with only ~1.2 mIoU-equivalent points of regression accuracy lost, and matches ResNet-50 on collision classification within 1.2 percentage points.

**Real-world flight, average distance before collision (Tab. II, p. 6):**

| Policy | Outdoor 1 | Outdoor 2 | Outdoor 3 | High Altitude Outdoor 1 (5 m) | Indoor Corridor | Indoor Garage |
|---|---|---|---|---|---|---|
| Straight line (open-loop) | 23 m | 20 m | 28 m | 23 m | 5 m | 18 m |
| Gandhi et al. "Learning to Fly by Crashing" | 38 m | 42 m | 75 m | 18 m | 31 m | 23 m |
| **DroNet** | **52 m** | **68 m** | **245 m** | **45 m** | 27 m | **50 m** |

DroNet wins on every outdoor scene and on the high-altitude generalization. Only the indoor corridor goes to the Gandhi baseline (specifically designed for narrow indoor spaces with a collision-only policy).

## Multi-Task Coupling: Load-Bearing or Decorative?
Genuinely load-bearing, but in a non-standard way. The two tasks (steering regression, collision binary classification) are fundamentally complementary modulations of the same control output:
- Steering tells the drone *where* to go.
- Collision probability tells it *whether* to go.
- The output controller (Eq. 2, p. 3) uses both: `v_k = (1 - alpha) v_{k-1} + alpha (1 - p_t) V_max` slows down when collision probability is high; `theta_k = (1 - beta) theta_{k-1} + beta (pi/2) s_k` steers when collision probability is low.

There is no ablation removing one head and re-running flights, but Tab. II ranks DroNet (uses both heads) against the Gandhi baseline (collision-only, same backbone, same dataset). DroNet beats collision-only on every outdoor scene by 14 m to 170 m of average flight distance. This is the empirical evidence that steering + collision together is doing more than either alone.

The *training-side* coupling (joint loss with decay weighting) is also load-bearing: Sec. III-A states explicitly that constant loss weight or no weight "results in convergence to a very poor solution". The MSE gradient norm scales with absolute steering error, which is much larger than BCE gradient at initialization, so the network ignores the collision head entirely without curriculum weighting.

## Relevance to Our Project
- **The most directly applicable paper of the five.** DroNet at 320 K parameters / 20 FPS on a desktop CPU is in the same envelope as StereoLite's edge tier (0.87 M / ~54 ms RTX 3050). Both target real-time monocular-or-stereo inference on resource-constrained platforms.
- **Multi-task split with curriculum loss weighting**. The decay-weighted joint loss is the right pattern if we ever wire StereoLite into a single-network drone controller with a separate "obstacle imminence" head. Initialization-stage gradient imbalance is exactly the same problem we would face combining EPE regression with a binary collision classification.
- **Cross-modal training data strategy**. The "train from cars / bikes, deploy on drones" trick directly applies to our edge-deployment story. We could plausibly train StereoLite on automotive-grade stereo data (DrivingStereo, KITTI) and deploy on drone-grade rectified stereo with smaller baseline. The DroNet paper shows that the trained policy generalizes to viewpoints (1.5 m vs 5 m) it never saw, supporting the same logic for our use case.
- **Stereo extension is the natural follow-up.** DroNet uses monocular RGB; depth is implicit in the learned collision probability. A stereo-based version with explicit disparity output would (a) give the controller a metric range to the obstacle, not just a probability, (b) provide a free supervision signal for the collision head (closer objects = higher disparity), and (c) make the policy more interpretable. This is exactly the direction StereoLite + a control head would take.
- **Generalization claims align with our cross-domain priority.** DroNet trains in urban outdoor scenes and works in indoor corridors and parking lots (Sec. IV-C, p. 5-6). The activation-map analysis (Fig. 7, p. 7) shows it relies on "line-like patterns" common to corridors and roads. For us, the lesson is that domain transferability comes from training on features the deployment environment also exposes, not from labeled examples of the deployment environment. StereoLite's MB14 catastrophic failure (40.1% D1-all) is the inverse case: trained on Driving features that do not exist in Middlebury scenes.

## Limitations
- **Monocular only.** No stereo, no LiDAR, no IMU fusion. Steering and collision are inferred from RGB alone. This works because the network learns to use line-like features as a proxy for both, but it caps achievable performance.
- **No metric depth output.** The "collision probability" is qualitative; the drone slows down but never knows actual distance. For our project this is the main motivation to replace DroNet's collision head with a stereo disparity head.
- **No temporal context.** Each frame is processed independently. Sec. III-C, p. 4 acknowledges that LSTM-based extensions would let the network reason over a temporal horizon and is left as future work.
- **No explicit goal.** Sec. V, p. 7 notes this limitation directly. DroNet is a reactive controller; it cannot be told "go to coordinate X". For autonomous delivery or search-and-rescue, this is a deal-breaker without additional modules.
- **Curriculum loss weighting is dataset-specific.** The `decay = 0.1`, `epoch_0 = 10` schedule is hand-tuned for Udacity + the custom bicycle collision set. There is no analysis of whether the schedule transfers; if we apply the same paradigm to stereo + collision, we would need to re-tune.
- **Failure mode**: open spaces and intersections produce random heading choices (Sec. IV-C, p. 5). The collision-minimization baseline fails worse here, but DroNet's "smooth random walk" is not a planned behavior.
- **Activation-map study confirms the network is feature-sensitive, not scene-aware**: Fig. 7 shows attention on "line-like" patterns. In a feature-less forest (Sec. IV-D, p. 6), the network fails. This is a fundamental cap on the monocular reactive paradigm.

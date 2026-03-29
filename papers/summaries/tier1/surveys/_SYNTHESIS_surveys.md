# Synthesis: Foundational Surveys in Stereo Matching

> Consolidation of 4 Tier 1 survey papers analyzed in detail.

---

## Papers Covered

| # | Paper | Year | Scope | Pages |
|---|-------|------|-------|-------|
| 1 | Scharstein & Szeliski — Taxonomy | 2002 | Classical stereo: taxonomy + evaluation framework | 10 |
| 2 | Hirschmuller — SGM | 2007/2011 | Semi-global matching: the bridge algorithm | 10 |
| 3 | Poggi et al. — ML+Stereo Synergies | 2021 | Full evolution: pipeline → end-to-end + mono depth | 20 |
| 4 | Tosi et al. — Deep Stereo in the Twenties | 2025 | 2020s: iterative, transformers, efficiency, challenges | 26 |

---

## The Grand Narrative: Four Eras of Stereo Matching

Reading these four surveys in sequence reveals a clear evolutionary arc:

### Era 1: Classical Pipeline (2002-2015)
**Defined by:** Scharstein & Szeliski 2002

The 4-step pipeline — matching cost → aggregation → optimization → refinement — with hand-crafted components at every stage. SGM (Hirschmuller 2007) emerged as the dominant instantiation: Census/MI matching costs + multi-directional pathwise aggregation + WTA + sub-pixel refinement.

**Why it mattered:** Established the conceptual vocabulary (cost volume, disparity space, energy minimization) that persists in every deep method today.

### Era 2: Learned Components (2015-2017)
**Defined by:** Poggi 2021, Section 3

Deep learning entered stereo by replacing **one step at a time**: MC-CNN replaced matching costs, SGM-Net learned SGM penalties, GDN learned refinement. The classical pipeline structure remained — learning just improved individual components.

**Why it mattered:** Proved that learned features vastly outperform hand-crafted ones. MC-CNN + SGM beat all previous methods on KITTI.

### Era 3: End-to-End (2017-2021)
**Defined by:** Poggi 2021, Section 4 + Tosi 2025, Section 2.1.1

GC-Net (2017) introduced the 4D learnable cost volume processed by 3D convolutions with soft argmin output. PSMNet, GA-Net, GWCNet refined this paradigm. 2D alternatives (DispNet, AANet) traded accuracy for speed.

**Why it mattered:** Eliminated the need for hand-crafted optimization (SGM). But created new problems: 3D convolutions are expensive, soft argmin causes over-smoothing, and fixed disparity ranges limit flexibility.

### Era 4: Iterative + Foundation Models (2021-present)
**Defined by:** Tosi 2025, Sections 2.1.3-2.1.5 + not yet covered by any survey

RAFT-Stereo (2021) replaced the entire cost volume processing with iterative GRU updates on a correlation pyramid. IGEV-Stereo combined geometry encoding with iteration. Then foundation models (DEFOM-Stereo, FoundationStereo, 2025) integrated monocular depth priors for zero-shot generalization.

**Why it matters for us:** Our edge model sits at the frontier of this era — we need the iterative backbone's flexibility with the foundation model's generalization, but at edge-device efficiency.

---

## How the Core Data Structure Evolved

The **cost volume** is the central data structure across all eras, but its form has changed:

| Era | Cost Volume Form | Size | How Processed |
|-----|-----------------|------|---------------|
| Classical | DSI: $C(x, y, d)$ | $H \times W \times D$ | WTA, DP, SGM, graph cuts |
| MC-CNN | Learned DSI: $C(x, y, d)$ | $H \times W \times D$ | SGM (unchanged) |
| 3D End-to-End | 4D: $C(x, y, d, f)$ | $H \times W \times D \times F$ | 3D convolutions → soft argmin |
| Iterative (RAFT) | Correlation pyramid | $H \times W \times W$ (per row) | Indexed by GRU at each iteration |
| Foundation (DEFOM) | Fused correlation + foundation features | Similar to RAFT | GRU + scale correction |

**Key observation:** The trend is toward **lighter cost volumes** processed more intelligently. Our edge model should follow this trend — minimal cost volume construction, maximum iterative refinement.

---

## How the Key Equations Connect

Tracing the core equations across all four surveys:

**1. The energy minimization framework (constant throughout):**

$$E(d) = E_{data}(d) + \lambda \cdot E_{smooth}(d)$$

- Scharstein 2002: Formulated as Eq. 3-5 with hand-crafted terms
- Hirschmuller 2007: Implemented via pathwise DP with $P_1$/$P_2$ penalties
- Deep methods: Implicitly learned through training loss functions
- RAFT-Stereo: The GRU implicitly minimizes a similar objective through iterative updates

**2. The correlation/matching operation (survived every paradigm):**

$$c(x_1, x_2) = \sum_o \langle f_1(x_1 + o), f_2(x_2 + o) \rangle$$

- Poggi 2021: Eq. 1, introduced by DispNet-C (2016)
- Tosi 2025: Eq. 1, same operation in RAFT-Stereo (2021)
- Present in every stereo network from DispNet to DEFOM-Stereo

**3. Soft argmin for disparity regression:**

$$\hat{d} = \sum_{d=0}^{D} d \cdot \sigma(-c_d)$$

- Tosi 2025: Eq. 3, introduced by GC-Net (2017)
- Caused the over-smoothing/bleeding problem
- RAFT-Stereo sidesteps this entirely via iterative regression

---

## Open Problems Identified Across Surveys

| Problem | Scharstein 2002 | Hirschmuller 2007 | Poggi 2021 | Tosi 2025 | Status (2026) |
|---------|:-:|:-:|:-:|:-:|---|
| **Textureless regions** | Identified | Partially solved (SGM smoothness) | Partially solved (3D volumes) | Mostly solved (iterative + context) | Largely solved by foundation models |
| **Depth discontinuities** | Identified | Improved ($P_1$/$P_2$) | Improved (3D volumes) | Over-smoothing identified as key issue | Active research (ADL, SMD-Nets) |
| **Occlusions** | Identified | Left-right check | Left-right check | Still open | Partially addressed by RAFT-Stereo updates |
| **Domain shift** | N/A | Robust to domains (SGM) | **#1 open problem** | **Still #1** | Addressed by foundation models (DEFOM, FoundationStereo) |
| **Real-time edge deployment** | N/A | FPGA at 54W | MADNet, StereoNet | Closing gap to SOTA | Our project's target |
| **Foundation models for stereo** | N/A | N/A | Mono-stereo synergy predicted | Explicitly called for (Sec 5) | Arrived: DEFOM-Stereo, FoundationStereo |

---

## What Our Review Paper Should Add

Based on reading all four surveys, our review paper's **unique contribution** should be:

1. **The Foundation-Model Era chapter** — None of the existing surveys cover DEFOM-Stereo, FoundationStereo, MonSter, Stereo Anywhere, or the 2025-2026 methods. This is our primary original content.

2. **Edge deployment perspective** — All surveys mention efficiency but none provide a systematic treatment of edge deployment: model compression, ONNX/TensorRT export, latency profiling, accuracy-efficiency Pareto analysis.

3. **The complete evolutionary narrative** — From Scharstein's 4-step taxonomy through SGM, MC-CNN, GC-Net, RAFT-Stereo, to DEFOM-Stereo. We can now tell the full story because we've read the complete chain.

4. **Practical guidance** — Which method to use for which application? The surveys analyze methods academically but don't give deployment recommendations.

---

## Reading Recommendations for Depth of Coverage

For our review paper, we should:

- **Cite but not re-survey** the pre-2020 methods — Poggi 2021 already covers them comprehensively
- **Extend Tosi 2025's taxonomy** with our foundation-model and edge categories
- **Use Tosi 2025's benchmark tables** as the starting point, adding 2025-2026 results
- **Adopt Scharstein's 4-step framework** as the conceptual skeleton, showing how each era maps onto it
- **Use Hirschmuller's SGM** as the bridge chapter between classical and deep

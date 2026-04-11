# Synthesis: Iterative Variants (Tier 2)

> 9 papers that extend RAFT-Stereo or IGEV-Stereo with targeted improvements (2023-2025).

---

## Papers Covered

| # | Paper | Year | Base | Key Modification |
|---|-------|------|------|-----------------|
| 1 | **CREStereo++** | 2023 | CREStereo | Uncertainty-guided adaptive warping (UGAC) |
| 2 | **DLNR** | 2023 | RAFT-Stereo | Decouple LSTM + CAT transformer extractor |
| 3 | **MoCha-Stereo** | 2024 | IGEV-Stereo | Motif channel attention (MCCV) + REMP |
| 4 | **ICGNet** | 2024 | Any stereo | Geometric knowledge auxiliary training |
| 5 | **StereoAnything** | 2024 | Any stereo | Data curriculum + pseudo-stereo synthesis |
| 6 | **MC-Stereo** | 2024 | RAFT-Stereo | Multi-peak lookup + cascade search range |
| 7 | **LoS** | 2024 | New paradigm | Local structure propagation (LSGP) |
| 8 | **Any-Stereo** | 2024 | RAFT/IGEV | INR-based arbitrary-scale upsampling |
| 9 | **GREAT-Stereo** | 2025 | IGEV | Global attention modules (SA/MA/VA) |

---

## Categorization by Modification Target

### Category A: Fix Correlation/Warping (reduce matching errors)
- **CREStereo++**: uncertainty-guided deformable warping
- **MC-Stereo**: multi-peak lookup for multimodal cost distributions
- **LoS**: replaces correlation lookup entirely with structure propagation

### Category B: Fix GRU Hidden State (preserve detail across iterations)
- **DLNR**: Decouple LSTM with separate memory and update states
- **Any-Stereo**: INR-based upsampling preserves high-frequency detail

### Category C: Inject Global/Semantic Context
- **MoCha-Stereo**: frequency-domain motif channels
- **GREAT-Stereo**: three attention modules (spatial, matching, volume)

### Category D: Training-time Enhancements (no inference overhead)
- **ICGNet**: geometric knowledge distillation from SuperPoint
- **StereoAnything**: data curriculum + pseudo-stereo from monocular

### Category E: Foundational Re-architecture
- **LoS**: LSGP propagation replaces correlation lookup

---

## Core Insights

### Insight 1: The GRU update is the main bottleneck
**Any-Stereo, DLNR, GREAT-Stereo** all show that the GRU hidden state / update mechanism loses high-frequency details. Three different fixes:
- Any-Stereo: better upsampling (INR vs convex)
- DLNR: decouple memory and update roles (LSTM-style)
- GREAT-Stereo: inject global context via attention

### Insight 2: Multi-modal cost distributions matter
**MC-Stereo** explicitly addresses this via top-K peak lookup. Reflective surfaces, repetitive textures, and thin structures all produce multi-modal distributions that single-peak lookup fails on.

### Insight 3: Training-time enhancements compound with architecture improvements
**ICGNet** (knowledge distillation from SuperPoint) and **StereoAnything** (data curriculum) add zero inference overhead while improving accuracy. **Any edge model should adopt both.**

### Insight 4: Convergence can be dramatically accelerated
- **GREAT-Stereo:** 4 iterations match IGEV's 22 iterations (5.5× reduction)
- **MoCha-Stereo:** 4 iterations beat IGEV's 16 iterations
- **Any-Stereo:** robustness at 25% input scale shows information is preserved
- **Pattern:** better initialization or context → far fewer iterations

### Insight 5: Non-correlation paradigms work
**LoS** replaces correlation lookup entirely with structure propagation, achieving ~30× lower per-iteration cost. This is the most radical departure but the biggest efficiency win.

---

## Benchmark Comparison (KITTI 2015 D1-all)

| Method | D1-all | Runtime | Notes |
|--------|--------|---------|-------|
| RAFT-Stereo (baseline) | 1.82% | 0.38s | — |
| IGEV-Stereo (baseline) | 1.59% | 0.18s | — |
| CREStereo++ | **1.88%** (RVC) | 56ms (lite) | Cross-domain champion |
| DLNR | 1.76% | 135ms | High-frequency detail |
| MoCha-Stereo | **1.53%** | — | Motif channel attention |
| ICGNet | 1.57% | unchanged | Zero-overhead training |
| MC-Stereo | **1.55%** | 0.40s | Multi-peak + cascade |
| LoS | 1.42% | 0.31s | Structure propagation |
| Any-IGEV | 1.58% | 0.42s | INR upsampling |
| GREAT-IGEV | 1.50% | — | Global attention |

**All improvements over RAFT-Stereo baseline**, with different trade-offs on speed, generalization, and parameter count.

---

## Relevance to Our Edge Model

### Must-Adopt (Zero Inference Overhead)
1. **ICGNet** — SuperPoint-based geometric knowledge distillation during training
2. **StereoAnything** — 3-dataset curriculum + pseudo-stereo synthesis
3. **Pip-Stereo's MPT** (from Tier 1) — distill DAv2 priors into encoder weights

### High-Value Components
1. **Any-Stereo's INR upsampling** — run GRU at 1/16, upsample to full via MLP (dramatic memory savings)
2. **LoS's LSGP** — if we can implement it, ~30× per-iteration speedup
3. **GREAT-Stereo's MA module alone** — epipolar cross-attention provides global context at +1.84M params

### Worth Testing But Not Core
1. **DLNR's Decouple LSTM** — may help cross-domain generalization
2. **MC-Stereo's multi-peak lookup** — helps reflective surfaces
3. **CREStereo++'s UGAC** — helps rectification errors

### Skip
1. **MoCha-Stereo's REMP** — UNet refinement too expensive for edge
2. **GREAT-Stereo's SA + VA** — quadratic attention prohibitive

---

## The Pattern: Iteration Count is the Primary Knob

Reading all 9 papers, the consistent finding is that **better context/initialization dramatically reduces required iterations**:

| Approach | Result |
|----------|--------|
| Warm start from GEV (IGEV-Stereo) | 3 iter matches RAFT-32 |
| Motif channel context (MoCha) | 4 iter beats IGEV-16 |
| Global attention (GREAT) | 4 iter matches IGEV-22 |
| Monocular priors (DEFOM, Pip-Stereo) | 1-2 iter suffices |
| Structure propagation (LoS) | Per-iter cost drops 30× |

**Implication for our edge model:** the compute budget should go into **better initialization + richer context per iteration**, not more iterations. 2-4 iterations with the right context will beat 32 iterations with vanilla RAFT features.

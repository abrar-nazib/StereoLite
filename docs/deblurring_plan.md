# Fixing the Blur — Literature-Backed Plan (2026-07-03)

Corpus sweep of disparity over-smoothing / edge fattening: every cause and
cure with citations, mapped to our chassis (soft-argmin TileInit at 1/16 ->
slanted-tile ConvGRU refinement to 1/4 -> ConvexUpsample to full,
multi-scale L1 loss). Full agent report in the session record; this file is
the actionable distillation.

## The diagnosis is canonical (we are a textbook case)

- Tosi survey has a dedicated over-smoothing section: soft-argmin's expected
  value at a bimodal boundary pixel lands BETWEEN surfaces ("bleeding
  artifact", flying 3D points). Cure taxonomy: unimodal modeling /
  multi-modal modeling / iterative refinement.
  (`papers/summaries/tier1/surveys/Tosi_Survey_IJCV2025.md:189-198`)
- StereoRisk: the between-modes value is "a systematic bias, not noise".
- EdgeStereo names our decode cause: downsampling + convolution smoothness
  act as a low-pass filter on the disparity field.
- Any-Stereo proves it for RAFT-lineage nets: "the bottleneck is in the
  upsampling stage, not the GRU iterations"
  (`papers/summaries/tier2/iterative_variants/Any-Stereo.md:22,47`).
- Our evidence agrees: bad-0.5 stuck at 44-46% across ALL nine arms while
  EPE moved -31% => a representation/decode ceiling, not a loss-tuning
  ceiling.

Two root causes, both ours:
  C1 finest evidence at 1/4 + ConvexUpsample = learned 9-neighbor low-pass
  C2 L1 + soft-argmin mean-regression at bimodal boundary pixels

## Who actually moved boundary metrics (evidence table)

| Fix | Boundary evidence | Cost | Treats |
|---|---|---|---|
| SMD-Nets bimodal Laplacian head | boundary SEE3 1.73->1.13 (-35%); KT15 val 1.10->0.90 | ~190 params | C2 (+C1 as decode) |
| ADL multi-modal CE + dominant-mode | zero-shot bad-3 KT15 16.3->4.78, MB bad-2 25.1->8.85 | 0 params, 0 FLOPs | C2 (needs full volume; only our TileInit qualifies) |
| Top-k soft-argmin (CoEx k=2..4) | SF EPE 0.74->0.685; MUST be trained in | ~0 | C2 at init |
| HITNet truncated loss (A=1) + contrastive init loss | crisp-edge record, ETH3D bad-1 2.79 @20ms | 0 (loss-only) | C2 |
| Plane-equation full-res rendering (HITNet) | "slanted parameterization enables crisp edges without refinement module" | ~0 | C1 |
| Any-Stereo INR upsample | KITTI D1-fg 2.62->2.27 (param-neutral, +1.7% time) | ~0 | C1 |
| BGNet CUBG bilateral grid | 1/8+CUBG ~= native 1/4 accuracy at 10x speed | custom kernel | C1 (poor fit: our CV dies at 1/16) |
| Selective-Stereo SRU+CSA | MB bad-1 9.41->6.53, bad-2 4.83->2.51 (best module numbers) | +0.53M | both, BUT our _sru variant failed; re-test only after top-3 |
| DLNR decouple-LSTM + full-res DNR | MB bad-2 3.20 rank-1 | heavy; idea portable | hidden-state detail erosion |
| Full/half-res residual refinement | works ONLY with explicit warp-error/occlusion inputs (StereoDRNet, DLNR); plain convs fail (BGNet verdict; our sharp/hrrefine/sharptail failures) | varies | C1 |

Not in corpus: dedicated peakedness/W-entropy loss papers. Nearest: HITNet
Eq.10 contrastive-on-cost (different mechanism from the IGEV L_init
regression we already tested and REJECTED — keep separable).

## The plan (ordered by bad-0.5 gain per unit risk)

**Fix 1 — distribution-commitment bundle (zero params, zero latency,
ships in the full run):**
- top-k=3 soft-argmin at TileInit (must be trained in, never retrofitted;
  CoEx caveat: k=2 too aggressive untrained)
- HITNet truncated robust loss A=1 px on the 1/16..1/4 L1 terms
  (saturates the wrong-surface gradient at boundary tiles -> no mean pull);
  keep full-res term untruncated
- HITNet contrastive init loss on the TileInit cost volume (Eq.10-11,
  margin beta=1) — separable switch, drop independently if it hurts

**Fix 2 — render what we already carry (C1, chassis-native):**
plane-equation rendering from the 1/4 slanted-tile state to full res
(each tile spawns its 4x4 patch via d + sx dx + sy dy), blended with
ConvexUpsample by a learned edge gate. PREREQUISITE: gated slant
supervision done RIGHT — robust 9x9 plane-fit GT slants, gate chi(|d_err|<1)
(HITNet Eq.13). Our failed p2_slope_sup likely lacked both details; verify
before concluding slants can't be supervised on this chassis.

**Fix 3 — SMD-Nets bimodal Laplacian aux head (~190 params):**
NLL + depth-discontinuity-aware sampling, run as an AUXILIARY head on 1/4
features first (can't regress the main path), switch to mode-pick output
if it wins. The only fix with dedicated boundary-metric evidence.

**Explicitly deferred:** SRU/CSA (controlled re-test later; published gains
at 11M/32-iter scale, our _sru failed), ADL on final head (wrong attachment
point for a refinement-regressed chassis), any full-res refinement without
warp-error/occlusion inputs (three failures on record, ours + published).

## Decisive next experiment (before/parallel to the full run)

100-pair 80/20 A/B on the matched harness: baseline (gev4_opt_narrow + aug)
vs +bundle-1, deciding metric bad-0.5. ~15 min/arm on A10.
- If bundle-1 moves bad-0.5 => C2 was the binding cause; ship it in the
  full run and do Fix 2 as the follow-up.
- If bad-0.5 doesn't move => the ceiling is decode (C1); prioritize Fix 2
  (plane rendering) and Fix 3, bundle-1 still free to keep.

## Sequencing with the full SceneFlow run

The full run should NOT wait for Fix 2/3 (surgery). It SHOULD carry
bundle-1 if the A/B above validates it (zero cost, must-be-trained-in).
Fixes 2-3 become the post-full-run finetune/ablation round, evaluated on
bad-0.5 + MB14 zero-shot + the annotated collages.

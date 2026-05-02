# Phase 2 ablation sweep — 9 variants apples-to-apples

**Run:** 2026-05-02. 9 parallel A100-40GB containers via Modal `.starmap()`. Wall time ≈ 26 min for slowest. Cost ≈ **$7** (corrected from earlier $10 estimate; A100-40GB is $0.000583/sec = $2.10/hr, not $2.78/hr).

**Methodology:** Apples-to-apples atop the ghostconv winner chassis (`costlookup_yolo26n_full + ghostconv`). 100 SF Driving pairs (cached), batch=4, **9000 steps** (Phase 1 plateau budget), seed=42. Each variant adds ONE intervention — never violates the edge envelope (< 2.5 M params, INT8-quantizable ops).

## Full results (eval mode on all 100 pairs)

| Variant | Params | EPE | RMSE | med | bad-0.5 | bad-1 | bad-2 | bad-3 | D1 | ms |
|---|---|---|---|---|---|---|---|---|---|---|
| **p2_baseline** (control) | 1.254M | 0.8644 | 2.607 | 0.160 | 25.2% | 16.7% | 10.06% | 7.05% | 6.91% | 25.82 |
| **p2_cascade_cv_4** ★ | 1.322M | **0.8385** | **2.534** | 0.161 | **25.0%** | **16.1%** | **9.73%** | **6.82%** | **6.70%** | 41.16 |
| **p2_slope_aware_warp** ★ | 1.254M | 0.8435 | 2.582 | **0.152** | **23.9%** | **16.0%** | 9.90% | 6.98% | 6.86% | 39.73 |
| p2_selective_gate | 1.254M | 0.8623 | 2.585 | 0.175 | 24.2% | 16.4% | 10.11% | 7.10% | 6.96% | 38.85 |
| p2_conf_aware | 1.254M | 0.8718 | 2.650 | 0.163 | 25.0% | 16.4% | 9.98% | 7.06% | 6.91% | 25.20 |
| p2_edge_smooth | 1.254M | 0.8734 | 2.619 | 0.175 | 25.9% | 16.4% | 10.05% | 7.08% | 6.91% | 25.66 |
| p2_context_branch | 1.496M | 0.8818 | 2.585 | 0.193 | 25.8% | 16.9% | 10.14% | 7.12% | 7.00% | 41.53 |
| p2_seq_loss | 1.254M | 0.8936 | 2.673 | 0.183 | 25.4% | 16.8% | 10.29% | 7.28% | 7.05% | 47.44 |
| p2_slope_sup | 1.254M | 0.8972 | 2.653 | 0.173 | 26.8% | 17.2% | 10.48% | 7.26% | 7.04% | 26.81 |

(Sorted: baseline first, then by EPE.)

## Deltas vs baseline (negative = better)

| Variant | EPE | bad-1 | bad-2 | D1 | latency |
|---|---|---|---|---|---|
| **cascade_cv_4** | **-3.0%** | **-3.7%** | **-3.2%** | **-2.9%** | +59.4% |
| **slope_aware_warp** | -2.4% | **-4.0%** | -1.6% | -0.7% | +53.9% |
| selective_gate | -0.2% | -2.0% | +0.5% | +0.7% | +50.5% |
| conf_aware | +0.9% | -2.0% | -0.7% | +0.0% | -2.4% |
| edge_smooth | +1.0% | -1.5% | -0.1% | +0.1% | -0.6% |
| context_branch | +2.0% | +1.1% | +0.9% | +1.3% | +60.9% |
| seq_loss | +3.4% | +0.6% | +2.4% | +2.0% | +83.7% |
| slope_sup | +3.8% | +3.3% | +4.2% | +1.9% | +3.8% |

## Findings

**Two variants beat baseline on balanced metrics:**

1. **`cascade_cv_4`** (A3) — narrow-range 3D-aggregated cost volume between TileRefine iters at 1/4. Wins on EPE (-3.0%), bad-1 (-3.7%), bad-2 (-3.2%), and D1 (-2.9%). +68k params (1.322M total). **Latency cost: +59% (25.8 → 41.2 ms).**

2. **`slope_aware_warp`** (A2) — slope-corrected fR sample (averages two grid_samples at d ± 0.5·sx). Wins on bad-1 (-4.0%, the BEST sub-pixel improvement), bad-0.5 (-1.3pp), median AE (-5%). EPE -2.4%. **Same params** (1.254M). **Latency cost: +54% (extra grid_sample per iter).**

**Loss-side variants all DIDN'T help** at this step budget:
- `seq_loss` (RAFT-style γ-weighted L1) is **WORSE** by 3.4% EPE. Surprising given strong literature support — possibly under-converged or our γ=0.9 schedule is wrong.
- `slope_sup` is the WORST (worse than baseline by 3.8% EPE). Direct supervision of slopes via finite-diff GT gradients hurts; the network learns better slopes implicitly through the multi-scale L1.
- `conf_aware` and `edge_smooth` are basically tied with baseline.

**`context_branch` (A1) UNDERPERFORMED** despite strong literature precedent. EPE +2.0%, params +20%, latency +61%. The likely reason: with our cost-lookup chassis, the matching features already carry "context" implicitly via feature warping. Adding a separate context encoder added optimization noise without contributing useful information.

**`selective_gate` (A5)** is essentially tied with baseline on accuracy (-0.2% EPE) but +51% latency. Not worth the complexity.

## Practical conclusion

**Two surviving candidates: `cascade_cv_4` and `slope_aware_warp`.** Both are within the edge envelope (< 2.5M params, RTX 3050 fp16 latency would still be < 80 ms). Both are improvements; both pay ~50-60% latency.

**Phase 3 composition** (probably combining cascade_cv_4 + slope_aware_warp) is the next obvious step. Both are architectural so they should compose without obvious interference. Estimated combined: EPE ~0.82, +1.4 GB memory, ~50 ms latency.

**Drop from further consideration:**
- All 4 loss-side variants (seq_loss, slope_sup, conf_aware, edge_smooth) — none beat baseline at this budget
- context_branch — added cost without benefit on this chassis
- selective_gate — wash on accuracy, costs latency

**Caveat:** all results are at 9000 steps, training-batch-noise plateau region. The seq_loss negative result is suspicious — RAFT-style sequence loss is well-supported in literature. Possibly:
1. Under-converged at 9000 steps (our γ=0.9 weight requires more iters to stabilize)
2. Our γ=0.9 weight choice is wrong (RAFT-Stereo uses γ=0.9 for 32 iters; we have only 10)
3. The `with_iter_stages=True` mode keeps more activations and may be hitting AMP fp16 instability

Worth a re-test of seq_loss alone with γ=0.7 or γ=0.8 before fully dropping. Fast follow-up.

## Cost note

A100-40GB at $0.000583/sec = $2.10/hr (not $2.78 as I quoted earlier).
9 parallel containers × 22 min average × $0.000583/sec = **~$7**. Higher than my $5 estimate but lower than my "$10" panic.

For future ablation runs (small model, ~4 GB peak), **T4 at $0.59/hr** is ~3.6× cheaper per hour. Even at 1.5-2× longer wall time (we're launch-overhead-bound, not GPU-bound), total cost drops to **~$3** for the same 9-config sweep.

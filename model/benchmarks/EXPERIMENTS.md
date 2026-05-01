# Experiments

Chronological log of every overfit / ablation / training run, newest first.
Variants that haven't finished (no `final_metrics_all` in meta.json) are
labelled `(running)`.

Re-build this file:
    python3 model/scripts/build_experiments_summary.py

Per-run methodology: [`OVERFIT_METHODOLOGY.md`](OVERFIT_METHODOLOGY.md).

## yolo_ablation_20260430-204526
**Type:** YOLO encoder ablation
**Started:** 2026-04-30T20:45:33  ·  **Config:** 3000 steps, 384×640, 10 pairs, batch=2

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo26n | 0.808 | (running) | | | | | | | | |
| yolo26s | 2.061 | (running) | | | | | | | | |

_Per-variant artefacts: [`benchmarks/yolo_ablation_20260430-204526/`](benchmarks/yolo_ablation_20260430-204526/)_

## yolo_ablation_20260430-202931
**Type:** YOLO encoder ablation
**Started:** 2026-04-30T20:29:40  ·  **Config:** 3000 steps, 384×640, 10 pairs, batch=2

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo26n | 0.808 | (running) | | | | | | | | |

_Per-variant artefacts: [`benchmarks/yolo_ablation_20260430-202931/`](benchmarks/yolo_ablation_20260430-202931/)_

## matched_overfit_20260430-234721
**Type:** Matched encoder overfit (ghost vs yolo26n vs yolo26s)
**Started:** 2026-04-30T23:47:28  ·  **Config:** 3000 steps, 384×640, 20 pairs, batch=4

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| ghost | 0.538 | **0.6245** | 1.425 | 0.289 | 30.62% | **13.31%** | 5.85% | 3.55% | 3.54% | 23.5 |
| yolo26n | 0.808 | **0.7121** | 1.353 | 0.497 | 49.73% | **18.07%** | 4.79% | 2.91% | 2.90% | 24.1 |
| yolo26s | 2.061 | **0.5283** | 1.193 | 0.256 | 27.96% | **10.88%** | 4.12% | 2.50% | 2.50% | 25.8 |

_Per-variant artefacts: [`benchmarks/matched_overfit_20260430-234721/`](benchmarks/matched_overfit_20260430-234721/)_

## matched_overfit_20260430-225114
**Type:** Matched encoder overfit (ghost vs yolo26n vs yolo26s)
**Started:** 2026-04-30T22:51:20  ·  **Config:** 3000 steps, 384×640, 10 pairs, batch=4

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| ghost | 0.538 | **0.6960** | 1.999 | 0.381 | 33.56% | **11.49%** | 5.42% | 3.51% | 3.51% | 24.0 |
| yolo26n | 0.808 | **0.7836** | 1.646 | 0.563 | 52.43% | **29.83%** | 3.98% | 2.63% | 2.62% | 24.9 |
| yolo26s | 2.061 | **0.3751** | 1.296 | 0.126 | 13.14% | **6.19%** | 3.17% | 2.05% | 2.05% | 29.7 |

_Per-variant artefacts: [`benchmarks/matched_overfit_20260430-225114/`](benchmarks/matched_overfit_20260430-225114/)_

## matched_overfit_20260430-214720
**Type:** Matched encoder overfit (ghost vs yolo26n vs yolo26s)
**Started:** 2026-04-30T21:47:27  ·  **Config:** 3000 steps, 384×640, 10 pairs, batch=4

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| ghost | 0.538 | (running) | | | | | | | | |
| yolo26n | 0.808 | (running) | | | | | | | | |
| yolo26s | 2.061 | (running) | | | | | | | | |

_Per-variant artefacts: [`benchmarks/matched_overfit_20260430-214720/`](benchmarks/matched_overfit_20260430-214720/)_

## loss_ablation_20260501-132948
**Type:** Loss formulation A/B
**Started:** 2026-05-01T13:29:55  ·  **Config:** 3000 steps, 384×640, 20 pairs, batch=4

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| current (StereoLite_yolo, backbone=ghost) | 0.538 | **0.6629** | 1.901 | 0.268 | 27.62% | **12.10%** | 6.09% | 3.99% | 3.99% | - |
| current (StereoLite_yolo, backbone=ghost) | 0.538 | **0.7631** | 1.486 | 0.449 | 45.82% | **18.59%** | 6.21% | 3.66% | 3.65% | - |
| current (StereoLite_yolo, backbone=ghost) | 0.538 | **0.6178** | 1.956 | 0.196 | 23.94% | **10.73%** | 5.93% | 3.92% | 3.92% | - |
| current (StereoLite_yolo, backbone=ghost) | 0.538 | **1.0053** | 2.107 | 0.562 | 53.49% | **28.65%** | 13.25% | 4.16% | 4.04% | - |
| current (StereoLite_yolo, backbone=ghost) | 0.538 | **0.6309** | 1.422 | 0.320 | 31.03% | **11.96%** | 5.78% | 3.58% | 3.58% | - |
| current (StereoLite_yolo, backbone=ghost) | 0.538 | **0.6915** | 1.461 | 0.405 | 40.04% | **13.02%** | 6.12% | 3.68% | 3.68% | - |
| current (StereoLite_yolo, backbone=ghost) | 0.538 | **0.6765** | 1.376 | 0.356 | 38.28% | **16.62%** | 6.21% | 3.51% | 3.50% | - |
| current (StereoLite_yolo, backbone=ghost) | 0.538 | **0.6716** | 1.310 | 0.403 | 38.03% | **14.05%** | 5.97% | 3.46% | 3.45% | - |
| current (StereoLite_yolo, backbone=ghost) | 0.538 | **0.5913** | 1.258 | 0.301 | 31.48% | **11.73%** | 5.64% | 3.35% | 3.35% | - |

_Per-variant artefacts: [`benchmarks/loss_ablation_20260501-132948/`](benchmarks/loss_ablation_20260501-132948/)_

## arch_ablation_20260501-122438
**Type:** Architecture A/B/C overfit (refinement+upsample design)
**Started:** 2026-05-01T12:24:43  ·  **Config:** 3000 steps, 384×640, 20 pairs, batch=4

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| current | 0.538 | **0.5776** | 1.390 | 0.284 | 25.32% | **10.72%** | 5.60% | 3.44% | 3.44% | 23.8 |
| v1_iter | 0.576 | **0.5729** | 1.363 | 0.257 | 28.02% | **11.12%** | 5.44% | 3.25% | 3.25% | 32.5 |
| v2_hitnet | 0.487 | **0.8444** | 1.751 | 0.456 | 45.98% | **19.30%** | 8.50% | 4.86% | 4.79% | 25.2 |

_Per-variant artefacts: [`benchmarks/arch_ablation_20260501-122438/`](benchmarks/arch_ablation_20260501-122438/)_

## arch_ablation_20260501-122340
**Type:** Architecture A/B/C overfit (refinement+upsample design)
**Started:** 2026-05-01T12:23:45  ·  **Config:** 3000 steps, 384×640, 20 pairs, batch=4

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| current | 0.538 | (running) | | | | | | | | |

_Per-variant artefacts: [`benchmarks/arch_ablation_20260501-122340/`](benchmarks/arch_ablation_20260501-122340/)_

## arch_ablation_20260501-122133
**Type:** Architecture A/B/C overfit (refinement+upsample design)
**Started:** 2026-05-01T12:21:41  ·  **Config:** 3000 steps, 384×640, 20 pairs, batch=4

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| current | 0.538 | (running) | | | | | | | | |

_Per-variant artefacts: [`benchmarks/arch_ablation_20260501-122133/`](benchmarks/arch_ablation_20260501-122133/)_

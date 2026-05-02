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

## widener_n100_20260502-013518
**Type:** Unknown
**Started:** 2026-05-02T03:30:15  ·  **Config:** 5000 steps, 384×640, 100 pairs, batch=3

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| costlookup | 1.288 | (running) | | | | | | | | |
| costlookup | 1.045 | (running) | | | | | | | | |
| costlookup | 1.677 | (running) | | | | | | | | |
| costlookup | 0.952 | (running) | | | | | | | | |
| costlookup | 0.918 | (running) | | | | | | | | |
| costlookup | 1.254 | (running) | | | | | | | | |
| costlookup | 0.904 | (running) | | | | | | | | |
| costlookup | 3.562 | (running) | | | | | | | | |
| costlookup | 0.904 | (running) | | | | | | | | |
| costlookup | 1.306 | (running) | | | | | | | | |
| costlookup | 2.136 | **1.0390** | 2.929 | 0.235 | 30.00% | **19.22%** | 11.96% | 8.52% | 8.25% | 56.1 |

_Per-variant artefacts: [`benchmarks/widener_n100_20260502-013518/`](benchmarks/widener_n100_20260502-013518/)_

## raftlike_sweep_20260501-211601
**Type:** Unknown
**Started:** 2026-05-01T21:16:06  ·  **Config:** 3000 steps, 384×640, 20 pairs, batch=8

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| costlookup | 0.590 | **0.5743** | 1.397 | 0.242 | 25.20% | **11.55%** | 5.75% | 3.50% | 3.50% | 39.9 |
| costlookup | 0.645 | **0.6102** | 1.393 | 0.301 | 30.78% | **12.35%** | 5.50% | 3.30% | 3.30% | 50.8 |
| costlookup | 0.860 | **0.7500** | 1.385 | 0.485 | 49.16% | **23.66%** | 4.95% | 2.82% | 2.82% | 36.4 |
| costlookup | 0.904 | **0.5984** | 1.328 | 0.332 | 38.23% | **10.33%** | 4.84% | 3.00% | 3.00% | 49.9 |
| raftlike | 0.546 | **0.7966** | 1.557 | 0.491 | 48.91% | **18.21%** | 7.09% | 4.08% | 4.08% | 41.8 |
| raftlike | 0.586 | **0.6141** | 1.479 | 0.220 | 30.19% | **13.61%** | 6.22% | 3.84% | 3.84% | 56.9 |
| raftlike | 0.816 | **0.6246** | 1.344 | 0.329 | 35.90% | **14.44%** | 4.73% | 2.96% | 2.96% | 41.8 |
| raftlike | 0.845 | **0.7303** | 1.500 | 0.374 | 41.04% | **20.37%** | 6.21% | 3.69% | 3.69% | 59.7 |
| tilegru | 0.494 | **0.6779** | 1.538 | 0.331 | 30.22% | **14.15%** | 6.57% | 4.13% | 4.12% | 27.7 |
| tilegru | 0.517 | **0.6402** | 1.495 | 0.290 | 28.78% | **13.42%** | 6.57% | 3.96% | 3.96% | 40.3 |
| tilegru | 0.764 | **0.6132** | 1.325 | 0.291 | 34.23% | **14.82%** | 4.88% | 2.92% | 2.92% | 29.3 |
| tilegru | 0.776 | **0.7764** | 1.508 | 0.486 | 48.95% | **18.62%** | 6.71% | 3.84% | 3.84% | 37.5 |

_Per-variant artefacts: [`benchmarks/raftlike_sweep_20260501-211601/`](benchmarks/raftlike_sweep_20260501-211601/)_

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

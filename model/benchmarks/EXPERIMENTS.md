# Experiments

Chronological log of every overfit / ablation / training run, newest first.
Variants that haven't finished (no `final_metrics_all` in meta.json) are
labelled `(running)`.

Re-build this file:
    python3 model/scripts/build_experiments_summary.py

Per-run methodology: [`OVERFIT_METHODOLOGY.md`](OVERFIT_METHODOLOGY.md).

## arch_ablation_20260625-062310
**Type:** Architecture A/B/C overfit (refinement+upsample design)
**Started:** 2026-06-25T06:23:15  ·  **Config:** 10000 steps, 384×640, 100 pairs, batch=4

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo_ctx_guided | 2.983 | **0.7642** | 2.485 | 0.091 | 22.50% | **14.97%** | 9.22% | 6.44% | 6.33% | 28.9 |

_Per-variant artefacts: [`benchmarks/arch_ablation_20260625-062310/`](benchmarks/arch_ablation_20260625-062310/)_

## arch_ablation_20260624-181105
**Type:** Architecture A/B/C overfit (refinement+upsample design)
**Started:** 2026-06-24T18:11:06  ·  **Config:** 7000 steps, 384×640, 20 pairs, batch=4

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo_ctx_gev4 | 2.962 | **0.2610** | 0.885 | 0.059 | 9.97% | **5.25%** | 2.56% | 1.57% | 1.56% | 24.6 |

_Per-variant artefacts: [`benchmarks/arch_ablation_20260624-181105/`](benchmarks/arch_ablation_20260624-181105/)_

## arch_ablation_20260624-165515
**Type:** Architecture A/B/C overfit (refinement+upsample design)
**Started:** 2026-06-24T16:55:15  ·  **Config:** 6000 steps, 384×640, 20 pairs, batch=4

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo_ctx_init4 | 2.948 | **0.4553** | 1.052 | 0.195 | 22.66% | **10.64%** | 3.47% | 1.92% | 1.91% | 20.7 |

_Per-variant artefacts: [`benchmarks/arch_ablation_20260624-165515/`](benchmarks/arch_ablation_20260624-165515/)_

## arch_ablation_20260624-124603
**Type:** Architecture A/B/C overfit (refinement+upsample design)
**Started:** 2026-06-24T12:46:04  ·  **Config:** 7000 steps, 384×640, 20 pairs, batch=4

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo_ctx_sru | 3.398 | **0.4738** | 1.029 | 0.255 | 27.09% | **8.20%** | 3.06% | 1.82% | 1.81% | 14.8 |

_Per-variant artefacts: [`benchmarks/arch_ablation_20260624-124603/`](benchmarks/arch_ablation_20260624-124603/)_

## arch_ablation_20260624-031337
**Type:** Architecture A/B/C overfit (refinement+upsample design)
**Started:** 2026-06-24T03:13:38  ·  **Config:** 7000 steps, 384×640, 20 pairs, batch=4

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo_ctx_hrrefine | 3.106 | (running) | | | | | | | | |

_Per-variant artefacts: [`benchmarks/arch_ablation_20260624-031337/`](benchmarks/arch_ablation_20260624-031337/)_

## arch_ablation_20260624-023629
**Type:** Architecture A/B/C overfit (refinement+upsample design)
**Started:** 2026-06-24T02:36:29  ·  **Config:** 7000 steps, 384×640, 20 pairs, batch=4

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo_ctx_hrrefine | 3.106 | **0.3512** | 0.886 | 0.186 | 13.03% | **5.18%** | 2.53% | 1.54% | 1.54% | 16.1 |

_Per-variant artefacts: [`benchmarks/arch_ablation_20260624-023629/`](benchmarks/arch_ablation_20260624-023629/)_

## arch_ablation_20260624-015952
**Type:** Architecture A/B/C overfit (refinement+upsample design)
**Started:** 2026-06-24T01:59:53  ·  **Config:** 7000 steps, 384×640, 20 pairs, batch=4

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo_ctx_sharp | 2.950 | **0.3217** | 0.878 | 0.136 | 12.98% | **5.21%** | 2.56% | 1.55% | 1.55% | 14.6 |

_Per-variant artefacts: [`benchmarks/arch_ablation_20260624-015952/`](benchmarks/arch_ablation_20260624-015952/)_

## arch_ablation_20260624-005948
**Type:** Architecture A/B/C overfit (refinement+upsample design)
**Started:** 2026-06-24T00:59:48  ·  **Config:** 8000 steps, 384×640, 20 pairs, batch=8

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo_ctx_gate | 2.921 | **0.3187** | 0.894 | 0.127 | 11.76% | **5.24%** | 2.54% | 1.54% | 1.54% | 11.9 |

_Per-variant artefacts: [`benchmarks/arch_ablation_20260624-005948/`](benchmarks/arch_ablation_20260624-005948/)_

## arch_ablation_20260624-003119
**Type:** Architecture A/B/C overfit (refinement+upsample design)
**Started:** 2026-06-24T00:31:20  ·  **Config:** 8000 steps, 384×640, 20 pairs, batch=8

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo_ctx_gate | 1.410 | **0.3873** | 1.026 | 0.182 | 15.32% | **5.98%** | 2.99% | 1.89% | 1.89% | 10.9 |

_Per-variant artefacts: [`benchmarks/arch_ablation_20260624-003119/`](benchmarks/arch_ablation_20260624-003119/)_

## arch_ablation_20260624-002023
**Type:** Architecture A/B/C overfit (refinement+upsample design)
**Started:** 2026-06-24T00:20:24  ·  **Config:** 3000 steps, 384×640, 20 pairs, batch=4

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo_ctx_gate | 2.921 | **0.4726** | 1.096 | 0.246 | 25.42% | **7.75%** | 3.41% | 2.12% | 2.12% | 12.2 |

_Per-variant artefacts: [`benchmarks/arch_ablation_20260624-002023/`](benchmarks/arch_ablation_20260624-002023/)_

## arch_ablation_20260623-234424
**Type:** Architecture A/B/C overfit (refinement+upsample design)
**Started:** 2026-06-23T23:44:24  ·  **Config:** 3000 steps, 384×640, 20 pairs, batch=2

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo_geomctx | 2.919 | **0.4585** | 1.110 | 0.218 | 21.42% | **8.16%** | 3.60% | 2.23% | 2.22% | 36.7 |

_Per-variant artefacts: [`benchmarks/arch_ablation_20260623-234424/`](benchmarks/arch_ablation_20260623-234424/)_

## arch_ablation_20260623-195239
**Type:** Architecture A/B/C overfit (refinement+upsample design)
**Started:** 2026-06-23T19:52:39  ·  **Config:** 3000 steps, 384×640, 20 pairs, batch=8

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo_ctx | 2.831 | **0.5418** | 1.161 | 0.278 | 29.45% | **11.44%** | 4.14% | 2.39% | 2.39% | 10.5 |

_Per-variant artefacts: [`benchmarks/arch_ablation_20260623-195239/`](benchmarks/arch_ablation_20260623-195239/)_

## arch_ablation_20260623-192301
**Type:** Architecture A/B/C overfit (refinement+upsample design)
**Started:** 2026-06-23T19:23:02  ·  **Config:** 3000 steps, 384×640, 20 pairs, batch=4

| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo_ctx | 2.831 | **0.4238** | 1.104 | 0.167 | 18.74% | **7.62%** | 3.60% | 2.23% | 2.23% | 10.7 |

_Per-variant artefacts: [`benchmarks/arch_ablation_20260623-192301/`](benchmarks/arch_ablation_20260623-192301/)_

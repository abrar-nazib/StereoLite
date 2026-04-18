"""Latency-profiling harness. Wiring is deferred until the non-RAFT designs
(designs/d1_tile, d2_cascade, d3_sgm) have first-pass implementations.

This file is intentionally a stub. Once a design has a runnable model class,
add it to `make_model` below. Measure median / p95 wall-clock latency after
a short warm-up.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch


def make_model(name: str):
    raise NotImplementedError(
        f"No model registered for {name!r}. Implement designs/d1_tile, "
        "d2_cascade, or d3_sgm and wire them here."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--size", type=int, nargs=2, default=[384, 640])
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(args.model).to(device).eval()
    H, W = args.size
    left = torch.rand(1, 3, H, W, device=device) * 255
    right = torch.rand(1, 3, H, W, device=device) * 255

    with torch.no_grad():
        for _ in range(args.warmup):
            model(left, right)
            if device.type == "cuda":
                torch.cuda.synchronize()

    latencies: list[float] = []
    with torch.no_grad():
        for _ in range(args.trials):
            t0 = time.time()
            model(left, right)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.time() - t0) * 1000)

    latencies.sort()
    med = statistics.median(latencies)
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
    n_params = sum(pt.numel() for pt in model.parameters())
    print(f"model={args.model}  params={n_params/1e6:.2f} M  size={H}x{W}")
    print(f"latency ms    median={med:.1f}   p95={p95:.1f}")
    print(f"fps @ median  {1000/med:.1f}")


if __name__ == "__main__":
    main()

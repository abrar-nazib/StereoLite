"""Evaluation harness: EPE / bad-1 / bad-3 / D1-all across a dataset.

This file is intentionally a stub. Once a non-RAFT design exposes a single
inference callable (`model(left, right) -> disparity`), register it below.
Dataset loaders go under model/data/ when they are built.
"""
from __future__ import annotations

import argparse


def make_model(name: str):
    raise NotImplementedError(
        f"No model registered for {name!r}. Implement designs/d1_tile, "
        "d2_cascade, or d3_sgm and wire them here."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="kitti15",
                   choices=["kitti15", "sceneflow", "middlebury"])
    p.add_argument("--split", default="val")
    args = p.parse_args()
    raise NotImplementedError(
        "Evaluation loader not yet wired. Needs datasets under model/data/ "
        "and a runnable design under model/designs/."
    )


if __name__ == "__main__":
    main()

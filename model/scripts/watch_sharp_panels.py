"""Watch a training-panel dir and display the latest step's panels in a
cv2 window as they arrive.

Usage:
    python3 model/scripts/watch_sharp_panels.py \
        --dir model/benchmarks/sharp_<ts>

Press Q/ESC to close the window (training continues).
"""
from __future__ import annotations

import argparse
import os
import re
import time

import cv2
import numpy as np


STEP_RE = re.compile(r"step_(\d+)\.png$")


def latest_step_panels(root: str) -> tuple[int, list[str]]:
    """Look in <root>/montage/ for the latest step_<N>.png."""
    montage_dir = os.path.join(root, "montage")
    if not os.path.isdir(montage_dir):
        return -1, []
    buckets: dict[int, str] = {}
    for f in os.listdir(montage_dir):
        m = STEP_RE.search(f)
        if m:
            buckets[int(m.group(1))] = os.path.join(montage_dir, f)
    if not buckets:
        return -1, []
    step = max(buckets.keys())
    return step, [buckets[step]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="panel directory to watch")
    p.add_argument("--poll_sec", type=float, default=2.0)
    p.add_argument("--max_w", type=int, default=1800)
    args = p.parse_args()

    print(f"watching {args.dir}  (Q/ESC to quit)")
    win = "sharpness progress  (Q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.max_w, 800)

    last_step = -1
    placeholder = np.zeros((400, 1600, 3), dtype=np.uint8)
    cv2.putText(placeholder, "waiting for first panel...", (40, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    while True:
        step, panels = latest_step_panels(args.dir)
        if step < 0:
            cv2.imshow(win, placeholder)
        elif step != last_step and panels:
            imgs = [cv2.imread(pp) for pp in panels]
            imgs = [i for i in imgs if i is not None]
            if not imgs:
                cv2.imshow(win, placeholder)
            else:
                rows = []
                for img in imgs:
                    row = img
                    rows.append(row)
                panel = np.vstack(rows)
                if panel.shape[1] > args.max_w:
                    scale = args.max_w / panel.shape[1]
                    panel = cv2.resize(panel, (args.max_w,
                                               int(panel.shape[0] * scale)))
                head = np.zeros((34, panel.shape[1], 3), dtype=np.uint8)
                cv2.putText(head, f"training step {step}  ({len(panels)} val panels)",
                            (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (255, 255, 255), 1, cv2.LINE_AA)
                canvas = np.vstack([head, panel])
                cv2.imshow(win, canvas)
                last_step = step
                print(f"  displayed step {step}")

        k = cv2.waitKey(int(args.poll_sec * 1000)) & 0xFF
        if k in (ord("q"), ord("Q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

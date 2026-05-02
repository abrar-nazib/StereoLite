"""Build comparison mosaics for the architecture sweep.

For a given run dir with 12 variant subdirs (3 archs × 2 backbones ×
2 phases), produce three PNGs:
    comparison_phase1.png  — Phase 1 grid (3 rows × 2 cols), no extension
    comparison_phase2.png  — Phase 2 grid (3 rows × 2 cols), extend_to_full
    comparison_master.png  — Master grid: 6 rows × 2 cols (both phases)

Each panel shows the variant's final viz step (the last `viz/step_*.png`)
plus an overlay caption with EPE / bad-1.0 / bad-2.0 / params / latency.

Usage:
    python3 model/scripts/build_arch_mosaic.py <RUN_DIR>
    python3 model/scripts/build_arch_mosaic.py <RUN_DIR> --phase 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ARCHES = ("costlookup", "tilegru", "raftlike")
BACKBONES = ("ghost", "yolo26n")
ARCH_LABEL = {
    "costlookup": "Cost-Lookup",
    "tilegru":    "Tile-GRU",
    "raftlike":   "RAFT-like",
}


def latest_viz(variant_dir: Path) -> Path | None:
    viz = variant_dir / "viz"
    if not viz.is_dir():
        return None
    pngs = sorted(viz.glob("step_*.png"))
    return pngs[-1] if pngs else None


def load_meta(variant_dir: Path) -> dict:
    p = variant_dir / "meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def caption_for(variant_dir: Path, fallback_tag: str) -> list[str]:
    meta = load_meta(variant_dir)
    arch = meta.get("arch", fallback_tag)
    epe = meta.get("final_epe_all")
    metrics = meta.get("final_metrics_all", {})
    inf = meta.get("inference_bench", {})
    params = meta.get("params_train_M")
    lines = [f"{fallback_tag}"]
    if epe is not None:
        lines.append(
            f"EPE {epe:.3f}  bad1 {metrics.get('bad_1.0', '?')}  "
            f"bad2 {metrics.get('bad_2.0', '?')}")
    if inf:
        lines.append(
            f"{params:.3f}M  {inf.get('ms_mean','?')}ms  "
            f"{inf.get('fps_mean','?')}fps")
    return lines


def render_panel(viz_path: Path | None, caption_lines: list[str],
                 panel_h: int, panel_w: int) -> np.ndarray:
    if viz_path is None or not viz_path.exists():
        # Empty placeholder.
        img = np.full((panel_h, panel_w, 3), 64, dtype=np.uint8)
        cv2.putText(img, "(no run)", (panel_w // 2 - 60, panel_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    else:
        img = cv2.imread(str(viz_path))
        if img is None:
            img = np.full((panel_h, panel_w, 3), 64, dtype=np.uint8)
        else:
            # The variant viz panels are typically 3-stack (L | pred | err)
            # with a stat sidebar. We resize to panel_h x panel_w to fit
            # the grid uniformly.
            img = cv2.resize(img, (panel_w, panel_h))

    # Caption strip (semi-transparent black bar with white text on top).
    overlay = img.copy()
    bar_h = 22 * len(caption_lines) + 12
    cv2.rectangle(overlay, (0, 0), (panel_w, bar_h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    for i, line in enumerate(caption_lines):
        cv2.putText(img, line, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return img


def build_grid(run_dir: Path, extend: bool) -> np.ndarray | None:
    """Build a (3 archs × 2 backbones) grid panel for one phase.

    Each panel: latest viz PNG of variant <arch>_<backbone>[_full]/.
    Layout:
                 ghost           yolo26n
        costlookup [panel]       [panel]
        tilegru    [panel]       [panel]
        raftlike   [panel]       [panel]
    """
    panel_h, panel_w = 360, 720
    rows = []
    found_any = False
    for arch in ARCHES:
        cells = []
        for bb in BACKBONES:
            tag = f"{arch}_{bb}{'_full' if extend else ''}"
            vd = run_dir / tag
            viz = latest_viz(vd) if vd.is_dir() else None
            cap = caption_for(vd, tag) if vd.is_dir() else [tag, "(missing)"]
            if viz is not None:
                found_any = True
            cells.append(render_panel(viz, cap, panel_h, panel_w))
        rows.append(np.concatenate(cells, axis=1))
    if not found_any:
        return None
    grid = np.concatenate(rows, axis=0)
    # Title bar.
    title_h = 36
    title = np.full((title_h, grid.shape[1], 3), 32, dtype=np.uint8)
    label = (f"Phase {'2 (TileRefine to 1/2 + plane-eq, no Convex)' if extend else '1 (TileRefine to 1/4 + Convex)'}")
    cv2.putText(title, label, (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2,
                cv2.LINE_AA)
    return np.concatenate([title, grid], axis=0)


def build_master(run_dir: Path) -> np.ndarray | None:
    """6×2 master: rows = (cost/gru/raft × {phase1, phase2}), cols = backbone."""
    panel_h, panel_w = 320, 640
    rows_specs = []
    for extend in (False, True):
        for arch in ARCHES:
            rows_specs.append((arch, extend))
    rows = []
    found_any = False
    for arch, extend in rows_specs:
        cells = []
        for bb in BACKBONES:
            tag = f"{arch}_{bb}{'_full' if extend else ''}"
            vd = run_dir / tag
            viz = latest_viz(vd) if vd.is_dir() else None
            cap = caption_for(vd, tag) if vd.is_dir() else [tag, "(missing)"]
            if viz is not None:
                found_any = True
            cells.append(render_panel(viz, cap, panel_h, panel_w))
        rows.append(np.concatenate(cells, axis=1))
    if not found_any:
        return None
    grid = np.concatenate(rows, axis=0)
    title_h = 40
    title = np.full((title_h, grid.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(title, "Master sweep: 3 archs * 2 encoders * 2 phases",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255, 255, 255), 2, cv2.LINE_AA)
    return np.concatenate([title, grid], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str,
                    help="path to a sweep output dir "
                          "(e.g. model/benchmarks/raftlike_sweep_<TS>)")
    ap.add_argument("--phase", choices=["1", "2", "master", "all"],
                    default="all")
    args = ap.parse_args()
    rd = Path(args.run_dir)
    if not rd.is_dir():
        print(f"not a directory: {rd}", file=sys.stderr)
        sys.exit(2)

    if args.phase in ("1", "all"):
        g = build_grid(rd, extend=False)
        if g is not None:
            out = rd / "comparison_phase1.png"
            cv2.imwrite(str(out), g)
            print(f"wrote {out}  ({g.shape[1]}x{g.shape[0]})")
        else:
            print("Phase 1: no variants found yet; skipping mosaic")

    if args.phase in ("2", "all"):
        g = build_grid(rd, extend=True)
        if g is not None:
            out = rd / "comparison_phase2.png"
            cv2.imwrite(str(out), g)
            print(f"wrote {out}  ({g.shape[1]}x{g.shape[0]})")
        else:
            print("Phase 2: no variants found yet; skipping mosaic")

    if args.phase in ("master", "all"):
        g = build_master(rd)
        if g is not None:
            out = rd / "comparison_master.png"
            cv2.imwrite(str(out), g)
            print(f"wrote {out}  ({g.shape[1]}x{g.shape[0]})")
        else:
            print("Master: no variants found yet; skipping mosaic")


if __name__ == "__main__":
    main()

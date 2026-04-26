"""Quality scan over a FoundationStereo pseudo-GT dataset.

Reads every (left, right, disp_pseudo) triple, computes per-pair statistics,
applies multi-criterion quality filters, and writes:

  <DATASET>/clean_pairs.txt        — basenames passing all filters
  <DATASET>/quality_report.md      — human-readable filter breakdown + stats
  <DATASET>/quality_histograms.png — disparity / luminance distributions

Filters applied (each can be disabled via flags):

  • npy_corrupt       — .npy file is 0 bytes or unreadable
  • disp_outlier      — max disparity > MAX_DISP_PX (default 200 px)
  • image_dark        — mean luminance of left image < MIN_LUM (default 20/255)
  • image_uniform     — left-image variance < MIN_VAR (default 50, very flat)
  • disp_flat         — disparity std < MIN_DISP_STD (default 1.0 px,
                        no learning signal)
  • shape_mismatch    — disparity shape doesn't match image shape
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def collect_stats(pairs_dir: Path) -> list[dict]:
    """Walk left/, right/, disp_pseudo/ and gather per-pair stats."""
    left_files = sorted((pairs_dir / "left").glob("*.png"))
    rows: list[dict] = []
    for i, lp in enumerate(left_files):
        base = lp.stem
        rp = pairs_dir / "right" / lp.name
        dp = pairs_dir / "disp_pseudo" / f"{base}.npy"
        row: dict = {
            "base": base,
            "left_exists": lp.exists(),
            "right_exists": rp.exists(),
            "disp_exists": dp.exists(),
            "disp_size_bytes": dp.stat().st_size if dp.exists() else 0,
            "fail_reasons": [],
        }
        if not (lp.exists() and rp.exists() and dp.exists()):
            row["fail_reasons"].append("missing_file")
            rows.append(row)
            continue
        if row["disp_size_bytes"] == 0:
            row["fail_reasons"].append("npy_corrupt")
            rows.append(row)
            continue

        # Load disp
        try:
            d = np.load(dp)
        except Exception as e:
            row["fail_reasons"].append(f"npy_unreadable:{type(e).__name__}")
            rows.append(row)
            continue
        finite = np.isfinite(d)
        valid = (d > 0.5) & finite
        row["disp_shape"] = list(d.shape)
        row["disp_dtype"] = str(d.dtype)
        row["disp_min"] = float(d[finite].min()) if finite.any() else None
        row["disp_max"] = float(d[finite].max()) if finite.any() else None
        row["disp_mean"] = float(d[valid].mean()) if valid.any() else 0.0
        row["disp_std"] = float(d[valid].std()) if valid.any() else 0.0
        row["disp_p98"] = float(np.percentile(d[valid], 98)) if valid.any() else 0.0
        row["valid_pct"] = 100.0 * valid.sum() / d.size
        row["nan_pct"] = 100.0 * (~finite).sum() / d.size

        # Load left for image stats
        L = cv2.imread(str(lp))
        if L is None:
            row["fail_reasons"].append("left_unreadable")
            rows.append(row)
            continue
        gray = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
        row["left_h"], row["left_w"] = gray.shape
        row["left_lum_mean"] = float(gray.mean())
        row["left_lum_std"] = float(gray.std())
        # Variance proxy for texture content
        row["left_variance"] = float(gray.var())

        # Shape match check
        if (d.shape[0], d.shape[1]) != gray.shape:
            row["fail_reasons"].append(
                f"shape_mismatch:disp{d.shape}_img{gray.shape}")

        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  scanned {i + 1}/{len(left_files)}", file=sys.stderr)
    return rows


def apply_filters(rows: list[dict], args) -> tuple[list[dict], dict]:
    """Apply quality filters. Returns (filtered_rows, drop_counts)."""
    drops: dict[str, int] = {}

    def add_drop(row, reason):
        if reason not in row["fail_reasons"]:
            row["fail_reasons"].append(reason)
        drops[reason] = drops.get(reason, 0) + 1

    for row in rows:
        if row["fail_reasons"]:  # already failed at scan
            for r in row["fail_reasons"]:
                drops[r] = drops.get(r, 0) + 1
            continue
        if row.get("disp_max", 0) and row["disp_max"] > args.max_disp:
            add_drop(row, "disp_outlier")
        if row.get("left_lum_mean", 0) < args.min_lum:
            add_drop(row, "image_dark")
        if row.get("left_variance", 0) < args.min_var:
            add_drop(row, "image_uniform")
        if row.get("disp_std", 0) < args.min_disp_std:
            add_drop(row, "disp_flat")
        if row.get("nan_pct", 0) > args.max_nan_pct:
            add_drop(row, "disp_nan_inf")
    return rows, drops


def write_clean_list(rows, out_path: Path) -> int:
    cleans = [r["base"] for r in rows if not r["fail_reasons"]]
    out_path.write_text("\n".join(cleans) + "\n")
    return len(cleans)


def write_report(rows, drops, args, report_path: Path):
    n_total = len(rows)
    n_clean = sum(1 for r in rows if not r["fail_reasons"])
    n_drop = n_total - n_clean

    # Stats over CLEANED rows
    cleaned = [r for r in rows if not r["fail_reasons"]]
    if cleaned:
        d_means = np.array([r["disp_mean"] for r in cleaned])
        d_maxs = np.array([r["disp_max"] for r in cleaned])
        d_stds = np.array([r["disp_std"] for r in cleaned])
        l_lums = np.array([r["left_lum_mean"] for r in cleaned])
        l_vars = np.array([r["left_variance"] for r in cleaned])
    else:
        d_means = d_maxs = d_stds = l_lums = l_vars = np.array([])

    md = []
    md.append(f"# Pseudo-dataset quality report")
    md.append("")
    md.append(f"- Source dir: `{args.pairs_dir}`")
    md.append(f"- Total pairs scanned: **{n_total}**")
    md.append(f"- Pairs passing all filters: **{n_clean}** "
              f"({100*n_clean/max(n_total,1):.1f}%)")
    md.append(f"- Pairs dropped: **{n_drop}**")
    md.append("")
    md.append("## Drop counts by reason")
    md.append("")
    md.append("| reason | count | % of total |")
    md.append("|---|---|---|")
    for reason, count in sorted(drops.items(), key=lambda x: -x[1]):
        md.append(f"| {reason} | {count} | "
                  f"{100*count/max(n_total,1):.1f}% |")
    md.append("")
    md.append("## Filter thresholds used")
    md.append("")
    md.append(f"- `disp_outlier`: max disparity > **{args.max_disp} px**")
    md.append(f"- `image_dark`: mean luminance < **{args.min_lum} / 255**")
    md.append(f"- `image_uniform`: left-image variance < **{args.min_var}**")
    md.append(f"- `disp_flat`: disparity std < **{args.min_disp_std} px**")
    md.append(f"- `disp_nan_inf`: NaN/Inf pixel fraction > **{args.max_nan_pct}%**")
    md.append("")
    if cleaned:
        md.append("## Disparity statistics (CLEANED set)")
        md.append("")
        md.append("| stat | min | p5 | median | p95 | max |")
        md.append("|---|---|---|---|---|---|")
        for label, arr in (("mean disp / pair", d_means),
                            ("max disp / pair", d_maxs),
                            ("disp std / pair", d_stds)):
            md.append(
                f"| {label} | {arr.min():.2f} | {np.percentile(arr,5):.2f} | "
                f"{np.median(arr):.2f} | {np.percentile(arr,95):.2f} | "
                f"{arr.max():.2f} |")
        md.append("")
        md.append("## Image statistics (CLEANED set)")
        md.append("")
        md.append("| stat | min | p5 | median | p95 | max |")
        md.append("|---|---|---|---|---|---|")
        for label, arr in (("luminance mean / pair", l_lums),
                            ("variance / pair", l_vars)):
            md.append(
                f"| {label} | {arr.min():.2f} | {np.percentile(arr,5):.2f} | "
                f"{np.median(arr):.2f} | {np.percentile(arr,95):.2f} | "
                f"{arr.max():.2f} |")
        md.append("")
    md.append("## Sample dropped pairs (first 20)")
    md.append("")
    md.append("| basename | reasons |")
    md.append("|---|---|")
    n_listed = 0
    for r in rows:
        if r["fail_reasons"] and n_listed < 20:
            md.append(f"| {r['base']} | "
                      f"{', '.join(set(r['fail_reasons']))} |")
            n_listed += 1
    report_path.write_text("\n".join(md) + "\n")


def write_histograms(rows, out_path: Path):
    cleaned = [r for r in rows if not r["fail_reasons"]]
    if not cleaned:
        return
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    arrs = [
        ("mean disparity per pair (px)",
         np.array([r["disp_mean"] for r in cleaned]), "#2d6a4f"),
        ("max disparity per pair (px)",
         np.array([r["disp_max"] for r in cleaned]), "#c1121f"),
        ("disparity std per pair (px)",
         np.array([r["disp_std"] for r in cleaned]), "#8a5a00"),
        ("left-image luminance mean (0-255)",
         np.array([r["left_lum_mean"] for r in cleaned]), "#1a659e"),
    ]
    for ax, (label, arr, c) in zip(axes.flat, arrs):
        ax.hist(arr, bins=40, color=c, alpha=0.85, edgecolor="white",
                linewidth=0.4)
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.suptitle(
        f"Pseudo-dataset cleaned set distributions ({len(cleaned)} pairs)",
        fontsize=11, fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_dir", required=True,
                   help="directory containing left/, right/, disp_pseudo/")
    p.add_argument("--max_disp", type=float, default=200.0)
    p.add_argument("--min_lum", type=float, default=20.0)
    p.add_argument("--min_var", type=float, default=50.0)
    p.add_argument("--min_disp_std", type=float, default=1.0)
    p.add_argument("--max_nan_pct", type=float, default=5.0)
    args = p.parse_args()

    pairs = Path(args.pairs_dir)
    print(f"scanning {pairs} ...")
    rows = collect_stats(pairs)
    print(f"scanned {len(rows)} pairs")

    rows, drops = apply_filters(rows, args)

    clean_path = pairs / "clean_pairs.txt"
    n_clean = write_clean_list(rows, clean_path)
    print(f"wrote {clean_path}: {n_clean} clean pairs")

    report_path = pairs / "quality_report.md"
    write_report(rows, drops, args, report_path)
    print(f"wrote {report_path}")

    hist_path = pairs / "quality_histograms.png"
    write_histograms(rows, hist_path)
    print(f"wrote {hist_path}")

    # Also dump raw stats as json for downstream automation
    json_path = pairs / "quality_stats.json"
    json_path.write_text(json.dumps(rows, indent=1, default=str))
    print(f"wrote {json_path}")

    print(f"\nSUMMARY: {n_clean}/{len(rows)} pairs accepted "
          f"({100*n_clean/max(len(rows),1):.1f}%)")


if __name__ == "__main__":
    main()

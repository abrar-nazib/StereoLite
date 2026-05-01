"""Build a master EXPERIMENTS.md summary by walking model/benchmarks/.

Scans every `<benchmark_dir>/<variant>/meta.json` file under
model/benchmarks/, extracts the final stereo metrics, training time,
parameter counts, and inference latency, and writes a single
EXPERIMENTS.md at model/benchmarks/EXPERIMENTS.md.

Run after every completed experiment:
    python3 model/scripts/build_experiments_summary.py

Output is a chronological log: newest experiments first, each with a
self-contained markdown section showing the variants it compared and
the headline metrics. Only variants that have completed (i.e. their
meta.json has `final_metrics_all` or `final_metrics_all10`) are shown.
In-progress variants are listed but marked as "(running)".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "model" / "benchmarks"
OUT = BENCH / "EXPERIMENTS.md"


def load_meta(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  skip {path}: {e}", file=sys.stderr)
        return None


def fmt_metric(m: dict, key: str, fmt: str = ".3f") -> str:
    v = m.get(key)
    if v is None:
        return "-"
    return f"{v:{fmt}}"


def render_run(run_dir: Path) -> str:
    """Render one experiment dir as a markdown section."""
    metas: list[tuple[str, dict, Path]] = []
    for variant_dir in sorted(run_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        meta_path = variant_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = load_meta(meta_path)
        if meta is None:
            continue
        # Variant name: prefer the meta's own label if present.
        label = (meta.get("backbone") or meta.get("arch")
                 or meta.get("loss") or variant_dir.name)
        metas.append((label, meta, variant_dir))

    if not metas:
        return ""

    name = run_dir.name
    started = metas[0][1].get("started_at", "")
    sf_size = metas[0][1].get("n_pairs", "?")
    steps = metas[0][1].get("steps", "?")
    H = metas[0][1].get("height", "?")
    W = metas[0][1].get("width", "?")

    # Try to identify experiment type from prefix.
    if name.startswith("matched_overfit"):
        kind = "Matched encoder overfit (ghost vs yolo26n vs yolo26s)"
    elif name.startswith("arch_ablation"):
        kind = "Architecture A/B/C overfit (refinement+upsample design)"
    elif name.startswith("yolo_ablation"):
        kind = "YOLO encoder ablation"
    elif name.startswith("loss_ablation"):
        kind = "Loss formulation A/B"
    elif name.startswith("stereolite_finetune"):
        kind = "Indoor finetune"
    elif name.startswith("stereolite_v"):
        kind = "Standalone training run"
    else:
        kind = "Unknown"

    out = []
    out.append(f"## {name}")
    out.append(f"**Type:** {kind}")
    out.append(f"**Started:** {started}  ·  **Config:** {steps} steps, "
               f"{H}×{W}, {sf_size} pairs, batch={metas[0][1].get('batch','?')}")
    out.append("")

    # Table.
    out.append("| Variant | Trainable (M) | EPE | RMSE | Median | bad-0.5 | bad-1.0 | bad-2.0 | bad-3.0 | D1-all | Latency (ms) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for label, meta, vdir in metas:
        fm = (meta.get("final_metrics_all")
              or meta.get("final_metrics_all10"))
        params = meta.get("params_train_M",
                          meta.get("params_total_M", "?"))
        params_s = f"{params:.3f}" if isinstance(params, (int, float)) else str(params)
        bench = meta.get("inference_bench") or {}
        ms = bench.get("ms_mean")
        ms_s = f"{ms:.1f}" if isinstance(ms, (int, float)) else "-"
        if fm is None:
            out.append(f"| {label} | {params_s} | (running) | | | | | | | | |")
            continue
        out.append(
            f"| {label} | {params_s} | "
            f"**{fmt_metric(fm,'epe','.4f')}** | "
            f"{fmt_metric(fm,'rmse','.3f')} | "
            f"{fmt_metric(fm,'median_ae','.3f')} | "
            f"{fmt_metric(fm,'bad_0.5','.2f')}% | "
            f"**{fmt_metric(fm,'bad_1.0','.2f')}%** | "
            f"{fmt_metric(fm,'bad_2.0','.2f')}% | "
            f"{fmt_metric(fm,'bad_3.0','.2f')}% | "
            f"{fmt_metric(fm,'d1_all','.2f')}% | "
            f"{ms_s} |"
        )
    out.append("")

    # Path link.
    rel = run_dir.relative_to(BENCH.parent)
    out.append(f"_Per-variant artefacts: [`{rel}/`]({rel}/)_")
    out.append("")
    return "\n".join(out)


def main():
    if not BENCH.exists():
        print(f"missing: {BENCH}", file=sys.stderr)
        sys.exit(1)

    runs = []
    for p in sorted(BENCH.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        if any((p / sub / "meta.json").exists()
               for sub in p.iterdir() if sub.is_dir()):
            runs.append(p)

    sections = []
    for run in runs:
        s = render_run(run)
        if s:
            sections.append(s)

    header = """# Experiments

Chronological log of every overfit / ablation / training run, newest first.
Variants that haven't finished (no `final_metrics_all` in meta.json) are
labelled `(running)`.

Re-build this file:
    python3 model/scripts/build_experiments_summary.py

Per-run methodology: [`OVERFIT_METHODOLOGY.md`](OVERFIT_METHODOLOGY.md).

"""

    OUT.write_text(header + "\n".join(sections))
    print(f"wrote {OUT} ({len(sections)} run sections, "
          f"total {sum(s.count(chr(10)) for s in sections)} lines)")


if __name__ == "__main__":
    main()

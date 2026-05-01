#!/bin/bash
# Sequential orchestrator for the loss ablation sweep.
#
# Runs 9 loss variants on the current chassis (ghost encoder) at
# overfit-on-20-pairs, 3000 steps each. Total wall time ~2-3 hours
# on RTX 3050. Each variant lands in its own subdirectory under
#   model/benchmarks/loss_ablation_<TS>/<loss>/
# with the standard outputs (train.csv, curve.png, viz/, README.md, meta.json).
#
# Live OpenCV view is OFF (--show 0) for unattended sequential runs.
# To see live, run a single variant manually:
#   python3 model/scripts/overfit_loss_ablation.py --loss cocktail --show 1
#
# Run:
#   bash model/scripts/run_loss_sweep.sh
#
# Or in background:
#   bash model/scripts/run_loss_sweep.sh > /tmp/loss_sweep.log 2>&1 &
set -e
cd "$(dirname "$0")/../.."
source venv/bin/activate

TS=$(date +%Y%m%d-%H%M%S)
OUT="model/benchmarks/loss_ablation_${TS}"
mkdir -p "$OUT"
echo "Output root: $OUT"
echo "Log file: $OUT/sweep.log"
echo

LOSSES=(L1 L1_seq L1_grad L1_bad1 cocktail cocktail_b05 stack stack_d1 charbonnier)

# Header
{
    echo "=== Loss ablation sweep ==="
    echo "Started: $(date -Iseconds)"
    echo "Variants: ${LOSSES[*]}"
    echo "Output:   $OUT"
    echo
} | tee "$OUT/sweep.log"

for loss in "${LOSSES[@]}"; do
    {
        echo
        echo "=== $loss ($(date -Iseconds)) ==="
    } | tee -a "$OUT/sweep.log"

    python3 model/scripts/overfit_loss_ablation.py \
        --loss "$loss" \
        --out_root "$OUT" \
        --show 0 \
        2>&1 | tee -a "$OUT/sweep.log"

    if [ $? -ne 0 ]; then
        echo "!!! $loss FAILED, continuing with next variant" | tee -a "$OUT/sweep.log"
    fi
done

# Final summary
{
    echo
    echo "=== Sweep done $(date -Iseconds) ==="
    echo
    echo "Final results:"
    printf "%-15s %10s %10s %10s %10s %10s\n" "loss" "EPE" "bad-0.5%" "bad-1.0%" "bad-2.0%" "D1-all%"
    for loss in "${LOSSES[@]}"; do
        meta="$OUT/$loss/meta.json"
        if [ -f "$meta" ]; then
            python3 -c "
import json, sys
m = json.load(open('$meta'))
fm = m.get('final_metrics_all', {})
print(f'{m[\"loss\"]:<15} {fm.get(\"epe\",-1):>10.4f} '
      f'{fm.get(\"bad_0.5\",-1):>10.2f} {fm.get(\"bad_1.0\",-1):>10.2f} '
      f'{fm.get(\"bad_2.0\",-1):>10.2f} {fm.get(\"d1_all\",-1):>10.2f}')
"
        else
            echo "$loss   (missing meta.json)"
        fi
    done
} | tee -a "$OUT/sweep.log"

echo
echo "Done. Output: $OUT"
echo "Log: $OUT/sweep.log"

# Build the visual mosaic for cross-variant comparison.
echo
echo "Building visual comparison mosaic..."
python3 model/scripts/build_loss_mosaic.py --root "$OUT" 2>&1 | tee -a "$OUT/sweep.log"

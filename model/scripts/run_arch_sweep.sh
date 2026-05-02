#!/usr/bin/env bash
# 12-test architecture sweep:
#   3 mechanics (costlookup, tilegru, raftlike) × 2 encoders (ghost, yolo26n)
#   × 2 phases (extend_to_full=0|1).
#
# Phase 1: TileRefine ends at 1/4 + ConvexUpsample to full (extend=0)
# Phase 2: TileRefine all the way to 1/2 + plane-eq to full (extend=1)
#
# Per-variant batch sizes were measured in a 1-step train probe and
# target ~80% RTX 3050 GPU memory. See top-of-file table.
#
# Usage:
#   bash model/scripts/run_arch_sweep.sh              # auto-timestamped
#   bash model/scripts/run_arch_sweep.sh /tmp/myrun   # explicit dir
#
# Output dir layout:
#   <OUT_ROOT>/<arch>_<backbone>[_full]/{train.csv, viz/, meta.json, README.md, curve.png}
#   <OUT_ROOT>/sweep.log         # combined stdout from all 12 runs
#   <OUT_ROOT>/SWEEP_STATUS.txt  # human-readable progress (updated per variant)

set -e
cd "$(dirname "$0")/../.."
source venv/bin/activate
# Reduce allocator fragmentation for the new memory-heavier arches.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Defensive cleanup of any zombie overfit processes before we start.
# Prior sweeps could have left orphans holding the GPU.
pkill -9 -f "overfit_arch_ablation" 2>/dev/null || true
sleep 2

# Wait for GPU memory to drop below 200 MB before starting the sweep.
# (cudnn / cuda runtime can hang onto a few MB; >200 MB means a real
# tensor-holder is alive.)
echo "[sweep] waiting for GPU to free..."
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 200 ]; do
    sleep 2
done
echo "[sweep] GPU clear."

if [ -n "$1" ]; then
    OUT_ROOT="$1"
else
    TS=$(date +%Y%m%d-%H%M%S)
    OUT_ROOT="model/benchmarks/raftlike_sweep_${TS}"
fi
mkdir -p "$OUT_ROOT"
echo "[sweep] OUT_ROOT=$OUT_ROOT"
LOG="$OUT_ROOT/sweep.log"
STATUS="$OUT_ROOT/SWEEP_STATUS.txt"

# Configs: arch backbone extend_to_full batch
# (12 entries; phase 1 first, then phase 2). Batch sizes were tuned by
# 50-step smoke runs targeting the largest fitting batch on RTX 3050
# (3.96 GB) under AMP + expandable_segments. extend=1 phase pays a
# memory tax for 1/2-resolution TileRefine activations.
read -r -d '' CONFIGS <<'EOF' || true
costlookup ghost   0 8
costlookup yolo26n 0 6
tilegru    ghost   0 8
tilegru    yolo26n 0 8
raftlike   ghost   0 6
raftlike   yolo26n 0 6
costlookup ghost   1 4
costlookup yolo26n 1 3
tilegru    ghost   1 4
tilegru    yolo26n 1 3
raftlike   ghost   1 3
raftlike   yolo26n 1 2
EOF

n_total=$(echo "$CONFIGS" | wc -l)
echo "[sweep] $n_total configs queued"
echo "queued at $(date)" > "$STATUS"
echo "" >> "$STATUS"

i=0
while IFS= read -r line; do
    [ -z "$line" ] && continue
    i=$((i+1))
    set -- $line
    ARCH="$1"; BACKBONE="$2"; EXT="$3"; BATCH="$4"
    suffix=""; if [ "$EXT" = "1" ]; then suffix="_full"; fi
    TAG="${ARCH}_${BACKBONE}${suffix}"
    echo "" | tee -a "$LOG"
    echo "===========================================================" | tee -a "$LOG"
    echo "[sweep $i/$n_total] $TAG  (batch=$BATCH)  starting $(date +%H:%M:%S)" | tee -a "$LOG"
    echo "===========================================================" | tee -a "$LOG"
    echo "[$i/$n_total] $TAG  STARTED  $(date +%H:%M:%S)" >> "$STATUS"

    # Run the harness; tee its output into both the per-variant log and
    # the global sweep log. --show 0 is required because we run headless.
    set +e
    python3 -u model/scripts/overfit_arch_ablation.py \
        --arch "$ARCH" \
        --backbone "$BACKBONE" \
        --extend_to_full "$EXT" \
        --batch "$BATCH" \
        --out_root "$OUT_ROOT" \
        --show 0 \
        2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    set -e

    # Update status with final EPE if available.
    if [ -f "$OUT_ROOT/$TAG/meta.json" ]; then
        EPE=$(python3 -c "import json; d=json.load(open('$OUT_ROOT/$TAG/meta.json')); print(round(d.get('final_epe_all', float('nan')), 4))" 2>/dev/null || echo "?")
    else
        EPE="?"
    fi
    echo "[$i/$n_total] $TAG  FINISHED rc=$rc EPE=$EPE  $(date +%H:%M:%S)" >> "$STATUS"

    if [ "$rc" != "0" ]; then
        echo "[sweep] $TAG returned rc=$rc; continuing to next variant" | tee -a "$LOG"
    fi

    # Wait for GPU to fully release between configs (avoid orphan
    # state from any partial cleanup in the previous run carrying over).
    until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 200 ]; do
        sleep 2
    done

    # After config 6 (last Phase 1), build the Phase 1 mosaic.
    if [ "$i" = "6" ]; then
        echo "[sweep] Phase 1 done; building Phase 1 mosaic" | tee -a "$LOG"
        python3 model/scripts/build_arch_mosaic.py "$OUT_ROOT" --phase 1 \
            2>&1 | tee -a "$LOG" || true
    fi
done <<< "$CONFIGS"

echo "" | tee -a "$LOG"
echo "[sweep] all 12 variants done at $(date)" | tee -a "$LOG"
echo "" >> "$STATUS"
echo "ALL DONE at $(date)" >> "$STATUS"

# Build mosaics (Phase 1, Phase 2, master).
echo "[sweep] building mosaics..." | tee -a "$LOG"
python3 model/scripts/build_arch_mosaic.py "$OUT_ROOT" 2>&1 | tee -a "$LOG" || true

# Refresh the master EXPERIMENTS.md.
python3 model/scripts/build_experiments_summary.py 2>&1 | tee -a "$LOG" || true

echo "[sweep] DONE. Open $OUT_ROOT/comparison_master.png" | tee -a "$LOG"

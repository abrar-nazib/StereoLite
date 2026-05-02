#!/usr/bin/env bash
# 11-config feature-widener sweep at n_pairs=100, 5000 steps.
# Mechanic: costlookup_yolo26n_full (the strongest yolo26n+full from the
# 12-config sweep — see model/benchmarks/raftlike_sweep_20260501-211601/REPORT.md).
#
# 10 wideners on yolo26n + 1 ceiling reference (yolo26s_native).

set -e
cd "$(dirname "$0")/../.."
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -n "$1" ]; then
    OUT_ROOT="$1"
else
    TS=$(date +%Y%m%d-%H%M%S)
    OUT_ROOT="model/benchmarks/widener_n100_${TS}"
fi
mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/sweep.log"
STATUS="$OUT_ROOT/SWEEP_STATUS.txt"

# Defensive cleanup.
pkill -9 -f "overfit_arch_ablation" 2>/dev/null || true
sleep 2
echo "[widener] waiting for GPU to free..."
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 200 ]; do
    sleep 2
done
echo "[widener] GPU clear. OUT_ROOT=$OUT_ROOT"

# Configs: backbone widener batch tag
read -r -d '' CONFIGS <<'EOF' || true
yolo26n none         4 yolo26n_w_none
yolo26n f2_only      4 yolo26n_w_f2_only
yolo26n f2_f4        4 yolo26n_w_f2_f4
yolo26n all_to_s     3 yolo26n_w_all_to_s
yolo26n dw           2 yolo26n_w_dw
yolo26n mbconv       1 yolo26n_w_mbconv
yolo26n ghostconv    3 yolo26n_w_ghostconv
yolo26n topdown_fpn  3 yolo26n_w_topdown_fpn
yolo26n bifpn        2 yolo26n_w_bifpn
yolo26n gn_replace   4 yolo26n_w_gn_replace
yolo26s none         3 yolo26s_native_ceiling
EOF

n_total=11
echo "[widener] queued at $(date)" > "$STATUS"
echo "" >> "$STATUS"

i=0
while IFS= read -r line; do
    [ -z "$line" ] && continue
    i=$((i+1))
    set -- $line
    BB="$1"; W="$2"; B="$3"; TAG="$4"
    echo "" | tee -a "$LOG"
    echo "===========================================================" | tee -a "$LOG"
    echo "[widener $i/$n_total] $TAG  bb=$BB widener=$W batch=$B  $(date +%H:%M:%S)" | tee -a "$LOG"
    echo "===========================================================" | tee -a "$LOG"
    echo "[$i/$n_total] $TAG  STARTED  $(date +%H:%M:%S)" >> "$STATUS"

    set +e
    python3 -u model/scripts/overfit_arch_ablation.py \
        --arch costlookup \
        --backbone "$BB" \
        --extend_to_full 1 \
        --widener "$W" \
        --batch "$B" \
        --n_pairs 100 \
        --steps 5000 \
        --out_root "$OUT_ROOT" \
        --variant_tag "$TAG" \
        --show 0 \
        --viz_interval_s 30 \
        2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    set -e

    if [ -f "$OUT_ROOT/$TAG/meta.json" ]; then
        EPE=$(python3 -c "import json; d=json.load(open('$OUT_ROOT/$TAG/meta.json')); print(round(d.get('final_epe_all', float('nan')), 4))" 2>/dev/null || echo "?")
    else
        EPE="?"
    fi
    echo "[$i/$n_total] $TAG  FINISHED rc=$rc EPE=$EPE  $(date +%H:%M:%S)" >> "$STATUS"

    if [ "$rc" != "0" ]; then
        echo "[widener] $TAG returned rc=$rc; continuing" | tee -a "$LOG"
    fi

    until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 200 ]; do
        sleep 2
    done
done <<< "$CONFIGS"

echo "" | tee -a "$LOG"
echo "[widener] all $n_total variants done at $(date)" | tee -a "$LOG"
echo "ALL DONE at $(date)" >> "$STATUS"

# Refresh master EXPERIMENTS.md
python3 model/scripts/build_experiments_summary.py 2>&1 | tee -a "$LOG" || true

echo "[widener] DONE. Reports next."

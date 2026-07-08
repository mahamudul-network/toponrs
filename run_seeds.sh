#!/bin/bash
# Sequential multi-seed runs for one variant on one GPU.
# Usage: bash run_seeds.sh <gpu> <variant> <seed1> [seed2 ...]
set -u
GPU=$1; VARIANT=$2; shift 2
LOGDIR=results/seed_runs/logs
mkdir -p "$LOGDIR"
for SEED in "$@"; do
  OUT="results/seed_runs/${VARIANT}_seed${SEED}_results.json"
  if [ -f "$OUT" ]; then
    echo "[skip] $OUT already exists"
    continue
  fi
  echo "[start] variant=$VARIANT seed=$SEED gpu=$GPU $(date)"
  python3 run_all_experiments.py --variant "$VARIANT" --seed "$SEED" --gpu "$GPU" \
      --epochs 6 --validate_every 500 \
      > "$LOGDIR/${VARIANT}_seed${SEED}.log" 2>&1
  RC=$?
  echo "[done] variant=$VARIANT seed=$SEED rc=$RC $(date)"
done
echo "QUEUE_COMPLETE variant=$VARIANT gpu=$GPU"

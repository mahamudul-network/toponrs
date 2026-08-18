#!/bin/bash
# MIND-large multi-seed runs. Usage: bash run_seeds_large.sh <gpu> <variant> <seed...>
# 2 epochs (14x more data/epoch than MIND-small); validation kept infrequent
# because MIND-large evaluation is expensive.
set -u
GPU=$1; VARIANT=$2; shift 2
LOGDIR=results/seed_runs/logs; mkdir -p "$LOGDIR"
for SEED in "$@"; do
  OUT="results/seed_runs/${VARIANT}_large_seed${SEED}_results.json"
  if [ -f "$OUT" ]; then echo "[skip] $OUT exists"; continue; fi
  echo "[start] large variant=$VARIANT seed=$SEED gpu=$GPU $(date)"
  python3 run_all_experiments.py --dataset large --variant "$VARIANT" --seed "$SEED" \
      --gpu "$GPU" --epochs 2 --validate_every 20000 \
      > "$LOGDIR/${VARIANT}_large_seed${SEED}.log" 2>&1
  echo "[done] large variant=$VARIANT seed=$SEED rc=$? $(date)"
done
echo "QUEUE_COMPLETE_LARGE variant=$VARIANT gpu=$GPU"

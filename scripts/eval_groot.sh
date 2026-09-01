#!/usr/bin/env bash
# GR00T N1.7 offline eval launch wrapper
# =========================================
# Thin wrapper around scripts/eval_groot.py, run inside the sibling GR00T
# venv set up by scripts/setup_groot_env.sh. Computes the same
# forward-pass loss lerobot-train uses for its own held-out evaluation,
# standalone, against an already-trained checkpoint.
#
# CAVEAT: unless the checkpoint was trained with --dataset.eval_split > 0,
# this is an IN-SAMPLE loss, not a generalization measure. See
# scripts/eval_groot.py's docstring / printed output.
#
# Usage:
#   scripts/eval_groot.sh --dataset-dir open_trashcan_50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_DIR="open_trashcan_50"
GROOT_DIR="../lerobot-groot"
CHECKPOINT=""
BATCH_SIZE=16
NUM_WORKERS=4
GPU=1

usage() {
    grep '^#' "$0" | sed 's/^#//; s/^ //'
    cat <<'EOF'

Flags:
  --dataset-dir PATH    Local LeRobot v3 dataset dir (default: open_trashcan_50)
  --checkpoint PATH     Saved pretrained_model dir (default: outputs/groot_<dataset-dir-basename>/checkpoints/last/pretrained_model)
  --groot-dir PATH      GR00T-enabled lerobot checkout from setup script (default: ../lerobot-groot)
  --batch-size N        (default: 16)
  --num-workers N       (default: 4)
  --gpu N               CUDA_VISIBLE_DEVICES (default: 1 -- check nvidia-smi and adjust)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --groot-dir) GROOT_DIR="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

DATASET_BASENAME="$(basename "$DATASET_DIR")"
CHECKPOINT="${CHECKPOINT:-outputs/groot_${DATASET_BASENAME}/checkpoints/last/pretrained_model}"

if [[ ! -f "$DATASET_DIR/meta/info.json" ]]; then
    echo "[ERROR] $DATASET_DIR does not look like a LeRobot dataset (missing meta/info.json)" >&2
    exit 1
fi
if [[ ! -f "$CHECKPOINT/config.json" ]]; then
    echo "[ERROR] $CHECKPOINT does not look like a saved policy (missing config.json)" >&2
    exit 1
fi
if [[ ! -d "$GROOT_DIR" ]]; then
    echo "[ERROR] $GROOT_DIR not found -- run scripts/setup_groot_env.sh first" >&2
    exit 1
fi

echo "==> Dataset:    $DATASET_DIR"
echo "==> Checkpoint: $CHECKPOINT"
echo "==> GR00T env:  $GROOT_DIR"
echo "==> GPU:        $GPU (CUDA_VISIBLE_DEVICES)"
echo

CUDA_VISIBLE_DEVICES="$GPU" uv run --project "$GROOT_DIR" python "$SCRIPT_DIR/eval_groot.py" \
    --checkpoint "$CHECKPOINT" \
    --dataset-dir "$DATASET_DIR" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --device cuda

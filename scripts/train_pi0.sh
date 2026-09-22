#!/usr/bin/env bash
# pi0 finetuning launch wrapper
# ==============================
# Thin wrapper around `lerobot-train --policy.path=lerobot/pi0_base`, run
# inside the sibling pi0 venv set up by scripts/setup_pi0_env.sh, pointed
# at a local LeRobot v3 dataset in this repo (e.g. open_trashcan_50/).
#
# open_trashcan has one camera (cam_wrist); pi0_base's pretrained config
# expects three generic slots (base_0_rgb / left_wrist_0_rgb /
# right_wrist_0_rgb). --rename_map below maps cam_wrist onto
# left_wrist_0_rgb (a real wrist camera, closest semantic match) --
# the other two stay unfilled, same "dataset visuals as a subset of the
# policy's declared visuals" mechanism used for SmolVLA (see
# eval_smolvla_open_trashcan.py / scripts/run_sample_size_sweep.sh).
#
# Usage:
#   scripts/train_pi0.sh --dataset-dir open_trashcan_50
#
# Smoke-test the pipeline first (fast, cheap, catches setup issues before
# committing to a real run):
#   scripts/train_pi0.sh --dataset-dir open_trashcan_10 \
#       --steps 30 --batch-size 8 --save-freq 30

set -euo pipefail

DATASET_DIR="open_trashcan_50"
PI0_DIR="../lerobot-pi0"
OUTPUT_DIR=""
JOB_NAME=""
BASE_MODEL="lerobot/pi0_base"
BATCH_SIZE=8
STEPS=50000
SAVE_FREQ=5000
GPU=1
WANDB_ENABLE=false
SEED=42
EXTRA_ARGS=""

usage() {
    grep '^#' "$0" | sed 's/^#//; s/^ //'
    cat <<'EOF'

Flags:
  --dataset-dir PATH    Local LeRobot v3 dataset dir (default: open_trashcan_50)
  --pi0-dir PATH        pi0-enabled lerobot checkout from setup script (default: ../lerobot-pi0)
  --output-dir PATH     Checkpoints/logs dir (default: outputs/pi0_<dataset-dir-basename>)
  --job-name NAME       (default: <dataset-dir-basename>)
  --base-model ID       HF model id to finetune from (default: lerobot/pi0_base)
  --batch-size N        (default: 8)
  --steps N             (default: 50000)
  --save-freq N         (default: 5000)
  --gpu N               CUDA_VISIBLE_DEVICES (default: 1 -- check nvidia-smi and adjust)
  --wandb               Enable W&B logging (default: off)
  --seed N              (default: 42)
  --extra-args "..."    Verbatim passthrough appended to the lerobot-train invocation
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
        --pi0-dir) PI0_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --job-name) JOB_NAME="$2"; shift 2 ;;
        --base-model) BASE_MODEL="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --save-freq) SAVE_FREQ="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --wandb) WANDB_ENABLE=true; shift ;;
        --seed) SEED="$2"; shift 2 ;;
        --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ ! -f "$DATASET_DIR/meta/info.json" ]]; then
    echo "[ERROR] $DATASET_DIR does not look like a LeRobot dataset (missing meta/info.json)" >&2
    exit 1
fi
if [[ ! -d "$PI0_DIR" ]]; then
    echo "[ERROR] $PI0_DIR not found -- run scripts/setup_pi0_env.sh first" >&2
    exit 1
fi

DATASET_BASENAME="$(basename "$DATASET_DIR")"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pi0_${DATASET_BASENAME}}"
JOB_NAME="${JOB_NAME:-$DATASET_BASENAME}"

CMD=(uv run --project "$PI0_DIR" lerobot-train
    "--dataset.repo_id=local/${DATASET_BASENAME}"
    "--dataset.root=${DATASET_DIR}"
    "--dataset.video_backend=pyav"
    "--policy.path=${BASE_MODEL}"
    "--policy.device=cuda"
    "--policy.push_to_hub=false"
    "--rename_map={\"observation.images.cam_wrist\": \"observation.images.left_wrist_0_rgb\"}"
    "--seed=${SEED}"
    "--batch_size=${BATCH_SIZE}"
    "--steps=${STEPS}"
    "--save_freq=${SAVE_FREQ}"
    "--output_dir=${OUTPUT_DIR}"
    "--job_name=${JOB_NAME}"
    "--wandb.enable=${WANDB_ENABLE}"
)
[[ -n "$EXTRA_ARGS" ]] && CMD+=($EXTRA_ARGS)

echo "==> Dataset:    $DATASET_DIR"
echo "==> pi0 env:    $PI0_DIR"
echo "==> Output dir: $OUTPUT_DIR"
echo "==> GPU:        $GPU (CUDA_VISIBLE_DEVICES)"
echo "==> Command:"
printf '    %s\n' "${CMD[@]}"
echo

CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"

cat <<EOF

==> Done. Checkpoints under: $OUTPUT_DIR/checkpoints/<step|last>/pretrained_model
EOF

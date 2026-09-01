#!/usr/bin/env bash
# GR00T N1.7 finetuning launch wrapper
# =====================================
# Thin wrapper around `lerobot-train --policy.type=groot`, run inside the
# sibling GR00T venv set up by scripts/setup_groot_env.sh, pointed at a
# local LeRobot v3 dataset in this repo (e.g. open_trashcan_50/).
#
# Usage:
#   scripts/train_groot.sh --dataset-dir open_trashcan_50
#
# Smoke-test the pipeline first (fast, cheap, catches setup/auth issues
# before committing to a real run):
#   scripts/train_groot.sh --dataset-dir open_trashcan_50 \
#       --steps 10 --batch-size 4 --save-freq 5

set -euo pipefail

DATASET_DIR="open_trashcan_50"
GROOT_DIR="../lerobot-groot"
OUTPUT_DIR=""
JOB_NAME=""
BASE_MODEL="nvidia/GR00T-N1.7-3B"
EMBODIMENT_TAG="new_embodiment"
CHUNK_SIZE=16
BATCH_SIZE=32
STEPS=3000
SAVE_FREQ=500
GPU=1
WANDB_ENABLE=false
PUSH_TO_HUB=false
HUB_REPO_ID=""
SEED=42
EXTRA_ARGS=""

usage() {
    grep '^#' "$0" | sed 's/^#//; s/^ //'
    cat <<'EOF'

Flags:
  --dataset-dir PATH     Local LeRobot v3 dataset dir (default: open_trashcan_50)
  --groot-dir PATH       GR00T-enabled lerobot checkout from setup script (default: ../lerobot-groot)
  --output-dir PATH      Checkpoints/logs dir (default: outputs/groot_<dataset-dir-basename>)
  --job-name NAME        (default: <dataset-dir-basename>)
  --base-model ID        HF model id (default: nvidia/GR00T-N1.7-3B)
  --embodiment-tag TAG   (default: new_embodiment)
  --chunk-size N         chunk_size / n_action_steps (default: 16)
  --batch-size N         (default: 32)
  --steps N              (default: 3000)
  --save-freq N          (default: 500)
  --gpu N                CUDA_VISIBLE_DEVICES (default: 1 -- GPU 0 may be busy, check nvidia-smi)
  --wandb                Enable W&B logging (default: off)
  --push-to-hub          Push resulting policy to the HF Hub (needs --hub-repo-id)
  --hub-repo-id ID       Destination repo id when --push-to-hub is set
  --seed N               (default: 42)
  --extra-args "..."     Verbatim passthrough appended to the lerobot-train invocation
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
        --groot-dir) GROOT_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --job-name) JOB_NAME="$2"; shift 2 ;;
        --base-model) BASE_MODEL="$2"; shift 2 ;;
        --embodiment-tag) EMBODIMENT_TAG="$2"; shift 2 ;;
        --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --save-freq) SAVE_FREQ="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --wandb) WANDB_ENABLE=true; shift ;;
        --push-to-hub) PUSH_TO_HUB=true; shift ;;
        --hub-repo-id) HUB_REPO_ID="$2"; shift 2 ;;
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
if [[ ! -d "$GROOT_DIR" ]]; then
    echo "[ERROR] $GROOT_DIR not found -- run scripts/setup_groot_env.sh first" >&2
    exit 1
fi
if [[ "$PUSH_TO_HUB" == true && -z "$HUB_REPO_ID" ]]; then
    echo "[ERROR] --push-to-hub requires --hub-repo-id" >&2
    exit 1
fi

DATASET_BASENAME="$(basename "$DATASET_DIR")"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/groot_${DATASET_BASENAME}}"
JOB_NAME="${JOB_NAME:-$DATASET_BASENAME}"

CMD=(uv run --project "$GROOT_DIR" lerobot-train
    "--dataset.repo_id=local/${DATASET_BASENAME}"
    "--dataset.root=${DATASET_DIR}"
    "--dataset.image_transforms.enable=true"
    "--policy.type=groot"
    "--policy.device=cuda"
    "--policy.base_model_path=${BASE_MODEL}"
    "--policy.embodiment_tag=${EMBODIMENT_TAG}"
    "--policy.chunk_size=${CHUNK_SIZE}"
    "--policy.n_action_steps=${CHUNK_SIZE}"
    "--policy.use_relative_actions=true"
    "--policy.relative_exclude_joints=[\"gripper\"]"
    "--policy.use_bf16=true"
    "--policy.push_to_hub=${PUSH_TO_HUB}"
    "--seed=${SEED}"
    "--batch_size=${BATCH_SIZE}"
    "--steps=${STEPS}"
    "--save_checkpoint=true"
    "--save_freq=${SAVE_FREQ}"
    "--use_policy_training_preset=true"
    "--env_eval_freq=0"
    "--eval_steps=0"
    "--log_freq=10"
    "--output_dir=${OUTPUT_DIR}"
    "--job_name=${JOB_NAME}"
    "--wandb.enable=${WANDB_ENABLE}"
)
[[ "$PUSH_TO_HUB" == true ]] && CMD+=("--policy.repo_id=${HUB_REPO_ID}")
[[ -n "$EXTRA_ARGS" ]] && CMD+=($EXTRA_ARGS)

echo "==> Dataset:    $DATASET_DIR"
echo "==> GR00T env:  $GROOT_DIR"
echo "==> Output dir: $OUTPUT_DIR"
echo "==> GPU:        $GPU (CUDA_VISIBLE_DEVICES)"
echo "==> Command:"
printf '    %s\n' "${CMD[@]}"
echo

CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"

cat <<EOF

==> Done. Checkpoints/logs under: $OUTPUT_DIR
    (exact checkpoint filename layout depends on the installed lerobot
    version -- check $OUTPUT_DIR directly.)
EOF

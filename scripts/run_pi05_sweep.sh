#!/usr/bin/env bash
# pi05 sample-size sweep: trains pi05 (frozen-backbone recipe, same as pi0's
# -- see scripts/run_pi0_sweep.sh) independently on the 10/25/50/100-episode
# open_trashcan subsets, then runs open-loop eval for each checkpoint
# against the same fixed held-out episode set used for the ACT/SmolVLA/pi0
# sweeps (outputs/sample_size_sweep/held_out_episodes.json).
#
# Reuses scripts/train_pi0.sh directly (it's generic over --base-model, no
# pi05-specific wrapper needed) with --base-model lerobot/pi05_base and
# explicit --output-dir overrides so checkpoints land under pi05_* instead
# of pi0_*. Same isolated env (../lerobot-pi0, already has pi05 available
# since it's part of the same lerobot checkout) and the same
# cam_wrist -> left_wrist_0_rgb camera mapping (pi05_base uses the identical
# base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb convention as pi0_base).
#
# Step budget: 20,000, same as pi0 -- pi05 runs at a comparable ~1.1 step/s
# even with the frozen-backbone recipe (693M/4.1B params trainable), so 50k
# would cost as much sequential wall-clock as pi0's did.
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=20000
BATCH_SIZE=8
SAVE_FREQ=5000
GPU=3
LOG_DIR=outputs/sample_size_sweep
mkdir -p "$LOG_DIR"

PI05_RECIPE="--policy.dtype=bfloat16 --policy.train_expert_only=true --policy.freeze_vision_encoder=true --policy.gradient_checkpointing=true"

HELD_OUT="15 20 23 32 43 48 55 60 69 73 75 83 87 93 96"

SIZES="10 25 50 100"

for size in $SIZES; do
    if [ "$size" = "100" ]; then
        dataset_dir="open_trashcan"
    else
        dataset_dir="open_trashcan_${size}"
    fi

    name="pi05_open_trashcan_${size}"
    echo "=== [$(date)] Training $name on $dataset_dir ==="
    bash scripts/train_pi0.sh \
        --dataset-dir "$dataset_dir" \
        --base-model lerobot/pi05_base \
        --output-dir "outputs/$name" \
        --job-name "$name" \
        --batch-size $BATCH_SIZE \
        --steps $STEPS \
        --save-freq $SAVE_FREQ \
        --gpu $GPU \
        --extra-args "$PI05_RECIPE" \
        > "$LOG_DIR/${name}.log" 2>&1

    echo "=== [$(date)] Evaluating $name ==="
    CUDA_VISIBLE_DEVICES=$GPU uv run --project ../lerobot-pi0 python eval_pi05_open_trashcan.py \
        --checkpoint "outputs/$name/checkpoints/last/pretrained_model" \
        --dataset-name open_trashcan \
        --episodes $HELD_OUT \
        --output "$LOG_DIR/${name}_eval.json" \
        > "$LOG_DIR/${name}_eval.log" 2>&1
done

echo "=== [$(date)] pi05 sweep complete ==="

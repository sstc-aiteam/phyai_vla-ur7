#!/usr/bin/env bash
# pi0 sample-size sweep: trains pi0 (frozen-backbone recipe, matching
# SmolVLA's efficient finetune approach) independently on the 10/25/50/100-
# episode open_trashcan subsets, then runs open-loop eval for each
# checkpoint against the same fixed held-out episode set used for the
# ACT/SmolVLA sweep (see scripts/run_sample_size_sweep.sh /
# outputs/sample_size_sweep/held_out_episodes.json).
#
# Step budget is 20,000 -- NOT the 50,000 ACT/SmolVLA trained at. pi0 (even
# with a frozen backbone) runs ~4.4x slower per step than SmolVLA on this
# hardware -- 50k steps x 4 sizes would take ~44h sequential vs ~17.5h at
# 20k. This is a genuine cross-policy budget difference, not an oversight;
# flag it wherever these numbers are compared to ACT/SmolVLA's.
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=20000
BATCH_SIZE=8
SAVE_FREQ=5000
LOG_DIR=outputs/sample_size_sweep
mkdir -p "$LOG_DIR"

PI0_RECIPE="--policy.dtype=bfloat16 --policy.train_expert_only=true --policy.freeze_vision_encoder=true --policy.gradient_checkpointing=true"

# Same fixed held-out episodes as the ACT/SmolVLA sweep (original
# open_trashcan indices) -- disjoint from the 50-episode subset (and
# therefore 10/25 too).
HELD_OUT="15 20 23 32 43 48 55 60 69 73 75 83 87 93 96"

SIZES="10 25 50 100"

for size in $SIZES; do
    if [ "$size" = "100" ]; then
        dataset_dir="open_trashcan"
    else
        dataset_dir="open_trashcan_${size}"
    fi

    name="pi0_open_trashcan_${size}"
    echo "=== [$(date)] Training $name on $dataset_dir ==="
    bash scripts/train_pi0.sh \
        --dataset-dir "$dataset_dir" \
        --output-dir "outputs/$name" \
        --job-name "$name" \
        --batch-size $BATCH_SIZE \
        --steps $STEPS \
        --save-freq $SAVE_FREQ \
        --gpu 3 \
        --extra-args "$PI0_RECIPE" \
        > "$LOG_DIR/${name}.log" 2>&1

    echo "=== [$(date)] Evaluating $name ==="
    CUDA_VISIBLE_DEVICES=3 uv run --project ../lerobot-pi0 python eval_pi0_open_trashcan.py \
        --checkpoint "outputs/$name/checkpoints/last/pretrained_model" \
        --dataset-name open_trashcan \
        --episodes $HELD_OUT \
        --output "$LOG_DIR/${name}_eval.json" \
        > "$LOG_DIR/${name}_eval.log" 2>&1
done

echo "=== [$(date)] pi0 sweep complete ==="

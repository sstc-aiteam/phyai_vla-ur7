#!/usr/bin/env python3
"""
Offline eval for a finetuned GR00T N1.7 checkpoint
====================================================
Computes the same forward-pass loss `lerobot-train` uses for its own
`--eval_steps` held-out evaluation (see `lerobot.scripts.lerobot_train`'s
`is_eval_step` block), standalone, against an already-saved checkpoint.

CAVEAT: unless the checkpoint was trained with `--dataset.eval_split` > 0,
every episode in --dataset-dir was seen during training, so this measures
in-sample fit, not generalization. See scripts/train_groot.sh's
--extra-args passthrough to add --dataset.eval_split/--eval_steps for a
genuine held-out number.

Run inside the GR00T venv (via scripts/eval_groot.sh, or directly):
    uv run --project ../lerobot-groot python scripts/eval_groot.py \\
        --checkpoint outputs/groot_open_trashcan_50/checkpoints/last/pretrained_model \\
        --dataset-dir open_trashcan_50
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_train import _preprocess_dataset_batch


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a saved pretrained_model dir (config.json + model.safetensors)")
    parser.add_argument("--dataset-dir", type=str, default="open_trashcan_50",
                        help="Local LeRobot v3 dataset directory")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def main():
    args = build_arg_parser().parse_args()

    checkpoint = Path(args.checkpoint)
    if not (checkpoint / "config.json").is_file():
        sys.exit(f"[ERROR] {checkpoint} does not look like a saved policy (missing config.json)")

    dataset_dir = Path(args.dataset_dir)
    if not (dataset_dir / "meta" / "info.json").is_file():
        sys.exit(f"[ERROR] {dataset_dir} does not look like a LeRobot dataset (missing meta/info.json)")

    print(f"==> Loading dataset: {dataset_dir}")
    dataset = LeRobotDataset(repo_id=f"local/{dataset_dir.name}", root=dataset_dir)

    print(f"==> Loading policy config + weights from: {checkpoint}")
    policy_cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    policy_cfg.pretrained_path = str(checkpoint)
    policy_cfg.device = args.device

    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    policy.eval()

    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"==> Running forward-pass loss over {dataset.num_frames} frames "
          f"({dataset.num_episodes} episodes), batch_size={args.batch_size} ...")

    losses = []
    episode_losses = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            episode_indices = batch["episode_index"].clone()
            batch = _preprocess_dataset_batch(batch, dataset.meta.camera_keys, {}, preprocessor)
            loss, _ = policy(batch)
            loss_value = loss.item()
            losses.append(loss_value)
            for ep in episode_indices.unique().tolist():
                episode_losses[ep].append(loss_value)

    if not losses:
        sys.exit("[ERROR] No batches were evaluated -- empty dataset?")

    mean_loss = sum(losses) / len(losses)
    print("\n==> Results")
    print(f"    Batches:     {len(losses)}")
    print(f"    Mean loss:   {mean_loss:.4f}")
    print(f"    Min / Max:   {min(losses):.4f} / {max(losses):.4f}")

    print("\n    Per-episode mean loss:")
    for ep in sorted(episode_losses):
        ep_losses = episode_losses[ep]
        print(f"      episode {ep:>3}: {sum(ep_losses) / len(ep_losses):.4f}  (n={len(ep_losses)} batches)")

    print(
        "\n==> CAVEAT: this checkpoint was trained with --dataset.eval_split=0.0 "
        "(all episodes seen during training), so this is an IN-SAMPLE loss -- it "
        "checks that the model fits its own training data, not that it generalizes. "
        "For a genuine held-out number, retrain with e.g.\n"
        '    scripts/train_groot.sh --dataset-dir <name> '
        '--extra-args "--dataset.eval_split=0.2 --eval_steps=250"\n'
        "and read the eval_loss lerobot-train logs periodically during that run."
    )


if __name__ == "__main__":
    main()

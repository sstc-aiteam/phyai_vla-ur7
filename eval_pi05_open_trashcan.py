#!/usr/bin/env python
"""Open-loop evaluation of a trained pi05 policy against recorded ground-truth
actions in the open_trashcan dataset.

Mirrors `eval_pi0_open_trashcan.py` (same open-loop replay/MAE methodology,
same held-out-episode protocol, same isolated-env requirement) but loads a
`PI05Policy` checkpoint. Must run inside the sibling pi0/pi05 venv set up by
`scripts/setup_pi0_env.sh` (this repo's own env can't import
lerobot.policies.pi05 -- same transformers-version reason as pi0):

    uv run --project ../lerobot-pi0 python eval_pi05_open_trashcan.py \\
        --checkpoint outputs/pi05_open_trashcan_100/checkpoints/last/pretrained_model \\
        --dataset-name open_trashcan \\
        --episodes 15 20 23 32 43 48 55 60 69 73 75 83 87 93 96 \\
        --output outputs/sample_size_sweep/pi05_open_trashcan_100_eval.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
    "gripper",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/pi05_open_trashcan/checkpoints/last/pretrained_model")
    parser.add_argument("--dataset-name", default="open_trashcan")
    parser.add_argument("--episodes", type=int, nargs="+", default=None, help="Episode indices to evaluate (default: evenly spaced sample)")
    parser.add_argument("--num-episodes", type=int, default=6, help="How many episodes to sample if --episodes not given")
    parser.add_argument("--output", default="outputs/pi05_open_trashcan_eval.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    print(f"Loading policy from {ckpt}")
    policy = PI05Policy.from_pretrained(ckpt, device=args.device)
    device_override = {"device_processor": {"device": policy.config.device}}

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=ckpt, dataset_stats=None,
        preprocessor_overrides=device_override, postprocessor_overrides=device_override,
    )

    root = Path(args.dataset_name)
    dataset = LeRobotDataset(args.dataset_name, root=str(root), video_backend="pyav")
    total_episodes = dataset.meta.total_episodes

    if args.episodes is not None:
        episodes = args.episodes
    else:
        n = min(args.num_episodes, total_episodes)
        episodes = sorted(set(np.linspace(0, total_episodes - 1, n).astype(int).tolist()))

    print(f"Evaluating episodes: {episodes} (of {total_episodes} total)")

    results = {"joint_names": JOINT_NAMES, "fps": dataset.meta.fps, "episodes": []}

    for ep_idx in episodes:
        ep_meta = dataset.meta.episodes[ep_idx]
        from_idx = ep_meta["dataset_from_index"]
        to_idx = ep_meta["dataset_to_index"]
        task = ep_meta["tasks"][0] if ep_meta.get("tasks") else ""

        policy.reset()
        preds, gts, timestamps = [], [], []

        for i in range(from_idx, to_idx):
            item = dataset[i]
            obs = {
                "observation.state": item["observation.state"],
                "observation.images.cam_wrist": item["observation.images.cam_wrist"],
                "task": task,
            }
            obs = preprocessor(obs)
            with torch.inference_mode():
                action = policy.select_action(obs)
            action = postprocessor(action).squeeze(0).cpu().numpy()
            pred = action[:7].tolist()
            gt = item["action"].numpy().tolist()

            preds.append(pred)
            gts.append(gt)
            timestamps.append(float(item["timestamp"]))

        preds_arr = np.array(preds)
        gts_arr = np.array(gts)
        mae_per_joint = np.abs(preds_arr - gts_arr).mean(axis=0).tolist()
        arm_range = (gts_arr[:, :6].max(axis=0) - gts_arr[:, :6].min(axis=0)).max()
        is_static = bool(arm_range < 0.01)

        print(
            f"  episode {ep_idx}: {len(preds)} frames, "
            f"{'STATIC' if is_static else 'moving'}, "
            f"MAE per joint = {[f'{m:.4f}' for m in mae_per_joint]}"
        )

        results["episodes"].append(
            {
                "episode_index": ep_idx,
                "task": task,
                "num_frames": len(preds),
                "is_static": is_static,
                "timestamps": timestamps,
                "predicted": preds,
                "ground_truth": gts,
                "mae_per_joint": mae_per_joint,
            }
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

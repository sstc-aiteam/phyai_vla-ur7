#!/usr/bin/env python
"""Open-loop evaluation of the trained ACT policy against recorded ground-truth
actions in the open_trashcan dataset.

For each selected episode, replays the recorded observations frame-by-frame
through the trained policy (exactly as it would run at inference time, with
the same action-queue behavior as `select_action`), and records the
predicted action alongside the ground-truth action actually taken. Dumps the
comparison to a JSON file for visualization.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

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
    parser.add_argument("--checkpoint", default="outputs/act_open_trashcan/checkpoints/last/pretrained_model")
    parser.add_argument("--dataset-name", default="open_trashcan")
    parser.add_argument("--episodes", type=int, nargs="+", default=None, help="Episode indices to evaluate (default: evenly spaced sample)")
    parser.add_argument("--num-episodes", type=int, default=6, help="How many episodes to sample if --episodes not given")
    parser.add_argument("--output", default="outputs/act_open_trashcan_eval.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    print(f"Loading policy from {ckpt}")
    policy = ACTPolicy.from_pretrained(ckpt, device=args.device)
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
            action = postprocessor(action)
            pred = action.squeeze(0).cpu().numpy().tolist()
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

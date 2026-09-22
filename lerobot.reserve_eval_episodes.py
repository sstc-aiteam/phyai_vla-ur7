#!/usr/bin/env python3
"""
LeRobot Eval-Episode Reservation
=================================
Splits a local LeRobot v3 dataset into a held-out eval set (untouched by
any training) and a training pool (everything else), by *original episode
index* -- e.g. "reserve episodes 100-129 of open_trashcan as unseen eval
data, train on the rest."

This is the first step ahead of lerobot.partition_dataset.py's sample-size
ablation: run this once to carve out the eval episodes, then partition the
resulting training-pool dataset into nested 10/25/50/100/... subsets. The
eval episodes themselves are never materialized as a separate directory --
scripts already index eval episodes straight out of the source dataset via
`--episodes <indices...>` (see eval_act_open_trashcan.py and friends), so
there's nothing to build for them beyond recording which indices they are.

Usage:
    python lerobot.reserve_eval_episodes.py open_trashcan --eval-episodes 100-129

Writes `<dataset>_trainpool/` (a full standalone LeRobot v3 dataset,
re-chunked and re-indexed by the underlying
`lerobot.datasets.dataset_tools.delete_episodes`) alongside a
`reservation_manifest.json` recording exactly which original episode
indices were reserved for eval vs kept for training, for reproducibility.
"""

import argparse
import json
import sys
from pathlib import Path

from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_episode_spec(spec: list[str]) -> list[int]:
    """Parse ["100-129"] or ["100", "101", ..., "129"] into a sorted list of ints."""
    indices: set[int] = set()
    for token in spec:
        if "-" in token:
            lo, hi = token.split("-", 1)
            indices.update(range(int(lo), int(hi) + 1))
        else:
            indices.add(int(token))
    return sorted(indices)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=str, help="Source LeRobot v3 dataset directory (e.g. open_trashcan)")
    parser.add_argument("--eval-episodes", type=str, nargs="+", required=True,
                        help="Original episode indices to reserve for eval, as a range (e.g. 100-129) "
                             "and/or individual indices (e.g. 100 101 129)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output training-pool dataset dir (default: '<dataset>_trainpool')")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory")
    return parser


def main():
    args = build_arg_parser().parse_args()

    source_dir = Path(args.dataset)
    if not (source_dir / "meta" / "info.json").is_file():
        sys.exit(f"[ERROR] {source_dir} does not look like a LeRobot dataset (missing meta/info.json)")

    dataset = LeRobotDataset(source_dir.name, root=source_dir)
    total_episodes = dataset.meta.total_episodes
    print(f"Source: {source_dir}/ -- {total_episodes} episodes, {dataset.meta.total_frames} frames "
          f"(fps={dataset.meta.fps}, robot_type={dataset.meta.robot_type})")

    eval_episodes = parse_episode_spec(args.eval_episodes)
    out_of_range = [i for i in eval_episodes if i < 0 or i >= total_episodes]
    if out_of_range:
        sys.exit(f"[ERROR] Eval episode indices out of range [0, {total_episodes - 1}]: {out_of_range}")

    train_episodes = [i for i in range(total_episodes) if i not in eval_episodes]
    if not train_episodes:
        sys.exit("[ERROR] Nothing left to train on -- all episodes were reserved for eval.")

    output_dir = Path(args.output or f"{source_dir.name}_trainpool")
    if output_dir.exists():
        if not args.force:
            sys.exit(f"[ERROR] {output_dir}/ already exists (use --force to overwrite)")
        import shutil
        shutil.rmtree(output_dir)

    print(f"\nReserving {len(eval_episodes)} episodes for eval: {eval_episodes}")
    print(f"Training pool: {len(train_episodes)} episodes -> {output_dir}/ ...")
    delete_episodes(dataset, episode_indices=eval_episodes, output_dir=output_dir, repo_id=output_dir.name)

    manifest = {
        "source_dataset": str(source_dir),
        "source_total_episodes": total_episodes,
        "eval_episode_indices": eval_episodes,
        "train_episode_indices": train_episodes,
        "train_pool_dir": str(output_dir),
    }
    (output_dir / "reservation_manifest.json").write_text(json.dumps(manifest, indent=2))

    info_path = output_dir / "meta" / "info.json"
    frames = json.loads(info_path.read_text())["total_frames"] if info_path.is_file() else "?"
    print(f"\nDone. {output_dir}/  ({len(train_episodes)} episodes, {frames} frames)")
    print(f"Eval episodes (never touched by training): {eval_episodes}")
    print("Evaluate against them directly on the source dataset, e.g.:")
    print(f"  python eval_act_open_trashcan.py --checkpoint <ckpt> --dataset-name {source_dir.name} "
          f"--episodes {' '.join(map(str, eval_episodes))} --output <out>.json")


if __name__ == "__main__":
    main()

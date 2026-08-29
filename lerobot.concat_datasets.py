#!/usr/bin/env python3
"""
LeRobot Dataset Concatenator
============================
Merges two or more local LeRobot v3 dataset directories into a single
combined dataset, ready for training on all of them together.

Requirements:
    pip install -r requirements.txt

Usage:
    python lerobot.concat_datasets.py open_trashcan_19 open_trashcan_49 \\
        --output open_trashcan_19_49

Source datasets must share the same fps, robot_type, and features schema
(e.g. all recorded with the same robot/cameras) -- this is enforced by
the underlying `lerobot.datasets.aggregate.aggregate_datasets`, which does
the real work here: it copies/re-chunks the parquet + video files, unions
the per-dataset task tables, re-indexes episodes/frames, and recomputes
stats. This script is just a thin CLI around it.

The output directory must not already exist -- point --output at a fresh
name to avoid ambiguity about what got merged into what.
"""

import argparse
import json
import sys
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Concatenate LeRobot v3 datasets into one")
    parser.add_argument("datasets", type=str, nargs="+",
                        help="Paths to the source dataset directories to concatenate, in order")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Path for the new combined dataset directory (must not exist yet)")
    return parser


def main():
    args = build_arg_parser().parse_args()

    dataset_dirs = [Path(d) for d in args.datasets]
    for d in dataset_dirs:
        if not (d / "meta" / "info.json").is_file():
            sys.exit(f"[ERROR] {d} does not look like a LeRobot dataset (missing meta/info.json)")

    output_dir = Path(args.output)
    if output_dir.exists():
        sys.exit(f"[ERROR] --output {output_dir} already exists; choose a new destination")

    repo_ids = [d.name for d in dataset_dirs]
    print(f"Concatenating {len(dataset_dirs)} datasets into {output_dir}/ ...")
    for d in dataset_dirs:
        print(f"  - {d}")

    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=output_dir.name,
        roots=dataset_dirs,
        aggr_root=output_dir,
    )

    info = json.loads((output_dir / "meta" / "info.json").read_text())
    print(f"\nDone! {output_dir}/ has {info['total_episodes']} episodes, "
          f"{info['total_frames']} frames (fps={info['fps']}, robot_type={info['robot_type']}).")
    print("\nNext steps:")
    print(f"  1. Inspect:  lerobot-dataset-viz --repo-id {output_dir.name} --root {output_dir} --mode local")
    print("  2. Train:    python -m lerobot.train \\")
    print(f"                 --dataset.repo_id={output_dir.name} \\")
    print("                 --policy.type=act \\")
    print(f"                 --output_dir=outputs/act_{output_dir.name}")


if __name__ == "__main__":
    main()

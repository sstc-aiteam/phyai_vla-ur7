#!/usr/bin/env python3
"""
LeRobot Dataset Partitioner
===========================
Splits a local LeRobot v3 dataset into smaller, *nested* subsets by
episode count, for sample-size ablations (e.g. "how does ACT/SmolVLA
performance scale with 10 vs 25 vs 50 vs 100 demonstrations?").

Requirements:
    pip install -r requirements.txt

Usage:
    python lerobot.partition_dataset.py open_trashcan --sizes 10 25 50

Subsets are nested: episodes are shuffled once with a fixed --seed, and
each subset of size N keeps the first N episodes of that shuffled order.
So subset(10) ⊆ subset(25) ⊆ subset(50) ⊆ ... ⊆ the full dataset --
sample-size effects aren't confounded with *which* episodes were picked
or with any drift over the recording session (e.g. technique improving
over time), which taking episodes 0..N-1 in recording order would risk.

A size equal to (or exceeding) the source dataset's total episode count
is skipped -- the source dataset already *is* that condition, so there's
no need to materialize an identical copy of it.

Each output subset is written to `<dataset>_<size>/` (a full standalone
LeRobot v3 dataset directory, re-chunked and re-indexed by the underlying
`lerobot.datasets.dataset_tools.delete_episodes`) alongside a
`partition_manifest.json` recording exactly which original episode
indices it contains, for reproducibility.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=str, help="Source LeRobot v3 dataset directory (e.g. open_trashcan)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 25, 50, 100],
                        help="Episode counts to partition into, as nested subsets (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the one-time episode shuffle that defines the nesting order (default: %(default)s)")
    parser.add_argument("--output-prefix", type=str, default=None,
                        help="Prefix for output dataset dirs, '<prefix>_<size>' (default: the source dataset's name)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directories")
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

    sizes = sorted(set(args.sizes))
    oversized = [n for n in sizes if n >= total_episodes]
    sizes = [n for n in sizes if n < total_episodes]
    if oversized:
        print(f"Skipping size(s) {oversized}: >= the source's {total_episodes} episodes "
              f"-- use {source_dir}/ itself for that condition.")
    if not sizes:
        sys.exit("[ERROR] Nothing to do: no requested size is smaller than the source dataset.")

    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(total_episodes).tolist()
    print(f"Shuffle order (seed={args.seed}): {shuffled}")

    prefix = args.output_prefix or source_dir.name
    results = []
    for n in sizes:
        output_dir = Path(f"{prefix}_{n}")
        keep = sorted(shuffled[:n])
        delete = [i for i in range(total_episodes) if i not in keep]

        if output_dir.exists():
            if not args.force:
                print(f"\n[SKIP] {output_dir}/ already exists (use --force to overwrite)")
                results.append((n, output_dir, keep))
                continue
            import shutil
            shutil.rmtree(output_dir)

        print(f"\nPartitioning {n} episodes -> {output_dir}/ ...")
        delete_episodes(dataset, episode_indices=delete, output_dir=output_dir, repo_id=output_dir.name)

        manifest = {
            "source_dataset": str(source_dir),
            "source_total_episodes": total_episodes,
            "size": n,
            "seed": args.seed,
            "kept_source_episode_indices": keep,
        }
        (output_dir / "partition_manifest.json").write_text(json.dumps(manifest, indent=2))
        results.append((n, output_dir, keep))

    print("\nDone! Subsets:")
    for n, output_dir, keep in results:
        info_path = output_dir / "meta" / "info.json"
        frames = json.loads(info_path.read_text())["total_frames"] if info_path.is_file() else "?"
        print(f"  - {output_dir}/  ({n} episodes, {frames} frames)  source episodes: {keep}")


if __name__ == "__main__":
    main()

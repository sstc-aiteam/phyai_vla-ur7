#!/usr/bin/env python3
"""
UR7e Dataset Motion Verifier
============================
Checks every episode of a recorded LeRobot v3 dataset for "frozen" arm
motion -- i.e. the 6 UR arm joints barely move across the whole episode,
which usually means the RTDE receive feed got stuck for that episode
rather than the human genuinely holding the arm still (see
`ur7e_recorder.session.FROZEN_EPISODE_RANGE_RAD` and its docstring,
`_episode_frozen`, for the live version of this same check run during
recording). This script runs the identical check after the fact, over an
entire saved dataset.

Usage:
    python lerobot.verify_dataset.py --dataset-name open_trashcan
    python lerobot.verify_dataset.py --dataset-name open_trashcan --threshold 0.02

Exits 0 if every episode clears the threshold, 1 if any is flagged
frozen -- so it can gate a subsequent training run.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from ur7e_recorder.dump import load_dataset_joint_states
from ur7e_recorder.session import FROZEN_EPISODE_RANGE_RAD

# The 6 UR arm joints carry the motion signal; `gripper` is excluded to
# match `_episode_frozen`/`eval_act_open_trashcan.py`'s definition of
# "arm motion".
ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-name", default="open_trashcan", help="Local dataset directory (default: %(default)s)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=FROZEN_EPISODE_RANGE_RAD,
        help="Minimum required arm-joint range in radians, below which an episode is flagged "
        "frozen (default: %(default)s rad, same as the live recording-time check)",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    dataset_dir = Path(args.dataset_name)

    df = load_dataset_joint_states(dataset_dir)
    state_cols = [f"observation.state.{name}" for name in ARM_JOINT_NAMES]

    rows = []
    for episode_index, group in df.groupby("episode_index"):
        values = group[state_cols].to_numpy()
        arm_range = float((values.max(axis=0) - values.min(axis=0)).max())
        rows.append((episode_index, len(group), arm_range))

    rows.sort(key=lambda row: row[2])

    frozen = [row for row in rows if row[2] < args.threshold]

    print(f"{'episode':>8}  {'frames':>7}  {'arm_range_rad':>14}  status")
    for episode_index, num_frames, arm_range in rows:
        status = "FROZEN" if arm_range < args.threshold else "ok"
        print(f"{episode_index:>8}  {num_frames:>7}  {arm_range:>14.4f}  {status}")

    print()
    if frozen:
        flagged = ", ".join(str(row[0]) for row in frozen)
        print(f"[FAIL] {len(frozen)}/{len(rows)} episode(s) below {args.threshold} rad threshold: {flagged}")
        sys.exit(1)

    ranges = np.array([row[2] for row in rows])
    print(
        f"[OK] All {len(rows)} episodes clear the {args.threshold} rad threshold "
        f"(min={ranges.min():.4f}, median={np.median(ranges):.4f}, max={ranges.max():.4f})"
    )


if __name__ == "__main__":
    main()

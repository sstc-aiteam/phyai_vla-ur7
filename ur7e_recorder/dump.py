"""Dumps recorded joint state (and action) data from a LeRobot v2 dataset
to stdout or a CSV file, for inspecting episodes without a training run.
"""

import json
from pathlib import Path

import pandas as pd

STATE_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3", "gripper",
]


def load_dataset_joint_states(dataset_dir: Path, episode_index: int | None = None) -> pd.DataFrame:
    """Joint state + action for every step of a dataset, or of one episode.

    Columns: episode_index, frame_index, timestamp, one
    `observation.state.<joint_name>` and `action.<joint_name>` per joint.
    """
    with open(dataset_dir / "meta" / "info.json") as f:
        total_episodes = json.load(f)["total_episodes"]

    episode_indices = [episode_index] if episode_index is not None else range(total_episodes)

    episode_frames = []
    for ep_idx in episode_indices:
        parquet_path = dataset_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"No such episode: {parquet_path}")
        episode_frames.append(pd.read_parquet(parquet_path).sort_values("frame_index"))

    df = pd.concat(episode_frames, ignore_index=True)

    rename = {f"observation.state.{j}": f"observation.state.{name}" for j, name in enumerate(STATE_NAMES)}
    rename |= {f"action.{j}": f"action.{name}" for j, name in enumerate(STATE_NAMES)}
    df = df.rename(columns=rename)

    cols = (["episode_index", "frame_index", "timestamp"]
            + [f"observation.state.{name}" for name in STATE_NAMES]
            + [f"action.{name}" for name in STATE_NAMES])
    return df[cols]

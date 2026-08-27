"""Dumps recorded joint state (and action) data from a LeRobot v3 dataset
to stdout or a CSV file, for inspecting episodes without a training run.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


def load_dataset_joint_states(dataset_dir: Path, episode_index: int | None = None) -> pd.DataFrame:
    """Joint state + action for every step of a dataset, or of one episode.

    Columns: episode_index, frame_index, timestamp, one
    `observation.state.<joint_name>` and `action.<joint_name>` per joint.
    """
    meta = LeRobotDatasetMetadata(repo_id=dataset_dir.name, root=dataset_dir)
    state_names = meta.features["observation.state"]["names"]
    action_names = meta.features["action"]["names"]

    episode_indices = [episode_index] if episode_index is not None else list(range(meta.total_episodes))
    for ep_idx in episode_indices:
        if ep_idx >= meta.total_episodes:
            raise FileNotFoundError(f"No such episode: {ep_idx} (dataset has {meta.total_episodes})")

    # v3 datasets pack multiple episodes into one Parquet file, so read
    # each backing file once even if several requested episodes share it.
    data_paths = {meta.root / meta.get_data_file_path(ep_idx) for ep_idx in episode_indices}
    df = pd.concat([pd.read_parquet(p) for p in data_paths], ignore_index=True)
    df = df[df["episode_index"].isin(episode_indices)].sort_values(["episode_index", "frame_index"])
    df = df.reset_index(drop=True)

    obs_state = np.stack(df["observation.state"].values)
    actions = np.stack(df["action"].values)
    for j, name in enumerate(state_names):
        df[f"observation.state.{name}"] = obs_state[:, j]
    for j, name in enumerate(action_names):
        df[f"action.{name}"] = actions[:, j]

    cols = (["episode_index", "frame_index", "timestamp"]
            + [f"observation.state.{name}" for name in state_names]
            + [f"action.{name}" for name in action_names])
    return df[cols]

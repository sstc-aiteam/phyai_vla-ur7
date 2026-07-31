"""LeRobot v2 dataset writer (Parquet + MP4)."""

import json
from pathlib import Path

import cv2
import pandas as pd

from .config import RecorderConfig
from .episode import EpisodeRecorder

STATE_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3", "gripper",
]


class LeRobotDatasetWriter:
    """Writes episodes to LeRobot v2 dataset format on disk."""

    def __init__(self, config: RecorderConfig, camera_names: list):
        self.dataset_dir = Path(config.dataset_name)
        self.fps = config.fps
        self.task = config.task
        self.camera_names = camera_names
        self.episode_count = 0

        self._init_directory_layout()
        self._init_tasks_file()
        self.episodes_file = self.dataset_dir / "meta" / "episodes.jsonl"
        open(self.episodes_file, "w").close()

    def _init_directory_layout(self):
        (self.dataset_dir / "meta").mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        for cam in self.camera_names:
            (self.dataset_dir / "videos" / "chunk-000"
             / f"observation.images.{cam}").mkdir(parents=True, exist_ok=True)

    def _init_tasks_file(self):
        self.tasks_file = self.dataset_dir / "meta" / "tasks.jsonl"
        with open(self.tasks_file, "w") as f:
            f.write(json.dumps({"task_index": 0, "task": self.task}) + "\n")

    def save_episode(self, episode: EpisodeRecorder) -> bool:
        """Save a single episode to parquet + mp4 files."""
        if episode.num_steps < 2:
            print("[WARN] Episode too short, skipping.")
            return False

        ep_idx = self.episode_count
        self._write_parquet(episode, ep_idx)
        self._write_videos(episode, ep_idx)
        self._append_episode_record(ep_idx, episode.num_steps)

        self.episode_count += 1
        print(f"[SAVED] Episode {ep_idx} — {episode.num_steps} steps")
        return True

    def _write_parquet(self, episode: EpisodeRecorder, ep_idx: int):
        n_steps = episode.num_steps
        # For actions, shift states by 1 (action = next state)
        actions = episode.states[1:] + [episode.states[-1]]

        rows = []
        for i in range(n_steps):
            row = {
                "episode_index": ep_idx,
                "frame_index": i,
                "timestamp": episode.timestamps[i] - episode.timestamps[0],
                "task_index": 0,
                "next.done": i == n_steps - 1,
            }
            for j, value in enumerate(episode.states[i]):
                row[f"observation.state.{j}"] = value
            for j, value in enumerate(actions[i]):
                row[f"action.{j}"] = value
            rows.append(row)

        df = pd.DataFrame(rows)
        parquet_path = (self.dataset_dir / "data" / "chunk-000"
                        / f"episode_{ep_idx:06d}.parquet")
        df.to_parquet(parquet_path, index=False)

    def _write_videos(self, episode: EpisodeRecorder, ep_idx: int):
        for cam_name in self.camera_names:
            frames = episode.frames.get(cam_name)
            if not frames:
                continue

            video_path = (self.dataset_dir / "videos" / "chunk-000"
                          / f"observation.images.{cam_name}"
                          / f"episode_{ep_idx:06d}.mp4")

            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (w, h))
            for frame in frames:
                writer.write(frame)
            writer.release()

    def _append_episode_record(self, ep_idx: int, length: int):
        with open(self.episodes_file, "a") as f:
            f.write(json.dumps({
                "episode_index": ep_idx,
                "length": length,
                "task_index": 0,
                "task": self.task,
            }) + "\n")

    def write_info(self):
        """Write the final info.json metadata file."""
        state_dim = len(STATE_NAMES)
        info = {
            "codebase_version": "v2.0",
            "robot_type": "ur7e",
            "fps": self.fps,
            "total_episodes": self.episode_count,
            "features": {
                "observation.state": {
                    "dtype": "float64",
                    "shape": [state_dim],
                    "names": STATE_NAMES,
                },
                "action": {
                    "dtype": "float64",
                    "shape": [state_dim],
                    "names": STATE_NAMES,
                },
                **{
                    f"observation.images.{cam}": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                        "names": None,
                    }
                    for cam in self.camera_names
                },
            },
        }
        with open(self.dataset_dir / "meta" / "info.json", "w") as f:
            json.dump(info, f, indent=2)
        print(f"[INFO] Dataset saved: {self.episode_count} episodes → {self.dataset_dir}")

"""LeRobot v3 dataset writer.

Thin wrapper around `lerobot.datasets.lerobot_dataset.LeRobotDataset` --
the official dataset class from the `lerobot` package -- so recordings
come out as spec-compliant LeRobot v3.0 datasets (chunked Parquet +
H.264 MP4, `meta/episodes/` + `meta/stats.json`) instead of a hand-rolled
approximation of the format.
"""

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .config import RecorderConfig

STATE_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3", "gripper",
]

# PNG frames for the episode currently being recorded are written to disk
# by background threads (one camera's worth of writes shouldn't stall the
# freedrive control loop waiting on the other camera's disk I/O).
IMAGE_WRITER_THREADS_PER_CAMERA = 4


class LeRobotDatasetWriter:
    """Records UR7e episodes directly into a LeRobot v3.0 dataset on disk."""

    def __init__(self, config: RecorderConfig, camera_shapes: dict[str, tuple[int, int, int]]):
        """camera_shapes: {name: (height, width, channels)} for every
        enabled camera, measured from a real captured frame -- the
        dataset's video features are fixed at creation time, so this must
        match what `add_step` will actually pass in.
        """
        self.task = config.task
        self.robot_type = config.robot_type
        self.camera_names = list(camera_shapes)

        if config.resume:
            self.ds = LeRobotDataset(
                repo_id=config.dataset_name,
                root=config.dataset_name,
                vcodec="h264",
            )
            self._validate_resume(config, camera_shapes)
            if self.camera_names:
                self.ds.start_image_writer(
                    num_processes=0,
                    num_threads=IMAGE_WRITER_THREADS_PER_CAMERA * len(self.camera_names),
                )
            self.episode_count = self.ds.meta.total_episodes
            return

        features = {
            "observation.state": {
                "dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES,
            },
            "action": {
                "dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES,
            },
            **{
                f"observation.images.{cam}": {
                    "dtype": "video", "shape": shape, "names": ["height", "width", "channels"],
                }
                for cam, shape in camera_shapes.items()
            },
        }

        self.ds = LeRobotDataset.create(
            repo_id=config.dataset_name,
            fps=config.fps,
            features=features,
            root=config.dataset_name,
            robot_type=self.robot_type,
            use_videos=bool(self.camera_names),
            vcodec="h264",
            image_writer_processes=0,
            image_writer_threads=IMAGE_WRITER_THREADS_PER_CAMERA * len(self.camera_names),
        )
        self.episode_count = 0

    def _validate_resume(self, config: RecorderConfig, camera_shapes: dict[str, tuple[int, int, int]]):
        """Guard against silently corrupting an existing dataset: a
        mismatched fps skews every new frame's timestamp, and the video
        features' (height, width, channels) are fixed at creation time,
        so a different camera set/resolution can't be appended to it."""
        errors = []
        if self.ds.meta.fps != config.fps:
            errors.append(f"fps={config.fps} does not match the existing dataset's fps={self.ds.meta.fps}")
        if self.ds.meta.robot_type != self.robot_type:
            errors.append(f"robot_type={self.robot_type!r} does not match the existing dataset's "
                           f"robot_type={self.ds.meta.robot_type!r}")

        existing_cams = {
            key.removeprefix("observation.images."): tuple(ft["shape"])
            for key, ft in self.ds.features.items()
            if key.startswith("observation.images.")
        }
        new_cams = {cam: tuple(shape) for cam, shape in camera_shapes.items()}
        if existing_cams != new_cams:
            errors.append(f"cameras {new_cams} do not match the existing dataset's cameras {existing_cams}")

        if errors:
            raise ValueError(
                f"Cannot resume {config.dataset_name!r}: " + "; ".join(errors)
            )

    @property
    def num_steps(self) -> int:
        """Steps buffered so far for the episode currently being recorded."""
        return self.ds.episode_buffer["size"] if self.ds.episode_buffer else 0

    def add_step(self, state: list, action: list, camera_frames: dict):
        """Buffer one step of the episode currently being recorded.
        `camera_frames` is {cam_name: BGR frame}, as produced by CameraManager.
        """
        frame = {
            "observation.state": np.asarray(state, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "task": self.task,
        }
        for cam_name, bgr_frame in camera_frames.items():
            frame[f"observation.images.{cam_name}"] = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        self.ds.add_frame(frame)

    def discard_episode(self):
        """Drop the episode currently being recorded without saving it."""
        self.ds.clear_episode_buffer()

    def save_episode(self) -> bool:
        """Encode and save the episode currently being recorded."""
        if self.num_steps < 2:
            print("[WARN] Episode too short, skipping.")
            self.discard_episode()
            return False

        ep_idx = self.episode_count
        n_steps = self.num_steps
        self.ds.save_episode()
        self.episode_count += 1
        print(f"[SAVED] Episode {ep_idx} — {n_steps} steps")
        return True

    def finalize(self):
        """Flush metadata to disk. Must be called once, after the last
        episode is saved -- without it the dataset's episode index isn't
        fully written and the dataset can't be loaded back."""
        self.ds.finalize()
        print(f"[INFO] Dataset saved: {self.episode_count} episodes → {self.ds.root}")

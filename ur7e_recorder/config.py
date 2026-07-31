"""Centralized run configuration.

`RecorderConfig` is the single source of truth for a recording run
(robot IP, dataset name, fps, task, cameras, ...). `cli.py` is the only
place that turns command-line flags into one; every other module takes
an already-built config (or a piece derived from it) instead of parsing
arguments or hard-coding defaults itself.
"""

from dataclasses import dataclass


#: Camera backends a CameraConfig can select.
CAMERA_BACKENDS = ("usb", "realsense")


@dataclass
class CameraConfig:
    index: int
    width: int = 640
    height: int = 480
    fps: int = 30
    backend: str = CAMERA_BACKENDS[0]  # "usb" (cv2.VideoCapture) or "realsense" (pyrealsense2)


@dataclass
class RecorderConfig:
    robot_ip: str = "192.168.50.76"
    dataset_name: str = "ur7e_pick_and_place"
    num_episodes: int = 5  #50
    fps: int = 5 #30
    task: str = "Pick up the object and place it in the bin"
    cam_overhead: int = 0
    cam_overhead_backend: str = CAMERA_BACKENDS[0]
    cam_wrist: int = -1
    cam_wrist_backend: str = CAMERA_BACKENDS[0]

    @property
    def camera_configs(self) -> dict:
        """{name: CameraConfig} for every enabled camera (index >= 0)."""
        configs = {}
        if self.cam_overhead >= 0:
            configs["cam_overhead"] = CameraConfig(
                index=self.cam_overhead, fps=self.fps, backend=self.cam_overhead_backend
            )
        if self.cam_wrist >= 0:
            configs["cam_wrist"] = CameraConfig(
                index=self.cam_wrist, fps=self.fps, backend=self.cam_wrist_backend
            )
        return configs

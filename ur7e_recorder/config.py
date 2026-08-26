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

#: Freedrive implementations a robot's controller generation supports:
#:   "e-series" -- freedriveMode()/endFreedriveMode() (UR7e and other e-Series arms)
#:   "cb3"      -- teachMode()/endTeachMode() (CB3 controllers, e.g. this UR5 on PolyScope 3.13)
CONTROLLER_GENERATIONS = ("e-series", "cb3")

#: Sentinel telling UR7eRobot to detect the generation itself via the
#: Dashboard Server instead of taking it from the caller/CLI.
#: See robot.detect_controller_generation.
CONTROLLER_AUTO = "auto"

#: Gripper hardware a run can record with.
GRIPPER_KINDS = ("robotiq", "none")


@dataclass
class CameraConfig:
    index: int
    width: int = 640
    height: int = 480
    fps: int = 5
    backend: str = CAMERA_BACKENDS[0]  # "usb" (cv2.VideoCapture) or "realsense" (pyrealsense2)


@dataclass
class RecorderConfig:
    robot_ip: str = "192.168.50.76"
    dataset_name: str = "ur7e_pick_and_place_dataset"
    num_episodes: int = 5                           #50
    fps: int = 10                                   #1
    task: str = "task description in here"
    cam_overhead: int = -1                          # -1 => disabled
    cam_overhead_backend: str = CAMERA_BACKENDS[0]
    cam_wrist: int = -1                             # -1 => disabled
    cam_wrist_backend: str = CAMERA_BACKENDS[1]
    controller: str = CONTROLLER_AUTO               # "auto", "e-series", or "cb3"
    gripper: str = GRIPPER_KINDS[0]                 # "robotiq" or "none"
    robot_type: str = "ur7e"                        # stored in the dataset's meta/info.json

    @property
    def camera_configs(self) -> dict:
        """{name: CameraConfig} for every enabled camera (index >= 0)."""
        configs = {}
        if self.cam_overhead >= 0:
            configs["cam_overhead"] = CameraConfig(
                index=self.cam_overhead, backend=self.cam_overhead_backend
            )
        if self.cam_wrist >= 0:
            configs["cam_wrist"] = CameraConfig(
                index=self.cam_wrist, backend=self.cam_wrist_backend
            )
        return configs


@dataclass
class ReplayConfig:
    robot_ip: str = "192.168.50.76"
    dataset_name: str = "ur7e_pick_and_place"
    episode: int = 0
    fps: int | None = None  # None => read from the dataset's meta/info.json
    start_speed: float = 0.3  # rad/s for the initial moveJ to the episode's first pose
    start_acceleration: float = 0.3  # rad/s^2 for the initial moveJ
    gripper: str = GRIPPER_KINDS[0]  # "robotiq" or "none" -- must match how the dataset was recorded

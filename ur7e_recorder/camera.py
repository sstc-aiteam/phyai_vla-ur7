"""Camera management for observation recording.

Each camera is a `Camera` (a single BGR frame source). Two backends are
provided, selected via `CameraConfig.backend`:
  - "usb":       a V4L2/UVC camera via cv2.VideoCapture
  - "realsense": an Intel RealSense camera's color stream via pyrealsense2

`CameraManager` treats every backend identically, so a session can mix
USB and RealSense cameras freely. Adding a new backend means adding a
`Camera` subclass and registering it in `_BACKENDS` below.
"""

from abc import ABC, abstractmethod

import cv2
import numpy as np

from .config import CameraConfig


class Camera(ABC):
    """A single video source that yields BGR frames."""

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Return the latest BGR frame, or None if one isn't available."""

    @abstractmethod
    def release(self):
        ...


class USBCamera(Camera):
    """A V4L2/UVC camera accessed through OpenCV."""

    def __init__(self, name: str, cfg: CameraConfig):
        self.cap = cv2.VideoCapture(cfg.index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        if not self.cap.isOpened():
            print(f"[WARN] Camera '{name}' (index {cfg.index}) failed to open.")
        else:
            print(f"[OK]   Camera '{name}' opened at index {cfg.index}.")

    def read(self) -> np.ndarray | None:
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()


class RealSenseCamera(Camera):
    """An Intel RealSense camera's aligned color stream via pyrealsense2.

    `cfg.index` selects which connected device to open (0 = first
    detected) when more than one RealSense camera is plugged in.
    """

    def __init__(self, name: str, cfg: CameraConfig):
        import pyrealsense2 as rs

        self.name = name
        self.pipeline = None
        self._warned_timeout = False

        devices = rs.context().devices
        if len(devices) == 0:
            print(f"[WARN] RealSense camera '{name}': no device connected.")
            return

        device_idx = cfg.index if cfg.index < len(devices) else 0
        if cfg.index >= len(devices):
            print(f"[WARN] RealSense camera '{name}': index {cfg.index} out of range "
                  f"({len(devices)} device(s) connected); using device 0.")
        serial = devices[device_idx].get_info(rs.camera_info.serial_number)

        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps)

        self.pipeline = rs.pipeline()
        self.pipeline.start(config)
        print(f"[OK]   RealSense camera '{name}' opened (serial {serial}).")

    def read(self) -> np.ndarray | None:
        if self.pipeline is None:
            return None
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=3000)
        except RuntimeError:
            # A dropped/late frame (e.g. a USB glitch) shouldn't kill the
            # whole recording session -- fall back like a bad USB read.
            if not self._warned_timeout:
                print(f"[WARN] RealSense camera '{self.name}': frame timed out; "
                      f"substituting blank frames until it recovers.")
                self._warned_timeout = True
            return None
        self._warned_timeout = False
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())

    def release(self):
        if self.pipeline is not None:
            self.pipeline.stop()


_BACKENDS = {
    "usb": USBCamera,
    "realsense": RealSenseCamera,
}


class CameraManager:
    """Manages one or more cameras, of any backend, for observation recording."""

    def __init__(self, camera_configs: dict[str, CameraConfig]):
        """camera_configs: {"cam_overhead": CameraConfig(index=0), ...}"""
        self.configs = camera_configs
        self.cameras: dict[str, Camera] = {}

        for name, cfg in camera_configs.items():
            camera_cls = _BACKENDS.get(cfg.backend)
            if camera_cls is None:
                raise ValueError(
                    f"Unknown camera backend '{cfg.backend}' for '{name}' "
                    f"(expected one of {list(_BACKENDS)})"
                )
            self.cameras[name] = camera_cls(name, cfg)

    @property
    def names(self) -> list:
        return list(self.configs.keys())

    def read_all(self) -> dict:
        """Read a frame from every camera. Returns {name: frame_bgr}."""
        frames = {}
        for name, camera in self.cameras.items():
            frame = camera.read()
            if frame is None:
                cfg = self.configs[name]
                frame = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)
            frames[name] = frame
        return frames

    def frame_shapes(self) -> dict:
        """{name: (height, width, channels)}, measured from a real
        captured frame -- a camera doesn't always honor its configured
        resolution, and the dataset's video features must match exactly
        what `read_all()` will actually produce."""
        return {name: frame.shape for name, frame in self.read_all().items()}

    def release(self):
        for camera in self.cameras.values():
            camera.release()

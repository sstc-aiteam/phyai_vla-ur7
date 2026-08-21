"""No-op gripper for arms with no gripper attached (e.g. a bare UR5 test
rig on PolyScope 3.13). Lets the rest of the recorder -- session, replay,
dataset writer -- run unmodified: `position` is always 0.0, so recorded
episodes just carry a constant gripper channel.
"""

from .base import Gripper


class NoGripper(Gripper):
    """Always-open, do-nothing gripper."""

    def __init__(self):
        self.is_open = True
        self.position = 0.0

    def open(self):
        pass

    def close(self):
        pass

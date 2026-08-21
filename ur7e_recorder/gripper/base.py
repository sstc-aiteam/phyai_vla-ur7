"""Gripper interface.

The rest of the recorder (session.py, replay.py) depends only on this
`Gripper` ABC, so new gripper hardware -- or no gripper at all -- can be
supported by adding a subclass here without touching camera, dataset, or
session code.
"""

from abc import ABC, abstractmethod


class Gripper(ABC):
    """Binary open/close gripper interface."""

    is_open: bool
    position: float  # 0.0 = open, 1.0 = closed

    @abstractmethod
    def open(self):
        ...

    @abstractmethod
    def close(self):
        ...

    def toggle(self):
        self.close() if self.is_open else self.open()

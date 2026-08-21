"""Robotiq gripper control.

`RobotiqGripper` drives a real Robotiq 2F gripper over UR RTDE using the
vendor URScript preamble (see ur_rtde docs:
https://sdurobotics.gitlab.io/ur_rtde/_static/robotiq_gripper_control.py),
adapted to satisfy the `Gripper` interface (see base.py).
"""

import time

from .base import Gripper
from .robotiq_preamble import ROBOTIQ_PREAMBLE


class RobotiqGripper(Gripper):
    """Robotiq 2F gripper control via the UR RTDE custom-script interface.

    Activates itself on construction (takes ~5s), so it's ready to use as
    soon as `RobotiqGripper(rtde_c)` returns.
    """

    def __init__(self, rtde_c):
        self.rtde_c = rtde_c
        self.is_open = True
        self.position = 0.0
        self.activate()

    def _call(self, script_name: str, script_function: str):
        """Send a URScript function wrapped in the Robotiq preamble."""
        return self.rtde_c.sendCustomScriptFunction(
            "ROBOTIQ_" + script_name,
            ROBOTIQ_PREAMBLE + script_function,
        )

    def activate(self) -> bool:
        """Activate the gripper. Takes ~5 seconds."""
        ret = self._call("ACTIVATE", "rq_activate()")
        time.sleep(5)  # HACK: activation has no completion signal to poll
        return ret

    def set_speed(self, speed: int) -> bool:
        """Set gripper speed as a percentage [0-100]."""
        return self._call("SET_SPEED", f"rq_set_speed_norm({speed})")

    def set_force(self, force: int) -> bool:
        """Set gripper force as a percentage [0-100]."""
        return self._call("SET_FORCE", f"rq_set_force_norm({force})")

    def move(self, pos_in_mm: float) -> bool:
        """Move the gripper to an absolute position, in millimeters."""
        return self._call("MOVE", f"rq_move_and_wait_mm({pos_in_mm})")

    def open(self) -> bool:
        ret = self._call("OPEN", "rq_open_and_wait()")
        self.is_open = True
        self.position = 0.0
        return ret

    def close(self) -> bool:
        ret = self._call("CLOSE", "rq_close_and_wait()")
        self.is_open = False
        self.position = 1.0
        return ret

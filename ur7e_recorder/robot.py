"""UR7e connection and motion-mode management."""

import logging

import rtde_control
import rtde_receive

logger = logging.getLogger(__name__)

# rtde_receive.RTDEReceiveInterface.getRobotMode() return value meaning
# "normal operation, capable of running a program" -- see its docstring.
ROBOT_MODE_RUNNING = 7


class UR7eRobot:
    """Owns the RTDE control/receive interfaces and their lifecycle.

    Also drives arms on other controller generations, despite the name --
    `controller="cb3"` targets CB3 controllers (e.g. a UR5 on PolyScope
    3.13), which expose freedrive as teachMode()/endTeachMode() instead of
    the e-Series freedriveMode()/endFreedriveMode().
    """

    def __init__(self, ip: str, controller: str = "e-series"):
        self.ip = ip
        self.controller = controller
        # rtde_c first: its constructor uploads/starts the External Control
        # URScript on the controller, which resets the robot's RTDE server
        # session. Connecting rtde_r before that reset can leave it stuck
        # silently replaying its last-received packet for the rest of the
        # process (no exception, just frozen readings).
        self.rtde_c = rtde_control.RTDEControlInterface(ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ip)

    def get_joint_positions(self) -> list:
        """Current joint positions in radians (6 values)."""
        return list(self.rtde_r.getActualQ())

    def _freedrive_preflight_issue(self) -> str | None:
        """Return a specific reason freedrive would be rejected, or None if clear."""
        if self.rtde_r.isEmergencyStopped():
            return "the robot is in an EMERGENCY STOP"
        if self.rtde_r.isProtectiveStopped():
            return "the robot is in a PROTECTIVE STOP"
        robot_mode = self.rtde_r.getRobotMode()
        if robot_mode != ROBOT_MODE_RUNNING:
            return (
                f"the robot isn't in RUNNING mode (robot mode={robot_mode}) -- "
                "the 'External Control' URCap program is likely not loaded/"
                "playing on the pendant, or the robot isn't in Remote Control mode"
            )
        return None

    def enable_freedrive(self):
        issue = self._freedrive_preflight_issue()
        if issue:
            msg = f"Cannot enable freedrive: {issue}."
            logger.error(msg)
            raise RuntimeError(msg)

        enter = self.rtde_c.teachMode if self.controller == "cb3" else self.rtde_c.freedriveMode
        ok = enter()
        if not ok:
            fn_name = "teachMode()" if self.controller == "cb3" else "freedriveMode()"
            msg = (
                f"{fn_name} was rejected by the robot. Common causes: the "
                "'External Control' URCap program isn't loaded/playing on the "
                "pendant, the robot isn't in Remote Control mode, or there's an "
                "active protective/safety stop."
            )
            logger.error(msg)
            raise RuntimeError(msg)

    def disable_freedrive(self):
        if self.controller == "cb3":
            self.rtde_c.endTeachMode()
        else:
            self.rtde_c.endFreedriveMode()

    def disconnect(self):
        self.rtde_c.stopScript()
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()

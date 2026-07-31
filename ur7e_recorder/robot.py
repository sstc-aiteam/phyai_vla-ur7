"""UR7e connection and motion-mode management."""

import rtde_control
import rtde_receive


class UR7eRobot:
    """Owns the RTDE control/receive interfaces and their lifecycle."""

    def __init__(self, ip: str):
        self.ip = ip
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ip)
        self.rtde_c = rtde_control.RTDEControlInterface(ip)

    def get_joint_positions(self) -> list:
        """Current joint positions in radians (6 values)."""
        return list(self.rtde_r.getActualQ())

    def enable_freedrive(self):
        ok = self.rtde_c.freedriveMode()
        if not ok:
            raise RuntimeError(
                "freedriveMode() was rejected by the robot. Common causes: the "
                "'External Control' URCap program isn't loaded/playing on the "
                "pendant, the robot isn't in Remote Control mode, or there's an "
                "active protective/safety stop."
            )

    def disable_freedrive(self):
        self.rtde_c.endFreedriveMode()

    def disconnect(self):
        self.rtde_c.stopScript()
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()

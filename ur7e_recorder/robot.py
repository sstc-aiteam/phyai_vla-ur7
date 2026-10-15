"""UR7e connection and motion-mode management."""

import atexit
import logging
import re
import signal
import socket
import threading
import time

import rtde_control
import rtde_receive

logger = logging.getLogger(__name__)

# rtde_receive.RTDEReceiveInterface.getRobotMode() return value meaning
# "normal operation, capable of running a program" -- see its docstring.
ROBOT_MODE_RUNNING = 7

# Dashboard Server: up whenever the controller software is running, on both
# CB3 and e-Series -- see cb3_record.py's preflight() for the same port used
# as a plain reachability canary.
DASHBOARD_PORT = 29999


# Connecting can fail transiently when the controller still holds an RTDE
# session (and its input registers) from a previous run that was killed
# before it could disconnect -- ur_rtde reports that as "End of file" /
# "input registers already in use". The stale session times out on the
# controller after a while, so retry with a growing delay before giving up.
CONNECT_ATTEMPTS = 4
CONNECT_RETRY_DELAY_S = 5.0

_termination_handlers_installed = False


def _install_termination_handlers():
    """Turn SIGTERM/SIGHUP into a normal interpreter exit.

    By default these signals kill the process without running `finally`
    blocks or atexit hooks, which leaves the controller holding this
    process's RTDE session and makes the next connect fail with "input
    registers already in use". Raising SystemExit instead lets the
    existing cleanup (`UR7eRobot.disconnect`) run. SIGINT already raises
    KeyboardInterrupt, so it needs nothing here.
    """
    global _termination_handlers_installed
    if _termination_handlers_installed:
        return
    if threading.current_thread() is not threading.main_thread():
        return

    def _handler(signum, _frame):
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _handler)
    _termination_handlers_installed = True


def detect_controller_generation(ip: str, timeout: float = 2.0) -> str:
    """Ask the Dashboard Server for its PolyScope version and classify the
    controller generation from it: PolyScope 3.x (CB3) reports "cb3";
    PolyScope 5.x/10.x (e-Series, incl. PolyScope X) reports "e-series".

    Raises RuntimeError if the dashboard can't be reached or parsed --
    callers should tell the user to pass --controller explicitly instead.
    """
    try:
        with socket.create_connection((ip, DASHBOARD_PORT), timeout=timeout) as sock:
            reader = sock.makefile("r", encoding="utf-8", newline="\n")
            reader.readline()  # "Connected: Universal Robots Dashboard Server"
            sock.sendall(b"PolyscopeVersion\n")
            reply = reader.readline().strip()
    except OSError as exc:
        raise RuntimeError(
            f"Could not reach the Dashboard Server at {ip}:{DASHBOARD_PORT} to "
            f"auto-detect the controller generation ({exc}). Pass --controller "
            "explicitly ('e-series' or 'cb3')."
        ) from exc

    # Reply is e.g. "URSoftware 3.13.0.10253 (三月 22 2020)" (CB3) or
    # "5.9.4.10151470 (Aug 2 2022)" (e-Series) -- prefix/locale vary, so pull
    # out the first x.y.z-style version number rather than assuming a shape.
    match = re.search(r"\d+(?:\.\d+){2,}", reply)
    if not match:
        raise RuntimeError(
            f"Could not parse a PolyScope version from Dashboard Server reply "
            f"{reply!r}. Pass --controller explicitly ('e-series' or 'cb3')."
        )
    major = int(match.group().split(".", 1)[0])

    generation = "cb3" if major < 5 else "e-series"
    logger.info("Detected controller=%s (PolyScope %s)", generation, reply)
    return generation


class UR7eRobot:
    """Owns the RTDE control/receive interfaces and their lifecycle.

    Also drives arms on other controller generations, despite the name --
    `controller="cb3"` targets CB3 controllers (e.g. a UR5 on PolyScope
    3.13), which expose freedrive as teachMode()/endTeachMode() instead of
    the e-Series freedriveMode()/endFreedriveMode().
    """

    def __init__(self, ip: str, controller: str = "auto"):
        self.ip = ip
        self.controller = (
            detect_controller_generation(ip) if controller == "auto" else controller
        )
        # rtde_c first: its constructor uploads/starts the External Control
        # URScript on the controller, which resets the robot's RTDE server
        # session. Connecting rtde_r before that reset can leave it stuck
        # silently replaying its last-received packet for the rest of the
        # process (no exception, just frozen readings).
        _install_termination_handlers()
        self.rtde_c = None
        self.rtde_r = None
        self._freedrive_active = False
        self._disconnected = False
        self._connect_with_retry()
        # Covers exits that bypass the caller's own try/finally (an
        # exception between construction and the main loop, sys.exit()).
        atexit.register(self.disconnect)

    def _connect_with_retry(self):
        for attempt in range(1, CONNECT_ATTEMPTS + 1):
            try:
                self.rtde_c = rtde_control.RTDEControlInterface(self.ip)
                self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ip)
                return
            except RuntimeError as exc:
                self._release_interfaces()
                if attempt == CONNECT_ATTEMPTS:
                    raise RuntimeError(
                        f"Could not open an RTDE connection to {self.ip} after "
                        f"{CONNECT_ATTEMPTS} attempts ({exc}). The controller "
                        "probably still holds an RTDE session from a previous run "
                        "that was killed without disconnecting. Make sure no other "
                        "process is connected to the robot, stop any program "
                        "playing on the pendant, then wait ~1 minute or restart "
                        "the controller's program and try again."
                    ) from exc
                delay = CONNECT_RETRY_DELAY_S * attempt
                print(
                    f"[WARN] RTDE connect attempt {attempt}/{CONNECT_ATTEMPTS} "
                    f"failed ({exc}); retrying in {delay:.0f}s..."
                )
                time.sleep(delay)

    def _release_interfaces(self):
        """Best-effort teardown of whatever was opened; never raises."""
        if self.rtde_c is not None:
            for call in (self.rtde_c.stopScript, self.rtde_c.disconnect):
                try:
                    call()
                except Exception:
                    pass
            self.rtde_c = None
        if self.rtde_r is not None:
            try:
                self.rtde_r.disconnect()
            except Exception:
                pass
            self.rtde_r = None

    def get_joint_positions(self) -> list:
        """Current joint positions in radians (6 values)."""
        return list(self.rtde_r.getActualQ())

    def resync_receive(self):
        """Rebuild the RTDE receive interface.

        Sending *any* URScript program to the controller -- not just the
        one-time External Control upload in `__init__` -- replaces
        whatever program is currently running there and can reset the
        robot's RTDE server session the same way (see the ordering
        comment above). When that happens, `self.rtde_r` doesn't raise;
        it just gets stuck silently replaying its last-received packet,
        so `get_joint_positions()` looks fine but stops changing.

        Call this after anything that sends a custom URScript to the
        controller mid-session (gripper commands via
        `sendCustomScriptFunction`, in particular) to make sure joint
        readings are live again afterward.
        """
        self.rtde_r.disconnect()
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ip)

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
        self._freedrive_active = True

    def disable_freedrive(self):
        # Calling endTeachMode()/endFreedriveMode() when freedrive was
        # never actually entered (e.g. enable_freedrive() raised) hits
        # ur_rtde's C++ binding in a state it doesn't handle gracefully
        # and crashes the whole process (segfault) instead of raising --
        # so only call it if we know we're actually in freedrive.
        if not self._freedrive_active:
            return
        if self.controller == "cb3":
            self.rtde_c.endTeachMode()
        else:
            self.rtde_c.endFreedriveMode()
        self._freedrive_active = False

    def disconnect(self):
        """Release the controller's RTDE session. Idempotent and never
        raises, so it's safe from both `finally` blocks and atexit."""
        if self._disconnected:
            return
        self._disconnected = True
        self._release_interfaces()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()

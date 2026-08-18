"""Interactive recording session: wires robot, gripper, cameras, and
keyboard input together and drives the fixed-rate recording loop.
"""

import time

from .camera import CameraManager
from .config import RecorderConfig
from .dataset import LeRobotDatasetWriter
from .episode import EpisodeRecorder
from .gripper import Gripper
from .keyboard import KeyListener
from .robot import UR7eRobot

CONTROLS_HELP = """\
  Controls:
    [S] Start / stop recording episode
    [SPACE] Toggle gripper open/close
    [D] Discard current episode
    [Q] Quit and save dataset"""


class RecordingSession:
    """Runs the free-drive recording loop until `num_episodes` is reached
    or the user quits.

    Key bindings are a dict of key -> handler, so new controls can be
    added by a caller without modifying the run loop.
    """

    def __init__(
        self,
        robot: UR7eRobot,
        gripper: Gripper,
        writer: LeRobotDatasetWriter,
        config: RecorderConfig,
        camera_manager: CameraManager | None = None,
    ):
        self.robot = robot
        self.gripper = gripper
        self.writer = writer
        self.cam_mgr = camera_manager
        self.fps = config.fps
        self.num_episodes = config.num_episodes

        self.keys = KeyListener()
        self.key_handlers = {
            "q": self._handle_quit,
            " ": self._handle_toggle_gripper,
            "d": self._handle_discard,
            "s": self._handle_toggle_recording,
        }

        self.episode = None
        self.recording = False
        self.episodes_recorded = 0
        self._quit = False

        # Tripwire for a stuck RTDE receive feed (e.g. getActualQ() frozen
        # on a stale packet): warn once if joint readings stop changing for
        # a full second of recording instead of silently saving dead data.
        self._last_joint_positions = None
        self._stuck_frames = 0
        self._stuck_warned = False

    def run(self):
        self.keys.start()
        self._print_banner()

        try:
            self.robot.enable_freedrive()
            print("[FREEDRIVE] Robot is now in free-drive mode. Guide it by hand.\n")
            self._loop()
        except KeyboardInterrupt:
            print("\n[INTERRUPTED]")
        finally:
            self.robot.disable_freedrive()
            self.robot.disconnect()
            if self.cam_mgr:
                self.cam_mgr.release()
            self.keys.stop()

        self.writer.write_info()
        return self.episodes_recorded

    def _loop(self):
        dt = 1.0 / self.fps
        while self.episodes_recorded < self.num_episodes and not self._quit:
            loop_start = time.time()

            self._handle_keys()

            if self.recording and self.episode is not None:
                self._record_step()

            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _handle_keys(self):
        for key, handler in self.key_handlers.items():
            if self.keys.pop(key):
                handler()

    def _handle_quit(self):
        print("\n[QUIT] Saving dataset...")
        self._quit = True

    def _handle_toggle_gripper(self):
        self.gripper.toggle()
        state_str = "OPEN" if self.gripper.is_open else "CLOSED"
        print(f"  Gripper: {state_str}")
        # Sending the gripper's URScript command replaces the running
        # freedrive control script on the controller, which silently ends
        # freedrive (the arm goes rigid). Re-enter freedrive so the user
        # can keep guiding it by hand.
        self.robot.enable_freedrive()

    def _handle_discard(self):
        if self.recording:
            self.recording = False
            self.episode = None
            print("  [DISCARDED] Episode discarded.")

    def _handle_toggle_recording(self):
        if not self.recording:
            self.episode = EpisodeRecorder()
            self.recording = True
            print(f"  [REC] Recording episode {self.episodes_recorded}...")
        else:
            self.recording = False
            if self.episode and self.episode.num_steps > 1:
                if self.writer.save_episode(self.episode):
                    self.episodes_recorded += 1
            self.episode = None
            print(f"  Progress: {self.episodes_recorded}/{self.num_episodes}")

    def _record_step(self):
        joint_positions = self.robot.get_joint_positions()
        self._check_stuck_readings(joint_positions)
        state = joint_positions + [self.gripper.position]
        action = state.copy()
        camera_frames = self.cam_mgr.read_all() if self.cam_mgr else {}

        self.episode.add_step(
            timestamp=time.time(),
            state=state,
            action=action,
            camera_frames=camera_frames,
        )

        if self.episode.num_steps % self.fps == 0:
            elapsed = self.episode.num_steps / self.fps
            print(f"    ... {self.episode.num_steps} steps ({elapsed:.1f}s)")

    def _check_stuck_readings(self, joint_positions: list):
        """Warn once if get_joint_positions() stops changing for a full
        second, which usually means the RTDE receive feed is stuck on a
        stale packet rather than the arm actually holding still."""
        if self._last_joint_positions is not None and all(
            abs(a - b) < 1e-9
            for a, b in zip(joint_positions, self._last_joint_positions)
        ):
            self._stuck_frames += 1
        else:
            self._stuck_frames = 0
            self._stuck_warned = False
        self._last_joint_positions = joint_positions

        if self._stuck_frames >= self.fps and not self._stuck_warned:
            print(
                "  [WARN] Joint readings haven't changed in ~1s. If the arm "
                "is actually moving, the RTDE receive feed is likely stuck "
                "— restart the recorder."
            )
            self._stuck_warned = True

    def _print_banner(self):
        camera_names = self.cam_mgr.names if self.cam_mgr else []
        print("\n" + "=" * 60)
        print("  UR7e Free-Drive Recorder")
        print("=" * 60)
        print(f"  Task:     {self.writer.task}")
        print(f"  Target:   {self.num_episodes} episodes @ {self.fps} Hz")
        print(f"  Cameras:  {camera_names or 'None'}")
        print()
        print(CONTROLS_HELP)
        print("=" * 60)
        print()

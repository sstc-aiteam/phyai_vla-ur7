"""Interactive recording session: wires robot, gripper, cameras, and
keyboard input together and drives the fixed-rate recording loop.
"""

import time

from .camera import CameraManager
from .config import RecorderConfig
from .dataset import LeRobotDatasetWriter
from .gripper import Gripper
from .keyboard import KeyListener
from .robot import UR7eRobot

CONTROLS_HELP = """\
  Controls:
    [S] Start / stop recording episode
    [SPACE] Toggle gripper open/close
    [D] Discard current episode
    [Q] Quit and save dataset"""

# Below this much total swing (radians) across an episode's six arm
# joints, treat it as the RTDE receive feed having been stuck for the
# whole episode rather than the human actually holding the arm still --
# see resync_receive()'s docstring. A real freedrive episode this long
# always has more drift than this even when the person is trying to
# hold position.
FROZEN_EPISODE_RANGE_RAD = 0.01


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

        self.recording = False
        self.episodes_recorded = 0
        self._quit = False

        # Tripwire for a stuck RTDE receive feed (e.g. getActualQ() frozen
        # on a stale packet): auto-resync once if joint readings stop
        # changing for a full second of recording, instead of silently
        # saving dead data.
        self._last_joint_positions = None
        self._stuck_frames = 0
        self._stuck_resynced = False

        # Per-episode min/max of the six arm joints, to catch a feed that
        # was stuck for the *entire* episode (resync didn't happen to be
        # triggered, or fired too late) before it ever reaches the dataset.
        self._episode_min = None
        self._episode_max = None

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
            # Quitting or an interrupt mid-episode shouldn't leave a
            # half-written episode buffer (and its temp frame images)
            # behind uncommitted.
            if self.recording:
                self.writer.discard_episode()

        self.writer.finalize()
        return self.episodes_recorded

    def _loop(self):
        dt = 1.0 / self.fps
        while self.episodes_recorded < self.num_episodes and not self._quit:
            loop_start = time.time()

            self._handle_keys()

            if self.recording:
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
        # The gripper command sends a URScript program to the controller,
        # which both replaces the running freedrive control script
        # (silently ending freedrive -- the arm goes rigid) and can leave
        # the RTDE receive feed stuck on a stale packet (see
        # resync_receive()). Re-enter freedrive so the user can keep
        # guiding it by hand, and resync the receive feed so joint
        # readings are live again.
        self.robot.enable_freedrive()
        self.robot.resync_receive()

    def _handle_discard(self):
        if self.recording:
            self.recording = False
            self.writer.discard_episode()
            print("  [DISCARDED] Episode discarded.")

    def _handle_toggle_recording(self):
        if not self.recording:
            self.recording = True
            self._episode_min = None
            self._episode_max = None
            print(f"  [REC] Recording episode {self.episodes_recorded}...")
        else:
            self.recording = False
            if self._episode_frozen():
                self.writer.discard_episode()
                print(
                    "  [DISCARDED] Joint readings never changed across the whole "
                    "episode (RTDE receive feed was stuck) -- this would have "
                    "been a dead episode. Not saved; please redo it."
                )
            elif self.writer.save_episode():
                self.episodes_recorded += 1
            print(f"  Progress: {self.episodes_recorded}/{self.num_episodes}")

    def _episode_frozen(self) -> bool:
        """True if the just-recorded episode's arm joints never moved
        beyond FROZEN_EPISODE_RANGE_RAD -- i.e. the receive feed was stuck
        for its entire duration rather than genuinely holding still."""
        if self._episode_min is None or self.writer.num_steps < self.fps:
            return False  # too short to distinguish from a deliberate pause
        max_range = max(hi - lo for lo, hi in zip(self._episode_min, self._episode_max))
        return max_range < FROZEN_EPISODE_RANGE_RAD

    def _record_step(self):
        joint_positions = self.robot.get_joint_positions()
        self._check_stuck_readings(joint_positions)
        self._track_episode_range(joint_positions)
        state = joint_positions + [self.gripper.position]
        action = state.copy()
        camera_frames = self.cam_mgr.read_all() if self.cam_mgr else {}

        self.writer.add_step(state=state, action=action, camera_frames=camera_frames)

        if self.writer.num_steps % self.fps == 0:
            elapsed = self.writer.num_steps / self.fps
            print(f"    ... {self.writer.num_steps} steps ({elapsed:.1f}s)")

    def _track_episode_range(self, joint_positions: list):
        if self._episode_min is None:
            self._episode_min = list(joint_positions)
            self._episode_max = list(joint_positions)
        else:
            self._episode_min = [min(a, b) for a, b in zip(self._episode_min, joint_positions)]
            self._episode_max = [max(a, b) for a, b in zip(self._episode_max, joint_positions)]

    def _check_stuck_readings(self, joint_positions: list):
        """Auto-recover once if get_joint_positions() stops changing for a
        full second, which usually means the RTDE receive feed is stuck on
        a stale packet rather than the arm actually holding still."""
        if self._last_joint_positions is not None and all(
            abs(a - b) < 1e-9
            for a, b in zip(joint_positions, self._last_joint_positions)
        ):
            self._stuck_frames += 1
        else:
            self._stuck_frames = 0
            self._stuck_resynced = False
        self._last_joint_positions = joint_positions

        if self._stuck_frames >= self.fps and not self._stuck_resynced:
            print(
                "  [WARN] Joint readings haven't changed in ~1s -- the RTDE "
                "receive feed looks stuck. Resyncing it..."
            )
            self.robot.resync_receive()
            self._stuck_resynced = True

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

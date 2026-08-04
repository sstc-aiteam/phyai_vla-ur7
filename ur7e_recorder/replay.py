"""Physical episode replay: reads recorded actions from a LeRobot v2
dataset episode and streams them back to the real UR7e over RTDE.

The first waypoint is reached with a slow, blocking `moveJ` so the robot
never jumps from wherever it currently is straight into a fast streamed
move. The rest of the episode is streamed with `servoJ` at the episode's
recording fps, which is what lets a low-rate (5-30 Hz) waypoint sequence
play back as smooth motion instead of a series of jerky point-to-point
moves.
"""

import json
import time
from pathlib import Path

import pandas as pd

from .gripper import Gripper
from .robot import UR7eRobot

ACTION_DIM = 7  # 6 joints + gripper
GRIPPER_CLOSED_THRESHOLD = 0.5

# servoJ tuning for tracking closely-spaced recorded waypoints.
# velocity/acceleration are unused by servoJ but required by its signature.
SERVO_VELOCITY = 0.0
SERVO_ACCELERATION = 0.0
SERVO_LOOKAHEAD_TIME = 0.1
SERVO_GAIN = 300


class EpisodeReplayer:
    """Replays one recorded episode's actions on the physical robot."""

    def __init__(self, robot: UR7eRobot, gripper: Gripper, fps: int):
        self.robot = robot
        self.gripper = gripper
        self.fps = fps

    def replay(self, dataset_dir: Path, episode_index: int,
               start_speed: float, start_acceleration: float):
        actions = self._load_actions(dataset_dir, episode_index)
        dt = 1.0 / self.fps

        first_q, first_gripper_closed = actions[0][:6], actions[0][6] > GRIPPER_CLOSED_THRESHOLD
        print(f"[REPLAY] Episode {episode_index}: moving to start pose...")
        self.robot.rtde_c.moveJ(first_q, start_speed, start_acceleration)
        self._apply_gripper(first_gripper_closed)

        print(f"[REPLAY] Streaming {len(actions)} steps @ {self.fps} Hz")
        for i, action in enumerate(actions):
            loop_start = time.time()

            q = action[:6]
            self._apply_gripper(action[6] > GRIPPER_CLOSED_THRESHOLD)
            self.robot.rtde_c.servoJ(
                q, SERVO_VELOCITY, SERVO_ACCELERATION, dt,
                SERVO_LOOKAHEAD_TIME, SERVO_GAIN,
            )

            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            if (i + 1) % self.fps == 0:
                print(f"    ... {i + 1}/{len(actions)} steps")

        self.robot.rtde_c.servoStop()
        print("[REPLAY] Done.")

    def _apply_gripper(self, should_close: bool):
        if should_close and self.gripper.is_open:
            self.gripper.close()
        elif not should_close and not self.gripper.is_open:
            self.gripper.open()

    @staticmethod
    def _load_actions(dataset_dir: Path, episode_index: int) -> list:
        parquet_path = (dataset_dir / "data" / "chunk-000"
                         / f"episode_{episode_index:06d}.parquet")
        if not parquet_path.exists():
            raise FileNotFoundError(f"No such episode: {parquet_path}")

        df = pd.read_parquet(parquet_path).sort_values("frame_index")
        action_cols = [f"action.{j}" for j in range(ACTION_DIM)]
        actions = df[action_cols].values.tolist()
        if not actions:
            raise ValueError(f"Episode {episode_index} has no steps.")
        return actions

    @staticmethod
    def dataset_fps(dataset_dir: Path) -> int:
        with open(dataset_dir / "meta" / "info.json") as f:
            return json.load(f)["fps"]

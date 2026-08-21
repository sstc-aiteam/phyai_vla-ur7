"""Command-line entry point: parses flags into a RecorderConfig and wires
up a RecordingSession. Defaults live on RecorderConfig itself, so this
module never hard-codes a value that isn't just a flag mirror.
"""

import argparse
from pathlib import Path

import pandas as pd

from .camera import CameraManager
from .config import CAMERA_BACKENDS, RecorderConfig, ReplayConfig
from .dataset import LeRobotDatasetWriter
from .dump import load_dataset_joint_states
from .gripper import RobotiqGripper
from .replay import EpisodeReplayer
from .robot import UR7eRobot
from .session import RecordingSession


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = RecorderConfig()
    parser = argparse.ArgumentParser(description="UR7e Free-Drive Recorder for LeRobot")
    parser.add_argument("--robot-ip", type=str, default=defaults.robot_ip,
                        help="UR7e IP address")
    parser.add_argument("--dataset-name", type=str, default=defaults.dataset_name,
                        help="Name for the output dataset directory")
    parser.add_argument("--num-episodes", type=int, default=defaults.num_episodes,
                        help="Target number of episodes to record")
    parser.add_argument("--fps", type=int, default=defaults.fps,
                        help="Recording frequency in Hz")
    parser.add_argument("--task", type=str, default=defaults.task,
                        help="Task description for the dataset")
    parser.add_argument("--cam-overhead", type=int, default=defaults.cam_overhead,
                        help="Camera index for overhead view (-1 to disable)")
    parser.add_argument("--cam-overhead-backend", type=str, choices=CAMERA_BACKENDS,
                        default=defaults.cam_overhead_backend,
                        help="Backend for the overhead camera")
    parser.add_argument("--cam-wrist", type=int, default=defaults.cam_wrist,
                        help="Camera index for wrist view (-1 to disable)")
    parser.add_argument("--cam-wrist-backend", type=str, choices=CAMERA_BACKENDS,
                        default=defaults.cam_wrist_backend,
                        help="Backend for the wrist camera")
    return parser


def parse_config(argv: list | None = None) -> RecorderConfig:
    args = build_arg_parser().parse_args(argv)
    return RecorderConfig(
        robot_ip=args.robot_ip,
        dataset_name=args.dataset_name,
        num_episodes=args.num_episodes,
        fps=args.fps,
        task=args.task,
        cam_overhead=args.cam_overhead,
        cam_overhead_backend=args.cam_overhead_backend,
        cam_wrist=args.cam_wrist,
        cam_wrist_backend=args.cam_wrist_backend,
    )


def print_next_steps(dataset_name: str, episodes_recorded: int):
    print(f"\nDone! {episodes_recorded} episodes saved to ./{dataset_name}/")
    print("\nNext steps:")
    print(f"  1. Inspect:  lerobot-dataset-viz --repo-id {dataset_name} --mode local")
    print("  2. Train:    python -m lerobot.train \\")
    print(f"                 --dataset.repo_id={dataset_name} \\")
    print("                 --policy.type=act \\")
    print(f"                 --output_dir=outputs/act_{dataset_name}")
    print(f"  3. Push:     huggingface-cli upload <user>/{dataset_name} ./{dataset_name}")


def main():
    config = parse_config()

    camera_configs = config.camera_configs
    if not camera_configs:
        print("[WARN] No cameras configured. Recording state-only dataset.")
    cam_mgr = CameraManager(camera_configs) if camera_configs else None

    print(f"Connecting to UR7e at {config.robot_ip}...")
    robot = UR7eRobot(config.robot_ip)
    print("[OK] Connected to UR robtic arm.")

    gripper = RobotiqGripper(robot.rtde_c)
    gripper.open()

    writer = LeRobotDatasetWriter(config, camera_names=list(camera_configs.keys()))

    session = RecordingSession(
        robot=robot,
        gripper=gripper,
        writer=writer,
        config=config,
        camera_manager=cam_mgr,
    )
    episodes_recorded = session.run()

    print_next_steps(config.dataset_name, episodes_recorded)


def build_replay_arg_parser() -> argparse.ArgumentParser:
    defaults = ReplayConfig()
    parser = argparse.ArgumentParser(description="UR7e Episode Replayer for LeRobot")
    parser.add_argument("--robot-ip", type=str, default=defaults.robot_ip,
                        help="UR7e IP address")
    parser.add_argument("--dataset-name", type=str, default=defaults.dataset_name,
                        help="Dataset directory to replay from")
    parser.add_argument("--episode", type=int, default=defaults.episode,
                        help="episode_index to replay")
    parser.add_argument("--fps", type=int, default=defaults.fps,
                        help="Playback rate in Hz (default: the dataset's recorded fps)")
    parser.add_argument("--start-speed", type=float, default=defaults.start_speed,
                        help="rad/s for the initial move to the episode's start pose")
    parser.add_argument("--start-acceleration", type=float, default=defaults.start_acceleration,
                        help="rad/s^2 for the initial move to the episode's start pose")
    return parser


def parse_replay_config(argv: list | None = None) -> ReplayConfig:
    args = build_replay_arg_parser().parse_args(argv)
    return ReplayConfig(
        robot_ip=args.robot_ip,
        dataset_name=args.dataset_name,
        episode=args.episode,
        fps=args.fps,
        start_speed=args.start_speed,
        start_acceleration=args.start_acceleration,
    )


def main_replay():
    config = parse_replay_config()
    dataset_dir = Path(config.dataset_name)
    fps = config.fps or EpisodeReplayer.dataset_fps(dataset_dir)

    print(f"Connecting to UR7e at {config.robot_ip}...")
    robot = UR7eRobot(config.robot_ip)
    print("[OK] Connected to UR robtic arm.")

    gripper = RobotiqGripper(robot.rtde_c)

    replayer = EpisodeReplayer(robot, gripper, fps=fps)
    try:
        replayer.replay(dataset_dir, config.episode,
                         config.start_speed, config.start_acceleration)
    finally:
        robot.disconnect()


def build_dump_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump recorded joint states from a LeRobot dataset")
    parser.add_argument("--dataset-name", type=str, default="ur7e_pick_and_place",
                        help="Dataset directory to read from")
    parser.add_argument("--episode", type=int, default=None,
                        help="Dump only this episode_index (default: all episodes)")
    parser.add_argument("--format", type=str, choices=["table", "csv", "json"], default=None,
                        help="Output format (default: table for stdout, or inferred from "
                             "--output's extension)")
    parser.add_argument("--output", type=str, default=None,
                        help="Write to this path instead of printing to stdout")
    return parser


def _resolve_dump_format(args) -> str:
    if args.format:
        return args.format
    if args.output:
        suffix = Path(args.output).suffix.lower()
        if suffix in (".json", ".csv"):
            return suffix.lstrip(".")
    return "table"


def main_dump():
    args = build_dump_arg_parser().parse_args()
    df = load_dataset_joint_states(Path(args.dataset_name), args.episode)
    fmt = _resolve_dump_format(args)

    if fmt == "json":
        text = df.to_json(orient="records", indent=2)
    elif fmt == "csv":
        text = df.to_csv(index=False)
    else:
        with pd.option_context("display.max_rows", None, "display.width", None):
            text = df.to_string(index=False)

    if args.output:
        Path(args.output).write_text(text)
        print(f"[OK] Wrote {len(df)} rows to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()

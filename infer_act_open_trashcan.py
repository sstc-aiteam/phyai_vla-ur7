#!/usr/bin/env python
"""Closed-loop ACT inference on the real UR7e arm.

Mirrors `eval_act_open_trashcan.py`'s policy-loading/pre-post-processing
and `ur7e_recorder.replay.EpisodeReplayer`'s physical motion pattern
(blocking moveJ to the first pose, then a servoJ stream at the recording
fps) -- except every action comes live from the policy instead of a
recorded episode.

SAFETY -- read before running:
  * The arm WILL move on its own. Stand clear, keep a hand near the
    pendant's e-stop, and be ready to hit [Q] (checked once per control
    step) or the physical e-stop at any time.
  * --max-step-rad clamps how far any single joint may move in one
    control step (default 0.05 rad ~= 2.9 deg at 5 Hz => ~14 deg/s max),
    guarding against a bad/OOD prediction commanding a large jump. It
    does not prevent collisions with anything in the arm's reach.
  * The `open_trashcan` dataset was recorded with `--gripper none`
    (meta/stats.json: the `gripper` channel is a constant 0.0 across all
    50 episodes), so the policy never saw a gripper open/close example.
    Its 7th action dimension carries no learned signal -- don't expect
    (or wire up) meaningful gripper control from this checkpoint.
  * Test at a low --fps/short --num-steps first, and watch the very
    first moveJ (it moves at --start-speed, blocking, from wherever the
    arm currently is to the policy's first predicted pose).
"""

import argparse
import time
from pathlib import Path

import cv2
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from ur7e_recorder.camera import CameraManager
from ur7e_recorder.config import CameraConfig, GRIPPER_KINDS
from ur7e_recorder.gripper import NoGripper, RobotiqGripper
from ur7e_recorder.keyboard import KeyListener
from ur7e_recorder.robot import UR7eRobot

# Matches ur7e_recorder.replay.EpisodeReplayer -- see its module docstring
# for why servoJ (not moveJ) is used to stream closely-spaced waypoints.
GRIPPER_CLOSED_THRESHOLD = 0.5
SERVO_VELOCITY = 0.0
SERVO_ACCELERATION = 0.0
SERVO_LOOKAHEAD_TIME = 0.1
SERVO_GAIN = 300


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="outputs/act_open_trashcan/checkpoints/last/pretrained_model")
    parser.add_argument("--robot-ip", default="192.168.50.76")
    parser.add_argument("--controller", default="e-series", choices=("e-series", "cb3"),
                         help="Inference only moves the arm, never freedrive, so this rarely matters")
    parser.add_argument("--gripper", default="none", choices=GRIPPER_KINDS,
                         help="Must match how open_trashcan was recorded (default: none)")
    parser.add_argument("--cam-wrist-index", type=int, required=True,
                         help="Camera index for the wrist view -- must match how open_trashcan was recorded")
    parser.add_argument("--cam-wrist-backend", default="realsense", choices=("usb", "realsense"))
    parser.add_argument("--task", default="task description in here",
                         help="Task string the policy was conditioned on (default: the placeholder "
                              "open_trashcan was actually recorded with)")
    parser.add_argument("--fps", type=int, default=5, help="Control rate in Hz -- match the training fps")
    parser.add_argument("--num-steps", type=int, default=100, help="Stop after this many control steps")
    parser.add_argument("--max-step-rad", type=float, default=0.05,
                         help="Safety clamp: max joint delta per control step")
    parser.add_argument("--start-speed", type=float, default=0.1, help="rad/s for the initial move to the first pose")
    parser.add_argument("--start-acceleration", type=float, default=0.3, help="rad/s^2 for that initial move")
    parser.add_argument("--device", default="cuda")
    return parser


def main():
    args = build_arg_parser().parse_args()

    print(f"Loading policy from {args.checkpoint}")
    policy = ACTPolicy.from_pretrained(Path(args.checkpoint), device=args.device)
    device_override = {"device_processor": {"device": policy.config.device}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=Path(args.checkpoint), dataset_stats=None,
        preprocessor_overrides=device_override, postprocessor_overrides=device_override,
    )

    cam_mgr = CameraManager({
        "cam_wrist": CameraConfig(index=args.cam_wrist_index, backend=args.cam_wrist_backend)
    })

    print(f"Connecting to UR arm at {args.robot_ip}...")
    robot = UR7eRobot(args.robot_ip, controller=args.controller)
    print(f"[OK] Connected to UR robotic arm (controller={robot.controller}).")
    gripper = NoGripper() if args.gripper == "none" else RobotiqGripper(robot.rtde_c)

    keys = KeyListener()
    keys.start()
    print("Press [Q] at any time to stop.\n")

    dt = 1.0 / args.fps
    policy.reset()
    prev_q = None

    try:
        for step in range(args.num_steps):
            if keys.pop("q"):
                print("[STOP] Q pressed.")
                break
            loop_start = time.time()

            q = robot.get_joint_positions()
            state = torch.tensor(q + [gripper.position], dtype=torch.float32)

            frame_bgr = cam_mgr.read_all()["cam_wrist"]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0

            obs = preprocessor({
                "observation.state": state,
                "observation.images.cam_wrist": image,
                "task": args.task,
            })
            with torch.inference_mode():
                action = policy.select_action(obs)
            action = postprocessor(action).squeeze(0).cpu().numpy()

            target_q = action[:6].tolist()
            if prev_q is not None:
                target_q = [
                    p + max(-args.max_step_rad, min(args.max_step_rad, t - p))
                    for p, t in zip(prev_q, target_q)
                ]

            if step == 0:
                print(f"Moving to first predicted pose at {args.start_speed} rad/s...")
                robot.rtde_c.moveJ(target_q, args.start_speed, args.start_acceleration)
            else:
                robot.rtde_c.servoJ(
                    target_q, SERVO_VELOCITY, SERVO_ACCELERATION, dt,
                    SERVO_LOOKAHEAD_TIME, SERVO_GAIN,
                )

            should_close = action[6] > GRIPPER_CLOSED_THRESHOLD
            if should_close and gripper.is_open:
                gripper.close()
            elif not should_close and not gripper.is_open:
                gripper.open()

            prev_q = target_q

            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            if (step + 1) % args.fps == 0:
                print(f"    ... step {step + 1}/{args.num_steps}")
    finally:
        robot.rtde_c.servoStop()
        keys.stop()
        cam_mgr.release()
        robot.disconnect()
        print("[DONE] Inference stopped, robot disconnected.")


if __name__ == "__main__":
    main()

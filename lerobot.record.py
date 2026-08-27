#!/usr/bin/env python3
"""
UR7e Free-Drive Pick-and-Place Recorder for LeRobot
====================================================
Records demonstrations by physically guiding the UR7e in freedrive mode.
Saves data in LeRobot v3 dataset format (Parquet + MP4), via the official
`lerobot` package's `LeRobotDataset`.

Requirements:
    pip install -r requirements.txt

Usage:
    python lerobot.record.py \
        --robot-ip 192.168.50.75 \
        --dataset-name my_pick_and_place \
        --num-episodes 50 \
        --fps 30 \
        --task "Pick up the object and place it in the bin" \
        --cam-overhead 0 --cam-wrist 2 --cam-wrist-backend realsense

Controls during recording:
    [SPACE]  - Toggle gripper open/close
    [S]      - Start/stop recording an episode
    [D]      - Discard current episode
    [Q]      - Quit and save dataset

Implementation lives in the `ur7e_recorder` package:
    config.py   - RecorderConfig / CameraConfig (single source of truth for settings)
    keyboard.py - non-blocking key input
    gripper.py  - Gripper interface + RobotiqGripper
    robot.py    - UR7e RTDE connection lifecycle
    camera.py   - Camera interface (USBCamera, RealSenseCamera) + CameraManager
    dataset.py  - LeRobotDatasetWriter (wraps lerobot's LeRobotDataset, v3 format on disk)
    session.py  - RecordingSession (the recording loop)
    cli.py      - argument parsing and wiring
"""

from ur7e_recorder.cli import main

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
UR7e Episode Replayer for LeRobot
==================================
Physically replays one recorded episode by streaming its saved joint and
gripper actions back to the real UR7e over RTDE.

Requirements:
    pip install -r requirements.txt

Usage:
    python lerobot.replay.py \
        --robot-ip 192.168.50.75 \
        --dataset-name ur7e_pick_and_place \
        --episode 3

By default playback runs at the fps recorded in the dataset's
meta/info.json; pass --fps to override it.

SAFETY: the robot moves to the episode's start pose with a slow, blocking
moveJ (see --start-speed / --start-acceleration) before streaming the rest
of the trajectory. Clear the workspace and keep a hand on the pendant's
e-stop before running.

Implementation lives in `ur7e_recorder.replay.EpisodeReplayer`, wired up by
`ur7e_recorder.cli.main_replay`.
"""

from ur7e_recorder.cli import main_replay

if __name__ == "__main__":
    main_replay()

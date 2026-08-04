#!/usr/bin/env python3
"""
UR7e Dataset Joint State Dumper
================================
Dumps the recorded joint state (and action) values for a dataset — every
episode by default, or a single one with --episode — to stdout or CSV.
Pure data inspection: does not connect to the robot.

Usage:
    python lerobot.dump_states.py --dataset-name ur7e_pick_and_place
    python lerobot.dump_states.py --dataset-name ur7e_pick_and_place --episode 3
    python lerobot.dump_states.py --dataset-name ur7e_pick_and_place --output states.csv
    python lerobot.dump_states.py --dataset-name ur7e_pick_and_place --output states.json
    python lerobot.dump_states.py --dataset-name ur7e_pick_and_place --format json

Implementation lives in `ur7e_recorder.dump.load_dataset_joint_states`,
wired up by `ur7e_recorder.cli.main_dump`.
"""

from ur7e_recorder.cli import main_dump

if __name__ == "__main__":
    main_dump()

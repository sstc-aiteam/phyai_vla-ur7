# phyai_vla-ur7

Free-drive pick-and-place demonstration recorder for UR arms, producing
datasets in [LeRobot](https://github.com/huggingface/lerobot) v3 format
(chunked Parquet state/action + H.264 MP4 video) for training VLA
policies, written via the official `lerobot` package's `LeRobotDataset`.

You physically guide the robot in freedrive mode while the recorder logs
joint state/action and camera frames for each episode.

Built for the UR7e (e-Series), but also drives older CB3 controllers --
e.g. a UR5 on PolyScope 3.13 -- via `--controller cb3` (see below); every
recording, regardless of arm, is written in the same LeRobot v3 layout.

## Requirements

- Python 3.10+
- A UR robot reachable over the network, with the RTDE interface enabled
- (Optional) a Robotiq gripper attached via the robot's tool I/O -- pass
  `--gripper none` for a bare arm with nothing attached
- (Optional) one or two cameras: USB (OpenCV) and/or Intel RealSense

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Three entry point scripts cover the dataset lifecycle: record a dataset,
replay one of its episodes on the robot, and dump its joint states for
inspection.

### Recording a dataset

Run the recorder, pointing it at your robot's IP and choosing which cameras
to use:

```bash
python lerobot.record.py \
    --robot-ip 192.168.50.75 \
    --dataset-name my_pick_and_place \
    --num-episodes 50 \
    --fps 1 \
    --task "Pick up the object and place it in the bin" \
    --cam-wrist 0 --cam-wrist-backend realsense
```

| Flag | Default | Description |
| --- | --- | --- |
| `--robot-ip` | `192.168.50.76` | UR7e IP address |
| `--dataset-name` | `ur7e_pick_and_place` | Output dataset directory name |
| `--num-episodes` | `5` | Target number of episodes to record |
| `--fps` | `5` | Recording frequency in Hz |
| `--task` | `Pick up the object and place it in the bin` | Task description stored in the dataset |
| `--cam-overhead` | `0` | Camera index for overhead view (`-1` to disable) |
| `--cam-overhead-backend` | `usb` | Backend for the overhead camera (`usb` or `realsense`) |
| `--cam-wrist` | `-1` | Camera index for wrist view (`-1` to disable) |
| `--cam-wrist-backend` | `usb` | Backend for the wrist camera (`usb` or `realsense`) |
| `--controller` | `e-series` | `e-series` (`freedriveMode`, UR7e/UR5e/...) or `cb3` (`teachMode`, e.g. a UR5 on PolyScope 3.13) |
| `--gripper` | `robotiq` | `robotiq`, or `none` for a bare arm with nothing attached |
| `--robot-type` | `ur7e` | Robot type recorded in the dataset's `meta/info.json` |

Camera indices of `-1` disable that camera. With no cameras enabled, the
recorder still runs and produces a state-only dataset.

To record from a UR5 on a CB3 controller (PolyScope 3.13) with no gripper
attached, everything else about the workflow -- controls, dataset layout,
replay, dump -- stays the same:

```bash
python lerobot.record.py \
    --robot-ip 192.168.50.50 \
    --dataset-name my_ur5_dataset \
    --controller cb3 \
    --gripper none \
    --robot-type ur5
```

#### Controls while recording

| Key | Action |
| --- | --- |
| `SPACE` | Toggle gripper open/close |
| `S` | Start/stop recording the current episode |
| `D` | Discard the current episode |
| `Q` | Quit and save the dataset |

#### After recording

```
Done! <N> episodes saved to ./<dataset_name>/

Next steps:
  1. Inspect:  lerobot-dataset-viz --repo-id <dataset_name> --mode local
  2. Train:    python -m lerobot.train \
                 --dataset.repo_id=<dataset_name> \
                 --policy.type=act \
                 --output_dir=outputs/act_<dataset_name>
  3. Push:     huggingface-cli upload <user>/<dataset_name> ./<dataset_name>
```

### Replaying an episode

Physically re-runs one recorded episode on the robot, streaming the saved
joint + gripper actions back over RTDE:

```bash
python lerobot.replay.py \
    --robot-ip 192.168.50.75 \
    --dataset-name ur7e_pick_and_place \
    --episode 3
    --fps 1
```

| Flag | Default | Description |
| --- | --- | --- |
| `--robot-ip` | `192.168.50.76` | UR7e IP address |
| `--dataset-name` | `ur7e_pick_and_place` | Dataset directory to replay from |
| `--episode` | `0` | `episode_index` to replay |
| `--fps` | dataset's recorded fps | Playback rate in Hz |
| `--start-speed` | `0.3` | rad/s for the initial move to the episode's start pose |
| `--start-acceleration` | `0.3` | rad/s² for the initial move to the episode's start pose |
| `--gripper` | `robotiq` | `robotiq`, or `none` -- must match how the dataset was recorded |

The robot first moves to the episode's start pose with a slow, blocking
move (`--start-speed` / `--start-acceleration`) before streaming the rest
of the trajectory via `servoJ` at `--fps`.

**Safety:** clear the workspace and keep a hand on the pendant's e-stop
before running — this drives the real robot with no collision checking.

### Dumping joint states

Inspects the recorded joint state/action values for a dataset without
connecting to the robot:

```bash
python lerobot.dump_states.py --dataset-name ur7e_pick_and_place             # all episodes, printed as a table
python lerobot.dump_states.py --dataset-name ur7e_pick_and_place --episode 3 # one episode
python lerobot.dump_states.py --dataset-name ur7e_pick_and_place --output states.csv
python lerobot.dump_states.py --dataset-name ur7e_pick_and_place --output states.json
python lerobot.dump_states.py --dataset-name ur7e_pick_and_place --format json  # JSON to stdout
```

| Flag | Default | Description |
| --- | --- | --- |
| `--dataset-name` | `ur7e_pick_and_place` | Dataset directory to read from |
| `--episode` | all episodes | Dump only this `episode_index` |
| `--format` | inferred, else `table` | Output shape: `table`, `csv`, or `json` |
| `--output` | stdout | Write to this path instead of printing |

`--format` picks the output shape; if omitted it's inferred from
`--output`'s file extension, defaulting to `table` when printing to stdout.

## Project layout

```
lerobot.record.py       Record entry point script
lerobot.replay.py       Replay entry point script
lerobot.dump_states.py  Joint state dump entry point script
requirements.txt        Python dependencies
ur7e_recorder/          Recorder implementation
    config.py           RecorderConfig / ReplayConfig / CameraConfig (single source of truth for settings)
    keyboard.py         Non-blocking key input
    gripper/            Gripper interface, RobotiqGripper, NoGripper (--gripper none)
    robot.py            UR RTDE connection lifecycle (e-Series freedriveMode or CB3 teachMode)
    camera.py           Camera interface (USBCamera, RealSenseCamera) + CameraManager
    dataset.py          LeRobotDatasetWriter (wraps lerobot's LeRobotDataset, v3 format on disk)
    session.py          RecordingSession (the recording loop)
    replay.py            EpisodeReplayer (streams a saved episode's actions back to the robot)
    dump.py               load_dataset_joint_states (reads joint state/action data from disk)
    cli.py               Argument parsing and wiring
ur7e_pick_and_place/    Example/output dataset metadata (LeRobot v3 layout)
```

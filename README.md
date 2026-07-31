# phyai_vla-ur7

Free-drive pick-and-place demonstration recorder for the UR7e arm, producing
datasets in [LeRobot](https://github.com/huggingface/lerobot) v2 format
(Parquet state/action + MP4 video) for training VLA policies.

You physically guide the robot in freedrive mode while the recorder logs
joint state/action and camera frames for each episode.

## Requirements

- Python 3.10+
- A UR7e robot reachable over the network, with the RTDE interface enabled
- (Optional) a Robotiq gripper attached via the robot's tool I/O
- (Optional) one or two cameras: USB (OpenCV) and/or Intel RealSense

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the recorder, pointing it at your robot's IP and choosing which cameras
to use:

```bash
python lerobot.train.py \
    --robot-ip 192.168.50.75 \
    --dataset-name my_pick_and_place \
    --num-episodes 50 \
    --fps 30 \
    --task "Pick up the object and place it in the bin" \
    --cam-overhead 0 --cam-wrist 2 --cam-wrist-backend realsense
```

### CLI options

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

Camera indices of `-1` disable that camera. With no cameras enabled, the
recorder still runs and produces a state-only dataset.

### Controls while recording

| Key | Action |
| --- | --- |
| `SPACE` | Toggle gripper open/close |
| `S` | Start/stop recording the current episode |
| `D` | Discard the current episode |
| `Q` | Quit and save the dataset |

### After recording

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

## Project layout

```
lerobot.train.py       Entry point script
requirements.txt       Python dependencies
ur7e_recorder/         Recorder implementation
    config.py          RecorderConfig / CameraConfig (single source of truth for settings)
    keyboard.py        Non-blocking key input
    gripper/           Gripper interface + RobotiqGripper
    robot.py           UR7e RTDE connection lifecycle
    camera.py           Camera interface (USBCamera, RealSenseCamera) + CameraManager
    episode.py          EpisodeRecorder (per-episode buffer)
    dataset.py          LeRobotDatasetWriter (LeRobot v2 format on disk)
    session.py          RecordingSession (the recording loop)
    cli.py               Argument parsing and wiring
ur7e_pick_and_place/    Example/output dataset metadata (LeRobot v2 layout)
```

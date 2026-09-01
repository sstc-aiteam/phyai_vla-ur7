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

Entry point scripts cover the full lifecycle: record a dataset, replay one
of its episodes on the robot, dump its joint states for inspection, verify
its motion quality, concatenate multiple datasets together, train an ACT
policy on the result, evaluate it open-loop against held-out episodes, and
run it closed-loop on the real robot.

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

#### Resuming a recording session

`--dataset-name` decides create-vs-resume on its own: point it at a name
that doesn't exist yet and a new dataset is created as usual; point it at
a dataset you already recorded and the recorder appends to it instead,
picking episode numbering back up where it left off -- no separate flag
needed:

```bash
python lerobot.record.py \
    --robot-ip 192.168.50.75 \
    --dataset-name my_pick_and_place \
    --num-episodes 20
```

If `--dataset-name` exists but doesn't look like a LeRobot dataset (no
`meta/info.json`), it errors out before touching any hardware rather than
recording into it. When resuming, it also validates that `--fps`,
`--robot-type`, and the enabled cameras match what the dataset was
originally recorded with, since those are fixed at creation time and
can't be changed mid-dataset.

If you instead have two *separate* dataset directories you want to merge
after the fact (e.g. recorded in different sessions under different
names), use [Concatenating datasets](#concatenating-datasets) instead.

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
  2. Verify:   python lerobot.verify_dataset.py --dataset-name <dataset_name>
  3. Train:    python -m lerobot.scripts.lerobot_train \
                 --dataset.repo_id=<dataset_name> \
                 --dataset.root=./<dataset_name> \
                 --dataset.video_backend=pyav \
                 --policy.type=act \
                 --policy.push_to_hub=false \
                 --output_dir=outputs/act_<dataset_name>
  4. Push:     huggingface-cli upload <user>/<dataset_name> ./<dataset_name>
```

See [Verifying dataset motion](#verifying-dataset-motion) and
[Training an ACT policy](#training-an-act-policy) below.

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

### Verifying dataset motion

Checks every episode of a saved dataset for "frozen" arm motion -- i.e.
the 6 UR arm joints barely move across the whole episode, which usually
means the RTDE receive feed got stuck for that episode rather than the
human genuinely holding the arm still (the same check `RecordingSession`
already runs live during recording via `FROZEN_EPISODE_RANGE_RAD`, here
run after the fact over an entire dataset):

```bash
python lerobot.verify_dataset.py --dataset-name open_trashcan
python lerobot.verify_dataset.py --dataset-name open_trashcan --threshold 0.02
```

| Flag | Default | Description |
| --- | --- | --- |
| `--dataset-name` | `open_trashcan` | Dataset directory to check |
| `--threshold` | `0.01` (rad) | Minimum required arm-joint range, below which an episode is flagged `FROZEN` |

Prints a per-episode table sorted by ascending arm range and exits `1` if
any episode is flagged, `0` otherwise -- worth running before spending a
training run on a dataset, and safe to script/gate on the exit code.

### Concatenating datasets

Merges two or more recorded datasets into a single combined dataset, e.g.
to train on multiple recording sessions of the same task together:

```bash
python lerobot.concat_datasets.py open_trashcan_19 open_trashcan_49 \
    --output open_trashcan_19_49
```

| Flag | Default | Description |
| --- | --- | --- |
| `datasets` | (required, positional) | Two or more source dataset directories to concatenate, in order |
| `--output` / `-o` | (required) | Path for the new combined dataset directory -- must not already exist |

Source datasets must share the same fps, robot_type, and features schema
(i.e. recorded with the same robot/camera setup) -- this is enforced by
the underlying `lerobot` package, which does the actual merge: copying
and re-chunking the parquet/video files, unioning the per-dataset task
tables, and re-indexing episodes/frames across the combined dataset.

### Training an ACT policy

This repo only produces the dataset -- training itself is delegated to
the `lerobot` package's own trainer:

```bash
python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=open_trashcan \
    --dataset.root=./open_trashcan \
    --dataset.video_backend=pyav \
    --policy.type=act \
    --policy.push_to_hub=false \
    --output_dir=outputs/act_open_trashcan \
    --job_name=act_open_trashcan \
    --batch_size=8 \
    --steps=100000 \
    --wandb.enable=false
```

Notes:

- `--dataset.root` is required since a dataset recorded by this repo is a
  local directory, not something pulled from the HF Hub.
- `--dataset.video_backend=pyav` matters: the trainer's default backend,
  `torchcodec`, needs FFmpeg shared libraries that aren't always
  installed, and fails the whole run mid-dataloader if they're missing.
  `pyav` (already a `lerobot` dependency) avoids that. If `torchcodec`
  does work in your environment you can drop this flag.
- `--policy.push_to_hub=false` avoids requiring HF write auth (`lerobot`
  defaults this to `true`).
- Checkpoints land at `outputs/act_<dataset_name>/checkpoints/<step|last>/pretrained_model`
  (`--save_freq`, default every 20000 steps, plus a final save aliased
  `last`) -- the path both `eval_act_open_trashcan.py` and
  `infer_act_open_trashcan.py` default to.
- `--batch_size`/`--steps` above are `lerobot`'s own defaults, a
  reasonable single-task baseline; a 50-episode/~5K-frame dataset like
  `open_trashcan` trains in roughly an hour on a single modern GPU.

### Evaluating a trained policy (open-loop)

`eval_act_open_trashcan.py` replays each selected episode's recorded
observations frame-by-frame through the trained policy (same action-queue
behavior as real inference) and compares predicted vs. ground-truth
actions -- a quick sanity check that doesn't touch the robot:

```bash
python eval_act_open_trashcan.py \
    --checkpoint outputs/act_open_trashcan/checkpoints/last/pretrained_model \
    --dataset-name open_trashcan \
    --num-episodes 6 \
    --output outputs/act_open_trashcan_eval.json
```

Prints per-episode MAE per joint and writes the full predicted/ground-truth
comparison to `--output` for later visualization.

### Running inference on the real robot

`infer_act_open_trashcan.py` runs the trained policy closed-loop on the
physical arm: read state + wrist camera, predict an action, stream it via
`servoJ` (same motion pattern as [Replaying an episode](#replaying-an-episode)),
repeat at `--fps`:

```bash
python infer_act_open_trashcan.py \
    --robot-ip 192.168.50.75 \
    --checkpoint outputs/act_open_trashcan/checkpoints/last/pretrained_model \
    --cam-wrist-index 0 --cam-wrist-backend realsense \
    --num-steps 100
```

| Flag | Default | Description |
| --- | --- | --- |
| `--checkpoint` | `outputs/act_open_trashcan/checkpoints/last/pretrained_model` | Trained policy to load |
| `--robot-ip` | `192.168.50.76` | UR7e IP address |
| `--gripper` | `none` | Must match how the dataset was recorded |
| `--cam-wrist-index` | (required) | Camera index for the wrist view -- must match how the dataset was recorded |
| `--cam-wrist-backend` | `realsense` | `usb` or `realsense` |
| `--task` | `task description in here` | Task string the policy was conditioned on |
| `--fps` | `5` | Control rate in Hz -- match the training fps |
| `--num-steps` | `100` | Stop after this many control steps |
| `--max-step-rad` | `0.05` | Safety clamp: max joint delta per control step |
| `--start-speed` / `--start-acceleration` | `0.1` / `0.3` | For the initial blocking move to the policy's first predicted pose |

**Safety:** the arm moves on its own from a live model prediction, with no
collision checking -- clear the workspace, keep a hand near the pendant's
e-stop, and press `Q` (checked once per control step) at the first sign of
trouble. `--max-step-rad` bounds how far any joint can move in a single
step, guarding against a bad/out-of-distribution prediction commanding a
large jump, but it does not know about obstacles. Start with a short
`--num-steps` run to sanity-check the initial move before letting it run
longer.

`eval_act_open_trashcan.py` and `infer_act_open_trashcan.py` are written
for the `open_trashcan` dataset/task specifically (single wrist camera,
7-dim state/action, hardcoded defaults) -- copy and adjust their defaults
for a different dataset.

## Project layout

```
lerobot.record.py           Record entry point script
lerobot.replay.py           Replay entry point script
lerobot.dump_states.py      Joint state dump entry point script
lerobot.verify_dataset.py   Dataset motion-verification entry point script
lerobot.concat_datasets.py  Dataset concatenation entry point script
eval_act_open_trashcan.py   Open-loop ACT policy evaluation against a dataset (open_trashcan-specific)
infer_act_open_trashcan.py  Closed-loop ACT policy inference on the real robot (open_trashcan-specific)
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
outputs/                Training runs: outputs/act_<dataset_name>/checkpoints/<step|last>/pretrained_model
```

"""In-memory buffer for a single episode's steps."""


class EpisodeRecorder:
    """Records a single episode: joint states, actions, gripper, and video."""

    def __init__(self):
        self.timestamps = []
        self.states = []       # observation.state (6 joint pos + 1 gripper)
        self.actions = []      # action (6 joint pos + 1 gripper)
        self.frames = {}       # {cam_name: [frame, frame, ...]}

    def add_step(self, timestamp, state, action, camera_frames):
        self.timestamps.append(timestamp)
        self.states.append(state)
        self.actions.append(action)
        for cam_name, frame in camera_frames.items():
            self.frames.setdefault(cam_name, []).append(frame)

    @property
    def num_steps(self) -> int:
        return len(self.timestamps)

import imageio
import numpy as np


class VideoRecorder:
    def __init__(self, fps=30):
        self.fps = fps
        self.frames = []

    def reset(self):
        self.frames = []

    def record_frame(self, env):
        frame = env.unwrapped.render()
        if not isinstance(frame, np.ndarray):
            # older envs (e.g. lbforaging 2.x) ignore render_mode and need the mode per call
            frame = env.unwrapped.render(mode="rgb_array")
        self.frames.append(frame)

    def save(self, filename):
        imageio.mimsave(f"{filename}", self.frames, fps=self.fps)

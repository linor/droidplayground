import math
import torch

class MotionPlayer:

    def __init__(self, keyframes, device, degrees=False):
        self.device = device

        self.times = torch.tensor(
            [k[0] for k in keyframes],
            device=device,
            dtype=torch.float32,
        )

        # Convert poses if they are specified in degrees
        if degrees:
            poses = [
                [math.radians(angle) for angle in pose]
                for _, pose in keyframes
            ]
        else:
            poses = [
                pose
                for _, pose in keyframes
            ]

        self.poses = torch.tensor(
            poses,
            device=device,
            dtype=torch.float32,
        )

        self.length = self.times[-1]

    def sample(self, t):

        t = t % self.length

        i = torch.searchsorted(self.times, t)

        i = torch.clamp(i, 1, len(self.times)-1)

        t0 = self.times[i-1]
        t1 = self.times[i]

        q0 = self.poses[i-1]
        q1 = self.poses[i]

        alpha = (t-t0)/(t1-t0)

        return q0 + alpha.unsqueeze(-1)*(q1-q0)

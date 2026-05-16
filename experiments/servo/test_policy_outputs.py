#!/usr/bin/env python3
"""
test_policy_outputs.py

Offline sanity-check for exported RSL-RL policy.

It sweeps target angle errors and prints the normalized action and resulting
velocity command.
"""

import argparse
import math

import numpy as np
import torch


def wrap_angle(x):
    return math.atan2(math.sin(x), math.cos(x))


def build_obs(current_velocity, angle_error):
    return torch.tensor(
        [
            angle_error,
            current_velocity,
        ],
        dtype=torch.float32,
    ).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--action-scale", type=float, default=10.8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--velocity", type=float, default=0.0)
    args = parser.parse_args()

    policy = torch.jit.load(args.policy, map_location=args.device)
    policy.eval()

    print("error_deg,error_rad,raw_action,clipped_action,velocity_rad_s")

    current_angle = 0.0

    for error_deg in np.linspace(-180, 180, 25):
        error_angle = math.radians(error_deg)

        obs = build_obs(
            current_velocity=args.velocity,
            angle_error=error_angle,
        ).to(args.device)

        with torch.no_grad():
            action = policy(obs)

        raw_action = float(action.squeeze().cpu().numpy())
        clipped_action = float(np.clip(raw_action, -1.0, 1.0))
        velocity = clipped_action * args.action_scale

        print(
            f"{error_deg:+.1f},"
            f"{math.radians(error_deg):+.4f},"
            f"{raw_action:+.4f},"
            f"{clipped_action:+.4f},"
            f"{velocity:+.4f}"
        )


if __name__ == "__main__":
    main()
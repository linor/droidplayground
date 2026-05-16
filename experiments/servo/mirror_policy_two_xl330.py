#!/usr/bin/env python3
"""
mirror_policy_two_xl330.py

Servo 1: passive/manual target input, torque disabled.
Servo 2: velocity-controlled follower using exported RSL-RL TorchScript policy.

Assumed policy observation:
    [current_angle, current_velocity, target_angle_error]

Adjust build_obs() if your trained observation order differs.
"""

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from dynamixel_sdk import PortHandler, PacketHandler


# XL330 / X-series Protocol 2.0 addresses
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_VELOCITY = 104
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132

TORQUE_DISABLE = 0
TORQUE_ENABLE = 1

OPERATING_MODE_VELOCITY = 1

PROTOCOL_VERSION = 2.0

# XL330 units
POSITION_TICKS_PER_REV = 4096
RAD_PER_TICK = 2.0 * math.pi / POSITION_TICKS_PER_REV

# Goal Velocity unit: 0.229 rpm
RPM_PER_VELOCITY_UNIT = 0.229
RAD_PER_SEC_PER_VELOCITY_UNIT = RPM_PER_VELOCITY_UNIT * 2.0 * math.pi / 60.0

XL330_MAX_RAD_S = 10.8


@dataclass
class DynamixelBus:
    device: str
    baudrate: int

    def __post_init__(self):
        self.port = PortHandler(self.device)
        self.packet = PacketHandler(PROTOCOL_VERSION)

        if not self.port.openPort():
            raise RuntimeError(f"Failed to open port: {self.device}")
        if not self.port.setBaudRate(self.baudrate):
            raise RuntimeError(f"Failed to set baudrate: {self.baudrate}")

    def close(self):
        self.port.closePort()

    def check(self, dxl_id: int, result: int, error: int, context: str):
        if result != 0:
            raise RuntimeError(
                f"[ID {dxl_id}] {context}: "
                f"{self.packet.getTxRxResult(result)}"
            )
        if error != 0:
            raise RuntimeError(
                f"[ID {dxl_id}] {context}: "
                f"{self.packet.getRxPacketError(error)}"
            )

    def write1(self, dxl_id: int, addr: int, value: int, context: str):
        result, error = self.packet.write1ByteTxRx(
            self.port, dxl_id, addr, value
        )
        self.check(dxl_id, result, error, context)

    def write4(self, dxl_id: int, addr: int, value: int, context: str):
        # Dynamixel SDK expects unsigned representation for signed 32-bit values.
        value_u32 = np.uint32(np.int32(value)).item()
        result, error = self.packet.write4ByteTxRx(
            self.port, dxl_id, addr, value_u32
        )
        self.check(dxl_id, result, error, context)

    def read4_signed(self, dxl_id: int, addr: int, context: str) -> int:
        value, result, error = self.packet.read4ByteTxRx(
            self.port, dxl_id, addr
        )
        self.check(dxl_id, result, error, context)
        return np.int32(np.uint32(value)).item()

    def set_velocity_mode(self, dxl_id: int):
        self.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, "disable torque")
        self.write1(dxl_id, ADDR_OPERATING_MODE, OPERATING_MODE_VELOCITY, "set velocity mode")

    def set_torque(self, dxl_id: int, enabled: bool):
        self.write1(
            dxl_id,
            ADDR_TORQUE_ENABLE,
            TORQUE_ENABLE if enabled else TORQUE_DISABLE,
            "set torque",
        )

    def read_position_rad(self, dxl_id: int) -> float:
        ticks = self.read4_signed(dxl_id, ADDR_PRESENT_POSITION, "read position")
        return ticks * RAD_PER_TICK

    def read_velocity_rad_s(self, dxl_id: int) -> float:
        raw = self.read4_signed(dxl_id, ADDR_PRESENT_VELOCITY, "read velocity")
        return raw * RAD_PER_SEC_PER_VELOCITY_UNIT

    def write_goal_velocity_rad_s(self, dxl_id: int, velocity_rad_s: float):
        velocity_rad_s = float(np.clip(velocity_rad_s, -XL330_MAX_RAD_S, XL330_MAX_RAD_S))
        raw = int(round(velocity_rad_s / RAD_PER_SEC_PER_VELOCITY_UNIT))
        self.write4(dxl_id, ADDR_GOAL_VELOCITY, raw, "write goal velocity")


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def build_obs(
    follower_angle: float,
    follower_velocity: float,
    target_angle: float,
    obs_scale_velocity: float = 1.0,
) -> torch.Tensor:
    """
    Match this to your Isaac Lab observation order.

    Current observation order:
        [wrapped target angle error, follower joint velocity]
    """
    error = wrap_angle(target_angle - follower_angle)

    obs = torch.tensor(
        [
            error,
            follower_velocity * obs_scale_velocity,
        ],
        dtype=torch.float32,
    )

    return obs.unsqueeze(0)

def load_policy(path: str, device: str):
    policy = torch.jit.load(path, map_location=device)
    policy.eval()
    return policy


def policy_to_velocity(action: torch.Tensor, action_scale: float) -> float:
    """
    RSL-RL exported policy usually outputs normalized action.
    Isaac Lab JointVelocityActionCfg(scale=10.8) means:
        velocity_command = action * 10.8
    """
    a = float(action.squeeze().detach().cpu().numpy())
    a = float(np.clip(a, -1.0, 1.0))
    return a * action_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=57600)
    parser.add_argument("--target-id", type=int, default=1)
    parser.add_argument("--follower-id", type=int, default=2)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--action-scale", type=float, default=10.8)
    parser.add_argument("--device-torch", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    policy = load_policy(args.policy, args.device_torch)

    bus = DynamixelBus(args.device, args.baudrate)

    dt = 1.0 / args.rate

    try:
        # Target servo: torque off, just used as position sensor.
        bus.set_torque(args.target_id, False)

        # Follower servo: velocity mode.
        bus.set_velocity_mode(args.follower_id)
        if not args.dry_run:
            bus.set_torque(args.follower_id, True)

        print("Running. Ctrl+C to stop.")
        print("target_rad follower_rad error_rad follower_vel action_vel")

        while True:
            t0 = time.perf_counter()

            target_angle = bus.read_position_rad(args.target_id)
            follower_angle = bus.read_position_rad(args.follower_id)
            follower_velocity = bus.read_velocity_rad_s(args.follower_id)

            obs = build_obs(
                follower_angle=follower_angle,
                follower_velocity=follower_velocity,
                target_angle=target_angle,
            ).to(args.device_torch)

            with torch.no_grad():
                action = policy(obs)

            goal_velocity = policy_to_velocity(action, args.action_scale)

            if not args.dry_run:
                bus.write_goal_velocity_rad_s(args.follower_id, goal_velocity)

            err = wrap_angle(target_angle - follower_angle)

            print(
                f"{target_angle:+.3f} "
                f"{follower_angle:+.3f} "
                f"{err:+.3f} "
                f"{follower_velocity:+.3f} "
                f"{goal_velocity:+.3f}"
            )

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        print("\nStopping.")

    finally:
        try:
            bus.write_goal_velocity_rad_s(args.follower_id, 0.0)
            bus.set_torque(args.follower_id, False)
        finally:
            bus.close()


if __name__ == "__main__":
    main()
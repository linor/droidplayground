#!/usr/bin/env python3
"""
imu_calibration_check.py

Standalone diagnostic/calibration tool for the Adafruit LSM6DSOX + LIS3MDL
IMU (product 4517) used by robot_deploy.py. Run this BEFORE trusting the
IMU for a real balance policy -- imu_sensor.py's axis remap (chip frame ->
robot base_link frame) is derived from base_link's axes as inspected in
Isaac Sim (see that module's docstring) but has NOT yet been verified
against the real sensor, and a wrong sign here won't crash anything on its
own -- it'll just make a balance policy actively push the robot the WRONG
way the instant it starts to tip, which is worse than doing nothing.

This imports imu_sensor.py directly (the exact same code robot_deploy.py
uses), so whatever you observe here is exactly what the control loop will
see -- there's no separate/divergent reading path to get out of sync.

USAGE
-----
Guided calibration sequence (recommended the first time):
    python3 imu_calibration_check.py --config robot_config_qmini.json

Free-running live monitor (Ctrl+C to stop) -- useful once you trust the
axis mapping and just want to watch it while moving the robot around:
    python3 imu_calibration_check.py --config robot_config_qmini.json --live

WHAT TO DO WITH THE RESULTS
-----------------------------
Each guided step now checks its result against a specific expected
axis+sign (derived from base_link's confirmed Isaac Sim axes -- see
imu_sensor.py's docstring) and prints OK/WARN. If a step comes back WARN,
edit "imu_axis_remap" in your robot_config.json (flip the sign, or swap
which chip axis index feeds that robot axis slot) and rerun this script
until every step matches.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import imu_sensor

EARTH_GRAVITY_MPS2 = 9.80665


def sample_avg(sensor: imu_sensor.ImuSensor, n: int = 20, dt: float = 0.05):
    """Averages n readings ~dt seconds apart to cut sensor noise. Returns
    an imu_sensor.ImuReading with each field replaced by its mean."""
    accel_robot = [0.0, 0.0, 0.0]
    gyro_robot = [0.0, 0.0, 0.0]
    accel_chip = [0.0, 0.0, 0.0]
    gyro_chip = [0.0, 0.0, 0.0]
    for _ in range(n):
        r = sensor.read()
        for i in range(3):
            accel_robot[i] += r.accel_robot_mps2[i] / n
            gyro_robot[i] += r.gyro_robot_rads[i] / n
            accel_chip[i] += r.accel_chip_mps2[i] / n
            gyro_chip[i] += r.gyro_chip_rads[i] / n
        time.sleep(dt)
    return imu_sensor.ImuReading(accel_robot, gyro_robot, accel_chip, gyro_chip)


def fmt3(v, unit=""):
    return f"({v[0]:+7.3f}, {v[1]:+7.3f}, {v[2]:+7.3f}){' ' + unit if unit else ''}"


def print_reading(label, r: imu_sensor.ImuReading):
    print(f"  {label}")
    print(f"    chip  accel: {fmt3(r.accel_chip_mps2, 'm/s^2')}   gyro: {fmt3([math.degrees(x) for x in r.gyro_chip_rads], 'deg/s')}")
    print(f"    robot accel: {fmt3(r.accel_robot_mps2, 'm/s^2')}   gyro: {fmt3([math.degrees(x) for x in r.gyro_robot_rads], 'deg/s')}")
    norm = math.sqrt(sum(a * a for a in r.accel_robot_mps2))
    gravity_dir = [-a / norm for a in r.accel_robot_mps2] if norm > 1e-6 else [0, 0, -1]
    print(f"    robot gravity_dir (matches projected_gravity_b): {fmt3(gravity_dir)}   |accel|={norm:6.3f} m/s^2 (expect ~{EARTH_GRAVITY_MPS2:.2f})")


def wait_for_enter(prompt):
    input(f"\n{prompt}\nPress Enter when ready (or Ctrl+C to quit)... ")


def step_level(sensor):
    wait_for_enter(
        "STEP 1/4 -- LEVEL: Place the robot (or just the IMU, rigidly "
        "attached as mounted) flat and stationary, base_link as level as "
        "you can manage by eye."
    )
    r = sample_avg(sensor)
    print_reading("Result:", r)
    norm = math.sqrt(sum(a * a for a in r.accel_robot_mps2))
    gravity_dir = [-a / norm for a in r.accel_robot_mps2] if norm > 1e-6 else [0, 0, -1]
    ok_mag = 9.0 <= norm <= 10.6
    ok_level = gravity_dir[2] < -0.9  # mostly -Z, allows a few degrees of "level by eye" error
    print(f"  [{'OK' if ok_mag else 'WARN'}] accelerometer magnitude {'looks sane' if ok_mag else 'is OFF -- check wiring/I2C, or a units bug (deg/s vs rad/s, g vs m/s^2)'}")
    print(f"  [{'OK' if ok_level else 'WARN'}] gravity_dir z={gravity_dir[2]:+.3f} {'is close to -1, consistent with Z axis mapped correctly and robot roughly level' if ok_level else 'is NOT close to -1 -- either the robot was not actually level, or imu_axis_remap Z is wrong (check for a swapped axis or missing sign flip)'}")
    gyro_mag_dps = math.degrees(math.sqrt(sum(g * g for g in r.gyro_robot_rads)))
    ok_gyro = gyro_mag_dps < 5.0
    print(f"  [{'OK' if ok_gyro else 'WARN'}] gyro magnitude at rest = {gyro_mag_dps:.2f} deg/s {'(small bias, normal)' if ok_gyro else '(unexpectedly large for a stationary sensor -- check for vibration or a bad gyro reading)'}")


def step_tilt(sensor, label, instruction, expected_axis, expected_sign, why):
    wait_for_enter(f"STEP {label} -- {instruction}")
    r = sample_avg(sensor)
    print_reading("Result:", r)
    norm = math.sqrt(sum(a * a for a in r.accel_robot_mps2))
    gravity_dir = [-a / norm for a in r.accel_robot_mps2] if norm > 1e-6 else [0, 0, -1]
    dominant_i = max(range(3), key=lambda i: abs(gravity_dir[i]))
    axis_name = ["X", "Y", "Z"][dominant_i]
    expected_name = ["X", "Y", "Z"][expected_axis]
    ok = dominant_i == expected_axis and (gravity_dir[expected_axis] * expected_sign) > 0
    print(
        f"  Axis that moved the most: robot-frame {axis_name} "
        f"(gravity_dir[{axis_name}]={gravity_dir[dominant_i]:+.3f})."
    )
    print(
        f"  [{'OK' if ok else 'WARN'}] expected robot-frame {expected_name} to go "
        f"{'negative' if expected_sign < 0 else 'positive'} ({why}). "
        f"{'Matches.' if ok else 'Does NOT match -- fix imu_axis_remap in robot_config.json (wrong axis index and/or sign for this slot) and rerun.'}"
    )


def step_yaw_rotation(sensor):
    wait_for_enter(
        "STEP 4/4 -- YAW: Slowly rotate the robot in the direction it "
        "would turn to walk to the RIGHT (holding it otherwise level). "
        "Keep rotating steadily while sampling starts."
    )
    r = sample_avg(sensor, n=15, dt=0.05)
    print_reading("Result:", r)
    gyro_dps = [math.degrees(g) for g in r.gyro_robot_rads]
    dominant_i = max(range(3), key=lambda i: abs(gyro_dps[i]))
    axis_name = ["X", "Y", "Z"][dominant_i]
    # Confirmed 2026-08-13: with base_link's +X=left, +Y=rear, +Z=up (Isaac
    # Sim), standard right-hand-rule yaw means +Z angular velocity = CCW
    # from above = a LEFT turn, so a RIGHT turn should read NEGATIVE Z.
    ok = dominant_i == 2 and gyro_dps[2] < 0
    print(
        f"  Axis with the largest angular velocity: robot-frame {axis_name} "
        f"({gyro_dps[dominant_i]:+.2f} deg/s)."
    )
    print(
        f"  [{'OK' if ok else 'WARN'}] expected robot-frame Z to be dominant "
        f"and NEGATIVE for a right turn (right-hand rule: +Z ang_vel = CCW "
        f"from above = a left turn). "
        f"{'Matches.' if ok else 'Does NOT match -- fix imu_axis_remap Z slot in robot_config.json and rerun.'}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="robot_config.json (reads imu_i2c_address/imu_axis_remap)")
    parser.add_argument("--live", action="store_true", help="Skip the guided steps -- just print live readings until Ctrl+C")
    parser.add_argument("--rate-hz", type=float, default=5.0, help="Print rate for --live mode")
    args = parser.parse_args()

    raw = json.loads(args.config.read_text())
    address = raw.get("imu_i2c_address")
    axis_remap = raw.get("imu_axis_remap") or imu_sensor.AXIS_REMAP
    print(f"Using imu_axis_remap={axis_remap} from {args.config}"
          f"{' (module default -- config had none)' if not raw.get('imu_axis_remap') else ''}")

    sensor = imu_sensor.ImuSensor(address=address, axis_remap=axis_remap)

    if args.live:
        print("Live IMU monitor -- Ctrl+C to stop.\n")
        try:
            while True:
                r = sensor.read()
                print_reading(f"t={time.strftime('%H:%M:%S')}", r)
                print()
                time.sleep(1.0 / args.rate_hz)
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    print(__doc__)
    try:
        step_level(sensor)
        step_tilt(
            sensor, "2/4",
            "PITCH: Tip the robot so its FRONT (the direction it walks "
            "forward) points DOWN, and hold steady.",
            expected_axis=1, expected_sign=-1,
            why="base_link's +Y points toward the REAR (confirmed in Isaac "
                "Sim), so the front is -Y -- tipping the front down should "
                "move gravity_dir toward -Y",
        )
        step_tilt(
            sensor, "3/4",
            "ROLL: Tip the robot so its RIGHT side points DOWN, and hold steady.",
            expected_axis=0, expected_sign=-1,
            why="base_link's +X points toward the LEFT leg (confirmed in "
                "Isaac Sim), so the right side is -X -- tipping right-down "
                "should move gravity_dir toward -X",
        )
        step_yaw_rotation(sensor)
    except KeyboardInterrupt:
        print("\nStopped.")
        return

    print(
        "\nDone. If every step above matched its expected axis (and you've "
        "independently confirmed base_link's forward/left/up directions in "
        "Isaac Sim -- see imu_sensor.py's docstring), imu_axis_remap is "
        "verified. If anything looked wrong, edit imu_axis_remap in "
        f"{args.config} and rerun this script."
    )


if __name__ == "__main__":
    main()

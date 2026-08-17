#!/usr/bin/env python3
"""
imu_sensor.py

Shared IMU reading + axis-remap code for the Adafruit LSM6DSOX + LIS3MDL
board (Adafruit product 4517: https://www.adafruit.com/product/4517),
connected via I2C to the Raspberry Pi running robot_deploy.py.

Both robot_deploy.py (the real control loop) and imu_calibration_check.py
(the standalone verification tool) import THIS module rather than talking
to the sensor directly, so they're guaranteed to agree on the exact same
reading/remap code path -- if you ever change the axis remap, both stay in
sync automatically.

WHAT THIS PROVIDES
-------------------
  - Raw chip-frame accelerometer (m/s^2) and gyroscope (rad/s) reads from
    the LSM6DSOX. The LIS3MDL magnetometer on the same board isn't used --
    yaw heading isn't needed for a "keep the body level" balance reward,
    only tilt, which the accelerometer already gives you.
  - A configurable axis remap from the chip's own silkscreen-labeled X/Y/Z
    to the robot's base_link frame (whatever qmini_urdf-2legs.usda calls
    "forward"/"left"/"up" for the articulation in Isaac Lab).
  - read_robot_frame(), returning (gravity_dir, ang_vel_rad_s) already
    remapped, matching QminiLegEnv._get_observations()'s
    (projected_gravity_b, root_ang_vel_b) exactly -- including the sign
    flip needed to turn a raw accelerometer reading (which points AWAY
    from gravity when at rest -- it measures the reaction force holding
    the sensor up, not gravity itself) into a gravity DIRECTION vector
    (which points the way gravity actually pulls, i.e. down, when level).

AXIS_REMAP DERIVATION -- confirmed against Isaac Sim, NOT yet against real hardware
-------------------------------------------------------------------------------------
base_link's actual local frame, per inspecting its axis gizmo in Isaac Sim
(qmini_urdf-2legs.usda): +X points toward the LEFT leg, +Y points toward
the REAR, +Z is up. This is NOT the ROS-style "X forward, Y left"
convention an earlier version of this file assumed -- it's rotated 90 deg
from that about Z.

Per your IMU mounting note: chip X+ = robot's rear, chip Y+ = towards the
right leg, chip Z+ = up. Matching physical directions between the two
frames (chip X+ / base Y+ both = "rear"; chip Y+ = "right" = the opposite
physical direction to base X+ = "left"; chip Z+ / base Z+ both = "up")
gives:

    base_X (points left)  = -chip_Y   (chip Y+ = right = -base_X)
    base_Y (points rear)  = +chip_X   (chip X+ = rear = +base_Y)
    base_Z (points up)    = +chip_Z

This is a proper rotation (a 90 deg rotation about Z, determinant +1 --
not a reflection), which is what you'd expect from a rigid IMU mount, so
the derivation is at least internally consistent. It has NOT been
independently verified against the real sensor yet, though -- before
trusting it for a real balance policy:

  1. Run imu_calibration_check.py and physically tilt the real robot in
     known directions (nose down, right side down, yaw rotation), and
     check the printed remapped output moves the way this derivation
     predicts (see that script's per-step hints).
  2. Only then trust AXIS_REMAP for a real balance policy -- a wrong sign
     here won't crash anything or throw an error, it'll just make the
     policy actively push the robot the WRONG way the instant it starts
     to tip, which is worse than having no balance behavior at all.

WIRING: Raspberry Pi native GPIO vs. MCP2221 USB-to-I2C (product 4471)
--------------------------------------------------------------------------
This module uses Blinka's portable `board`/`busio` API, so the actual
sensor-reading code below is IDENTICAL either way -- Blinka picks the
backend from the BLINKA_MCP2221 environment variable (set automatically
below, before `import board`, so this "just works" once the adapter is
plugged in). Switching between the two needs no code changes, only setup:

  Native Pi GPIO I2C (original wiring):
    pip3 install adafruit-blinka adafruit-circuitpython-lsm6ds
    Enable I2C: raspi-config -> Interface Options -> I2C.

  MCP2221 USB-to-I2C bridge (Adafruit product 4471 -- moves the IMU off
  the Pi's onboard I2C entirely, the fix for I2C corruption coupling in
  from motor PWM noise/grounding, see ImuTelemetryFault's docstring).
  Identify yours first with `lsusb` -- Microchip's MCP2221(a) reports as
  04d8:00dd; if you get a different adapter later, adjust the VID:PID and
  BLINKA_* variable to match (Blinka supports several USB bridges, e.g.
  FT232H uses 0403:6014 and BLINKA_FT232H instead):
    pip3 install adafruit-blinka adafruit-circuitpython-lsm6ds hidapi
    (MCP2221 is a USB HID device, not a libusb one like FT232H -- hidapi
    is Blinka's dependency for talking to it, no pyftdi/libusb needed.)
    Udev rule so it's usable without root -- create
    /etc/udev/rules.d/99-mcp2221.rules containing:
        SUBSYSTEM=="usb", ATTRS{idVendor}=="04d8", ATTRS{idProduct}=="00dd", GROUP="plugdev", MODE="0666"
    then: sudo udevadm control --reload-rules && sudo udevadm trigger
    (unplug/replug the adapter after this). Make sure your user is in the
    `plugdev` group (or adjust GROUP= above), or just use MODE="0666" as
    shown, which Adafruit's own guides use for simplicity.
    Wire the IMU's SCL/SDA/GND/VIN to the breakout's labeled SCL/SDA/GND/
    3V3 (or 5V, check the LSM6DSOX+LIS3MDL board's supported range) pins.

  One thing to watch either way: USB-bridged I2C has meaningfully higher
  per-transaction latency than native GPIO I2C (USB round-trip overhead).
  IMU read time isn't currently broken out in robot_deploy.py's
  policy_ms/bus_ms timing log (it happens inside build_obs(), before
  policy_ms starts timing) -- if control-loop overrun becomes a concern
  again after switching, that's a place worth adding visibility, since
  right now it's silently part of the unaccounted loop time.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Sequence

# Must be set before `import board` -- Blinka reads this at import time to
# decide whether to talk to the Pi's native GPIO or bridge through an
# MCP2221 USB-to-I2C adapter (Adafruit product 4471, confirmed via `lsusb`
# as 04d8:00dd Microchip MCP2221(a)). setdefault (not a plain assignment)
# so an explicit BLINKA_MCP2221 already set in the shell -- e.g. "0" to
# force native GPIO -- still wins over this default.
# os.environ.setdefault("BLINKA_MCP2221", "1")

try:
    import board
    import busio
    from adafruit_lsm6ds.lsm6dsox import LSM6DSOX
    _IMU_LIB_AVAILABLE = True
except ImportError as e:  # pragma: no cover - depends on target machine
    _IMU_LIB_AVAILABLE = False
    _IMU_IMPORT_ERROR = e

# Default LSM6DSOX I2C address on the Adafruit 4517 board. The board's SDO
# pin can be pulled high to switch to 0x6B if you need a second IMU (or a
# conflicting device) on the same bus -- pass address=0x6B in that case.
DEFAULT_LSM6DSOX_ADDRESS = 0x6A

# (source_index_into_chip_xyz, sign) for each of base_link's own (X, Y, Z)
# axes -- X points toward the LEFT leg, Y points toward the REAR, Z is up
# (confirmed in Isaac Sim, see the derivation in the module docstring
# above). NOT yet verified against the real sensor -- run
# imu_calibration_check.py before trusting this.
AXIS_REMAP: list[tuple[int, float]] = [
    (1, -1.0),  # base_link X (left) = -chip Y  (chip Y+ is towards the right leg)
    (0, 1.0),   # base_link Y (rear) =  chip X  (chip X+ is the robot's rear)
    (2, 1.0),   # base_link Z (up)   =  chip Z
]


def remap_to_robot_frame(chip_xyz: Sequence[float], axis_remap=AXIS_REMAP) -> list[float]:
    """Reorders/flips a 3-vector from the IMU chip's own frame into the
    robot's base_link frame, per `axis_remap`."""
    return [sign * chip_xyz[idx] for idx, sign in axis_remap]


class ImuTelemetryFault(RuntimeError):
    """Raised when the accelerometer disagrees with the gyroscope about how
    much the robot has rotated, PERSISTENTLY across several consecutive
    reads -- consistent with an actually broken/disconnected sensor, not
    the normal brief disagreement any single accelerometer reading has
    during a real shock or sudden deceleration (see read_robot_frame()'s
    complementary filter, which absorbs those on its own without needing
    to raise this). Mirrors robot_deploy.py's MotorBus._validate_reading /
    TelemetryFaultError, which catches the analogous problem for motor
    position readings."""


# Per-step accelerometer/gyro consistency threshold -- see
# read_robot_frame(). A rotation of `angle` over `dt` seconds needs an
# average angular rate of angle/dt -- GRAVITY_JUMP_MARGIN allows the
# accelerometer-implied rate to exceed the gyro's own reading by this
# factor before that single step counts as a "mismatch" (which only turns
# into an ImuTelemetryFault after SUSTAINED_MISMATCH_STEPS in a row -- see
# below), and GRAVITY_JUMP_SLACK_DEG is an absolute floor (so accelerometer
# noise near zero gyro rate, e.g. while genuinely standing still, never
# counts as a mismatch).
GRAVITY_JUMP_MARGIN = 5.0
GRAVITY_JUMP_SLACK_DEG = 15.0

# How many CONSECUTIVE per-step mismatches (see above) before
# read_robot_frame() gives up and raises ImuTelemetryFault instead of
# continuing to filter through it. 5 steps = 100ms at the 20ms control
# rate this was tuned against -- real shocks/decelerations resolve much
# faster than that (both incidents that motivated this design, 2026-08-16,
# were single-step mismatches immediately preceded and followed by
# consistent readings); a mismatch that persists past this is much more
# likely a genuinely broken/disconnected sensor.
SUSTAINED_MISMATCH_STEPS = 5

# Complementary filter weight: how much each new accelerometer reading
# corrects the gyro-propagated gravity_dir estimate, per step. Small on
# purpose -- a raw accelerometer only measures gravity direction correctly
# when the sensor isn't accelerating translationally, which is violated by
# ANY shock or sudden deceleration (a real, unavoidable event on a real
# robot: footstep impacts during actual standing/walking will cause this
# too, not just manual handling). The gyro has no such blind spot (it
# measures rotation directly), so it carries the estimate through those
# moments; the accelerometer's job is just to correct long-term gyro drift
# during otherwise-quiet periods. 0.02 per 20ms step is roughly a 1-second
# correction time constant (dt/weight) -- fast enough to track real gyro
# bias over a run, slow enough that a single bad/shock-corrupted sample
# only nudges the estimate a little instead of defining it outright.
ACCEL_CORRECTION_WEIGHT = 0.02


def _rotate_vector_by_gyro(v, gyro_rad_s, dt):
    """Propagates a WORLD-fixed vector `v` (expressed in the body/sensor
    frame -- e.g. gravity_dir) forward by `dt` seconds of rotation at
    `gyro_rad_s` (the body's own measured angular velocity, body frame).

    Uses Rodrigues' rotation formula with the gyro reading NEGATED -- a
    world-fixed vector expressed in a frame that's itself rotating at +omega
    appears, from inside that frame, to rotate at -omega (standard rotating
    -frame kinematics: d(v_body)/dt = -omega x v_body). Verified numerically
    against a full rotation-matrix reference before relying on it here --
    easy to get backwards, and a sign error would make the filter track
    rotation in the wrong direction rather than fail loudly.
    """
    wx, wy, wz = -gyro_rad_s[0], -gyro_rad_s[1], -gyro_rad_s[2]
    mag = math.sqrt(wx * wx + wy * wy + wz * wz)
    if mag < 1e-9:
        return list(v)
    angle = mag * dt
    kx, ky, kz = wx / mag, wy / mag, wz / mag
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    k_cross_v = (ky * v[2] - kz * v[1], kz * v[0] - kx * v[2], kx * v[1] - ky * v[0])
    k_dot_v = kx * v[0] + ky * v[1] + kz * v[2]
    k = (kx, ky, kz)
    return [v[i] * cos_a + k_cross_v[i] * sin_a + k[i] * k_dot_v * (1 - cos_a) for i in range(3)]


@dataclass
class ImuReading:
    accel_robot_mps2: list  # [x, y, z], base_link frame, m/s^2 (raw accelerometer, NOT gravity-direction)
    gyro_robot_rads: list   # [x, y, z], base_link frame, rad/s
    accel_chip_mps2: list   # raw, chip frame -- kept for diagnostics/calibration
    gyro_chip_rads: list    # raw, chip frame -- kept for diagnostics/calibration


class ImuSensor:
    """Wraps an Adafruit LSM6DSOX (product 4517) over I2C."""

    def __init__(self, address: Optional[int] = None, axis_remap=AXIS_REMAP):
        if not _IMU_LIB_AVAILABLE:
            raise RuntimeError(
                f"Adafruit CircuitPython IMU libraries could not be imported "
                f"({_IMU_IMPORT_ERROR}). On the Raspberry Pi, install with: "
                f"pip3 install adafruit-blinka adafruit-circuitpython-lsm6ds "
                f"-- and make sure I2C is enabled (raspi-config -> Interface "
                f"Options -> I2C)."
            )
        self.axis_remap = axis_remap
        i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = LSM6DSOX(i2c, address=address or DEFAULT_LSM6DSOX_ADDRESS)

        # State for read_robot_frame()'s complementary filter + sustained-
        # mismatch check -- see ImuTelemetryFault and ACCEL_CORRECTION_WEIGHT.
        self._filtered_gravity_dir = None
        self._last_read_time = None
        self._consecutive_mismatches = 0

    def read(self) -> ImuReading:
        # adafruit_lsm6ds reports .acceleration in m/s^2 and .gyro in
        # rad/s (SI units, per the library's own docstrings) -- if your
        # installed version differs, imu_calibration_check.py's at-rest
        # sanity check (accel magnitude should read ~9.81 m/s^2, gyro
        # near 0) will make a unit mismatch obvious immediately.
        accel_chip = list(self.sensor.acceleration)
        gyro_chip = list(self.sensor.gyro)

        return ImuReading(
            accel_robot_mps2=remap_to_robot_frame(accel_chip, self.axis_remap),
            gyro_robot_rads=remap_to_robot_frame(gyro_chip, self.axis_remap),
            accel_chip_mps2=accel_chip,
            gyro_chip_rads=gyro_chip,
        )

    def read_robot_frame(self):
        """Returns (gravity_dir, ang_vel_rad_s), matching
        QminiLegEnv._get_observations()'s (projected_gravity_b,
        root_ang_vel_b) exactly: gravity_dir is a unit vector pointing the
        way gravity pulls (down when level), NOT the raw accelerometer
        reading (which points the opposite way at rest -- see the module
        docstring).

        gravity_dir is a COMPLEMENTARY-FILTERED estimate, not the raw
        per-sample accelerometer reading: each step, the previous estimate
        is propagated forward using ONLY the gyroscope (immune to linear
        acceleration), then nudged gently toward the new raw accelerometer
        reading (ACCEL_CORRECTION_WEIGHT). This matters because a raw
        accelerometer only measures gravity direction correctly when the
        sensor isn't accelerating translationally -- which briefly fails
        during ANY shock or sudden deceleration, including completely
        normal ones (a footstep impact during real standing/walking, not
        just someone tilting the robot by hand). Trusting each raw sample
        outright (this function's original implementation) meant every one
        of those normal events looked identical to real sensor corruption.
        ang_vel_rad_s is NOT filtered -- the gyro is directly trustworthy
        (no translational-acceleration blind spot), so there's nothing to
        correct."""
        r = self.read()
        norm = math.sqrt(sum(a * a for a in r.accel_robot_mps2))
        if norm < 1e-6:
            # Sensor gave an all-zero reading (comms failure) -- fall back
            # to "level" rather than dividing by ~0 and returning garbage.
            accel_gravity_dir = [0.0, 0.0, -1.0]
        else:
            accel_gravity_dir = [-a / norm for a in r.accel_robot_mps2]

        now = time.perf_counter()

        if self._filtered_gravity_dir is None:
            # First reading ever -- no prior estimate to propagate, and no
            # dt to propagate it over. Bootstrap by trusting the raw
            # reading outright (reasonable: startup_sequence() calibrates
            # with the robot held still).
            self._filtered_gravity_dir = accel_gravity_dir
            self._last_read_time = now
            return list(self._filtered_gravity_dir), r.gyro_robot_rads

        dt = now - self._last_read_time
        self._last_read_time = now

        gyro_predicted_dir = _rotate_vector_by_gyro(self._filtered_gravity_dir, r.gyro_robot_rads, dt)

        # Sustained-mismatch check: does the raw accelerometer disagree
        # with where the gyro says we ended up? A LONE mismatch is
        # expected/normal (see docstring above) and the filter blend below
        # absorbs it on its own -- this only raises once it persists for
        # SUSTAINED_MISMATCH_STEPS in a row, which a real shock shouldn't
        # do but a genuinely broken sensor would.
        gyro_mag_rads = math.sqrt(sum(g * g for g in r.gyro_robot_rads))
        cos_angle = max(-1.0, min(1.0, sum(a * b for a, b in zip(accel_gravity_dir, gyro_predicted_dir))))
        mismatch_deg = math.degrees(math.acos(cos_angle))
        plausible_deg = math.degrees(gyro_mag_rads * dt) * GRAVITY_JUMP_MARGIN + GRAVITY_JUMP_SLACK_DEG
        if mismatch_deg > plausible_deg:
            self._consecutive_mismatches += 1
            if self._consecutive_mismatches >= SUSTAINED_MISMATCH_STEPS:
                raise ImuTelemetryFault(
                    f"accelerometer vs. gyro mismatch ({mismatch_deg:.1f} deg vs. "
                    f"{plausible_deg:.1f} deg plausible) has persisted for "
                    f"{self._consecutive_mismatches} consecutive reads -- this is "
                    f"well past a normal shock/deceleration blip (see "
                    f"ImuTelemetryFault's docstring), treating as a broken sensor."
                )
        else:
            self._consecutive_mismatches = 0

        blended = [
            (1.0 - ACCEL_CORRECTION_WEIGHT) * g + ACCEL_CORRECTION_WEIGHT * a
            for g, a in zip(gyro_predicted_dir, accel_gravity_dir)
        ]
        blended_norm = math.sqrt(sum(x * x for x in blended))
        self._filtered_gravity_dir = (
            [x / blended_norm for x in blended] if blended_norm > 1e-6 else gyro_predicted_dir
        )
        return list(self._filtered_gravity_dir), r.gyro_robot_rads

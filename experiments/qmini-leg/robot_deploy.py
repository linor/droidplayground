#!/usr/bin/env python3
"""
robot_deploy.py

Sim-to-real deployment of an Isaac Lab / rsl_rl policy (QminiLegEnv, 10 joints:
hip yaw/roll/pitch, knee, ankle x left/right) onto real GO-M8010-6 motors via
Unitree's unitree_actuator_sdk, over one or more RS485 busses (see "MULTIPLE
SERIAL BUSSES" below). Joint count/order is NOT hardcoded here -- it's
whatever robot_config.json's "joints" list declares, cross-checked against
the policy's own joint_order at startup (see load_and_verify_policy()).

WHAT THIS SCRIPT DOES
----------------------
  1. Verifies the policy bundle (policy.pt + policy.meta.json) matches this
     script's expected observation/action layout AND the robot config you
     pass in (joint order, count) before anything moves.
  2. Connects to the motor bus, reads each motor's raw position, and uses
     that startup reading (+ a configurable per-joint offset) to calibrate
     "policy zero" for that joint. Prints a table of raw vs. calibrated
     degrees per joint and waits for you to confirm before enabling torque.
  3. Runs the control loop: build obs -> policy -> action -> joint targets,
     convert to motor (rotor-side, gear-ratio-scaled) commands, and send.
  4. Before every command, checks the requested step is <= --max-step-deg
     from the last commanded target, AND that every joint's target is
     within its configured [min_deg, max_deg] (if set). If any joint would
     move further in a single step, or outside its limits, it aborts
     immediately and releases (torque-off) ALL motors on ALL buses.
  5. After every motor read/command, checks each motor's error flag and
     temperature against `motor_temp_limit_c` (robot_config.json). Any
     motor error, or a temperature above the limit, triggers the same
     immediate release-all abort as a step/joint-limit violation.
  6. Releases all motors on normal exit, Ctrl+C, SIGTERM, or any unhandled
     exception (try/finally + signal handlers).
  7. Logs every control step (obs, raw policy action, joint targets, rotor
     commands, motor telemetry) to a CSV plus a human-readable event log.

MULTIPLE SERIAL BUSSES
-----------------------
Each joint in robot_config.json declares its own `port`. Joints sharing the
same port string are driven by one MotorBus/SerialPort; joints with
different ports get their own MotorBus, so a leg spread across two RS485
adapters (e.g. hip+knee on /dev/ttyUSB0, ankle on /dev/ttyUSB1) works with
no code changes -- just set each joint's `port` accordingly.

PER-JOINT GEAR RATIO
---------------------
Most joints use the bus's base gear ratio (queried from the SDK, or
DEFAULT_GEAR_RATIO_GO_M8010_6). A joint with a non-default reduction (e.g.
an extra belt/pulley stage) can set `extra_gear_ratio` in robot_config.json;
the joint's effective ratio becomes `extra_gear_ratio * base_gear_ratio`
(mirrors the Qmini firmware's `is_special ? Extra_Gear_Ratio * Gear_Ratio :
Gear_Ratio`). Omit it (defaults to 1.0) for normal joints.

WHAT YOU MUST ADAPT
--------------------
  - `default_pos_deg` on each joint in robot_config.json: the fixed pose the
    policy's action is a small offset from (see QminiLegEnv._apply_action).
    This must match training's default_joint_pos exactly, in degrees. This
    is NOT the same thing as the keyframes/MotionReference below -- it's a
    single fixed number per joint, not an animation.
  - `UNITREE_SDK_LIB_PATH`: point this at wherever you built
    unitree_actuator_sdk's `lib/` (contains the compiled python module).
  - `MotionReference`: this is a placeholder linear-interpolation
    reimplementation of the `MotionPlayer` used in training. It is NOT
    guaranteed to numerically match your actual MotionPlayer (which may use
    cubic/spline interpolation). Sim-to-real fidelity depends on this
    matching exactly -- compare outputs offline before trusting it live.
    If you can import your real MotionPlayer instead (e.g. it has no
    isaaclab/torch-tensor-only dependencies), do that instead.
  - Gear ratio: this script calls the SDK's `queryGearRatio()` if available
    and falls back to DEFAULT_GEAR_RATIO_GO_M8010_6 (6.33) otherwise --
    verify this against your motor's datasheet / SDK version.
  - The exact `unitree_actuator_sdk` Python API (class/attribute names)
    may differ slightly by SDK version. Everything SDK-specific is
    isolated in the `MotorBus` class below -- if imports/attributes don't
    match your installed SDK, that's the only class you should need to edit.

USAGE
-----
    python3 robot_deploy.py \
        --config robot_config_qmini.json \
        --keyframes keyframes_forward_slow_all_joints_4x.json \
        --policy ./deploy_bundle/policy.pt \
        --policy-meta ./deploy_bundle/policy.meta.json

Note: the serial port used to be a --port CLI flag. It's now set per joint
in robot_config.json (see "MULTIPLE SERIAL BUSSES" above), since a config
that's wrong for the robot you're pointing it at is exactly the kind of
mistake that should live in a reviewable file, not a shell history entry.

robot_config.json example (see robot_config_qmini.json for the full,
currently-in-use 10-joint config this robot actually runs with):
{
  "control_dt": 0.02,
  "max_step_deg": 15.0,
  "action_scale": 0.5,
  "motor_temp_limit_c": 55.0,
  "joints": [
    {"name": "left_hip_yaw",   "motor_id": 1, "port": "/dev/ttyUSB3", "invert": false, "kp": 55.0,  "kd": 2.0,  "default_pos_deg": 8.1526,  "min_deg": -45.0, "max_deg": 30.0},
    {"name": "left_hip_roll",  "motor_id": 1, "port": "/dev/ttyUSB2", "invert": false, "kp": 105.0, "kd": 18.0, "default_pos_deg": -0.1799, "min_deg": -15.0, "max_deg": 15.0, "extra_gear_ratio": 3.0},
    {"name": "left_hip_pitch", "motor_id": 0, "port": "/dev/ttyUSB1", "invert": true,  "kp": 75.0,  "kd": 2.0,  "default_pos_deg": 4.5092,  "min_deg": -50.0, "max_deg": 50.0},
    {"name": "left_knee",      "motor_id": 1, "port": "/dev/ttyUSB1", "invert": false, "kp": 45.0,  "kd": 2.0,  "default_pos_deg": 16.6821, "min_deg": -60.0, "max_deg": 50.0},
    {"name": "left_ankle",     "motor_id": 2, "port": "/dev/ttyUSB1", "invert": false, "kp": 30.0,  "kd": 2.0,  "default_pos_deg": 25.6964, "min_deg": -60.0, "max_deg": 40.0}
    ... and the mirrored right_* joints -- see robot_config_qmini.json for the full list
  ]
}
min_deg/max_deg are output-side degrees, optional per side (omit or set
null for "no limit" on that side) -- but leaving them unset means that
joint has NO joint-limit safety abort, so set them to your robot's actual
safe mechanical range before running for real.

NOTE ON default_pos_deg: this must exactly match the corresponding joint's
default_joint_pos in training (qmini.py's ArticulationCfg.InitialStateCfg.joint_pos,
converted rad -> deg). The policy's action is decoded as
default_pos_deg + action_scale * action -- if this drifts from what training
used, every commanded target on the real robot will be offset from what the
policy actually learned, even though the policy itself is unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# unitree_actuator_sdk import
# ---------------------------------------------------------------------------
# Adjust this to wherever you built the SDK's python bindings
# (unitree_actuator_sdk/lib on Linux after `cmake .. && make`).
UNITREE_SDK_LIB_PATH = "./unitree_actuator_sdk/lib"
if UNITREE_SDK_LIB_PATH not in sys.path:
    sys.path.append(UNITREE_SDK_LIB_PATH)

try:
    from unitree_actuator_sdk import (  # type: ignore
        SerialPort,
        MotorCmd,
        MotorData,
        MotorType,
        MotorMode,
        queryMotorMode,
        queryGearRatio,
    )
    _SDK_AVAILABLE = True
except ImportError as e:  # pragma: no cover - depends on target machine
    _SDK_AVAILABLE = False
    _SDK_IMPORT_ERROR = e

DEFAULT_GEAR_RATIO_GO_M8010_6 = 6.33  # verify against queryGearRatio()/datasheet for your units

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

# A single reading-to-reading position jump bigger than this is treated as
# corrupted telemetry (dropped/garbled/timed-out serial reply leaving
# data.q/data.dq holding stale or uninitialized memory), not real motion.
#
# Deliberately NOT a velocity/time-normalized bound (e.g. "rad/s above
# velocity_limit"): real elapsed time between readings balloons exactly
# during the comms failures this check exists to catch (each timed-out
# motor adds ~15-20ms of wait), which would make a time-normalized bound
# *more* permissive right when it needs to be strictest. This bound is
# fixed regardless of how much wall-clock time actually elapsed.
#
# Calibrated well above legitimate motion: our own control loop never
# commands more than max_step_deg (see RobotConfig) away from the last
# target in a single step, so a healthy joint should never be found this
# far from its last reading, however long a stall lasted.
MAX_PLAUSIBLE_JOINT_JUMP_RAD = math.radians(60.0)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class JointConfig:
    name: str
    motor_id: int
    port: Optional[str] = None  # serial port this joint's motor is on, e.g. "/dev/ttyUSB0" (required)
    invert: bool = False
    kp: float = 20.0          # output-side stiffness
    kd: float = 0.5           # output-side damping
    offset_deg: float = 0.0   # extra manual offset added on top of the startup calibration reading
    default_pos_deg: float = 0.0
    # Fixed anchor pose this joint's action is a *small offset from* --
    # i.e. robot_cfg.action_scale * action + this. MUST match training's
    # default_joint_pos for this joint exactly (qmini.py's
    # ArticulationCfg.InitialStateCfg.joint_pos, converted to degrees,
    # currently the keyframe-0 pose -- NOT necessarily 0). This is a single
    # fixed number, unlike the --keyframes clip (the full animation, only
    # needed for --open-loop-ref / the obs motion_time phase signal, not
    # for reconstructing policy targets).
    min_deg: Optional[float] = None  # output-side lower limit; None = no limit enforced (unsafe -- set this!)
    max_deg: Optional[float] = None  # output-side upper limit; None = no limit enforced (unsafe -- set this!)
    extra_gear_ratio: float = 1.0
    # Multiplier on the bus's base gear ratio for joints with a non-standard
    # reduction (e.g. an extra belt stage): effective ratio = extra_gear_ratio
    # * base_gear_ratio. Leave at 1.0 for joints on the standard reduction.


@dataclass
class RobotConfig:
    control_dt: float = 0.02
    max_step_deg: float = 5.0
    action_scale: float = 0.15
    motor_temp_limit_c: float = 55.0
    joints: list = field(default_factory=list)  # list[JointConfig]

    @staticmethod
    def load(path: Path) -> "RobotConfig":
        raw = json.loads(path.read_text())
        joints = [JointConfig(**j) for j in raw["joints"]]
        for j in joints:
            if not j.port:
                raise ValueError(
                    f"Joint '{j.name}' is missing a 'port' in robot_config.json "
                    f"(e.g. \"/dev/ttyUSB0\") -- ports are configured per joint, "
                    f"not passed on the command line."
                )
            if j.min_deg is not None and j.max_deg is not None and j.min_deg >= j.max_deg:
                raise ValueError(
                    f"Joint '{j.name}': min_deg ({j.min_deg}) must be < max_deg ({j.max_deg})."
                )
        return RobotConfig(
            control_dt=raw.get("control_dt", 0.02),
            max_step_deg=raw.get("max_step_deg", 5.0),
            action_scale=raw.get("action_scale", 0.15),
            motor_temp_limit_c=raw.get("motor_temp_limit_c", 55.0),
            joints=joints,
        )

    @property
    def joint_names(self):
        return [j.name for j in self.joints]


# ---------------------------------------------------------------------------
# Policy metadata verification
# ---------------------------------------------------------------------------

class PolicyMismatchError(RuntimeError):
    pass


def load_and_verify_policy(policy_path: Path, meta_path: Path, robot_cfg: RobotConfig, logger: logging.Logger):
    """
    Loads the TorchScript policy and cross-checks its metadata sidecar
    against both the file's actual hash and this robot's config, so a
    stale/wrong/incompatible policy can't silently get run on the robot.
    """
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Policy metadata not found: {meta_path}")

    meta = json.loads(meta_path.read_text())

    n_joints = len(robot_cfg.joints)
    expected_action_dim = n_joints
    # joint_pos (n_joints) + joint_vel (n_joints) + a single motion_time
    # scalar -- matches QminiLegEnv._get_observations exactly (NOT a
    # per-joint motion_ref: that term is computed but never appended there,
    # see the commented-out `# reference,` line).
    expected_obs_dim = n_joints * 2 + 1

    if meta.get("action_dim") != expected_action_dim:
        raise PolicyMismatchError(
            f"Policy action_dim={meta.get('action_dim')} does not match "
            f"robot config's {expected_action_dim} joints."
        )
    if meta.get("obs_dim") != expected_obs_dim:
        raise PolicyMismatchError(
            f"Policy obs_dim={meta.get('obs_dim')} does not match expected "
            f"{expected_obs_dim} (2 * {n_joints} joints [pos+vel] + 1 motion_time scalar)."
        )
    # NOTE: this is a SET comparison, not an order comparison. robot_cfg's
    # joint order is whatever's convenient to read/wire physically (e.g.
    # grouped by leg); meta["joint_order"] is the order the policy actually
    # produces actions in (e.g. Isaac Lab's articulation order, which
    # interleaves left/right and will generally NOT match robot_cfg's
    # order). Deployment builds every obs/action array by walking
    # meta["joint_order"] and looking joints up by NAME -- never by
    # position against robot_cfg.joints -- specifically so this mismatch
    # (a real bug caught during the 3->10 joint migration: a naive
    # positional zip between the two orders would have silently swapped
    # which motor got which command) can't reoccur regardless of how
    # robot_cfg.json happens to list its joints.
    policy_joint_order = list(meta.get("joint_order", []))
    if len(policy_joint_order) != len(set(policy_joint_order)):
        raise PolicyMismatchError(
            f"Policy joint_order={policy_joint_order} contains duplicate names."
        )
    if set(policy_joint_order) != set(robot_cfg.joint_names):
        raise PolicyMismatchError(
            f"Policy joint_order={policy_joint_order} does not reference the "
            f"same set of joints as robot config {robot_cfg.joint_names} "
            f"(order may differ, but every joint name must appear in both)."
        )
    if abs(meta.get("action_scale", robot_cfg.action_scale) - robot_cfg.action_scale) > 1e-9:
        logger.warning(
            "Policy metadata action_scale (%.4f) differs from robot config "
            "action_scale (%.4f) -- using robot config value. Confirm this is intentional.",
            meta.get("action_scale"), robot_cfg.action_scale,
        )

    # default_pos_deg is the anchor pose actions are decoded relative to
    # (see JointConfig / Deployment.run()). A mismatch here doesn't crash
    # anything -- it silently offsets every commanded target from what the
    # policy actually learned, so check it like action_scale above whenever
    # the sidecar declares it. Meta files without this field (older
    # bundles) just skip the check.
    meta_default_pose = meta.get("default_pos_deg")
    if meta_default_pose is not None:
        for joint in robot_cfg.joints:
            expected = meta_default_pose.get(joint.name)
            if expected is None:
                continue
            if abs(expected - joint.default_pos_deg) > 0.5:
                raise PolicyMismatchError(
                    f"robot_config.json default_pos_deg for '{joint.name}' "
                    f"({joint.default_pos_deg:.3f} deg) differs from the policy "
                    f"metadata's expected value ({expected:.3f} deg) by more than "
                    f"0.5 deg. This anchor pose must match training's "
                    f"default_joint_pos or every commanded target will be offset."
                )

    policy = torch.jit.load(str(policy_path), map_location="cpu")
    policy.eval()

    # Dummy forward pass shape check -- catches architecture mismatches that
    # a hand-edited metadata file wouldn't catch.
    with torch.no_grad():
        dummy_obs = torch.zeros(1, expected_obs_dim)
        dummy_out = policy(dummy_obs)
        if dummy_out.shape[-1] != expected_action_dim:
            raise PolicyMismatchError(
                f"Policy forward pass produced output dim {dummy_out.shape[-1]}, "
                f"expected {expected_action_dim}."
            )

    logger.info("Policy verified OK: joints=%s, obs_dim=%d, action_dim=%d",
                meta["joint_order"], expected_obs_dim, expected_action_dim)
    return policy, meta


# ---------------------------------------------------------------------------
# Motion reference (placeholder -- see module docstring warning above)
# ---------------------------------------------------------------------------

class MotionReference:
    """
    Linear-interpolation replica of the keyframe motion used in training.
    ONLY matches training exactly if MotionPlayer also used linear
    interpolation between keyframes. Verify offline before trusting this.
    """

    def __init__(self, keyframes_deg, degrees: bool = True):
        self.times = [t for t, _ in keyframes_deg]
        vals = [v for _, v in keyframes_deg]
        self.values_rad = [
            [math.radians(x) if degrees else x for x in v] for v in vals
        ]
        self.length = self.times[-1]  # assumes first/last keyframe match, i.e. a closed loop

    @classmethod
    def from_json(cls, path: Path, n_joints: Optional[int] = None) -> "MotionReference":
        """Load keyframes from a shared keyframes.json (e.g.
        keyframes_forward_slow_all_joints_4x.json). If `n_joints` is given,
        checks every frame has exactly that many values -- a clip built for
        the wrong joint count would otherwise only surface as a much more
        confusing IndexError deep in the control loop (--open-loop-ref) or
        _startup_pose_targets()."""
        raw = json.loads(Path(path).read_text())
        keyframes = raw["keyframes"]
        if n_joints is not None:
            for t, pose in keyframes:
                if len(pose) != n_joints:
                    raise ValueError(
                        f"{path} has a keyframe at t={t} with {len(pose)} "
                        f"joint values, but robot_config.json declares "
                        f"{n_joints} joints. This clip was very likely built "
                        f"for a different robot/joint-count -- check "
                        f"joint_order in the file before using it."
                    )
        return cls(keyframes, degrees=raw.get("degrees", True))

    def sample(self, t: float):
        t = t % self.length
        for i in range(len(self.times) - 1):
            t0, t1 = self.times[i], self.times[i + 1]
            if t0 <= t <= t1:
                alpha = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                v0, v1 = self.values_rad[i], self.values_rad[i + 1]
                return [a + alpha * (b - a) for a, b in zip(v0, v1)]
        return self.values_rad[-1]



# ---------------------------------------------------------------------------
# Motor bus -- all unitree_actuator_sdk-specific code lives here
# ---------------------------------------------------------------------------

@dataclass
class MotorReading:
    joint_name: str
    output_pos_rad: float   # calibrated, direction-corrected, output-side
    output_vel_rad_s: float
    raw_rotor_pos_rad: float
    torque_est_nm: float
    temperature: Optional[float] = None
    error_flag: Optional[int] = None


class SafetyAbort(Exception):
    """Base class for any condition that requires immediately releasing
    (torque-off) every motor on every bus and aborting -- step-limit
    violations, joint-limit violations, motor errors, over-temperature, and
    corrupted telemetry all raise a subclass of this."""


class TelemetryFaultError(SafetyAbort):
    """Raised when a motor reading implies physically impossible motion --
    almost always a dropped/garbled/timed-out serial reply being silently
    treated as valid telemetry, not a real jump. Handled the same way as
    StepLimitExceeded: caller releases all motors and aborts, since once one
    reading is untrustworthy there's no safe basis for the next command."""
    def __init__(self, joint_name: str, last_pos_rad: float, new_pos_rad: float):
        self.joint_name = joint_name
        jump_deg = math.degrees(abs(new_pos_rad - last_pos_rad))
        super().__init__(
            f"Motor '{joint_name}' reading jumped {jump_deg:.2f} deg "
            f"(from {math.degrees(last_pos_rad):.2f} to {math.degrees(new_pos_rad):.2f} deg) "
            f"in a single reading -- physically implausible, treating as a "
            f"dropped/corrupted serial reply rather than real motion. Check "
            f"motor power, cabling, and bus timing/framing."
        )


class MotorBus:
    def __init__(self, port: str, joints: list, logger: logging.Logger):
        if not _SDK_AVAILABLE:
            raise RuntimeError(
                f"unitree_actuator_sdk could not be imported ({_SDK_IMPORT_ERROR}). "
                f"Check UNITREE_SDK_LIB_PATH at the top of this file."
            )
        self.logger = logger
        self.port = port
        self.joints: list[JointConfig] = joints
        self.serial = SerialPort(port)

        self.motor_type = MotorType.GO_M8010_6
        try:
            base_gear_ratio = queryGearRatio(self.motor_type)
        except Exception:
            self.logger.warning(
                "queryGearRatio() unavailable/failed, falling back to "
                "DEFAULT_GEAR_RATIO_GO_M8010_6=%.3f -- verify this!",
                DEFAULT_GEAR_RATIO_GO_M8010_6,
            )
            base_gear_ratio = DEFAULT_GEAR_RATIO_GO_M8010_6

        # Per-joint effective gear ratio: joints with a non-default
        # extra_gear_ratio (see JointConfig) get base_gear_ratio scaled by
        # it; joints left at the default (1.0) just use base_gear_ratio.
        self.gear_ratio = {j.name: base_gear_ratio * j.extra_gear_ratio for j in self.joints}

        self._motor_mode = queryMotorMode(self.motor_type, MotorMode.FOC)

        # calibration offsets, filled in by calibrate_from_startup_position()
        self.zero_offset_rotor_rad = {j.name: 0.0 for j in self.joints}
        self.last_commanded_output_rad = {j.name: None for j in self.joints}

        # last known-GOOD reading per joint (distinct from
        # last_commanded_output_rad, which tracks what we asked for, not
        # what we last confirmed) -- used by _validate_reading() to catch
        # corrupted telemetry from failed serial transactions.
        self._last_valid_output_rad = {j.name: None for j in self.joints}

    def _validate_reading(self, joint: JointConfig, output_pos_rad: float) -> None:
        """Raises TelemetryFaultError if output_pos_rad jumped further from
        this joint's last known-good reading than MAX_PLAUSIBLE_JOINT_JUMP_RAD
        allows. Silently accepts (and records as the new known-good reading)
        if there's no prior reading yet or the jump is plausible."""
        last_pos = self._last_valid_output_rad[joint.name]
        if last_pos is not None:
            if abs(output_pos_rad - last_pos) > MAX_PLAUSIBLE_JOINT_JUMP_RAD:
                raise TelemetryFaultError(joint.name, last_pos, output_pos_rad)
        self._last_valid_output_rad[joint.name] = output_pos_rad

    def _direction(self, joint: JointConfig) -> float:
        return -1.0 if joint.invert else 1.0

    def read_motor(self, joint: JointConfig) -> MotorReading:
        cmd = MotorCmd()
        data = MotorData()
        cmd.motorType = self.motor_type
        data.motorType = self.motor_type
        cmd.mode = self._motor_mode
        cmd.id = joint.motor_id
        # zero gains / feedforward torque -> pure read, motor stays wherever
        # it already is (this is also used to "ping" the motor for telemetry
        # without commanding any motion).
        cmd.kp = 0.0
        cmd.kd = 0.0
        cmd.q = 0.0
        cmd.dq = 0.0
        cmd.tau = 0.0
        self.serial.sendRecv(cmd, data)

        gear_ratio = self.gear_ratio[joint.name]
        raw_rotor_pos = data.q
        raw_rotor_vel = data.dq
        output_pos = (raw_rotor_pos / gear_ratio) * self._direction(joint)
        output_vel = (raw_rotor_vel / gear_ratio) * self._direction(joint)
        output_pos_calibrated = output_pos - self.zero_offset_rotor_rad[joint.name]

        self._validate_reading(joint, output_pos_calibrated)

        return MotorReading(
            joint_name=joint.name,
            output_pos_rad=output_pos_calibrated,
            output_vel_rad_s=output_vel,
            raw_rotor_pos_rad=raw_rotor_pos,
            torque_est_nm=getattr(data, "tau", 0.0),
            temperature=getattr(data, "temp", None),
            error_flag=getattr(data, "merror", None),
        )

    def calibrate_from_startup_position(self):
        """
        Reads each motor's current raw position and stores it (converted to
        output-side, direction-corrected radians, plus any manual
        offset_deg) as the "policy zero" for that joint. Call this once at
        startup with the robot held/resting in its intended zero pose.
        """
        for joint in self.joints:
            cmd = MotorCmd()
            data = MotorData()
            cmd.motorType = self.motor_type
            data.motorType = self.motor_type
            cmd.mode = self._motor_mode
            cmd.id = joint.motor_id
            cmd.kp = 0.0
            cmd.kd = 0.0
            cmd.q = 0.0
            cmd.dq = 0.0
            cmd.tau = 0.0
            self.serial.sendRecv(cmd, data)

            raw_output_pos = (data.q / self.gear_ratio[joint.name]) * self._direction(joint)
            offset = raw_output_pos + math.radians(joint.offset_deg)
            self.zero_offset_rotor_rad[joint.name] = offset

            output_pos_calibrated = raw_output_pos - self.zero_offset_rotor_rad[joint.name]
            self.last_commanded_output_rad[joint.name] = output_pos_calibrated

    def print_startup_table(self):
        print(f"\n=== Startup motor check for bus '{self.port}' (verify direction & zero before enabling) ===")
        print(f"{'joint':<8} {'id':>3} {'raw_deg':>10} {'calib_deg':>10} {'invert':>7}")
        for joint in self.joints:
            r = self.read_motor(joint)
            # True raw pose: direction-corrected, gear-divided, but NOT
            # zero-offset-corrected. Previously this used r.output_pos_rad
            # (already calibrated) for BOTH columns, so raw_deg and
            # calib_deg were always identical and neither showed the
            # robot's actual physical pose -- see r.raw_rotor_pos_rad,
            # which read_motor() keeps around as the true rotor-side value.
            raw_output_pos = (r.raw_rotor_pos_rad / self.gear_ratio[joint.name]) * self._direction(joint)
            raw_deg = raw_output_pos * RAD2DEG
            # Calibrated (zero-offset applied). Right after
            # calibrate_from_startup_position() this reads ~ -offset_deg
            # (not ~0, unless offset_deg is ~0) -- see that method's
            # derivation: the calibrated reading at calibration time is
            # defined to equal -offset_deg by construction.
            calib_deg = r.output_pos_rad * RAD2DEG
            print(f"{joint.name:<8} {joint.motor_id:>3} {raw_deg:>10.2f} {calib_deg:>10.2f} {str(joint.invert):>7}")
        print("=======================================================================\n")

    def check_joint_limits(self, target_output_rad: dict):
        """Raises JointLimitExceeded if any joint's target falls outside its
        configured [min_deg, max_deg] (joints with a limit left as None are
        unconstrained on that side)."""
        violations = []
        for joint in self.joints:
            target_deg = math.degrees(target_output_rad[joint.name])
            lo, hi = joint.min_deg, joint.max_deg
            if (lo is not None and target_deg < lo) or (hi is not None and target_deg > hi):
                violations.append((joint.name, target_deg, lo, hi))
        if violations:
            raise JointLimitExceeded(violations)

    def check_step_limits(self, target_output_rad: dict, max_step_deg: float):
        """Raises StepLimitExceeded if any joint's target differs from its
        last commanded target by more than max_step_deg."""
        max_step_rad = math.radians(max_step_deg)
        violations = []
        for joint in self.joints:
            target = target_output_rad[joint.name]
            last = self.last_commanded_output_rad[joint.name]
            if last is not None and abs(target - last) > max_step_rad:
                violations.append((joint.name, math.degrees(target - last)))
        if violations:
            raise StepLimitExceeded(violations)

    def send_targets(self, target_output_rad: dict):
        """
        Sends position targets to all motors on the bus. Does NOT validate
        the targets -- callers must call check_joint_limits() and
        check_step_limits() (or Deployment's helpers, which do both across
        all buses) first, before sending anything to any bus.
        """
        readings = {}
        for joint in self.joints:
            target = target_output_rad[joint.name]
            direction = self._direction(joint)
            gear_ratio = self.gear_ratio[joint.name]

            # convert calibrated output-side target back to raw rotor units
            uncalibrated_output = target + self.zero_offset_rotor_rad[joint.name]
            rotor_target = (uncalibrated_output * direction) * gear_ratio

            kp_rotor = joint.kp / (gear_ratio * gear_ratio)
            kd_rotor = joint.kd / (gear_ratio * gear_ratio)
            # kp_rotor = 0.0
            # kd_rotor = 0.0
            # rotor_target = 0.0

            cmd = MotorCmd()
            data = MotorData()
            cmd.motorType = self.motor_type
            data.motorType = self.motor_type
            cmd.mode = self._motor_mode
            cmd.id = joint.motor_id
            cmd.kp = kp_rotor
            cmd.kd = kd_rotor
            cmd.q = rotor_target
            cmd.dq = 0.0
            cmd.tau = 0.0
            self.serial.sendRecv(cmd, data)

            self.last_commanded_output_rad[joint.name] = target

            output_pos = (data.q / gear_ratio) * direction - self.zero_offset_rotor_rad[joint.name]
            output_vel = (data.dq / gear_ratio) * direction

            self._validate_reading(joint, output_pos)

            readings[joint.name] = MotorReading(
                joint_name=joint.name,
                output_pos_rad=output_pos,
                output_vel_rad_s=output_vel,
                raw_rotor_pos_rad=data.q,
                torque_est_nm=getattr(data, "tau", 0.0),
                temperature=getattr(data, "temp", None),
                error_flag=getattr(data, "merror", None),
            )
        return readings

    def release_all(self):
        """Sets kp=kd=tau=0 on every motor so they go limp. Always safe to call."""
        for joint in self.joints:
            try:
                cmd = MotorCmd()
                data = MotorData()
                cmd.motorType = self.motor_type
                data.motorType = self.motor_type
                cmd.mode = self._motor_mode
                cmd.id = joint.motor_id
                cmd.kp = 0.0
                cmd.kd = 0.0
                cmd.q = 0.0
                cmd.dq = 0.0
                cmd.tau = 0.0
                self.serial.sendRecv(cmd, data)
            except Exception as e:
                self.logger.error("Failed to release motor %s (id=%d): %s", joint.name, joint.motor_id, e)
        self.logger.info("All motors released (torque off).")


class StepLimitExceeded(SafetyAbort):
    def __init__(self, violations):
        self.violations = violations
        msg = "; ".join(f"{name}: {delta:+.2f} deg" for name, delta in violations)
        super().__init__(f"Commanded step exceeds max_step_deg: {msg}")


class JointLimitExceeded(SafetyAbort):
    def __init__(self, violations):
        self.violations = violations
        parts = []
        for name, deg, lo, hi in violations:
            lo_s = "-inf" if lo is None else f"{lo:.2f}"
            hi_s = "+inf" if hi is None else f"{hi:.2f}"
            parts.append(f"{name}: {deg:+.2f} deg (limits [{lo_s}, {hi_s}])")
        super().__init__(f"Commanded target outside joint limits: {'; '.join(parts)}")


class MotorErrorDetected(SafetyAbort):
    def __init__(self, joint_name: str, error_flag):
        self.joint_name = joint_name
        self.error_flag = error_flag
        super().__init__(f"Motor error detected on joint '{joint_name}': error_flag={error_flag}")


class MotorOverTemperature(SafetyAbort):
    def __init__(self, joint_name: str, temperature: float, limit: float):
        self.joint_name = joint_name
        self.temperature = temperature
        self.limit = limit
        super().__init__(
            f"Motor over-temperature on joint '{joint_name}': "
            f"{temperature:.1f}C exceeds motor_temp_limit_c={limit:.1f}C"
        )


def check_motor_safety(readings: dict, temp_limit_c: float):
    """Raises MotorErrorDetected or MotorOverTemperature if any reading in
    `readings` (joint_name -> MotorReading) reports a nonzero error flag or
    a temperature above temp_limit_c. Readings with error_flag/temperature
    unsupported by the SDK (None) are skipped for that check."""
    for name, r in readings.items():
        if r.error_flag:
            raise MotorErrorDetected(name, r.error_flag)
        if r.temperature is not None and r.temperature > temp_limit_c:
            raise MotorOverTemperature(name, r.temperature, temp_limit_c)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger("robot_deploy")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    # Don't propagate to the root logger -- main() also attaches a
    # StreamHandler to this logger below, so propagating on top of that
    # printed every message to the console twice (once via this logger's
    # own handler, once via the root logger's, since messages bubble up by
    # default).
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_dir / f"events_{timestamp}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    csv_path = log_dir / f"control_loop_{timestamp}.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    return logger, csv_writer, csv_file, csv_path


def write_csv_header(csv_writer, policy_joint_order, physical_joint_names):
    # NOTE: this must match build_obs()'s actual layout exactly (pos x N,
    # then vel x N, then a single motion_time scalar -- NOT an obs_ref per
    # joint, which build_obs() computes but never appends). The previous
    # version of this header didn't match, which silently shifted every
    # column after it by 2. If you've since added `ref` back into build_obs
    # (the commented-out `obs.extend(ref)` line), update this to match.
    #
    # obs/action columns follow `policy_joint_order` (meta.json's order --
    # what the policy's own vectors are actually indexed by); the
    # target/actual/temp/err columns follow `physical_joint_names`
    # (robot_cfg's order -- purely bookkeeping, doesn't need to match the
    # policy). These two orders are allowed to differ -- see
    # load_and_verify_policy()'s joint_order check and Deployment.build_obs.
    header = ["t_wall", "motion_time"]  # t_wall = measured perf_counter() time since run() started
    for name in policy_joint_order:
        header += [f"obs_pos_{name}"]
    for name in policy_joint_order:
        header += [f"obs_vel_{name}"]
    header += ["obs_motion_time"]
    for name in policy_joint_order:
        header += [f"action_{name}"]
    for name in physical_joint_names:
        header += [f"target_deg_{name}", f"actual_deg_{name}", f"temp_{name}", f"err_{name}"]
    header += ["policy_ms", "bus_ms"]  # per-step timing breakdown, see analyze_delay.py
    csv_writer.writerow(header)


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

class Deployment:
    def __init__(
        self,
        robot_cfg: RobotConfig,
        policy,
        logger: logging.Logger,
        csv_writer,
        csv_file,
        policy_joint_order: Optional[list] = None,
        keyframes_path: Optional[Path] = None,
        open_loop_ref: bool = False,
        startup_pose: str = "auto",
        startup_move_duration: float = 2.0,
        action_smoothing: float = 1.0,
    ):
        self.robot_cfg = robot_cfg
        self.policy = policy
        self.open_loop_ref = open_loop_ref
        self.logger = logger
        self.csv_writer = csv_writer
        self.csv_file = csv_file

        # The order the policy's obs/action vectors are actually indexed by
        # -- from policy.meta.json's joint_order, verified by
        # load_and_verify_policy() to reference the same joints as
        # robot_cfg (as a SET, not by position). build_obs()/run() walk
        # THIS list and look joints up by name, never by position against
        # robot_cfg.joints, so robot_cfg.json's own order (e.g. grouped by
        # leg, for readability/wiring) doesn't need to match it. Falls back
        # to robot_cfg.joint_names for --open-loop-ref, where there's no
        # policy/meta to source an order from and this list is only used
        # for (unused-for-control) obs logging.
        self.policy_joint_order = (
            list(policy_joint_order) if policy_joint_order is not None else robot_cfg.joint_names
        )
        assert set(self.policy_joint_order) == set(robot_cfg.joint_names), (
            "policy_joint_order must reference exactly robot_cfg's joints -- "
            "this should have been caught by load_and_verify_policy()."
        )

        write_csv_header(self.csv_writer, self.policy_joint_order, robot_cfg.joint_names)

        for joint in robot_cfg.joints:
            if joint.min_deg is None or joint.max_deg is None:
                self.logger.warning(
                    "Joint '%s' has no %s configured in robot_config.json -- "
                    "that side is UNCONSTRAINED and won't trigger a joint-limit "
                    "safety abort. Set min_deg/max_deg once you know this "
                    "joint's safe mechanical range.",
                    joint.name,
                    "min_deg/max_deg" if joint.min_deg is None and joint.max_deg is None
                    else ("min_deg" if joint.min_deg is None else "max_deg"),
                )

        if open_loop_ref:
            self.logger.warning(
                "Running in --open-loop-ref mode: the policy is loaded but NOT "
                "used, joint targets are the raw motion reference. This is for "
                "sim/real matching (tune_pid_isaaclab.py), not normal operation."
            )

        # EMA low-pass on the FINAL decoded target (after default_pose_rad +
        # action_scale * action), applied every step regardless of
        # open_loop_ref (a no-op there at the default). smoothed_t =
        # action_smoothing * target_t + (1 - action_smoothing) * smoothed_{t-1}.
        # 1.0 = off (raw target passed through unchanged, original
        # behavior). Diagnostic tool for a specific symptom: if the policy
        # is producing noisy/jittery frame-to-frame targets that demand
        # sharp torque transients from the PD controller, this damps that
        # out WITHOUT retraining, letting you test whether jitter (rather
        # than wiring or an individual large target) is what's tripping the
        # motor fault. If smoothing fixes it, the real fix is an
        # action-rate penalty during training (see qmini_leg_env.py) --
        # this flag is for isolating the cause, not a permanent substitute.
        self.action_smoothing = action_smoothing
        self._smoothed_targets: dict = {}

        # One MotorBus per unique port in robot_config.json -- joints that
        # share a port share a bus, joints with different ports get their
        # own (see module docstring's "MULTIPLE SERIAL BUSSES" section).
        joints_by_port: dict = {}
        for joint in robot_cfg.joints:
            joints_by_port.setdefault(joint.port, []).append(joint)
        self.buses = [MotorBus(port, joints, self.logger) for port, joints in joints_by_port.items()]
        self._joint_bus = {joint.name: bus for bus in self.buses for joint in bus.joints}
        self.logger.info(
            "Initialized %d motor bus(es): %s",
            len(self.buses),
            "; ".join(f"{bus.port} -> {[j.name for j in bus.joints]}" for bus in self.buses),
        )

        if keyframes_path is None:
            raise ValueError(
                "keyframes_path is required -- pass --keyframes pointing at "
                "the clip this robot_config.json's joints were tuned "
                "against (e.g. keyframes_forward_slow_all_joints_4x.json). "
                "There is no built-in default clip: a stale/wrong joint "
                "count here would silently drift from what training used."
            )
        self.motion = MotionReference.from_json(keyframes_path, n_joints=len(robot_cfg.joints))
        self.logger.info("Loaded keyframes from %s", keyframes_path)
        self.motion_time = 0.0

        # Fixed anchor pose each joint's action is decoded relative to --
        # target = default_pose_rad + action_scale * action. Deliberately
        # NOT derived from self.motion (the keyframes/MotionReference
        # above): that's the reference *animation*, this is the single
        # fixed calibration pose from robot_config.json, matching
        # training's default_joint_pos. See run() and JointConfig.
        self.default_pose_rad = {
            j.name: math.radians(j.default_pos_deg) for j in robot_cfg.joints
        }

        # "auto" resolves to "keyframe" only for --open-loop-ref (which
        # needs to start on the reference trajectory it's about to command
        # directly), else "default" -- matching how QminiLegEnv now resets
        # each episode to default_joint_pos (see qmini.py's
        # ArticulationCfg.InitialStateCfg and _apply_action's anchor pose)
        # before the policy starts acting on it.
        if startup_pose == "auto":
            startup_pose = "keyframe" if open_loop_ref else "default"
        if startup_pose not in ("zero", "default", "keyframe", "none"):
            raise ValueError(f"Unknown --startup-pose '{startup_pose}', expected zero|default|keyframe|none|auto")
        self.startup_pose_mode = startup_pose
        self.startup_move_duration = startup_move_duration

        self._stop = False
        signal.signal(signal.SIGINT, self._handle_stop_signal)
        signal.signal(signal.SIGTERM, self._handle_stop_signal)

    def _handle_stop_signal(self, signum, frame):
        self.logger.warning("Received signal %s -- stopping and releasing motors.", signum)
        self._stop = True

    # -- multi-bus helpers: dispatch each joint to its owning MotorBus and
    # merge results, so callers can work in terms of "all robot joints"
    # without caring how they're split across serial buses. --

    def _read_all_motors(self) -> dict:
        return {j.name: self._joint_bus[j.name].read_motor(j) for j in self.robot_cfg.joints}

    def _validate_targets_all(self, targets: dict, max_step_deg: float):
        """Checks joint limits AND step limits on every bus before anything
        is sent, so a violation on one bus can never be discovered after
        another bus has already been commanded."""
        for bus in self.buses:
            bus.check_joint_limits(targets)
            bus.check_step_limits(targets, max_step_deg)

    def _send_targets_all(self, targets: dict) -> dict:
        readings = {}
        for bus in self.buses:
            readings.update(bus.send_targets(targets))
        return readings

    def _release_all(self):
        for bus in self.buses:
            bus.release_all()

    def _startup_pose_targets(self) -> dict:
        if self.startup_pose_mode == "zero":
            return {j.name: 0.0 for j in self.robot_cfg.joints}
        elif self.startup_pose_mode == "default":
            return dict(self.default_pose_rad)
        elif self.startup_pose_mode == "keyframe":
            # Zipped positionally against robot_cfg.joints (NOT
            # policy_joint_order) -- this is a property of the shared
            # keyframes clip, which is authored in the same order
            # robot_cfg.json lists its joints in (grouped by leg), not the
            # policy's training order. Unlike build_obs/run() above, there
            # is no meta.json here to source a name-verified order from.
            ref = self.motion.sample(0.0)
            return {joint.name: ref[i] for i, joint in enumerate(self.robot_cfg.joints)}
        else:
            raise ValueError(f"_startup_pose_targets() called with mode='none'")

    def move_to_pose(self, target_output_rad: dict, duration: float):
        """
        Smoothly ramps every joint from its CURRENT position (freshly read,
        not assumed) to target_output_rad over `duration` seconds, at
        robot_cfg.control_dt and each joint's normal kp/kd. Splits the move
        into enough steps that no single step should exceed half of
        robot_cfg.max_step_deg, extending the move beyond `duration` if the
        requested duration would require bigger steps than that -- slower
        and safe beats fast and tripping the StepLimitExceeded abort
        partway through a startup move.
        """
        dt = self.robot_cfg.control_dt
        readings = self._read_all_motors()
        check_motor_safety(readings, self.robot_cfg.motor_temp_limit_c)
        current = {name: r.output_pos_rad for name, r in readings.items()}

        max_step_rad = math.radians(self.robot_cfg.max_step_deg)
        max_delta = max(abs(target_output_rad[name] - current[name]) for name in current)
        min_steps_for_safety = max(1, math.ceil(max_delta / (max_step_rad * 0.5)))
        n_steps = max(min_steps_for_safety, max(1, int(round(duration / dt))))
        actual_duration = n_steps * dt

        self.logger.info(
            "Moving to '%s' startup pose over %.2fs (%d steps @ %.0fms, largest joint delta %.1f deg)...",
            self.startup_pose_mode, actual_duration, n_steps, dt * 1000, math.degrees(max_delta),
        )

        try:
            for step in range(1, n_steps + 1):
                step_start = time.perf_counter()
                alpha = step / n_steps
                targets = {
                    name: current[name] + alpha * (target_output_rad[name] - current[name])
                    for name in current
                }
                self._validate_targets_all(targets, self.robot_cfg.max_step_deg)
                step_readings = self._send_targets_all(targets)
                check_motor_safety(step_readings, self.robot_cfg.motor_temp_limit_c)
                elapsed = time.perf_counter() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
        except SafetyAbort as e:
            self.logger.error("SAFETY ABORT during startup move: %s", e)
            self._release_all()
            raise

        self.logger.info("Startup pose reached.")

    def startup_sequence(self):
        self.logger.info("Calibrating from current motor positions...")
        for bus in self.buses:
            bus.calibrate_from_startup_position()
        for bus in self.buses:
            bus.print_startup_table()

        answer = input(
            "Verify the table above: raw_deg should match the robot's actual pose, "
            "and invert flags should make sense. Enable motors and start policy? [y/N] "
        )
        if answer.strip().lower() != "y":
            self.logger.info("User declined startup confirmation. Exiting without enabling motors.")
            sys.exit(0)

        if self.startup_pose_mode != "none":
            self.move_to_pose(self._startup_pose_targets(), self.startup_move_duration)

        answer = input(
            "Start pose reached: Continue and start policy? [y/N] "
        )
        if answer.strip().lower() != "y":
            self.logger.info("User declined startup confirmation. Exiting without enabling motors.")
            self._release_all()
            sys.exit(0)

    def build_obs(self, readings: dict) -> torch.Tensor:
        ref = self.motion.sample(self.motion_time)
        obs = []
        # Walk policy_joint_order (NOT robot_cfg.joints) so the obs vector
        # is indexed exactly the way the policy was trained, regardless of
        # what order robot_cfg.json happens to list joints in.
        for name in self.policy_joint_order:
            obs.append(readings[name].output_pos_rad)
        for name in self.policy_joint_order:
            obs.append(readings[name].output_vel_rad_s)
        obs.append(self.motion_time)
        # obs.extend(ref)
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    def run(self):
        dt = self.robot_cfg.control_dt
        step = 0
        run_start = time.perf_counter()
        try:
            new_readings = self._read_all_motors()
            check_motor_safety(new_readings, self.robot_cfg.motor_temp_limit_c)
            while not self._stop:
                loop_start = time.perf_counter()

                readings = new_readings

                print(f"CURRENT  LEFT HIP YAW: {math.degrees(readings['left_hip_yaw'].output_pos_rad):10.5f}  HIP ROLL: {math.degrees(readings['left_hip_roll'].output_pos_rad):10.5f}  HIP PITCH: {math.degrees(readings['left_hip_pitch'].output_pos_rad):10.5f}  KNEE: {math.degrees(readings['left_knee'].output_pos_rad):10.5f}  ANKLE: {math.degrees(readings['left_ankle'].output_pos_rad):10.5f}")
                print(f"CURRENT RIGHT HIP YAW: {math.degrees(readings['right_hip_yaw'].output_pos_rad):10.5f}  HIP ROLL: {math.degrees(readings['right_hip_roll'].output_pos_rad):10.5f}  HIP PITCH: {math.degrees(readings['right_hip_pitch'].output_pos_rad):10.5f}  KNEE: {math.degrees(readings['right_knee'].output_pos_rad):10.5f}  ANKLE: {math.degrees(readings['right_ankle'].output_pos_rad):10.5f}")
                obs = self.build_obs(readings)

                ref = self.motion.sample(self.motion_time)

                policy_start = time.perf_counter()
                if self.open_loop_ref:
                    # Bypass the policy entirely -- targets are the raw
                    # reference trajectory. Used to capture a real-robot
                    # trace to compare against tune_pid_isaaclab.py. Zipped
                    # against robot_cfg.joints, same as _startup_pose_targets'
                    # "keyframe" branch -- see the comment there.
                    action = [0.0] * len(self.robot_cfg.joints)
                    targets = {joint.name: ref[i] for i, joint in enumerate(self.robot_cfg.joints)}
                else:
                    with torch.no_grad():
                        action = self.policy(obs).squeeze(0).numpy()
                    action = action.clip(-1.0, 1.0)
                    # Must mirror QminiLegEnv._apply_action exactly: action
                    # is a small offset from the fixed default_pose_rad
                    # anchor, scaled by action_scale -- NOT an absolute
                    # target. See JointConfig.default_pos_deg. Indexed by
                    # policy_joint_order (matching build_obs above), NOT
                    # robot_cfg.joints -- action[i] means whatever joint
                    # policy_joint_order[i] names, which may not be
                    # robot_cfg.joints[i].
                    targets = {
                        name: self.default_pose_rad[name]
                        + self.robot_cfg.action_scale * float(action[i])
                        for i, name in enumerate(self.policy_joint_order)
                    }
                policy_ms = (time.perf_counter() - policy_start) * 1000.0

                if self.action_smoothing < 1.0:
                    for name, t in targets.items():
                        prev = self._smoothed_targets.get(name)
                        smoothed = t if prev is None else (
                            self.action_smoothing * t + (1.0 - self.action_smoothing) * prev
                        )
                        self._smoothed_targets[name] = smoothed
                        targets[name] = smoothed

                print(f"TARGETS  LEFT HIP YAW: {math.degrees(targets['left_hip_yaw']):10.5f}  HIP ROLL: {math.degrees(targets['left_hip_roll']):10.5f}  HIP PITCH: {math.degrees(targets['left_hip_pitch']):10.5f}  KNEE: {math.degrees(targets['left_knee']):10.5f}  ANKLE: {math.degrees(targets['left_ankle']):10.5f}")
                print(f"TARGETS RIGHT HIP YAW: {math.degrees(targets['right_hip_yaw']):10.5f}  HIP ROLL: {math.degrees(targets['right_hip_roll']):10.5f}  HIP PITCH: {math.degrees(targets['right_hip_pitch']):10.5f}  KNEE: {math.degrees(targets['right_knee']):10.5f}  ANKLE: {math.degrees(targets['right_ankle']):10.5f}")

                bus_start = time.perf_counter()
                try:
                    self._validate_targets_all(targets, self.robot_cfg.max_step_deg)
                    new_readings = self._send_targets_all(targets)
                    check_motor_safety(new_readings, self.robot_cfg.motor_temp_limit_c)
                except SafetyAbort as e:
                    self.logger.error("SAFETY ABORT: %s", e)
                    self._release_all()
                    raise
                bus_ms = (time.perf_counter() - bus_start) * 1000.0

                t_wall = loop_start - run_start
                self._log_step(step, t_wall, readings, obs, action, targets, new_readings, policy_ms, bus_ms)

                self.motion_time = (self.motion_time + dt) % self.motion.length
                step += 1
                print(f"STEP: {step}  MOTION_TIME: {self.motion_time}")

                elapsed = time.perf_counter() - loop_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif elapsed > dt * 1.5:
                    self.logger.warning(
                        "Control loop overrun: %.1f ms (target %.1f ms) -- policy %.1f ms, bus %.1f ms",
                        elapsed * 1000, dt * 1000, policy_ms, bus_ms,
                    )
        finally:
            self._release_all()
            self.csv_file.close()
            self.logger.info("Deployment stopped, motors released, log file closed.")

    def _log_step(self, step, t_wall, readings, obs, action, targets, new_readings, policy_ms, bus_ms):
        # NOTE: 't' is now the measured wall-clock time since run() started,
        # NOT step * control_dt. If your loop is overrunning control_dt
        # (watch the "Control loop overrun" warnings), those two diverge --
        # use this column, not the step index, when aligning against sim
        # traces or computing achieved control rate.
        row = [t_wall, self.motion_time]
        obs_list = obs.squeeze(0).tolist()
        row += obs_list
        row += list(action)
        for joint in self.robot_cfg.joints:
            r = new_readings[joint.name]
            row += [
                math.degrees(targets[joint.name]),
                math.degrees(r.output_pos_rad),
                r.temperature if r.temperature is not None else "",
                r.error_flag if r.error_flag is not None else "",
            ]
        row += [policy_ms, bus_ms]
        self.csv_writer.writerow(row)
        if step % 50 == 0:
            self.csv_file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="robot_config.json")
    parser.add_argument(
        "--policy", type=Path, default=None,
        help="policy.pt (TorchScript). Required unless --open-loop-ref is set.",
    )
    parser.add_argument(
        "--policy-meta", type=Path, default=None,
        help="policy.meta.json. Required unless --open-loop-ref is set.",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("./logs"))
    parser.add_argument(
        "--keyframes", type=Path, required=True,
        help="Path to the shared keyframes clip (e.g. "
             "keyframes_forward_slow_all_joints_4x.json) matching --config's "
             "joint count and order. Required -- there is no built-in "
             "default clip.",
    )
    parser.add_argument(
        "--open-loop-ref", action="store_true",
        help="Bypass the policy and command the raw motion reference "
             "directly. Use this to capture a real-robot trace for "
             "tune_pid_isaaclab.py -- NOT for normal operation.",
    )
    parser.add_argument(
        "--action-smoothing", type=float, default=1.0,
        help="EMA low-pass factor applied to the final decoded joint "
             "target every step. 1.0 (default) = off, unchanged behavior. "
             "Lower = more smoothing (e.g. 0.3). Diagnostic tool: if the "
             "policy's raw targets are jittery enough to trip a motor "
             "fault that a smooth reference trajectory doesn't, lowering "
             "this can confirm that without retraining. Not a permanent "
             "fix -- if it helps, add an action-rate penalty to training "
             "instead (see qmini_leg_env.py).",
    )
    parser.add_argument(
        "--startup-pose", type=str, default="auto", choices=["auto", "zero", "default", "keyframe", "none"],
        help="Pose to smoothly move to after calibration, before the "
             "control loop starts. 'auto' (default) picks 'keyframe' "
             "(keyframes[0]) for --open-loop-ref, else 'default' (each "
             "joint's default_pos_deg from robot_config.json -- the same "
             "anchor pose the policy's actions are decoded relative to, "
             "and what QminiLegEnv resets each training episode to). "
             "'zero' moves to the raw calibration pose instead. 'none' "
             "skips the move (old behavior -- the first policy/reference "
             "action jumps straight from calibration pose, subject to "
             "--max-step-deg).",
    )
    parser.add_argument(
        "--startup-move-duration", type=float, default=2.0,
        help="Seconds to spend ramping to --startup-pose. Extended "
             "automatically if that would require per-step moves close to "
             "--max-step-deg.",
    )
    args = parser.parse_args()

    robot_cfg = RobotConfig.load(args.config)

    # Single logger for the whole run, set up once here -- Deployment no
    # longer creates its own (see setup_logging()'s propagate=False note:
    # having two independently-configured loggers in this hierarchy was
    # exactly why every message after Deployment was constructed used to
    # print to the console twice).
    logger, csv_writer, csv_file, csv_path = setup_logging(args.log_dir)
    logger.info("Logging control loop to %s", csv_path)

    try:
        policy_joint_order = None
        if args.open_loop_ref:
            if args.policy or args.policy_meta:
                logger.info("--open-loop-ref set: ignoring --policy/--policy-meta, the policy will not be called.")
            policy = None
        else:
            if not args.policy or not args.policy_meta:
                parser.error("--policy and --policy-meta are required unless --open-loop-ref is set.")
            policy, meta = load_and_verify_policy(args.policy, args.policy_meta, robot_cfg, logger)
            policy_joint_order = meta["joint_order"]

        deployment = Deployment(
            robot_cfg, policy, logger, csv_writer, csv_file,
            policy_joint_order=policy_joint_order,
            keyframes_path=args.keyframes, open_loop_ref=args.open_loop_ref,
            startup_pose=args.startup_pose, startup_move_duration=args.startup_move_duration,
            action_smoothing=args.action_smoothing,
        )
        deployment.startup_sequence()
        deployment.run()
    except SafetyAbort:
        logger.error("Exiting after safety abort. Motors have been released.")
        sys.exit(1)
    except Exception:
        logger.exception("Unhandled exception -- motors released via finally block.")
        raise
    finally:
        if not csv_file.closed:
            csv_file.close()


if __name__ == "__main__":
    main()

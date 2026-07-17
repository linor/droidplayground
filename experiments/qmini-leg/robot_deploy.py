#!/usr/bin/env python3
"""
robot_deploy.py

Sim-to-real deployment of an Isaac Lab / rsl_rl policy (QminiLegEnv, 3 joints:
hip, knee, ankle) onto real GO-M8010-6 motors via Unitree's unitree_actuator_sdk
over a single RS485 bus.

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
     from the last commanded target. If any joint would move further in a
     single step, it aborts immediately and releases (torque-off) ALL
     motors on the bus.
  5. Releases all motors on normal exit, Ctrl+C, SIGTERM, or any unhandled
     exception (try/finally + signal handlers).
  6. Logs every control step (obs, raw policy action, joint targets, rotor
     commands, motor telemetry) to a CSV plus a human-readable event log.

WHAT YOU MUST ADAPT
--------------------
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
        --config robot_config.json \
        --policy ./deploy_bundle/policy.pt \
        --policy-meta ./deploy_bundle/policy.meta.json \
        --port /dev/ttyUSB0

robot_config.json example:
{
  "control_dt": 0.02,
  "max_step_deg": 5.0,
  "action_scale": 0.15,
  "joints": [
    {"name": "hip",   "motor_id": 0, "invert": false, "kp": 140.0, "kd": 5.0},
    {"name": "knee",  "motor_id": 1, "invert": true,  "kp": 180.0, "kd": 6.0},
    {"name": "ankle", "motor_id": 2, "invert": false, "kp": 90.0,  "kd": 3.0}
  ]
}
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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class JointConfig:
    name: str
    motor_id: int
    invert: bool = False
    kp: float = 20.0          # output-side stiffness
    kd: float = 0.5           # output-side damping
    offset_deg: float = 0.0   # extra manual offset added on top of the startup calibration reading


@dataclass
class RobotConfig:
    control_dt: float = 0.02
    max_step_deg: float = 5.0
    action_scale: float = 0.15
    joints: list = field(default_factory=list)  # list[JointConfig]

    @staticmethod
    def load(path: Path) -> "RobotConfig":
        raw = json.loads(path.read_text())
        joints = [JointConfig(**j) for j in raw["joints"]]
        return RobotConfig(
            control_dt=raw.get("control_dt", 0.02),
            max_step_deg=raw.get("max_step_deg", 5.0),
            action_scale=raw.get("action_scale", 0.15),
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

    actual_sha256 = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    expected_sha256 = meta.get("policy_sha256")
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise PolicyMismatchError(
            f"policy.pt sha256 mismatch!\n"
            f"  expected (from metadata): {expected_sha256}\n"
            f"  actual file on disk:      {actual_sha256}\n"
            f"The policy file does not match its metadata sidecar -- refusing to run. "
            f"Re-export both together with export_policy_for_deployment.py."
        )

    n_joints = len(robot_cfg.joints)
    expected_action_dim = n_joints
    expected_obs_dim = n_joints * 3  # joint_pos + joint_vel + motion_ref, per QminiLegEnv

    if meta.get("action_dim") != expected_action_dim:
        raise PolicyMismatchError(
            f"Policy action_dim={meta.get('action_dim')} does not match "
            f"robot config's {expected_action_dim} joints."
        )
    if meta.get("obs_dim") != expected_obs_dim:
        raise PolicyMismatchError(
            f"Policy obs_dim={meta.get('obs_dim')} does not match expected "
            f"{expected_obs_dim} (3 * {n_joints} joints: pos+vel+ref)."
        )
    if list(meta.get("joint_order", [])) != robot_cfg.joint_names:
        raise PolicyMismatchError(
            f"Policy joint_order={meta.get('joint_order')} does not match "
            f"robot config joint order {robot_cfg.joint_names}. Joint order "
            f"mismatches are especially dangerous -- they silently swap which "
            f"motor gets which command."
        )
    if abs(meta.get("action_scale", robot_cfg.action_scale) - robot_cfg.action_scale) > 1e-9:
        logger.warning(
            "Policy metadata action_scale (%.4f) differs from robot config "
            "action_scale (%.4f) -- using robot config value. Confirm this is intentional.",
            meta.get("action_scale"), robot_cfg.action_scale,
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

    logger.info("Policy verified OK: sha256=%s, joints=%s, obs_dim=%d, action_dim=%d",
                actual_sha256[:12], meta["joint_order"], expected_obs_dim, expected_action_dim)
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

    def sample(self, t: float):
        t = t % self.length
        for i in range(len(self.times) - 1):
            t0, t1 = self.times[i], self.times[i + 1]
            if t0 <= t <= t1:
                alpha = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                v0, v1 = self.values_rad[i], self.values_rad[i + 1]
                return [a + alpha * (b - a) for a, b in zip(v0, v1)]
        return self.values_rad[-1]


DEFAULT_KEYFRAMES_DEG = [
    (0.0, [20, -50, 30]),
    (0.4, [5, -20, 10]),
    (0.8, [-15, 10, 5]),
    (1.2, [5, -20, 10]),
    (1.6, [20, -50, 30]),
]
# DEFAULT_KEYFRAMES_DEG = [
#     (0.0, [20, -50, 30]),
#     (1.0, [5, -20, 10]),
#     (2.0, [-15, 10, 5]),
#     (3.0, [5, -20, 10]),
#     (4.0, [20, -50, 30]),
# ]


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


class MotorBus:
    def __init__(self, port: str, joints: list, logger: logging.Logger):
        if not _SDK_AVAILABLE:
            raise RuntimeError(
                f"unitree_actuator_sdk could not be imported ({_SDK_IMPORT_ERROR}). "
                f"Check UNITREE_SDK_LIB_PATH at the top of this file."
            )
        self.logger = logger
        self.joints: list[JointConfig] = joints
        self.serial = SerialPort(port)

        self.motor_type = MotorType.GO_M8010_6
        try:
            self.gear_ratio = queryGearRatio(self.motor_type)
        except Exception:
            self.logger.warning(
                "queryGearRatio() unavailable/failed, falling back to "
                "DEFAULT_GEAR_RATIO_GO_M8010_6=%.3f -- verify this!",
                DEFAULT_GEAR_RATIO_GO_M8010_6,
            )
            self.gear_ratio = DEFAULT_GEAR_RATIO_GO_M8010_6

        self._motor_mode = queryMotorMode(self.motor_type, MotorMode.FOC)

        # calibration offsets, filled in by calibrate_from_startup_position()
        self.zero_offset_rotor_rad = {j.name: 0.0 for j in self.joints}
        self.last_commanded_output_rad = {j.name: None for j in self.joints}

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

        raw_rotor_pos = data.q
        raw_rotor_vel = data.dq
        output_pos = (raw_rotor_pos / self.gear_ratio) * self._direction(joint)
        output_vel = (raw_rotor_vel / self.gear_ratio) * self._direction(joint)
        output_pos_calibrated = output_pos - self.zero_offset_rotor_rad[joint.name]

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

            raw_output_pos = (data.q / self.gear_ratio) * self._direction(joint)
            offset = raw_output_pos + math.radians(joint.offset_deg)
            self.zero_offset_rotor_rad[joint.name] = offset
            self.last_commanded_output_rad[joint.name] = 0.0

    def print_startup_table(self):
        print("\n=== Startup motor check (verify direction & zero before enabling) ===")
        print(f"{'joint':<8} {'id':>3} {'raw_deg':>10} {'calib_deg':>10} {'invert':>7}")
        for joint in self.joints:
            r = self.read_motor(joint)
            raw_deg = r.output_pos_rad * RAD2DEG
            calib_deg = raw_deg  # right after calibrate_from_startup_position(), calib should read ~0
            print(f"{joint.name:<8} {joint.motor_id:>3} {raw_deg:>10.2f} {calib_deg:>10.2f} {str(joint.invert):>7}")
        print("=======================================================================\n")

    def send_targets(self, target_output_rad: dict, max_step_deg: float, logger: logging.Logger):
        """
        Sends position targets to all motors on the bus. Raises
        StepLimitExceeded (without sending anything) if any joint's target
        differs from its last commanded target by more than max_step_deg.
        Caller is responsible for catching that and calling release_all().
        """
        max_step_rad = math.radians(max_step_deg)
        violations = []
        for joint in self.joints:
            target = target_output_rad[joint.name]
            last = self.last_commanded_output_rad[joint.name]
            if last is not None and abs(target - last) > max_step_rad:
                violations.append((joint.name, math.degrees(target - last)))

        if violations:
            raise StepLimitExceeded(violations)

        readings = {}
        for joint in self.joints:
            target = target_output_rad[joint.name]
            direction = self._direction(joint)

            # convert calibrated output-side target back to raw rotor units
            uncalibrated_output = target + self.zero_offset_rotor_rad[joint.name]
            rotor_target = (uncalibrated_output * direction) * self.gear_ratio

            r = self.gear_ratio
            kp_rotor = joint.kp / (r * r)
            kd_rotor = joint.kd / (r * r)
            # kp_rotor = 0.05
            # kd_rotor = 0.001
            # print(f"KP: {kp_rotor}  KD: {kd_rotor}")

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

            output_pos = (data.q / self.gear_ratio) * direction - self.zero_offset_rotor_rad[joint.name]
            output_vel = (data.dq / self.gear_ratio) * direction
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


class StepLimitExceeded(Exception):
    def __init__(self, violations):
        self.violations = violations
        msg = "; ".join(f"{name}: {delta:+.2f} deg" for name, delta in violations)
        super().__init__(f"Commanded step exceeds max_step_deg: {msg}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger("robot_deploy")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

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


def write_csv_header(csv_writer, joint_names):
    header = ["t", "motion_time"]
    for name in joint_names:
        header += [f"obs_pos_{name}", f"obs_vel_{name}", f"obs_ref_{name}"]
    for name in joint_names:
        header += [f"action_{name}"]
    for name in joint_names:
        header += [f"target_deg_{name}", f"actual_deg_{name}", f"temp_{name}", f"err_{name}"]
    csv_writer.writerow(header)


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

class Deployment:
    def __init__(self, robot_cfg: RobotConfig, policy, port: str, log_dir: Path):
        self.robot_cfg = robot_cfg
        self.policy = policy
        self.logger, self.csv_writer, self.csv_file, csv_path = setup_logging(log_dir)
        self.logger.info("Logging control loop to %s", csv_path)
        write_csv_header(self.csv_writer, robot_cfg.joint_names)

        self.bus = MotorBus(port, robot_cfg.joints, self.logger)
        self.motion = MotionReference(DEFAULT_KEYFRAMES_DEG, degrees=True)
        self.motion_time = 0.0

        self._stop = False
        signal.signal(signal.SIGINT, self._handle_stop_signal)
        signal.signal(signal.SIGTERM, self._handle_stop_signal)

    def _handle_stop_signal(self, signum, frame):
        self.logger.warning("Received signal %s -- stopping and releasing motors.", signum)
        self._stop = True

    def startup_sequence(self):
        self.logger.info("Calibrating from current motor positions...")
        self.bus.calibrate_from_startup_position()
        self.bus.print_startup_table()

        answer = input(
            "Verify the table above: raw_deg should match the robot's actual pose, "
            "and invert flags should make sense. Enable motors and start policy? [y/N] "
        )
        if answer.strip().lower() != "y":
            self.logger.info("User declined startup confirmation. Exiting without enabling motors.")
            sys.exit(0)

    def build_obs(self, readings: dict) -> torch.Tensor:
        ref = self.motion.sample(self.motion_time)
        obs = []
        for i, joint in enumerate(self.robot_cfg.joints):
            obs.append(readings[joint.name].output_pos_rad)
        for i, joint in enumerate(self.robot_cfg.joints):
            obs.append(readings[joint.name].output_vel_rad_s)
        obs.extend(ref)
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    def run(self):
        dt = self.robot_cfg.control_dt
        # dt = 0.05
        # dt = 1 / 120
        step = 0
        try:
            new_readings = {j.name: self.bus.read_motor(j) for j in self.robot_cfg.joints}
            while not self._stop:
                loop_start = time.perf_counter()

                # readings = {j.name: self.bus.read_motor(j) for j in self.robot_cfg.joints}
                readings = new_readings
                # print(f"READINGS: {readings}")

                print(f"HIP: {math.degrees(readings['hip'].output_pos_rad):10.5f}  KNEE: {math.degrees(readings['knee'].output_pos_rad):10.5f}  ANKLE: {math.degrees(readings['ankle'].output_pos_rad):10.5f}")

                obs = self.build_obs(readings)
                # print(f"OBS: {obs}")

                with torch.no_grad():
                    action = self.policy(obs).squeeze(0).numpy()
                # print(f"ACTION: {action}")
                action = action.clip(-1.0, 1.0)
                # print(f"ACTION (CLIPPED): {action}")
                # action = [0,0,0]
                # self.robot_cfg.action_scale = 0.05

                ref = self.motion.sample(self.motion_time)
                # print(f"TIME: {self.motion_time}")
                # print(f"REF: {ref}")
                targets = {}
                for i, joint in enumerate(self.robot_cfg.joints):
                    targets[joint.name] = ref[i] + self.robot_cfg.action_scale * float(action[i])
                # print(f"TARGETS: {targets}")
                # self.robot_cfg.max_step_deg = 150

                try:
                    new_readings = self.bus.send_targets(targets, self.robot_cfg.max_step_deg, self.logger)
                except StepLimitExceeded as e:
                    self.logger.error("SAFETY ABORT: %s", e)
                    self.bus.release_all()
                    raise

                self._log_step(step, readings, obs, action, targets, new_readings)

                self.motion_time = (self.motion_time + dt) % self.motion.length
                step += 1

                elapsed = time.perf_counter() - loop_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif elapsed > dt * 1.5:
                    self.logger.warning("Control loop overrun: %.1f ms (target %.1f ms)", elapsed * 1000, dt * 1000)
        finally:
            self.bus.release_all()
            self.csv_file.close()
            self.logger.info("Deployment stopped, motors released, log file closed.")

    def _log_step(self, step, readings, obs, action, targets, new_readings):
        row = [step * self.robot_cfg.control_dt, self.motion_time]
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
        self.csv_writer.writerow(row)
        if step % 50 == 0:
            self.csv_file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="robot_config.json")
    parser.add_argument("--policy", type=Path, required=True, help="policy.pt (TorchScript)")
    parser.add_argument("--policy-meta", type=Path, required=True, help="policy.meta.json")
    parser.add_argument("--port", type=str, required=True, help="e.g. /dev/ttyUSB0")
    parser.add_argument("--log-dir", type=Path, default=Path("./logs"))
    args = parser.parse_args()

    logger = logging.getLogger("robot_deploy.startup")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    robot_cfg = RobotConfig.load(args.config)

    policy, meta = load_and_verify_policy(args.policy, args.policy_meta, robot_cfg, logger)

    deployment = Deployment(robot_cfg, policy, args.port, args.log_dir)
    try:
        deployment.startup_sequence()
        deployment.run()
    except StepLimitExceeded:
        logger.error("Exiting after safety abort. Motors have been released.")
        sys.exit(1)
    except Exception:
        logger.exception("Unhandled exception -- motors released via finally block.")
        raise


if __name__ == "__main__":
    main()

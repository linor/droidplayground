"""
mock_unitree_actuator_sdk.py

A drop-in stand-in for unitree_actuator_sdk that simulates GO-M8010-6-like
motor physics in software, so you can run robot_deploy.py end-to-end (startup
calibration, control loop, policy verification, step-limit abort, logging)
with zero hardware attached.

USAGE
-----
Don't import this directly into robot_deploy.py. Instead, at the top of
robot_deploy.py, temporarily point UNITREE_SDK_LIB_PATH's sys.path insertion
at the folder containing this file, OR simpler: run robot_deploy.py with this
file's directory first on PYTHONPATH and this module named exactly
"unitree_actuator_sdk" so the existing `from unitree_actuator_sdk import ...`
line picks it up unmodified:

    mkdir -p mock_sdk
    cp mock_unitree_actuator_sdk.py mock_sdk/unitree_actuator_sdk.py
    PYTHONPATH=mock_sdk python3 robot_deploy.py \
        --config robot_config.json \
        --policy deploy_bundle/policy.pt \
        --policy-meta deploy_bundle/policy.meta.json \
        --port MOCK

No real serial port is opened -- SerialPort(port) here ignores the port
string entirely and just simulates.

PHYSICS MODEL
-------------
Each motor is a simple critically-damped-ish 1-DOF system on the OUTPUT
side, converted to rotor units the same way real GO-M8010-6 commands are
(divide/multiply by gear ratio, kp/kd scaled by 1/r^2) so the round-trip
math in robot_deploy.py gets properly exercised, including the calibration
offset math and the step-limit check.

    tau = kp_rotor * (q_target_rotor - q_rotor) + kd_rotor * (dq_target - dq_rotor)
    ddq = tau / rotor_inertia
    (semi-implicit Euler integration at a fixed internal dt)

This is NOT a faithful simulation of the real actuator's FOC control loop,
electrical dynamics, or friction -- it exists purely to exercise the
software path (calibration, gear-ratio math, direction inversion, step
limiting, logging, policy verification), not to validate control gains or
sim-to-real transfer. Tune ROTOR_INERTIA / add friction below if you want a
more realistic feel for manual testing.
"""

import time


class MotorType:
    GO_M8010_6 = "GO_M8010_6"
    A1 = "A1"
    B1 = "B1"


class MotorMode:
    FOC = "FOC"


def queryMotorMode(motor_type, mode):
    return "FOC_MODE"


def queryGearRatio(motor_type):
    if motor_type == MotorType.GO_M8010_6:
        return 6.33
    return 1.0


class MotorCmd:
    def __init__(self):
        self.motorType = None
        self.mode = None
        self.id = 0
        self.kp = 0.0
        self.kd = 0.0
        self.q = 0.0
        self.dq = 0.0
        self.tau = 0.0


class MotorData:
    def __init__(self):
        self.motorType = None
        self.q = 0.0
        self.dq = 0.0
        self.tau = 0.0
        self.temp = 25.0
        self.merror = 0


class _SimMotor:
    ROTOR_INERTIA = 0.0006      # kg*m^2, rough guess, rotor side
    FRICTION_COEFF = 0.01       # simple linear damping term
    MAX_TORQUE_ROTOR = 3.0      # crude saturation so silly kp values don't blow up
    TEMP_BASE = 28.0

    def __init__(self, start_pos_rotor_rad: float = 0.0):
        self.q = start_pos_rotor_rad
        self.dq = 0.0
        self.temp = self.TEMP_BASE
        self.merror = 0
        self._last_t = time.perf_counter()

    def step(self, kp: float, kd: float, q_target: float, dq_target: float, tau_ff: float):
        now = time.perf_counter()
        sim_dt = min(max(now - self._last_t, 1e-4), 0.05)  # clamp so a slow debugger step doesn't explode
        self._last_t = now

        tau = kp * (q_target - self.q) + kd * (dq_target - self.dq) + tau_ff
        tau -= self.FRICTION_COEFF * self.dq
        tau = max(-self.MAX_TORQUE_ROTOR, min(self.MAX_TORQUE_ROTOR, tau))

        ddq = tau / self.ROTOR_INERTIA
        self.dq += ddq * sim_dt
        self.q += self.dq * sim_dt

        # gentle fake heating proportional to |tau|, decaying toward base
        self.temp += (0.002 * abs(tau) - 0.01 * (self.temp - self.TEMP_BASE))
        return tau


class SerialPort:
    """
    Mimics the real SerialPort's sendRecv-per-motor-id interface, but
    simulates each distinct `cmd.id` as its own independent motor. No actual
    port is opened -- `port` is accepted and ignored (kept for interface
    compatibility with real code).
    """

    def __init__(self, port: str):
        self.port = port
        self._motors: dict[int, _SimMotor] = {}
        print(f"[MOCK SDK] SerialPort('{port}') opened (simulated, no real hardware).")

    def _get_motor(self, motor_id: int) -> _SimMotor:
        if motor_id not in self._motors:
            # start each simulated motor at a random-ish but fixed nonzero
            # position so calibration has something realistic to read,
            # instead of everything conveniently already being zero.
            start = 0.15 * ((motor_id * 37) % 7 - 3)  # deterministic per id, in rotor rad
            self._motors[motor_id] = _SimMotor(start_pos_rotor_rad=start)
        return self._motors[motor_id]

    def sendRecv(self, cmd: MotorCmd, data: MotorData):
        motor = self._get_motor(cmd.id)
        tau = motor.step(cmd.kp, cmd.kd, cmd.q, cmd.dq, cmd.tau)

        data.motorType = cmd.motorType
        data.q = motor.q
        data.dq = motor.dq
        data.tau = tau
        data.temp = motor.temp
        data.merror = motor.merror

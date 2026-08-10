"""
isaac_sim_unitree_backend.py

Drop-in replacement for unitree_actuator_sdk, backed by the real QMINI
articulation running live in Isaac Sim, instead of the crude double-
integrator in mock_unitree_actuator_sdk.py. This gets you actual mass/
inertia/gear-coupling dynamics from the same USD/articulation you trained
on -- much closer to "real robot" than the arithmetic mock, without any
hardware.

*** NOT EXECUTED OR TESTED IN THIS SESSION ***
I don't have Isaac Sim / a GPU available in this sandbox, so unlike
mock_unitree_actuator_sdk.py (which I actually ran robot_deploy.py against),
this file is written from the Isaac Lab API as shown in your QminiLegEnv
code and Isaac Lab's public docs, but I could not import isaaclab or step a
simulation to confirm it runs. Treat it as a strong first draft -- run it on
your own Isaac Lab machine and expect to debug API details (this changes
between Isaac Lab versions).

WHAT THIS DOES NOT FAITHFULLY REPRODUCE
-----------------------------------------
1. Per-command kp/kd: robot_deploy.py sends a fresh kp/kd with every
   MotorCmd (like the real GO-M8010-6 firmware allows). Isaac Lab's
   articulation actuators (DCMotorCfg in your QMINI_CFG) use fixed
   stiffness/damping configured on the ArticulationCfg, not overridable
   per physics step through set_joint_position_target(). This backend
   *ignores* the incoming cmd.kp/cmd.kd and instead relies on whatever
   stiffness/damping QMINI_CFG's DCMotorCfg specifies. If you need the
   robot_config.json kp/kd values to actually apply here, you'd need to
   rebuild the ArticulationCfg's actuators with those gains, or switch to
   direct torque control (set_joint_effort_target) and compute the PD
   torque yourself in this file to mirror what robot_deploy.py's targets
   imply.
2. Timing: robot_deploy.py calls sendRecv once per joint per phase
   (a no-op "read" phase, then a "command" phase), i.e. 2*n_joints calls
   per control loop iteration, not one call per control loop. This shim
   steps physics by (decimation physics substeps) only on the LAST joint
   of the command phase in each cycle, tracked via a simple call counter --
   verify this actually lines up for your joint count/ordering.
3. No contact/ground -- this spawns just the articulation for simplicity.
   Add spawn_ground_plane() back in if your policy's success depends on
   ground contact/support reactions on the legs (for a single dangling leg
   test it likely doesn't matter, but say so if you're testing standing).

SETUP
-----
Run this the same way you'd run any Isaac Lab standalone script (from
inside your Isaac Lab python env, with droidplayground on PYTHONPATH so
`from droidplayground.assets.qmini import QMINI_CFG` resolves):

    mkdir -p isaac_sdk
    cp isaac_sim_unitree_backend.py isaac_sdk/unitree_actuator_sdk.py
    PYTHONPATH=isaac_sdk ./isaaclab.sh -p robot_deploy.py \
        --config robot_config.json \
        --policy deploy_bundle/policy.pt \
        --policy-meta deploy_bundle/policy.meta.json \
        --port ISAAC

The AppLauncher below defaults to headless=False so you can watch the leg
move; pass --headless via ISAAC_SIM_HEADLESS=1 env var if you want it
off-screen (e.g. running over SSH).
"""

import os

# --- Isaac Sim app must be launched before importing isaaclab.* anything ---
from isaaclab.app import AppLauncher  # noqa: E402

_headless = os.environ.get("ISAAC_SIM_HEADLESS", "0") == "1"
_app_launcher = AppLauncher(headless=_headless)
simulation_app = _app_launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationCfg  # noqa: E402
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane  # noqa: E402

# Reuse the exact articulation config you trained with.
from droidplayground.assets.qmini import QMINI_CFG  # noqa: E402


# ---------------------------------------------------------------------------
# Map your generic joint names (robot_config.json: "hip"/"knee"/"ankle") to
# the actual USD joint names in QMINI_CFG. EDIT THIS to match your asset --
# I don't know the real joint names, only that the actuators regex on
# ".*pitch" / ".*knee" / ".*ankle".
# ---------------------------------------------------------------------------
JOINT_NAME_MAP = {
    "hip": "Revolute_left_pitch",
    "knee": "Revolute_left_knee",
    "ankle": "Revolute_left_ankle",
}

PHYSICS_DT = 1 / 200.0
DECIMATION = 4  # matches QminiLegEnvCfg.decimation


class MotorType:
    GO_M8010_6 = "GO_M8010_6"


class MotorMode:
    FOC = "FOC"


def queryMotorMode(motor_type, mode):
    return "FOC_MODE"


def queryGearRatio(motor_type):
    return 6.33  # verify against your real SDK / datasheet


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


class _IsaacSimBackend:
    """
    Singleton-ish holder for the live sim -- SerialPort instances share this
    so all three joints/motor ids drive the same running simulation instead
    of each spinning up their own Isaac Sim app.
    """
    _instance = None

    def __init__(self):
        self.sim = sim_utils.SimulationContext(SimulationCfg(dt=PHYSICS_DT))
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        robot_cfg = QMINI_CFG.replace(prim_path="/World/Robot")
        self.robot = Articulation(robot_cfg)
        self.sim.reset()

        print("\n[ISAAC BACKEND] Robot joints in this USD:")
        for i, name in enumerate(self.robot.joint_names):
            print(f"  {i:2d}: {name}")
        print("[ISAAC BACKEND] Verify JOINT_NAME_MAP above against this list.\n")

        self._joint_idx = {}
        for generic_name, usd_name in JOINT_NAME_MAP.items():
            try:
                self._joint_idx[generic_name] = self.robot.joint_names.index(usd_name)
            except ValueError:
                raise RuntimeError(
                    f"Joint '{usd_name}' (mapped from '{generic_name}') not found in "
                    f"USD joint list above. Fix JOINT_NAME_MAP."
                )

        self._pending_targets_rotor = {}  # generic_name -> rotor-side q target
        self._gear_ratio = queryGearRatio(MotorType.GO_M8010_6)

    @classmethod
    def get(cls) -> "_IsaacSimBackend":
        if cls._instance is None:
            cls._instance = _IsaacSimBackend()
        return cls._instance

    def _generic_name_for_id(self, motor_id: int) -> str:
        # robot_deploy.py's MotorBus doesn't pass joint names into sendRecv,
        # only cmd.id -- so this shim needs its own id->name mapping,
        # matching the motor_id values in your robot_config.json.
        id_to_name = {0: "hip", 1: "knee", 2: "ankle"}  # EDIT to match robot_config.json
        if motor_id not in id_to_name:
            raise RuntimeError(f"Unknown motor_id {motor_id}, update id_to_name mapping.")
        return id_to_name[motor_id]

    def read_or_command(self, cmd: MotorCmd, data: MotorData):
        name = self._generic_name_for_id(cmd.id)
        idx = self._joint_idx[name]

        is_command = cmd.kp > 0.0 or cmd.kd > 0.0

        if is_command:
            # cmd.q is rotor-side (already gear-ratio- and direction-scaled
            # by robot_deploy.py's MotorBus) -- convert back to the raw
            # physical joint angle Isaac Lab expects.
            physical_target = cmd.q / self._gear_ratio
            self._pending_targets_rotor[name] = physical_target
            self.robot.set_joint_position_target(
                torch.tensor([[physical_target]]), joint_ids=[idx]
            )

            # Step physics once we've received a command for all mapped
            # joints this cycle (approximation -- see module docstring).
            if len(self._pending_targets_rotor) >= len(self._joint_idx):
                for _ in range(DECIMATION):
                    self.robot.write_data_to_sim()
                    self.sim.step()
                    self.robot.update(PHYSICS_DT)
                self._pending_targets_rotor.clear()

        # Report current physical state, converted to rotor units, regardless
        # of whether this call was a read or a command (matches real SDK
        # semantics: every sendRecv returns current telemetry).
        physical_pos = float(self.robot.data.joint_pos[0, idx])
        physical_vel = float(self.robot.data.joint_vel[0, idx])
        data.motorType = cmd.motorType
        data.q = physical_pos * self._gear_ratio
        data.dq = physical_vel * self._gear_ratio
        data.tau = float(self.robot.data.applied_torque[0, idx]) if hasattr(self.robot.data, "applied_torque") else 0.0
        data.temp = 30.0  # Isaac Lab has no thermal model here; placeholder
        data.merror = 0


class SerialPort:
    def __init__(self, port: str):
        self.port = port
        self.backend = _IsaacSimBackend.get()
        print(f"[ISAAC BACKEND] SerialPort('{port}') -> live Isaac Sim articulation (not real hardware).")

    def sendRecv(self, cmd: MotorCmd, data: MotorData):
        self.backend.read_or_command(cmd, data)

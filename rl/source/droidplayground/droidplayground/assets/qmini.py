"""Configuration for a simple servo, based on the Dynamixel XT-330."""


import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg
from droidplayground.assets import ASSET_USD_DIRECTORY

##
# Configuration
##

# Output-side (joint-space) stiffness/damping/armature per actuator group.
# These are also the values a real GO-M8010-6 deployment must convert to
# rotor-side via kp_rotor = kp_output / gear_ratio**2 (see robot_deploy.py's
# MotorBus.send_targets). Keep this dict as the single source of truth for
# "current best sim gains" -- both QMINI_CFG below and build_qmini_cfg()
# read from it.
DEFAULT_GAINS = {
    "hip_pitch": dict(stiffness=75.0, damping=2.0, armature=0.02),
    "knee": dict(stiffness=45.0, damping=2.0, armature=0.02),
    "ankle": dict(stiffness=30.0, damping=2.0, armature=0.02),
}


def build_qmini_cfg(gains: dict | None = None) -> ArticulationCfg:
    """Return a QMINI ArticulationCfg with actuator gains optionally overridden.

    Use this instead of the module-level QMINI_CFG constant whenever you need
    to change stiffness/damping/armature after import time -- e.g. sweeping
    values in tune_pid_isaaclab.py, or sampling per-episode randomized gains
    in QminiLegEnv's domain randomization.

    Args:
        gains: optional dict keyed by actuator group name
            ("hip_pitch" | "knee" | "ankle"), each value a dict with any of
            "stiffness" / "damping" / "armature". Fields not given for a
            group fall back to DEFAULT_GAINS for that group. Example:

                build_qmini_cfg({
                    "hip_pitch": {"stiffness": 75.0, "damping": 0.3},
                    "knee":      {"stiffness": 45.0, "damping": 0.5},
                    "ankle":     {"stiffness": 30.0, "damping": 0.25},
                })

    NOTE: not run against a real isaaclab install in this session -- this
    relies on DCMotorCfg / ArticulationCfg supporting .replace(**kwargs) the
    same way QMINI_CFG.replace(prim_path=...) is used elsewhere in this repo
    (qmini_leg_env.py, isaac_sim_unitree_backend.py). If your isaaclab
    version's configclass doesn't support .replace() the way I'm assuming,
    swap this for dataclasses.replace(actuator_cfg, **overrides).
    """
    merged = {name: dict(vals) for name, vals in DEFAULT_GAINS.items()}
    if gains:
        for name, overrides in gains.items():
            if name not in merged:
                raise KeyError(
                    f"Unknown actuator group '{name}', expected one of {list(merged)}"
                )
            merged[name].update(overrides)

    new_actuators = {}
    for name, actuator_cfg in QMINI_CFG.actuators.items():
        new_actuators[name] = actuator_cfg.replace(**merged[name])
    return QMINI_CFG.replace(actuators=new_actuators)


QMINI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_USD_DIRECTORY}/qmini_urdf.usda",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0), # joint_pos={".*": 0.0}
        joint_pos={
            "Revolute_left_pitch": -0.216651,
            "Revolute_left_knee": 0.331148,
            "Revolute_left_ankle": 0.254123
        },
    ),
    actuators={
        # "xl330_velocity_actuator": DCMotorCfg(
        #     joint_names_expr=[".*"],

        #     # effort_limit=23.7,          # continuous output torque
        #     # The datasheet advertises maximum torque, not continuous torque. Looking at the specifications:
        #     # A more conservative simulation would use something like
        #     effort_limit=16.0,   # or 18 Nm

        #     saturation_effort=23.7,     # peak torque used for saturation model
        #     velocity_limit=30.0,        # rad/s

        #     # velocity drive: stiffness = 0, damping = velocity gain
        #     stiffness=60.0,      # tune
        #     damping=1.5,         # tune
        # )

        # For the GO-M8010-6 with its 6.33:1 reduction, I'd start with:
        # armature = 0.002  # kg·m²

        # If the joint oscillates, increase damping first. If it's sluggish but stable, increase stiffness.
        # Gains come from DEFAULT_GAINS above -- edit that dict, not these
        # literals, so build_qmini_cfg()'s "fall back to defaults" behavior
        # stays consistent with what QMINI_CFG itself spawns with.
        "hip_pitch": DCMotorCfg(
            joint_names_expr=[".*pitch"],
            effort_limit=18.0,
            saturation_effort=23.7,
            velocity_limit=30.0,
            **DEFAULT_GAINS["hip_pitch"],
        ),

        "knee": DCMotorCfg(
            joint_names_expr=[".*knee"],
            effort_limit=18.0,
            saturation_effort=23.7,
            velocity_limit=30.0,
            **DEFAULT_GAINS["knee"],
        ),

        "ankle": DCMotorCfg(
            joint_names_expr=[".*ankle"],
            effort_limit=18.0,
            saturation_effort=23.7,
            velocity_limit=30.0,
            **DEFAULT_GAINS["ankle"],
        ),
    }
)

"""Configuration for a simple servo, based on the Dynamixel XT-330."""


import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg
from droidplayground.assets import ASSET_USD_DIRECTORY

##
# Configuration
##

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
        pos=(0.0, 0.0, 2.0), joint_pos={".*": 0.0}
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
        "hip_pitch": DCMotorCfg(
            joint_names_expr=[".*pitch"],
            stiffness=140.0,
            damping=5.0,
            effort_limit=18.0,
            saturation_effort=23.7,
            velocity_limit=30.0,
            armature = 0.002,  # kg·m²
        ),

        "knee": DCMotorCfg(
            joint_names_expr=[".*knee"],
            stiffness=180.0,
            damping=6.0,
            effort_limit=18.0,
            saturation_effort=23.7,
            velocity_limit=30.0,
            armature = 0.002,  # kg·m²
        ),

        "ankle": DCMotorCfg(
            joint_names_expr=[".*ankle"],
            stiffness=90.0,
            damping=3.0,
            effort_limit=18.0,
            saturation_effort=23.7,
            velocity_limit=30.0,
            armature = 0.002,  # kg·m²
        )


    }
)

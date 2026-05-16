"""Configuration for a simple servo, based on the Dynamixel XT-330."""


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from droidplayground.assets import ASSET_USD_DIRECTORY

##
# Configuration
##

XL330_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_USD_DIRECTORY}/xl330/xl330.usd",
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
        pos=(0.0, 0.0, 2.0), joint_pos={"Revolute_1": 0.0}
    ),
    actuators={
        "xl330_velocity_actuator": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_1"],

            # XL330-M288-T @ 5V
            #effort_limit_sim=0.52,          # Nm, stall torque
            effort_limit_sim=2.0,          
            velocity_limit_sim=10.8,        # rad/s, 103 rpm

            # velocity drive: stiffness = 0, damping = velocity gain
            stiffness=0.0,
            damping=4,
        )
    }
)

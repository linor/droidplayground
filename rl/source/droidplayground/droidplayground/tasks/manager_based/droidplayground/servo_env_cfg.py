import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##

from droidplayground.assets.xl330 import XL330_CFG


##
# Scene definition
##


@configclass
class ServoSceneCfg(InteractiveSceneCfg):
    """Configuration for a simple servo scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = XL330_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    target_angle = mdp.RandomJointAngleCommandCfg(
        class_type=mdp.RandomJointAngleCommand,
        angle_range=(-math.pi, math.pi),
        resampling_time_range=(1.0, 3.0),
        debug_vis=True,
        line_length=0.030,
        line_width=3.0,
        line_z_offset=0.001,
        visual_angle_offset=math.pi / 2.0,
        debug_env_count=99,
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["Revolute_1"],
        scale=10.8,       # action -1..1 maps to -10.8..10.8 rad/s
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        target_angle_error = ObsTerm(func=mdp.target_angle_error, params={ "asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute_1"])})
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute_1"])})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    
    reset_env = EventTerm(
        func=mdp.reset_joints_by_offset, 
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute_1"]),
            "position_range": (-math.pi, math.pi),
            "velocity_range": (-0.5, 0.5),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    track_target = RewTerm(
        func=mdp.track_target_angle_exp,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute_1"]),
            "std": 0.25,
        },
    )

    joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_l2_penalty,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Revolute_1"]),
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class ServoEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: ServoSceneCfg = ServoSceneCfg(num_envs=4096, env_spacing=0.1)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (0.0, 0.0, 4.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from droidplayground.assets.qmini import QMINI_CFG
# from .phase_modulator import PhaseModulator
from .motion_player import MotionPlayer

@configclass
class QminiLegEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 3
    observation_space = 9
    state_space = 0
    action_scale = 0.5

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = QMINI_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=1.0, replicate_physics=True)

class QminiLegEnv(DirectRLEnv):
    cfg: QminiLegEnvCfg

    def __init__(self, cfg: QminiLegEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        print("\nRobot joints:")
        for joint_id, joint_name in enumerate(self.robot.joint_names):
            print(f"  {joint_id:2d}: {joint_name}")

        self.num_phases = 1
        self.phase_frequency = 0.5  # Hz: one complete cycle every 2 seconds

        # self.phase_modulator = PhaseModulator(
        #     time_step=self.step_dt,
        #     num_envs=self.num_envs,
        #     device=self.device,
        # )
        self.motion = MotionPlayer(
            keyframes=[
                # time, [hip, knee, ankle]

                # leg lifted
                (0.0, [20, -50, 30]),

                # leg passing through
                (0.4, [5, -20, 10]),

                # leg extended
                (0.8, [-15, 10, 5]),

                # return swing
                (1.2, [5, -20, 10]),

                # back to lifted
                (1.6, [20, -50, 30]),
            ],
            # Same motion, half the speed.
            # keyframes=[
            #     (0.0, [20, -50, -30]),
            #     (0.6, [5, -20, -10]),
            #     (1.2, [-15, 10, -5]),
            #     (1.8, [5, -20, -10]),
            #     (2.4, [20, -50, -30]),
            # ],
            device=self.device,
            degrees=True,
        )

        self.motion_time = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float32,
        )

        # all_env_ids = torch.arange(
        #     self.num_envs,
        #     dtype=torch.long,
        #     device=self.device,
        # )

        # self.phase_modulator.reset(
        #     env_ids=all_env_ids,
        #     deterministic=self.render_mode is not None,
        # )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        # self.phase_modulator.compute()
        # self.motion_time += self.step_dt
        # self.motion_time = torch.zeros(
        #     self.num_envs,
        #     device=self.device,
        # )
        self.motion_time = (
            self.motion_time + self.step_dt
        ) % self.motion.length

        # self.actions[:, 0] = 0.0
        # self.actions[:, 2] = 0.0

    def _get_observations(self):
        # phase = self.phase_modulator.phase
        reference = self.motion.sample(self.motion_time)

        # observations = torch.cat(
        #     (
        #         torch.sin(phase),                    # 1
        #         torch.cos(phase),                    # 1
        #         self.robot.data.joint_pos[:, :3],   # 3
        #         self.robot.data.joint_vel[:, :3],   # 3
        #     ),
        #     dim=-1,
        # )
        observations = torch.cat(
            (
                self.robot.data.joint_pos[:, :3],
                self.robot.data.joint_vel[:, :3],
                reference,
            ),
            dim=-1
        )
        
        return {
            "policy": observations
        }

    def _apply_action(self) -> None:
        num_actions = self.cfg.action_space
        joint_ids = slice(0, num_actions)

        # position_targets = (
        #     self.robot.data.default_joint_pos[:, joint_ids]
        #     + self.cfg.action_scale * self.actions[:, :num_actions]
        # )
        reference = self.motion.sample(self.motion_time)

        position_targets = (
            reference
            + 0.15*self.actions
        )

        self.robot.set_joint_position_target(
            position_targets,
            joint_ids=joint_ids,
        )

    def _get_rewards(self) -> torch.Tensor:
        # phase = self.phase_modulator.phase[:, 0]

        # target_position = 0.5 * torch.sin(phase)
        # actual_position = self.robot.data.joint_pos[:, 1]
        # actual_velocity = self.robot.data.joint_vel[:, 1]

        # tracking_error = actual_position - target_position

        # tracking_reward = torch.exp(
        #     -5.0 * tracking_error.square()
        # )

        # velocity_penalty = 0.001 * actual_velocity.square()

        # return tracking_reward - velocity_penalty

        reference = self.motion.sample(self.motion_time)

        error = self.robot.data.joint_pos[:, :3] - reference

        tracking_reward = torch.exp(
            -5.0 * torch.sum(error**2, dim=1)
        )


        reference_next = self.motion.sample(
            self.motion_time + self.step_dt
        )

        reference_velocity = (
            reference_next-reference
        )/self.step_dt

        velocity_error = (
            self.robot.data.joint_vel[:, :3]
            - reference_velocity
        )

        velocity_reward = torch.exp(
            -0.5*torch.sum(
                velocity_error**2,
                dim=1,
            )
        )

        reward = (
            2.0*tracking_reward
            + velocity_reward
        )

        # ---------------------------------
        # Trajectory logging (env 0 only)
        # ---------------------------------
        self.extras["log"] = {
            "tracking/hip_error": torch.rad2deg(torch.mean(torch.abs(error[:, 0]))),
            "tracking/knee_error": torch.rad2deg(torch.mean(torch.abs(error[:, 1]))),
            "tracking/ankle_error": torch.rad2deg(torch.mean(torch.abs(error[:, 2]))),
            "tracking/reward": reward.mean(),

            # Reference vs actual joint positions
            "motion/ref_hip": torch.rad2deg(reference[0, 0]),
            "motion/ref_knee": torch.rad2deg(reference[0, 1]),
            "motion/ref_ankle": torch.rad2deg(reference[0, 2]),

            "motion/actual_hip": torch.rad2deg(self.robot.data.joint_pos[0, 0]),
            "motion/actual_knee": torch.rad2deg(self.robot.data.joint_pos[0, 1]),
            "motion/actual_ankle": torch.rad2deg(self.robot.data.joint_pos[0, 2]),
        }

        # Print every 500 simulation steps
        if self.common_step_counter % 100 == 0:
            ref = torch.rad2deg(reference[0]).cpu()
            act = torch.rad2deg(self.robot.data.joint_pos[0, :3]).cpu()

            print("\n----------------------------")
            print(f"Step {self.common_step_counter}")
            print(f"Reference : {ref.numpy()}")
            print(f"Actual    : {act.numpy()}")
            print(f"Error (°) : {(act-ref).numpy()}")

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # self.phase_modulator.reset(
        #     env_ids=env_ids,
        #     deterministic=self.render_mode is not None,
        # )

        self.motion_time[env_ids] = torch.rand(
            len(env_ids),
            device=self.device,
        ) * self.motion.length


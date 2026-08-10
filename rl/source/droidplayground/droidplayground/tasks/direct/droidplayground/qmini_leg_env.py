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
    decimation = 4
    episode_length_s = 5.0
    # - spaces definition
    action_space = 3
    observation_space = 7
    state_space = 0
    action_scale = 0.5

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = QMINI_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=1.0, replicate_physics=True)

    # --- sim-to-real robustness ---------------------------------------
    # Extra zero-order-hold action delay, in control steps (each step =
    # decimation * sim.dt = 20ms at the defaults above), sampled per-env
    # per-episode from this (min, max) range. Set this from what
    # analyze_delay.py measures on the real robot -- e.g. if it reports
    # ~25-60ms of target->actual lag, that's roughly 1-3 steps here.
    action_delay_range_steps: tuple[int, int] = (0, 3)

    # Per-episode multiplicative randomization applied to each actuator
    # group's stiffness/damping/armature, as a fraction of qmini.py's
    # DEFAULT_GAINS (or whatever build_qmini_cfg() gains this env was built
    # with). (0.7, 1.3) means each episode samples a value in
    # [0.7x, 1.3x] of the base gain, independently per env and per group.
    gain_randomization_range: dict = {
        "stiffness": (0.7, 1.3),
        "damping": (0.7, 1.3),
        "armature": (0.8, 1.2),
    }

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
                (0.0, [-20, 50, 30]),

                # leg passing through
                (0.4, [-5, 20, 10]),

                # leg extended
                (0.8, [15, -10, 5]),

                # return swing
                (1.2, [-5, 20, 10]),

                # back to lifted
                (1.6, [-20, 50, 30]),
            ],
            device=self.device,
            degrees=True,
        )

        self.motion_time = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float32,
        )

        # NOTE on ordering: like self.motion_time above, these are created
        # AFTER super().__init__() returns, but _reset_idx() (which uses
        # them) is written assuming they already exist. This only works if
        # your DirectRLEnv doesn't call _reset_idx during __init__ itself
        # (i.e. reset() is called externally afterward) -- true for
        # self.motion_time in the code as you gave it to me, so I'm
        # following the same assumption here. If your Isaac Lab version
        # does trigger an implicit reset inside __init__, guard
        # _randomize_action_delay/_randomize_actuator_gains with an
        # `if not hasattr(self, "_action_buffer"): return`.

        # --- action delay buffer ---------------------------------------
        # Circular buffer of the last `_action_buffer_len` raw actions per
        # env. _apply_action reads back `action_delay_steps[env]` steps
        # behind the write pointer, so each env sees a per-episode-fixed
        # zero-order-hold delay instead of the instantaneous action Isaac
        # Lab would otherwise apply. See action_delay_range_steps in cfg.
        self._action_buffer_len = max(1, self.cfg.action_delay_range_steps[1] + 1)
        self._action_buffer = torch.zeros(
            self.num_envs, self._action_buffer_len, self.cfg.action_space,
            device=self.device, dtype=torch.float32,
        )
        self._action_delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._buffer_ptr = 0
        self._delayed_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # --- default (unrandomized) actuator gains, captured once so
        # per-episode randomization always scales from the same baseline
        # instead of compounding across resets. Verify these tensor shapes
        # against your installed Isaac Lab version -- assumed here to be
        # (num_envs, num_joints_in_group), matching how DCMotorCfg's scalar
        # stiffness/damping/armature get broadcast at Articulation init.
        self._default_actuator_gains = {}
        for name, actuator in self.robot.actuators.items():
            self._default_actuator_gains[name] = {
                "stiffness": actuator.stiffness.clone(),
                "damping": actuator.damping.clone(),
                "armature": actuator.armature.clone(),
            }

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

        # --- action delay ------------------------------------------------
        # Write this step's action into the circular buffer, then read back
        # each env's own delayed action (fixed for the episode, resampled
        # on reset -- see _reset_idx). _apply_action uses
        # self._delayed_actions instead of self.actions directly.
        self._action_buffer[:, self._buffer_ptr, :] = self.actions
        read_idx = (self._buffer_ptr - self._action_delay_steps) % self._action_buffer_len
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._delayed_actions = self._action_buffer[env_ids, read_idx, :]
        self._buffer_ptr = (self._buffer_ptr + 1) % self._action_buffer_len

    def _get_observations(self):
        # phase = self.phase_modulator.phase
        # reference = self.motion.sample(self.motion_time)

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
                # reference,
                self.motion_time.unsqueeze(1)
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
        # reference = self.motion.sample(self.motion_time)

        # position_targets = (
        #     reference
        #     + 0.15*self.actions
        # )

        # Was: position_targets = (self.actions) -- now goes through the
        # per-env action-delay buffer computed in _pre_physics_step, so the
        # policy has to be robust to the same lag analyze_delay.py measures
        # on the real robot instead of assuming instantaneous actuation.
        position_targets = self._delayed_actions[:, :num_actions]

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

        self._randomize_action_delay(env_ids)
        self._randomize_actuator_gains(env_ids)

    def _randomize_action_delay(self, env_ids: Sequence[int]):
        low, high = self.cfg.action_delay_range_steps
        n = len(env_ids)
        self._action_delay_steps[env_ids] = torch.randint(
            low, high + 1, (n,), device=self.device, dtype=torch.long,
        )
        # Clear stale pre-reset actions out of the buffer for these envs so
        # a delayed readback right after reset can't replay an action from
        # a different episode.
        self._action_buffer[env_ids, :, :] = 0.0

    def _randomize_actuator_gains(self, env_ids: Sequence[int]):
        for name, actuator in self.robot.actuators.items():
            defaults = self._default_actuator_gains[name]
            for field, (low, high) in self.cfg.gain_randomization_range.items():
                base = defaults[field][env_ids]
                scale = sample_uniform(low, high, base.shape, device=self.device)
                getattr(actuator, field)[env_ids] = base * scale


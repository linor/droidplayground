from __future__ import annotations

import json
import math
import torch
from collections.abc import Sequence
from pathlib import Path

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

# Reference gait clip: time (s) -> 10 joint angles, in the clip's own
# `joint_order` (see the file's "joint_order" key) -- NOT necessarily the
# articulation's joint order. _load_reference_keyframes() below reorders
# columns to match self.robot.joint_names at load time.
KEYFRAMES_PATH = Path(__file__).parent / "keyframes_forward_slow_all_joints_4x.json"


def _load_reference_keyframes(json_path: Path, joint_names: list[str]):
    """Load a keyframe clip and reorder its columns to match `joint_names`.

    The clip stores columns in its own `joint_order` (plain names like
    "left_yaw", grouped left-then-right). The articulation's joint order can
    differ (e.g. "Revolute_left_yaw", interleaved left/right/left/right) --
    so each articulation joint is looked up by name rather than assuming the
    two orderings already match.
    """
    with open(json_path) as f:
        data = json.load(f)

    clip_order = data["joint_order"]
    column_for_name = {name: i for i, name in enumerate(clip_order)}

    columns = []
    for joint_name in joint_names:
        key = joint_name.removeprefix("Revolute_")
        if key not in column_for_name:
            raise KeyError(
                f"Robot joint '{joint_name}' (looked up as '{key}') has no "
                f"matching column in {json_path}'s joint_order={clip_order}"
            )
        columns.append(column_for_name[key])

    keyframes = [(t, [pose[c] for c in columns]) for t, pose in data["keyframes"]]
    return keyframes, bool(data.get("degrees", True))


@configclass
class QminiLegEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 10.0
    # - spaces definition
    # 10 actuated joints (yaw/roll/pitch/knee/ankle x left/right), see
    # qmini_step_in_place isaac joint listing / keyframes_forward_slow_all_joints_4x.json.
    action_space = 10
    observation_space = 21  # 10 joint_pos + 10 joint_vel + 1 motion_time
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

    # Weight on the action-rate penalty in _get_rewards (-weight *
    # sum((action_t - action_{t-1})**2)). 0.0 = off (previous behavior).
    # Start small (e.g. 0.01-0.05) and increase if deployed targets are
    # still visibly jittery -- too high will fight the tracking_reward and
    # produce a sluggish policy that can't keep up with the gait.
    action_rate_penalty_weight: float = 0.02

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
        self.num_joints = self.cfg.action_space
        assert len(self.robot.joint_names) == self.num_joints, (
            f"Expected {self.num_joints} actuated joints (action_space), "
            f"but the articulation has {len(self.robot.joint_names)}: "
            f"{self.robot.joint_names}"
        )

        keyframes, degrees = _load_reference_keyframes(KEYFRAMES_PATH, self.robot.joint_names)
        self.motion = MotionPlayer(
            keyframes=keyframes,
            device=self.device,
            degrees=degrees,
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

        # --- action-rate penalty state ------------------------------------
        # Tracks the raw action from the previous step (pre-delay, i.e.
        # self.actions, not self._delayed_actions) so _get_rewards can
        # penalize step-to-step jitter. Without this, nothing in the
        # reward discourages a noisy/twitchy action sequence that averages
        # out to good tracking but demands sharp torque transients from the
        # real motor's PD controller on every step -- exactly the kind of
        # policy behavior that can trip a real overcurrent/fault protection
        # even when a smooth reference trajectory at similar positions
        # doesn't (see robot_deploy.py's --action-smoothing flag, added as
        # a deploy-side diagnostic for this same symptom).
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

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
                self.robot.data.joint_pos[:, :self.num_joints],
                self.robot.data.joint_vel[:, :self.num_joints],
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

        # reference = self.motion.sample(self.motion_time)

        # position_targets = (
        #     reference
        #     + 0.15*self.actions
        # )

        # Actions are a small offset from a FIXED anchor pose
        # (default_joint_pos, i.e. keyframe-0 / the robot's calibrated
        # "home" pose -- see qmini.py's ArticulationCfg.InitialStateCfg),
        # not an absolute joint target. This gives the policy a built-in
        # "don't jump" bias (a near-zero-mean action net starts out
        # commanding targets close to the current/reset pose) and, just as
        # important, keeps deployment simple: robot_deploy.py only needs to
        # know this one fixed calibration pose, not the full keyframe
        # animation, to reconstruct the same targets on the real robot --
        # see MotorBus / Deployment.run()'s use of default_pose_rad there.
        #
        # Goes through the per-env action-delay buffer computed in
        # _pre_physics_step, so the policy has to be robust to the same lag
        # analyze_delay.py measures on the real robot instead of assuming
        # instantaneous actuation.
        position_targets = (
            self.robot.data.default_joint_pos[:, joint_ids]
            + self.cfg.action_scale * self._delayed_actions[:, :num_actions]
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

        error = self.robot.data.joint_pos[:, :self.num_joints] - reference

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
            self.robot.data.joint_vel[:, :self.num_joints]
            - reference_velocity
        )

        velocity_reward = torch.exp(
            -0.5*torch.sum(
                velocity_error**2,
                dim=1,
            )
        )

        # Penalize step-to-step action jitter. Uses the raw (pre-delay)
        # action, since the delay buffer's job is to simulate real
        # target->actuation lag -- jitter is a property of what the policy
        # *outputs*, independent of when it actually reaches the motor.
        action_rate_penalty = self.cfg.action_rate_penalty_weight * torch.sum(
            (self.actions - self._prev_actions) ** 2, dim=1
        )
        self._prev_actions = self.actions.clone()

        reward = (
            2.0*tracking_reward
            + velocity_reward
            - action_rate_penalty
        )

        # ---------------------------------
        # Trajectory logging (env 0 only)
        # ---------------------------------
        joint_labels = [n.removeprefix("Revolute_") for n in self.robot.joint_names[:self.num_joints]]

        log = {
            "tracking/reward": reward.mean(),
            "tracking/action_rate_penalty": action_rate_penalty.mean(),
        }
        for i, label in enumerate(joint_labels):
            log[f"tracking/{label}_error"] = torch.rad2deg(torch.mean(torch.abs(error[:, i])))
            # Reference vs actual joint positions
            log[f"motion/ref_{label}"] = torch.rad2deg(reference[0, i])
            log[f"motion/actual_{label}"] = torch.rad2deg(self.robot.data.joint_pos[0, i])
        self.extras["log"] = log

        # Print every 500 simulation steps
        if self.common_step_counter % 100 == 0:
            ref = torch.rad2deg(reference[0]).cpu()
            act = torch.rad2deg(self.robot.data.joint_pos[0, :self.num_joints]).cpu()

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

        # Reset to phase 0, matching the actual robot pose after reset
        # (default_joint_pos == keyframe-0, see qmini.py's
        # ArticulationCfg.InitialStateCfg). Randomizing this independently
        # of the physical reset pose (as an earlier version of this file
        # did) trains the network on (joint_pos=default, motion_time=random)
        # pairs that never occur together in reality -- the joint can't
        # teleport to match a random phase -- which is exactly what caused
        # the bad first-action-of-episode jump previously. Don't reintroduce
        # phase randomization here without also moving the robot's actual
        # joint state to match (see the earlier discussion in this thread).
        self.motion_time[env_ids] = 0.0

        # Avoid penalizing the first post-reset action against whatever
        # action happened to be in _prev_actions from a different episode
        # (or, for envs reset at __init__ time, from the zero-init default,
        # which is fine and intentional -- zero is a neutral "no offset
        # from default_joint_pos" action).
        self._prev_actions[env_ids] = 0.0

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


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

# Reference clip: time (s) -> 10 joint angles, in the clip's own
# `joint_order` (see the file's "joint_order" key) -- NOT necessarily the
# articulation's joint order. _load_reference_keyframes() below reorders
# columns to match self.robot.joint_names at load time.
#
# Currently a static standing pose (both keyframes identical), NOT the
# walking gait -- see keyframes_standing_still.json's _comment. Switched
# from keyframes_forward_slow_all_joints_4x.json because the robot's real
# mass distribution (battery + Pi mount + decoration, all rear-mounted on
# base_link, added in qmini_urdf-2legs.usda) made the walking gait's
# start pose statically unbalanced -- it fell over before a policy could
# ever get a useful action in, regardless of training (see qmini.py's
# init_state.joint_pos comment). Standing is also just an easier first
# problem than walking + balancing at the same time. Point this back at
# keyframes_forward_slow_all_joints_4x.json (and revert qmini.py's
# init_state.joint_pos to match its frame-0) once standing balance works
# and you're ready to reintroduce the gait -- the lean bias found by
# tune_stance_lean_isaaclab.py will very likely need to be re-applied
# throughout that clip too, not just at frame 0.
KEYFRAMES_PATH = Path(__file__).parent / "keyframes_standing_still.json"


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
    # 10 joint_pos + 10 joint_vel + 3 projected_gravity_b (IMU accel-like,
    # unit vector) + 3 root_ang_vel_b (IMU gyro-like, rad/s) + 1 motion_time.
    # See _get_observations -- this layout must match robot_deploy.py's
    # build_obs() exactly (joint pos/vel, then the 6 IMU terms, then
    # motion_time), since that's what a real IMU reading gets slotted into.
    observation_space = 27
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

    # Max per-joint-TYPE random offset (degrees) added to default_joint_pos
    # at every episode reset, independently per env and per joint -- see
    # _reset_idx (matched against each joint's name suffix, same convention
    # as qmini.py's actuator groups -- e.g. "Revolute_left_pitch" matches
    # "pitch"). On real hardware, the calibrated startup pose never matches
    # default_joint_pos exactly (calibration imprecision, gear backlash,
    # wherever the robot happened to be sitting), and the policy had only
    # ever seen an exact reset-to-default state in sim, so small real-world
    # deviations were out-of-distribution and could provoke a large,
    # erratic corrective action (seen on hardware: multiple joints
    # commanded >10 deg in one step, tripping max_step_deg). Randomizing
    # the reset pose teaches the policy to correct back toward the
    # reference from a nearby-but-not-exact start, instead of only ever
    # knowing how to hold the one exact pose it always saw before.
    # Per-joint magnitudes from observed real calibration deviation: pitch
    # and knee run noticeably looser (~10-15 deg) than yaw/roll/ankle.
    startup_joint_pos_noise_deg: dict = {
        "yaw": 5.0,
        "roll": 5.0,
        "pitch": 15.0,
        "knee": 15.0,
        "ankle": 5.0,
    }

    # IMU realism: sim currently hands the policy a PERFECT
    # projected_gravity_b/root_ang_vel_b -- ground truth, zero noise, zero
    # bias, zero lag. The real IMU (imu_sensor.py) is nowhere near that
    # clean: even a static imu_calibration_check.py run shows measurable
    # accelerometer/gyro noise, and real gyros carry a roughly
    # session-constant bias (temperature, power-cycle, etc.). Applied only
    # in _get_observations -- NOT to _get_rewards' or _get_dones' use of
    # projected_gravity_b, which should keep using the true simulated
    # state (the agent's OBSERVATION should be imperfect, matching what it
    # would actually perceive; the reward/termination signal is a training
    # tool that has no reason to also be noisy).
    #
    # Per-step, i.i.d. each step (sensor noise proper):
    imu_gyro_noise_std_deg_s: float = 0.5
    # gravity_dir noise is applied as small per-component Gaussian noise
    # then renormalized -- an approximation of "angular noise of roughly
    # this many degrees", not an exact conversion, but standard practice
    # for small angles and much simpler than replicating imu_sensor.py's
    # complementary filter dynamics inside the sim step.
    imu_gravity_noise_std_deg: float = 1.0
    # Per-episode, sampled once per env at reset and held constant for the
    # whole episode (real gyro bias, not step noise) -- see _reset_idx.
    imu_gyro_bias_range_deg_s: float = 1.0

    # Per-episode constant ROTATIONAL bias applied to observed gravity_dir
    # -- distinct from imu_gravity_noise_std_deg's per-step noise, this
    # models a fixed few-degree IMU MOUNTING/CALIBRATION error (AXIS_REMAP
    # or the physical mount not being exactly what imu_calibration_check.py
    # measured) that persists for an entire deployment run rather than
    # averaging out step to step. Applied as a small random axis-angle
    # rotation, magnitude sampled in [0, this], see _apply_imu_mount_bias.
    imu_mount_bias_range_deg: float = 2.0

    # Per-episode mass SCALE randomization applied to base_link only.
    # Directly motivated by this project's own history: the real
    # battery/Pi-mount/decoration mass (qmini_urdf-2legs.usda's
    # centerOfMass/mass values) was initially completely unmodeled, and
    # even now those are hand-corrected estimates, not a precise
    # measurement -- there's no reason to assume they're exact. (0.9, 1.1)
    # = base mass scaled by a random factor in that range each episode.
    base_mass_randomization_range: tuple[float, float] = (0.9, 1.1)

    # Push disturbances: every push_interval_s (synchronized across envs,
    # but the random velocity per env still differs), a random horizontal
    # velocity kick is added directly to the base's linear velocity --
    # simulates a bump/nudge/uneven-footing event. Standard technique for
    # legged balance robustness: without this, the policy only ever has to
    # reject its own comparatively gentle tracking-error disturbances, not
    # an external shove. Set push_interval_s very large to effectively
    # disable.
    push_interval_s: float = 4.0
    push_velocity_range_mps: tuple[float, float] = (-0.4, 0.4)

    # Weight on the action-rate penalty in _get_rewards (-weight *
    # sum((action_t - action_{t-1})**2)). 0.0 = off (previous behavior).
    # Start small (e.g. 0.01-0.05) and increase if deployed targets are
    # still visibly jittery -- too high will fight the tracking_reward and
    # produce a sluggish policy that can't keep up with the gait.
    action_rate_penalty_weight: float = 0.02

    # --- balance (now that the base is free-floating, not welded to the
    # world -- see qmini_urdf-2legs.usda) --------------------------------
    # Weight on the "keep the body level" bonus in _get_rewards
    # (orientation_reward_weight * exp(-orientation_reward_scale *
    # |projected_gravity_b_xy|^2)). projected_gravity_b is (0,0,-1) when
    # base_link is exactly level, regardless of yaw heading, so its xy
    # components are a direct, facing-independent tilt measure.
    #
    # Raised from the original 1.0/20.0: tracking_reward+velocity_reward
    # alone are worth up to 3.0 and DON'T care about body orientation at
    # all (they only compare joint angles to the reference), so a robot
    # lying on the ground still wiggling its legs toward the keyframe
    # pattern collected almost as much reward as one standing -- with
    # orientation_reward_weight=1.0 that wasn't enough to outweigh it, and
    # scale=20 made the bonus collapse to ~0 past a ~13 deg tilt, giving
    # no useful gradient to correct a *small* lean before it became a
    # fall. weight=3.0 makes staying level worth as much as tracking;
    # scale=8.0 keeps a meaningful gradient out to a much larger tilt.
    orientation_reward_weight: float = 3.0
    orientation_reward_scale: float = 8.0

    # Terminate the episode once the base has tipped this far from
    # upright, measured as projected_gravity_b's z component (-1.0 =
    # perfectly upright, 0.0 = tipped 90 deg, +1.0 = upside down). -0.5
    # corresponds to roughly a 60 deg tilt -- past that the robot has
    # essentially fallen and continuing to simulate it lying on the ground
    # just wastes the rest of the episode. UNTUNED starting guess -- watch
    # early training rollouts and loosen/tighten as needed.
    fall_orientation_threshold: float = -0.5

    # One-time penalty applied on the step a fall is detected (see
    # fall_orientation_threshold), on top of losing all remaining reward
    # for that episode. Early in training, "lose future reward" alone is a
    # weak signal -- the agent hasn't discovered standing yet, so it can't
    # tell the difference between "this rollout ended" and "I did
    # something bad", especially with episodes already averaging ~20
    # steps (see the log you shared). An explicit penalty gives PPO's
    # value function a sharp, immediate signal to associate with the fall
    # itself rather than only with the reward it stopped collecting.
    #
    # Lowered from 5.0 -- a 30k-iteration run with 5.0 collapsed around
    # iteration ~19k (action noise std ~0.5 -> ~0.01, episode length -> 1,
    # fall_rate -> 1.0). Standing up from a free-floating spawn is hard
    # enough that early random exploration rarely succeeds, so a penalty
    # this large relative to tracking_reward (up to 2.0) + orientation_reward
    # (up to 3.0) can end up an almost-unavoidable constant early on, giving
    # PPO's advantage estimator little to differentiate between actions --
    # at which point shrinking action noise (to minimize surrogate-loss
    # variance on a landscape it can't improve) can look like the locally
    # "cheaper" move than continuing to explore. Paired with raising
    # entropy_coef in rsl_rl_qmini_leg_ppo_cfg.py; if collapse recurs, try
    # lowering this further (or to 0.0, relying only on lost future reward).
    termination_penalty_weight: float = 1.0

    # Penalize joints for sitting near their hard mechanical limits
    # (self.robot.data.joint_pos_limits, enforced by PhysX from the USD's
    # physics:lowerLimit/upperLimit -- same numbers as
    # robot_config_qmini.json's min_deg/max_deg). Added after training
    # converged (action noise std settling near 0.2, tracking errors
    # frozen bit-for-bit across iterations) onto locking every major joint
    # exactly at its limit and standing rigidly on the hard stops: a
    # legitimate local optimum given the OTHER reward terms, not
    # instability -- gravity_z was -0.9999 and fall_rate was 0.0, i.e.
    # *maximally* stable, because a joint jammed against a physical stop
    # needs no active balancing effort and has zero velocity (matching the
    # static reference's zero velocity, so velocity_reward maxes out too).
    # tracking_reward alone doesn't discourage this: exp(-5*error^2)
    # bottoms out near 0 once error is already large, so once a joint is
    # e.g. 40 degrees off, drifting to 50 degrees off costs nothing further
    # -- no gradient pulls it back. This term adds one that doesn't
    # saturate the same way, and is worth having regardless of that
    # exploit: repeatedly commanding a real joint into its hard stop is
    # bad for the gearbox.
    #
    # joint_limit_margin: normalized distance from a joint's center
    # ((pos-lower)/(upper-lower)*2-1, so 0=centered, +-1=at a limit) inside
    # which no penalty applies. 0.2 starts penalizing once a joint enters
    # the outer 20% of its range on either side.
    joint_limit_margin: float = 0.2
    joint_limit_penalty_weight: float = 1.0

    # Penalizes the RAW commanded joint target for exceeding a joint's
    # hard limits -- see _get_rewards' target_limit_penalty comment for
    # why this is a DIFFERENT, necessary signal from joint_limit_penalty
    # above (that one only sees the physically-clamped joint_pos, which in
    # sim is identical whether the raw target overshot the limit by 1deg
    # or 20deg -- real hardware's safety check has no such clamping, it
    # just refuses the command outright, so the policy needs a reason in
    # training to never produce it in the first place). Quadratic and
    # unbounded (unlike joint_limit_penalty, which saturates near the
    # physical position boundary) specifically because a real safety abort
    # doesn't care how far over the limit the target was, but a bigger
    # overshoot in training is a stronger signal something is systematically
    # wrong, not just marginal, so the gradient should scale with it.
    target_limit_penalty_weight: float = 2.0

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
        # Used by the joint-limit penalty in _get_rewards. Fail loudly and
        # early (not mid-training with a cryptic AttributeError) if this
        # Isaac Lab version names it differently -- run
        # `dir(self.robot.data)` and grep for "limit" to find the right one.
        assert hasattr(self.robot.data, "joint_pos_limits"), (
            "self.robot.data has no 'joint_pos_limits' attribute -- the "
            "joint-limit penalty in _get_rewards() needs updating for this "
            "Isaac Lab version's actual attribute name."
        )

        # Per-joint startup-pose randomization range, in radians -- see
        # cfg.startup_joint_pos_noise_deg and _reset_idx. Matched by name
        # suffix once here rather than every reset.
        self._startup_joint_pos_noise_range_rad = torch.zeros(self.num_joints, device=self.device)
        for i, name in enumerate(self.robot.joint_names[:self.num_joints]):
            for suffix, deg in self.cfg.startup_joint_pos_noise_deg.items():
                if name.endswith(suffix):
                    self._startup_joint_pos_noise_range_rad[i] = math.radians(deg)
                    break
            else:
                raise ValueError(
                    f"Robot joint '{name}' doesn't end with any of "
                    f"{list(self.cfg.startup_joint_pos_noise_deg)} -- add an "
                    f"entry to cfg.startup_joint_pos_noise_deg covering it."
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

        # --- IMU realism state ---------------------------------------------
        # Per-env, per-axis gyro bias -- constant for an episode, resampled
        # on reset (see _reset_idx). See cfg.imu_gyro_bias_range_deg_s.
        self._gyro_bias_rad = torch.zeros(self.num_envs, 3, device=self.device)
        # Per-env constant IMU mounting-bias rotation, as an axis-angle
        # vector (direction = rotation axis, magnitude = angle in
        # radians) -- see cfg.imu_mount_bias_range_deg and
        # _apply_imu_mount_bias.
        self._imu_mount_bias_axis_angle_rad = torch.zeros(self.num_envs, 3, device=self.device)

        # --- base mass randomization state ---------------------------------
        # *** UNVERIFIED in this session (no Isaac Sim access) -- root_physx_view
        # is a lower-level PhysX tensor API than the rest of this file's Articulation
        # calls (write_root_pose_to_sim etc.), more prone to differing between
        # Isaac Lab versions. This assertion is here so a mismatch fails loudly
        # at startup with a clear message, not deep into a training run. If it
        # fails, run `dir(self.robot.root_physx_view)` and grep for "mass". ***
        assert hasattr(self.robot, "root_physx_view") and hasattr(self.robot.root_physx_view, "get_masses"), (
            "self.robot.root_physx_view has no 'get_masses' -- base mass "
            "randomization in _randomize_base_mass() needs updating for this "
            "Isaac Lab version's actual API."
        )
        assert "base_link" in self.robot.body_names, (
            f"'base_link' not found in self.robot.body_names={self.robot.body_names} "
            f"-- base mass randomization needs the correct body name for this USD."
        )
        self._base_body_idx = self.robot.body_names.index("base_link")
        # (num_envs, num_bodies) -- captured once, same reasoning as
        # _default_actuator_gains below: randomization always scales from
        # this fixed baseline instead of compounding across resets.
        self._default_base_mass = self.robot.root_physx_view.get_masses()[:, self._base_body_idx].clone()

        # --- push disturbance timing ----------------------------------------
        # See cfg.push_interval_s/push_velocity_range_mps and _pre_physics_step.
        self._push_interval_steps = max(1, round(self.cfg.push_interval_s / self.step_dt))

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

        # --- push disturbances --------------------------------------------
        # See cfg.push_interval_s/push_velocity_range_mps. Synchronized
        # across envs (all get pushed the same step), but each env's
        # velocity kick is independently randomized.
        if self.common_step_counter > 0 and self.common_step_counter % self._push_interval_steps == 0:
            self._apply_random_push()

    def _apply_random_push(self):
        push_vel = sample_uniform(
            *self.cfg.push_velocity_range_mps, (self.num_envs, 3), device=self.device,
        )
        lin_vel = self.robot.data.root_lin_vel_w.clone() + push_vel
        ang_vel = self.robot.data.root_ang_vel_w.clone()
        self.robot.write_root_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=-1))

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
        # IMU-equivalent terms, in base_link's own local frame:
        #  - projected_gravity_b: unit vector, the direction gravity points
        #    in the body frame. (0,0,-1) when level. What a normalized
        #    accelerometer reading approximates at rest (up to sign -- see
        #    robot_deploy.py's imu_sensor.py, which negates the raw
        #    accelerometer reading to match this convention).
        #  - root_ang_vel_b: base angular velocity in the body frame,
        #    rad/s. Exactly what a gyroscope measures, no approximation.
        imu_gravity = self.robot.data.projected_gravity_b
        imu_ang_vel = self.robot.data.root_ang_vel_b

        # Sensor-realism noise -- see cfg.imu_gyro_noise_std_deg_s /
        # imu_gravity_noise_std_deg / imu_gyro_bias_range_deg_s. Deliberately
        # NOT applied to self.robot.data.projected_gravity_b as read
        # directly in _get_rewards/_get_dones -- those represent the true
        # physical state (what actually happened), only the agent's
        # OBSERVATION of it should be imperfect.
        gyro_noise = torch.randn_like(imu_ang_vel) * math.radians(self.cfg.imu_gyro_noise_std_deg_s)
        imu_ang_vel = imu_ang_vel + self._gyro_bias_rad + gyro_noise

        imu_gravity = self._apply_imu_mount_bias(imu_gravity)
        gravity_noise = torch.randn_like(imu_gravity) * math.radians(self.cfg.imu_gravity_noise_std_deg)
        imu_gravity = imu_gravity + gravity_noise
        imu_gravity = imu_gravity / imu_gravity.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        observations = torch.cat(
            (
                self.robot.data.joint_pos[:, :self.num_joints],
                self.robot.data.joint_vel[:, :self.num_joints],
                imu_gravity,
                imu_ang_vel,
                # reference,
                self.motion_time.unsqueeze(1)
            ),
            dim=-1
        )

        return {
            "policy": observations
        }

    def _apply_imu_mount_bias(self, gravity_dir: torch.Tensor) -> torch.Tensor:
        """Rotates `gravity_dir` (num_envs, 3) by each env's fixed
        per-episode IMU mounting-bias rotation (self._imu_mount_bias_axis_angle_rad,
        see cfg.imu_mount_bias_range_deg), via batched Rodrigues' rotation
        formula. Same formula (and the same care about sign/derivation) as
        imu_sensor.py's _rotate_vector_by_gyro on the real-robot side, just
        vectorized across envs here instead of applied to a single reading."""
        aa = self._imu_mount_bias_axis_angle_rad  # (N, 3)
        angle = aa.norm(dim=-1, keepdim=True).clamp_min(1e-9)  # (N, 1)
        axis = aa / angle
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        cross = torch.cross(axis, gravity_dir, dim=-1)
        dot = (axis * gravity_dir).sum(dim=-1, keepdim=True)
        return gravity_dir * cos_a + cross * sin_a + axis * dot * (1 - cos_a)

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
        # For _get_rewards' target_limit_penalty -- see that comment for
        # why this needs to be the RAW target (pre-physics-clamping), not
        # the resulting joint_pos.
        self._last_position_targets = position_targets

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

        # Penalize joints sitting near their hard mechanical limits -- see
        # cfg.joint_limit_margin/joint_limit_penalty_weight's comment for
        # why this exists (a real exploit this training run found: locking
        # every major joint against its limit is a "free" way to maximize
        # orientation_reward + velocity_reward once tracking_reward has
        # already saturated near 0, since exp(-5*error^2) gives no more
        # gradient once error is already large).
        limits = self.robot.data.joint_pos_limits[:, :self.num_joints, :]
        lower, upper = limits[..., 0], limits[..., 1]
        joint_range = upper - lower
        normalized_pos = 2.0 * (self.robot.data.joint_pos[:, :self.num_joints] - lower) / joint_range - 1.0
        joint_limit_penalty = self.cfg.joint_limit_penalty_weight * torch.sum(
            torch.clamp(normalized_pos.abs() - self.cfg.joint_limit_margin, min=0.0) ** 2,
            dim=1,
        )

        # Penalize the RAW commanded target for exceeding a joint's hard
        # limits -- distinct from joint_limit_penalty above, which only
        # looks at the resulting (physically clamped) joint_pos. PhysX
        # enforces these limits as a hard stop, so in sim a target of
        # -18deg and a target of -15deg on a joint limited to [-15,15]
        # produce the EXACT SAME joint_pos and the EXACT SAME
        # joint_limit_penalty -- nothing here previously told the policy
        # those two actions were any different, so it had zero incentive
        # to keep its raw outputs inside the physical envelope. Real
        # hardware's safety check (robot_deploy.py's check_joint_limits)
        # validates the commanded TARGET itself and refuses to send
        # anything out of range at all -- a real robot doesn't get PhysX's
        # "clamps for free" behavior, it just aborts. This is what a
        # deployed policy actually tripped: a -18.18deg target on a
        # right_hip_roll limited to [-15,15].
        target_over_limit = (
            torch.clamp(self._last_position_targets - upper, min=0.0)
            + torch.clamp(lower - self._last_position_targets, min=0.0)
        )
        target_limit_penalty = self.cfg.target_limit_penalty_weight * torch.sum(
            target_over_limit ** 2, dim=1,
        )

        # Keep the body level: projected_gravity_b's xy components vanish
        # exactly when base_link is upright, independent of yaw heading
        # (see _get_observations). This is now load-bearing, not cosmetic
        # -- the base is free-floating (qmini_urdf-2legs.usda's root fixed
        # joint was removed), so nothing else in the reward stops the
        # policy from just letting the robot tip over while it chases
        # tracking_reward with its legs.
        projected_gravity = self.robot.data.projected_gravity_b
        orientation_error = torch.sum(projected_gravity[:, :2] ** 2, dim=1)
        orientation_reward = torch.exp(-self.cfg.orientation_reward_scale * orientation_error)

        # Same fall condition as _get_dones() -- recomputed independently
        # here rather than reading self.reset_terminated, since this repo
        # hasn't verified whether Isaac Lab's DirectRLEnv.step() calls
        # _get_dones() before or after _get_rewards() (order differs
        # across versions/task templates), and projected_gravity_b is
        # cheap enough that recomputing it is simpler than depending on
        # that ordering being one particular way.
        fell = projected_gravity[:, 2] > self.cfg.fall_orientation_threshold
        termination_penalty = self.cfg.termination_penalty_weight * fell.float()

        reward = (
            2.0*tracking_reward
            + velocity_reward
            + self.cfg.orientation_reward_weight * orientation_reward
            - action_rate_penalty
            - termination_penalty
            - joint_limit_penalty
            - target_limit_penalty
        )

        # ---------------------------------
        # Trajectory logging (env 0 only)
        # ---------------------------------
        joint_labels = [n.removeprefix("Revolute_") for n in self.robot.joint_names[:self.num_joints]]

        log = {
            "tracking/reward": reward.mean(),
            "tracking/action_rate_penalty": action_rate_penalty.mean(),
            "tracking/joint_limit_penalty": joint_limit_penalty.mean(),
            "tracking/target_limit_penalty": target_limit_penalty.mean(),
            "orientation/reward": orientation_reward.mean(),
            "orientation/termination_penalty": termination_penalty.mean(),
            "orientation/fall_rate": fell.float().mean(),
            "orientation/gravity_x": projected_gravity[0, 0],
            "orientation/gravity_y": projected_gravity[0, 1],
            "orientation/gravity_z": projected_gravity[0, 2],
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
        # Now that the base is free-floating (see qmini_urdf-2legs.usda),
        # a policy that tips over can otherwise spend the rest of the
        # episode lying on the ground doing nothing useful -- terminate
        # early instead. See cfg.fall_orientation_threshold.
        projected_gravity = self.robot.data.projected_gravity_b
        terminated = projected_gravity[:, 2] > self.cfg.fall_orientation_threshold

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Actually write the robot back to its default root pose/velocity
        # and default joint state -- super()._reset_idx() (DirectRLEnv's
        # base implementation) only resets bookkeeping (episode_length_buf,
        # logging), NOT physics state; that's each task's own
        # responsibility, and this env never did it. Harmless while the
        # base was welded to the world (root pose was irrelevant, see
        # qmini_urdf-2legs.usda's now-removed root_joint) and reset joint
        # drift was comparatively minor -- but with a free-floating base, a
        # robot that fell would otherwise NEVER physically leave that
        # state: _get_dones() would keep seeing it as fallen and
        # re-terminate it every single following step, forever, regardless
        # of fall_orientation_threshold or episode_length_s actually being
        # correct. Root position needs the per-env origin offset added
        # (env_origins) since scene.clone_environments spreads envs out in
        # world space -- default_root_state stores env-LOCAL positions.
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        default_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        # Small per-joint random offset, independent per env and per joint
        # -- see cfg.startup_joint_pos_noise_deg's comment for why (real
        # calibrated startup poses never land exactly on default_joint_pos,
        # and an exact-reset-only policy treated that mismatch as
        # out-of-distribution on hardware). Sampled in [-1, 1] and scaled
        # by the precomputed per-joint range (self._startup_joint_pos_noise_range_rad,
        # shape (num_joints,)) rather than passing per-joint bounds
        # straight to sample_uniform, since that helper's low/high are
        # documented for the same scalar-range-per-call usage the gain
        # randomization above uses, not per-element bounds.
        unit_noise = sample_uniform(-1.0, 1.0, default_joint_pos.shape, device=self.device)
        joint_pos_noise = unit_noise * self._startup_joint_pos_noise_range_rad
        randomized_joint_pos = default_joint_pos + joint_pos_noise
        self.robot.write_joint_state_to_sim(randomized_joint_pos, default_joint_vel, env_ids=env_ids)

        # self.phase_modulator.reset(
        #     env_ids=env_ids,
        #     deterministic=self.render_mode is not None,
        # )

        # Reset to phase 0 (reference = keyframe-0 = default_joint_pos, see
        # qmini.py's ArticulationCfg.InitialStateCfg). The actual joint_pos
        # written above is now default_joint_pos PLUS the small random
        # offset above -- so unlike the original design (where this
        # comment described an exact match, specifically to avoid a
        # bad first-action jump), there IS deliberately a small gap
        # between joint_pos and the phase-0 reference at reset now. That's
        # intentional: it's what teaches the policy to correct back toward
        # the reference from a nearby-but-not-exact start, rather than only
        # ever knowing how to hold one exact memorized pose. Don't
        # reintroduce full PHASE randomization here though (motion_time
        # starting somewhere other than 0) -- that reintroduces the
        # original bad-jump problem this comment used to warn about, since
        # the joint offset above is small/local while a random phase could
        # put the reference anywhere in the gait, arbitrarily far from
        # wherever joint_pos actually is.
        self.motion_time[env_ids] = 0.0

        # Avoid penalizing the first post-reset action against whatever
        # action happened to be in _prev_actions from a different episode
        # (or, for envs reset at __init__ time, from the zero-init default,
        # which is fine and intentional -- zero is a neutral "no offset
        # from default_joint_pos" action).
        self._prev_actions[env_ids] = 0.0

        self._randomize_action_delay(env_ids)
        self._randomize_actuator_gains(env_ids)
        self._randomize_gyro_bias(env_ids)
        self._randomize_imu_mount_bias(env_ids)
        self._randomize_base_mass(env_ids)

    def _randomize_gyro_bias(self, env_ids: Sequence[int]):
        bias_range_rad = math.radians(self.cfg.imu_gyro_bias_range_deg_s)
        self._gyro_bias_rad[env_ids] = sample_uniform(
            -bias_range_rad, bias_range_rad, (len(env_ids), 3), device=self.device,
        )

    def _randomize_imu_mount_bias(self, env_ids: Sequence[int]):
        n = len(env_ids)
        range_rad = math.radians(self.cfg.imu_mount_bias_range_deg)
        random_axis = sample_uniform(-1.0, 1.0, (n, 3), device=self.device)
        random_axis = random_axis / random_axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        random_angle = sample_uniform(0.0, range_rad, (n, 1), device=self.device)
        self._imu_mount_bias_axis_angle_rad[env_ids] = random_axis * random_angle

    def _randomize_base_mass(self, env_ids: Sequence[int]):
        low, high = self.cfg.base_mass_randomization_range
        n = len(env_ids)
        mass_device = self._default_base_mass.device  # root_physx_view tensors may not be on self.device
        scale = sample_uniform(low, high, (n,), device=mass_device)
        env_ids_mass = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(list(env_ids), device=mass_device)
        masses = self.robot.root_physx_view.get_masses()
        masses[env_ids_mass.to(masses.device), self._base_body_idx] = (
            self._default_base_mass[env_ids_mass.to(mass_device)] * scale
        )
        self.robot.root_physx_view.set_masses(masses, env_ids_mass.to(masses.device))

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


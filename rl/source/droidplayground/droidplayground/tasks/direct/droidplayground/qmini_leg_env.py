from __future__ import annotations

import json
import math
import torch
from collections.abc import Sequence
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, sample_uniform, wrap_to_pi
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
KEYFRAMES_PATH = Path(__file__).parent / "keyframes_step_in_place_all_joints_2x_base_offset.json"


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


def _load_reference_foot_heights(json_path: Path):
    """Load the per-frame reference foot-height curve, same `times` as the
    joint keyframes. See `reference_foot_heights_m`'s own comment in the
    keyframes JSON (written by gen_reference_heights.py) for how these 4
    columns (left_heel, left_toe, right_heel, right_toe, meters, each
    foot's own height above ITS OWN cycle minimum) were computed. Raises
    if the field is missing, rather than silently falling back to
    something -- see foot_height_target_floor_m's comment in this file for
    why this replaced the old constant target and must not go quietly
    missing.
    """
    with open(json_path) as f:
        data = json.load(f)
    if "reference_foot_heights_m" not in data:
        raise KeyError(
            f"{json_path} has no 'reference_foot_heights_m' field -- run "
            f"gen_reference_heights.py to generate it (see "
            f"foot_height_target_floor_m's comment in qmini_leg_env.py)."
        )
    heights = data["reference_foot_heights_m"]
    times = [t for t, _ in data["keyframes"]]
    if len(times) != len(heights):
        raise ValueError(
            f"{json_path}: 'keyframes' has {len(times)} frames but "
            f"'reference_foot_heights_m' has {len(heights)} -- re-run "
            f"gen_reference_heights.py, they must stay in sync."
        )
    return list(zip(times, heights))


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
    # unit vector) + 3 root_ang_vel_b (IMU gyro-like, rad/s) + 1
    # motion_time. See _get_observations -- this layout must match
    # robot_deploy.py's build_obs() exactly (joint pos/vel, then the 6 IMU
    # terms, then motion_time), since that's what a real IMU reading gets
    # slotted into.
    #
    # RE-ADOPTED sin(phase)/cos(phase) (observation_space=28) on 2026-09-08,
    # back OUT of the raw motion_time scalar (observation_space=27) that had
    # been in place since the 2026-09-07 revert. Full story, for whoever
    # reads this next:
    #
    # sin/cos was first tried 2026-09-04 and confirmed (via the motion_time
    # sweep methodology) to fully close a real, video-confirmed real-
    # hardware phase-discontinuity landmine -- necessary for deployment, not
    # cosmetic. But no training run under it ever produced genuine,
    # video-confirmed stepping, so on 2026-09-07 it was reverted back to the
    # raw scalar as a controlled A/B test, isolating phase encoding as the
    # one remaining untested variable (see git history for that revert's
    # full comment). That test came back NEGATIVE -- raw scalar alone did
    # NOT bring stepping back either, ruling out phase encoding as the
    # blocker. Continued investigation (reconstructing the last confirmed-
    # good config from this file's own dated comment history) found the
    # real cause: tracking_linear_penalty_weight had been raised 0.2 -> 1.0
    # on 2026-09-06 (see that field's comment), an always-on, unbounded
    # per-step penalty that was suppressing exactly the bigger joint
    # excursions genuine stepping requires. Reverting THAT back to 0.2
    # (2026-09-07/08, run 2026-09-07_22-00-51), together with a cluster of
    # other stay-still pressure already loosened that same night
    # (push_interval_range_s, orientation_margin_deg, orientation_reward_scale,
    # position_reward_weight/heading_reward_weight), finally produced real,
    # sustained, video-confirmed stepping -- still under the raw scalar,
    # confirming phase encoding was never the issue.
    #
    # So: now that the actual blocker is fixed, re-adopting sin/cos to get
    # back the real-hardware fix it provides, on top of a reward config
    # that's now proven to support genuine stepping. If stepping survives
    # this switch, sin/cos and genuine stepping are compatible and further
    # polishing (gait aggressiveness/fall rate, see
    # tracking_linear_penalty_weight and heading_reward_weight's comments)
    # can continue on top of it. If stepping does NOT survive this specific
    # switch, that's real, well-isolated evidence sin/cos itself is
    # incompatible with this reward config specifically (not with stepping
    # in general, which is now well established) and the next step is a
    # richer, still-continuous phase representation rather than either
    # extreme. Requires a fresh training run, not a resume -- the input
    # layer's shape changes either direction.
    observation_space = 28
    state_space = 0
    action_scale = 0.5

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # Viewport steps:
    # Launch non-headless (drop --headless from train.py/play.py).
    # Once the sim is running, orbit/pan/zoom the viewport with the mouse until the shot looks right.
    # Open Window → Script Editor in Isaac Sim, and run:
    # from omni.kit.viewport.utility.camera_state import ViewportCameraState
    # cam = ViewportCameraState()
    # print("eye:", tuple(cam.position_world))
    # print("lookat:", tuple(cam.target_world))
    # Copy those two tuples straight into ViewerCfg(eye=..., lookat=...) in qmini_leg_env.py:97.

    # viewport/video camera (used for the non-headless viewer and --video recording)
    viewer: ViewerCfg = ViewerCfg(eye=(1.5, 3.5, 1.0), lookat=(-0.3, -1.5, -0.5))

    # robot(s)
    robot_cfg: ArticulationCfg = QMINI_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=1.0, replicate_physics=True)

    # --- sim-to-real robustness ---------------------------------------
    # Extra zero-order-hold action delay, in control steps (each step =
    # decimation * sim.dt = 20ms at the defaults above), sampled per-env
    # PER-JOINT per-episode from each joint-type's own (min, max) range.
    # Was a single robot-wide range until 2026-08-29 -- see
    # analyze_delay.py, run against a real --open-loop-ref log
    # (control_loop_20260829_144353.csv): real per-joint delay is NOT
    # uniform. Measured (high confidence, correlation 0.87-0.994):
    # hip_roll 200ms/10 steps (before a kp fix, see below), knee 80ms/4
    # steps, hip_pitch 40-60ms/2-3 steps, ankle 40-60ms/2-3 steps. hip_yaw
    # showed 0 correlation -- the reference barely moves yaw, not a real
    # "zero delay" measurement, left at a guessed range.
    #
    # hip_roll's 200ms was chased down to a real cause, not just modeled
    # around: robot_config_qmini_stepinplace.json's hip_roll has an
    # extra_gear_ratio of 3.0 on top of the base reduction, and kp_rotor =
    # kp_output / gear_ratio^2 -- squaring that extra 3x into the
    # denominator left hip_roll's rotor-side stiffness (~0.29) far below
    # every other joint's (~0.75-1.87) even though kd_rotor came out
    # matched (~0.05) everywhere. Lower kp with matched kd is a lower-
    # bandwidth, slower-settling response, not oscillatory on its own --
    # consistent with a clean measured LAG rather than ringing. Raising
    # hip_roll's output-side kp 105 -> 360 (re-measured, same
    # --open-loop-ref method, control_loop_20260829_150136.csv) brought it
    # down to 80ms/4 steps, in line with knee rather than a 2.5-5x outlier.
    # kp_rotor is now ~1.0, still below pitch's ~1.87, so there may be
    # room to push it further, but 360 already roughly doubles the
    # deployment's own historical kp~=0.5 baseline (vs the qmini source
    # default kp=105/kp_rotor~=0.29), so leaving it here for now rather
    # than continuing to chase it blind.
    # NARROWED from the first attempt at these per-joint ranges (yaw/pitch/
    # ankle (1,4), roll/knee (2,5)) on 2026-08-29: resuming with that full
    # jump straight to the measured values -- combined with the hip_roll
    # stiffness change above -- produced the roughest transient of the
    # whole project (see DEFAULT_GAINS["hip_roll"]'s comment in qmini.py
    # for the full isolation-test story: the delay widening alone, even
    # WITHOUT the stiffness change, was still enough to collapse
    # foot_swing_reward to ~0 with video-confirmed no lifting). Stepping
    # down to a smaller jump from the old uniform (0,3) here rather than
    # reverting the delay change entirely -- real per-joint delay is still
    # a genuine measured fact worth training against, just apparently too
    # big a step to absorb from an already-converged policy in one go.
    # Revisit widening further, gradually, once this trains cleanly.
    #
    # WIDENED again on 2026-09-03, this time as the direct fix for a
    # separate, better-understood problem: real robot_deploy.py logs from
    # 2026-09-01/02 showed a reproducible runaway at the reference's two
    # fastest transitions (mt~1.0-1.3 and mt~2.1-2.2), and probing the
    # exported policy directly (sweeping motion_time alone, all other obs
    # held fixed) proved this ISN'T sensor noise or real dynamics -- it's
    # the network having a sharp discontinuity in its own phase response,
    # because sim always keeps motion_time and true joint state in tight
    # lockstep while real actuator lag lets them desync exactly at those
    # two transitions. Two attempts at modeling that desync as observation
    # noise (motion_time_noise_std_s, now disabled -- see its comment) did
    # not narrow it. Widening the delay itself is more direct: more lag
    # means the sim joint state genuinely lags the phase during a fast
    # transition, the same mismatch real hardware produces, rather than an
    # artificial jitter on top of a still-perfectly-synced state. Stepped
    # roll/pitch/knee/ankle (the joints that actually spiked in the sweep)
    # up by one step each on the HIGH end only, leaving the low end
    # unchanged this time -- the 2026-08-29 jump that caused the worst
    # transient of the project raised the low end too (roll/knee 1->2),
    # forcing every episode to a higher minimum delay; this keeps the same
    # floor and just samples longer tails more often. yaw left alone --
    # it never showed up in either sweep. Revisit widening further if this
    # trains cleanly and the sweep still shows either zone.
    action_delay_range_steps: dict = {
        "yaw": (1, 3),
        "roll": (1, 5),
        "pitch": (1, 4),
        "knee": (1, 5),
        "ankle": (1, 4),
    }

    # EMA low-pass filter on the FINAL decoded position target (after
    # default_joint_pos + action_scale * action, see _apply_action):
    # smoothed_t = action_smoothing * target_t + (1 - action_smoothing) *
    # smoothed_{t-1}. 1.0 = off (raw target, previous behavior).
    #
    # This mechanism already existed on the deploy side only
    # (robot_deploy.py's --action-smoothing flag), explicitly documented
    # there as a diagnostic tool for isolating jitter as a cause of motor
    # faults, NOT a permanent fix -- "the real fix is an action-rate
    # penalty during training". That assumption turned out wrong: THREE
    # separate attempts at fixing a persistent ankle shake/twitch via
    # training (narrower gain_randomization_range, higher
    # DEFAULT_GAINS["ankle"]["damping"], a per-joint action_rate_penalty
    # weight on ankle -- see each cfg's comment) all failed the same way,
    # because none of them could distinguish "reacting to noise" from
    # "reacting to the genuine swing the gait needs" -- both got suppressed
    # together, or neither did. A low-pass filter is fundamentally
    # different: it's frequency-selective rather than amplitude-
    # suppressing. Video showed the shake as a genuine ~2-4Hz oscillation;
    # the actual gait cycle is ~0.45Hz
    # (keyframes_step_in_place_all_joints_2x_base_offset.json's ~2.2s
    # period). A filter cutoff well below the shake and well above the
    # gait cycle can attenuate one while leaving the other essentially
    # untouched -- exactly what none of the reward/gain-based attempts
    # could do. Matches BD-X's own low-level control design (Sec VI-D of
    # that paper): "we perform a first-order-hold... followed by a low-
    # pass filter... these low-level control aspects are identically
    # implemented in the RL and runtime environments" -- i.e. this must be
    # graduated from a deploy-only diagnostic to a permanent part of BOTH
    # qmini_leg_env.py and robot_deploy.py, at the SAME value, or the
    # policy trains against dynamics it won't actually see on hardware.
    #
    # RESULT of alpha=0.5 (2026-08-26, run 2026-08-26_19-40-00, resumed
    # from the same model_367800.pt checkpoint as all three earlier
    # attempts, for a clean isolated test): negative, same failure shape
    # AGAIN and faster than the previous attempts -- foot_swing_reward
    # already down to 0.37 within 1.3hr/~4800 iterations (the action_rate_
    # joint_weight attempt took ~5hr to reach a similar level), video
    # showing the same static-crouch look instead of confident alternating
    # stepping, right ankle STILL showing the shake. Reverted to 1.0 (off).
    #
    # The frequency-separation argument behind this had a real gap: it
    # compared the shake (~2-4Hz) against the FULL gait cycle (~0.45Hz),
    # but the actual swing motion only occupies a fraction of that cycle,
    # not the whole thing -- so its own frequency content is much higher
    # than 0.45Hz, likely overlapping the shake band rather than sitting
    # cleanly below it. A filter tuned to sit between the wrong two numbers
    # doesn't separate noise from signal, it just cuts into the signal --
    # explaining why this failed the same way as the other three attempts
    # (narrower gain_randomization_range, higher ankle damping, ankle-
    # weighted action_rate_penalty) despite being a structurally different
    # mechanism. FOUR attempts across four different levers have now all
    # hit the identical trade: suppress the shake a bit, lose genuine
    # lifting. Don't try a fifth blanket-suppression variant (lower alpha,
    # anything similar) without a fundamentally different diagnosis first.
    #
    # NOT the explanation: the reference keyframe's own timing. The "2x" in
    # keyframes_step_in_place_all_joints_2x_base_offset.json means 2x
    # SLOWER than the baseline clip, not faster/exaggerated -- confirmed
    # directly, this file's own earlier guess at that had it backwards. A
    # slower reference means LESS required ankle velocity, more slack, not
    # less -- so tight swing timing doesn't explain why every suppression
    # attempt has cut into genuine lift. What actually does explain it is
    # still open. The shake itself doesn't affect fall_rate or tracking
    # quality -- it's cosmetic, not a stability problem -- so deprioritizing
    # it rather than continuing to trade real regressions for it is
    # reasonable until there's a better-grounded next hypothesis.
    action_smoothing: float = 1.0

    # Per-episode multiplicative randomization applied to each actuator
    # group's stiffness/damping/armature, as a fraction of qmini.py's
    # DEFAULT_GAINS (or whatever build_qmini_cfg() gains this env was built
    # with). (0.7, 1.3) means each episode samples a value in
    # [0.7x, 1.3x] of the base gain. See _randomize_actuator_gains --
    # actuator groups are per JOINT TYPE (e.g. "ankle" matches both
    # left_ankle and right_ankle via regex), but the scale is sampled with
    # shape (num_envs, num_joints_in_group), i.e. INDEPENDENTLY per joint,
    # not once per group. So left_ankle and right_ankle already draw
    # separate random gains every reset -- a single joint (e.g. only the
    # right ankle) drawing an unlucky low-stiffness/low-damping combination
    # while its counterpart draws normally is expected, not a bug.
    #
    # Briefly narrowed stiffness/damping from (0.7,1.3) to (0.85,1.15) on
    # 2026-08-26 as a diagnostic test for a right-ankle-specific
    # shake/twitch after lifting seen in video (present on some envs and
    # not others, persisting for a whole episode once present). RESULT:
    # negative -- a training run at the narrower range (up to iteration
    # 370825) showed the exact same pattern, ruling out unlucky per-episode
    # gain draws as the cause. Reverted back to (0.7, 1.3) here since the
    # narrower range bought nothing but did cost real coverage of the
    # actuator's real gain variation. The actual fix being tested instead:
    # qmini.py's DEFAULT_GAINS["ankle"]["damping"], raised 2.0 -> 4.0 --
    # ankle is the lowest-stiffness joint and (per that file's own
    # troubleshooting comment) an oscillating joint calls for more damping,
    # not more stiffness. See that file's comment for the full reasoning.
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

    # Joint sensor realism: like the IMU block above, sim currently hands
    # the policy PERFECT self.robot.data.joint_pos/joint_vel every step --
    # zero encoder noise, zero velocity-estimation noise. Added after
    # analyzing real robot_deploy.py logs from 2026-09-01: a real deployed
    # policy produced a reproducible runaway (one joint's commanded target
    # exploding over 3-4 consecutive steps -- e.g. left_hip_roll going
    # -0.6 deg -> -19.2 deg in 60ms) in a window where the IMU, the actual
    # encoder position, AND the reference keyframe were all independently
    # confirmed calm (keyframe was gliding through target at ~5 deg/s, no
    # fast transition) -- and a same-day --open-loop-ref capture through
    # the identical motion_time window (control_loop_20260829_144353.csv,
    # ~2.5 full gait cycles) showed zero anomalies, ruling out hardware/
    # telemetry. That leaves the trained policy's own sensitivity to
    # small, real per-step sensor jitter it never saw in training (only
    # startup_joint_pos_noise_deg above -- a one-time RESET perturbation --
    # existed; nothing perturbs the observation step to step) as the most
    # likely cause. Magnitudes below are measured, not guessed: computed
    # as the residual of each real joint_pos/joint_vel reading against a
    # local linear fit of its two neighbors (isolates high-frequency noise
    # from genuine smooth motion) over that same open-loop-ref capture,
    # ~0.01-0.04 deg position / ~1-3 deg/s velocity across the three
    # joints checked (hip_roll, hip_pitch, ankle) -- values below add
    # margin on top of that measured floor rather than using it exactly,
    # since only 3 of 10 joints were sampled and real noise likely varies
    # somewhat by joint. Flat (not per-joint-suffix) for now, matching how
    # imu_gyro_noise_std_deg_s/imu_gravity_noise_std_deg above are also
    # flat despite covering physically different sensors -- revisit with
    # per-joint values if training shows some joints need more/less.
    joint_pos_noise_std_deg: float = 0.15
    joint_vel_noise_std_deg_s: float = 5.0

    # motion_time phase-input realism: added after the joint/IMU noise above
    # still didn't stop a reproducible real-hardware runaway on 2026-09-02
    # (multiple attempts, always right around the reference's two fast leg-
    # swing transitions). Direct probing of the exported policy.pt found the
    # cause: sweeping ONLY motion_time through the exported policy, with
    # every joint_pos/joint_vel/IMU reading held FIXED at a real calm
    # reading, reproduces the real runaway almost exactly on its own --
    # e.g. left_hip_roll's action went -0.09 -> -0.34 -> -0.64 for
    # motion_time 1.00 -> 1.02 -> 1.04 (a 20ms step) with the physical state
    # completely unchanged. So this isn't sensor noise or real dynamics at
    # all (a 2000-sample Monte Carlo test with realistic joint_pos/vel noise
    # only moved that same action by <=0.015) -- it's the trained network
    # itself having a sharp, narrow discontinuity in its response to its own
    # phase input, at exactly the two points in the 2.2s cycle where the
    # keyframe reference moves fastest. Root cause: sim always keeps
    # motion_time and true joint state in tight lockstep, so the network
    # never had to handle "phase says transition, joints still read
    # pre-transition" -- which is exactly what real actuator lag produces at
    # those two transitions. Modeling this as per-step Gaussian jitter on
    # the OBSERVED motion_time (not the true one used for reward/reference
    # sampling/action-delay indexing) directly forces the network to learn a
    # smoother, more robust function of phase instead of the brittle sharp
    # one found above. std chosen as ~2-3x the control step (0.02s) -- wide
    # enough to repeatedly sample across the narrow (~20-40ms) instability
    # band found above, small relative to the keyframe's own ~33ms average
    # spacing so overall phase tracking still stays coherent.
    #
    # DISABLED (0.05 -> 0.0) on 2026-09-03 after two resumed runs (~63k
    # combined iterations) failed to move the actual defect: re-running the
    # exact same motion_time sweep against the final checkpoint of the
    # second run found BOTH landmine zones still present at comparable or
    # greater severity, and the affected mt range had gotten WIDER (1.04-
    # 1.16 -> 0.98-1.30 for the first zone) rather than narrower. Meanwhile
    # the second run (with foot_swing_reward_weight also raised to 8.0)
    # never fully recovered -- noise_std plateaued around 0.44 instead of
    # the usual ~0.18-0.20, heading/yaw_rate_dps regressed to ~100 deg/s
    # from the ~35-45 deg/s this project had already reached, and
    # orientation/reward settled at ~0.77 instead of ~0.90+. Adding noise
    # around a sharp discontinuity apparently smeared the bad region wider
    # rather than flattening it, and cost real quality elsewhere in the
    # process. Root-causing (see action_delay_range_steps' comment) points
    # at real actuator lag creating a phase/state mismatch sim doesn't
    # model closely enough -- widening the delay range directly, rather
    # than jittering the phase observation, is the next attempt at this.
    motion_time_noise_std_s: float = 0.0

    # Per-episode constant ROTATIONAL bias applied to observed gravity_dir
    # -- distinct from imu_gravity_noise_std_deg's per-step noise, this
    # models a fixed few-degree IMU MOUNTING/CALIBRATION error (AXIS_REMAP
    # or the physical mount not being exactly what imu_calibration_check.py
    # measured) that persists for an entire deployment run rather than
    # averaging out step to step. Applied as a small random axis-angle
    # rotation, magnitude sampled in [0, this], see _apply_imu_mount_bias.
    imu_mount_bias_range_deg: float = 2.0

    # Per-episode joint ZERO-CALIBRATION error, per joint, uniform in
    # [-range, +range] (deg, matched by joint-name suffix), held constant
    # for the whole episode. Added 2026-10-05. Models robot_deploy.py's
    # power-on calibration landing slightly off the true zero:
    #   true angle = believed angle + offset
    # so the policy OBSERVES joint_pos - offset, the applied PD target is
    # default + action_scale*action + offset, the episode starts at
    # default + offset (the believed default pose), and target_limit_penalty
    # is measured on the believed target (what robot_deploy.py validates).
    # Distinct from startup_joint_pos_noise_deg, which only perturbs the
    # initial joint STATE and is then corrected away by the PD loop.
    #
    # Why: margin13 (2026-10-04_13-32-56/model_159996) on hardware leaned
    # back and hung in the rope with the rope loose until the ankles were
    # hand-calibrated 3 deg forward (robot_config_qmini_stepinplace_ankle_fwd3.json);
    # with that it stepped 7 minutes rope-free. check_calibration_offset.py
    # showed the policy is very sensitive here (ankle +4 deg -> 42% falls,
    # ankle +2 -> mean pitch +3.6 deg back), so it should learn to read
    # the resulting lean from the IMU and compensate, instead of relying
    # on a hand-tuned deploy offset. Ankle widest since that is where the
    # real error showed up; roll/yaw small (roll target limits are tight).
    joint_zero_offset_range_deg: dict = {
        "yaw": 1.0,
        "roll": 1.0,
        "pitch": 2.0,
        "knee": 2.0,
        "ankle": 3.0,
    }

    # Per-episode mass SCALE randomization applied to base_link only.
    # Directly motivated by this project's own history: the real
    # battery/Pi-mount/decoration mass (qmini_urdf-2legs.usda's
    # centerOfMass/mass values) was initially completely unmodeled, and
    # even now those are hand-corrected estimates, not a precise
    # measurement -- there's no reason to assume they're exact. (0.9, 1.1)
    # = base mass scaled by a random factor in that range each episode.
    base_mass_randomization_range: tuple[float, float] = (0.9, 1.1)

    # Per-episode foot/ground FRICTION, added 2026-10-06. Before this the
    # sim always used Isaac Lab's default ground material (0.5/0.5,
    # combine "average", robot USD has no material of its own -> effective
    # 0.5, never varied). Measured on the real test floor (hard feet +
    # sticky layer): 11.7 kg robot needs 3.5-4.0 kg of sideways pull to
    # slide -> mu ~0.30-0.34, i.e. 2/3 of what every policy so far trained
    # on; the real feet visibly slip and the robot turns 8-13 deg/s.
    # The ground plane is shared by all envs, so it is spawned with 1.0 and
    # combine mode "multiply" (wins over the robot's "average") and the
    # per-env value is set on the robot's own shapes in _randomize_friction:
    # effective static friction = U(range), dynamic = static * U(ratio).
    friction_randomization_range: tuple[float, float] = (0.25, 0.8)
    dynamic_friction_ratio_range: tuple[float, float] = (0.8, 1.0)

    # Push disturbances: every push_interval_range_s (independently
    # randomized and re-randomized per env, see that cfg's comment for why
    # -- NOT synchronized across envs), a random horizontal velocity kick
    # is added directly to the base's linear velocity -- simulates a
    # bump/nudge/uneven-footing event. Standard technique for legged
    # balance robustness: without this, the policy only ever has to reject
    # its own comparatively gentle tracking-error disturbances, not an
    # external shove. Set push_interval_range_s very large (both ends) to
    # effectively disable.
    #
    # DISABLED AGAIN (1000.0, effectively off) -- history worth reading in
    # full before re-enabling this again:
    #   1. Originally disabled while bringing up the step-in-place gait
    #      from nothing (stacking a shove on an undiscovered skill just
    #      made the safe never-lift-a-leg local optimum more attractive).
    #   2. Re-enabled at 4.0 once stepping LOOKED demonstrably working
    #      (swing_target/foot_clearance correlating, tracking down to
    #      1-4deg) -- that evidence turned out to be contaminated (see
    #      point 4), but at the time it looked like a genuine skill a
    #      "never has to react to anything" policy could safely regress
    #      away from, which is exactly what happened next.
    #   3. That re-enabling was itself followed by a real collapse to a
    #      frozen, statically roll-tilted pose within ~5700 iterations --
    #      addressed separately via joint_tracking_weight (roll-specific)
    #      and foot_swing_reward_weight (raised 1.0 -> 3.0).
    #   4. CRITICAL finding: foot_swing_reward's height computation
    #      (body_pos_w relative to a captured stance reference) can't
    #      distinguish a genuinely-lifted foot from one thrown into the
    #      air by a push -- and pushes are active during training, so the
    #      apparent "stepping working" jump in tracking/foot_swing_reward
    #      after raising the weight (point 2/3) was likely substantially
    #      farmed from passive push-recovery motion, not real intentional
    #      stepping. Direct video inspection confirmed this: the only
    #      visible leg lift happens right when pushed, and even that is
    #      minimal -- confirmed to still be true even after gating
    #      foot_swing_reward off for foot_swing_push_cooldown_steps after
    #      each push (tracking/foot_swing_reward barely moved), meaning
    #      either that cooldown is too short for how long a real push
    #      actually takes to settle, or push-adjacent influence bleeds
    #      further into the cycle than a short fixed window can catch.
    # Disabling again removes the confound entirely rather than guessing
    # at a longer cooldown: with pushes off, tracking/foot_swing_reward
    # and video behavior should both be trustworthy measures of GENUINE
    # gait-driven stepping again. Once that's unambiguously confirmed on
    # video (not just the metric, given the above), re-enable pushes as a
    # deliberate hardening step -- not before, and this time verify with
    # video immediately after re-enabling rather than trusting the metric
    # alone.
    #
    # RE-ENABLED (2026-08-28, 4.0s, matching the value from point 2 above)
    # -- not for the training-metric reasons this was originally toggled
    # over, but for a real-hardware deploy failure: robot_deploy.py safety-
    # aborted repeatedly on right_hip_roll/left_knee targets WAY beyond
    # both the joint's hard limit AND the reference clip's own tiny ~2.4deg
    # roll range (an 18deg target against a 15deg limit and a <3deg
    # reference peak -- not a boundary-grazing issue, a genuinely wild,
    # out-of-distribution first action). This policy has never once, in
    # this entire session, had to recover from an unexpected/perturbed
    # state -- pushes were off the whole time genuine stepping was being
    # debugged. The standing policy that deployed successfully WAS trained
    # with pushes on. Real hardware's actual physical startup state
    # (calibration slop, loading transients, backlash) is exactly the kind
    # of "surprising, not seen in the clean reference trajectory" state
    # pushes are meant to prepare a policy for. Safe to retry now in a way
    # it wasn't during points 1-3 above: stepping itself has held up
    # stably for tens of thousands of iterations at this point (not a
    # fragile, just-discovered skill anymore), and the specific failure
    # mode from the SECOND attempt (collapse to a frozen, statically roll-
    # tilted pose, point 3) has since been separately addressed by
    # joint_tracking_weight's roll=3.0 fix and foot_swing_reward_weight
    # being raised further (3.0 -> 5.0) since then. Per this comment's own
    # standing instruction: verify with VIDEO immediately after re-
    # enabling, don't just trust tracking/foot_swing_reward.
    #
    # PUSH TIMING RANDOMIZED PER-ENV (2026-08-28), replacing the single
    # fixed push_interval_s above -- that was a single float shared via
    # self.common_step_counter, i.e. LITERALLY every env got pushed on the
    # exact same global step, every time (this cfg used to say so
    # directly: "Synchronized across envs... all get pushed the same
    # step"). Caught before it trained long enough to matter: episode
    # length is a fixed 499 steps (only real falls end one early, and
    # fall_rate is ~0), and the push interval was ALSO a fixed number of
    # steps -- two fixed-period signals with no randomization between
    # them, so nearly every episode would see its pushes land at close to
    # the same points in "steps since this episode's own reset", and
    # therefore close to the same points in the ~2.2s gait cycle
    # (keyframes_step_in_place_all_joints_2x_base_offset.json's period),
    # every time. The whole point of pushes is to cover disturbances that
    # can happen at ANY point in the gait, not train the policy to expect
    # a shove at one or two predictable phases -- which would have been a
    # narrower, less useful kind of robustness than intended, and not
    # necessarily representative of when a real push/stumble would
    # actually occur relative to the gait cycle. Fixed range still gives a
    # reasonable target frequency; the randomization is what actually
    # matters here, sampled and re-sampled independently per env (see
    # _resample_push_countdown / _steps_until_push), not tied to any
    # shared global counter anymore.
    # DISABLED AGAIN (2026-09-07), as part of a controlled revert-to-
    # last-known-good test -- see position_reward_weight's comment for the
    # full reasoning. This is the same 1000.0-effectively-off value used
    # the last (and only) time this project had genuine, video-confirmed
    # full-leg stepping (run 2026-08-24_22-29-06). Pushes were re-enabled
    # 2026-08-28, AFTER that run, and this project's own history above
    # (point 4) already documents that pushes contaminate foot_swing_reward
    # by making passive push-recovery motion look like real stepping --
    # exactly the ambiguity that kept coming up reviewing videos this
    # session. Re-enable only once genuine stepping is unambiguously back,
    # and verify with video immediately after, per this field's standing
    # instruction above.
    #
    # RE-ENABLED AGAIN (2026-09-23, 4.0s, same value as the 2026-08-28
    # attempt) as a FRESH START, not a resume. Motivated by 2026-09-22/23
    # hardware: with the power supply issue resolved (see
    # imu_fault_diagnosis notes), the real remaining failures are (a) a
    # phase-locked disturbance at motion_time~1.0-1.5s the joint_tracking_
    # weight work has been chasing joint-by-joint without eliminating it,
    # and (b) the policy occasionally grazing the hip-roll hard limit --
    # both consistent with a policy that has never had to recover from an
    # off-reference state, only ever track the clean reference. That's
    # exactly this field's own 2026-08-28 rationale, for exactly the same
    # kind of hardware failure (targets near/past joint limits). NOT
    # expected to directly shrink the phase-locked disturbance itself
    # (that's the policy's OWN commanded motion diverging from sim under
    # real dynamics, not an external-disturbance-recovery problem) -- this
    # is a complementary robustness step, not a replacement for that work.
    # Safe to try via fresh start where a resume already failed once
    # (collapse to a frozen roll-tilted pose within ~5700 iterations, see
    # point 3 above) -- same lesson as roll_reward_weight's whole history:
    # introducing a genuinely new disturbance onto an already-converged,
    # low-entropy policy (noise_std ~0.05 currently) is fragile via
    # resume. The reward terms suspended during push cooldown
    # (foot_swing_reward, position_reward, heading_reward, plus
    # foot_overswing_penalty/heading_deviation_penalty) are the same as
    # before, but target_limit_penalty/step_limit_penalty/
    # action_rate_penalty/joint_limit_penalty and the core tracking/
    # orientation/roll rewards all stay active THROUGHOUT a push now --
    # none of those existed during the 2026-08-28 attempt, so recovery
    # motion is more constrained this time than what caused that collapse.
    # Per this field's standing instruction: verify with video immediately
    # after training, and check compare_policies_isaaclab.py's
    # target_violation_rate/step_over5_rate specifically, not just
    # foot_swing_reward (still contaminable by push-recovery motion).
    #
    # RESULT: failed, and by the OTHER mechanism this field's own point 1
    # already warned about (not the resume-collapse point 3 this was
    # written to avoid). Fresh start, stopped at iteration ~4975/40000:
    # foot_swing_reward peaked ~0.08 near iteration 100 then collapsed to
    # ~0.008 and stayed there, while orientation/reward sat at ~0.997,
    # fall_rate at ~0, episode length maxed at ~496 -- the classic
    # never-discovers-stepping local optimum, not a slow-to-converge run
    # (every healthy fresh start in this project reaches ~0.7-1.3 by
    # iteration 2000-3000). Pushes active from iteration 0, before
    # stepping was ever discovered, made "survive every episode
    # (including pushes) by standing still" more attractive than
    # discovering the riskier genuine-stepping skill -- exactly point 1's
    # mechanism. Both fresh-start (this) and resume (point 3, 2026-08-28)
    # have now independently failed at introducing pushes in this project;
    # a real future attempt would need something in between (introduce
    # pushes AFTER stepping is established but BEFORE exploration noise
    # collapses, not at either endpoint) rather than repeating either of
    # these two. DISABLED AGAIN (1000.0) -- back to the last known-good
    # state. Not attempted further as of 2026-09-23; two other, better-
    # localized hardware issues (hip-roll grazing its limit at
    # motion_time~1.0s, yaw3's right-hip-pitch step-size violations at
    # ~0.36-0.40s) are the priority instead.
    push_interval_range_s: tuple[float, float] = (1000.0, 1000.0)
    push_velocity_range_mps: tuple[float, float] = (-0.4, 0.4)

    # --- startup support-then-release (rope simulation) -----------------
    # Added 2026-09-29. Real deployment currently ALWAYS starts with the
    # robot held up by a safety rope through the startup pose and into the
    # first second or two of stepping, then the rope is deliberately
    # loosened -- and per direct user testing, the robot genuinely falls
    # backward at the startup pose whenever the rope is slack (confirmed
    # by holding the pose with the rope loose, no policy running -- this
    # matches the independent finding that the current anchor pose pins at
    # the tilt-termination boundary when held statically, see
    # check_static_pose_stability.py's 2026-09-25 results). Once the rope
    # is let go on hardware, video + control_loop CSVs show ~2x the normal
    # commanded step size for several seconds (mean_maxstep 3.2-3.6
    # deg/step vs 0.8-1.9 deg/step in calmer windows) -- the policy visibly
    # struggling with a transition it has NEVER experienced, since this is
    # a single large step-change in effective external support, not the
    # small periodic velocity kicks push_velocity_range_mps applies (and
    # which have twice failed to train successfully regardless, see
    # push_interval_range_s's own history -- a different disturbance
    # PROFILE, not just a retry of the same one).
    #
    # Simulates the rope as a sustained upward external force on base_link
    # (NOT a full unilateral rope-distance-constraint -- that would need
    # its own physics and this is meant to be a tractable first attempt,
    # not a literal rope model) applied from episode start, held for a
    # per-env randomized duration, then dropped to exactly zero for the
    # rest of the episode -- matching "rope tight through startup, then
    # released once and not re-tightened" rather than push's repeating
    # schedule. See _pre_physics_step's application and
    # cfg.startup_support_force_frac below.
    #
    # UNTUNED starting guess: 0.5-2.0s covers both "released almost
    # immediately" and "held through ~1 gait cycle" -- real durations
    # varied a lot across the 2026-09-29 hardware attempts and weren't
    # precisely timed. Resume (not fresh start) is NOT expected to work
    # cleanly here despite being additive-only, for the SAME reason
    # push_interval_range_s's point 1 gives: an already-converged, low-
    # entropy policy meeting genuinely novel starting dynamics (the first
    # ~1s of EVERY episode now begins differently than anything it was
    # ever trained on) is closer to "new disturbance on a converged
    # policy" than "new disturbance during exploration" -- try fresh
    # start first; if that repeats push's OWN fresh-start failure (blocks
    # discovering stepping at all, since standing supported doing nothing
    # is safer when unsupported balance is suddenly also required from
    # step 0), fall back to resume and watch closely.
    # DISABLED (0.5,2.0 -> 0.0,0.0) on 2026-09-30, resuming from pitchstep5
    # (which never trained with this active) to test the roll margin
    # widening above in isolation -- support_end_step becomes 0 for every
    # env, so still_supported (episode_length_buf < support_end_step) is
    # never true and the applied force is always zero. Re-enable once the
    # margin-only change is confirmed safe on hardware; don't stack this
    # back in on the same run as an untested margin change, or a hardware
    # result won't tell us which one mattered. See
    # startup_support_force_frac's comment for the full mechanism history
    # (0.9 fresh-start failed outright; 0.7 resume kept stepping but
    # regressed hip-roll margin -- the thing just re-widened above).
    startup_support_duration_range_s: tuple[float, float] = (0.0, 0.0)

    # Fraction of the robot's own current (possibly base_mass_randomization
    # -scaled) weight counteracted by the upward support force while
    # active. 0.9, not 1.0 -- a real taut rope still lets the robot's own
    # controller do essentially all the balancing work while it's holding
    # roughly level (matching "the body was relatively flat because the
    # rope was holding the body up" -- user, 2026-09-29), it doesn't
    # perfectly zero gravity; leaving 10% real weight on the legs keeps
    # ground contact/stepping meaningful throughout the supported phase
    # rather than having the robot dangle just clear of the floor.
    #
    # LOWERED 0.9 -> 0.7 on 2026-09-29 (run 2026-09-29_17-03-52, fresh
    # start, iteration 14254/40000, stopped). RESULT: this mechanism, at
    # full strength from iteration 0, reproduced the exact fragility this
    # field's own comment above warned about -- NOT the "never discovers
    # stepping" failure (foot_swing_reward actually reached ~2.3-2.5,
    # above the normal ~1.5 ceiling, and held there iterations ~1500-3500)
    # but a LATER collapse: a visible crisis around iteration 3500-4000
    # (episode_length dip, a second fall_rate spike, a roll_deg spike),
    # coming out the other side onto the standing-still local optimum
    # instead -- foot_swing_reward crashed to ~0.16 and sat there flat for
    # the next 10000+ iterations with noise_std already down at ~0.08-0.1
    # (left/right_foot_clearance_cm -6.7/-2.5, i.e. not lifting at all).
    # tracking/startup_support_active_frac read 0.13 at the stop point,
    # matching the expected steady-state fraction (mean duration ~1.25s /
    # ~9s episode) -- the mechanism itself was working as designed, the
    # DISTURBANCE MAGNITUDE was plausibly just too severe for a policy
    # whose exploration had already mostly shut off by the time the
    # crisis hit. Dropped to 0.7 as a hedge alongside switching from fresh
    # start to RESUME (see this field's own comment on trying fresh start
    # first, then falling back to resume) -- both prior full-strength (0.9)
    # attempts at introducing a genuinely new disturbance in this project
    # (this one, and push_interval_range_s's original fresh-start/resume
    # pair) only ever tried max strength from the start; a gentler version
    # is a real, untried lever, not just a retry. Watch foot_swing_reward
    # and left/right_foot_clearance_cm specifically for the same collapse
    # signature if this also fails.
    startup_support_force_frac: float = 0.7

    # Weight on the action-rate penalty in _get_rewards (-weight *
    # sum((action_t - action_{t-1})**2)). 0.0 = off (previous behavior).
    # Start small (e.g. 0.01-0.05) and increase if deployed targets are
    # still visibly jittery -- too high will fight the tracking_reward and
    # produce a sluggish policy that can't keep up with the gait.
    #
    # RAISED 0.02 -> 0.1 on 2026-09-09, directly triggered by this field's
    # own documented condition above: first real-hardware attempts with the
    # 2026-09-08 sin/cos checkpoint (model_39999, run 2026-09-08_22-14-51)
    # hit repeated SAFETY ABORTs on max_step_deg (5deg default, still
    # aborted at 15deg loosened) within ~0.2s of policy control starting.
    # Cross-checked the actual deploy-side decode math against the logged
    # control_loop CSV (target_deg = default_pos_deg + degrees(action_scale
    # * action)) and it matches exactly -- not a decode bug. The real
    # per-step TARGET deltas were genuinely escalating (roughly 2-8deg/step
    # early, climbing to 7-15deg/step by the 8th-11th control tick) with
    # IMU ang_vel_deg_s climbing alongside it (single digits -> 50-70+
    # deg/s over the same ~10 steps) -- a real, growing oscillation, not a
    # one-off spike. This is the same checkpoint already characterized in
    # sim as "aggressive, sometimes falls" (large tracking errors, high
    # yaw_rate_dps) -- expected, since this checkpoint was deliberately
    # picked to verify sin/cos survives genuine stepping BEFORE doing any
    # gait-quality polishing (see tracking_linear_penalty_weight and
    # heading_reward_weight's comments for that whole arc). Real hardware's
    # actuator latency/backlash/structural compliance isn't perfectly
    # modeled in sim, so a jerky, large-step-to-step policy that looks
    # merely energetic in sim is exactly the kind most exposed by that
    # sim-to-real gap. NOT raising max_step_deg further -- that would mask
    # the safety margin rather than fix the underlying jerkiness. 0.1 (5x)
    # is a first, moderate step per this field's own "start small and
    # increase if" guidance; resume (not fresh-start) from model_39999.pt
    # to keep the already-hard-won genuine stepping skill while adding this
    # new smoothness pressure on top. Watch tracking/action_rate_penalty
    # and whether deployed step sizes actually shrink; if still tripping
    # max_step_deg, raise further before considering action_smoothing
    # (already tried once as a permanent EMA and reverted, see that cfg's
    # comment -- this reward-based penalty is more surgical, doesn't blindly
    # low-pass filter every command the way a fixed EMA does).
    #
    # RAISED AGAIN 0.1 -> 0.3 on 2026-09-10, after the resumed run trained
    # under 0.1 was checked with compare_policies_isaaclab.py against the
    # pre-change checkpoint (experiments/qmini-leg/compare_action_smoothness.csv)
    # and the result was underwhelming: mean_action_delta_deg only dropped
    # ~1.274 -> ~1.227 deg (~4%) and mean_tracking_err_deg barely moved --
    # a 5x weight increase (0.02->0.1) bought almost nothing. That csv is a
    # MEAN over every joint/env/step, though, not a worst-case -- the real
    # hardware SAFETY ABORT this whole change was meant to fix was a
    # single-joint, single-step OUTLIER (10-15+ deg), which a small drop in
    # the average doesn't confirm or rule out either way. Raising
    # aggressively rather than incrementally this time specifically to get
    # a clearer signal one way or the other; re-verify with
    # compare_policies_isaaclab.py again after this run (one-checkpoint-
    # per-invocation, per that script's own multi-checkpoint-session
    # caveat) before ever attempting real hardware again.
    #
    # REVERTED 0.3 -> 0.02 (its original pre-2026-09-09 value) on 2026-09-16.
    # Two fresh-start runs since the 2026-09-07/08 breakthrough (which found
    # genuine stepping at 0.02) failed to find any stepping at all
    # (foot_swing_reward stayed ~0.006-0.008 for 40k/5600+ iterations, vs
    # ~1.5 by iteration 3000 on both prior successful fresh starts). Dug
    # into the SECOND failed run's own per-iteration tensorboard data
    # (free, no extra training needed) rather than guessing further: right
    # as foot_swing_reward briefly rose during a transient rediscovery
    # (iterations ~3075-3210), THIS term grew ~9x (0.11->1.38) in lockstep,
    # BEFORE episode_length or fall_rate moved at all -- i.e. a direct,
    # immediate mechanical consequence of genuine stepping's bigger/faster
    # joint motions, not a side effect of falling. See
    # target_limit_penalty_weight's comment for the paired finding (that
    # term moved even more, ~150x). Reverting both for a fresh-start
    # confirmatory test; once genuine stepping is re-established, harden
    # this back up gradually via RESUME (not fresh start) the same way it
    # worked the first time, watching compare_policies_isaaclab.py's
    # mean_action_delta_deg/target_violation_rate at each step rather than
    # jumping straight back to 0.3.
    action_rate_penalty_weight: float = 0.02

    # Per-joint-TYPE multiplier on the action-rate penalty's squared term,
    # same name-suffix matching convention (and same tensor-building code)
    # as cfg.joint_tracking_weight. Added after a close-up screen recording
    # of a single leg (not the usual wide training-eval video) showed the
    # right ankle specifically doing several real direction reversals
    # within ~0.5s right after reset -- a genuine several-Hz rocking
    # oscillation, not a one-off "pop up and down". That rules out a
    # reward-hacked flick (would show as one clean motion, not repeated
    # reversals) and points at underdamped step response to the fresh
    # post-reset setpoint instead -- but two attempts at fixing this via
    # actuator gains already failed (narrower gain_randomization_range: no
    # effect; DEFAULT_GAINS["ankle"]["damping"] 2.0->4.0: didn't stop it
    # AND broke tracking, see qmini.py's comment). action_rate_penalty
    # already exists specifically to punish step-to-step jitter but is
    # currently a flat, equal-weight sum across all 10 joints -- weighting
    # it toward ankle targets the exact symptom directly, the same way
    # joint_tracking_weight's roll=3.0 fixed a different joint-specific
    # problem earlier.
    #
    # RESULT (2026-08-26, run 2026-08-26_14-05-28, ~5hr/16000 iterations):
    # negative, same failure shape as the damping attempt -- didn't stop
    # the right-ankle shake ("right leg still has the same issue"), AND
    # this time suppressed the LEFT leg's swing too (foot_swing_reward
    # dropped from a ~0.65-0.70 baseline to 0.35, left_swing_target~0.54
    # with left_foot_clearance~0 -- calling for a real lift and getting
    # none). Reverted to 1.0/flat. Three attempts now (narrower
    # randomization, higher damping, this) have all tried to make the
    # ankle react LESS overall and all failed the same way: unable to
    # discriminate "reacting to noise" from "reacting to the genuine swing
    # it needs to do", so genuine swing gets suppressed right along with
    # the oscillation, without the oscillation actually going away. Don't
    # try a fourth blanket-suppression variant on this joint without a
    # mechanism that's frequency-selective instead (e.g. a low-pass filter
    # on the action itself, which BD-X's own low-level control uses
    # exactly for this kind of reason -- see Table I / Sec VI-D of the
    # BD-X paper).
    action_rate_joint_weight: dict = {
        "yaw": 1.0,
        "roll": 1.0,
        "pitch": 1.0,
        "knee": 1.0,
        "ankle": 1.0,
    }

    # Linear penalty added ON TOP of tracking_reward/velocity_reward's
    # existing exp(-scale*error^2) term -- see that comment in
    # _get_rewards. The pure-exponential shape saturates: once error is
    # already large, its gradient vanishes and nothing pulls it back
    # toward the reference. That's the exact mechanism behind a real
    # exploit this training run found (locking joints against their hard
    # limits, see joint_limit_penalty/target_limit_penalty's comments) --
    # this doesn't replace those penalties (they're still needed as a
    # backstop), it fixes the underlying cause so the policy has a reason
    # to reduce a large error in the first place. Inspired by Disney
    # Research's BD-X paper (Table I), which uses a purely linear
    # -||q-q_hat||^2 penalty for joint tracking instead of an exponential
    # bonus. Kept small and additive (not a full replacement) so
    # near-zero-error behavior -- where the exponential term already
    # provides a strong, separately-tuned incentive -- is barely affected,
    # avoiding the need to re-tune every other already-established reward
    # weight in this file. UNTUNED starting guess.
    #
    # RAISED 0.2 -> 1.0 on 2026-09-06, before a fresh from-scratch run.
    # Re-read BD-X's Table I directly: leg joint positions get weight 15.0
    # against torso orientation's 1.0 (a 15:1 ratio) -- ours currently has
    # this backwards (tracking_reward's outer weight is 2.0 against
    # orientation_reward_weight's 3.0). More importantly, their leg term is
    # PURELY the unbounded -||q-q_hat||^2 penalty, no exponential at all.
    # That reconciles two separate findings in this file under one cause:
    # today's (a policy can flatten/skip the hard, fast parts of the gait
    # and only pay a bounded, shrinking cost under the exp term) and the
    # earlier one a few hundred lines below re: foot_swing_reward_weight's
    # 1.0->3.0 raise (a policy that already tracks well gets almost no
    # marginal reward left for the extra risk of real foot clearance,
    # because the exp term is already near its max). An exponential is flat
    # at BOTH tails -- saturates near-max for already-good tracking (no
    # incentive to keep polishing) and near-zero for already-bad tracking
    # (no growing penalty for further slacking). The unbounded linear term
    # doesn't have either flat region, so it's the more surgical fix versus
    # just raising the outer 2.0x weight, which would only rescale the
    # still-flat exponential without touching the shape problem at either
    # end. Not adopting BD-X's exact 15.0 or its squared (vs. linear) form
    # outright -- our error/reward scales aren't directly comparable to
    # theirs, and a smaller, directional correction is more appropriate
    # given this is genuinely untested at any value above 0.2. Safe to be
    # more assertive than a typical single-change tweak here specifically
    # because this is going into a fresh from-scratch run, not a resume --
    # no existing policy/optimizer state to destabilize.
    #
    # REVERTED 1.0 -> 0.2 on 2026-09-07, as the next step in the same
    # revert-to-last-known-good test as push_interval_range_s/
    # orientation_margin_deg/orientation_reward_scale/position_reward_weight/
    # heading_reward_weight (see position_reward_weight's comment for the
    # full reasoning/history). Reverting that whole cluster and running
    # fresh to iteration ~22.5k did NOT reproduce genuine stepping either
    # (video-confirmed: standing/swaying in place, same as every run since
    # the 2026-08-24 reference, with foot_swing_reward peaks again tracing
    # back to falls, not real lift) -- ruling out the "stay-still pressure"
    # cluster as the SOLE blocker, on top of the phase-encoding revert
    # already having ruled out sin/cos. This term is the next largest
    # remaining divergence from the 2026-08-24 config that hasn't been
    # tested in isolation: unlike the terms above, it was never gated by a
    # margin or push-cooldown -- it's a small but unbounded, always-active
    # per-step penalty on joint-angle deviation across all 10 joints, which
    # plausibly discourages exactly the bigger transient joint excursions a
    # real step requires more than a small in-place wobble does. Keeping
    # foot_swing_reward_weight=7.0 and action_delay_range_steps' widening
    # unchanged for now -- this is a single-variable isolation, not a full
    # revert to Aug 24 -- so if stepping still doesn't reappear, those two
    # become the next candidates.
    tracking_linear_penalty_weight: float = 0.2
    velocity_linear_penalty_weight: float = 0.05

    # Per-joint-TYPE multiplier on tracking_reward's error term (matched by
    # name suffix, same convention as startup_joint_pos_noise_deg) --
    # everything defaults to 1.0 except roll. Added after FOUR separate
    # training attempts (varying entropy_coef 0.005/0.03/0.01/0.02+pushes)
    # all converged to the exact same exploit: a static, widened stance via
    # a persistent roll deviation (once even a full sign flip vs. the
    # reference), which bought real passive stability -- resists pushes,
    # keeps orientation_reward near 1.0 -- without ever needing genuine
    # single-support balance. tracking_reward sums squared error across all
    # 10 joints with equal weight, but this gait's own roll reference
    # amplitude is tiny (~1.4-2.5 deg) next to pitch's (~15-22 deg), so a
    # roll deviation that's proportionally huge relative to what the
    # reference actually asks for (4x overshoot, sometimes a full sign
    # flip) still reads as numerically small next to the other 9 joints'
    # errors -- the exponential/linear tracking penalty barely notices it,
    # even though it's the entire mechanism the policy is exploiting.
    # Four independent search strategies finding the identical loophole is
    # strong evidence this needs a structural reward fix, not more
    # exploration tuning. UNTUNED starting guess -- watch tracking/*_roll_error
    # specifically and raise further if this still isn't enough to prevent
    # the same collapse recurring.
    #
    # ankle RAISED to 2.0 after a long, otherwise-healthy training run
    # (foot_swing_reward stable ~0.77, best tracking of the project, zero
    # falls -- not a collapse) plateaued at a partial "toe-lift" instead of
    # a full foot-lift, left side only. Different situation from roll's
    # exploit -- ankle isn't being gamed, it's just consistently the
    # worst-tracked joint throughout the project (persistently 3-8 deg
    # errors when other joints settle to 1-4 deg) and completing a
    # toe-lift into a real, full-clearance step specifically requires
    # ankle dorsiflexion during swing. Less aggressive than roll's 3.0
    # since this is an underperforming joint needing more incentive, not
    # a runaway exploit needing to be shut down. UNTUNED -- watch
    # tracking/*_ankle_error and whether the video shows full (not just
    # toe) clearance, and whether the right foot starts participating too.
    # RAISED 2026-09-23: roll 3.0->6.0, ankle 2.0->3.0. Root-caused via
    # hardware -- 10 separate 2026-09-22 runs (both the original tighter
    # IMU-fault threshold and the loosened one) all hit their worst
    # disturbance at motion_time~1.0-1.3s, every single cycle (30+ cycles
    # observed). That's exactly the keyframes' own documented swing-onset
    # discontinuity window (see this file's _comment: pitch/knee/ankle
    # re-smoothed there on 2026-09-03 after a real robot_deploy.py
    # runaway) -- but that same comment explicitly says roll's OWN
    # reference was checked and is already smooth through this region, and
    # ankle's was re-smoothed too. So the policy's large roll/ankle
    # excursion there (hardware FK-style replay: ~17deg hip-roll swing and
    # ~23-25deg right-ankle swing, against references of ~2.7deg and
    # ~6deg respectively) isn't the reference asking for it -- it's the
    # policy choosing it, right when pitch/knee are making their fastest
    # transition, almost certainly as a fast balance lever. This is the
    # SAME exploit category roll=3.0 was originally added to shut down
    # (see that history above) recurring in a new form 3.0 wasn't enough
    # to prevent. Resume, not fresh start -- this reshapes tracking_reward's
    # existing per-joint weighting, not a separate additive penalty term
    # with its own escalation dynamics (unlike foot_overswing_penalty_weight,
    # which broke at big single jumps); the original roll=3.0/ankle=2.0
    # values were themselves introduced this way. Still watch
    # foot_swing_reward and video closely -- pushing joint tracking too
    # hard on ANY joint has suppressed genuine stepping before
    # (tracking_linear_penalty_weight's whole history), and this is
    # deliberately a real, not timid, step (2x on roll) given the
    # measured excess is ~3x the reference amplitude.
    #
    # RESULT (2026-09-23, compare_policies_isaaclab.py on model_159996.pt
    # vs the pre-change checkpoint): roll error improved 1.85/1.56 ->
    # 1.51/1.17deg, ankle 2.31/2.33 -> 2.13/1.55deg -- both real, confirmed
    # in eval not just training curves. But NOT a clean fix: yaw error got
    # notably worse (1.61/0.81 -> 2.67/2.04deg) and pitch somewhat worse
    # (4.13/2.79 -> 4.66/4.03deg); mean_tracking_err_deg and body tilt/
    # roll/pitch all ticked up slightly too. Read: the balance-correction
    # need at that gait phase didn't go away, it partly moved to yaw and
    # pitch instead of roll/ankle -- the same whack-a-mole pattern
    # roll=3.0 was originally introduced to fight, recurring one level
    # down.
    #
    # RAISED yaw 1.0 -> 3.0 on 2026-09-23 to follow it there. Yaw
    # specifically, not pitch, even though pitch also got worse: yaw's
    # reference is EXACTLY 0 always (this gait never asks for any yaw
    # motion), so tightening it has essentially no risk of suppressing
    # genuine stepping -- any yaw error is pure unwanted behavior by
    # construction. Pitch is different: it's the primary joint actually
    # driving genuine stepping (hip-pitch/knee-pitch lift), with a large
    # legitimate reference range (~15-22deg) that a further-tightened
    # weight could bite into, repeating the exact suppression failure mode
    # tracking_linear_penalty_weight caused earlier in this project (see
    # that field's whole history). Leaving pitch/knee alone this round to
    # isolate whether fixing yaw alone is enough before touching the one
    # joint most dangerous to over-constrain. Resume from this same
    # checkpoint (2026-09-22_13-45-07/model_159996.pt), same reasoning as
    # the roll/ankle change re: why resume is fine here. Watch
    # tracking/left_yaw_error /right_yaw_error, whether pitch/tracking_err
    # keep degrading anyway (would mean yaw wasn't the real outlet), and
    # foot_swing_reward for any suppression.
    joint_tracking_weight: dict = {
        "yaw": 3.0,
        "roll": 6.0,
        "pitch": 1.0,
        "knee": 1.0,
        "ankle": 3.0,
    }

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
    #
    # scale RAISED BACK 8.0 -> 20.0 (2026-08-31), now that
    # orientation_margin_deg=2.0 exists (added later than this comment's
    # own history above) and already solves the exact problem that
    # justified lowering scale in the first place -- small, legitimate
    # lean is fully free up to the margin regardless of scale, so scale
    # no longer needs to stay gentle to protect it. With margin active,
    # this is mathematically NOT the same as the original bare scale=20
    # (which measured tilt from zero); (sin(tilt)-sin(margin)) is smaller
    # than sin(tilt) alone, so the penalty ramps up more gradually past
    # the margin than the original comment's "collapses by 13deg" implies.
    # Root cause this addresses: real hardware never once got a leg off
    # the ground across 4 attempts, all failing on a smooth, accelerating
    # left_hip_roll divergence within the first ~1s -- and video (both the
    # headless sim eval and a close-up recording) confirmed real,
    # cycle-synced torso lean well beyond 2deg during otherwise-healthy
    # sim stepping. Checked the actual reward curve: at scale=8.0, 10deg
    # of real tilt still kept 86% of max orientation_reward, 15deg kept
    # 67%, 20deg kept 47% -- the margin was creating a genuine free zone
    # up to 2deg, but past it the penalty was too gentle to be a real
    # behavioral constraint, letting the policy lean far more than the
    # margin's name implies without much cost. At scale=20.0: 10deg ~=
    # 68%, 15deg ~= 37%, 20deg ~= 15% -- a real, felt cost for leaning
    # well past the margin instead of an almost-free one. UNTUNED --
    # watch for the original failure mode recurring (a fall from
    # insufficient gradient on a *small*, legitimate lean) now that this
    # is back near its pre-margin value; if that happens, split the
    # difference rather than reverting all the way to 8.0, since the
    # margin is still doing real work 8.0 didn't have.
    # scale REVERTED 20.0 -> 8.0 (2026-09-07), back to the pre-2026-08-31
    # value, as part of the same revert-to-last-known-good test as
    # push_interval_range_s/orientation_margin_deg/position_reward_weight/
    # heading_reward_weight -- see position_reward_weight's comment for the
    # full reasoning. This scale was raised specifically to make leaning
    # past orientation_margin_deg cost more, which is direct tension with
    # what genuine single-support stepping requires (leaning to shift
    # weight onto the stance leg).
    #
    # scale RAISED 8.0 -> 12.0 (2026-09-10), a deliberately PARTIAL move
    # (not back to 20.0). Across the three action_rate_penalty /
    # target_limit_penalty tuning iterations, mean_tilt_deg in the
    # compare_policies_isaaclab.py eval crept 8.16 -> 8.56 -> 9.00 -- the
    # policy leans progressively more as it gets more dynamic. Still 0%
    # fall_rate in that eval so it's tolerable now, but it's a consistent
    # trend and left unchecked could eventually cost stability. 12.0 adds a
    # real, felt cost specifically for the larger leans (per the reward-
    # curve numbers above: ~7-10% more orientation_reward lost at 12-20deg
    # tilt vs scale 8.0) while barely touching routine 8-9deg leans, so it
    # arrests the tail without re-creating the "too rigid to step" pressure
    # that reverting fully to 20.0 would. Keeping orientation_margin_deg at
    # 6.0 -- margin is the riskier knob (it was the one deliberately
    # loosened 2->6 to unlock stepping); tighten it only if scale=12.0
    # doesn't hold the tilt.
    orientation_reward_weight: float = 3.0
    orientation_reward_scale: float = 12.0

    # Degrees of lean given a FULL orientation_reward bonus (no penalty at
    # all) before tilt beyond this starts costing reward -- see the margin
    # comment inside _get_rewards' orientation_reward computation for why
    # this exists: it wasn't needed while training standing-still (zero
    # lean really was always best there), but the step-in-place gait
    # (keyframes_step_in_place_all_joints_2x_base_offset.json) exposed a
    # real conflict -- lifting one foot requires shifting the CoM over the
    # stance leg, a genuine lean, which this term used to punish starting
    # from zero regardless of cause. UNTUNED starting guess: a bit above
    # the reference gait's own ~5.4 deg peak-to-peak roll sway (checked via
    # keyframes_step_in_place_all_joints_2x_base_offset.json directly),
    # since the real CoM lean during single support likely needs more than
    # that one joint's angle alone -- hip roll + ankle roll + actual body
    # tilt all contribute to it.
    # Lowered from 6.0 to 2.0 once real step-in-place stepping was working
    # (both feet lifting, foot_swing_reward ~1.0) but the body was visibly
    # rocking more than needed (2026-08-24_22-29-06 run, iterations
    # 282000/286999 video). With a 6deg margin, tilt_sin - margin_sin was
    # already 0 for a lot of that sway, so orientation_reward gave zero
    # gradient against it -- it was "free" body lean while tracking_reward
    # still penalized deviating hip_pitch/knee further from the reference
    # to balance a different way instead. 2deg keeps just enough margin for
    # a real weight shift onto the stance leg without smuggling in several
    # extra degrees of unpenalized rocking; if this proves too tight
    # (fighting the genuine CoM shift needed to lift a foot at all) watch
    # for foot_swing_reward regressing and split the difference.
    #
    # That's exactly what happened: run 2026-08-26_21-10-07 (resumed clean
    # from model_367800.pt, none of the four reverted gain/action
    # experiments active) trained fine for ~10000 more iterations, then
    # foot_swing_reward and left_foot_clearance_cm collapsed and stayed
    # collapsed through the rest of the run (noise_std also settling low,
    # ~0.05->0.04, at the same time -- the "collapsed into a safe local
    # optimum, can't explore back out" pattern, not random instability).
    # Notable that it took this long (2° was introduced ~80k iterations
    # before the collapse) rather than showing up immediately -- consistent
    # with slowly mounting pressure against genuine CoM shift rather than
    # an immediate hard conflict, only becoming an unrecoverable collapse
    # once noise_std had also drifted low enough to lock it in. Splitting
    # the difference to 4.0 here, per the plan above, to retest from the
    # same model_367800.pt baseline.
    #
    # RESULT of 4.0: also negative, and actually WORSE than 2.0 -- collapsed
    # almost immediately on resume (sharp drop within ~6000 iterations,
    # visible as an unusually large noise_std spike, much bigger than any
    # previous resume transient) and stayed flat at foot_swing_reward~0.38
    # for the entire rest of a ~30000-iteration run once noise_std settled
    # back down -- genuinely stuck, not still adjusting. This is the
    # OPPOSITE of what the margin-tightness hypothesis predicts (looser
    # should mean more forgiving, not faster to collapse), so that
    # hypothesis doesn't hold up. Reverted to 2.0 -- not because 2.0 is
    # confirmed correct, but because it's the last configuration that
    # actually trained stably for a long stretch (~80k iterations) before
    # its own collapse, i.e. the best-evidenced "good" state available.
    #
    # Bigger picture after FIVE different changes (four action/gain
    # experiments, now this) all producing the same collapse after
    # resuming model_367800.pt: the common factor probably isn't any one
    # of the specific things changed. That checkpoint is already very
    # converged/low-noise (~0.045 noise_std BEFORE any of these changes),
    # so it has very little exploration budget left to recover through a
    # value-function shock from ANY reward perturbation -- it appears to
    # reliably slide toward "stop lifting" (lower-risk, easier to fall
    # into) rather than rediscovering confident stepping, regardless of
    # which specific thing perturbed it. Don't keep tuning reward/gain
    # parameters against this checkpoint looking for the "right" value --
    # if a plain, unmodified resume from here ALSO eventually collapses on
    # its own, that points at needing a different checkpoint (or more
    # entropy/exploration right at this stage) rather than more reward
    # tuning.
    #
    # LOOSENED 2.0 -> 6.0 (2026-09-07), back to the pre-tightening value,
    # as part of the same revert-to-last-known-good test as
    # push_interval_range_s and position_reward_weight above/below -- this
    # was tightened 6->2 on 2026-08-25, AFTER the only run this project has
    # ever had genuine, video-confirmed full-leg stepping on (see
    # position_reward_weight's comment). See that comment for the full
    # reasoning; this field is reverted for the same test.
    orientation_margin_deg: float = 6.0

    # --- roll specifically ----------------------------------------------
    # Added 2026-09-15 after extending compare_policies_isaaclab.py to
    # split orientation_reward's combined tilt metric into per-axis
    # numbers (see that script's mean_roll_deg/mean_pitch_deg) and finding
    # roll at 7.33deg vs pitch at only 2.70deg -- roll is the dominant,
    # almost the ENTIRE contributor to mean_tilt_deg (8.20deg =~
    # sqrt(7.33^2 + 2.70^2)), not a balanced mix. Confirms a real-hardware
    # observation (video comparison against reference footage of this
    # robot class showed a visibly stable torso there vs visible roll here)
    # with an actual number, and lines up with this project's oldest
    # documented real-hardware failure mode -- the very first attempts,
    # months before any of the heading/overswing work, all failed on "a
    # smooth, accelerating left_hip_roll divergence within the first ~1s"
    # (see orientation_reward_scale's history above).
    #
    # projected_gravity_b's X component encodes ROLL specifically (Y is
    # pitch) -- see robot_config_qmini_stepinplace.json's _imu_comment
    # (base_link +X points toward the LEFT leg, the lateral axis; a roll
    # rotation, about the fore-aft/Y axis, is what projects gravity onto
    # X) and compare_policies_isaaclab.py's matching comment.
    #
    # This is an ADDITIONAL bonus on top of orientation_reward above, not
    # a replacement -- orientation_reward's combined margin/scale are
    # already working (best-yet orientation/reward this whole series), so
    # this adds dedicated, undiluted pressure on roll specifically rather
    # than risking that by restructuring it. Reuses orientation_margin_deg
    # (6deg) rather than a tighter number for now -- see the conversation
    # this was added from: that margin was deliberately chosen (and a
    # tightening to 2deg deliberately reverted) because single-support
    # stepping physically requires some roll to shift weight onto the
    # stance leg, not just error to eliminate. Giving roll its own
    # UNDILUTED margin/weight (rather than sharing one budget with pitch,
    # which barely uses its share) is already a real tightening in
    # practice without needing to shrink the number too. If roll is still
    # too high after this, tighten roll_margin_deg specifically as the
    # next single-variable step -- don't change this AND the margin in the
    # same run, or a failure won't tell you which one mattered.
    # UNTUNED starting guess for the weight.
    #
    # LOWERED 2.0 -> 1.0 on 2026-09-15. The first run at 2.0 (resumed
    # 2026-09-14_12-14-23/model_279993.pt -> 2026-09-15_00-14-19,
    # iterations 280000-319992) DID work -- model_315000.pt showed
    # mean_roll_deg down 17% (7.33->6.10) and mean_tilt_deg at its best
    # value of the whole tuning series (7.09) -- but the run also finished
    # in an apparent catastrophic state (model_319992.pt: fall_rate 87%,
    # mean_action_delta_deg 126). Dug into the raw per-iteration
    # tensorboard data (not just the coarse checkpoints) rather than
    # guessing: this was NOT a single one-way collapse. Loss/value_function
    # showed 17 distinct spike events (>3000, vs a ~300-800 baseline)
    # across the 40k-iteration run, at ESCALATING frequency (3 in
    # 280k-290k, 1 in 290k-300k, 5 in 300k-310k, 8 in 310k-320k) and
    # escalating peak severity (up to 59614 at step ~315956) -- and 16 of
    # those 17 fully self-recovered within a few hundred iterations,
    # including one bigger than the "catastrophic" final one. Training just
    # happened to hit its iteration budget mid-spike on the 17th (started
    # step 319936), which is why the final checkpoint looked uniquely bad --
    # it wasn't a point of no return, just unlucky timing. Every spike
    # inspected showed orientation/roll_deg jumping in lockstep with
    # Loss/value_function and tracking/target_limit_penalty, which is why
    # this is being lowered rather than left alone: roll_reward and
    # orientation_reward both respond to the same physical signal (body
    # tilt), so a rare large-roll event now moves TWO reward terms at once,
    # plausibly making the value function's job harder around exactly that
    # part of state space -- and that's a real, escalating-frequency cost
    # even though every individual event (but the last, unlucky one) self-
    # corrected. Resuming from model_315000.pt (the last verified-stable
    # checkpoint, NOT the collapsed 319992), watching whether spike
    # frequency drops at this lower weight while still holding most of the
    # roll improvement.
    # DISABLED (1.0 -> 0.0) on 2026-09-16. Lowering the weight (2.0->1.0,
    # see history above) was meant to test whether roll_reward was driving
    # the recurring resumed-run instability -- instead the SAME lower
    # weight run showed FAR worse instability (70 spike events vs 17,
    # peaks up to 306945 vs 59614, one sustained 2634-iteration collapse)
    # and the final checkpoint never recovered, unlike the 2.0 run's
    # otherwise-comparable collapse. That doesn't fit "weight too high" as
    # the mechanism. A subsequent completely FRESH start (no resume, to
    # rule out resume-chain state as the cause) was run with everything
    # else unchanged, and it revealed something more fundamental: genuine
    # stepping never appeared at all this time (foot_swing_reward ~0.006
    # for the whole 40k-iteration run, vs ~1.5 by iteration 3000 on both
    # prior successful fresh starts, 2026-09-07/08) -- the policy instead
    # converged onto the "stand very still, track reference joint angles
    # tightly" local optimum this project has hit and had to fix multiple
    # times before (see tracking_linear_penalty_weight and
    # position_reward_weight's comments): tracking/reward abnormally HIGH
    # (6.49 vs the usual ~3-4.5), orientation/reward and roll_reward both
    # pinned near-max (0.999/1.000), noise_std very low (0.09, confidently
    # converged onto something -- just not stepping).
    #
    # Likely mechanism: roll_reward is a SECOND, independent reward channel
    # for staying level, stacked on top of orientation_reward rather than
    # sharing its existing budget (both use the same 6deg margin on the
    # same physical signal) -- roughly doubling the "stay level" pressure
    # against what genuine single-support stepping requires (leaning to
    # shift weight onto the stance leg). A RESUMED run has an established
    # stepping skill with some inertia defending it against that pull; a
    # FRESH run has nothing to defend and falls straight into the safer
    # optimum instead of ever discovering stepping. This also reframes the
    # resumed-run instability above: those runs still had real stepping
    # (with recurring destabilization), which is a different, less
    # fundamental problem than never finding it at all -- suggesting
    # roll_reward's strength is a real, independent problem, not purely a
    # resume-chain artifact.
    #
    # Disabling entirely as a clean, single-variable confirmatory test:
    # another fresh start with everything else unchanged. If genuine
    # stepping reappears reliably (matching the 2026-09-07/08 pattern),
    # that confirms the diagnosis, and roll stability should be
    # reintroduced more gently next time -- e.g. folded into
    # orientation_reward's existing budget instead of stacked as a second
    # channel, or at a much lower weight -- rather than as-is.
    #
    # That confirmatory test passed (fresh start under this AND
    # action_rate_penalty_weight/target_limit_penalty_weight/
    # target_limit_penalty_linear_coef all reverted together found genuine
    # stepping again by iteration ~2000-3000, matching 2026-09-07/08) -- but
    # video review of that run showed the body itself swinging/rotating a
    # lot to shift weight, unlike reference footage of this robot class
    # (Qmini_simulation.gif, the Disney BD-X recording), where the torso
    # stays visually still and weight-shift happens at the hip-roll JOINT
    # instead (confirmed in that run's own log: reference hip-roll amplitude
    # was tiny, ~0.5deg, while actual roll was -3 to -7deg -- the policy
    # wasn't tracking the small joint-level reference, the body was
    # absorbing the difference).
    #
    # RE-ENABLED (0.0 -> 1.0) on 2026-09-16 for a NEW fresh start (not a
    # resume) alongside the now-reverted action_rate_penalty_weight/
    # target_limit_penalty_weight/target_limit_penalty_linear_coef --
    # testing whether shaping roll from the very start of training, rather
    # than layering it onto an already-established aggressive gait via
    # resume (which is what caused the earlier instability, see the history
    # above), lets the policy find a genuinely calmer gait across the board
    # from the outset instead of learning "big and fast" first and fighting
    # to rein it in after. This is a different, cleaner scenario than every
    # previous roll_reward attempt: those were all resumes on top of an
    # already-aggressive policy; this is present from iteration 0, alongside
    # the gentler action_rate/target_limit values already confirmed to
    # allow fresh-start stepping discovery on their own. 1.0 chosen as a
    # middle value (not the original 2.0, not the even-more-unstable-in-
    # resume 1.0 from before -- though note that instability was measured
    # under resume conditions this isn't repeating). If stepping fails to
    # appear again, roll_reward is confirmed to fight cold-start discovery
    # regardless of the other terms and needs a fundamentally gentler shape
    # (e.g. folded into orientation_reward's budget rather than a stacked
    # second channel, per the note above) rather than just a lower weight.
    #
    # That fresh start at 1.0/margin=2.0 worked cleanly: only 1 tiny
    # Loss/value_function spike across the whole 40k-iteration run (vs. 17
    # and 70 in the two resumed attempts), and mean_roll_deg/mean_tilt_deg/
    # mean_pitch_deg all hit new project bests (4.84/5.67/2.02 deg via
    # compare_policies_isaaclab.py). Still clearly more body roll than
    # reference footage of this robot class shows (near-zero visible torso
    # movement), so pushing further. RAISED 1.0 -> 2.0 on 2026-09-17, back
    # to the original pre-resume-chaos value, for ANOTHER fresh start (not
    # a resume) -- deliberately not tightening roll_margin_deg further in
    # the same step (already a 3x cut, 6->2, don't stack two changes).
    # Important: resuming was considered and rejected here even though it
    # would be cheaper -- the second failed resume attempt (2.0 -> 1.0)
    # wasn't introducing a new term, it was changing the weight of an
    # ALREADY-active roll_reward, and that was the WORSE of the two
    # failures (70 spike events, one 2634-iteration sustained collapse).
    # So "already has roll_reward trained in, just changing the weight" is
    # not evidence resuming is safer for this term -- only training it in
    # from iteration 0 has ever worked cleanly. If this fresh start also
    # trains stably, that's real evidence the weight itself isn't what
    # caused the earlier resume failures (it was resuming specifically);
    # if THIS destabilizes too, that's real evidence 2.0 is simply too
    # aggressive regardless of resume vs fresh, and 1.0 was closer to the
    # ceiling.
    #
    # That fresh start at 2.0 answered the question: it destabilized
    # genuine stepping specifically (tracking/foot_swing_reward plateaued
    # ~0.66-0.70 instead of climbing to the ~1.35 the 1.0 run reached, with
    # general tracking error also markedly worse across almost every
    # joint), even though orientation/roll_deg DID improve further
    # (~4.25deg vs ~5.15deg at weight=1.0). Real trade-off, not worth it --
    # roughly halving genuine stepping quality for another ~1deg of roll.
    # REVERTED 2.0 -> 1.0 on 2026-09-17, back to the confirmed-clean value
    # (only 1 stability spike across its whole 40k-iteration run, best-yet
    # tilt/roll/pitch). Next lever for pushing roll down further is
    # roll_margin_deg, not this weight -- see that field's comment.
    roll_reward_weight: float = 1.0
    # TIGHTENED 6.0 -> 2.0 on 2026-09-16, before ever training with
    # roll_reward_weight=1.0 active (so this is the only change relative to
    # the previous, roll_reward=0 fresh-start baseline -- not bundled with
    # the weight re-enable itself, which was already its own single-
    # variable step). Motivated by two concrete numbers, not just the
    # reference footage looking calm: (1) the roll_reward=0 fresh-start run
    # already measured orientation/roll_deg ~5.86deg -- sitting right at
    # the edge of the OLD 6deg margin, meaning roll_reward at that margin
    # would have given this near-maximum reward already, with almost no
    # gradient pushing it lower (a plausible reason the very first
    # roll_reward attempt, back when it was margin=6/weight=2, only bought
    # a 17% reduction); (2) the REFERENCE motion's own hip-roll joint
    # amplitude (what tracking_reward is already trying to match) is only
    # ~0.5-2.5deg across the gait cycle -- if the intended gait doesn't
    # need much joint-level roll, the body shouldn't need 6deg of slack
    # either. Deliberately NOT touching orientation_margin_deg (still 6.0) --
    # that one also covers pitch and was already proven necessary to stay
    # loose for stepping in general (the 6->2 tightening that broke
    # stepping was on THAT combined field, not this roll-only one), so this
    # is a narrower, lower-risk bet than repeating that mistake. Still a
    # real guess, not a measured reference number -- watch whether this
    # reopens the "safe standing" suppression pattern (foot_swing_reward
    # failing to reach ~1.5 by iteration ~3000) the same way the combined
    # margin did; if so, this specific number was too tight, not the
    # roll-only approach itself.
    #
    # TIGHTENED 2.0 -> 1.0 on 2026-09-17, alongside reverting
    # roll_reward_weight back to 1.0 (see that field's comment -- raising
    # the weight to 2.0 instead hurt genuine stepping too much for the
    # roll improvement it bought). Trying the other lever instead: direct
    # video review showed the body still visibly rolling to shift weight
    # rather than doing that through hip-roll/hip-pitch/knee-pitch/
    # ankle-pitch joint articulation the way reference footage of this
    # robot class does it, with the torso staying flat -- the goal right
    # now is specifically a flatter torso via more joint-driven stepping,
    # not a broader "calm everything down" pass (yaw/heading are a
    # separate, deliberately deprioritized concern for later). Same
    # single-variable discipline as the weight test: only this field
    # changes this run, weight stays at the confirmed-good 1.0. Watch the
    # same signal as every fresh-start test -- foot_swing_reward reaching
    # ~1.5 by iteration ~3000 means stepping wasn't blocked; if it plateaus
    # low like the weight=2.0 attempt did, 1.0deg was too tight for this
    # lever too, and margin/weight together may just have a joint ceiling
    # around the 1.0 run's ~5deg roll for now.
    #
    # REVERTED 1.0 -> 2.0 on 2026-09-18: the 1.0 run's own eval data
    # confirmed the predicted outcome above -- roll_deg did NOT improve
    # over the margin=2.0/weight=1.0 baseline, and tracking got worse on
    # top of that (a strictly worse tradeoff, not just a smaller-than-hoped
    # gain). margin=2.0/weight=1.0 remains the best-evidenced setting; both
    # escalation levers tried so far (weight 1.0->2.0, margin 2.0->1.0)
    # have failed to beat it, suggesting either a real floor for this
    # reward shape or that roll needs to come from elsewhere (e.g. the
    # foot_overswing_margin_m tightening alongside this revert, which
    # targets the excess-lift symptom directly instead of squeezing roll
    # harder).
    roll_margin_deg: float = 2.0

    # --- stay in place -----------------------------------------------------
    # Nothing before this penalized the base for translating in the world --
    # tracking_reward only compares joint angles, orientation_reward only
    # cares about tilt, and the reference clip itself is a stationary
    # step-in-place gait with no forward-motion target at all. Added after
    # the step-in-place gait was working (video: 2026-08-24_22-29-06,
    # iterations 282000/286999) but visibly wandering/drifting across the
    # floor while stepping instead of staying put -- "we don't have a stay
    # in current position reward" was exactly right, there wasn't one.
    #
    # Tracks BASE LINEAR VELOCITY (root_lin_vel_w) against a commanded xy
    # velocity, currently always (0, 0) -- NOT absolute position drift from
    # a fixed per-episode anchor, which was the first version of this term
    # and had two real problems: (1) it's step-in-place specific by
    # construction (a fixed anchor point directly fights any future
    # commanded forward motion, requiring a reward-shape change, not just a
    # retune, to support walking), and (2) it compounds unboundedly over a
    # long episode -- a training rollout's exploration noise continuously
    # perturbing a free-floating base with no learned correction habit yet
    # produces a real random-walk position drift (confirmed: training env 0
    # read 126cm and climbing within one noisy episode, iteration 287145,
    # while the SAME policy's deterministic eval video, a different, noise-
    # free rollout, showed no visible drift at all against the floor grid
    # over the full clip -- not a contradiction, just two different
    # simulations). Velocity error has neither problem: it's a per-step
    # bounded quantity regardless of episode length, and walking later is
    # just changing the commanded velocity from zero to nonzero, no reward
    # redesign. Same exp(-scale * error^2) shape as orientation_reward.
    #
    # UNTUNED starting guess: scale=10 on velocity error (m/s)^2 gives
    # ~0.60 reward at 0.3 m/s of unwanted drift speed and ~0.14 at 0.6 m/s
    # -- cheap for the kind of brief sway that comes from shifting weight
    # onto a stance leg, real cost for sustained walking-off. No margin
    # (unlike orientation) -- stepping in place shouldn't require nonzero
    # average velocity the way lifting a foot requires some lean, so
    # there's no legitimate drift speed to protect.
    #
    # DISABLED (1.0 -> 0.0) on 2026-09-07, as part of a controlled revert-
    # to-last-known-good test. This term, heading_reward_weight below,
    # orientation_margin_deg, orientation_reward_scale, and
    # push_interval_range_s were reconstructed (via qmini_leg_env.py's own
    # dated comment history) to be the full set of changes made SINCE the
    # only run in this project's history with genuine, video-confirmed
    # full-leg stepping (run 2026-08-24_22-29-06, iterations 282000-286999,
    # foot_swing_reward ~0.83-0.85 real left-leg lift -- see
    # foot_swing_left_weight's comment). Every training run since --
    # multiple entropy_coef values, both phase encodings, the BD-X tracking
    # penalty raise, foot_swing_reward_weight escalations, ClampedActorCritic
    # -- has failed to reproduce genuine stepping, and single-variable tests
    # (sin/cos phase revert) already ruled out the phase encoding. This
    # term specifically penalizes the base for ANY translation, which is
    # direct tension with what genuine single-support stepping physically
    # requires (shifting weight off-center) -- added right after the Aug 24
    # run specifically because that run drifted, without ever being tested
    # against whether it makes rediscovering stepping harder. Disabling
    # here, not deleting, so this is a clean, reversible A/B test: if
    # genuine stepping reappears with this whole cluster of "stay still"
    # pressure removed, re-enable this and the others one at a time to find
    # how much stepping can actually tolerate before it's undone again.
    #
    # RE-ENABLED 2026-09-25 (0.0 -> 1.0, scale 10 -> 25), by resume. The
    # 2026-09-07 disable was a controlled test at a time when stepping was
    # not yet reproducible; the real blocker turned out to be
    # tracking_linear_penalty_weight (see the stepping-breakthrough notes),
    # and stepping has been robust since, so the "stay still fights genuine
    # stepping" worry no longer blocks re-trying it. Motivation is hardware:
    # 2026-09-25 rate-limited runs (10 attempts, ~145 cycles) now end almost
    # only because the robot walks off the doormat or into the support
    # frame, not from faults; sim shows the same drift (position/
    # base_speed_cmps 7.7-19 across recent runs, position/reward ~0.67-
    # 0.77). heading_reward/heading_deviation_penalty are already on.
    # Scale raised because at 10 a 0.15 m/s drift still scores exp(-0.225)
    # =0.80 vs 0.98 at 0.05 m/s, a weak gradient; at 25 those are 0.57 vs
    # 0.94. Additive positive reward (max = weight), suspended during push
    # cooldown like before, so a resume is the same kind of change as the
    # earlier weight bumps. UNTUNED: watch foot_swing_reward (should stay
    # ~1.5; if it falls below ~1.3 or fall_rate rises, back scale off to
    # ~15 or weight to 0.5), position/base_speed_cmps (should fall from
    # ~8-19 toward <5), and the per-joint tracking errors. Compare with
    # compare_policies_isaaclab.py and check the hardware drift on the mat.
    #
    # REVERTED 1.0 -> 0.0 on 2026-09-28 (run 2026-09-27_21-20-39,
    # model_357999, ~40000 iterations). RESULT: no effect. position/reward
    # plateaued at 0.55-0.59 within the FIRST 7000 iterations of the resume
    # and never moved from there; position/base_speed_cmps stayed flat at
    # 16-18 cm/s the entire run (was already ~14-17 before this term was
    # even active). compare_policies_isaaclab.py's --dump-heights was
    # extended with root_lin_vel_b (see that flag's comment) to see WHY:
    # base x-velocity averages +23 cm/s while the left foot swings and -20
    # cm/s while the right foot swings (vs -0.3 cm/s in double support) --
    # this term penalizes INSTANTANEOUS speed, and that +-20cm/s swing-
    # phase surge is the weight shift every single-support step physically
    # needs, already present in the reference motion, not a policy mistake.
    # It mostly self-cancels (per-cycle net x displacement across all
    # envs/cycles: mean +1.33cm, i.e. ~0.6cm/s of REAL residual drift, vs a
    # std of 4.21cm from cycle-to-cycle noise) -- so the actual problem
    # this term was meant to fix is a small quantity buried inside a much
    # larger, necessary oscillation the reward can't tell apart from it.
    # To raise position/reward above ~0.58 the policy would have to shrink
    # the swing surge itself, i.e. shrink genuine stepping -- the same
    # can't-distinguish-noise-from-signal trap that killed THREE earlier
    # ankle-shake fixes (gain_randomization_range, ankle damping,
    # action_rate_joint_weight) and the action_smoothing EMA attempt (see
    # that cfg's comment). Don't re-enable this exact shape (instantaneous
    # root_lin_vel_w/b tracked to zero) again without a materially
    # different mechanism -- e.g. tracking NET displacement over a window
    # (a gait cycle or an EMA-filtered velocity) instead of per-step speed,
    # which would be blind to the swing's own cancelling oscillation and
    # should only see the ~1cm/cycle residual that's the actual target.
    # Not yet implemented. Hardware corroborates the "no benefit" read:
    # 2026-09-28 attempts 5-8 on this exported bundle
    # (deploy_bundle_2026-09-28_position1) all aborted on IMU faults, none
    # ran long enough to compare drift distance against the pre-change
    # pitchstep5 bundle.
    position_reward_weight: float = 0.0
    position_reward_scale: float = 25.0

    # --- stay facing the same way -------------------------------------
    # Same gap as position_reward closed for translation, but for
    # rotation: orientation_reward only checks projected_gravity_b's xy
    # components, i.e. whether the base is LEVEL -- it says nothing about
    # which way the base is FACING. position_reward only checks linear
    # velocity -- nothing about rotation either. So nothing currently
    # penalizes the base slowly spinning in place while staying level and
    # not translating; a robot that rotates 360 deg over an episode while
    # holding position scores identically to one that holds its heading.
    # Added after exactly that showed up on video ("turning around more")
    # once foot_swing_reward_weight was raised (5.0) enough to make genuine
    # stepping worth pursuing again -- more real stepping meant more
    # opportunity for uncorrected yaw drift to accumulate, since nothing
    # was pulling it back.
    #
    # Same shape and same reasoning as position_reward: track base YAW
    # ANGULAR VELOCITY (root_ang_vel_w's z component) against a commanded
    # rate, currently always 0 (stay in place, don't turn) -- not absolute
    # heading against a fixed per-episode anchor, for the same reasons
    # position_reward moved away from that (bounded per-step regardless of
    # episode length, and turning later is just changing the commanded
    # rate from zero to nonzero, no reward redesign). Analogous to BD-X's
    # "Torso orientation" term (Table I: exp(-20*||theta ⊖ theta_hat||^2)),
    # which tracks full base orientation against a reference -- this is the
    # rotational-velocity version of the same idea, matching how
    # position_reward is the translational-velocity version of their
    # "Torso position xy" term.
    #
    # UNTUNED starting guess: scale=5.0 on yaw-rate error (rad/s)^2 gives
    # ~0.64 reward at ~17deg/s of unwanted turning and ~0.17 at ~34deg/s --
    # cheap for brief yaw wobble from footfall impacts, real cost for
    # sustained spinning.
    #
    # weight RAISED 1.0 -> 2.5 (2026-08-31). Unlike orientation_reward_scale
    # (which really was too gentle a curve), the curve here already checks
    # out: at the yaw rates actually observed (~50-65deg/s, per
    # heading/yaw_rate_dps), scale=5.0 already collapses this term to
    # ~0.02-0.002 -- essentially zero, not a soft plateau. So the problem
    # isn't the gradient, it's that the maximum this term can ever cost is
    # capped at weight=1.0, against a total reward regularly running
    # 9-11+ -- even collecting nothing here is a small fraction of what's
    # available elsewhere, so a policy that gains more than ~1 point by
    # tolerating yaw drift (however that happens) has no reason not to.
    # 2.5 makes that ceiling a meaningfully bigger slice of total reward
    # without dominating it. Watch whether yaw_rate_dps actually drops --
    # if it barely moves even with more weight on an already-steep curve,
    # that's evidence the drift isn't a reward trade-off the policy is
    # choosing at all (more likely a passive side-effect of the swing
    # motion imparting net reaction torque) and weight/scale tuning on
    # this term won't fix it either.
    # DISABLED (2.5 -> 0.0) on 2026-09-07, same revert-to-last-known-good
    # test as position_reward_weight above -- see that field's comment for
    # the full reasoning. This term didn't exist at all during the last
    # genuinely-confirmed-stepping run (added 2026-08-31).
    #
    # RE-ENABLED (0.0 -> 2.5) on 2026-09-08, resuming from
    # 2026-09-07_22-00-51/model_39999.pt -- that run, with
    # tracking_linear_penalty_weight reverted 1.0 -> 0.2 (see that field's
    # comment), finally produced real, video-confirmed, sustained genuine
    # stepping (foot_swing_reward steady ~1.6-1.65 the whole run, not a
    # spike; heel/toe clearance actually positive; close-up frame review
    # showed real alternating single-support swing/stance cycling). So
    # tracking_linear_penalty_weight, not the phase encoding or the
    # position/heading/push cluster, was the actual blocker all along. But
    # that run's gait is aggressive and falls more than it should
    # (heading/yaw_rate_dps ~100-127, way above anything seen in a healthy
    # run before; mean_episode_length only ~250-260 despite
    # orientation/fall_rate reading a "low" ~0.003 -- misleading at a
    # glance, but that's a PER-STEP rate, and compounded over up to 499
    # steps/episode it implies most episodes are in fact ending in a fall,
    # matching what video shows). This term exists specifically to penalize
    # uncommanded yaw spinning (see its addition above, originally added for
    # the exact same "turning around more" symptom once real stepping made
    # it possible) and does NOT touch tracking_linear_penalty_weight, so
    # re-enabling it alone is a single-variable test for whether it calms
    # the spin/fall rate down without undoing the stepping breakthrough.
    # Resuming, not restarting fresh -- obs/action shapes are unchanged, and
    # a resume preserves the actual learned stepping skill this run just
    # found rather than asking a from-scratch run to rediscover it.
    heading_reward_weight: float = 2.5
    heading_reward_scale: float = 5.0

    # --- stay facing the same way, part 2: ABSOLUTE heading -----------
    # Added 2026-09-11 after real-hardware attempts (2026-09-11, sin/cos
    # checkpoint with the target_limit_penalty fixes) diverged into a
    # visible, escalating body rotation and safety-aborted, even though
    # sim's own compare_policies_isaaclab.py eval on the exact same
    # checkpoint showed 0% fall_rate. Root cause, pieced together from
    # real control_loop CSVs: heading_reward above (and everything else in
    # this file) only ever tracks yaw RATE -- nothing anywhere tracks or
    # restores absolute heading. In sim that's tolerable (left/right
    # dynamics are symmetric by construction, episodes just keep running
    # regardless of net rotation, no hard trip-wire). Real hardware isn't
    # symmetric: replaying the reference open-loop (--open-loop-ref, no
    # policy at all, control_loop_20260829_144353.csv /
    # control_loop_20260829_150136.csv) showed a small but completely
    # consistent real gyro bias (+0.78 and +1.28 deg/s in two separate
    # recordings, same sign both times -- integrating to +85deg and +78deg
    # of real drift over 110s/61s) that has NOTHING to do with the policy.
    # On top of that, the policy's OWN gait already produces much larger
    # yaw rate from genuine stepping reaction torque (attempt 2's first 15
    # real control steps: already +8 to +17 deg/s, matching sim's own
    # long-standing heading/yaw_rate_dps ~47-70 deg/s that no amount of
    # raising heading_reward_weight ever fixed -- see that field's
    # comment). Neither of these is corrected by a rate-only reward; they
    # just integrate. On real hardware that integration compounds with the
    # confirmed physical bias and, per the control_loop CSVs, correlates
    # with the broader divergence (hip roll, then everything else) that
    # eventually trips the safety abort.
    #
    # This term gives the policy an actual restoring incentive: track
    # ACCUMULATED yaw deviation from this env's own reset-time heading
    # (self._reset_yaw, captured in _reset_idx), not just its rate.
    # Margined (heading_deviation_margin_deg) so ordinary within-stride yaw
    # wobble from genuine stepping isn't penalized, only sustained drift
    # beyond it. LINEAR, not quadratic -- same non-vanishing-gradient
    # reasoning as target_limit_penalty_linear_coef's fix: a quadratic
    # penalty barely notices the first several degrees of real drift,
    # exactly the region that matters here given how fast this compounds
    # on hardware (attempt 2 went from ~0 to unrecoverable in ~1.2s).
    # Capped before scaling, same reasoning as
    # target_limit_penalty_max_overshoot_deg and position_reward_weight's
    # own move away from an unbounded absolute-anchor penalty (a single
    # bad episode's random-walk drift could otherwise blow up the critic
    # the same way an uncapped target overshoot once did). UNTUNED
    # starting guess for both weight and margin -- watch
    # heading/deviation_deg in tensorboard (population value, unlike the
    # env-0-only gravity_x/y readings) and whether real hardware's net
    # rotation over ~1-2s actually shrinks; also watch that this doesn't
    # fight genuine turning later if a nonzero commanded yaw rate is ever
    # added (the margin would need to move with the command at that
    # point, not just wrap around a fixed reset heading).
    #
    # DISABLED (1.0 -> 0.0) on 2026-10-02 for a fresh start, as part of a
    # broader pivot after THREE consecutive fresh-start attempts all hit
    # the same collapse (foot_swing_reward climbs to a real peak around
    # iteration ~3000 then crashes to ~0 and stays there) regardless of
    # which ONE term was toggled (target_limit_margin_deg roll 13 vs 3,
    # position_deviation_penalty_weight 1.0 vs 0.0). Checked what the
    # ORIGINAL successful fresh start (2026-09-08, see the stepping-
    # breakthrough notes) actually had: tracking_linear_penalty_weight=0.2,
    # heading_reward_weight=2.5, push/position disabled -- NO
    # target_limit_penalty, step_limit_penalty, heading_deviation_penalty,
    # or position_deviation_penalty at all. Every one of those was added
    # LATER and has only ever been validated via resume onto an
    # already-stepping policy, never through fresh-start discovery --
    # disabling one at a time didn't find a single culprit, so trying all
    # four at once (this field, target_limit_penalty_weight,
    # step_limit_penalty_weight, and position_deviation_penalty_weight
    # already at 0) to see if that reproduces reliable, HELD stepping like
    # the original recipe did. If it does, reintroduce each one at a time
    # via resume exactly as already validated individually in the past.
    # RE-ENABLED (0.0 -> 1.0) on 2026-10-03, resuming from the
    # 2026-10-02_22-08-07 fresh start (model_39999, foot_swing_reward held
    # ~1.2-1.3 through all 40000 iterations with all four post-breakthrough
    # terms off -- see position_deviation_penalty_weight/target_limit_
    # penalty_weight/step_limit_penalty_weight for the matching re-enables
    # and the full fresh-start saga). Back to its original validated
    # value, same as every prior successful reintroduction of this term.
    heading_deviation_penalty_weight: float = 1.0
    heading_deviation_margin_deg: float = 15.0
    heading_deviation_penalty_max_deg: float = 60.0

    # --- stay in place, part 2: ABSOLUTE position (same fix as heading's
    # part 2, applied to translation) -----------------------------------
    # Added 2026-09-29. position_reward above only ever tracked base
    # VELOCITY -- same rate-only gap heading_deviation_penalty was built
    # to close for yaw (see that field's comment: "heading_reward above
    # only ever tracks yaw RATE -- nothing anywhere tracks or restores
    # absolute heading"). Identical problem existed for position and was
    # never fixed: position_reward_weight raised 0.0->1.0 on 2026-09-25
    # (hardware showed 2026-09-25 rate-limited runs now ending almost only
    # because the robot walks off the mat/into the support frame, not from
    # faults) had NO effect after ~40000 iterations -- position/reward
    # plateaued at 0.55-0.59 within the first 7000 and never moved.
    # compare_policies_isaaclab.py's --dump-heights was extended with
    # root_lin_vel_b to find out why: base x-velocity averages +23cm/s
    # while the left foot swings and -20cm/s while the right foot swings
    # (vs -0.3cm/s in double support) -- a large, mostly self-cancelling
    # swing-phase surge the reference motion itself requires, not a
    # mistake. Per-cycle net x displacement across all envs/cycles: mean
    # +1.33cm (~0.6cm/s of REAL residual drift), std 4.21cm of cycle-to-
    # cycle noise -- a per-step VELOCITY reward pays for the full +-20cm/s
    # surge every step even though ~95% of it cancels, so it can't resolve
    # the much smaller residual that's the actual problem. See
    # position_reward_weight's own comment for the full writeup (reverted
    # back to 0.0 there).
    #
    # This term is the position equivalent of heading_deviation_penalty:
    # tracks ACCUMULATED xy displacement from this env's own reset-time
    # position (self._reset_pos_xy, captured in _reset_idx), not velocity
    # -- structurally blind to a within-cycle oscillation that returns
    # close to its start point every ~1.1s, and only sees genuine
    # sustained drift. Margined/linear/capped, same shape and same
    # reasoning as heading_deviation_penalty (non-vanishing gradient from
    # the first cm over the margin; capped so a bad early-training
    # episode's random-walk drift can't blow up the value function the
    # same way an uncapped target overshoot once did -- this is exactly
    # the failure mode position_reward's OWN original anchor-based version
    # hit before it was changed to velocity in the first place, i.e. the
    # mechanism this field's history warns about; the cap is what's
    # supposed to prevent a repeat).
    #
    # UNTUNED starting guess: margin=8cm is roughly one cycle's typical
    # residual (mean 1.33cm, but real per-cycle spread std=4.21cm, so 8cm
    # leaves room for ordinary cycle-to-cycle sway without penalizing it);
    # max=40cm (5x margin) bounds the penalty well before it could
    # dominate; weight=1.0 matches heading_deviation_penalty_weight's
    # starting point. Resume, not fresh start -- new additive term, same
    # category of change as heading_deviation_penalty's own introduction.
    # Watch position/deviation_cm in tensorboard (population value) and,
    # more importantly, actual hardware run duration/how far it walks
    # before hitting the mat edge or a wall -- that's what this term is
    # ultimately trying to fix and the sim metric is a proxy for it.
    #
    # DISABLED (1.0 -> 0.0) on 2026-10-02 for a fresh start specifically.
    # Like this run was originally validated only via resume (see comment
    # above), never through fresh-start discovery. Two consecutive fresh
    # starts (target_limit_margin_deg roll=13, then roll=3 with this term
    # still at weight=1.0) BOTH showed the same signature: foot_swing_
    # reward climbs to a genuinely good peak (0.47, then 1.5 -- actually
    # above the normal ~0.7-1.3 healthy-fresh-start benchmark the SECOND
    # time) around iteration ~3000, then collapses to ~0 within a few
    # hundred iterations and stays there for 4000+ iterations after.
    # fall_rate does NOT spike back up at the collapse point (it's already
    # near 0 and stays there right through it) -- ruling out "it kept
    # falling while trying to step", and pointing at "standing still
    # became relatively safer" instead. Since reverting the roll margin
    # (13->3, matching the ORIGINAL successful fresh-start recipe exactly)
    # did NOT fix this -- if anything the peak got higher before still
    # collapsing -- the margin was likely not the (sole) cause, which
    # reopens the question of what else differs from that original
    # recipe. This term is the one other always-on, per-step, margined
    # penalty added since (2026-09-29) that has never been tested through
    # fresh-start discovery either, and it has a plausible mechanism here:
    # real stepping causes genuine transient base displacement (measured
    # ~20cm/s swing-phase surges elsewhere in this project), so as the
    # policy was first discovering real lift around iteration 3000, it may
    # have started crossing the 8cm margin often enough that retreating to
    # minimal motion looked like the better trade -- timing matches.
    # UNCONFIRMED, same caveat as the margin hypothesis before it. If
    # disabling this ALSO doesn't fix it, the next suspects are
    # heading_deviation_penalty (same category, never fresh-start-tested
    # either) or something not yet identified. Re-enable via resume once
    # this fresh start finds genuine, HELD stepping -- same pattern as
    # every other tightening in this project's history.
    # RE-ENABLED (0.0 -> 1.0) on 2026-10-03, resuming from the
    # 2026-10-02_22-08-07 fresh start together with heading_deviation_
    # penalty_weight/target_limit_penalty_weight/step_limit_penalty_weight
    # -- all four reintroduced in one resume rather than one at a time,
    # since each was already individually validated via resume before and
    # this checkpoint's noise_std never collapsed (still ~0.48-0.5 after
    # 40000 iterations, unlike the low-entropy pitchstep5 lineage), so it
    # should have real room left to adapt rather than being brittle.
    position_deviation_penalty_weight: float = 1.0
    position_deviation_margin_m: float = 0.08
    position_deviation_penalty_max_m: float = 0.40

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
    # UNCAPPED IN DIRECTION (unlike joint_limit_penalty, which saturates
    # near the physical position boundary) specifically because a real
    # safety abort doesn't care how far over the limit the target was, but
    # a bigger overshoot in training is a stronger signal something is
    # systematically wrong, not just marginal, so the gradient should
    # scale with it.
    #
    # RAISED 2.0 -> 5.0 (2026-08-28), alongside adding
    # target_limit_margin_deg below -- both part of the same fix for a
    # real hardware deploy failure (robot_deploy.py safety-aborted on
    # right_hip_roll/left_knee targets well beyond their hard limits).
    # This penalty existed the whole time that happened, but was
    # quantitatively negligible: a real 3.45deg overshoot from that deploy
    # log costs weight(2.0)*radians(3.45)^2 ~= 0.007 reward, against a
    # total per-step reward now regularly 9-11 (tracking ~6.5, foot_swing
    # ~1.3, orientation*weight ~2.9, position ~0.8, heading ~0.65) --
    # completely invisible to the policy, and quadratic-near-zero means it
    # STAYS invisible for exactly the small overshoots that matter here (a
    # real safety abort has zero tolerance regardless of size, unlike this
    # penalty's vanishing gradient near the boundary). Also worth having
    # simply because total reward scale has grown ~2.5-3x since this
    # weight was first tuned (foot_swing_reward_weight 3->5,
    # heading_reward added) -- the same physical overshoot matters even
    # less now than when 2.0 was chosen, independent of the margin fix.
    # RAISED AGAIN 5.0 -> 15.0 on 2026-09-10. compare_policies_isaaclab.py
    # against the 2026-09-08 sin/cos checkpoint (before the
    # action_rate_penalty raise) and its resumed successor (after) showed
    # target_violation_rate barely moved (10.99% -> 9.14%) despite this
    # weight already having been raised once for exactly this reason --
    # see the 2.0->5.0 history above. Same root cause recurring: total
    # reward scale has grown further since 5.0 was chosen
    # (foot_swing_reward_weight, heading_reward re-enabling, etc.), and this
    # term is QUADRATIC near the boundary, so its gradient vanishes for
    # precisely the small-but-real overshoots a real hard limit has zero
    # tolerance for -- raising the weight partially compensates but doesn't
    # fix the underlying near-zero-gradient shape, worth remembering if
    # 15.0 still doesn't move target_violation_rate meaningfully (next
    # lever would be the shape itself, e.g. a linear-near-zero term, not
    # another weight increase). Re-verify with compare_policies_isaaclab.py
    # (one checkpoint per invocation -- see that script's multi-checkpoint-
    # session caveat) before any further real hardware attempt.
    #
    # REVERTED 15.0 -> 5.0 on 2026-09-16, its value at the 2026-08-28
    # raise, i.e. what was active through both 2026-09-07/08 fresh starts
    # that found genuine stepping from scratch. Two fresh starts since
    # (with this at 15.0 and the linear-coef shape added, see
    # target_limit_penalty_linear_coef's comment) found NO stepping at all
    # -- dug into the second failure's raw per-iteration tensorboard data
    # rather than guessing further (see action_rate_penalty_weight's
    # comment for the paired finding): right as foot_swing_reward briefly
    # rose during a transient rediscovery, THIS term grew ~150x
    # (0.01->1.5) in lockstep, before any fall/episode-length crash --
    # i.e. genuine stepping's larger joint excursions were being hit hard
    # and immediately, not just failing policies. This is the term that
    # actually matters most for real-hardware safety (target_violation_rate
    # went from ~11% to ~0.01% because of the 5.0->15.0 raise + the linear
    # shape), so don't just leave it at 5.0 once stepping is back --
    # re-harden it gradually via RESUME on top of the re-established
    # stepping skill, the same escalation that worked the first time,
    # checking compare_policies_isaaclab.py's target_violation_rate AND
    # tracking/foot_swing_reward at each step rather than jumping straight
    # back to 15.0.
    #
    # RE-HARDENING STARTED 2026-09-21: 5.0 -> 15.0 (stage 1, paired with
    # target_limit_penalty_linear_coef 0.0 -> 0.5), via RESUME from
    # 2026-09-20_20-50-50/model_39999.pt (the fresh start trained under
    # the current dynamic reference-foot-height reward). Trigger: first
    # hardware runs with correctly normalized policies (2026-09-21, see
    # deploy_bundle_2026-09-21_overswing_w25) all aborted at gait phase
    # t~1.0s. Replaying the logged observations through the policy showed
    # the abort is the policy's own phase-locked feed-forward, not a
    # sim2real input mismatch: with every sensor input replaced by its
    # training mean the left hip-roll target still ramped 0 -> -16deg in 6
    # steps (limit +-15), and a full-cycle sweep shows hip-roll kicks to
    # -24.5deg (t~1.1s) and +29deg (t~2.1s) against a reference of +-2.7deg
    # -- ~12% of the cycle has a target beyond a joint limit, matching the
    # sim eval's ~9% target_violation_rate. Sim just saturates the joint at
    # the limit, so this was nearly free; hardware's safety check aborts.
    # Escalate in the same stages that worked before (weight 15 + linear
    # 0.5, then linear 1.0), checking target_violation_rate AND
    # foot_swing_reward after each; do NOT jump straight to 15 + 1.0.
    #
    # DISABLED (15.0 -> 0.0) on 2026-10-02 for a fresh start -- see
    # heading_deviation_penalty_weight's comment for the full reasoning
    # (one of four post-breakthrough penalty terms disabled together,
    # none of which existed in the original 2026-09-08 successful fresh
    # start, after toggling them one at a time across three attempts
    # failed to find a single culprit for the same foot_swing_reward
    # spike-then-collapse pattern). Re-enable via resume, same
    # weight=15/linear_coef=0.5 staged escalation this comment already
    # describes, once this fresh start finds genuine held stepping.
    # RE-ENABLED (0.0 -> 15.0) on 2026-10-03, resuming from the
    # 2026-10-02_22-08-07 fresh start, same batch of four re-enables as
    # heading_deviation_penalty_weight's comment describes. Back to the
    # ORIGINAL weight=15/linear_coef=0.5 combo (not the later roll-margin
    # widening saga's 8/11/13 -- target_limit_margin_deg is back at a
    # uniform 3.0 for every joint, see that field's own comment) --
    # deliberately not re-litigating the roll-margin question in the same
    # resume as everything else; revisit that separately once this new
    # lineage has a genuinely good baseline to diagnose from.
    target_limit_penalty_weight: float = 15.0

    # Degrees of buffer inside each hard limit before target_limit_penalty
    # starts applying at all -- same idea as joint_limit_margin above, just
    # never previously given to this term. Added alongside the weight
    # raise above: without a margin, this penalty is exactly zero right up
    # until a target actually crosses the hard limit, so there was NO
    # gradient at all pushing the policy to stay clear of the boundary
    # with any buffer -- only a weak one once already over it. A real
    # safety envelope needs "rarely gets close", not just "doesn't
    # overshoot by much when it does". UNTUNED starting guess: 3deg is
    # small relative to hip_roll's +-15deg range (the joint that actually
    # tripped) without meaningfully constraining legitimate motion --
    # the reference clip's own roll range is under 3deg peak-to-peak
    # already, well inside a target range that keeps this margin
    # untouched.
    #
    # CONVERTED to a per-joint-TYPE dict on 2026-09-23 (same matching
    # convention as joint_tracking_weight/action_rate_joint_weight), roll
    # widened 3.0 -> 8.0, everything else left at the original 3.0.
    # Motivated by hardware, not sim: at weight=15/linear_coef=0.5 this
    # term already drove sim's target_violation_rate down to ~0.0016%, yet
    # 2026-09-22/23 battery-powered hardware testing (comms confound
    # resolved, see imu_fault_diagnosis notes) hit a real "target outside
    # joint limits: left_hip_roll" abort on 4 of 7 runs across BOTH
    # roll6_ankle3 and yaw3 -- always the same joint, always within
    # motion_time 1.02-1.05s of each other, always grazing by only
    # 0.1-0.6deg (e.g. -15.10, -15.29, -15.31, -15.56 against the -15.00
    # limit). A uniform 3deg margin is a much smaller buffer, in absolute
    # terms, for a joint with a +-15deg range and a ~2.7deg reference
    # amplitude than for e.g. knee (+-50/60deg range, ~16-30deg
    # amplitude) -- the same margin doesn't mean the same safety headroom
    # per joint. Raising ONLY roll's margin (not the global weight/
    # linear_coef, which would also squeeze knee/pitch's much larger
    # legitimate excursions) targets the joint that's actually tripping
    # without repeating the "broad tracking tightening suppresses genuine
    # stepping" mistake from earlier in this project. 8deg leaves the
    # policy 2-3x the reference's own roll amplitude before the penalty
    # even starts, well short of the limit at +-15deg. Resume, not fresh
    # start -- reshapes an existing term's margin like joint_tracking_
    # weight's roll/ankle/yaw increases did, not a new penalty term.
    # 2026-09-24: roll widened again 8.0 -> 11.0. Offline replay of the
    # hardware logs through the resume_roll_margin8 policy still put the
    # left hip-roll target at -12.2..-13.2deg around motion_time 1.0s
    # (yaw3: -14.8..-15.6), i.e. 2-3deg from the +-15 limit and well inside
    # the 7deg zone this margin opened -- helped, not solved. 11deg starts
    # the penalty at +-4deg, still above the ~2.7deg reference amplitude.
    # At 11deg this held up well: pitchstep5 (the checkpoint built on this
    # margin) went 0/10+ on roll-limit aborts across every 2026-09-25 and
    # 2026-09-28/29 hardware batch, target consistently -8..-13deg.
    #
    # 2026-09-30: widened again 11.0 -> 13.0, NOT because pitchstep5 itself
    # showed a roll problem, but as a hedge/backstop before resuming
    # further work on it. deploy_bundle_2026-09-30_support_resume (the
    # startup_support-trained checkpoint built on TOP of this same 11deg
    # margin) failed 4/4 real hardware attempts with left/right_hip_roll
    # violations up to -20.87deg -- a genuine policy regression, confirmed
    # by replaying those same real observations through pitchstep5 (stayed
    # inside +-15deg in 3/4 cases on the identical inputs), invisible to
    # every sim metric checked beforehand (target_violation_rate was
    # IDENTICAL between the two checkpoints, roll tracking error looked
    # BETTER on the checkpoint that then failed on hardware). That result
    # means margin=11 is not necessarily enough buffer once a future
    # change (retrying startup_support, or anything else) puts renewed
    # pressure on this specific joint -- widening now, on the known-good
    # pitchstep5 baseline, is insurance against a repeat, not a fix for a
    # currently-observed failure. 13deg starts the penalty at +-2deg,
    # getting close to the reference's own ~2.7deg roll amplitude -- this
    # is deliberately NOT pushed further than that in this step (e.g. to
    # 14-15deg) specifically to avoid the over-constraining mistake this
    # field's own history warns about elsewhere (broad tracking tightening
    # suppressing genuine stepping). Resume, not fresh start, from
    # pitchstep5 (NOT support_resume) -- same category of change as every
    # prior margin widening.
    #
    # REVERTED roll 13.0 -> 3.0 on 2026-10-02 for a fresh start specifically
    # (matching every other joint -- the value the ORIGINAL successful
    # fresh start in this project used, before this dict's per-joint
    # overrides existed at all). Two consecutive fresh-start attempts at
    # step_limit_penalty_weight=20 (margin=13 active both times) failed --
    # first catastrophically (couldn't balance at all, see that field's
    # own comment), then after reverting the weight to 10, a SECOND,
    # more familiar failure: real balance was found (episode_length ->
    # ~450, fall_rate -> 0 by iteration ~3000) but foot_swing_reward
    # spiked then crashed to ~0 and stayed there 5000+ iterations
    # (iteration 8178 check: still 0.0092) -- the classic standing-still
    # local optimum. step_limit_penalty_weight was already back at its
    # only-ever-validated value, leaving margin=13 as the one remaining
    # variable different from every PRIOR successful fresh start in this
    # project's history. At margin=13 the penalty starts at +-2deg,
    # INSIDE the reference's own ~2.7deg normal roll swing -- i.e.
    # ordinary, necessary roll motion was incurring some penalty from
    # iteration 0, before the policy had any chance to learn useful roll
    # control. Every margin value above 3.0 (8, 11, 13) has, without
    # exception, only ever been introduced via RESUME onto an
    # already-stepping policy -- never through a fresh start -- matching
    # this project's own repeated lesson (tracking_linear_penalty_weight's
    # history, push_interval_range_s's history, startup_support's
    # history): a broad-enough constraint active from iteration 0 can
    # make "don't move much" look safer than ever discovering the real
    # skill, regardless of how large the reward for that skill is. Fresh
    # start again with roll=3.0 (all joints equal, matching the original
    # recipe); once this one finds genuine stepping, reintroduce roll=13
    # the same way every other tightening in this project has actually
    # worked -- as a resume on top of a confirmed-stepping checkpoint, not
    # from iteration 0 again.
    target_limit_margin_deg: dict = {
        "yaw": 3.0,
        "roll": 13.0,
        "pitch": 3.0,
        "knee": 3.0,
        "ankle": 3.0,
    }

    # The overshoot itself IS capped here (in degrees, before squaring) --
    # added after a 40000-iteration run diverged catastrophically at the
    # very end (value_function_loss hit 1.7 BILLION, mean_reward crashed to
    # -28385, this term alone averaged 818/step). A single oversized raw
    # action (nothing bounded one before this -- see clip_actions in
    # rsl_rl_qmini_leg_ppo_cfg.py, the OTHER half of this same fix) fed a
    # huge overshoot into an unbounded square, producing a reward magnitude
    # that dwarfed every other term and blew up the critic. Capping the
    # overshoot means even a wildly-out-of-range target still produces a
    # LARGE, clearly-bad, but FINITE penalty -- still exactly as strong a
    # "never do this" signal as before for any realistic overshoot, just
    # with a numerical ceiling so one outlier step can't destabilize the
    # whole run. 30 deg is generous headroom above any overshoot a
    # correctly-behaving policy should ever produce.
    target_limit_penalty_max_overshoot_deg: float = 30.0

    # Coefficient on a LINEAR overshoot term added alongside the quadratic
    # one (final penalty per joint = weight * (over**2 + linear_coef *
    # over), over in radians, both margined and capped as above). Added
    # 2026-09-10. Two rounds of raising target_limit_penalty_weight
    # (2.0->5.0->15.0) only got target_violation_rate from ~11% down to
    # ~7% (see experiments/qmini-leg/compare_action_smoothness.csv) --
    # diminishing returns, for the structural reason this field's own
    # neighbor comments already spell out: a pure quadratic has a gradient
    # that VANISHES as the overshoot approaches zero, i.e. it gives almost
    # no push to close out precisely the small 1-3deg overshoots that a
    # real hardware safety check still refuses outright (it has zero
    # tolerance regardless of magnitude). A linear term has a constant,
    # non-vanishing gradient all the way down to zero overshoot, so it
    # keeps pushing the policy fully inside the envelope rather than just
    # "not far past it". Kept the quadratic too -- it still does the useful
    # job of scaling the signal up for large overshoots ("this is very
    # wrong, not just marginally"). 0.5 chosen so that at a ~2deg overshoot
    # past the margin the linear part contributes ~15x the quadratic part
    # (making it the dominant near-boundary signal) while at the 30deg cap
    # the two are the same order (quadratic still meaningfully present).
    # UNTUNED starting guess -- if genuine stepping quality regresses
    # (foot_swing_reward drops, tracking degrades), lower this or
    # target_limit_penalty_weight; if target_violation_rate still doesn't
    # approach zero, raise it. Re-verify with compare_policies_isaaclab.py
    # (one checkpoint per invocation) after the run.
    #
    # RAISED 0.5 -> 1.0 (2026-09-11). The 0.5 run worked well:
    # target_violation_rate 7.13% -> 5.33% and mean_overshoot_deg 4.26 ->
    # 1.75 (now BELOW target_limit_margin_deg=3.0, i.e. the typical
    # overshoot sits inside the safety buffer rather than past the hard
    # limit), with mean_tilt_deg creep also reversed by the paired
    # orientation_reward_scale bump. Still ~1-in-19 commanded targets would
    # be refused by real hardware's safety check though -- probably enough
    # to still trip aborts. Doubling the linear coefficient to push the
    # violation rate toward ~2-3%, per the trajectory
    # (10.99->9.14->7.13->5.33%). Watch foot_swing_reward: it dipped
    # slightly under 0.5 (~1.96 -> ~1.83 smoothed) as constraint pressure
    # rose -- still clearly genuine stepping on video, but if it keeps
    # eroding under 1.0, that's the signal this lever has gone far enough
    # and the remaining violation rate needs a different fix (e.g. reducing
    # action_scale so the raw action range can't reach as far past a limit
    # in the first place). Resume from model_159996.pt.
    #
    # DISABLED (1.0 -> 0.0) on 2026-09-16, alongside target_limit_penalty_weight's
    # revert to 5.0 -- see that field's comment. This whole linear term
    # didn't exist yet during the 2026-09-07/08 fresh starts that found
    # genuine stepping from scratch (pure quadratic then, weight 5.0); 0.0
    # here reproduces that exact shape. Re-introduce this the same way as
    # the weight -- gradually, via resume on top of re-established
    # stepping, not from a fresh start.
    #
    # RE-INTRODUCED at 0.5 on 2026-09-21 (stage 1 of the re-hardening --
    # see target_limit_penalty_weight's comment for why). Raise to 1.0 as
    # stage 2 if target_violation_rate is still well above ~1% and
    # foot_swing_reward held.
    target_limit_penalty_linear_coef: float = 0.5

    # Penalty on the per-step CHANGE of the commanded joint target beyond a
    # threshold (degrees). Added 2026-09-22. robot_deploy.py aborts the
    # whole run if any joint's commanded target moves more than
    # max_step_deg (robot_config: 5.0 for step-in-place, 15.0 when relaxed)
    # between consecutive control steps; sim has no such check, and
    # action_rate_penalty is quadratic on RAW action deltas (tiny for the
    # 10-25deg single-step "snaps" seen on hardware-replay). Found by
    # sweeping the phase input of the 2026-09-21_12-52-39 policy with
    # neutral sensor inputs (matches the logged hardware behavior at the
    # one phase hardware confirmed): right knee extends slowly to +11.5deg
    # then snaps -24deg in ONE 20ms step at t~1.2s; other joints reach
    # 5-6deg steps. Computed on the raw pre-delay action exactly like
    # action_rate_penalty, in target degrees = degrees(action_scale *
    # |a_t - a_{t-1}|), which is what the deploy check measures. The first
    # step after reset is deliberately INCLUDED (prev action is 0, so the
    # delta is the first target's distance from the default pose): resets
    # always start at motion_time=0 like hardware, and hardware's first
    # commanded step is the same quantity (first hardware logs had 8-9deg
    # first-step jumps). LINEAR in the excess over the threshold (constant
    # gradient, same reasoning as target_limit_penalty_linear_coef) and
    # capped (a single outlier once blew up the value function).
    # threshold 4.0 sits under the 5.0 deploy limit as a safety buffer.
    # Weight 10 (per rad of excess). Measured on the 2026-09-21_12-52-39
    # policy with compare_policies_isaaclab.py (deterministic eval, obs
    # noise + pushes on): 18.3% of env-steps have SOME joint step >5deg and
    # the p99 of the per-step max is 38.6deg -- much more than the neutral-
    # sensor phase sweep showed, i.e. sim obs noise/pushes add jitter on
    # top of the feed-forward snap. At weight 20 that policy would pay
    # 0.56/step on average (p99 12.6, max 33.8) against ~8/step total
    # reward -- judged too big a first shock for a resume (see the
    # foot_overswing_penalty_weight 40/70 history), so 10: ~0.28/step
    # mean, p99 ~6, max ~17, while a 20deg snap still costs ~3.5 (about
    # 40% of a step's reward). Meant as a nudge the feed-forward policy can
    # satisfy by smoothing the snap, not by giving up stepping. UNTUNED:
    # watch foot_swing_reward, and tracking/step_over_rate /
    # mean_max_step_deg (should fall). If they barely move, raise to ~20.
    #
    # RAISED 10.0 -> 20.0 on 2026-10-01, following exactly that pre-
    # committed plan -- p99_step_deg didn't barely move, it nearly doubled
    # across essentially every resume since this field was introduced
    # (resume_roll6_ankle3 6.18 -> yaw3 6.77 -> roll_margin8 7.60 ->
    # roll11_pitchstep2 8.61 -> pitchstep5 9.42 -> position_deviation
    # 10.01 -> support_resume 9.89 -> target_limit_margin_deg roll 13
    # 11.46), regardless of what the specific change was. User's own
    # observation, 2026-10-01: every change so far has made hardware
    # behavior worse alongside this number climbing -- matches
    # support_resume's real 4/4 hardware failure and the roll-margin-13
    # checkpoint's own worse-than-pitchstep5 replay result, both at the
    # high end of this trend. Resume from pitchstep5 (not rollmargin13 or
    # support_resume -- avoid stacking on an already-once-resumed,
    # not-yet-hardware-validated checkpoint), with target_limit_margin_deg
    # roll=13 (see that field's own 2026-09-30 comment) ALSO still active
    # -- not a clean single-variable test against the margin change, but
    # training runs are ~10-13h each and both changes are independently
    # well-motivated, so testing them together here is a deliberate
    # tradeoff, not an oversight. Watch p99_step_deg specifically (should
    # fall back toward ~9 or lower) and re-run the same 4-attempt
    # open-loop replay this whole investigation has been using before
    # trusting any dashboard recovery.
    #
    # REVERTED 20.0 -> 10.0 on 2026-10-02. The steplimit20 resume (above)
    # did bring p99_step_deg down to 6.14 -- the best in this whole
    # lineage -- but check_roll_regression.py showed it was the WORST of
    # three checkpoints on real hardware-replay roll safety despite that,
    # so a FRESH START was tried next with weight=20 kept, specifically to
    # test whether training it in from scratch (rather than as a late
    # perturbation on an already-converged policy) would avoid that
    # regression. RESULT: much worse failure than expected, stopped at
    # iteration 9579/40000. mean_episode_length pinned near 11 steps the
    # entire time (every healthy fresh start in this project reaches
    # ~450-500 within 1500-2500 iterations), fall_rate still ~8%/step,
    # orientation/reward only 0.58, gravity_y 0.26 (matches the real
    # hardware-measured ~12deg backward tip at the unsupported startup
    # pose), knee tracking error 33-35deg (actual knees sitting near flat
    # regardless of what the reference wanted) -- not "hasn't discovered
    # stepping yet", genuinely failing to balance at all. target_limit_
    # penalty (0.81) and step_limit_penalty (1.83) were both far above any
    # steady-state value seen elsewhere in this project -- plausible
    # mechanism: early random exploration produces large, erratic
    # corrective actions by nature, and weight=20 (stacked with the also-
    # widened target_limit_margin_deg roll=13, which starts penalizing
    # sooner too) may punish the correction needed to catch a stumble
    # almost as hard as the stumble itself, a trap distinct from (worse
    # than) the "safe standing" local optimum this project has hit before.
    # Reverted to the only value ever actually validated (10.0, resume-
    # only history above) rather than splitting the difference -- the
    # failure was severe enough not to guess at a middle ground.
    # target_limit_margin_deg roll=13 left UNCHANGED for the retry (see
    # that field's own comment) -- narrower constraint, only bites near
    # the hard limit specifically, less likely to be what blocked basic
    # balance across the board. If this fresh start ALSO fails the same
    # way, margin=13 becomes the next suspect.
    # DISABLED (10.0 -> 0.0) on 2026-10-02 for a fresh start -- see
    # heading_deviation_penalty_weight's comment for the full reasoning
    # (one of four post-breakthrough penalty terms disabled together,
    # none present in the original successful fresh start). Re-enable via
    # resume at 10.0 once this fresh start finds genuine held stepping --
    # that value has a real, if resume-only, track record; don't jump
    # back to 20.0 (see this field's own comment above for why that
    # failed even harder).
    # RE-ENABLED (0.0 -> 10.0) on 2026-10-03, resuming from the
    # 2026-10-02_22-08-07 fresh start, same batch of four re-enables as
    # heading_deviation_penalty_weight's comment describes. Back to 10.0,
    # not the 20.0 that caused the catastrophic fresh-start failure above
    # -- 10.0 is the only value with an actual resume track record.
    step_limit_penalty_weight: float = 10.0
    step_limit_threshold_deg: float = 4.0
    step_limit_penalty_max_excess_deg: float = 30.0

    # Per-joint-TYPE multiplier on step_limit_penalty_weight (same suffix
    # matching as joint_tracking_weight). Added 2026-09-24: on 2026-09-23
    # battery hardware runs (yaw3 attempts 12 and 14) right_hip_pitch
    # commanded steps of +23.4/+16.9deg at motion_time 0.36-0.40s -- a
    # different phase from the roll-limit problem -- and offline replay of
    # those logs through the roll_margin8 policy still gave 25.0/13.5deg,
    # so the uniform weight-10 penalty isn't shaping this joint enough
    # (consistent with yaw tightening having shifted balance correction
    # into pitch). Only pitch is doubled, not the global weight, so
    # knee/ankle keep the already-working shaping. UNTUNED: watch
    # foot_swing_reward, tracking/mean_max_step_deg and compare_policies'
    # p99_step_deg; if stepping is suppressed drop back toward 1.5.
    # 2026-09-24 (later): pitch 2.0 -> 5.0. The x2 run (resume_roll11_
    # pitchstep2) barely helped: offline replay of hardware attempts 2-4
    # (right_hip_pitch steps 16/22/34.5deg under roll_margin8) gave
    # 20.2/19.8/28.6deg, still past the 15deg max_step_deg, and sim's
    # p99_step_deg rose 7.6 -> 8.6. The trigger is NOT an out-of-
    # distribution observation (max |z| < 3.5 vs the training normalizer
    # during the spike), so this is an in-distribution over-reaction to a
    # real event at mt~1.15s, and only a stronger cost on pitch steps can
    # reshape it. Resume from resume_roll11_pitchstep2 model_279993 (a
    # weight change on an existing term, like the earlier resumes). If
    # foot_swing_reward falls below ~1.3 or fall_rate rises, back off to
    # ~3.5. Compare hardware: roll11_pitchstep2 vs this one.
    step_limit_joint_weight: dict = {
        "yaw": 1.0,
        "roll": 1.0,
        "pitch": 5.0,
        "knee": 1.0,
        "ankle": 1.0,
    }

    # --- foot-height swing reward -----------------------------------------
    # Added because pure joint-angle tracking let the policy alias "roughly
    # matching the step-in-place gait's joint angles on average" without
    # ever actually lifting a foot (see the orientation_margin_deg comment
    # above for the full story -- this is the complementary, more direct
    # fix: an explicit, physically-grounded reward for the literal thing
    # that was missing, "is the foot actually off the ground", rather than
    # relying entirely on joint-angle tracking to imply it).
    #
    # PURELY A TRAINING-TIME SIGNAL. body_pos_w below is ground-truth
    # simulator state (Isaac Lab reading PhysX, no ContactSensor needed --
    # this USD doesn't have one) -- it is NOT added to the observation
    # space, so the deployed policy and robot_deploy.py need no changes at
    # all for this.
    #
    # UPDATE, correcting the original claim above: this DOES turn out to
    # need heel-specific precision, just not full multi-point contact
    # sensing. Originally tracked the FOOT BODY'S OWN ORIGIN (body_pos_w),
    # which is anchored at the ankle joint axis, not any point that
    # actually contacts the ground. Confirmed via direct video: the policy
    # learned to rotate the ankle joint (toes up) while the heel stayed
    # planted the entire time -- not a step at all -- and the ankle-axis
    # origin still rose during that rotation (it's off-axis from the
    # rotation's own pivot within the foot, so it moves even though
    # nothing left the ground), so the reward couldn't tell the two apart.
    # Fixed by tracking the HEEL specifically (left/right_heel_local_m
    # below) instead of the body origin -- a heel-pivot rotation leaves
    # the heel's own height essentially unchanged, so it correctly reads
    # as "not lifted", while a genuine full-foot lift raises it along with
    # everything else.
    #
    # Body (LINK, not joint) names for each foot -- the child link of each
    # ankle joint (Revolute_{left,right}_ankle), confirmed directly from
    # qmini_urdf-2legs.usda's PhysicsRevoluteJoint body0/body1 relations.
    # Note the "Riggt" typo is literally the USD prim name, not a bug here.
    left_foot_body_name: str = "Left_Foot_1"
    right_foot_body_name: str = "Riggt_Foot_1"

    # Heel point, as a fixed LOCAL offset (meters, in each foot body's own
    # frame) from that body's origin -- measured directly from the
    # collision mesh's actual vertices (not guessed): within the
    # back 15% of the mesh by the local Y axis (confirmed convention:
    # X=left-right +left, Y=forward-back +back, Z=up-down +up), the
    # lowest-Z (bottom-most) point. Verified this makes physical sense:
    # the ankle joint's own local Y position sits closer to this back
    # region than to the front, meaning the toe end is FARTHER from the
    # ankle's rotation axis than the heel end -- so a pure ankle rotation
    # sweeps the toe through a much larger vertical arc than the heel,
    # exactly matching what was observed on video (toe visibly lifts,
    # heel barely moves). Computed via a standalone USD geometry query
    # (pxr, no simulation) against qmini_urdf-2legs.usda directly -- see
    # left/right_foot_body_name's comment above for the story. Re-derive
    # these if the foot mesh geometry ever changes.
    left_heel_local_m: tuple[float, float, float] = (0.16286, 0.16930, -0.42102)
    right_heel_local_m: tuple[float, float, float] = (-0.12667, 0.17208, -0.41420)

    # Toe point, same idea and same geometry query as the heel above (front
    # 15% of the mesh by local Y, lowest-Z point within that region).
    # NEEDED, not optional -- tracking the heel ALONE turned out to still
    # be gameable: video confirmed the policy found the MIRROR-IMAGE
    # exploit almost immediately after the heel-only fix landed --
    # plantarflexion (heel lifts, toes stay planted on the ground) genuinely
    # raises the heel point in world space, since the foot pivots about
    # the still-grounded TOE instead of the heel this time. One tracked
    # point can always be gamed by rotating the foot about whatever OTHER
    # point stays planted; only requiring BOTH the heel AND the toe to be
    # simultaneously off the ground (see the min() in _get_rewards) rules
    # out rotating around either one alone. This is the general lesson,
    # not just a one-off patch -- if a THIRD rotation axis or contact
    # point (e.g. rolling onto the outer/inner edge of the foot) ever
    # turns up as a new exploit, the same "pick two points, take the
    # minimum" pattern applies.
    left_toe_local_m: tuple[float, float, float] = (0.14439, 0.04049, -0.42415)
    right_toe_local_m: tuple[float, float, float] = (-0.16138, 0.03790, -0.41897)

    # Foot height (meters, above the flat ground plane) at which the swing
    # reward saturates to its max weight -- doesn't need to match any real
    # clearance target precisely, just needs to be "clearly off the
    # ground" for this robot's scale. UNTUNED guess -- watch actual foot
    # trajectories in sim (or the height itself, if you log it) and adjust.
    # LOWERED 0.03 -> 0.02 on 2026-09-18 (fresh start). The corrected FK
    # pass (see foot_overswing_margin_m's CORRECTION note) shows the
    # reference itself only lifts the swing foot ~2.2cm, so 3cm was already
    # above anything the animation asks for while the policy's p99 heights
    # ran 12-13cm, partly via torso lean rather than joint lift (close-up
    # video). Reward now saturates at 2cm; the overswing free zone becomes
    # 2 + 1 = 3cm total. Watch that foot_swing_reward still reaches ~1.5 by
    # ~3k iterations on the fresh start (a lower target could also weaken
    # the lift incentive). Note compare_policies_isaaclab.py reads this cfg
    # value for its clearance/swing_product diagnostics, so those columns
    # are not comparable with rows recorded at 0.03.
    #
    # SUPERSEDED 2026-09-20 as the value driving _get_rewards' clearance/
    # overswing calc -- see foot_height_target_floor_m's comment for the
    # full story. Kept alive ONLY as a legacy constant that
    # compare_policies_isaaclab.py's diagnostic columns still read; nothing
    # in the actual training reward uses this field anymore. Do not raise
    # or lower this expecting it to change training behavior.
    foot_swing_target_height_m: float = 0.02

    # Numerical floor (meters) under the PER-PHASE reference foot-height
    # target (see reference_foot_heights_m in the keyframes JSON, sampled
    # every step via self.foot_height_motion alongside the joint-angle
    # reference). Added 2026-09-20, replacing the flat
    # foot_swing_target_height_m constant as what foot_swing_reward's
    # clearance and foot_overswing_penalty's threshold are actually
    # measured against -- three separate attempts to fix the "policy
    # lifts way higher than the reference" problem by pushing
    # foot_overswing_penalty_weight (5->25 worked, 25->40 and 25->70 both
    # destabilized training, once via resume and once even from a fresh
    # start) established that hand-tuning a single constant target height
    # is fundamentally the wrong lever: it doesn't generalize to a
    # different future reference clip (e.g. a walking gait with a taller
    # natural lift) without manual re-tuning, and the failed attempts were
    # really about a structural asymmetry (foot_swing_reward gives smooth
    # partial credit for UNDER-lifting, foot_overswing_penalty grows
    # sharply for OVER-lifting, so a strong enough penalty makes
    # under-lifting the cheap escape) that a bigger penalty weight can't
    # fix on its own. The reference's OWN foot-height curve (precomputed
    # via forward kinematics in fk_reference_heights.py/
    # gen_reference_heights.py against qmini_urdf-2legs.usda, validated
    # against the known standing-pose heel height) is now the target at
    # every phase, so it automatically matches whatever gait is loaded.
    # This floor only exists so the clearance division
    # (height / max(reference_height, floor)) doesn't blow up near-zero
    # during stance -- it is NOT a tunable "how high should it lift" knob;
    # swing_target already zeroes foot_swing_reward during stance
    # regardless of what this is set to. Keep small.
    foot_height_target_floor_m: float = 0.005

    # Penalty for foot height PAST foot_swing_target_height_m (plus this
    # margin) -- e.g. a real swing to 10cm gets flagged, one that peaks at
    # 5cm doesn't. Added 2026-09-14: foot_swing_reward above SATURATES at
    # foot_swing_target_height_m (clamped to max=1.0, see left/right_clearance)
    # but nothing has ever penalized going higher -- a fast, tall flick and
    # a controlled swing that both clear 3cm score identically. This is a
    # long-standing suspected-but-never-tested gap: DEFAULT_GAINS["ankle"]'s
    # damping comment (qmini.py) already names this exact mechanism ("a fast
    # flick cheaper than a controlled swing") as the likely explanation for
    # an earlier, unrelated shake problem, but nothing was ever added to
    # test it. Motivated freshly by comparing sim video against real
    # reference footage of this robot class (Recording_2026-09-14_112246.mp4,
    # Qmini_simulation.gif) -- both show a nearly rock-solid torso through
    # the whole gait cycle and a moderate foot lift, versus this project's
    # p99 foot heights running 10-13cm (3-4x past where foot_swing_reward
    # even cares) alongside persistent ~8-9deg mean_tilt_deg that's never
    # budged across six full tuning iterations. A bigger/faster swing than
    # necessary imparts more real reaction torque on the body for zero
    # extra reward -- a plausible direct physical driver of the roll (and
    # possibly yaw) instability, not yet addressed by anything tried so
    # far (all of which targeted joint-angle tracking or overall tilt
    # magnitude, never swing dynamics itself). Margin set equal to
    # foot_swing_target_height_m itself (free up to 6cm total) so genuine,
    # already-working ~3-5cm swings aren't taxed -- only the excess above
    # that. LINEAR (constant gradient, same non-vanishing-gradient
    # reasoning as target_limit_penalty_linear_coef) and capped (same
    # anti-blowup reasoning as target_limit_penalty_max_overshoot_deg --
    # foot height is physically bounded by leg length so a runaway blowup
    # is less likely here, but capping costs nothing and keeps the pattern
    # consistent). UNTUNED starting guess for both weight and margin --
    # watch p99 foot heights actually shrink toward something closer to
    # foot_swing_target_height_m, and whether mean_tilt_deg (and the
    # heading/deviation numbers) improve as a side effect; if p99 heights
    # don't move, this wasn't the mechanism and the next lead is a direct
    # roll-specific term (orientation_reward currently only tracks total
    # tilt magnitude, not per-axis).
    #
    # TIGHTENED 0.03 -> 0.015 on 2026-09-18 (fresh start 2026-09-18_10-40-15).
    # RESULT: no measurable effect -- training-time penalty rose 0.063 ->
    # 0.079 but eval p99 foot heights stayed ~12-13cm; at weight 5.0 the
    # penalty is ~0.08/step vs foot_swing_reward ~1.4, too weak to change
    # behavior. Margin alone is not the lever; weight (or the target
    # height itself) would be.
    #
    # CORRECTION (2026-09-18, later): an earlier version of this comment
    # cited a reference swing of ~4.8cm and "offset causes no tilt (0.01
    # deg)". Both came from a forward-kinematics script with two bugs (ankle
    # joint rotation sign flipped because that joint's body0/body1 order is
    # reversed vs the others; tilt measured about the wrong axis -- the
    # sagittal/pitch axis is the foot's local X, so "tilt about X" was
    # invisible to it). Validated fix: at the init stance pose heel/toe
    # heights match and base-to-heel is ~0.42m only with the corrected sign.
    # Corrected numbers for keyframes_step_in_place_all_joints_2x_base_offset
    # .json (left leg, base_link fixed): swing-foot heel/toe rises only
    # ~2.2cm above the opposite foot (BELOW foot_swing_target_height_m=3cm),
    # identical with/without the stance offset; foot pitch relative to the
    # torso is exactly constant through the cycle -- 0deg in the original
    # (ankle cancels hip+knee, flat foot) vs a constant 7deg toe-up with
    # the offset (pitch+knee-ankle offsets = -16+10-1). So the offset does
    # not change the swing shape, only tilts the whole foot by a constant
    # 7deg -- the same tilt as the tune_stance_lean stance pose.
    # 0.015 -> 0.01 alongside foot_swing_target_height_m 0.03 -> 0.02
    # (total free zone 3cm).
    foot_overswing_margin_m: float = 0.01
    # RAISED 5.0 -> 25.0 on 2026-09-19, alongside a resume (not fresh start)
    # from the foot_swing_target_height_m=0.02 fresh-start checkpoint.
    # RATIONALE: that fresh start lowered p99 eval foot heights only
    # ~18-20% (11.75/12.98cm -> 9.68/10.69cm, still ~4-5x the 2cm target
    # and the reference's own ~2.2cm swing) despite tightening the free
    # zone to 3cm total, because this penalty is a PER-STEP cost and only
    # fires near the swing peak -- most of the gait cycle the foot is near
    # the ground, so the time-averaged penalty stayed low (~0.096) even
    # with the peak far past the free zone. At weight 5.0 that's cheap
    # against tracking_reward+foot_swing_reward's combined ~3.9 budget.
    # 25.0 is a straight 5x, chosen to make the peak-height excess (~6-7cm
    # over the free zone) cost something comparable to that budget instead
    # of ~2.5%. Resume, not fresh start: this term isn't roll_reward or one
    # of the two terms (action_rate_penalty/target_limit_penalty) known to
    # block cold-start discovery via resume -- target_limit_penalty itself
    # was tightened via resume successfully before. Watch foot_swing_reward
    # doesn't collapse (this is now a much bigger penalty relative to it)
    # and whether p99 heights actually approach the 3cm free zone this
    # time.
    #
    # RESULT (2026-09-19, resume from the 0.02 fresh start, 40k more
    # iterations, logs/rsl_rl/qmini-leg/2026-09-19_11-41-39): p99 heights
    # 9.68/10.69cm -> 8.34/8.59cm (~14% further, ~30% cumulative from the
    # original 11.75/12.98cm baseline) -- real but still ~3-4x the 3cm free
    # zone. First run where tracking_err/tilt/roll ALL improved together
    # (4.66/5.46/4.18deg, best-yet on every one) instead of trading off --
    # the direction is working, just not enough bite yet at this weight.
    # Two brief instability spikes during the resume (steps ~65-66k and
    # ~70-71k: peak value_function loss 2302, fall_rate peak 2.9%,
    # episode_length dipped to ~224-233) but both fully recovered within
    # ~1-2k iterations and are ~100x smaller than the sustained roll_reward
    # resume failures.
    #
    # RAISED 25.0 -> 70.0 on 2026-09-19, resuming again from
    # 2026-09-19_11-41-39/model_79998.pt. Same lever, same reasoning as the
    # 5->25 step. 70 is a much bigger bite than the 5x step that worked
    # cleanly last time -- it's not yet established where the point is
    # where this starts suppressing genuine lift instead of just the
    # excess, so watch foot_swing_reward for a real sustained collapse
    # (not just a brief spike like last time) more carefully than before.
    #
    # RESULT: bad. Two collapses (steps ~85k and ~108-112k, episode_length
    # down to ~108, fall_rate up to 2-5%) that this time did NOT fully
    # recover -- last 2000 iterations averaged foot_swing_reward 0.67 (was
    # 1.36-1.5 going in), with both left AND right swing_target down
    # across the whole run (video: user reported the left leg visibly not
    # lifting anymore). orientation/reward and roll_deg hit best-ever
    # values (0.98, 3.25deg) in the same run -- the classic exploit this
    # project has hit before (see qmini.py's DEFAULT_GAINS damping
    # comment): a policy that barely steps also barely tilts, so it read
    # as a very good, not a bad, checkpoint on the orientation/roll
    # metrics while actually having found a "safe, don't lift" local
    # optimum. REVERTED 70.0 -> 40.0 and reset to resume from the last
    # known-good checkpoint (2026-09-19_11-41-39/model_79998.pt, NOT the
    # 120k checkpoint this produced) rather than continuing on top of a
    # policy that already found the shortcut -- past project history
    # (qmini.py, same comment) found this kind of collapse hard to
    # explore back out of once settled. 40 is a smaller step up from 25
    # than 70 was (1.6x vs 2.8x) -- watch foot_swing_reward closely again;
    # if even 40 trends toward a sustained plateau below ~1.2, that's the
    # ceiling for this lever and the next move should be a different one
    # (e.g. lowering foot_swing_target_height_m further) rather than
    # continuing to push this weight.
    #
    # RESULT: also bad, and a DIFFERENT failure shape than 70 -- not one
    # suppression plateau but repeated, WORSENING collapses (episode_length
    # crashes to <150 for 330, then 1486, then 3861 iterations, each one
    # longer than the last, with a 4th still ongoing and 35% of all
    # iterations since the resume at fall_rate>1% when this was stopped
    # partway through, well before its scheduled end). Between this and
    # the 70 result: two different target weights (1.6x and 2.8x over 25),
    # both destabilizing the SAME resumed checkpoint. That points at the
    # resume itself being the fragile part here, not the specific
    # magnitude -- the same category of problem already found with
    # roll_reward (see roll_reward_weight's comment): changing this term's
    # weight via resume on an already-converged policy is what breaks,
    # independent of the target value. STOPPED this run early (the
    # worsening trend made waiting out the remaining iterations not worth
    # it) and switching strategy to match how roll_reward's resume-
    # fragility was actually solved: same target value (40), but as a
    # FRESH START instead of a resume. If a fresh start with this weight
    # active from iteration 0 trains cleanly (the roll_reward precedent),
    # that confirms resume-on-this-checkpoint was the actual problem, not
    # the weight itself.
    #
    # RESULT: also bad -- a fresh start with weight=40 plateaued at
    # foot_swing_reward ~0.5-0.65 from iteration ~3000 onward (a known-good
    # fresh start at the same target-height config reached ~1.3 by the
    # same point and held it), so 40 suppresses genuine lift even with no
    # resume involved. That rules out resume-fragility as the explanation
    # here and points at the structural problem described in
    # foot_height_target_floor_m's comment instead: foot_swing_reward
    # gives smooth partial credit for UNDER-lifting while this penalty
    # grows for OVER-lifting, so past some weight, under-lifting becomes
    # the cheap escape regardless of how it's introduced. REVERTED 40.0 ->
    # 25.0 (the one value that's actually worked) and stopped pushing this
    # weight further -- see foot_height_target_floor_m's comment for the
    # replacement approach (a per-phase reference target instead of a
    # flat constant) tried instead of continuing to escalate this weight.
    foot_overswing_penalty_weight: float = 25.0
    foot_overswing_penalty_max_m: float = 0.15

    # RAISED from 1.0 to 3.0 -- after the roll-specific fix
    # (joint_tracking_weight) closed that particular exploit, the SAME
    # underlying avoidance pattern recurred in a subtler form: a resumed
    # run reached the best per-joint tracking accuracy of the whole project
    # (yaw/pitch/knee/ankle all 1-3 deg, roll a modest 1.6-1.9 deg, no
    # blowup) while tracking/foot_swing_reward kept DECLINING (0.15 -> 0.12)
    # and foot clearance stayed negative even at a moderate-high
    # swing_target (~0.57) -- i.e. the policy found it could closely match
    # the reference's joint ANGLES while still not completing genuine
    # weight transfer. With foot_swing_reward_weight=1.0 realistically
    # peaking around 0.15-0.2 against 2.0*tracking_reward sitting near its
    # own ~2.0 max once tracking is this good, there was very little
    # marginal incentive left to risk a bigger, riskier motion for real
    # clearance -- tracking alone already captured nearly all the
    # achievable reward. 3.0 makes genuine clearance a comparably-sized
    # slice of total reward instead of a rounding error next to tracking.
    # UNTUNED escalation -- if foot_swing_reward still doesn't climb
    # meaningfully above its historical ~0.15-0.17 ceiling, that would be
    # real evidence the bottleneck isn't reward MAGNITUDE at all (worth
    # revisiting foot_swing_target_height_m, or whether the swing_target
    # derivation itself needs to look at more than just knee angle).
    #
    # RAISED AGAIN, 3.0 -> 5.0 (2026-08-27): same underlying pattern as the
    # 1.0->3.0 raise above, recurring against a different competitor.
    # orientation_margin_deg (tightened 6->2) and position_reward (added
    # new) both reward the base staying still/level -- genuine stepping
    # always costs something against both, and foot_swing_reward is the
    # only term pulling the other way. FIVE separate resumes from
    # model_367800.pt (four unrelated action/gain experiments, plus one
    # completely unmodified control) all independently collapsed into
    # "stop lifting" within roughly the same iteration budget regardless of
    # what else was changed -- consistent with "stand still" having become
    # a genuinely competitive-or-better strategy under the current weights
    # once orientation_reward_weight/position_reward_weight's combined
    # pull is counted, not an exploration/checkpoint fluke each of those
    # five attempts could plausibly have fixed on its own (and didn't).
    # This raise directly targets that balance rather than trying another
    # exploration/gain-side lever. If foot_swing_reward still can't hold up
    # over a long resumed run at 5.0, that's real evidence against the
    # reward-balance theory entirely and toward something else (see the
    # options discussed when this was raised -- pushing is explicitly NOT
    # one of them right now, see push_interval_s's point 1).
    #
    # Raised again 5.0 -> 8.0 on 2026-09-02, but NOT under the same
    # conditions as the paragraph above -- this time cfg.motion_time_noise_std_s
    # (new that same day) was added alongside it, and THAT run (resumed from
    # model_661300.pt) collapsed into "stop lifting" around iteration 684k,
    # ~23k iterations in, after 5.0 alone had held for many prior runs
    # without motion_time_noise_std_s present. So this isn't the same
    # collapse recurring unchanged -- it's a genuinely harder problem (the
    # policy now has to stay robust through a real phase-input perturbation
    # at its two roughest transitions) tipping the same balance over again.
    # Direct evidence the collapse bought something: sweeping the
    # pre-collapse checkpoint (iter 683900) vs. the final one showed the
    # motion_time-noise fix genuinely working at the cycle-end transition
    # (mt~2.16-2.20, action peak -0.51 -> -0.19) but NOT at the other one
    # (mt~1.06-1.28, left_hip_roll peak -0.69 -> -0.68, unmoved) -- the
    # policy partially solved the new problem by giving up stepping instead
    # of by getting robust, exactly what this raise is meant to make less
    # attractive. Resuming this next run from model_683900.pt (last
    # checkpoint before the collapse, stepping still intact), NOT the
    # collapsed final checkpoint. If foot_swing_reward collapses again at
    # 8.0 under this same motion_time_noise_std_s setting, the honest
    # reading is that reward-balance tuning alone can't buy robustness at
    # the hip_roll transition -- worth trying easing motion_time_noise_std_s
    # instead of raising this again, or a curriculum that oversamples
    # resets near the two known bad windows so gradient signal there isn't
    # diluted across the whole 2.2s cycle.
    #
    # That resumed run finished on 2026-09-02: REVERTED 8.0 -> 5.0 on
    # 2026-09-03. It didn't recover cleanly -- noise_std plateaued at ~0.44
    # (vs the usual ~0.18-0.20), heading/yaw_rate_dps regressed to ~100
    # deg/s, orientation/reward settled at ~0.77 -- and the sweep showed the
    # motion_time landmines were untouched (see motion_time_noise_std_s'
    # comment, now also disabled). The apparent jump in the logged
    # tracking/foot_swing_reward number (~0.80 -> ~1.30) was also mostly an
    # artifact of the weight change itself, not real improvement: this
    # metric is the WEIGHTED contribution (weight * raw term, see
    # _get_rewards), so normalizing back out (1.30/8 ~= 0.16 vs 0.80/5 ~=
    # 0.16) shows the underlying raw stepping quality barely moved. Given
    # neither of the two changes made together this round bought anything
    # measurable and one plausibly made things worse, reverting both back
    # to the last independently-validated state before trying
    # action_delay_range_steps' widening in isolation.
    #
    # Raised again 5.0 -> 7.0 on 2026-09-05, for an unrelated reason this
    # time: the first from-scratch run under the new sin/cos phase encoding
    # (see observation_space's comment) fully converged (noise_std down to
    # ~0.13-0.18, stable) but into a genuine "stop lifting" collapse --
    # foot_swing_reward flat at ~0.003 for the whole second half of
    # training, orientation/reward at an all-time high ~0.98, fall_rate 0,
    # right_swing_target pinned at 0.0. The noise_std/foot_swing_reward
    # curves both show a repeated sawtooth through iterations 0-20k (the
    # policy visibly rediscovering then losing real stepping several times)
    # before settling into standing-still for good around 20-25k -- this
    # reads as a genuine reward-balance loss, not incomplete training (no
    # exploration budget left to rediscover it from here). 5.0 clearly
    # isn't enough under this new encoding. Not jumping straight back to
    # 8.0 -- that attempt was confounded with motion_time_noise_std_s
    # (active at the same time, since disabled), so 8.0 alone was never
    # actually isolated and shouldn't be written off. 7.0 splits the
    # difference: a real push in the same proven direction without
    # repeating that specific combination. Resuming from
    # model_18500.pt (this run's own lineage, from before the final
    # collapse locked in -- the last visible sawtooth peak in both
    # noise_std and foot_swing_reward), not the fully-collapsed final
    # checkpoint.
    foot_swing_reward_weight: float = 7.0

    # TEMPORARY per-side multipliers on foot_swing_reward, on top of the
    # weight above. Added after the ankle-tracking fix (joint_tracking_weight)
    # delivered a clean, SYMMETRIC improvement (both ankles down to ~1 deg
    # error, from a persistent 3-8 deg) with zero change in the underlying
    # left-only behavior. At the time this looked like an asymmetric local
    # optimum needing a curriculum nudge to break (right boosted to 1.5x
    # to catch up to left's apparent "head start"). REVERTED back to
    # 1.0/1.0 once the real cause was found: the foot-height signal itself
    # was measuring the wrong point (ankle-anchored body origin, see
    # left/right_heel_local_m's comment), which a pure ankle rotation
    # could satisfy with the heel still planted -- not real stepping at
    # all. Left never had a genuine head start to compensate for; there
    # was nothing asymmetric to fix, just a broken measurement that
    # happened to be easier to fake on one side. Now that the heel-based
    # fix is in, starting clean (1.0/1.0) gives an honest read on whether
    # any real asymmetry remains -- keeping the old boost would make that
    # impossible to tell apart from an artifact of the multiplier itself.
    #
    # RESULT of that clean 1.0/1.0 test (~40k more iterations, heel-based
    # signal): real, video-confirmed answer this time, not an artifact --
    # left genuinely learned to lift (foot_swing_reward climbed for real,
    # to ~0.83-0.85, matching visible full-leg lift with some post-lift
    # balance instability, exactly what a newly-learned single-support
    # skill should look like), while right is still just going up on its
    # toes with "not much lift" per direct video. Symmetric weights alone
    # didn't close this gap in a full run's worth of iterations, so
    # re-applying the same 1.5x boost on right -- this time targeting a
    # confirmed real asymmetry instead of compensating for a broken
    # measurement. REBALANCE BACK TO 1.0/1.0 once the right foot shows
    # real, video-confirmed full-leg lift matching the left.
    #
    # Reverted to 1.0/1.0 here (2026-08-25, run 2026-08-25_12-02-11,
    # ~iteration 292600): after orientation_margin_deg was tightened (6->2)
    # and position_reward was added, video showed the RIGHT foot doing a
    # sharp pop-up/twitch/drop instead of a controlled swing. Suspected
    # mechanism: foot_swing_reward is scored per-frame with no smoothness
    # or sustain requirement (swing_target * clearance, see _get_rewards),
    # so once sustained single-support lean got more expensive (tighter
    # margin + position penalty), a brief flick that briefly touches
    # clearance=1.0 became a cheaper way to collect the same reward than a
    # smooth swing-through -- and right, being weighted 1.5x, was under the
    # most pressure to find that shortcut first. Testing in isolation: if
    # right's clearance collapses back down at 1.0/1.0, the twitch wasn't
    # about this weight and the boost should go back; if the twitch clears
    # and clearance holds, this was it.
    foot_swing_left_weight: float = 1.0
    foot_swing_right_weight: float = 1.0

    # Steps to withhold foot_swing_reward entirely after each push
    # disturbance (cfg.push_interval_s). Added after the weight=3.0 raise
    # above produced a large jump in tracking/foot_swing_reward and in an
    # eval-time diagnostic's reported foot height -- but a real, direct
    # video inspection of the trained policy showed almost NO genuine
    # stepping: the only visible leg lift happened right when a push
    # disturbance landed, and even that was minimal. Root cause: this
    # reward's foot-height computation (body_pos_w relative to a captured
    # stance reference) can't distinguish "the policy lifted its foot on
    # purpose" from "the foot got knocked into the air by an external
    # shove that happened to land during a swing_target window" -- and
    # since pushes are active during training too, the policy could farm
    # real reward from a passive push-recovery motion without ever
    # learning genuine intentional stepping. Mirrors the identical fix
    # applied to compare_policies_isaaclab.py's SETTLE_STEPS (same
    # underlying computation, same blind spot, confirmed by both the
    # eval diagnostic AND the trained policy's actual behavior pointing
    # at the same cause). 25 steps (~0.5s at the default step_dt) is a
    # guess at push-recovery duration, matching the eval script's value --
    # not independently verified.
    #
    # Also now gates position_reward/heading_reward (2026-08-28) -- same
    # underlying reasoning, a push forces velocity/yaw-rate the policy
    # didn't choose, so none of the push-sensitive reward terms should
    # score it. Name kept as-is (foot_swing was first) rather than
    # renamed to something more generic -- all three read this same field.
    foot_swing_push_cooldown_steps: int = 25

    # Steps to wait after EACH env's reset before capturing its
    # foot-swing stance-height reference (self._left/right_foot_stance_height)
    # -- see that state's __init__ comment for the full story. Capturing on
    # the very first possible step after reset caught a settling transient
    # from that reset's randomized startup pose, not a representative
    # "quiet stance", producing a persistent (not swing-correlated) height
    # offset for the rest of the episode. Matches
    # compare_policies_isaaclab.py's SETTLE_STEPS -- same underlying
    # transient, same fix, just applied to the actual reward this time
    # instead of only a measurement tool.
    foot_swing_stance_settle_steps: int = 25

class QminiLegEnv(DirectRLEnv):
    cfg: QminiLegEnvCfg

    def __init__(self, cfg: QminiLegEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        print("\nRobot joints:")
        for joint_id, joint_name in enumerate(self.robot.joint_names):
            print(f"  {joint_id:2d}: {joint_name}")

        print("\nRobot bodies:")
        for body_id, body_name in enumerate(self.robot.body_names):
            print(f"  {body_id:2d}: {body_name}")

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

        # Used by the foot-height swing reward in _get_rewards -- see
        # cfg.foot_swing_reward_weight's comment.
        assert hasattr(self.robot.data, "body_pos_w"), (
            "self.robot.data has no 'body_pos_w' attribute -- the foot-height "
            "swing reward needs updating for this Isaac Lab version's actual "
            "attribute name (try `dir(self.robot.data)` and grep for 'body_pos')."
        )
        for name in (self.cfg.left_foot_body_name, self.cfg.right_foot_body_name):
            assert name in self.robot.body_names, (
                f"cfg.left_foot_body_name/right_foot_body_name={name!r} not "
                f"found in self.robot.body_names={self.robot.body_names} -- "
                f"see the 'Robot bodies:' printout above and fix these cfg "
                f"fields to match your USD's actual foot link names."
            )
        self._left_foot_body_idx = self.robot.body_names.index(self.cfg.left_foot_body_name)
        self._right_foot_body_idx = self.robot.body_names.index(self.cfg.right_foot_body_name)

        # Heel/toe offsets, broadcast-ready (num_envs, 3) -- see
        # cfg.left/right_heel_local_m and left/right_toe_local_m's
        # comments for where these numbers come from and why BOTH are
        # needed. Expanded once here rather than per-step.
        self._left_heel_local_offset = torch.tensor(
            self.cfg.left_heel_local_m, device=self.device
        ).expand(self.num_envs, 3)
        self._right_heel_local_offset = torch.tensor(
            self.cfg.right_heel_local_m, device=self.device
        ).expand(self.num_envs, 3)
        self._left_toe_local_offset = torch.tensor(
            self.cfg.left_toe_local_m, device=self.device
        ).expand(self.num_envs, 3)
        self._right_toe_local_offset = torch.tensor(
            self.cfg.right_toe_local_m, device=self.device
        ).expand(self.num_envs, 3)

        # body_pos_w's Z for a link is NOT "height of that part above the
        # ground" -- for a URDF-imported USD, a link's own coordinate
        # origin sits wherever the joint attaching it to its PARENT is
        # defined (here, the ankle joint axis), not at the physical
        # sole/ground-contact surface. Confirmed live: base_link read
        # 0.3665 while Left_Foot_1 read 0.4081 (HIGHER) with
        # gravity=(-0.003, 0.000, -1.000) -- i.e. dead level, not fallen --
        # so absolute world height isn't usable as "is the foot on the
        # ground" at all. What IS usable: the CHANGE in a foot's height
        # relative to its own value in a known stance pose.
        #
        # PER-ENV, captured lazily on the first _get_rewards() call AFTER
        # EACH reset (not once ever, and not here in __init__) -- reads
        # body_pos_w only after at least one real physics step has
        # happened post-reset (write_joint_state_to_sim alone doesn't
        # retroactively update body_pos_w). Originally a single GLOBAL
        # capture (one bool, captured once for the whole env instance) --
        # that was a real, confirmed bug: cfg.startup_joint_pos_noise_deg
        # randomizes joint pose by up to +-15deg (knee/pitch) at EVERY
        # reset, so any episode after the very first one was being
        # compared against a stale reference from a DIFFERENT random
        # starting pose. A robot that happened to reset with extra knee
        # flexion would read as having "clearance" relative to that stale
        # reference while standing completely still -- earning real
        # foot_swing_reward for pure luck of the random draw, with zero
        # actual stepping. Confirmed by an eval run where video showed two
        # frames 3.75s apart with IDENTICAL leg positions (no motion at
        # all) while foot_swing_reward still read ~0.55, and this held
        # true independent of push disturbances (tested both with and
        # without), ruling those out and pointing at something present in
        # literally every episode instead -- exactly what per-episode
        # startup randomization is.
        self._foot_stance_height_captured = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # "_foot_stance_height" (no "heel"/"toe" in the name) refers to the
        # HEEL specifically, kept as-is for continuity with existing code
        # (including compare_policies_isaaclab.py, which reads these exact
        # attribute names). The toe gets its own, separately-named pair
        # below -- both captured together, gated by the same flag above.
        self._left_foot_stance_height = torch.zeros(self.num_envs, device=self.device)
        self._right_foot_stance_height = torch.zeros(self.num_envs, device=self.device)
        self._left_toe_stance_height = torch.zeros(self.num_envs, device=self.device)
        self._right_toe_stance_height = torch.zeros(self.num_envs, device=self.device)

        # SECOND bug in the same mechanism, found after the per-episode
        # fix above still didn't move tracking/foot_swing_reward: capture
        # was still happening on the very FIRST possible _get_rewards()
        # call after each reset -- likely mid-settling-transient from that
        # reset's randomized startup pose (cfg.startup_joint_pos_noise_deg),
        # not a representative "quiet stance". Confirmed via a live diag
        # dump: left_height_cm read persistently POSITIVE (1.85-4.52cm)
        # across an entire 500-step window, including at moments
        # left_swing_target==0 (the reference's own lowest point) -- i.e.
        # uncorrelated with swing phase, a flat offset rather than a
        # dynamic signal, exactly what a too-early anchor produces. This
        # tracks the same settling-transient issue already fixed in
        # compare_policies_isaaclab.py's SETTLE_STEPS, just never applied
        # to the actual training reward's own capture. See
        # cfg.foot_swing_stance_settle_steps and _steps_since_reset below.
        self._steps_since_reset = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

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

        # Per-joint zero-calibration error range (rad) and the per-env
        # offsets themselves -- see cfg.joint_zero_offset_range_deg.
        self._joint_zero_offset_range_rad = torch.zeros(self.num_joints, device=self.device)
        for i, name in enumerate(self.robot.joint_names[:self.num_joints]):
            for suffix, deg in self.cfg.joint_zero_offset_range_deg.items():
                if name.endswith(suffix):
                    self._joint_zero_offset_range_rad[i] = math.radians(deg)
                    break
            else:
                raise ValueError(
                    f"Robot joint '{name}' doesn't end with any of "
                    f"{list(self.cfg.joint_zero_offset_range_deg)} -- add an "
                    f"entry to cfg.joint_zero_offset_range_deg covering it."
                )
        self._joint_zero_offset_rad = torch.zeros(self.num_envs, self.num_joints, device=self.device)

        # Per-joint-TYPE tracking weight -- see cfg.joint_tracking_weight's
        # comment. Same name-suffix matching convention as the noise range
        # just above.
        self._joint_tracking_weight = torch.ones(self.num_joints, device=self.device)
        for i, name in enumerate(self.robot.joint_names[:self.num_joints]):
            for suffix, weight in self.cfg.joint_tracking_weight.items():
                if name.endswith(suffix):
                    self._joint_tracking_weight[i] = weight
                    break
            else:
                raise ValueError(
                    f"Robot joint '{name}' doesn't end with any of "
                    f"{list(self.cfg.joint_tracking_weight)} -- add an "
                    f"entry to cfg.joint_tracking_weight covering it."
                )

        # Per-joint-TYPE target-limit margin (radians) -- see
        # cfg.target_limit_margin_deg's comment. Same matching convention
        # as _joint_tracking_weight just above.
        self._target_limit_margin_rad = torch.zeros(self.num_joints, device=self.device)
        for i, name in enumerate(self.robot.joint_names[:self.num_joints]):
            for suffix, deg in self.cfg.target_limit_margin_deg.items():
                if name.endswith(suffix):
                    self._target_limit_margin_rad[i] = math.radians(deg)
                    break
            else:
                raise ValueError(
                    f"Robot joint '{name}' doesn't end with any of "
                    f"{list(self.cfg.target_limit_margin_deg)} -- add an "
                    f"entry to cfg.target_limit_margin_deg covering it."
                )

        # Per-joint-TYPE step-limit multiplier -- see
        # cfg.step_limit_joint_weight's comment.
        self._step_limit_joint_weight = torch.ones(self.num_joints, device=self.device)
        for i, name in enumerate(self.robot.joint_names[:self.num_joints]):
            for suffix, weight in self.cfg.step_limit_joint_weight.items():
                if name.endswith(suffix):
                    self._step_limit_joint_weight[i] = weight
                    break
            else:
                raise ValueError(
                    f"Robot joint '{name}' doesn't end with any of "
                    f"{list(self.cfg.step_limit_joint_weight)} -- add an "
                    f"entry to cfg.step_limit_joint_weight covering it."
                )

        # Per-joint-TYPE action-rate weight -- see
        # cfg.action_rate_joint_weight's comment. Same matching convention
        # as _joint_tracking_weight just above.
        self._action_rate_joint_weight = torch.ones(self.num_joints, device=self.device)
        for i, name in enumerate(self.robot.joint_names[:self.num_joints]):
            for suffix, weight in self.cfg.action_rate_joint_weight.items():
                if name.endswith(suffix):
                    self._action_rate_joint_weight[i] = weight
                    break
            else:
                raise ValueError(
                    f"Robot joint '{name}' doesn't end with any of "
                    f"{list(self.cfg.action_rate_joint_weight)} -- add an "
                    f"entry to cfg.action_rate_joint_weight covering it."
                )

        # Per-joint-TYPE action-delay (min, max) range, in steps -- see
        # cfg.action_delay_range_steps' comment. Same matching convention
        # as _joint_tracking_weight above; used by _randomize_action_delay
        # to sample each joint's own per-episode delay independently.
        self._action_delay_low = torch.zeros(self.num_joints, dtype=torch.long, device=self.device)
        self._action_delay_high = torch.zeros(self.num_joints, dtype=torch.long, device=self.device)
        for i, name in enumerate(self.robot.joint_names[:self.num_joints]):
            for suffix, (low, high) in self.cfg.action_delay_range_steps.items():
                if name.endswith(suffix):
                    self._action_delay_low[i] = low
                    self._action_delay_high[i] = high
                    break
            else:
                raise ValueError(
                    f"Robot joint '{name}' doesn't end with any of "
                    f"{list(self.cfg.action_delay_range_steps)} -- add an "
                    f"entry to cfg.action_delay_range_steps covering it."
                )

        keyframes, degrees = _load_reference_keyframes(KEYFRAMES_PATH, self.robot.joint_names)
        self.motion = MotionPlayer(
            keyframes=keyframes,
            device=self.device,
            degrees=degrees,
        )

        # Per-phase reference foot-height target -- see
        # cfg.foot_height_target_floor_m's comment for why this replaced
        # the flat foot_swing_target_height_m constant. Reuses MotionPlayer
        # (already a generic multi-channel linear interpolator over
        # `times`) for a second, independent set of channels
        # ([left_heel, left_toe, right_heel, right_toe], meters, not
        # degrees) sampled at the SAME self.motion_time as the joint-angle
        # reference every step in _get_rewards.
        foot_height_keyframes = _load_reference_foot_heights(KEYFRAMES_PATH)
        self.foot_height_motion = MotionPlayer(
            keyframes=foot_height_keyframes,
            device=self.device,
            degrees=False,
        )

        # --- foot-height swing reward: per-foot swing-phase target -------
        # "How much should this foot be lifted right now" is derived from
        # the reference clip's OWN knee angle at the current motion_time
        # (already sampled every step in _get_rewards as `reference`),
        # normalized between that knee's DEFAULT/calibrated-standing value
        # (0 = stance) and whichever direction it swings FARTHEST toward in
        # the clip (1 = full swing). No forward kinematics needed: for this
        # leg's kinematic design (hip_pitch + knee + ankle in series, all
        # rotating in the sagittal plane), a knee bent further than its
        # standing value directly means a higher foot for a roughly-fixed
        # hip height, and the step-in-place clip's own knee excursion (see
        # keyframes_step_in_place_all_joints_2x_base_offset.json) was
        # specifically authored to lift the foot, not to bend the knee
        # while keeping it planted -- so knee angle is a robust,
        # already-available proxy for swing phase. Anchored on the
        # DEFAULT value (not the clip's own min) rather than assuming a
        # fixed sign per leg, since left/right joints are mirrored (see
        # qmini.py/tune_stance_lean_isaaclab.py's sign-convention notes) --
        # this handles either sign automatically.
        left_knee_idx = self.robot.joint_names.index("Revolute_left_knee")
        right_knee_idx = self.robot.joint_names.index("Revolute_right_knee")
        # `keyframes` stores the clip's RAW values, in whatever unit the
        # JSON's own "degrees" flag says (True for
        # keyframes_step_in_place_all_joints_2x_base_offset.json) --
        # self.motion (MotionPlayer) converts to radians internally when
        # sampled via .sample(), but that conversion doesn't apply here
        # since this reads `keyframes` directly, bypassing MotionPlayer.
        # Missing this conversion was a real, confirmed bug: comparing
        # degree-scale clip values (~17-34) against a radian-scale stance
        # value (~0.39) made _swing_extreme's denominator ~165x too large,
        # capping swing_target at ~0.006 max for an entire 40k-iteration
        # training run regardless of policy -- confirmed via
        # compare_policies_isaaclab.py's population-max diagnostic showing
        # max_left/right_swing_target pinned at exactly 0.006 across three
        # very different checkpoints (10k/20k/30k), which fully explains
        # why tracking/foot_swing_reward never grew past ~8e-4 the whole run.
        clip_values = torch.tensor([pose for _, pose in keyframes], dtype=torch.float32)
        if degrees:
            clip_values = torch.deg2rad(clip_values)

        def _swing_extreme(values: torch.Tensor, stance: float) -> float:
            lo, hi = float(values.min()), float(values.max())
            return hi if (hi - stance) >= (stance - lo) else lo

        left_knee_stance = float(self.robot.data.default_joint_pos[0, left_knee_idx])
        right_knee_stance = float(self.robot.data.default_joint_pos[0, right_knee_idx])
        self._left_knee_idx = left_knee_idx
        self._right_knee_idx = right_knee_idx
        self._left_knee_stance_rad = left_knee_stance
        self._left_knee_swing_extreme_rad = _swing_extreme(clip_values[:, left_knee_idx], left_knee_stance)
        self._right_knee_stance_rad = right_knee_stance
        self._right_knee_swing_extreme_rad = _swing_extreme(clip_values[:, right_knee_idx], right_knee_stance)

        self.motion_time = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float32,
        )

        # Per-env world-frame yaw captured at each env's last reset -- see
        # cfg.heading_deviation_penalty_weight's comment for why this
        # exists (heading_reward above only ever tracks yaw RATE, never
        # absolute heading). Set for real in _reset_idx right after the
        # root pose is written there; zeros here is just a safe placeholder
        # before the first reset.
        self._reset_yaw = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float32,
        )

        # Per-env world-frame xy position captured at each env's last
        # reset -- see cfg.position_deviation_penalty_weight's comment.
        # Same placeholder-until-first-reset pattern as self._reset_yaw
        # just above.
        self._reset_pos_xy = torch.zeros(
            self.num_envs, 2,
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
        # env. _apply_action reads back `action_delay_steps[env, joint]`
        # steps behind the write pointer, PER JOINT now (see
        # cfg.action_delay_range_steps' comment for why), so each env sees
        # a per-episode-fixed, per-joint-type zero-order-hold delay instead
        # of the instantaneous action Isaac Lab would otherwise apply.
        self._action_buffer_len = max(1, max(h for _, h in self.cfg.action_delay_range_steps.values()) + 1)
        self._action_buffer = torch.zeros(
            self.num_envs, self._action_buffer_len, self.cfg.action_space,
            device=self.device, dtype=torch.float32,
        )
        # Shape (num_envs, action_space) now, not (num_envs,) -- per-joint
        # delay, not one shared value per env. Populated by
        # _randomize_action_delay, which needs self._action_delay_low/high
        # (built further down alongside _joint_tracking_weight, once
        # self.robot.joint_names is available) -- safe to zero-init here
        # since _reset_idx always runs before the first real step.
        self._action_delay_steps = torch.zeros(self.num_envs, self.cfg.action_space, dtype=torch.long, device=self.device)
        self._buffer_ptr = 0
        self._delayed_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # --- action smoothing state ----------------------------------------
        # EMA state for cfg.action_smoothing, one per env -- see that cfg's
        # comment. Zero-initialized here; _reset_idx sets each reset env's
        # entry to its actual default_joint_pos (the action decode's own
        # anchor pose) once self.robot is available, rather than leaving it
        # at a stale value from a previous episode.
        self._smoothed_position_targets = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

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
        # See cfg.push_interval_range_s/push_velocity_range_mps and
        # _pre_physics_step. Per-env countdown (steps until this env's next
        # push), independently sampled and re-sampled after every push and
        # every reset -- see _resample_push_countdown -- so push timing
        # decorrelates from both the gait cycle and the global step
        # counter instead of every env sharing one synchronized schedule.
        self._steps_until_push = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._resample_push_countdown(self.robot._ALL_INDICES)
        # Per-env steps since this env's own last push (or since reset, if
        # it hasn't been pushed yet this episode) -- see
        # cfg.foot_swing_push_cooldown_steps' comment for why
        # foot_swing_reward needs to be withheld per-env now, not with one
        # shared global on/off window.
        self._steps_since_push = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # --- startup support-then-release (rope simulation) ------------------
        # See cfg.startup_support_duration_range_s/startup_support_force_frac.
        # Per-env step count at which support drops to zero this episode --
        # resampled for real in _reset_idx (unlike push's countdown above,
        # this MUST resample every reset, not carry over, since every
        # episode needs its own fresh "supported for the first bit" start).
        # Zero here is just a safe placeholder before the first reset.
        self._support_end_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Set for real every _pre_physics_step; this is just a safe
        # placeholder in case _get_rewards' logging ever runs first.
        self._support_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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
        # friction 1.0 + "multiply": the effective friction is the robot
        # shapes' own per-env value, see cfg.friction_randomization_range.
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(
            physics_material=RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
                friction_combine_mode="multiply",
            )
        ))
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
        #
        # Per-joint now (see cfg.action_delay_range_steps' comment) --
        # read_idx is (num_envs, action_space), a different delay per
        # joint, so a plain env-indexed gather isn't enough; use
        # torch.gather along the buffer-length dimension instead.
        self._action_buffer[:, self._buffer_ptr, :] = self.actions
        read_idx = (self._buffer_ptr - self._action_delay_steps) % self._action_buffer_len
        self._delayed_actions = self._action_buffer.gather(1, read_idx.unsqueeze(1)).squeeze(1)
        self._buffer_ptr = (self._buffer_ptr + 1) % self._action_buffer_len

        # --- push disturbances --------------------------------------------
        # See cfg.push_interval_range_s/push_velocity_range_mps. Per-env
        # countdown, NOT a shared global schedule -- see that cfg's
        # comment for why a single fixed interval was a real problem
        # (every env's pushes landing at close to the same gait phase).
        self._steps_until_push -= 1
        self._steps_since_push += 1
        due = (self._steps_until_push <= 0).nonzero(as_tuple=True)[0]
        if due.numel() > 0:
            self._apply_random_push(due)
            self._resample_push_countdown(due)
            self._steps_since_push[due] = 0

        # --- startup support-then-release (rope simulation) ------------------
        # See cfg.startup_support_duration_range_s/startup_support_force_frac.
        # Called every step (not just at the on/off transition) because
        # set_external_force_and_torque's buffer holds whatever was last
        # written until overwritten -- the zero-force call once support
        # ends is what actually disables it, not a one-time "turn it off"
        # event. Applied at base_link's own origin as a pure vertical
        # force (no torque, no offset toward the real handle's mount
        # point) -- a deliberate simplification, not a claim this is where
        # the real rope attaches.
        # Stored on self (not just a local) so _get_rewards' logging can
        # read it too, for tracking/startup_support_active_frac below.
        self._support_active = self.episode_length_buf < self._support_end_step
        still_supported = self._support_active
        total_mass_kg = self.robot.root_physx_view.get_masses().sum(dim=1).to(self.device)
        support_newtons = torch.where(
            still_supported,
            self.cfg.startup_support_force_frac * total_mass_kg * 9.81,
            torch.zeros_like(total_mass_kg),
        )
        support_forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        support_forces[:, 0, 2] = support_newtons
        support_torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.robot.set_external_force_and_torque(
            support_forces, support_torques, body_ids=[self._base_body_idx], is_global=True,
        )

    def _apply_random_push(self, env_ids: torch.Tensor):
        push_vel = sample_uniform(
            *self.cfg.push_velocity_range_mps, (env_ids.numel(), 3), device=self.device,
        )
        lin_vel = self.robot.data.root_lin_vel_w[env_ids].clone() + push_vel
        ang_vel = self.robot.data.root_ang_vel_w[env_ids].clone()
        self.robot.write_root_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=-1), env_ids)

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

        # Joint sensor-realism noise -- see cfg.joint_pos_noise_std_deg /
        # joint_vel_noise_std_deg_s's comment. Same pattern as the IMU
        # noise above: applied only to what the policy OBSERVES here, not
        # to self.robot.data.joint_pos/joint_vel as read directly by
        # _get_rewards/_get_dones/_apply_action, which should keep
        # reflecting true simulated state.
        # Believed angle = true - zero-calibration offset, see
        # cfg.joint_zero_offset_range_deg.
        joint_pos = self.robot.data.joint_pos[:, :self.num_joints] - self._joint_zero_offset_rad
        joint_vel = self.robot.data.joint_vel[:, :self.num_joints]
        joint_pos = joint_pos + torch.randn_like(joint_pos) * math.radians(self.cfg.joint_pos_noise_std_deg)
        joint_vel = joint_vel + torch.randn_like(joint_vel) * math.radians(self.cfg.joint_vel_noise_std_deg_s)

        # Phase-input realism noise -- see cfg.motion_time_noise_std_s's
        # comment (currently disabled, 0.0). Same pattern as the other
        # observation noise above: perturbs only what the policy OBSERVES
        # here. self.motion_time itself stays exact everywhere else
        # (reference sampling for rewards, the action-delay buffer's
        # indexing, episode timing), only this observed copy is jittered.
        #
        # RE-ADOPTED sin(phase)/cos(phase) 2026-09-08 -- see
        # observation_space's comment for the full reasoning. Noise is
        # applied to the raw scalar BEFORE the sin/cos transform (not to
        # sin/cos independently), so it stays a physically meaningful
        # phase-timing jitter rather than an arbitrary perturbation of two
        # otherwise-coupled unit-circle coordinates.
        motion_time_obs = self.motion_time.unsqueeze(1) + torch.randn_like(
            self.motion_time.unsqueeze(1)
        ) * self.cfg.motion_time_noise_std_s
        phase_obs = 2.0 * math.pi * motion_time_obs / self.motion.length
        phase_sin_cos_obs = torch.cat((torch.sin(phase_obs), torch.cos(phase_obs)), dim=-1)

        observations = torch.cat(
            (
                joint_pos,
                joint_vel,
                imu_gravity,
                imu_ang_vel,
                # reference,
                phase_sin_cos_obs
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
        # + zero-calibration offset: the policy commands a BELIEVED angle,
        # the joint physically goes to believed + offset (see
        # cfg.joint_zero_offset_range_deg).
        position_targets = (
            self.robot.data.default_joint_pos[:, joint_ids]
            + self.cfg.action_scale * self._delayed_actions[:, :num_actions]
            + self._joint_zero_offset_rad[:, :num_actions]
        )

        # EMA low-pass on the final decoded target -- see cfg.action_smoothing.
        # Mirrors robot_deploy.py's --action-smoothing exactly (same formula,
        # applied at the same point in the pipeline: after decode, before
        # anything downstream sees the target) so training and deployment
        # share identical low-level dynamics.
        self._smoothed_position_targets = (
            self.cfg.action_smoothing * position_targets
            + (1.0 - self.cfg.action_smoothing) * self._smoothed_position_targets
        )
        position_targets = self._smoothed_position_targets

        # For _get_rewards' target_limit_penalty -- see that comment for
        # why this needs to be the RAW target (pre-physics-clamping), not
        # the resulting joint_pos. Post-smoothing here to match what
        # robot_deploy.py actually validates/sends -- a real safety abort
        # would see the smoothed target, not the pre-filter one.
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
        # weighted_error feeds tracking_reward only -- error itself stays
        # unweighted for logging (tracking/*_error below reports true
        # degrees, not inflated by joint_tracking_weight).
        weighted_error = error * self._joint_tracking_weight

        tracking_reward = torch.exp(
            -5.0 * torch.sum(weighted_error**2, dim=1)
        ) - self.cfg.tracking_linear_penalty_weight * torch.sum(weighted_error.abs(), dim=1)


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
        ) - self.cfg.velocity_linear_penalty_weight * torch.sum(velocity_error.abs(), dim=1)

        # Penalize step-to-step action jitter. Uses the raw (pre-delay)
        # action, since the delay buffer's job is to simulate real
        # target->actuation lag -- jitter is a property of what the policy
        # *outputs*, independent of when it actually reaches the motor.
        action_rate_penalty = self.cfg.action_rate_penalty_weight * torch.sum(
            self._action_rate_joint_weight * (self.actions - self._prev_actions) ** 2, dim=1
        )
        # Per-step target change beyond cfg.step_limit_threshold_deg -- see
        # cfg.step_limit_penalty_weight's comment. Must be computed BEFORE
        # _prev_actions is overwritten just below.
        step_delta_rad = self.cfg.action_scale * (self.actions - self._prev_actions).abs()
        step_excess_rad = torch.clamp(
            step_delta_rad - math.radians(self.cfg.step_limit_threshold_deg),
            min=0.0,
            max=math.radians(self.cfg.step_limit_penalty_max_excess_deg),
        )
        step_limit_penalty = self.cfg.step_limit_penalty_weight * torch.sum(self._step_limit_joint_weight * step_excess_rad, dim=1)
        step_over_rate = (step_delta_rad > math.radians(self.cfg.step_limit_threshold_deg)).any(dim=1).float().mean()
        mean_max_step_deg = torch.rad2deg(step_delta_rad.max(dim=1).values).mean()
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
        #
        # Margined the same way joint_limit_penalty is (cfg.target_limit_margin_deg)
        # -- previously this was exactly zero until a target actually
        # crossed the hard limit, so nothing pushed the policy to stay
        # clear of it with any buffer, only to not overshoot by much once
        # already past it. A real safety abort has zero tolerance
        # regardless of overshoot size, so the training signal needs to
        # start well before the boundary, not at it.
        # Per-joint margin now (cfg.target_limit_margin_deg is a dict,
        # matched to self._target_limit_margin_rad in __init__) -- roll
        # gets a wider buffer than the rest, see that field's comment.
        margin_rad = self._target_limit_margin_rad
        # robot_deploy.py checks the BELIEVED target against its limits,
        # so remove the zero-calibration offset (cfg.joint_zero_offset_range_deg).
        believed_targets = self._last_position_targets - self._joint_zero_offset_rad[:, :self._last_position_targets.shape[1]]
        target_over_limit = (
            torch.clamp(believed_targets - (upper - margin_rad), min=0.0)
            + torch.clamp((lower + margin_rad) - believed_targets, min=0.0)
        )
        # Capped before squaring -- see cfg.target_limit_penalty_max_overshoot_deg's
        # comment. A single uncapped outlier here is what blew up the value
        # function in a prior run.
        max_overshoot_rad = math.radians(self.cfg.target_limit_penalty_max_overshoot_deg)
        target_over_limit = torch.clamp(target_over_limit, max=max_overshoot_rad)
        # over**2 + linear_coef*over -- the linear term gives a
        # non-vanishing gradient down to zero overshoot (see
        # cfg.target_limit_penalty_linear_coef's comment); the quadratic
        # still scales the signal up for large overshoots.
        target_limit_penalty = self.cfg.target_limit_penalty_weight * torch.sum(
            target_over_limit ** 2
            + self.cfg.target_limit_penalty_linear_coef * target_over_limit,
            dim=1,
        )

        # Keep the body level: projected_gravity_b's xy components vanish
        # exactly when base_link is upright, independent of yaw heading
        # (see _get_observations). This is now load-bearing, not cosmetic
        # -- the base is free-floating (qmini_urdf-2legs.usda's root fixed
        # joint was removed), so nothing else in the reward stops the
        # policy from just letting the robot tip over while it chases
        # tracking_reward with its legs.
        #
        # MARGINED around zero tilt, not measured from dead-upright -- see
        # cfg.orientation_margin_deg's comment. tilt_sin is the xy-plane
        # norm of the (unit) gravity vector, i.e. sin(tilt angle); only the
        # amount past margin_sin (also a sin, so the two subtract directly
        # in the same units) is penalized, so a lean up to the margin gets
        # the full undiscounted bonus instead of already paying for it.
        projected_gravity = self.robot.data.projected_gravity_b
        tilt_sin = torch.sqrt(torch.clamp(torch.sum(projected_gravity[:, :2] ** 2, dim=1), min=0.0))
        margin_sin = math.sin(math.radians(self.cfg.orientation_margin_deg))
        orientation_error = torch.clamp(tilt_sin - margin_sin, min=0.0) ** 2
        orientation_reward = torch.exp(-self.cfg.orientation_reward_scale * orientation_error)

        # ROLL specifically -- see cfg.roll_reward_weight's comment. Same
        # margined-exp shape as orientation_reward above, just isolated to
        # projected_gravity_b's X component (roll) instead of the combined
        # xy norm, with its own undiluted margin/weight.
        roll_sin = projected_gravity[:, 0].abs()
        roll_margin_sin = math.sin(math.radians(self.cfg.roll_margin_deg))
        roll_error = torch.clamp(roll_sin - roll_margin_sin, min=0.0) ** 2
        roll_reward = torch.exp(-self.cfg.orientation_reward_scale * roll_error)

        # Foot-height swing reward -- see cfg.foot_swing_reward_weight's
        # comment for the full rationale. left/right_swing_target is how
        # much the reference clip's own knee angle says this foot should
        # be lifted right now (0=stance, 1=full swing, see the
        # _swing_extreme derivation in __init__); left/right_clearance is
        # how much of the target height the foot has actually achieved.
        # Multiplying them means: no reward for lifting when the reference
        # doesn't call for it (swing_target~0 keeps this near 0 regardless
        # of clearance, so it can't be farmed by just standing on tiptoes
        # all the time), and no reward during swing until the foot is
        # actually off the ground (clearance~0 until it climbs).
        left_swing_target = torch.clamp(
            (reference[:, self._left_knee_idx] - self._left_knee_stance_rad)
            / (self._left_knee_swing_extreme_rad - self._left_knee_stance_rad),
            min=0.0, max=1.0,
        )
        right_swing_target = torch.clamp(
            (reference[:, self._right_knee_idx] - self._right_knee_stance_rad)
            / (self._right_knee_swing_extreme_rad - self._right_knee_stance_rad),
            min=0.0, max=1.0,
        )
        # World-frame HEEL and TOE positions -- see cfg.left/right_heel_local_m
        # and left/right_toe_local_m's comments for why BOTH are required.
        # Tracking the heel alone replaced one exploit (dorsiflexion --
        # toes up, heel planted -- gaming the old ankle-anchored body
        # origin) with its mirror image (plantarflexion -- heel up, toes
        # planted -- gaming a heel-only measurement, since the heel
        # genuinely rises in world space while the foot pivots on the
        # still-grounded toe). Requiring BOTH points to clear rules out
        # rotating around either one alone.
        left_heel_pos_w = self.robot.data.body_pos_w[:, self._left_foot_body_idx, :] + quat_apply(
            self.robot.data.body_quat_w[:, self._left_foot_body_idx, :], self._left_heel_local_offset
        )
        right_heel_pos_w = self.robot.data.body_pos_w[:, self._right_foot_body_idx, :] + quat_apply(
            self.robot.data.body_quat_w[:, self._right_foot_body_idx, :], self._right_heel_local_offset
        )
        left_toe_pos_w = self.robot.data.body_pos_w[:, self._left_foot_body_idx, :] + quat_apply(
            self.robot.data.body_quat_w[:, self._left_foot_body_idx, :], self._left_toe_local_offset
        )
        right_toe_pos_w = self.robot.data.body_pos_w[:, self._right_foot_body_idx, :] + quat_apply(
            self.robot.data.body_quat_w[:, self._right_foot_body_idx, :], self._right_toe_local_offset
        )

        # Per-env stance-height reference, (re-)captured once EACH env has
        # been settled for foot_swing_stance_settle_steps since its last
        # reset (see self._foot_stance_height_captured's __init__ comment
        # for why this must be per-episode, and self._steps_since_reset's
        # comment for why it can't be the very first post-reset step
        # either) -- avoids anchoring on a settling transient from that
        # reset's randomized startup pose, so this reads a genuinely
        # representative "quiet stance" that every later comparison in
        # THIS episode will be measured against. Heel and toe references
        # are captured together, gated by the same flag.
        self._steps_since_reset += 1
        needs_capture = (~self._foot_stance_height_captured) & (
            self._steps_since_reset >= self.cfg.foot_swing_stance_settle_steps
        )
        if needs_capture.any():
            self._left_foot_stance_height[needs_capture] = left_heel_pos_w[needs_capture, 2]
            self._right_foot_stance_height[needs_capture] = right_heel_pos_w[needs_capture, 2]
            self._left_toe_stance_height[needs_capture] = left_toe_pos_w[needs_capture, 2]
            self._right_toe_stance_height[needs_capture] = right_toe_pos_w[needs_capture, 2]
            self._foot_stance_height_captured[needs_capture] = True

        left_heel_height = left_heel_pos_w[:, 2] - self._left_foot_stance_height
        right_heel_height = right_heel_pos_w[:, 2] - self._right_foot_stance_height
        left_toe_height = left_toe_pos_w[:, 2] - self._left_toe_stance_height
        right_toe_height = right_toe_pos_w[:, 2] - self._right_toe_stance_height

        # Per-phase reference height target -- see cfg.foot_height_target_
        # floor_m's comment for why this replaced a flat constant. Columns
        # match gen_reference_heights.py's write order:
        # [left_heel, left_toe, right_heel, right_toe].
        foot_height_reference = self.foot_height_motion.sample(self.motion_time)
        target_left_heel = torch.clamp(foot_height_reference[:, 0], min=self.cfg.foot_height_target_floor_m)
        target_left_toe = torch.clamp(foot_height_reference[:, 1], min=self.cfg.foot_height_target_floor_m)
        target_right_heel = torch.clamp(foot_height_reference[:, 2], min=self.cfg.foot_height_target_floor_m)
        target_right_toe = torch.clamp(foot_height_reference[:, 3], min=self.cfg.foot_height_target_floor_m)

        # Clearance requires BOTH the heel AND the toe to have risen --
        # the min(), not e.g. an average, is what actually rules out
        # rotating around either fixed contact point (a rotation that
        # lifts one point while the other stays near zero would otherwise
        # still earn partial credit through an average).
        left_heel_clearance = torch.clamp(left_heel_height / target_left_heel, min=0.0, max=1.0)
        right_heel_clearance = torch.clamp(right_heel_height / target_right_heel, min=0.0, max=1.0)
        left_toe_clearance = torch.clamp(left_toe_height / target_left_toe, min=0.0, max=1.0)
        right_toe_clearance = torch.clamp(right_toe_height / target_right_toe, min=0.0, max=1.0)
        left_clearance = torch.minimum(left_heel_clearance, left_toe_clearance)
        right_clearance = torch.minimum(right_heel_clearance, right_toe_clearance)
        foot_swing_reward = self.cfg.foot_swing_reward_weight * (
            self.cfg.foot_swing_left_weight * left_swing_target * left_clearance
            + self.cfg.foot_swing_right_weight * right_swing_target * right_clearance
        )
        # Withhold entirely during the push-recovery cooldown window -- see
        # cfg.foot_swing_push_cooldown_steps' comment for why. Per-env now
        # (see cfg.push_interval_range_s's comment) -- pushes no longer
        # fire for every env on the same step, so "steps since the last
        # push" is a per-env tensor, not one shared global scalar.
        in_push_cooldown = self._steps_since_push < self.cfg.foot_swing_push_cooldown_steps

        # DIAGNOSTIC ONLY -- not fed into the training reward, computed here
        # purely so it can be logged below. Added 2026-09-06 after a direct
        # video review (2026-09-06_01-16-02 run, iteration ~29800) noted the
        # policy appeared to lift a leg briefly right after being pushed --
        # a real, if small, protective-stepping reflex -- but this can never
        # show up in tracking/foot_swing_reward or motion/*_clearance above,
        # since both are unconditionally zeroed/computed pre-push during
        # exactly this window by design (see this cfg's comment for why
        # that's still correct for the actual reward: push-induced motion
        # isn't a genuine policy choice). This metric exists only so we can
        # watch whether that reflex is real and improving over training,
        # without changing what the policy is actually optimized against.
        cooldown_mask = in_push_cooldown.float()
        n_in_cooldown = cooldown_mask.sum().clamp_min(1.0)
        foot_swing_during_push_cooldown = (foot_swing_reward * cooldown_mask).sum() / n_in_cooldown

        foot_swing_reward = torch.where(in_push_cooldown, torch.zeros_like(foot_swing_reward), foot_swing_reward)

        # Penalize foot height past target+margin -- see
        # cfg.foot_overswing_penalty_weight's comment. Uses the max of
        # heel/toe (either point being far too high means the foot is too
        # high), unlike foot_swing_reward's min() (which exists to rule out
        # gaming clearance by rotating around one fixed point -- not a
        # concern here, a real overswing lifts both). Each point is
        # measured against its OWN per-phase reference target (heel and
        # toe targets differ slightly), not a single shared constant --
        # see foot_height_target_floor_m's comment.
        left_heel_overswing = torch.clamp(
            left_heel_height - (target_left_heel + self.cfg.foot_overswing_margin_m), min=0.0
        )
        left_toe_overswing = torch.clamp(
            left_toe_height - (target_left_toe + self.cfg.foot_overswing_margin_m), min=0.0
        )
        right_heel_overswing = torch.clamp(
            right_heel_height - (target_right_heel + self.cfg.foot_overswing_margin_m), min=0.0
        )
        right_toe_overswing = torch.clamp(
            right_toe_height - (target_right_toe + self.cfg.foot_overswing_margin_m), min=0.0
        )
        left_overswing = torch.maximum(left_heel_overswing, left_toe_overswing)
        right_overswing = torch.maximum(right_heel_overswing, right_toe_overswing)
        overswing_max_m = self.cfg.foot_overswing_penalty_max_m
        left_overswing = torch.clamp(left_overswing, max=overswing_max_m)
        right_overswing = torch.clamp(right_overswing, max=overswing_max_m)
        foot_overswing_penalty = self.cfg.foot_overswing_penalty_weight * (left_overswing + right_overswing)
        foot_overswing_penalty = torch.where(
            in_push_cooldown, torch.zeros_like(foot_overswing_penalty), foot_overswing_penalty
        )

        # Track commanded base xy velocity (currently always zero -- stay
        # in place) -- see cfg.position_reward_weight's comment for why
        # this is velocity, not a position anchor.
        commanded_lin_vel_xy = torch.zeros(self.num_envs, 2, device=self.device)
        base_vel_error = torch.sum(
            (self.robot.data.root_lin_vel_w[:, :2] - commanded_lin_vel_xy) ** 2, dim=1,
        )
        position_reward = torch.exp(-self.cfg.position_reward_scale * base_vel_error)

        # Track commanded base yaw rate (currently always zero -- stay
        # facing the same way) -- see cfg.heading_reward_weight's comment.
        commanded_yaw_rate = torch.zeros(self.num_envs, device=self.device)
        yaw_rate_error = (self.robot.data.root_ang_vel_w[:, 2] - commanded_yaw_rate) ** 2
        heading_reward = torch.exp(-self.cfg.heading_reward_scale * yaw_rate_error)

        # ABSOLUTE heading tracking -- see cfg.heading_deviation_penalty_weight's
        # comment for the full story (heading_reward above only tracks
        # yaw RATE, never corrects accumulated drift, which is exactly
        # what real hardware showed). Margined and linear-then-capped, same
        # pattern as target_limit_penalty.
        _, _, current_yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
        heading_deviation_rad = wrap_to_pi(current_yaw - self._reset_yaw).abs()
        heading_deviation_margin_rad = math.radians(self.cfg.heading_deviation_margin_deg)
        heading_deviation_over = torch.clamp(heading_deviation_rad - heading_deviation_margin_rad, min=0.0)
        heading_deviation_max_rad = math.radians(self.cfg.heading_deviation_penalty_max_deg)
        heading_deviation_over = torch.clamp(heading_deviation_over, max=heading_deviation_max_rad)
        heading_deviation_penalty = self.cfg.heading_deviation_penalty_weight * heading_deviation_over

        # ABSOLUTE position tracking -- see cfg.position_deviation_penalty_weight's
        # comment. Same margin/linear/capped pattern as heading_deviation_penalty
        # just above, applied to xy displacement instead of yaw.
        position_deviation_m = torch.norm(
            self.robot.data.root_pos_w[:, :2] - self._reset_pos_xy, dim=1,
        )
        position_deviation_over = torch.clamp(
            position_deviation_m - self.cfg.position_deviation_margin_m, min=0.0
        )
        position_deviation_over = torch.clamp(
            position_deviation_over, max=self.cfg.position_deviation_penalty_max_m
        )
        position_deviation_penalty = self.cfg.position_deviation_penalty_weight * position_deviation_over

        # Same push-recovery cooldown as foot_swing_reward above, and for
        # the same underlying reason: a push forces real base velocity/yaw
        # rate that the policy didn't choose and can't avoid, so scoring it
        # against these terms during the cooldown window is pure noise, not
        # signal -- and unlike foot_swing_reward, nothing was protecting
        # these two from it, because both were built while pushes were
        # disabled and the interaction never came up. Added after
        # re-enabling push_interval_range_s made position_reward/
        # heading_reward visibly noisier (base_speed_cmps/yaw_rate_dps
        # spiking right when a push lands) without that being a real
        # behavior change.
        position_reward = torch.where(in_push_cooldown, torch.zeros_like(position_reward), position_reward)
        heading_reward = torch.where(in_push_cooldown, torch.zeros_like(heading_reward), heading_reward)
        position_deviation_penalty = torch.where(
            in_push_cooldown, torch.zeros_like(position_deviation_penalty), position_deviation_penalty
        )
        heading_deviation_penalty = torch.where(
            in_push_cooldown, torch.zeros_like(heading_deviation_penalty), heading_deviation_penalty
        )

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
            + self.cfg.roll_reward_weight * roll_reward
            + foot_swing_reward
            + self.cfg.position_reward_weight * position_reward
            + self.cfg.heading_reward_weight * heading_reward
            - action_rate_penalty
            - termination_penalty
            - joint_limit_penalty
            - target_limit_penalty
            - heading_deviation_penalty
            - position_deviation_penalty
            - foot_overswing_penalty
            - step_limit_penalty
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
            "tracking/foot_swing_reward": foot_swing_reward.mean(),
            # See foot_swing_during_push_cooldown's comment above -- watch
            # this to see whether push-recovery stepping is real and
            # improving, since it's invisible in the reward-facing metric
            # above by design.
            "tracking/foot_swing_during_push_cooldown": foot_swing_during_push_cooldown,
            "tracking/foot_overswing_penalty": foot_overswing_penalty.mean(),
            "tracking/step_limit_penalty": step_limit_penalty.mean(),
            "tracking/step_over_rate": step_over_rate,
            "tracking/mean_max_step_deg": mean_max_step_deg,
            # Logged as the MINIMUM of heel/toe height (the bottleneck that
            # actually determines clearance below), not heel alone -- see
            # cfg.left/right_toe_local_m's comment for why heel alone isn't
            # the whole picture anymore.
            "motion/left_foot_clearance_cm": torch.minimum(left_heel_height, left_toe_height)[0] * 100.0,
            "motion/right_foot_clearance_cm": torch.minimum(right_heel_height, right_toe_height)[0] * 100.0,
            "motion/left_swing_target": left_swing_target[0],
            "motion/right_swing_target": right_swing_target[0],
            "orientation/reward": orientation_reward.mean(),
            "orientation/roll_reward": roll_reward.mean(),
            "orientation/roll_deg": torch.rad2deg(torch.asin(torch.clamp(roll_sin, max=1.0))).mean(),
            "orientation/termination_penalty": termination_penalty.mean(),
            "orientation/fall_rate": fell.float().mean(),
            "orientation/gravity_x": projected_gravity[0, 0],
            "orientation/gravity_y": projected_gravity[0, 1],
            "orientation/gravity_z": projected_gravity[0, 2],
            "position/reward": position_reward.mean(),
            "position/base_speed_cmps": torch.sqrt(base_vel_error[0]) * 100.0,
            "position/deviation_penalty": position_deviation_penalty.mean(),
            "position/deviation_cm": position_deviation_m.mean() * 100.0,
            # Fraction of envs still under startup support THIS step -- see
            # cfg.startup_support_duration_range_s's comment. Should track
            # roughly (mean duration / episode_length_s) at steady state;
            # watch this alongside foot_swing_reward/fall_rate right after
            # each env's own support ends (not directly loggable per-env
            # here, but a sudden foot_swing_reward dip population-wide
            # around when most envs' support has just ended would be the
            # signature of the same "struggles right after release" hardware showed).
            "tracking/startup_support_active_frac": self._support_active.float().mean(),
            "heading/reward": heading_reward.mean(),
            "heading/yaw_rate_dps": torch.rad2deg(torch.sqrt(yaw_rate_error[0])),
            "heading/deviation_penalty": heading_deviation_penalty.mean(),
            "heading/deviation_deg": torch.rad2deg(heading_deviation_rad.mean()),
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

            # TEMP DIAGNOSTIC -- now tracks BOTH heel and toe (see
            # cfg.left/right_toe_local_m's comment for why heel alone was
            # still gameable via the mirror-image plantarflexion exploit,
            # confirmed via direct video: heel visibly lifting while toes
            # stayed planted on the ground). Watch that left_clearance
            # stays near zero whenever EITHER heel_height OR toe_height is
            # near zero -- if one is high and clearance still reads high,
            # the min() below has a bug. Remove once foot_swing_reward
            # looks trustworthy against video again.
            print(
                f"foot_swing diag: heel_left_cm={float(left_heel_height[0])*100:.3f}  "
                f"toe_left_cm={float(left_toe_height[0])*100:.3f}  "
                f"left_clearance={float(left_clearance[0]):.3f}  "
                f"left_swing_target={float(left_swing_target[0]):.3f}  needs_capture_now={bool(needs_capture[0])}  "
                f"steps_since_reset={int(self._steps_since_reset[0])}  "
                f"pct_envs_captured={float(self._foot_stance_height_captured.float().mean())*100:.1f}%"
            )

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

        # Capture this env's world-frame yaw right now as its heading
        # reference for cfg.heading_deviation_penalty_weight -- computed
        # from the quaternion actually just written above (not assumed to
        # be exactly 0) so this stays correct even if default_root_state's
        # orientation ever changes.
        _, _, reset_yaw = euler_xyz_from_quat(default_root_state[:, 3:7])
        self._reset_yaw[env_ids] = reset_yaw
        self._reset_pos_xy[env_ids] = default_root_state[:, :2]

        # Fresh per-env support duration for this new episode -- see
        # cfg.startup_support_duration_range_s's comment for why this
        # resamples every reset (unlike push's countdown).
        support_duration_s = sample_uniform(
            *self.cfg.startup_support_duration_range_s, (env_ids.numel(),), device=self.device,
        )
        self._support_end_step[env_ids] = (support_duration_s / self.step_dt).round().long()

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
        # Sample this episode's zero-calibration offsets first: the robot
        # starts at its BELIEVED default pose = true default + offset.
        self._randomize_joint_zero_offset(env_ids)
        randomized_joint_pos = default_joint_pos + joint_pos_noise + self._joint_zero_offset_rad[env_ids]
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

        # Force a fresh foot-swing stance-height capture for these envs on
        # their next _get_rewards() call -- see that flag's __init__
        # comment for why this must happen every reset, not just once.
        # env_ids may be a plain Sequence (e.g. a Python list, per this
        # method's own type hint) rather than a tensor, so index via
        # as_tensor rather than assuming boolean/tensor indexing works
        # directly (mirrors _randomize_base_mass's same env_ids handling).
        env_ids_t = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(list(env_ids), device=self.device)
        self._foot_stance_height_captured[env_ids_t] = False
        self._steps_since_reset[env_ids_t] = 0

        # Reset EMA smoothing state to this pipeline's own anchor pose (see
        # cfg.action_smoothing) so a fresh episode's first smoothed target
        # isn't pulled toward whatever a DIFFERENT episode's rollout last
        # commanded.
        self._smoothed_position_targets[env_ids_t] = (
            self.robot.data.default_joint_pos[env_ids_t, :self.cfg.action_space]
            + self._joint_zero_offset_rad[env_ids_t, :self.cfg.action_space]
        )

        self._randomize_action_delay(env_ids)
        self._randomize_actuator_gains(env_ids)
        self._randomize_gyro_bias(env_ids)
        self._randomize_imu_mount_bias(env_ids)
        self._randomize_base_mass(env_ids)
        self._randomize_friction(env_ids)
        # Fresh, independent push countdown for these envs -- see
        # cfg.push_interval_range_s's comment for why this must be
        # per-env, not a shared global schedule.
        self._resample_push_countdown(env_ids_t)
        self._steps_since_push[env_ids_t] = 0

    def _randomize_joint_zero_offset(self, env_ids: Sequence[int]):
        """Per-episode zero-calibration error, see cfg.joint_zero_offset_range_deg."""
        unit = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_joints), device=self.device)
        self._joint_zero_offset_rad[env_ids] = unit * self._joint_zero_offset_range_rad

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

    def _randomize_friction(self, env_ids: Sequence[int]):
        """Per-episode friction on all of this env's robot shapes, see
        cfg.friction_randomization_range (ground is 1.0 with "multiply", so
        these are the effective values)."""
        materials = self.robot.root_physx_view.get_material_properties()  # (N, shapes, 3) on CPU
        ids = (env_ids if torch.is_tensor(env_ids) else torch.as_tensor(list(env_ids))).to(materials.device)
        n = ids.numel()
        static = sample_uniform(*self.cfg.friction_randomization_range, (n, 1), device=materials.device)
        dynamic = static * sample_uniform(*self.cfg.dynamic_friction_ratio_range, (n, 1), device=materials.device)
        materials[ids, :, 0] = static
        materials[ids, :, 1] = dynamic
        self.robot.root_physx_view.set_material_properties(materials, ids)

    def _randomize_action_delay(self, env_ids: Sequence[int]):
        # Per-joint now (see cfg.action_delay_range_steps' comment) --
        # self._action_delay_low/high are (num_joints,), built once in
        # __init__ by matching each joint's own suffix. Sampled as a float
        # unit draw scaled per-joint rather than torch.randint (which
        # doesn't take per-element low/high), then floored into the
        # inclusive [low, high] range.
        n = len(env_ids)
        unit = torch.rand(n, self.cfg.action_space, device=self.device)
        span = (self._action_delay_high - self._action_delay_low + 1).float()
        sampled = self._action_delay_low + (unit * span).long()
        sampled = torch.minimum(sampled, self._action_delay_high)
        self._action_delay_steps[env_ids] = sampled
        # Clear stale pre-reset actions out of the buffer for these envs so
        # a delayed readback right after reset can't replay an action from
        # a different episode.
        self._action_buffer[env_ids, :, :] = 0.0

    def _resample_push_countdown(self, env_ids: Sequence[int]):
        """Draws a fresh, independent steps-until-next-push for these envs
        from cfg.push_interval_range_s -- called at reset and again
        immediately after each push, so no two envs stay locked to a
        shared schedule the way a single global interval would. See
        cfg.push_interval_range_s's comment."""
        env_ids_t = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(list(env_ids), device=self.device)
        low_s, high_s = self.cfg.push_interval_range_s
        interval_s = sample_uniform(low_s, high_s, (len(env_ids_t),), device=self.device)
        self._steps_until_push[env_ids_t] = torch.clamp((interval_s / self.step_dt).round().long(), min=1)

    def _randomize_actuator_gains(self, env_ids: Sequence[int]):
        for name, actuator in self.robot.actuators.items():
            defaults = self._default_actuator_gains[name]
            for field, (low, high) in self.cfg.gain_randomization_range.items():
                base = defaults[field][env_ids]
                scale = sample_uniform(low, high, base.shape, device=self.device)
                getattr(actuator, field)[env_ids] = base * scale


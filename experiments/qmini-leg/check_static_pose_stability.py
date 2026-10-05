#!/usr/bin/env python3
"""
check_static_pose_stability.py

Does a given STATIC leg pose, held constant by PD control under real
gravity/contact physics, stay upright? Unlike check_reference_stability.py's
open-loop GAIT tracking (which turned out uninformative -- single-support
stepping isn't statically stable regardless of reference quality, see that
script's 2026-09-28 notes), holding one fixed double-support pose IS a
meaningful, answerable dynamics question: if the pose's implied CoM sits
over the base of support, PD holding it stays upright; if not, it tips, and
that's a real signal, not a structural false negative.

Built 2026-09-29 to directly test the user's report that
robot_config_qmini_standing.json's default_pos_deg (left_pitch/knee/ankle
-16/+10/+1) holds stable in sim, while robot_config_qmini_stepinplace.json's
(-16.5656/+22.5569/+12.9913, == qmini.py's InitialStateCfg == the CURRENT
training/deployment anchor pose, == keyframes_standing_still.json's
keyframe-0) falls over backward -- despite that second pose's own code
comment claiming tune_stance_lean_isaaclab.py validated it (worst_tilt
3.2deg). Also used to test whether keyframes_step_in_place_..._no_offset.
json's own keyframe-0 (the "remove the offset" candidate) would hold up
better or worse than the current stepinplace reference's keyframe-0 --
motivated by check_reference_balance_fk.py's finding that removing the
offset moves both feet further BEHIND the body-frame origin at that frame,
not closer to center (a purely kinematic proxy with no real CoM data; this
script is the real-dynamics follow-up that FK check called for).

Give it EITHER --left-pitch/--left-knee/--left-ankle directly (right leg
is assumed mirrored -- true of every pose seen in this project so far,
yaw/roll assumed 0) OR --from-keyframes PATH --frame-index N to pull the
pose from an existing keyframes_*.json (any frame, not just 0).

Holds the pose CONSTANT (re-issues the SAME target action every step,
regardless of elapsed time) for --seconds and reports fall_rate/tilt over
time -- a rising tilt trend or any fall is real evidence this pose is not a
statically-held-stable double-support stance.

USAGE
-----
    # direct pose (degrees)
    ./isaaclab.sh -p check_static_pose_stability.py \\
        --label robot_config_standing --left-pitch -16.0 --left-knee 10.0 --left-ankle 1.0 \\
        --seconds 5 --headless

    # pose pulled from a keyframes file's frame 0
    ./isaaclab.sh -p check_static_pose_stability.py \\
        --label no_offset_kf0 --from-keyframes keyframes_step_in_place_all_joints_2x_no_offset.json \\
        --frame-index 0 --seconds 5 --headless

ONE POSE PER INVOCATION -- see check_reference_stability.py's docstring for
the confirmed multi-gym.make()-per-process hang; this script has the exact
same risk and isn't worth re-verifying independently.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--left-pitch", type=float, default=None, help="degrees")
    parser.add_argument("--left-knee", type=float, default=None, help="degrees")
    parser.add_argument("--left-ankle", type=float, default=None, help="degrees")
    parser.add_argument("--from-keyframes", type=str, default=None)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--task", type=str, default="DroidPlayground-QMini-Leg")
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spawn-in-pose", action="store_true",
                        help="reset directly into the held pose instead of ramping from the (unstable) anchor")
    args_cli, extra = parser.parse_known_args()

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import copy
    import json

    import gymnasium as gym
    import torch

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.hydra import hydra_task_config

    import droidplayground.tasks  # noqa: F401
    from droidplayground.tasks.direct.droidplayground import qmini_leg_env as qle

    if args_cli.from_keyframes:
        kf_path = Path(args_cli.from_keyframes)
        if not kf_path.is_absolute():
            kf_path = Path(qle.__file__).parent / kf_path
        data = json.load(open(kf_path))
        order = data["joint_order"]
        idx = {n: i for i, n in enumerate(order)}
        frame = data["keyframes"][args_cli.frame_index][1]
        lp = frame[idx["left_pitch"]]
        lk = frame[idx["left_knee"]]
        la = frame[idx["left_ankle"]]
        print(f"[DIAG] pose from {kf_path.name} frame {args_cli.frame_index}: pitch={lp:.4f} knee={lk:.4f} ankle={la:.4f}", flush=True)
    else:
        if None in (args_cli.left_pitch, args_cli.left_knee, args_cli.left_ankle):
            parser.error("either --from-keyframes or all of --left-pitch/--left-knee/--left-ankle")
        lp, lk, la = args_cli.left_pitch, args_cli.left_knee, args_cli.left_ankle

    @hydra_task_config(args_cli.task, args_cli.agent)
    def run(env_cfg, agent_cfg):
        print("[DIAG] entered run (hydra config composition done)", flush=True)
        cfg_i = copy.deepcopy(env_cfg)
        cfg_i.scene.num_envs = args_cli.num_envs
        cfg_i.seed = args_cli.seed

        print("[DIAG] about to gym.make()", flush=True)
        env = gym.make(args_cli.task, cfg=cfg_i, render_mode=None)
        print("[DIAG] gym.make() returned", flush=True)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        unwrapped = env.unwrapped
        device = unwrapped.device
        num_joints = unwrapped.num_joints
        action_scale = cfg_i.action_scale

        # Build the held target in the ROBOT's own joint order -- NOTE this
        # is the raw USD/articulation naming (e.g. "Revolute_left_pitch"),
        # NOT the "left_hip_pitch" hardware-side naming used in
        # robot_config json/robot_deploy.py (those only match up via
        # export_policy_for_deployment.py's explicit --joint-order at
        # export time). Same removeprefix("Revolute_") lookup
        # _load_reference_keyframes() uses in qmini_leg_env.py, mapped to
        # keyframes-file-style plain names ("left_pitch" etc). Mirrors
        # right = -left for pitch/knee/ankle, 0 for yaw/roll -- true of
        # every pose tested in this project so far.
        names = list(unwrapped.robot.joint_names[:num_joints])
        target_deg = {
            "left_yaw": 0.0, "right_yaw": 0.0,
            "left_roll": 0.0, "right_roll": 0.0,
            "left_pitch": lp, "right_pitch": -lp,
            "left_knee": lk, "right_knee": -lk,
            "left_ankle": la, "right_ankle": -la,
        }
        target_rad = torch.tensor(
            [math.radians(target_deg[n.removeprefix("Revolute_")]) for n in names], device=device
        )
        print(f"[DIAG] held target (deg, robot order {names}): "
              f"{[round(target_deg[n.removeprefix('Revolute_')], 2) for n in names]}", flush=True)

        num_steps = max(1, round(args_cli.seconds / unwrapped.step_dt))

        ever_fell = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
        tilt_sum = tilt_count = 0
        tilt_max = 0.0
        per_second_tilt_max = [0.0] * (int(args_cli.seconds) + 1)
        steps_per_second = max(1, round(1.0 / unwrapped.step_dt))

        # Ramp from the reset pose (default_joint_pos -- the current
        # training/deployment anchor) to the held target over RAMP_STEPS,
        # THEN hold constant -- matching robot_deploy.py's real startup_
        # sequence (a 2.00s move), not an instant snap to the target.
        # Added after a first pass without this ramp showed a 31% fall
        # rate on a pose the user separately confirmed holds stable on
        # hardware/in a prior sim check -- an abrupt multi-joint snap on
        # step 0 is itself a real (if artificial) shock this test
        # shouldn't be attributing to the target pose's own stability.
        RAMP_STEPS = max(1, round(2.0 / unwrapped.step_dt))

        # --spawn-in-pose (2026-10-04): the ramp above starts from the
        # training anchor, which is itself statically unstable (tips
        # backward), so with the ramp EVERY env falls before reaching the
        # target and the "settled" numbers come from post-reset envs. This
        # instead makes the target the reset pose and corrects the spawn
        # height so the feet start on the ground (foot-body z after one
        # step from the anchor reset is the on-ground reference).
        if args_cli.spawn_in_pose:
            feet = [unwrapped.robot.body_names.index(cfg_i.left_foot_body_name),
                    unwrapped.robot.body_names.index(cfg_i.right_foot_body_name)]
            hold = torch.zeros(args_cli.num_envs, num_joints, device=device)

            def foot_z():
                env.step(hold)
                return float(unwrapped.robot.data.body_pos_w[:, feet, 2].min(dim=1).values.mean())

            lift = 0.05
            with torch.inference_mode():
                env.reset()
                z_ground = foot_z()
                unwrapped.robot.data.default_joint_pos[:, :num_joints] = target_rad.unsqueeze(0)
                unwrapped.robot.data.default_root_state[:, 2] += lift
                env.reset()
                z_lifted = foot_z()
                unwrapped.robot.data.default_root_state[:, 2] -= (z_lifted - z_ground)
                env.reset()
            print(f"[DIAG] spawn-in-pose root z correction {1000 * (lift - (z_lifted - z_ground)):+.1f} mm", flush=True)
        default_pos0 = unwrapped.robot.data.default_joint_pos[:, :num_joints].clone()

        late_pitch, late_err = [], []
        obs = env.get_observations()
        for step in range(num_steps):
            with torch.inference_mode():
                if step < RAMP_STEPS:
                    frac = (step + 1) / RAMP_STEPS
                    ramped_target = default_pos0 + frac * (target_rad.unsqueeze(0) - default_pos0)
                else:
                    ramped_target = target_rad.unsqueeze(0)
                default_pos = unwrapped.robot.data.default_joint_pos[:, :num_joints]
                actions = (ramped_target - default_pos) / action_scale
                obs, _rew, dones, extras = env.step(actions)

            gxy = unwrapped.robot.data.projected_gravity_b[:, :2]
            tilt_deg = torch.rad2deg(torch.asin(torch.clamp(torch.norm(gxy, dim=1), max=1.0)))
            tilt_sum += float(tilt_deg.sum())
            tilt_count += args_cli.num_envs
            tilt_max = max(tilt_max, float(tilt_deg.max()))
            sec = step // steps_per_second
            if sec < len(per_second_tilt_max):
                per_second_tilt_max[sec] = max(per_second_tilt_max[sec], float(tilt_deg.max()))

            dones_bool = dones.bool()
            time_outs = extras.get("time_outs")
            fell = dones_bool & ~time_outs.bool() if time_outs is not None else dones_bool
            ever_fell |= fell

            # Last-half signed pitch and actual - target per joint, over envs
            # that have not fallen -- directly comparable to the hardware
            # open-loop-ref CSVs (2026-10-04 ankle sweep).
            if step >= num_steps // 2:
                ok = ~ever_fell
                if ok.any():
                    g = unwrapped.robot.data.projected_gravity_b[ok]
                    late_pitch.append(torch.rad2deg(torch.asin(g[:, 1].clamp(-1, 1))).cpu())
                    late_err.append(torch.rad2deg(
                        unwrapped.robot.data.joint_pos[ok, :num_joints] - target_rad.unsqueeze(0)).cpu())

        print(
            f"\n[{args_cli.label}] frac_envs_ever_fell={float(ever_fell.float().mean()):.4f} "
            f"(of {args_cli.num_envs} envs over {args_cli.seconds}s)  "
            f"mean_tilt={tilt_sum / max(1, tilt_count):.2f}deg  max_tilt={tilt_max:.2f}deg"
        )
        print(f"[{args_cli.label}] per-second max tilt (deg): {[round(x, 1) for x in per_second_tilt_max]}")
        if late_pitch:
            lp_all = torch.cat(late_pitch)
            le_all = torch.cat(late_err)
            print(f"[{args_cli.label}] last-half (non-fallen envs) signed pitch mean={float(lp_all.mean()):+.2f}deg (+ = backward)")
            print(f"[{args_cli.label}] last-half mean actual - target (deg): "
                  + "  ".join(f"{n.removeprefix('Revolute_')}={float(le_all[:, i].mean()):+.2f}"
                              for i, n in enumerate(names)))
        print(
            "  (should stay small and flat if this pose is a real statically-stable stance; "
            "rising toward the episode's tilt-termination boundary means it's tipping over)"
        )

        env.close()

    run()
    simulation_app.close()


if __name__ == "__main__":
    main()

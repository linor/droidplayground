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
pose from an existing keyframes_*.json (any frame, not just 0) OR
--from-gait-library NPZ [--phase P] to hold a walking gait library's full
10-joint reference at zero command (2026-10-14: the walking deploy rest pose,
library v3's stance carries yaw/roll offsets, so nothing is mirrored here).

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
    parser.add_argument("--from-gait-library", type=str, default=None,
                        help="gait_library*.npz: hold its reference at zero command, --phase")
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--blend-from", type=str, default=None,
                        help="left pitch,knee,ankle (mirrored, yaw/roll 0): envs are split into --blend-steps groups "
                             "holding from + a*(pose - from), a = 0..1, one group per a")
    parser.add_argument("--blend-steps", type=int, default=8)
    parser.add_argument("--ankle-offsets", type=str, default=None,
                        help="comma list of left-ankle offsets in deg (right mirrored; negative = toes down / "
                             "body forward): envs are split into one group per offset added to the pose")
    parser.add_argument("--rigid-soles", action="store_true", help="foot_contact_stiffness = None (pre-2026-10-07 contact)")
    parser.add_argument("--friction", type=float, default=0.5, help="static = dynamic friction (real floor ~0.3)")
    parser.add_argument("--task", type=str, default="DroidPlayground-QMini-Leg")
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--usd", type=str, default=None, help="override the robot USD (see fix_foot_collisions.py)")
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

    full_pose = None
    if args_cli.from_gait_library:
        sys.path.insert(0, str(Path(__file__).parent))
        from walk_command import ReferenceGait
        lib_path = Path(args_cli.from_gait_library)
        if not lib_path.is_absolute() and not lib_path.exists():
            lib_path = Path(qle.__file__).parent / lib_path
        ref = ReferenceGait(lib_path).joint_ref([0.0, 0.0, 0.0], args_cli.phase)
        lib_to_usd = {"hip_yaw": "yaw", "hip_roll": "roll", "hip_pitch": "pitch", "knee": "knee", "ankle": "ankle"}
        full_pose = {}
        for k, v in ref.items():
            side, j = k.split("_", 1)
            full_pose[f"{side}_{lib_to_usd[j]}"] = math.degrees(v)
        lp = lk = la = None
        print(f"[DIAG] pose from {lib_path.name} cmd 0 phase {args_cli.phase}: "
              + " ".join(f"{k}={v:+.2f}" for k, v in full_pose.items()), flush=True)
    elif args_cli.from_keyframes:
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
        if args_cli.usd:
            cfg_i.robot_cfg.spawn.usd_path = args_cli.usd
            print(f"[DIAG] robot USD override: {args_cli.usd}", flush=True)
        # deterministic: no random joint zero offsets / friction (2026-10-05/06 additions)
        if hasattr(cfg_i, "joint_zero_offset_range_deg"):
            cfg_i.joint_zero_offset_range_deg = {k: 0.0 for k in cfg_i.joint_zero_offset_range_deg}
        if hasattr(cfg_i, "friction_randomization_range"):
            cfg_i.friction_randomization_range = (args_cli.friction, args_cli.friction)
            cfg_i.dynamic_friction_ratio_range = (1.0, 1.0)
        if hasattr(cfg_i, "gain_randomization_range"):
            cfg_i.gain_randomization_range = {k: (1.0, 1.0) for k in cfg_i.gain_randomization_range}
        if hasattr(cfg_i, "base_mass_randomization_range"):
            cfg_i.base_mass_randomization_range = (1.0, 1.0)
        if hasattr(cfg_i, "startup_joint_pos_noise_deg"):
            cfg_i.startup_joint_pos_noise_deg = {k: 0.0 for k in cfg_i.startup_joint_pos_noise_deg}
        if args_cli.rigid_soles:
            cfg_i.foot_contact_stiffness = None

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
        } if full_pose is None else full_pose
        target_rad = torch.tensor(
            [math.radians(target_deg[n.removeprefix("Revolute_")]) for n in names], device=device
        )
        print(f"[DIAG] held target (deg, robot order {names}): "
              f"{[round(target_deg[n.removeprefix('Revolute_')], 2) for n in names]}", flush=True)
        group = None
        group_labels = None
        if args_cli.ankle_offsets:
            offs = [float(v) for v in args_cli.ankle_offsets.split(",")]
            group = torch.arange(args_cli.num_envs, device=device) % len(offs)
            sign = torch.tensor([(1.0 if n.endswith("left_ankle") else -1.0 if n.endswith("right_ankle") else 0.0)
                                 for n in names], device=device)
            off = torch.tensor([math.radians(o) for o in offs], device=device)[group]
            target_rad = target_rad.unsqueeze(0) + off.unsqueeze(1) * sign.unsqueeze(0)
            group_labels = [f"ankle{o:+.1f}" for o in offs]
        elif args_cli.blend_from:
            bp, bk, ba = (float(v) for v in args_cli.blend_from.split(","))
            from_deg = {"left_yaw": 0.0, "right_yaw": 0.0, "left_roll": 0.0, "right_roll": 0.0,
                        "left_pitch": bp, "right_pitch": -bp, "left_knee": bk, "right_knee": -bk,
                        "left_ankle": ba, "right_ankle": -ba}
            from_rad = torch.tensor([math.radians(from_deg[n.removeprefix("Revolute_")]) for n in names], device=device)
            nb = args_cli.blend_steps
            group = torch.arange(args_cli.num_envs, device=device) % nb
            alpha = (group.float() / max(1, nb - 1)).unsqueeze(1)
            group_labels = [f"a={g / max(1, nb - 1):.2f}" for g in range(nb)]
            target_rad = from_rad.unsqueeze(0) + alpha * (target_rad.unsqueeze(0) - from_rad.unsqueeze(0))

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
                return unwrapped.robot.data.body_pos_w[:, feet, 2].min(dim=1).values

            lift = 0.05
            with torch.inference_mode():
                env.reset()
                z_ground = foot_z()
                unwrapped.robot.data.default_joint_pos[:, :num_joints] = target_rad.expand(args_cli.num_envs, -1)
                unwrapped.robot.data.default_root_state[:, 2] += lift
                env.reset()
                z_lifted = foot_z()
                unwrapped.robot.data.default_root_state[:, 2] -= (z_lifted - z_ground)
                env.reset()
            print(f"[DIAG] spawn-in-pose root z correction {float(1000 * (lift - (z_lifted - z_ground)).mean()):+.1f} mm", flush=True)
        default_pos0 = unwrapped.robot.data.default_joint_pos[:, :num_joints].clone()

        late_pitch, late_err = [], []
        pitch_sum = torch.zeros(args_cli.num_envs, device=device)
        pitch_n = torch.zeros(args_cli.num_envs, device=device)
        prev_pitch = torch.zeros(args_cli.num_envs, device=device)
        fall_pitch = torch.full((args_cli.num_envs,), float("nan"), device=device)
        fall_time = torch.full((args_cli.num_envs,), float("nan"), device=device)
        obs = env.get_observations()
        for step in range(num_steps):
            with torch.inference_mode():
                if step < RAMP_STEPS:
                    frac = (step + 1) / RAMP_STEPS
                    ramped_target = default_pos0 + frac * (target_rad.expand(args_cli.num_envs, -1) - default_pos0)
                else:
                    ramped_target = target_rad.expand(args_cli.num_envs, -1)
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
            first = fell & ~ever_fell
            fall_pitch[first] = prev_pitch[first]          # pitch just before the reset
            fall_time[first] = (step + 1) * unwrapped.step_dt
            ever_fell |= fell
            prev_pitch = torch.rad2deg(torch.asin(unwrapped.robot.data.projected_gravity_b[:, 1].clamp(-1, 1)))

            # Last-half signed pitch and actual - target per joint, over envs
            # that have not fallen -- directly comparable to the hardware
            # open-loop-ref CSVs (2026-10-04 ankle sweep).
            if step >= num_steps // 2:
                ok = ~ever_fell
                gp = unwrapped.robot.data.projected_gravity_b[:, 1].clamp(-1, 1)
                pitch_sum += torch.where(ok, torch.rad2deg(torch.asin(gp)), torch.zeros_like(gp))
                pitch_n += ok.float()
                if ok.any():
                    g = unwrapped.robot.data.projected_gravity_b[ok]
                    late_pitch.append(torch.rad2deg(torch.asin(g[:, 1].clamp(-1, 1))).cpu())
                    late_err.append(torch.rad2deg(
                        unwrapped.robot.data.joint_pos[ok, :num_joints] - target_rad.expand(args_cli.num_envs, -1)[ok]).cpu())

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
        if group is not None:
            print(f"[{args_cli.label}] sweep (blend a=0: --blend-from, a=1: the pose), friction {args_cli.friction}:")
            for g, glabel in enumerate(group_labels):
                m = group == g
                pose = target_rad[m][0]
                n_ok = float(pitch_n[m].sum())
                pm = float(pitch_sum[m].sum()) / n_ok if n_ok else float("nan")
                fp, ft = fall_pitch[m], fall_time[m]
                fok = ~torch.isnan(fp)
                fall_s = (f"first fall {float(ft[fok].mean()):4.1f}s pitch {float(fp[fok].mean()):+5.1f}" if fok.any()
                          else "no falls")
                print(f"  {glabel} fell={float(ever_fell[m].float().mean()):.2f} late pitch={pm:+6.2f} {fall_s}  pose "
                      + " ".join(f"{n.removeprefix('Revolute_')}={math.degrees(float(v)):+.1f}"
                                 for n, v in zip(names, pose) if n.startswith("Revolute_left")))
        print(
            "  (should stay small and flat if this pose is a real statically-stable stance; "
            "rising toward the episode's tilt-termination boundary means it's tipping over)"
        )

        env.close()

    run()
    simulation_app.close()


if __name__ == "__main__":
    main()

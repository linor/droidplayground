#!/usr/bin/env python3
"""
check_calibration_offset.py

Simulates a real-robot joint-zero calibration error and measures whether
it makes a trained policy lean/fall backward the way real hardware does.

Why: robot_deploy.py's calibrate_from_startup_position() takes each motor's
position AT POWER-ON as that joint's zero (offset_deg is 0 for every joint
in robot_config_qmini_stepinplace.json). Any error in the physical power-on
pose becomes a constant offset for the whole run: true_angle =
believed_angle + delta. With both feet flat on the floor, an ankle or
hip-pitch offset tilts the whole body.

Model (exact, given how obs/targets are built): the policy observes absolute
joint positions, and targets are default_joint_pos + action_scale * action.
A zero error delta means
    observed = true - delta          (policy sees the believed angle)
    true target = believed target + delta
so this shifts robot.data.default_joint_pos by +delta (moves targets AND the
reset pose, matching the real robot moving to its believed default pose)
and subtracts delta from the joint-position part of every observation.

Offsets are given per joint TYPE in physical terms and mirrored like the
poses are (right = -left for pitch/knee/ankle), so "+4 ankle" means both
ankles rotated the same way physically.

ONE CONDITION PER INVOCATION (multi gym.make per process hangs, see
check_reference_stability.py).
"""
from __future__ import annotations

import argparse
import math
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--pitch-deg", type=float, default=0.0, help="hip pitch zero error (left; right mirrored)")
    parser.add_argument("--knee-deg", type=float, default=0.0)
    parser.add_argument("--ankle-deg", type=float, default=0.0)
    parser.add_argument("--imu-pitch-bias-deg", type=float, default=0.0,
                        help="IMU mounting pitch error: observed gravity rotated so pitch reads "
                             "this much more backward (+) / forward (-) than the truth. The "
                             "2026-10-04 foot-world-pitch check suggests ~-2.5 on hardware.")
    parser.add_argument("--kp-scale", type=float, default=1.0,
                        help="multiply every actuator's stiffness (like robot_config_qmini_standing-2x.json); "
                             "per-reset gain randomization still applies on top")
    parser.add_argument("--friction", type=float, default=0.5,
                        help="fixed foot/ground friction (0.5 = sim value before 2026-10-06; real floor ~0.3)")
    parser.add_argument("--kd-scale", type=float, default=1.0, help="same for damping")
    parser.add_argument("--task", type=str, default="DroidPlayground-QMini-Leg")
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    args_cli, _ = parser.parse_known_args()

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import copy

    import gymnasium as gym
    import numpy as np
    import torch

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.hydra import hydra_task_config

    import droidplayground.tasks  # noqa: F401

    @hydra_task_config(args_cli.task, args_cli.agent)
    def run(env_cfg, agent_cfg):
        cfg_i = copy.deepcopy(env_cfg)
        cfg_i.scene.num_envs = args_cli.num_envs
        cfg_i.seed = args_cli.seed
        # The env's own per-episode random zero offsets (added 2026-10-05)
        # are switched off so this script's fixed offsets stay deterministic
        # and comparable with results from before that change.
        if hasattr(cfg_i, "joint_zero_offset_range_deg"):
            cfg_i.joint_zero_offset_range_deg = {k: 0.0 for k in cfg_i.joint_zero_offset_range_deg}
        # Same for friction randomization (added 2026-10-06): fixed value,
        # default 0.5 = what every eval before that date ran on.
        if hasattr(cfg_i, "friction_randomization_range"):
            cfg_i.friction_randomization_range = (args_cli.friction, args_cli.friction)
            cfg_i.dynamic_friction_ratio_range = (1.0, 1.0)
        print("[DIAG] about to gym.make()", flush=True)
        env = gym.make(args_cli.task, cfg=cfg_i, render_mode=None)
        print("[DIAG] gym.make() returned", flush=True)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        u = env.unwrapped
        dev = u.device
        nj = u.num_joints
        policy = torch.jit.load(args_cli.policy, map_location=str(dev)).eval()

        per_type = {"yaw": 0.0, "roll": 0.0, "pitch": args_cli.pitch_deg,
                    "knee": args_cli.knee_deg, "ankle": args_cli.ankle_deg}
        delta = torch.zeros(nj, device=dev)
        for i, name in enumerate(u.robot.joint_names[:nj]):
            plain = name.removeprefix("Revolute_")
            side, jtype = plain.split("_", 1)
            sign = 1.0 if side == "left" else -1.0
            delta[i] = math.radians(per_type[jtype] * sign)
        print(f"[DIAG] delta deg per joint {list(u.robot.joint_names[:nj])}: "
              f"{[round(math.degrees(float(d)), 2) for d in delta]}", flush=True)

        u.robot.data.default_joint_pos[:, :nj] += delta
        # Deploy-side gain change without retraining (2026-10-05 feet/gain tests):
        # _randomize_actuator_gains re-draws from these defaults at every reset.
        for gains in u._default_actuator_gains.values():
            gains["stiffness"] *= args_cli.kp_scale
            gains["damping"] *= args_cli.kd_scale
        cb = math.cos(math.radians(args_cli.imu_pitch_bias_deg))
        sb = math.sin(math.radians(args_cli.imu_pitch_bias_deg))

        def policy_obs(obs):
            o = obs if isinstance(obs, torch.Tensor) else obs["policy"]
            o = o.clone()
            o[:, :nj] -= delta
            # obs layout: joint_pos, joint_vel, gravity(3), ang_vel(3), phase(2)
            gy, gz = o[:, 2 * nj + 1].clone(), o[:, 2 * nj + 2].clone()
            o[:, 2 * nj + 1] = cb * gy - sb * gz
            o[:, 2 * nj + 2] = sb * gy + cb * gz
            return o

        env.reset()
        obs = env.get_observations()
        steps = max(1, round(args_cli.seconds / u.step_dt))
        pitches = []
        yaw_rate_abs = []  # |world yaw rate| per step, deg/s
        yaw_net = torch.zeros(args_cli.num_envs, device=dev)  # integrated heading change, deg
        track_err = []
        true_pos = []  # true joint angles (deg), for left/right symmetry checks vs hardware  # true joint pos - applied target (deg), comparable to robot_deploy CSV obs_pos - target_deg
        ever_fell = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=dev)
        for k in range(steps):
            with torch.inference_mode():
                obs, _r, dones, extras = env.step(policy(policy_obs(obs)))
            if k >= 25:
                true_pos.append(torch.rad2deg(u.robot.data.joint_pos[:, :nj]).cpu())
                track_err.append(torch.rad2deg(
                    u.robot.data.joint_pos[:, :nj] - u._last_position_targets[:, :nj]).cpu())
            wz = torch.rad2deg(u.robot.data.root_ang_vel_w[:, 2])
            yaw_rate_abs.append(wz.abs().mean().item())
            yaw_net += wz * u.step_dt
            g = u.robot.data.projected_gravity_b
            pitches.append(torch.rad2deg(torch.asin(g[:, 1].clamp(-1, 1))).cpu())
            d = dones.bool()
            to = extras.get("time_outs")
            ever_fell |= (d & ~to.bool()) if to is not None else d

        p = torch.stack(pitches).numpy()
        print(f"\n[{args_cli.label}] pitch(ankle={args_cli.ankle_deg}, hip_pitch={args_cli.pitch_deg}, knee={args_cli.knee_deg}, imu_bias={args_cli.imu_pitch_bias_deg}, kp x{args_cli.kp_scale}, kd x{args_cli.kd_scale}, mu {args_cli.friction})"
              f"  frac_envs_fell={float(ever_fell.float().mean()):.3f}"
              f"  mean_pitch={p.mean():+.2f}deg  p95={np.percentile(p, 95):+.2f}  p5={np.percentile(p, 5):+.2f}"
              f"  (+ = backward)", flush=True)
        print(f"[{args_cli.label}] yaw: mean |yaw rate| {np.mean(yaw_rate_abs):.1f} deg/s, net heading drift "
              f"{float(yaw_net.abs().mean()) / args_cli.seconds:.2f} deg/s (|mean over envs| {abs(float(yaw_net.mean())) / args_cli.seconds:.2f})", flush=True)
        tp = torch.cat(true_pos).numpy()
        print(f"[{args_cli.label}] mean true joint angle (deg): "
              + "  ".join(f"{n.removeprefix('Revolute_')}={tp[:, i].mean():+.2f}" for i, n in enumerate(u.robot.joint_names[:nj])), flush=True)
        te = torch.cat(track_err).numpy()
        names = [n.removeprefix("Revolute_") for n in u.robot.joint_names[:nj]]
        print(f"[{args_cli.label}] mean signed tracking error (actual - target, deg): "
              + "  ".join(f"{n}={te[:, i].mean():+.2f}" for i, n in enumerate(names)), flush=True)
        env.close()

    run()
    simulation_app.close()


if __name__ == "__main__":
    main()

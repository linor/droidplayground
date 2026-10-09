#!/usr/bin/env python3
"""
check_lateral_sway.py

Sim counterpart of the hardware lateral-sway analysis (2026-10-08): runs a
deploy-bundle policy and reports, per gait phase, the body roll, the STANCE
foot's world roll and the stance hip roll (actual / target) -- the same
numbers computed from real control_loop CSVs (IMU + encoder FK), so sim and
hardware can be compared directly.

Why: on hardware the final feetfix policy sways +-13..15 deg in body roll
(p5..p95) and the stance foot rolls 8-13 deg in the world along with it,
while the stance hip roll encoder moves only 1-3 deg. This script shows
how much of that the sim reproduces.

Gait phases from the step-in-place reference (2.2 s cycle): left foot
swings at motion_time 0.03-0.57 s (right stance), right foot 1.07-1.70 s
(left stance), double support otherwise.

ONE CONDITION PER INVOCATION (multi gym.make per process hangs).
"""
from __future__ import annotations

import argparse
import math
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--friction", type=float, default=0.3, help="fixed foot/ground friction (real floor ~0.3)")
    parser.add_argument("--usd", type=str, default=None, help="override the robot USD")
    parser.add_argument("--hip-roll-kp-scale", type=float, default=1.0,
                        help="scale hip_roll stiffness (and damping by its sqrt) -- tests whether a softer roll chain reproduces the real sway")
    parser.add_argument("--delay-steps", type=str, default=None,
                        help="override cfg.action_delay_range_steps for every joint type, 'lo,hi' in control steps")
    parser.add_argument("--hip-roll-backlash-deg", type=float, default=0.0,
                        help="dead band (+-deg) on the hip_roll position error: no spring torque until the joint is this far from its target (models free play after the encoder)")
    parser.add_argument("--dt-div", type=int, default=1,
                        help="divide the physics dt by N (decimation x N, same 50 Hz policy rate). The env default is 2.5 ms since "
                             "2026-10-09; at 5 ms the explicit hip-roll PD damps sway and chatters with *_flex USDs")
    parser.add_argument("--task", type=str, default="DroidPlayground-QMini-Leg")
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    args_cli, _ = parser.parse_known_args()

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args_cli).app

    import copy

    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab.utils.math import quat_apply_inverse
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.hydra import hydra_task_config

    import droidplayground.tasks  # noqa: F401

    @hydra_task_config(args_cli.task, args_cli.agent)
    def run(env_cfg, agent_cfg):
        cfg = copy.deepcopy(env_cfg)
        cfg.scene.num_envs = args_cli.num_envs
        cfg.seed = args_cli.seed
        # deterministic conditions: no random zero offsets, fixed friction
        cfg.joint_zero_offset_range_deg = {k: 0.0 for k in cfg.joint_zero_offset_range_deg}
        cfg.friction_randomization_range = (args_cli.friction, args_cli.friction)
        cfg.dynamic_friction_ratio_range = (1.0, 1.0)
        if args_cli.usd:
            cfg.robot_cfg.spawn.usd_path = args_cli.usd
        if args_cli.dt_div > 1:
            cfg.sim.dt = cfg.sim.dt / args_cli.dt_div
            cfg.decimation = cfg.decimation * args_cli.dt_div
            cfg.sim.render_interval = cfg.decimation
        if args_cli.delay_steps:
            lo, hi = (int(v) for v in args_cli.delay_steps.split(","))
            cfg.action_delay_range_steps = {k: (lo, hi) for k in cfg.action_delay_range_steps}
        env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=cfg, render_mode=None), clip_actions=agent_cfg.clip_actions)
        u = env.unwrapped
        r = u.robot
        policy = torch.jit.load(args_cli.policy, map_location=str(u.device)).eval()
        if args_cli.hip_roll_backlash_deg > 0:
            act = r.actuators["hip_roll"]
            band = math.radians(args_cli.hip_roll_backlash_deg)
            orig_compute = act.compute

            def compute_with_backlash(control_action, joint_pos, joint_vel):
                err = control_action.joint_positions - joint_pos
                err = torch.sign(err) * torch.clamp(err.abs() - band, min=0.0)
                control_action.joint_positions = joint_pos + err
                return orig_compute(control_action, joint_pos, joint_vel)

            act.compute = compute_with_backlash
        if args_cli.hip_roll_kp_scale != 1.0:
            g = u._default_actuator_gains["hip_roll"]
            g["stiffness"] *= args_cli.hip_roll_kp_scale
            g["damping"] *= math.sqrt(args_cli.hip_roll_kp_scale)
        names = [n.removeprefix("Revolute_") for n in r.joint_names[: u.num_joints]]
        flex_idx = [i for i, n in enumerate(r.joint_names) if "flex" in n]
        if flex_idx:
            print(f"[DIAG] joint order: {r.joint_names}  (passive flex joints at {flex_idx})", flush=True)
        iroll = {s: names.index(f"{s}_roll") for s in ("left", "right")}
        feet = {"left": u._left_foot_body_idx, "right": u._right_foot_body_idx}
        down = torch.tensor([0.0, 0.0, -1.0], device=u.device)

        env.reset()
        obs = env.get_observations()
        rec = {k: [] for k in ("roll", "mt", "lfoot", "rfoot", "lact", "ract", "ltgt", "rtgt", "done", "cmd", "act_all")}
        steps = round(args_cli.seconds / u.step_dt)
        for k in range(steps):
            with torch.inference_mode():
                action = policy(obs if isinstance(obs, torch.Tensor) else obs["policy"])
                # undelayed command, like the target robot_deploy.py logs at send time
                cmd = torch.rad2deg(r.data.default_joint_pos[:, : u.num_joints] + u.cfg.action_scale * action.clamp(-3, 3))
                obs, _r, dones, _e = env.step(action)
            if k < 50:
                continue
            g = r.data.projected_gravity_b
            rec["roll"].append(torch.rad2deg(torch.asin(g[:, 0].clamp(-1, 1))).cpu())
            rec["mt"].append((u.motion_time % u.motion.length).cpu())
            for s, key in (("left", "lfoot"), ("right", "rfoot")):
                q = r.data.body_link_quat_w[:, feet[s]]
                gf = quat_apply_inverse(q, down.expand(q.shape[0], 3))
                rec[key].append(torch.rad2deg(torch.asin(gf[:, 0].clamp(-1, 1))).cpu())
            for s, a, t in (("left", "lact", "ltgt"), ("right", "ract", "rtgt")):
                rec[a].append(torch.rad2deg(r.data.joint_pos[:, iroll[s]]).cpu())
                rec[t].append(torch.rad2deg(u._last_position_targets[:, iroll[s]]).cpu())
            rec["done"].append(dones.bool().cpu())
            if flex_idx:
                rec.setdefault("flex", []).append(torch.rad2deg(r.data.joint_pos[:, flex_idx]).cpu())
            rec["cmd"].append(cmd.cpu())
            rec["act_all"].append(torch.rad2deg(r.data.joint_pos[:, : u.num_joints]).cpu())
        D = {k: torch.stack(v).numpy() for k, v in rec.items()}
        ok = ~np.cumsum(D["done"], axis=0).astype(bool)  # drop everything after an env's first reset
        R = D["roll"][ok]
        mt = D["mt"][ok]
        print(f"\n[{args_cli.label}] mu={args_cli.friction} hip_roll_kp x{args_cli.hip_roll_kp_scale} delay {args_cli.delay_steps or 'cfg'} backlash {args_cli.hip_roll_backlash_deg}: body roll p5..p95 {np.percentile(R, 5):+5.1f}..{np.percentile(R, 95):+5.1f}, mean|roll| {np.abs(R).mean():4.1f}")
        lsw = (mt > 0.03) & (mt < 0.57)
        rsw = (mt > 1.07) & (mt < 1.70)
        for name, mask, stance in (("left swing (right stance)", lsw, "right"), ("right swing (left stance)", rsw, "left"), ("double support", ~(lsw | rsw), None)):
            line = f"   {name:26s} body roll {R[mask].mean():+5.1f} [{np.percentile(R[mask], 5):+5.1f},{np.percentile(R[mask], 95):+5.1f}]"
            if stance:
                c = stance[0]
                fr = D[c + "foot"][ok][mask]
                line += (f" | stance({stance}) foot world roll {fr.mean():+5.1f} [{np.percentile(fr, 5):+5.1f},{np.percentile(fr, 95):+5.1f}]"
                         f" | stance hip roll act {D[c + 'act'][ok][mask].mean():+5.1f} tgt {D[c + 'tgt'][ok][mask].mean():+5.1f}")
            print(line)
        if "flex" in D:
            for i, side in enumerate(("first", "second")):
                fx = D["flex"][..., i][ok]
                print(f"   passive flex joint {side}: mean {fx.mean():+5.2f} deg, p5..p95 {np.percentile(fx, 5):+5.2f}..{np.percentile(fx, 95):+5.2f}, double-support std {fx[~(lsw | rsw)].std():.2f}")
        for stance, mask in (("right", lsw), ("left", rsw)):
            fr = D[stance[0] + "foot"][ok][mask]
            sl = np.polyfit(R[mask], fr, 1)[0]
            print(f"   {stance} stance: d(foot world roll)/d(body roll) = {sl:.2f} (corr {np.corrcoef(R[mask], fr)[0, 1]:.2f})")
        # command -> actual lag per joint (same measure as the real-log analysis):
        # lag of the max cross-correlation, envs without a reset only
        # per env: the stretch before its first reset (episodes are 10 s), >= 4 s long
        first_done = [int(np.argmax(D["done"][:, e])) if D["done"][:, e].any() else D["done"].shape[0] for e in range(D["done"].shape[1])]
        segs = [(e, n) for e, n in enumerate(first_done) if n * u.step_dt >= 4.0]
        print(f"[{args_cli.label}] command->actual lag / amplitude ratio (envs used: {len(segs)}):")
        for j, name in enumerate(names):
            lags, amps = [], []
            for e, n in segs:
                t = D["cmd"][:n, e, j] - D["cmd"][:n, e, j].mean()
                a = D["act_all"][:n, e, j] - D["act_all"][:n, e, j].mean()
                cs = [np.dot(t[: len(t) - k], a[k:]) for k in range(15)]
                lags.append(int(np.argmax(cs)) * u.step_dt * 1000)
                amps.append(np.std(a) / max(np.std(t), 1e-6))
            print(f"   {name:12s} lag {np.mean(lags):5.0f} ms (p10..p90 {np.percentile(lags, 10):.0f}..{np.percentile(lags, 90):.0f}), amplitude {np.mean(amps):.2f}")
        env.close()

    run()
    app.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
check_battery_mass_pitch.py

Tests whether a heavier rear-mounted battery (Go1_________1, qmini.py's own
name for it, currently 1.2kg) makes backward tipping worse -- motivated by
2026-09-29 hardware data (attempts 5-7, loosened rope) showing a backward
pitch excursion tightly phase-locked to motion_time~1.0-1.2s (mean +4.0 to
+5.1deg, individual peaks 9-14deg), the same window as the already-known
hip-roll/IMU-fault/yaw-torque disturbance.

IMPORTANT SCOPE CAVEAT (read before interpreting results): the mt~1.0-1.2s
timing strongly suggests the TRIGGER is gait-dynamics/timing related (same
window as multiple other symptoms), not a static CoM offset -- a static
rear-heavy CoM would be expected to show up as a persistent mean pitch
bias throughout the cycle, which hardware does NOT show (mean pitch was
near-neutral, +0.09 to +1.64deg, only that one phase spikes). So this test
is NOT expected to explain WHY the disturbance happens at that phase. What
it CAN check: given the same triggering event, does added rear mass make
the CONSEQUENCE (how far it tips, whether it recovers, fall rate) worse?
That's a real, useful, narrower question even if the broader "why" stays
unanswered by this test.

METHOD: same harness as check_mass_asymmetry.py, but shifts ONE body's
mass (default Go1_________1) by --delta-g and reports fall_rate plus pitch
statistics specifically in the mt 1.0-1.2s window (matching the hardware
finding) alongside the full-cycle mean/max, for direct before/after
comparison against stock mass.

ONE CONDITION PER INVOCATION (see check_reference_stability.py's confirmed
multi-gym.make()-per-process hang).
"""
from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--body", type=str, default="Go1_________1")
    parser.add_argument("--delta-g", type=float, default=0.0, help="grams added (+) or removed (-) from --body")
    parser.add_argument("--task", type=str, default="DroidPlayground-QMini-Leg")
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=42)
    args_cli, extra = parser.parse_known_args()

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import copy
    import math as _m

    import gymnasium as gym
    import torch

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.hydra import hydra_task_config

    import droidplayground.tasks  # noqa: F401

    @hydra_task_config(args_cli.task, args_cli.agent)
    def run(env_cfg, agent_cfg):
        print("[DIAG] entered run", flush=True)
        cfg_i = copy.deepcopy(env_cfg)
        cfg_i.scene.num_envs = args_cli.num_envs
        cfg_i.seed = args_cli.seed

        print("[DIAG] about to gym.make()", flush=True)
        env = gym.make(args_cli.task, cfg=cfg_i, render_mode=None)
        print("[DIAG] gym.make() returned", flush=True)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        unwrapped = env.unwrapped
        device = unwrapped.device

        policy = torch.jit.load(args_cli.policy, map_location=str(device)).eval()

        if args_cli.delta_g != 0.0:
            names = list(unwrapped.robot.body_names)
            body_i = names.index(args_cli.body)
            masses = unwrapped.robot.root_physx_view.get_masses()
            m0 = float(masses[0, body_i])
            new_m = m0 + args_cli.delta_g / 1000.0
            print(f"[DIAG] {args_cli.body} {m0*1000:.1f}g -> {new_m*1000:.1f}g", flush=True)
            masses[:, body_i] = new_m
            all_env_ids = torch.arange(args_cli.num_envs, device=masses.device)
            unwrapped.robot.root_physx_view.set_masses(masses, all_env_ids)

        def _flat_policy_obs(obs):
            if isinstance(obs, torch.Tensor):
                return obs
            return obs["policy"]

        num_steps = max(1, round(args_cli.seconds / unwrapped.step_dt))
        obs = env.get_observations()

        all_mt = []
        all_pitch = []
        ever_fell = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
        for step in range(num_steps):
            with torch.inference_mode():
                actions = policy(_flat_policy_obs(obs))
                obs, _rew, dones, extras = env.step(actions)
            gxy = unwrapped.robot.data.projected_gravity_b[:, :2]
            # pitch = asin(gravity_y), same convention validated against
            # the known backward-tip reading (gravity_y=+0.20 at ~11.7deg
            # backward, hardware_test_2026-09-21 notes) -- positive = backward.
            pitch_deg = torch.rad2deg(torch.asin(torch.clamp(gxy[:, 1], min=-1.0, max=1.0)))
            all_pitch.append(pitch_deg.cpu())
            mt = (unwrapped.motion_time % 2.2).cpu()
            all_mt.append(mt)
            dones_bool = dones.bool()
            time_outs = extras.get("time_outs")
            fell = dones_bool & ~time_outs.bool() if time_outs is not None else dones_bool
            ever_fell |= fell

        mt = torch.cat(all_mt).numpy()
        pitch = torch.cat(all_pitch).numpy()
        window = (mt >= 1.0) & (mt < 1.2)

        print(f"\n[{args_cli.label}] over {args_cli.seconds:.0f}s, {args_cli.num_envs} envs, "
              f"fall_rate={float(ever_fell.float().mean()):.3f}")
        print(f"[{args_cli.label}] pitch (deg, +backward): full-cycle mean {pitch.mean():+.2f} "
              f"max {pitch.max():+.2f} | mt1.0-1.2 window mean {pitch[window].mean():+.2f} "
              f"p95 {__import__('numpy').percentile(pitch[window], 95):+.2f} max {pitch[window].max():+.2f}")

        env.close()

    run()
    simulation_app.close()


if __name__ == "__main__":
    main()

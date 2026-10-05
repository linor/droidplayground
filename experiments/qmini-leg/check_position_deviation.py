#!/usr/bin/env python3
"""
check_position_deviation.py

Direct, apples-to-apples measurement of the SAME quantity
position_deviation_penalty tracks during training (norm of xy displacement
from each env's own per-episode reset anchor, unwrapped._reset_pos_xy),
sampled every step over the whole eval window and reported as a
distribution -- not a single start/end snapshot (check_mass_asymmetry.py's
approach, which turned out to mostly reflect wherever in its current
episode each env happened to be at the one instant sampled, not a
representative measure). Built 2026-09-29 to get a real before/after number
comparing pitchstep5 (no position_deviation_penalty) against the
2026-09-28_21-10-54 run that added it.

ONE POLICY PER INVOCATION (see check_reference_stability.py's confirmed
multi-gym.make()-per-process hang).
"""
from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
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

        def _flat_policy_obs(obs):
            if isinstance(obs, torch.Tensor):
                return obs
            return obs["policy"]

        num_steps = max(1, round(args_cli.seconds / unwrapped.step_dt))
        obs = env.get_observations()

        all_dev_cm = []
        fall_count = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
        for step in range(num_steps):
            with torch.inference_mode():
                actions = policy(_flat_policy_obs(obs))
                obs, _rew, dones, extras = env.step(actions)
            # Same quantity qmini_leg_env.py's _get_rewards computes for
            # position_deviation_m -- read directly, not re-derived.
            dev_m = torch.norm(unwrapped.robot.data.root_pos_w[:, :2] - unwrapped._reset_pos_xy, dim=1)
            all_dev_cm.append((dev_m * 100.0).cpu())
            dones_bool = dones.bool()
            time_outs = extras.get("time_outs")
            fell = dones_bool & ~time_outs.bool() if time_outs is not None else dones_bool
            fall_count |= fell

        dev = torch.cat(all_dev_cm)
        print(f"\n[{args_cli.label}] over {args_cli.seconds:.0f}s, {args_cli.num_envs} envs, "
              f"fall_rate={float(fall_count.float().mean()):.3f}")
        print(f"[{args_cli.label}] position_deviation_cm (per-step, all envs): "
              f"mean {dev.mean():.2f}  median {dev.median():.2f}  p90 {dev.quantile(0.9):.2f}  "
              f"p99 {dev.quantile(0.99):.2f}  max {dev.max():.2f}")

        env.close()

    run()
    simulation_app.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
check_mass_asymmetry.py

Tests the user's 2026-09-29 hypothesis directly: real hardware drifts
consistently BACKWARD and RIGHTWARD (~50cm back, 1.5-2m right, ~90deg net
yaw over 4 attempts), and the left leg is currently ~100g lighter than the
right (missing 2 cover panels). Sim's own USD already has a smaller,
unmodeled asymmetry in the same direction -- Left_Thigh_1/Left_Calf_1 are
34g/12g lighter than their right counterparts (leg-only total: L=908g,
R=958g, 50g right-heavy) -- and this is NOT covered by
base_mass_randomization_range (base_link only, symmetric across legs).

METHOD
------ Runs a trained policy (TorchScript bundle, e.g.
deploy_bundle_*/policy.pt) through a normal eval, ONCE with sim's stock
masses and ONCE with an extra --extra-left-deficit-g removed from
Left_Thigh_1/Left_Calf_1 (split proportional to their existing masses,
approximating "2 missing cover panels" since the exact attachment point
isn't known) -- then reports net xy displacement (world frame) and net
yaw over the eval window for both, so the shift between them can be
compared directly against the reported real-world direction/magnitude.
ONE CONDITION PER INVOCATION (see check_reference_stability.py's
confirmed multi-gym.make()-per-process hang).

USAGE
-----
    ./isaaclab.sh -p check_mass_asymmetry.py --policy PATH/policy.pt \\
        --label stock_mass --seconds 40 --headless

    ./isaaclab.sh -p check_mass_asymmetry.py --policy PATH/policy.pt \\
        --label left_100g_lighter --extra-left-deficit-g 100 --seconds 40 --headless
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", type=str, required=True, help="TorchScript policy.pt (deploy bundle)")
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--extra-left-deficit-g", type=float, default=0.0,
                         help="grams removed from Left_Thigh_1+Left_Calf_1 combined (split proportionally), on top of sim's existing 50g right-heavy asymmetry")
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

        if args_cli.extra_left_deficit_g > 0:
            names = list(unwrapped.robot.body_names)
            thigh_i, calf_i = names.index("Left_Thigh_1"), names.index("Left_Calf_1")
            masses = unwrapped.robot.root_physx_view.get_masses()
            m_thigh0 = float(masses[0, thigh_i])
            m_calf0 = float(masses[0, calf_i])
            total0 = m_thigh0 + m_calf0
            deficit_kg = args_cli.extra_left_deficit_g / 1000.0
            frac_thigh = m_thigh0 / total0
            new_thigh = m_thigh0 - deficit_kg * frac_thigh
            new_calf = m_calf0 - deficit_kg * (1 - frac_thigh)
            print(f"[DIAG] Left_Thigh_1 {m_thigh0*1000:.1f}g -> {new_thigh*1000:.1f}g, "
                  f"Left_Calf_1 {m_calf0*1000:.1f}g -> {new_calf*1000:.1f}g", flush=True)
            masses[:, thigh_i] = new_thigh
            masses[:, calf_i] = new_calf
            all_env_ids = torch.arange(args_cli.num_envs, device=masses.device)
            unwrapped.robot.root_physx_view.set_masses(masses, all_env_ids)

        def _flat_policy_obs(obs):
            # Same extraction compare_policies_isaaclab.py uses -- a
            # TorchScript-exported policy wants a plain flat tensor, but
            # RslRlVecEnvWrapper hands back a TensorDict/dict keyed by obs
            # group ("policy" is the group qmini_leg_env.py returns).
            if isinstance(obs, torch.Tensor):
                return obs
            return obs["policy"]

        num_steps = max(1, round(args_cli.seconds / unwrapped.step_dt))
        obs = env.get_observations()

        start_xy = unwrapped.robot.data.root_pos_w[:, :2].clone()
        import math as _m
        from isaaclab.utils.math import euler_xyz_from_quat
        _, _, start_yaw = euler_xyz_from_quat(unwrapped.robot.data.root_quat_w)
        start_yaw = start_yaw.clone()

        fall_count = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
        for step in range(num_steps):
            with torch.inference_mode():
                actions = policy(_flat_policy_obs(obs))
                obs, _rew, dones, extras = env.step(actions)
            dones_bool = dones.bool()
            time_outs = extras.get("time_outs")
            fell = dones_bool & ~time_outs.bool() if time_outs is not None else dones_bool
            fall_count |= fell

        end_xy = unwrapped.robot.data.root_pos_w[:, :2]
        _, _, end_yaw = euler_xyz_from_quat(unwrapped.robot.data.root_quat_w)
        disp = end_xy - start_xy  # world-frame, but env starts facing a fixed default heading so this is comparable across envs
        # Rotate world-frame displacement into the START heading's own body frame
        # (forward = -Y_body, right = -X_body per the _imu_comment convention used
        # throughout this project) so "left/right/fore/aft" is meaningful per-env
        # regardless of the arbitrary world frame each env spawns in.
        cy, sy = torch.cos(-start_yaw), torch.sin(-start_yaw)
        dx_body = disp[:, 0] * cy - disp[:, 1] * sy
        dy_body = disp[:, 0] * sy + disp[:, 1] * cy
        lateral_cm = -dx_body * 100.0  # +right, -left (body +X = left, so -X = right)
        foreaft_cm = dy_body * 100.0  # +Y body = rear/backward
        yaw_drift_deg = torch.rad2deg(((end_yaw - start_yaw + _m.pi) % (2 * _m.pi)) - _m.pi)

        print(
            f"\n[{args_cli.label}] over {args_cli.seconds:.0f}s, {args_cli.num_envs} envs, "
            f"fall_rate={float(fall_count.float().mean()):.3f}"
        )
        print(
            f"[{args_cli.label}] net lateral (cm, +right/-left): mean {lateral_cm.mean():+.1f} "
            f"median {lateral_cm.median():+.1f} std {lateral_cm.std():.1f}"
        )
        print(
            f"[{args_cli.label}] net fore/aft (cm, +back/-fwd): mean {foreaft_cm.mean():+.1f} "
            f"median {foreaft_cm.median():+.1f} std {foreaft_cm.std():.1f}"
        )
        print(
            f"[{args_cli.label}] net yaw (deg, +?): mean {yaw_drift_deg.mean():+.1f} "
            f"median {yaw_drift_deg.median():+.1f} std {yaw_drift_deg.std():.1f}"
        )

        env.close()

    run()
    simulation_app.close()


if __name__ == "__main__":
    main()

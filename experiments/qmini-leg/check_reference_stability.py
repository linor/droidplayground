#!/usr/bin/env python3
"""
check_reference_stability.py

Answers a narrower, cheaper question than a full RL training run: does a
given reference motion (a keyframes_*.json file), tracked OPEN-LOOP by a
plain PD controller with no RL policy at all, keep the robot upright? If a
candidate reference can't even survive its own trajectory under simple PD
tracking, no amount of RL training will fix that -- the reference's implied
CoM/momentum trajectory is dynamically infeasible for this robot's mass and
actuator limits, not a policy-learning problem. This is NOT a claim that a
reference passing this check will train a good policy (the RL policy still
has all its own reward-shaping and exploration questions on top) -- only
that it rules out the reference itself as the failure cause before spending
a ~10 hour training run to find that out the hard way.

Built 2026-09-29 alongside the question of whether removing the
"stable base pose" pitch/knee/ankle offset from
keyframes_step_in_place_all_joints_2x_base_offset.json (see that file's
_comment -- the offset was derived from the STANDING policy's pose and
reused for stepping without independent verification) makes for a better
step-in-place reference. Works for any keyframes_*.json in this directory's
parent, not just that comparison.

METHOD
------
Every control step: sample the reference at the env's OWN motion_time
(same MotionPlayer qmini_leg_env.py uses, same 2x-slowed timing), convert
to the action that would make _apply_action's position_targets formula
(default_joint_pos + action_scale * action) come out EXACTLY equal to the
reference, and step the env with that -- i.e. reference tracking with a
PERFECT, instantaneous "policy" (no delay/smoothing IS still applied,
since action_delay_range_steps and action_smoothing are real env dynamics,
not something worth bypassing for this check). Reports fall_rate, tilt
(mean/p95/max, split roll/pitch), and PER-CYCLE tilt trend across the eval
so a genuine progressive instability (tilt growing cycle over cycle) can be
told apart from a bounded, repeating wobble.

Domain randomization stays at whatever the env cfg's current defaults are
(startup pose noise, IMU noise, gain randomization, etc -- NOT disabled),
same reasoning as compare_policies_isaaclab.py's fixed --seed: every
candidate reference sees the identical sequence of "bad luck", so
differences in fall_rate are attributable to the reference, not to which
policy got an easier draw. push_interval_range_s is a training-cfg default
(currently disabled, see that field's comment) and is left as whatever the
cfg says, not forced off.

USAGE
-----
    ./isaaclab.sh -p check_reference_stability.py \\
        --keyframes keyframes_step_in_place_all_joints_2x_base_offset.json \\
        --keyframes keyframes_step_in_place_all_joints_2x_no_offset.json \\
        --cycles 8 --headless

Pass multiple --keyframes to compare them in one Isaac Sim session (same
caveat as compare_policies_isaaclab.py about multi-env-construction not
being independently verified clean between runs in this repo -- rerun
one-at-a-time if results look suspicious).

CONFIRMED 2026-09-28: multi-keyframes-per-invocation HANGS, not just "not
independently verified" -- the first candidate's env.close() does not
release GPU/physx resources cleanly enough for a second gym.make() in the
same process (matches the "PhysXFoundation: Calling createGpuFoundation
without first releasing already acquired instances" warning seen in a
prior, unrelated eval log). First candidate completes and prints its
result; the second candidate's process then sits at "about to gym.make()"
indefinitely -- confirmed via added [DIAG] print statements and a 500s
foreground timeout, no crash, no further output, no error, genuinely stuck
(killed cleanly by `timeout`, no orphaned process or leftover GPU memory
after). ALWAYS run ONE --keyframes per invocation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--keyframes", type=str, action="append", required=True,
        help="keyframes_*.json filename (relative to the qmini_leg_env.py directory) or an absolute path. "
             "Repeat for multiple candidates in one session.",
    )
    parser.add_argument("--task", type=str, default="DroidPlayground-QMini-Leg")
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--cycles", type=float, default=8.0, help="how many 2.2s gait cycles to run each candidate for")
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
    from droidplayground.tasks.direct.droidplayground import qmini_leg_env as qle

    env_dir = Path(qle.__file__).parent
    results = []

    @hydra_task_config(args_cli.task, args_cli.agent)
    def run_all(env_cfg, agent_cfg):
        print("[DIAG] entered run_all (hydra config composition done)", flush=True)
        for kf in args_cli.keyframes:
            kf_path = Path(kf)
            if not kf_path.is_absolute():
                kf_path = env_dir / kf_path
            if not kf_path.exists():
                raise FileNotFoundError(f"--keyframes {kf} not found (looked for {kf_path})")

            print(f"\n=== {kf_path.name} ===", flush=True)

            # KEYFRAMES_PATH is a module-level global read inside
            # QminiLegEnv.__init__ (NOT a cfg field), so it has to be
            # monkeypatched before gym.make() constructs the env -- see
            # this script's docstring / qmini_leg_env.py's own
            # `keyframes, degrees = _load_reference_keyframes(KEYFRAMES_PATH, ...)`.
            qle.KEYFRAMES_PATH = kf_path

            cfg_i = copy.deepcopy(env_cfg)
            cfg_i.scene.num_envs = args_cli.num_envs
            print(f"[DIAG] about to gym.make() num_envs={args_cli.num_envs}", flush=True)
            cfg_i.seed = args_cli.seed

            env = gym.make(args_cli.task, cfg=cfg_i, render_mode=None)
            print("[DIAG] gym.make() returned", flush=True)
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            unwrapped = env.unwrapped
            device = unwrapped.device
            num_joints = unwrapped.num_joints
            action_scale = cfg_i.action_scale

            num_steps = max(1, round((args_cli.cycles * 2.2) / unwrapped.step_dt))
            steps_per_cycle = max(1, round(2.2 / unwrapped.step_dt))

            # Tracks, per env, whether it EVER fell during the eval window --
            # simpler and more directly answers "does this reference cause
            # falling" than a per-completed-episode rate (compare_policies_
            # isaaclab.py's convention) would, since a reference that's
            # dynamically unstable could keep resetting/falling repeatedly
            # within the eval and a per-episode rate wouldn't distinguish
            # that from "fell once, then was fine".
            ever_fell = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
            steps_since_reset = torch.zeros(args_cli.num_envs, device=device)
            tilt_sum = tilt_count = 0
            roll_sum = pitch_sum = 0.0
            tilt_max = 0.0
            per_cycle_tilt_max = [0.0] * (int(args_cli.cycles) + 1)

            obs = env.get_observations()
            for step in range(num_steps):
                with torch.inference_mode():
                    reference = unwrapped.motion.sample(unwrapped.motion_time)
                    default_pos = unwrapped.robot.data.default_joint_pos[:, :num_joints]
                    actions = (reference[:, :num_joints] - default_pos) / action_scale
                    obs, _rew, dones, extras = env.step(actions)

                steps_since_reset += 1

                gxy = unwrapped.robot.data.projected_gravity_b[:, :2]
                tilt_deg = torch.rad2deg(torch.asin(torch.clamp(torch.norm(gxy, dim=1), max=1.0)))
                roll_deg = torch.rad2deg(torch.asin(torch.clamp(gxy[:, 0].abs(), max=1.0)))
                pitch_deg = torch.rad2deg(torch.asin(torch.clamp(gxy[:, 1].abs(), max=1.0)))
                tilt_sum += float(tilt_deg.sum())
                roll_sum += float(roll_deg.sum())
                pitch_sum += float(pitch_deg.sum())
                tilt_count += args_cli.num_envs
                tilt_max = max(tilt_max, float(tilt_deg.max()))
                cyc = step // steps_per_cycle
                if cyc < len(per_cycle_tilt_max):
                    per_cycle_tilt_max[cyc] = max(per_cycle_tilt_max[cyc], float(tilt_deg.max()))

                dones_bool = dones.bool()
                time_outs = extras.get("time_outs")
                if time_outs is not None:
                    fell = dones_bool & ~time_outs.bool()
                else:
                    fell = dones_bool
                ever_fell |= fell
                steps_since_reset[dones_bool] = 0

            row = {
                "keyframes": kf_path.name,
                "cycles": args_cli.cycles,
                "frac_envs_ever_fell": float(ever_fell.float().mean()),
                "mean_tilt_deg": tilt_sum / max(1, tilt_count),
                "mean_roll_deg": roll_sum / max(1, tilt_count),
                "mean_pitch_deg": pitch_sum / max(1, tilt_count),
                "max_tilt_deg": tilt_max,
                "per_cycle_max_tilt_deg": per_cycle_tilt_max[: int(args_cli.cycles) + 1],
            }
            results.append(row)
            print(
                f"  frac_envs_ever_fell={row['frac_envs_ever_fell']:.4f} (of {args_cli.num_envs} envs, over {args_cli.cycles} cycles)  "
                f"mean_tilt={row['mean_tilt_deg']:.2f}deg  (roll {row['mean_roll_deg']:.2f}, pitch {row['mean_pitch_deg']:.2f})  "
                f"max_tilt={row['max_tilt_deg']:.2f}deg"
            )
            print(f"  per-cycle max tilt (deg): {[round(x, 1) for x in row['per_cycle_max_tilt_deg']]}")
            print(
                "  (a rising trend here means progressive/cumulative instability -- the reference "
                "itself can't be tracked open-loop for this many cycles without an RL policy actively "
                "correcting it; a flat/bounded trend means the reference alone is dynamically viable)"
            )

            env.close()

    run_all()

    print("\n=== SUMMARY ===")
    print(f"{'keyframes':<55s} {'frac_fell':>9s} {'mean_tilt':>10s} {'max_tilt':>9s}")
    for r in results:
        print(f"{r['keyframes']:<55s} {r['frac_envs_ever_fell']:9.4f} {r['mean_tilt_deg']:10.2f} {r['max_tilt_deg']:9.2f}")

    simulation_app.close()


if __name__ == "__main__":
    main()

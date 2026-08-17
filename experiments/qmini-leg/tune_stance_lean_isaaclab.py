#!/usr/bin/env python3
"""
tune_stance_lean_isaaclab.py

The qmini-leg robot's default standing pose (qmini.py's
ArticulationCfg.InitialStateCfg.joint_pos, == keyframe-0 of the reference
gait) was designed assuming the body's mass was roughly centered over the
feet. It isn't anymore: the real battery ("Go1_________1" in
qmini_urdf-2legs.usda, ~1.2kg) plus the Raspberry Pi mount / exterior
decoration / handle add ~2.36kg mounted toward the REAR of base_link,
shifting the combined center of mass ~13cm behind base_link's own origin.
With the base now free-floating (see qmini_urdf-2legs.usda's removed
root_joint), that's enough to topple the robot backward in well under one
control step, from pure gravity alone -- independent of any RL policy (see
the qmini-leg training thread this script came out of: episode length was
stuck at 1 from iteration 0 of a 30k-iteration run).

This script does NOT run any RL policy. It holds the robot at a STATIC
candidate pose via ordinary PD position control (the same gains QMINI_CFG
trains with) and reports whether it stays upright over a short hold, so you
can verify/refine a stance found by hand in the Isaac Sim viewer without
guessing at Isaac's exact PD/contact behavior over time.

LEFT/RIGHT SIGN CONVENTION -- confirmed inverted, not matched
-----------------------------------------------------------------
An earlier version of this script assumed left and right legs use the SAME
sign for a given physical motion (based on how the reference keyframes JSON
describes its own sign convention). That was wrong for THIS USD: manually
testing in Isaac Sim showed the right leg moves in the OPPOSITE direction
from the left leg for the same joint value -- i.e. right = -left, for
pitch, knee, AND ankle. This script now mirrors (negates) every joint value
between legs accordingly. If you find this ISN'T exactly a clean negation
for some future USD revision, the asymmetry will show up directly as a
nonzero final_gravity_x in the results below (a pure fore/aft lean should
leave gravity_x near 0 -- see the summary printout).

STANDING POSE, NOT WALKING
-----------------------------
This only searches for a STATIC balanced stance -- it does not touch the
walking gait/reference tracking at all. Worth doing regardless of whether
you end up training "stand still" or "walk forward" first: a policy can't
learn to walk from a start pose that's already toppling over before it gets
a single useful action in.

*** NOT EXECUTED OR TESTED IN THIS SESSION *** (no Isaac Sim/GPU available
here) -- written against the same isaaclab APIs qmini_leg_env.py and
tune_pid_isaaclab.py already use in this repo (Articulation,
write_root_pose_to_sim / write_root_velocity_to_sim / write_joint_state_to_sim,
robot.data.projected_gravity_b). Expect to debug small API differences for
your installed Isaac Lab version.

USAGE
-----
Just verify the pose you already found by hand (no sweep, single candidate):

    ./isaaclab.sh -p tune_stance_lean_isaaclab.py \\
        --left-pitch-deg -20 --left-knee-deg 20 --left-ankle-deg 1 \\
        --vary-joint none --headless

Sweep a range of hip_pitch around that same center (holds knee/ankle fixed
at their --left-*-deg values, right leg always mirrored):

    ./isaaclab.sh -p tune_stance_lean_isaaclab.py \\
        --left-pitch-deg -20 --left-knee-deg 20 --left-ankle-deg 1 \\
        --vary-joint pitch --delta-range -10 10 --delta-step 2 \\
        --headless

Then rerun with --vary-joint knee or --vary-joint ankle to explore those
independently around the same center.

WHAT TO DO WITH THE RESULT
----------------------------
Once you have a pose that survives the hold with a small final tilt: add it
to BOTH
  1. qmini.py's ArticulationCfg.InitialStateCfg.joint_pos (all six of
     Revolute_{left,right}_{pitch,knee,ankle}), and
  2. keyframes_forward_slow_all_joints_4x.json -- IF you're keeping the
     walking gait goal, the same pose needs to become the new frame-0 (and
     really bias the whole gait, not just frame 0, the way a person
     carrying a heavy backpack leans forward through their entire stride,
     not just at standstill). If you're trying "stand still and balance"
     first instead (a very reasonable simplification -- ask me about
     restructuring qmini_leg_env.py's reward/reference for that), the
     keyframes file may not even be needed for a first pass.
default_joint_pos and keyframe-0 must still match exactly if you keep the
gait (see qmini_leg_env.py's comment on why) -- ask me to apply either edit
once you have a number, rather than hand-editing every keyframe yourself.
"""

from __future__ import annotations

import argparse
import csv as csv_module
import math
from pathlib import Path


def run_sweep(args):
    # Isaac Sim app must be launched before importing isaaclab.* anything.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    import torch
    from isaaclab.assets import Articulation
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

    from droidplayground.assets.qmini import QMINI_CFG

    physics_dt = 1.0 / 200.0  # matches QminiLegEnvCfg.sim.dt
    decimation = 4  # matches QminiLegEnvCfg.decimation -> control dt = 0.02
    control_dt = physics_dt * decimation

    sim = SimulationContext(SimulationCfg(dt=physics_dt))
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

    cfg = QMINI_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(cfg)
    sim.reset()

    print("\nRobot joints in this USD:")
    for i, name in enumerate(robot.joint_names):
        print(f"  {i:2d}: {name}")

    def joint_index(name: str) -> int:
        return robot.joint_names.index(name)

    # left = as-given; right = mirrored (negated) -- confirmed by hand in
    # Isaac Sim, see the module docstring's "LEFT/RIGHT SIGN CONVENTION" note.
    joint_idx = {
        "pitch": (joint_index("Revolute_left_pitch"), joint_index("Revolute_right_pitch")),
        "knee": (joint_index("Revolute_left_knee"), joint_index("Revolute_right_knee")),
        "ankle": (joint_index("Revolute_left_ankle"), joint_index("Revolute_right_ankle")),
    }
    left_center_deg = {"pitch": args.left_pitch_deg, "knee": args.left_knee_deg, "ankle": args.left_ankle_deg}

    sim_device = getattr(robot, "device", None) or sim.device
    print(f"Using device: {sim_device}")

    default_joint_pos = robot.data.default_joint_pos[0].clone()  # (num_joints,) -- yaw/roll/etc. stay at this
    default_root_pos = robot.data.default_root_state[0, :3].clone()
    if args.spawn_z is not None:
        default_root_pos[2] = args.spawn_z
    default_root_quat = robot.data.default_root_state[0, 3:7].clone()  # (w, x, y, z), should be identity/level

    def build_target(left_deg: dict) -> "torch.Tensor":
        target = default_joint_pos.clone()
        for joint, left_val_deg in left_deg.items():
            left_i, right_i = joint_idx[joint]
            target[left_i] = math.radians(left_val_deg)
            target[right_i] = math.radians(-left_val_deg)
        return target

    if args.vary_joint == "none":
        deltas_deg = [0.0]
    else:
        lo, hi = args.delta_range
        n = int(round((hi - lo) / args.delta_step)) + 1
        deltas_deg = [lo + i * args.delta_step for i in range(n)]

    results = []
    for delta_deg in deltas_deg:
        left_deg = dict(left_center_deg)
        if args.vary_joint != "none":
            left_deg[args.vary_joint] = left_center_deg[args.vary_joint] + delta_deg
        target = build_target(left_deg)

        # Hard reset to a clean state before every candidate -- root pose,
        # root velocity, AND joint state, so a violent fall from the
        # previous candidate can't bleed into this one.
        root_pose = torch.cat([default_root_pos, default_root_quat]).unsqueeze(0).to(sim_device)
        root_vel = torch.zeros(1, 6, device=sim_device)
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(root_vel)
        robot.write_joint_state_to_sim(
            target.unsqueeze(0).to(sim_device),
            torch.zeros_like(target).unsqueeze(0).to(sim_device),
        )
        robot.reset()

        fell_at_step = None
        worst_tilt_sq = 0.0
        for step in range(args.settle_steps):
            robot.set_joint_position_target(target.unsqueeze(0).to(sim_device))
            for _ in range(decimation):
                robot.write_data_to_sim()
                sim.step(render=not args.headless)
                robot.update(physics_dt)

            g = robot.data.projected_gravity_b[0]
            tilt_sq = float(g[0] ** 2 + g[1] ** 2)
            worst_tilt_sq = max(worst_tilt_sq, tilt_sq)
            if fell_at_step is None and float(g[2]) > args.fall_threshold:
                fell_at_step = step

        g_final = robot.data.projected_gravity_b[0].tolist()
        results.append({
            "vary_joint": args.vary_joint,
            "delta_deg": delta_deg,
            "left_pitch_deg": left_deg["pitch"], "left_knee_deg": left_deg["knee"], "left_ankle_deg": left_deg["ankle"],
            "fell_at_step": fell_at_step,
            "fell_at_s": None if fell_at_step is None else fell_at_step * control_dt,
            "worst_tilt_deg": math.degrees(math.asin(min(1.0, math.sqrt(worst_tilt_sq)))),
            "final_gravity_x": g_final[0],
            "final_gravity_y": g_final[1],
            "final_gravity_z": g_final[2],
        })
        status = "FELL" if fell_at_step is not None else "held"
        print(
            f"  {args.vary_joint}={left_deg[args.vary_joint] if args.vary_joint != 'none' else 0:+6.1f} deg "
            f"(delta={delta_deg:+5.1f})  [{status:4s}]  "
            f"worst_tilt={results[-1]['worst_tilt_deg']:5.1f} deg  "
            f"final_gravity=({g_final[0]:+.3f}, {g_final[1]:+.3f}, {g_final[2]:+.3f})"
        )

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv_module.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote {len(results)} candidates to {args.out}")

    held = [r for r in results if r["fell_at_step"] is None]
    print("\n=== Summary ===")
    if not held:
        print(
            "No candidate held for the full settle duration -- including, "
            "if you passed --vary-joint none, your hand-found pose itself. "
            "That could mean it's less stable under a sustained PD hold "
            "than it looked in the viewer (e.g. contact/friction settling "
            "differently over 2s vs a quick visual check), or that "
            "--fall-threshold / --spawn-z need adjusting. Try widening "
            "--delta-range, or increasing --settle-steps to see if it's "
            "failing early vs. slowly drifting."
        )
    else:
        best = min(held, key=lambda r: r["worst_tilt_deg"])
        print(
            f"Best held candidate: {args.vary_joint}="
            f"{best['left_pitch_deg'] if args.vary_joint == 'pitch' else best['left_knee_deg'] if args.vary_joint == 'knee' else best['left_ankle_deg']:+.1f} deg "
            f"(left_pitch={best['left_pitch_deg']:+.1f}, left_knee={best['left_knee_deg']:+.1f}, "
            f"left_ankle={best['left_ankle_deg']:+.1f}), "
            f"worst_tilt={best['worst_tilt_deg']:.1f} deg during the hold, "
            f"final_gravity=({best['final_gravity_x']:+.3f}, "
            f"{best['final_gravity_y']:+.3f}, {best['final_gravity_z']:+.3f})."
        )
        print(
            "If |final_gravity_x| is not small, the mirrored left/right "
            "values aren't producing a pure fore/aft lean -- double check "
            "the negation assumption still holds (see the module "
            "docstring's sign-convention note)."
        )

    _teardown(sim, simulation_app, force_exit=args.force_exit, timeout=args.teardown_timeout)


def _teardown(sim, simulation_app, force_exit: bool, timeout: float):
    """Same bounded-wait shutdown as tune_pid_isaaclab.py -- results are
    already printed/written to disk before this runs, so a slow/hanging
    Isaac Sim teardown can never lose them."""
    import os

    if force_exit:
        print("--force-exit set: skipping graceful shutdown, os._exit(0) now.")
        os._exit(0)

    import threading

    done = threading.Event()

    def _graceful_shutdown():
        try:
            sim.stop()
        except Exception as e:  # noqa: BLE001 - best-effort, shutting down regardless
            print(f"(sim.stop() raised {e!r}, continuing)")
        try:
            simulation_app.update()
            simulation_app.close()
        except Exception as e:  # noqa: BLE001
            print(f"(simulation_app.close() raised {e!r}, continuing)")
        done.set()

    t = threading.Thread(target=_graceful_shutdown, daemon=True)
    t.start()
    if done.wait(timeout):
        print("Isaac Sim shut down cleanly.")
    else:
        print(
            f"Graceful shutdown did not finish within {timeout:.0f}s "
            f"(known Isaac Sim issue, not a problem with your results above "
            f"-- they're already on disk) -- forcing process exit."
        )
        os._exit(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left-pitch-deg", type=float, default=-20.0, help="left hip_pitch center value, degrees (right = mirrored)")
    parser.add_argument("--left-knee-deg", type=float, default=20.0, help="left knee center value, degrees (right = mirrored)")
    parser.add_argument("--left-ankle-deg", type=float, default=1.0, help="left ankle center value, degrees (right = mirrored)")
    parser.add_argument("--vary-joint", choices=["pitch", "knee", "ankle", "none"], default="none",
                         help="which joint to sweep a delta around its center value; 'none' just tests the center pose once")
    parser.add_argument("--delta-range", type=float, nargs=2, default=[-10.0, 10.0], metavar=("LOW", "HIGH"),
                         help="sweep range added to --vary-joint's center value, degrees")
    parser.add_argument("--delta-step", type=float, default=2.0, help="degrees between sweep candidates")
    parser.add_argument("--settle-steps", type=int, default=100, help="control steps to hold each candidate (100 = 2s)")
    parser.add_argument("--fall-threshold", type=float, default=-0.5,
                         help="projected_gravity_b.z above this = fallen, matches qmini_leg_env.py's fall_orientation_threshold")
    parser.add_argument("--spawn-z", type=float, default=None,
                         help="override spawn height (meters); default uses QMINI_CFG's current init_state.pos")
    parser.add_argument("--out", type=Path, default=Path("stance_lean_sweep.csv"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--force-exit", action="store_true",
        help="Skip graceful sim.stop()/close() entirely and os._exit(0) as "
             "soon as results are printed/written. See tune_pid_isaaclab.py's "
             "identical flag for why this exists.",
    )
    parser.add_argument("--teardown-timeout", type=float, default=5.0)
    args = parser.parse_args()

    run_sweep(args)


if __name__ == "__main__":
    main()

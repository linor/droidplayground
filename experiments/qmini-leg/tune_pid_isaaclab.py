#!/usr/bin/env python3
"""
tune_pid_isaaclab.py

Sim-side counterpart to RoboTamer4Qmini's tune_pid.py, ported to Isaac Lab /
your QMINI setup. Drives a single QMINI leg through the SAME reference
trajectory used in training (qmini_leg_env.py's MotionPlayer) and on the
real robot (robot_deploy.py --open-loop-ref), open-loop -- no RL policy
involved -- and logs joint_pos/joint_vel/applied_torque at the same control
rate the real deployment uses. Optionally overlays that trace against a real
control_loop_*.csv (captured with `robot_deploy.py --open-loop-ref
--keyframes keyframes.json`) so you can see exactly where sim and real
diverge and adjust stiffness/damping/armature accordingly.

This is a *manual* tuning loop, same as the original tune_pid.py -- run it,
look at the overlay plot + printed RMSE, adjust --stiffness/--damping/
--armature, run again. See the bottom of this file for a note on
automating the search with scipy.optimize if you want that instead.

*** NOT EXECUTED OR TESTED IN THIS SESSION ***
I don't have Isaac Sim / a GPU available in this sandbox, so unlike
robot_deploy.py (verified logically but also not run against real hardware
here), this script has not been run at all. It's written against the
isaaclab APIs used in your own qmini_leg_env.py and the
isaac_sim_unitree_backend.py I gave you earlier (SimulationContext,
Articulation, set_joint_position_target, robot.data.applied_torque). Expect
to debug small API differences for your installed Isaac Lab version --
in particular:
  - `robot.data.applied_torque` may not exist on all versions; there's a
    fallback to 0.0 like isaac_sim_unitree_backend.py used, guard against it
    printing all-zero torque and silently invalidating any torque-based
    comparison.
  - Whether QMINI's base is fixed in the USD itself. This script does not
    add a ground plane or fix the base explicitly (matching item 3 in
    isaac_sim_unitree_backend.py's docstring) -- if the leg falls under
    gravity instead of just swinging from a fixed hip, either your USD
    already pins the base, or you need to add that here.

USAGE
-----
    ./isaaclab.sh -p tune_pid_isaaclab.py \\
        --keyframes keyframes.json \\
        --duration 6.0 \\
        --stiffness hip_pitch=55 knee=45 ankle=30 \\
        --damping   hip_pitch=0.3 knee=0.5 ankle=0.25 \\
        --out sim_trace.csv \\
        --real-log logs/control_loop_20260810_120000.csv \\
        --headless

If --real-log is omitted, this just produces sim_trace.csv (and, with
--plot, plots of ref-vs-actual in sim alone) so you can sanity check the sim
side first.
"""

from __future__ import annotations

import argparse
import csv as csv_module
import json
import math
import os
from pathlib import Path


def parse_gain_kv(pairs: list[str] | None, field: str) -> dict:
    """Parse ["hip_pitch=55", "knee=45"] into {"hip_pitch": {"stiffness": 55.0}, ...}."""
    out = {}
    for pair in pairs or []:
        name, value = pair.split("=")
        out.setdefault(name, {})[field] = float(value)
    return out


def merge_gain_dicts(*dicts: dict) -> dict:
    merged: dict = {}
    for d in dicts:
        for name, fields in d.items():
            merged.setdefault(name, {}).update(fields)
    return merged


class LinearMotionReference:
    """Deliberately identical logic to robot_deploy.py's MotionReference, so
    the two scripts can't silently diverge on interpolation method. If your
    real MotionPlayer (used in training) is NOT piecewise-linear, this will
    not match it -- import the real class instead if you can."""

    def __init__(self, keyframes_deg, degrees: bool = True):
        self.times = [t for t, _ in keyframes_deg]
        vals = [v for _, v in keyframes_deg]
        self.values_rad = [[math.radians(x) if degrees else x for x in v] for v in vals]
        self.length = self.times[-1]

    @classmethod
    def from_json(cls, path: Path) -> "LinearMotionReference":
        raw = json.loads(Path(path).read_text())
        return cls(raw["keyframes"], degrees=raw.get("degrees", True))

    def sample(self, t: float):
        t = t % self.length
        for i in range(len(self.times) - 1):
            t0, t1 = self.times[i], self.times[i + 1]
            if t0 <= t <= t1:
                alpha = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                v0, v1 = self.values_rad[i], self.values_rad[i + 1]
                return [a + alpha * (b - a) for a, b in zip(v0, v1)]
        return self.values_rad[-1]


def run_sim_trace(args, gains: dict) -> Path:
    # Isaac Sim app must be launched before importing isaaclab.* anything.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    import torch
    from isaaclab.assets import Articulation
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

    from droidplayground.assets.qmini import build_qmini_cfg

    physics_dt = 1.0 / 200.0  # matches QminiLegEnvCfg.sim.dt
    decimation = 4  # matches QminiLegEnvCfg.decimation -> control dt = 0.02
    control_dt = physics_dt * decimation
    if abs(control_dt - args.control_dt) > 1e-6:
        print(
            f"WARNING: this script's control_dt ({control_dt*1000:.1f} ms, from "
            f"physics_dt*decimation) does not match --control-dt "
            f"({args.control_dt*1000:.1f} ms). Fix physics_dt/decimation above "
            f"or --control-dt so sim and real logs are on the same rate."
        )

    sim = SimulationContext(SimulationCfg(dt=physics_dt))
    if args.ground:
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

    cfg = build_qmini_cfg(gains).replace(prim_path="/World/Robot")
    robot = Articulation(cfg)
    sim.reset()

    print(f"\nRobot joints in this USD:")
    for i, name in enumerate(robot.joint_names):
        print(f"  {i:2d}: {name}")
    joint_ids = list(range(min(3, robot.num_joints)))
    print(f"Driving joint_ids={joint_ids} as [hip, knee, ankle] -- verify this ordering "
          f"matches keyframes.json's joint_order against the printout above.\n")

    # Articulation should expose .device, but fall back to the
    # SimulationContext's device if a future/older isaaclab version doesn't
    # -- every tensor we build below must live on this device or ops like
    # set_joint_position_target's internal indexing throw a CPU/CUDA
    # mismatch (as happened here originally).
    sim_device = getattr(robot, "device", None) or sim.device
    print(f"Using device: {sim_device}")

    motion = LinearMotionReference.from_json(args.keyframes)

    rows = []
    t = 0.0
    n_steps = int(args.duration / control_dt)
    for step in range(n_steps):
        ref = motion.sample(t)
        target = torch.tensor([ref], dtype=torch.float32, device=sim_device)
        robot.set_joint_position_target(target, joint_ids=joint_ids)

        for _ in range(decimation):
            robot.write_data_to_sim()
            # render=not args.headless matters: without it the physics
            # loop never pumps Kit's UI event loop, so the window shows as
            # "Not Responding" for the whole run (not just hanging at the
            # end) when --headless isn't passed.
            sim.step(render=not args.headless)
            robot.update(physics_dt)

        pos = robot.data.joint_pos[0, joint_ids].tolist()
        vel = robot.data.joint_vel[0, joint_ids].tolist()
        if hasattr(robot.data, "applied_torque"):
            tau = robot.data.applied_torque[0, joint_ids].tolist()
        else:
            tau = [0.0, 0.0, 0.0]

        rows.append([t] + [math.degrees(x) for x in ref] + [math.degrees(x) for x in pos]
                    + [math.degrees(x) for x in vel] + tau)
        t += control_dt

    out_path = Path(args.out)
    with open(out_path, "w", newline="") as f:
        writer = csv_module.writer(f)
        writer.writerow(
            ["t"]
            + [f"target_deg_{n}" for n in ("hip", "knee", "ankle")]
            + [f"actual_deg_{n}" for n in ("hip", "knee", "ankle")]
            + [f"vel_deg_{n}" for n in ("hip", "knee", "ankle")]
            + [f"tau_{n}" for n in ("hip", "knee", "ankle")]
        )
        writer.writerows(rows)
    print(f"Wrote {len(rows)} sim steps to {out_path}")

    # Do the comparison/plotting BEFORE attempting to tear down Isaac Sim.
    # compare_traces only needs numpy/matplotlib -- nothing isaaclab-
    # specific -- so there's no reason its output should be at the mercy of
    # whether simulation_app.close() hangs. This is what was actually
    # costing you the comparison before: --force-exit killed the process
    # (correctly) but *after* main() would have called compare_traces(),
    # since that call happened outside/after run_sim_trace() returned.
    if args.real_log:
        compare_traces(out_path, args.real_log, args.joints, args.plot)
    else:
        print("\nNo --real-log given -- run robot_deploy.py with "
              "--open-loop-ref --keyframes keyframes.json on the real robot, "
              "then pass that CSV here with --real-log to compare.")

    _teardown(sim, simulation_app, force_exit=args.force_exit, timeout=args.teardown_timeout)
    return out_path


def _teardown(sim, simulation_app, force_exit: bool, timeout: float):
    """
    Best-effort Isaac Sim shutdown that cannot hang the process, because by
    the time this runs, everything you actually care about (CSV, RMSE,
    plots) has already been written. If --force-exit is set, skip straight
    to os._exit(0). Otherwise try a graceful sim.stop() + close() on a
    background thread and give it `timeout` seconds; if that thread hasn't
    finished by then (the actual failure mode you hit -- sim.stop() itself
    hanging, not just close()), force-exit anyway rather than wait forever.
    """
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


def compare_traces(sim_csv: Path, real_csv: Path, joint_names, plot: bool):
    import numpy as np

    def load(path):
        with open(path, newline="") as f:
            reader = csv_module.reader(f)
            header = next(reader)
            rows = [row for row in reader]
        return {name: np.array([float(row[i]) for row in rows]) for i, name in enumerate(header)}

    sim = load(sim_csv)
    real = load(real_csv)

    # real_csv comes from robot_deploy.py, whose time column is t_wall
    # (measured, not nominal) -- see the timing patch in that file. Resample
    # sim onto real's timestamps for a fair comparison.
    t_sim, t_real = sim["t"], real["t_wall"]

    print("\n=== Sim vs. real tracking comparison ===")
    for name in joint_names:
        sim_key, real_key = f"actual_deg_{name}", f"actual_deg_{name}"
        if sim_key not in sim or real_key not in real:
            print(f"  {name:8s}: missing columns, skipping")
            continue
        sim_resampled = np.interp(t_real, t_sim, sim[sim_key])
        err = sim_resampled - real[real_key]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        print(f"  {name:8s}: RMSE(sim - real) = {rmse:6.2f} deg over {len(t_real)} samples")

    if plot:
        import matplotlib.pyplot as plt

        out_dir = sim_csv.parent / "compare_plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in joint_names:
            sim_key, real_key = f"actual_deg_{name}", f"actual_deg_{name}"
            if sim_key not in sim or real_key not in real:
                continue
            plt.figure()
            plt.plot(t_real, np.interp(t_real, t_sim, sim[f"target_deg_{name}"]), "k--", label="reference")
            plt.plot(t_sim, sim[sim_key], label="sim")
            plt.plot(t_real, real[real_key], label="real")
            plt.xlabel("t (s)")
            plt.ylabel("deg")
            plt.title(f"{name}: sim vs real")
            plt.legend()
            plt.grid(True)
            path = out_dir / f"compare_{name}.png"
            plt.savefig(path)
            plt.close()
            print(f"  saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keyframes", type=Path, default=Path("keyframes.json"))
    parser.add_argument("--duration", type=float, default=6.0, help="seconds of trajectory to run")
    parser.add_argument("--control-dt", type=float, default=0.02, help="expected control dt, for a mismatch warning only")
    parser.add_argument("--stiffness", type=str, nargs="*", default=None, help="e.g. hip_pitch=55 knee=45 ankle=30")
    parser.add_argument("--damping", type=str, nargs="*", default=None, help="e.g. hip_pitch=0.3 knee=0.5 ankle=0.25")
    parser.add_argument("--armature", type=str, nargs="*", default=None, help="e.g. hip_pitch=0.002 knee=0.002 ankle=0.002")
    parser.add_argument("--ground", action="store_true", help="spawn a ground plane (usually not needed for a single dangling leg)")
    parser.add_argument("--out", type=Path, default=Path("sim_trace.csv"))
    parser.add_argument("--real-log", type=Path, default=None, help="control_loop_*.csv from robot_deploy.py --open-loop-ref")
    parser.add_argument("--joints", type=str, nargs="*", default=["hip", "knee", "ankle"])
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--force-exit", action="store_true",
        help="Skip graceful sim.stop()/close() entirely and os._exit(0) as "
             "soon as the CSV + comparison/plots are written. Use this if "
             "you've confirmed graceful shutdown always hangs on your "
             "machine; otherwise --teardown-timeout already force-exits "
             "automatically after a bounded wait.",
    )
    parser.add_argument(
        "--teardown-timeout", type=float, default=5.0,
        help="Seconds to wait for graceful Isaac Sim shutdown (on a "
             "background thread) before force-exiting anyway. Your CSV/"
             "comparison output is written before this runs, so a timeout "
             "here never loses results.",
    )
    args = parser.parse_args()

    gains = merge_gain_dicts(
        parse_gain_kv(args.stiffness, "stiffness"),
        parse_gain_kv(args.damping, "damping"),
        parse_gain_kv(args.armature, "armature"),
    )
    print(f"Gain overrides: {gains or '(none, using DEFAULT_GAINS from qmini.py)'}")

    sim_csv = run_sim_trace(args, gains)
    print(f"\nDone. Sim trace: {sim_csv}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Automating the search (optional)
# ---------------------------------------------------------------------------
# Once this manual loop works, wrapping it in scipy.optimize.minimize is a
# small step: write a function gains_vector -> RMSE (call run_sim_trace +
# compare_traces's RMSE computation, without launching/closing the sim app
# every call -- keep one SimulationContext alive and just rebuild the
# articulation's actuator gains between trials via build_qmini_cfg) and
# minimize it with e.g. Nelder-Mead, which tolerates the non-smooth,
# noisy objective this comparison produces better than gradient methods.
# I haven't written that loop here since it involves restructuring
# run_sim_trace to not tear down the app each call, which is easier to do
# once you've confirmed the manual version above actually runs on your
# machine.

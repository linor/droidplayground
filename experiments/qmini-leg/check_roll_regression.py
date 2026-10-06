#!/usr/bin/env python3
"""
check_roll_regression.py

Mandatory pre-hardware gate. Replays a candidate policy.pt, open-loop,
against a growing registry of REAL hardware control_loop_*.csv logs that
previously caused a left/right_hip_roll joint-limit abort (or came close),
and reports the worst commanded roll target it would have produced on
each one. Built 2026-10-02 after THREE independent resumes from
pitchstep5 (support_resume, rollmargin13, steplimit20) each looked better
or equal on every sim aggregate metric (fall_rate, p99_step_deg,
step_over5_rate, target_violation_rate, roll tracking error) than
pitchstep5 itself, while actually being WORSE on this exact replay check
every single time -- and all three then failed on real hardware the same
way. This check has caught three real regressions sim's own eval missed
each time; treat a FAIL here as disqualifying, regardless of how good the
sim aggregate numbers look.

METHOD: for each registered log, feeds the real recorded observation
vectors through the candidate policy one step at a time (no closed-loop
simulation, no physics -- just "what would this policy have commanded
given what the real robot's sensors actually saw"), independently for
each policy compared. This is NOT a claim that the candidate would have
produced the exact same subsequent trajectory in closed loop (its own
actions would diverge from what was actually logged after the first
step) -- it's a cheap, fast, no-GPU-needed proxy that has empirically
been a better predictor of real hardware roll-limit failures than
compare_policies_isaaclab.py's full closed-loop sim eval has been, twice
running. Use both, trust this one more for this specific joint/phase.

USAGE
-----
    python3 check_roll_regression.py deploy_bundle_2026-10-02_steplimit20/policy.pt \\
        [more policy.pt paths...] [--csv-dir /path/to/control_loop/csvs]

No Isaac Sim / GPU required -- pure torch + csv, runs in seconds. Add new
entries to CSV_REGISTRY below as new hardware failures (or near-misses)
come in; keep every one, don't prune passing cases out, since a policy
that regresses on an OLD case is just as disqualifying as a new one.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import numpy as np

# (label, control_loop csv filename, real outcome) -- filenames are as
# produced by robot_deploy.py, expected directly in --csv-dir (default:
# the repo root, where these have all landed so far).
CSV_REGISTRY = [
    ("support_resume_a1", "control_loop_20260930_135833.csv", "FAILED: left_hip_roll -20.87deg"),
    ("support_resume_a2", "control_loop_20260930_140018.csv", "FAILED: left_hip_roll -17.73deg"),
    ("support_resume_a3", "control_loop_20260930_140210.csv", "FAILED (different side): right_hip_roll +15.22deg, mt2.12"),
    ("support_resume_a4", "control_loop_20260930_140350.csv", "FAILED: left_hip_roll -17.16deg"),
    ("rollmargin13_a1",   "control_loop_20261001_132203.csv", "IMU fault (roll not the cause here, kept as a roll-safety control case)"),
    ("rollmargin13_a2",   "control_loop_20261001_132350.csv", "FAILED: left_hip_roll -15.73deg"),
    ("rollmargin13_a3",   "control_loop_20261001_132551.csv", "FAILED: left_hip_roll -19.00deg"),
    ("rollmargin13_a4",   "control_loop_20261001_132717.csv", "IMU fault (roll not the cause here, kept as a roll-safety control case)"),
    # 2026-10-05, margin13 bundle, hard feet, rope loose -- all during a
    # backward-lean episode. a12/a18 ran with the ankle forward offset
    # (robot_config_qmini_stepinplace_ankle_fwd2/fwd3.json).
    ("margin13_hw_a5",    "control_loop_20261005_124820.csv", "FAILED: left_hip_roll -15.31deg"),
    ("margin13_hw_a8",    "control_loop_20261005_125504.csv", "FAILED: left_hip_roll -18.84deg"),
    ("margin13_fwd2_a12", "control_loop_20261005_135958.csv", "FAILED: left_hip_roll -15.55deg"),
    ("margin13_fwd3_a18", "control_loop_20261005_143656.csv", "FAILED: left_hip_roll -17.21deg"),
]

ROLL_LIMIT_DEG = 15.0
ACTION_SCALE = 0.5


def load_obs(csv_path: Path) -> tuple[torch.Tensor, np.ndarray]:
    with open(csv_path) as f:
        rows = [x for x in csv.DictReader(f) if x.get("motion_time") and x.get("obs_pos_left_hip_yaw")]
    obs_cols = [k for k in rows[0] if k.startswith("obs_")][:28]
    obs = torch.tensor([[float(x[k]) for k in obs_cols] for x in rows], dtype=torch.float32)
    mt = np.array([float(x["motion_time"]) for x in rows])
    return obs, mt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("policies", nargs="+", help="one or more policy.pt (TorchScript, from export_policy_for_deployment.py)")
    parser.add_argument("--csv-dir", type=str, default=".", help="directory containing the registered control_loop_*.csv files")
    parser.add_argument("--labels", nargs="*", default=None, help="display name per policy, same order/count as policies")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    labels = args.labels or [Path(p).parent.name for p in args.policies]
    if len(labels) != len(args.policies):
        parser.error("--labels must match --policies count")

    policies = {lbl: torch.jit.load(p, map_location="cpu").eval() for lbl, p in zip(labels, args.policies)}

    # joint order from export_policy_for_deployment.py / qmini_leg_env.py's
    # articulation order: yaw, yaw, roll, roll, pitch, pitch, knee, knee,
    # ankle, ankle (left, right interleaved) -- roll is columns 2/3.
    LEFT_ROLL_COL, RIGHT_ROLL_COL = 2, 3

    missing = [name for _, name, _ in CSV_REGISTRY if not (csv_dir / name).exists()]
    if missing:
        print(f"WARNING: {len(missing)} registered CSV(s) not found in {csv_dir}: {missing}")

    results = {lbl: [] for lbl in labels}
    print(f"{'case':<22s} {'real outcome':<55s} " + "  ".join(f"{lbl:>22s}" for lbl in labels))
    for case_label, fname, outcome in CSV_REGISTRY:
        path = csv_dir / fname
        if not path.exists():
            continue
        obs, mt = load_obs(path)
        row_strs = []
        for lbl in labels:
            with torch.no_grad():
                a = policies[lbl](obs).numpy() * ACTION_SCALE * 57.29577951308232
            left_min = a[:, LEFT_ROLL_COL].min()
            right_max = a[:, RIGHT_ROLL_COL].max()
            worst = left_min if abs(left_min) >= abs(right_max) else right_max
            over = abs(worst) - ROLL_LIMIT_DEG
            flag = "FAIL" if over > 0 else ("close" if over > -2 else "ok")
            results[lbl].append(over)
            row_strs.append(f"{worst:7.2f}deg[{flag:>5s}]      ")
        print(f"{case_label:<22s} {outcome:<55s} " + "  ".join(row_strs))

    print("\n=== SUMMARY (worst-case margin to +-15deg limit, negative = violation) ===")
    for lbl in labels:
        overs = np.array(results[lbl])
        n_fail = int((overs > 0).sum())
        n_close = int(((overs <= 0) & (overs > -2)).sum())
        print(f"{lbl:<22s} worst_over={overs.max():+6.2f}deg  mean_over={overs.mean():+6.2f}deg  "
              f"FAIL={n_fail}/{len(overs)}  close={n_close}/{len(overs)}")
    print(
        "\nFAIL = this policy's replayed target exceeded +-15deg on this real log.\n"
        "A policy with MORE fails/closer margins than pitchstep5 (or whichever\n"
        "baseline you're comparing against) should NOT go to hardware, regardless\n"
        "of how its compare_policies_isaaclab.py aggregate metrics look -- see\n"
        "this script's docstring."
    )


if __name__ == "__main__":
    main()

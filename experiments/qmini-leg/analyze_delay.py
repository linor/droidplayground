#!/usr/bin/env python3
"""
analyze_delay.py

Answers two questions from a robot_deploy.py control_loop_*.csv log:

  1. What control rate is the real robot actually achieving? (not the
     nominal control_dt in robot_config.json -- the measured one, from the
     t_wall column added to robot_deploy.py.)
  2. How much delay is there between a commanded target and the motor
     actually getting there? Estimated per joint via cross-correlation
     between target_deg_<joint> and actual_deg_<joint>.

Run this against a log captured with `--open-loop-ref` (so target_deg is
the raw motion reference, a clean, information-rich signal to correlate
against -- a policy's targets are a worse signal for this because they
already contain closed-loop corrections).

USAGE
-----
    python3 analyze_delay.py logs/control_loop_20260810_120000.csv

    # save plots too
    python3 analyze_delay.py logs/control_loop_20260810_120000.csv --plot

NOT RUN IN THIS SESSION -- I don't have a real control_loop_*.csv to test
this against, only the format written by write_csv_header()/_log_step() in
the patched robot_deploy.py. Column names are read from the CSV header, not
hardcoded, so it should tolerate minor future header changes; verify the
printed joint list matches your robot_config.json joints.
"""

from __future__ import annotations

import argparse
import csv as csv_module
from pathlib import Path

import numpy as np


def load_log(path: Path):
    with open(path, newline="") as f:
        reader = csv_module.reader(f)
        header = next(reader)
        rows = [row for row in reader]

    data = {}
    skipped = []
    for i, name in enumerate(header):
        try:
            data[name] = np.array([float(row[i]) for row in rows])
        except ValueError:
            # Non-numeric column (e.g. temp_*/err_* can be empty strings
            # when the SDK doesn't report them) -- not needed for timing/
            # delay analysis, so skip rather than aborting the whole load.
            skipped.append(name)
    if skipped:
        print(f"(skipped non-numeric columns: {', '.join(skipped)})")
    return data, header


def report_loop_timing(data: dict, control_dt_hint: float | None):
    t = data["t_wall"]
    dt = np.diff(t)
    if len(dt) == 0:
        print("Not enough rows to measure loop timing.")
        return
    print("=== Control loop timing (measured from t_wall) ===")
    print(f"  n_steps        : {len(t)}")
    print(f"  mean dt        : {dt.mean()*1000:7.2f} ms  ({1.0/dt.mean():6.1f} Hz)")
    print(f"  median dt      : {np.median(dt)*1000:7.2f} ms")
    print(f"  p95 dt         : {np.percentile(dt, 95)*1000:7.2f} ms")
    print(f"  max dt         : {dt.max()*1000:7.2f} ms")
    if control_dt_hint:
        print(f"  target dt      : {control_dt_hint*1000:7.2f} ms  ({1.0/control_dt_hint:6.1f} Hz)")
        overrun_frac = np.mean(dt > 1.5 * control_dt_hint)
        print(f"  frac. overrun  : {overrun_frac*100:5.1f}% of steps exceed 1.5x target dt")
    if "imu_ms" in data:
        print(f"  mean imu_ms    : {data['imu_ms'].mean():7.2f} ms  "
              f"(<- IMU read, happens before policy_ms starts timing; worth "
              f"watching if you're on a USB-bridged I2C adapter, e.g. "
              f"MCP2221/FT232H, which has higher per-transaction latency "
              f"than the Pi's native GPIO I2C)")
    if "policy_ms" in data and "bus_ms" in data:
        print(f"  mean policy_ms : {data['policy_ms'].mean():7.2f} ms")
        print(f"  mean bus_ms    : {data['bus_ms'].mean():7.2f} ms  "
              f"(<- serial round trips to the motors; this is usually where "
              f"the real cost is, since it's N blocking sendRecv() calls)")
    print()
    print("  If mean dt is well above your intended control_dt, your policy "
          "was trained assuming a faster action-update rate than the real "
          "robot delivers. Either speed up the loop (see bus_ms above -- "
          "batching the sendRecv calls or using a faster bus/baud rate "
          "usually matters more than policy_ms), or retrain/fine-tune with "
          "a decimation matched to the dt you actually measured here.\n")


def estimate_delay(data: dict, joint_names: list[str], control_dt_hint: float | None, max_lag_s: float = 0.3):
    t = data["t_wall"]
    dt_est = float(np.median(np.diff(t))) if len(t) > 1 else (control_dt_hint or 0.02)
    max_lag_steps = max(1, int(round(max_lag_s / dt_est)))

    print("=== Estimated actuation delay (target -> actual cross-correlation) ===")
    print(f"  (search window: 0 to {max_lag_steps} steps, ~{max_lag_steps*dt_est*1000:.0f} ms, at dt~={dt_est*1000:.1f} ms)\n")

    for name in joint_names:
        tgt_key, act_key = f"target_deg_{name}", f"actual_deg_{name}"
        if tgt_key not in data or act_key not in data:
            print(f"  {name:8s}: columns {tgt_key}/{act_key} not found, skipping")
            continue
        target = data[tgt_key] - data[tgt_key].mean()
        actual = data[act_key] - data[act_key].mean()
        n = len(target)
        if n < 2 * max_lag_steps:
            print(f"  {name:8s}: log too short for this max_lag ({n} samples), skipping")
            continue

        # Cross-correlation restricted to non-negative lags (actual should
        # lag target, not lead it -- causality check doubles as a sanity
        # check on the log itself).
        best_lag, best_corr = 0, -np.inf
        denom = np.linalg.norm(target) * np.linalg.norm(actual) + 1e-9
        for lag in range(0, max_lag_steps + 1):
            if lag == 0:
                a, b = target, actual
            else:
                a, b = target[:-lag], actual[lag:]
            if len(a) < 10:
                break
            corr = float(np.dot(a, b)) / denom
            if corr > best_corr:
                best_corr, best_lag = corr, lag

        delay_ms = best_lag * dt_est * 1000.0
        print(f"  {name:8s}: lag = {best_lag:3d} steps  ~= {delay_ms:6.1f} ms   (peak normalized corr = {best_corr:.3f})")

    print()
    print("  This delay includes EVERYTHING between 'policy/reference decided "
          "a target' and 'motor position actually got there': control loop "
          "period, serial round trip, the GO-M8010-6's internal FOC/current "
          "loop, and mechanical response. That whole number is what you want "
          "to reproduce as an action delay in sim (see the QminiLegEnvCfg "
          "action_delay_range_steps addition) -- not just the communication "
          "latency in isolation.\n")


def maybe_plot(data: dict, joint_names: list[str], out_dir: Path):
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    t = data["t_wall"]
    for name in joint_names:
        tgt_key, act_key = f"target_deg_{name}", f"actual_deg_{name}"
        if tgt_key not in data or act_key not in data:
            continue
        plt.figure()
        plt.plot(t, data[tgt_key], label="target", linestyle="--")
        plt.plot(t, data[act_key], label="actual")
        plt.xlabel("t_wall (s)")
        plt.ylabel("deg")
        plt.title(f"{name}: target vs actual")
        plt.legend()
        plt.grid(True)
        out_path = out_dir / f"delay_{name}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"  saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--control-dt", type=float, default=0.02, help="nominal control_dt from robot_config.json, for comparison only")
    parser.add_argument("--joints", type=str, nargs="*", default=["hip", "knee", "ankle"])
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    data, header = load_log(args.csv_path)
    print(f"Loaded {args.csv_path} ({len(header)} columns, {len(data['t_wall'])} rows)\n")

    report_loop_timing(data, args.control_dt)
    estimate_delay(data, args.joints, args.control_dt)

    if args.plot:
        maybe_plot(data, args.joints, args.csv_path.parent / "delay_plots")


if __name__ == "__main__":
    main()

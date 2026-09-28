#!/usr/bin/env python3
"""
compare_policies_isaaclab.py

Standalone eval/comparison tool for qmini-leg checkpoints -- built alongside
the tracking_reward/velocity_reward linear-penalty fix in qmini_leg_env.py
(inspired by the Disney Research BD-X paper's reward shape) to answer the
question that came up when discussing that change: once you've changed the
reward function, raw reward totals between an old checkpoint and a new one
aren't comparable (they're not even measuring the same thing anymore) -- you
need PHYSICAL metrics instead: fall rate, episode length, per-joint tracking
error in degrees, how often the raw commanded target exceeds a joint's hard
limit (see target_limit_penalty's comment in qmini_leg_env.py -- this is
exactly the class of real-hardware safety abort FROZEN_JOINTS in
robot_deploy.py is currently working around), and action smoothness. This
script runs one or more checkpoints through a FIXED, deterministic eval
(same env seed => same domain-randomization draws: startup pose noise,
pushes, IMU noise/bias, mass -- every policy compared sees the identical
sequence of "bad luck") and reports those metrics side by side.

HOW TO USE THIS AS TRAINING PROGRESSES
---------------------------------------
Run it ONCE PER CHECKPOINT, right after each run/resume you want to record,
pointing --out at the SAME csv every time:

    ./isaaclab.sh -p compare_policies_isaaclab.py \\
        logs/rsl_rl/qmini-leg/2026-08-10_.../model_28000.pt \\
        --out compare.csv --headless

    ./isaaclab.sh -p compare_policies_isaaclab.py \\
        logs/rsl_rl/qmini-leg/2026-08-16_.../model_44800.pt \\
        --out compare.csv --headless

Each run APPENDS a row (creating the header if the file is new) and prints
the comparison table built from every row accumulated in the file so far --
not just the checkpoint(s) from this invocation -- so the table naturally
grows across separate training runs finishing on different days without you
tracking numbers by hand. You can also just view the accumulated table
without touching Isaac Sim at all:

    python3 compare_policies_isaaclab.py --print-only compare.csv

Passing multiple checkpoints to one invocation is also supported (they run
sequentially in the same Isaac Sim session and get appended as separate
rows) -- but this repo has NOT verified whether Isaac Lab's DirectRLEnv
tears down cleanly enough to construct a second, independent env instance
in the same process (most scripts in this repo, like tune_pid_isaaclab.py,
only ever construct one). If the second checkpoint's eval hangs, errors, or
gets contaminated by leftover state from the first, fall back to the
one-checkpoint-per-invocation form above -- it sidesteps the question
entirely since --out already accumulates across processes.

WHY A FIXED SEED MATTERS
--------------------------
qmini_leg_env.py's domain randomization (startup_joint_pos_noise_deg,
push_velocity_range_mps, imu_gyro_bias_range_deg_s, imu_mount_bias_range_deg,
base_mass_randomization_range, gain_randomization_range, action_delay_range_steps)
all draw from per-env RNG state seeded off cfg.seed at env construction. This
script pins --seed to the same value (default 42) for every checkpoint it
evaluates, specifically so no policy gets an easier or harder random draw
than another purely by chance -- any metric difference you see should
reflect the policy, not the dice roll.

*** NOT EXECUTED OR TESTED IN THIS SESSION *** (no Isaac Sim/GPU available
here) -- written against the same isaaclab / rsl_rl APIs
rl/scripts/rsl_rl/play.py already uses in this repo (hydra_task_config,
RslRlVecEnvWrapper, OnPolicyRunner.load/get_inference_policy) plus direct
introspection of env.unwrapped, the same pattern qmini_leg_env.py's own
_get_rewards() uses internally. Expect to debug small API differences
(e.g. whether RslRlVecEnvWrapper.step's extras dict really carries
"time_outs" under that exact key in your installed Isaac Lab version --
if not, fall_rate/mean_episode_length_s will silently read as N/A rather
than crash, see the comment at collect_metrics() below).
"""

from __future__ import annotations

import argparse
import csv as csv_module
import json
import sys
from datetime import datetime
from pathlib import Path

# --- metrics reported per checkpoint, in the order printed/written -------
# "physical" metrics are reward-function-agnostic (degrees, booleans,
# seconds) -- safe to compare across checkpoints even if tracking_reward's
# shape/weights changed between the runs that produced them. Kept as a
# module-level list (not just a dict) so the printed table and the CSV
# columns are guaranteed to agree on ordering.
SUMMARY_FIELDS = [
    "label", "checkpoint", "timestamp", "num_envs", "eval_seconds", "seed",
    "completed_episodes", "fall_rate", "mean_episode_length_s",
    "mean_tracking_err_deg", "mean_tilt_deg", "mean_roll_deg", "mean_pitch_deg",
    "target_violation_rate", "mean_overshoot_deg", "mean_action_delta_deg",
    "p99_left_swing_target", "p99_right_swing_target",
    "p99_left_foot_height_cm", "p99_right_foot_height_cm",
    "p99_left_swing_product", "p99_right_swing_product",
]


def print_table(rows: list[dict]):
    if not rows:
        print("(no rows to show)")
        return

    def fmt(key: str, row: dict) -> str:
        v = row.get(key, "")
        if v == "" or v is None:
            return "n/a"
        try:
            f = float(v)
        except (TypeError, ValueError):
            return str(v)
        if key in ("fall_rate", "target_violation_rate"):
            return f"{f * 100:6.2f}%"
        if key in ("num_envs", "seed", "completed_episodes"):
            return f"{int(f):d}"
        return f"{f:7.3f}"

    display_cols = [
        ("label", "policy"),
        ("fall_rate", "fall_rate"),
        ("mean_episode_length_s", "ep_len_s"),
        ("mean_tracking_err_deg", "track_err_deg"),
        ("mean_tilt_deg", "tilt_deg"),
        ("mean_roll_deg", "roll_deg"),
        ("mean_pitch_deg", "pitch_deg"),
        ("target_violation_rate", "tgt_viol_rate"),
        ("mean_overshoot_deg", "overshoot_deg"),
        ("mean_action_delta_deg", "act_delta_deg"),
        ("p99_left_swing_product", "p99_L_swing"),
        ("p99_right_swing_product", "p99_R_swing"),
        ("completed_episodes", "n_ep"),
    ]
    # Column widths sized from the actual formatted content, not a fixed
    # guess -- default checkpoint labels are "<run_name>/model_XXXX.pt" or
    # "<dir>/policy.pt" (see load_csv_rows()'s default label), routinely
    # 30+ chars, well past any fixed floor.
    formatted_rows = [{key: fmt(key, row) for key, _ in display_cols} for row in rows]
    widths = {
        key: max(len(header), max((len(fr[key]) for fr in formatted_rows), default=0))
        for key, header in display_cols
    }
    header_line = "  ".join(f"{header:<{widths[key]}}" for key, header in display_cols)
    print(header_line)
    print("-" * len(header_line))
    for fr in formatted_rows:
        print("  ".join(f"{fr[key]:<{widths[key]}}" for key, _ in display_cols))

    # Per-joint breakdown, since the main table above collapses all joints
    # into one mean_tracking_err_deg for readability.
    joint_keys = sorted({k for row in rows for k in row if k.startswith("err_deg_")})
    if joint_keys:
        print("\nPer-joint tracking error (deg):")
        joint_labels = [k.removeprefix("err_deg_") for k in joint_keys]
        label_width = max(len("policy"), max((len(str(row.get("label", "?"))) for row in rows), default=6))
        col_width = max(10, max((len(lbl) for lbl in joint_labels), default=10))
        header_line = f"{'policy':<{label_width}s}  " + "  ".join(f"{lbl:>{col_width}s}" for lbl in joint_labels)
        print(header_line)
        print("-" * len(header_line))
        for row in rows:
            label = str(row.get("label", "?"))
            vals = []
            for k in joint_keys:
                v = row.get(k, "")
                try:
                    vals.append(f"{float(v):{col_width}.2f}")
                except (TypeError, ValueError):
                    vals.append(f"{'n/a':>{col_width}s}")
            print(f"{label:<{label_width}s}  " + "  ".join(vals))

    # Foot-swing diagnostic breakdown -- p99_*_swing_target/foot_height_cm
    # separate the swing-product column above into its two factors, so you
    # can tell WHICH part is missing when p99_*_swing is near 0: does
    # swing_target (the reference's own gait-phase signal) ever get high,
    # does foot_height (actual measured lift) ever get high, or does
    # neither? See collect_metrics()'s comment for what each combination
    # implies.
    swing_diag_cols = [
        ("p99_left_swing_target", "L_swing_tgt"),
        ("p99_right_swing_target", "R_swing_tgt"),
        ("p99_left_foot_height_cm", "L_height_cm"),
        ("p99_right_foot_height_cm", "R_height_cm"),
    ]
    if any(col in row for row in rows for col, _ in swing_diag_cols):
        print("\nFoot-swing diagnostic (population 99th percentile over the whole eval):")
        label_width = max(len("policy"), max((len(str(row.get("label", "?"))) for row in rows), default=6))
        widths2 = {
            col: max(len(header), max((len(f"{float(row[col]):.3f}") for row in rows if col in row), default=0))
            for col, header in swing_diag_cols
        }
        header_line = f"{'policy':<{label_width}s}  " + "  ".join(
            f"{header:>{widths2[col]}}" for col, header in swing_diag_cols
        )
        print(header_line)
        print("-" * len(header_line))
        for row in rows:
            label = str(row.get("label", "?"))
            vals = []
            for col, _ in swing_diag_cols:
                v = row.get(col, "")
                try:
                    vals.append(f"{float(v):{widths2[col]}.3f}")
                except (TypeError, ValueError):
                    vals.append(f"{'n/a':>{widths2[col]}}")
            print(f"{label:<{label_width}s}  " + "  ".join(vals))


def _label_for_checkpoint(resume_path) -> str:
    """Default display label for a checkpoint. A raw rsl_rl checkpoint
    naturally carries a date via its run directory name
    (logs/rsl_rl/qmini-leg/<date>_<time>/model_XXXX.pt). An exported
    TorchScript policy.pt loses that -- it's two directories this project
    exports one to, both dateless on their own:
      1. export_policy_for_deployment.py, writing to whatever fixed
         --output-dir you gave it -- but its sidecar policy.meta.json
         records "source_checkpoint", the original dated raw checkpoint
         path, so recover the date from there when that sidecar is sitting
         next to it.
      2. rl/scripts/rsl_rl/play.py's own export (export_policy_as_jit/onnx,
         no policy.meta.json sidecar), which writes straight into a fixed
         "exported/" subfolder of the run directory it just loaded from
         (see play.py's export_model_dir = dirname(resume_path)/exported)
         -- climb one more level to recover the run's date in that case.
    """
    resume_path = Path(resume_path)

    meta_path = resume_path.parent / "policy.meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            source = meta.get("source_checkpoint")
            if source:
                return f"{Path(source).parent.name}/{resume_path.name}"
        except (json.JSONDecodeError, OSError):
            pass  # sidecar present but unreadable/malformed -- fall through

    if resume_path.parent.name == "exported":
        return f"{resume_path.parent.parent.name}/{resume_path.parent.name}/{resume_path.name}"

    return f"{resume_path.parent.name}/{resume_path.name}"


def load_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv_module.DictReader(f))


# Added 2026-09-22 (per-step commanded-target change, what robot_deploy.py's
# max_step_deg check measures). Deliberately appended AFTER the err_deg_*
# columns so they never shift the position of any column already in an
# accumulated CSV (write_csv_rows appends without rewriting the header --
# an existing CSV must be migrated to include these columns once, see git
# history / the 2026-09-22 migration).
EXTRA_FIELDS = ["step_over5_rate", "p99_step_deg"]


def write_csv_rows(path: Path, rows: list[dict]):
    fieldnames = list(SUMMARY_FIELDS)
    joint_keys = sorted({k for row in rows for k in row if k.startswith("err_deg_")})
    fieldnames += [k for k in joint_keys if k not in fieldnames]
    fieldnames += [k for k in EXTRA_FIELDS if k not in fieldnames]
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _flat_policy_obs(obs):
    """RslRlVecEnvWrapper.get_observations()/step() hand back a TensorDict
    (or a plain dict, in older Isaac Lab versions) keyed by obs group --
    "policy" is the group qmini_leg_env.py's _get_observations() returns.
    OnPolicyRunner.get_inference_policy()'s wrapper apparently unwraps this
    internally (that path works with `obs` passed straight through, as
    play.py does), but a TorchScript module exported by
    export_policy_for_deployment.py / export_policy_as_jit expects a plain
    flat tensor -- exactly what robot_deploy.py's build_obs() constructs by
    hand on the real robot -- so extract that tensor explicitly here."""
    import torch

    if isinstance(obs, torch.Tensor):
        return obs
    return obs["policy"]


DUMP_HEIGHTS_PATH = None


def collect_metrics(env, policy, num_steps: int, num_envs: int, action_scale: float) -> dict:
    """Runs `num_steps` deterministic control steps and returns aggregate
    physical metrics. Reads env.unwrapped directly (same fields
    qmini_leg_env.py's own _get_rewards()/_apply_action() populate) rather
    than re-deriving joint tracking / limit logic here, so this stays in
    sync with that file automatically -- the one exception is target
    over-limit and orientation tilt, computed fresh in degrees so they're
    independent of target_limit_penalty_weight/orientation_reward_scale
    (reward-function tuning shouldn't change what "tilt" or "overshoot"
    MEANS in degrees, only how much it's penalized)."""
    import torch
    from isaaclab.utils.math import quat_apply

    unwrapped = env.unwrapped
    device = unwrapped.device
    num_joints = unwrapped.num_joints

    steps_since_reset = torch.zeros(num_envs, device=device)
    episode_lengths_s = []
    fall_flags = []
    log_sums: dict[str, float] = {}
    log_count = 0
    tilt_deg_sum = 0.0
    # Split out of tilt_deg (which is the combined xy-gravity-vector norm,
    # i.e. total lean regardless of axis) -- added 2026-09-15 to check
    # whether roll or pitch is the bigger contributor after video review
    # flagged roll specifically as visibly worse than reference footage of
    # this robot class, something mean_tilt_deg alone can't distinguish.
    # projected_gravity_b's X component is ROLL, Y is PITCH -- per
    # robot_config_qmini_stepinplace.json's _imu_comment (base_link +X
    # points toward the LEFT leg, +Y toward the REAR) and the standard
    # convention (roll = rotation about the fore-aft/sagittal axis, i.e.
    # about body Y here; that rotation projects world gravity onto body X).
    roll_deg_sum = 0.0
    pitch_deg_sum = 0.0
    target_violation_step_count = 0
    overshoot_deg_sum = 0.0
    overshoot_deg_count = 0
    action_delta_deg_sum = 0.0
    prev_actions = torch.zeros(num_envs, unwrapped.cfg.action_space, device=device)
    # Per-step commanded-target change (degrees), per env, reset to 0 on env
    # reset exactly like the training env's _prev_actions -- so the first
    # step of an episode counts (it is hardware's first commanded step too).
    prev_actions_step = torch.zeros(num_envs, unwrapped.cfg.action_space, device=device)
    step_over5_count = 0
    step_max_deg_samples: list = []
    step_excess_sum_samples: list = []  # per env-step: sum_j clamp(delta_j - 4deg, 0, 30deg) in rad (x weight = training penalty)

    # Population-wide 99th PERCENTILE (not mean, and NOT a raw max) of the
    # foot-swing-reward signal's components -- added to investigate a run
    # where tracking/foot_swing_reward (a .mean() across all envs, logged
    # every training step) stayed pinned at a tiny ceiling (~8e-4, vs. a
    # theoretical max around 1.0 per foot) for an entire 40k-iteration run.
    # A mean that never moves despite real exploration looks structural,
    # not just "policy avoiding it" -- but a MEAN across ~4096 envs at
    # asynchronous gait phases can hide a real signal that only a few envs
    # produce at any instant, so some population-wide "how good does it get"
    # statistic is still needed.
    #
    # ORIGINALLY a raw running max, not a percentile -- changed after TWO
    # separate real evals showed multiple, visibly-different-quality
    # checkpoints reporting suspiciously identical (sometimes bit-for-bit)
    # readings. Root cause: a raw max is extremely sensitive to a SINGLE
    # rare, large, policy-INDEPENDENT transient -- e.g. a violent
    # settling snap right after a large random startup-pose draw
    # (cfg.startup_joint_pos_noise_deg), or a push disturbance
    # (cfg.push_interval_range_s, re-enabled during training AND active here
    # since eval reuses the same env cfg) throwing a foot into a large but
    # gait-unrelated excursion. Both are seed-deterministic (same --seed
    # every checkpoint eval, by design, for apples-to-apples comparison),
    # so different POLICIES sharing the same domain-randomization draws
    # would report the identical max even though real per-policy stepping
    # behavior differs -- confirmed via checkpoints with 0% fall_rate (i.e.
    # identical reset/no-early-termination timing) reporting bit-for-bit
    # identical max_left/right_foot_height_cm. A 99th percentile over every
    # settled sample in the whole eval is far less sensitive to a handful
    # of such outlier events while still reflecting genuine, sustained
    # clearance rather than collapsing to a mean that dilutes across
    # asynchronous gait phases -- BUT this alone still wasn't enough:
    # pushes fire for every env at once and can account for well over 1% of
    # total samples, so the percentile alone didn't exclude them either.
    # SETTLE_STEPS below now explicitly excludes samples shortly after
    # EITHER a reset or a push, on top of the percentile -- belt and
    # suspenders, since either alone proved insufficient in practice.
    left_swing_target_samples: list = []
    right_swing_target_samples: list = []
    left_foot_height_cm_samples: list = []
    right_foot_height_cm_samples: list = []
    left_swing_product_samples: list = []
    right_swing_product_samples: list = []
    PERCENTILE = 0.99

    # OPTIONAL raw dump (--dump-heights PATH): full per-step, per-env arrays
    # (NOT masked/percentiled) so per-swing peak heights, medians, and left/
    # right asymmetry can be analysed offline -- p99 alone is a tail number
    # that can hide what a typical swing looks like.
    dump_rows: list = []

    # Skip this many steps right after EACH env's reset, AND right after
    # each push-disturbance event, before trusting body_pos_w-derived
    # readings above. Reset: a large random startup-pose draw can produce a
    # violent settling transient in the first fraction of a second, before
    # the PD controller catches up. Push (cfg.push_interval_range_s, re-enabled
    # during training and active here since eval reuses the same env cfg):
    # a real, seed-deterministic velocity kick applied to ALL envs
    # simultaneously that can throw a foot into a large but gait-unrelated
    # excursion. The 99th-percentile switch above wasn't aggressive enough
    # to exclude these on its own -- pushes fire 5 times in a 20s eval and
    # hit every env at once, so their recovery transient can easily exceed
    # 1% of total samples. ~0.5s (25 steps at the default 0.02s step_dt) is
    # a guess at how long either kind of settling takes -- not verified
    # against an actual measured transient duration.
    SETTLE_STEPS = 25

    obs = env.get_observations()
    for step in range(num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _rew, dones, extras = env.step(actions)

        steps_since_reset += 1
        settled_mask = steps_since_reset >= SETTLE_STEPS
        # Push timing is per-env now (cfg.push_interval_range_s, see that
        # cfg's comment -- no longer a shared global schedule every env
        # hits on the same step), so read the env's own per-env cooldown
        # tensor directly instead of reconstructing it from a global step
        # counter.
        steps_since_push = getattr(unwrapped, "_steps_since_push", None)
        if steps_since_push is not None:
            settled_mask = settled_mask & (steps_since_push >= SETTLE_STEPS)

        reference = unwrapped.motion.sample(unwrapped.motion_time)
        left_swing_target = torch.clamp(
            (reference[:, unwrapped._left_knee_idx] - unwrapped._left_knee_stance_rad)
            / (unwrapped._left_knee_swing_extreme_rad - unwrapped._left_knee_stance_rad),
            min=0.0, max=1.0,
        )
        right_swing_target = torch.clamp(
            (reference[:, unwrapped._right_knee_idx] - unwrapped._right_knee_stance_rad)
            / (unwrapped._right_knee_swing_extreme_rad - unwrapped._right_knee_stance_rad),
            min=0.0, max=1.0,
        )
        # World-frame HEEL and TOE positions, matching qmini_leg_env.py's
        # own _get_rewards() exactly (see cfg.left/right_heel_local_m and
        # left/right_toe_local_m's comments there) -- NOT the raw foot
        # body origin (anchored at the ankle joint axis), and NOT heel
        # alone either: heel-only was still gameable via the mirror-image
        # plantarflexion exploit (heel lifts, toes stay planted). Requires
        # BOTH points to clear -- see the min() below. This eval script
        # has its own, separate height computation (doesn't reuse the
        # env's internal one), so it needed this fix applied independently
        # too, same as the heel-only fix before it.
        left_heel_pos_w = unwrapped.robot.data.body_pos_w[:, unwrapped._left_foot_body_idx, :] + quat_apply(
            unwrapped.robot.data.body_quat_w[:, unwrapped._left_foot_body_idx, :],
            unwrapped._left_heel_local_offset,
        )
        right_heel_pos_w = unwrapped.robot.data.body_pos_w[:, unwrapped._right_foot_body_idx, :] + quat_apply(
            unwrapped.robot.data.body_quat_w[:, unwrapped._right_foot_body_idx, :],
            unwrapped._right_heel_local_offset,
        )
        left_toe_pos_w = unwrapped.robot.data.body_pos_w[:, unwrapped._left_foot_body_idx, :] + quat_apply(
            unwrapped.robot.data.body_quat_w[:, unwrapped._left_foot_body_idx, :],
            unwrapped._left_toe_local_offset,
        )
        right_toe_pos_w = unwrapped.robot.data.body_pos_w[:, unwrapped._right_foot_body_idx, :] + quat_apply(
            unwrapped.robot.data.body_quat_w[:, unwrapped._right_foot_body_idx, :],
            unwrapped._right_toe_local_offset,
        )
        left_heel_height = left_heel_pos_w[:, 2] - unwrapped._left_foot_stance_height
        right_heel_height = right_heel_pos_w[:, 2] - unwrapped._right_foot_stance_height
        left_toe_height = left_toe_pos_w[:, 2] - unwrapped._left_toe_stance_height
        right_toe_height = right_toe_pos_w[:, 2] - unwrapped._right_toe_stance_height
        left_foot_height = torch.minimum(left_heel_height, left_toe_height)
        right_foot_height = torch.minimum(right_heel_height, right_toe_height)
        left_clearance = torch.clamp(left_foot_height / unwrapped.cfg.foot_swing_target_height_m, min=0.0, max=1.0)
        right_clearance = torch.clamp(right_foot_height / unwrapped.cfg.foot_swing_target_height_m, min=0.0, max=1.0)

        if DUMP_HEIGHTS_PATH is not None:
            _fh = unwrapped.foot_height_motion.sample(unwrapped.motion_time)
            dump_rows.append(torch.stack([
                torch.minimum(left_heel_height, left_toe_height), torch.maximum(left_heel_height, left_toe_height),
                torch.minimum(right_heel_height, right_toe_height), torch.maximum(right_heel_height, right_toe_height),
                left_swing_target, right_swing_target, settled_mask.float(),
                _fh[:, 0], _fh[:, 1], _fh[:, 2], _fh[:, 3],
                # cols 11-20 actual joint_pos, 21-30 reference joint_pos (rad, articulation order,
                # names saved next to the .npy), 31 base z, 32-34 projected_gravity_b, 35-36
                # root_lin_vel_b xy (m/s, base frame -- added 2026-09-28 alongside
                # position_reward_weight going 0.0->1.0, to see WHERE/WHEN drift happens instead
                # of just training's aggregate position/base_speed_cmps scalar)
                *unwrapped.robot.data.joint_pos[:, :num_joints].T,
                *reference[:, :num_joints].T,
                unwrapped.robot.data.root_pos_w[:, 2],
                *unwrapped.robot.data.projected_gravity_b.T,
                *unwrapped.robot.data.root_lin_vel_b[:, :2].T,
            ], dim=1).cpu())

        if settled_mask.any():
            left_swing_target_samples.append(left_swing_target[settled_mask].cpu())
            right_swing_target_samples.append(right_swing_target[settled_mask].cpu())
            left_foot_height_cm_samples.append((left_foot_height[settled_mask] * 100.0).cpu())
            right_foot_height_cm_samples.append((right_foot_height[settled_mask] * 100.0).cpu())
            left_swing_product_samples.append((left_swing_target * left_clearance)[settled_mask].cpu())
            right_swing_product_samples.append((right_swing_target * right_clearance)[settled_mask].cpu())

        dones_bool = dones.bool()
        # rsl_rl / Isaac Lab convention: extras["time_outs"] is truncation
        # (episode_length_s reached), separate from dones which combines
        # both terminated (fell) and truncated. If a future Isaac Lab
        # version doesn't expose this key, degrade to "can't distinguish
        # fall from timeout" rather than crashing the whole eval.
        time_outs = extras.get("time_outs")
        if time_outs is not None:
            time_outs_bool = time_outs.bool()
            fell_mask = dones_bool & ~time_outs_bool
        else:
            fell_mask = dones_bool
        for i in torch.nonzero(dones_bool).flatten().tolist():
            episode_lengths_s.append(float(steps_since_reset[i]) * unwrapped.step_dt)
            fall_flags.append(bool(fell_mask[i]))
        steps_since_reset[dones_bool] = 0.0

        step_log = getattr(unwrapped, "extras", {}).get("log", {})
        for k, v in step_log.items():
            log_sums[k] = log_sums.get(k, 0.0) + float(v)
        log_count += 1

        gxy = unwrapped.robot.data.projected_gravity_b[:, :2]
        tilt_rad = torch.asin(torch.clamp(gxy.norm(dim=1), max=1.0))
        tilt_deg_sum += float(torch.rad2deg(tilt_rad).mean())
        roll_rad = torch.asin(torch.clamp(gxy[:, 0].abs(), max=1.0))
        pitch_rad = torch.asin(torch.clamp(gxy[:, 1].abs(), max=1.0))
        roll_deg_sum += float(torch.rad2deg(roll_rad).mean())
        pitch_deg_sum += float(torch.rad2deg(pitch_rad).mean())

        limits = unwrapped.robot.data.joint_pos_limits[:, :num_joints, :]
        lower, upper = limits[..., 0], limits[..., 1]
        targets = unwrapped._last_position_targets
        over = torch.clamp(targets - upper, min=0.0) + torch.clamp(lower - targets, min=0.0)
        viol_mask = over > 0
        target_violation_step_count += int(viol_mask.any(dim=1).sum())
        if viol_mask.any():
            overshoot_deg_sum += float(torch.rad2deg(over[viol_mask]).sum())
            overshoot_deg_count += int(viol_mask.sum())

        action_delta_deg = torch.rad2deg(action_scale * (actions - prev_actions).abs()).mean()
        action_delta_deg_sum += float(action_delta_deg)
        prev_actions = actions.clone()
        _step_deg = torch.rad2deg(action_scale * (actions - prev_actions_step).abs()).max(dim=1).values
        step_over5_count += int((_step_deg > 5.0).sum())
        step_max_deg_samples.append(_step_deg.cpu())
        _dj = torch.rad2deg(action_scale * (actions - prev_actions_step).abs())
        step_excess_sum_samples.append(torch.deg2rad(torch.clamp(_dj - unwrapped.cfg.step_limit_threshold_deg, min=0.0, max=unwrapped.cfg.step_limit_penalty_max_excess_deg)).sum(dim=1).cpu())
        prev_actions_step = actions.clone()
        prev_actions_step[dones_bool] = 0.0

        if (step + 1) % 100 == 0 or step + 1 == num_steps:
            print(f"    step {step + 1}/{num_steps}")

    if DUMP_HEIGHTS_PATH is not None and dump_rows:
        import numpy as _np
        _np.save(DUMP_HEIGHTS_PATH, torch.stack(dump_rows).numpy())  # [steps, envs, 37]
        with open(str(DUMP_HEIGHTS_PATH) + ".joints.json", "w") as _f:
            json.dump(list(unwrapped.robot.joint_names[:num_joints]), _f)
        print(f"    dumped raw heights to {DUMP_HEIGHTS_PATH}")

    if step_excess_sum_samples:
        _ex = torch.cat(step_excess_sum_samples)
        print(f"    step-limit penalty at cfg weight {unwrapped.cfg.step_limit_penalty_weight}: mean {float(_ex.mean()) * unwrapped.cfg.step_limit_penalty_weight:.3f}/step, p99 {float(torch.quantile(_ex, 0.99)) * unwrapped.cfg.step_limit_penalty_weight:.2f}, max {float(_ex.max()) * unwrapped.cfg.step_limit_penalty_weight:.2f}")
    mean_log = {k: v / log_count for k, v in log_sums.items()} if log_count else {}
    joint_err_keys = [k for k in mean_log if k.startswith("tracking/") and k.endswith("_error")]
    mean_tracking_err_deg = (
        sum(mean_log[k] for k in joint_err_keys) / len(joint_err_keys) if joint_err_keys else float("nan")
    )
    completed = len(episode_lengths_s)

    def _percentile(samples: list) -> float:
        if not samples:
            return float("nan")
        return float(torch.quantile(torch.cat(samples), PERCENTILE))

    result = {
        "completed_episodes": completed,
        "fall_rate": (sum(fall_flags) / completed) if completed else float("nan"),
        "mean_episode_length_s": (sum(episode_lengths_s) / completed) if completed else float("nan"),
        "mean_tracking_err_deg": mean_tracking_err_deg,
        "mean_tilt_deg": tilt_deg_sum / num_steps,
        "mean_roll_deg": roll_deg_sum / num_steps,
        "mean_pitch_deg": pitch_deg_sum / num_steps,
        "target_violation_rate": target_violation_step_count / (num_steps * num_envs),
        "mean_overshoot_deg": (overshoot_deg_sum / overshoot_deg_count) if overshoot_deg_count else 0.0,
        "mean_action_delta_deg": action_delta_deg_sum / num_steps,
        "step_over5_rate": step_over5_count / (num_steps * num_envs),
        "p99_step_deg": float(torch.quantile(torch.cat(step_max_deg_samples), PERCENTILE)),
        "p99_left_swing_target": _percentile(left_swing_target_samples),
        "p99_right_swing_target": _percentile(right_swing_target_samples),
        "p99_left_foot_height_cm": _percentile(left_foot_height_cm_samples),
        "p99_right_foot_height_cm": _percentile(right_foot_height_cm_samples),
        "p99_left_swing_product": _percentile(left_swing_product_samples),
        "p99_right_swing_product": _percentile(right_swing_product_samples),
    }
    for k in joint_err_keys:
        label = k.removeprefix("tracking/").removesuffix("_error")
        result[f"err_deg_{label}"] = mean_log[k]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", type=str, nargs="*", help="one or more rsl_rl OnPolicyRunner checkpoint .pt files to evaluate")
    parser.add_argument("--labels", type=str, nargs="*", default=None, help="display name per checkpoint, same order/count as checkpoints (default: checkpoint's parent run dir + filename)")
    parser.add_argument("--task", type=str, default="DroidPlayground-QMini-Leg")
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
    parser.add_argument("--num_envs", type=int, default=128, help="parallel envs for the eval (more envs = less noisy aggregate stats, not more episodes per env)")
    parser.add_argument("--eval-seconds", type=float, default=20.0, help="deterministic eval duration in sim seconds (default 20s ~= 2x episode_length_s, so most envs complete at least one full episode)")
    parser.add_argument("--seed", type=int, default=42, help="env seed, held identical across every checkpoint compared -- see module docstring's WHY A FIXED SEED MATTERS")
    parser.add_argument("--out", type=Path, default=None, help="CSV path to append results to (created with header if new); the printed table is built from the FULL accumulated file, not just this run's checkpoints")
    parser.add_argument("--print-only", type=Path, default=None, help="skip Isaac Sim entirely, just load and print an existing --out csv")
    parser.add_argument("--dump-heights", type=str, default=None, help="optional .npy path: save raw per-step per-env foot heights [steps, envs, 11] for offline distribution analysis")
    args_cli, extra = parser.parse_known_args()
    global DUMP_HEIGHTS_PATH
    DUMP_HEIGHTS_PATH = args_cli.dump_heights

    if args_cli.print_only:
        print_table(load_csv_rows(args_cli.print_only))
        return

    if not args_cli.checkpoints:
        parser.error("provide at least one checkpoint, or use --print-only <csv>")
    if args_cli.labels and len(args_cli.labels) != len(args_cli.checkpoints):
        parser.error("--labels must have the same count as checkpoints")
    if len(args_cli.checkpoints) > 1:
        # Multiple checkpoints in one Isaac Sim session is still not
        # independently verified to tear down cleanly between gym.make()
        # calls (see the module docstring). It initially looked like the
        # cause of visibly-different checkpoints reporting identical
        # foot-swing diagnostic values -- but a SEPARATE, one-checkpoint-
        # per-invocation run reproduced the same identical readings, which
        # ruled that out as the (sole) explanation. The real cause was the
        # diagnostic itself: a raw MAX is extremely sensitive to a single
        # rare, seed-deterministic transient (startup-pose settling, or a
        # push disturbance) that's identical across different policies
        # sharing the same --seed. Now computed as a 99th percentile
        # instead (see collect_metrics()'s comment), which should be
        # trustworthy in either mode. Left this notice in case multi-
        # checkpoint sessions turn out to have a genuine, separate leakage
        # issue after all -- if results still look suspiciously identical
        # across checkpoints, rerun with one checkpoint per invocation (the
        # --out csv append behavior exists exactly to make that convenient)
        # to rule that back in.
        print(
            f"[INFO] Evaluating {len(args_cli.checkpoints)} checkpoints in one "
            f"invocation. Foot-swing diagnostics now use a 99th percentile, not a "
            f"raw max, so they should be trustworthy here -- but if any column "
            f"looks suspiciously identical across checkpoints, rerun with one "
            f"checkpoint per invocation to rule out a session-leakage issue."
        )

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import copy

    import gymnasium as gym
    import torch

    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from rsl_rl.runners import OnPolicyRunner

    import droidplayground.tasks  # noqa: F401

    def load_policy(resume_path, env, agent_cfg):
        """Accepts either checkpoint kind this project produces:

          1. A raw rsl_rl OnPolicyRunner checkpoint -- the
             logs/rsl_rl/qmini-leg/<run>/model_XXXX.pt files a training run
             saves periodically (a dict with a "model_state_dict" key).
          2. An already-exported TorchScript policy -- e.g.
             experiments/qmini-leg/policy.pt, produced by
             export_policy_for_deployment.py or rsl_rl's own
             export_policy_as_jit (which is what play.py calls). This is
             exactly the file robot_deploy.py loads on the real robot, so
             comparing a deployed policy.pt against a new candidate
             checkpoint is a natural use of this script.

        Detected the same way export_policy_for_deployment.py's own
        load_policy() does: try torch.jit.load first (raises RuntimeError,
        not some other exception, if the file isn't a TorchScript archive),
        fall back to OnPolicyRunner.load() otherwise. A TorchScript module
        already has the obs normalizer baked in (see play.py's
        export_policy_as_jit(..., normalizer=normalizer, ...) call) so it's
        called directly with the same raw obs tensor env.get_observations()
        returns -- no OnPolicyRunner needed for that path at all.
        """
        try:
            scripted = torch.jit.load(str(resume_path), map_location=str(env.unwrapped.device))
            scripted.eval()
            return (lambda obs: scripted(_flat_policy_obs(obs))), "torchscript"
        except RuntimeError:
            pass  # not a TorchScript archive -- fall through to raw rsl_rl checkpoint

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(resume_path))
        return runner.get_inference_policy(device=env.unwrapped.device), "rsl_rl_checkpoint"

    results = []

    @hydra_task_config(args_cli.task, args_cli.agent)
    def run_all(env_cfg, agent_cfg):
        for idx, checkpoint_str in enumerate(args_cli.checkpoints):
            resume_path = retrieve_file_path(checkpoint_str)
            label = args_cli.labels[idx] if args_cli.labels else _label_for_checkpoint(resume_path)
            print(f"\n=== [{idx + 1}/{len(args_cli.checkpoints)}] {label} ({resume_path}) ===")

            cfg_i = copy.deepcopy(env_cfg)
            cfg_i.scene.num_envs = args_cli.num_envs
            cfg_i.seed = args_cli.seed

            env = gym.make(args_cli.task, cfg=cfg_i, render_mode=None)
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

            policy, source_kind = load_policy(resume_path, env, agent_cfg)
            print(f"  loaded as: {source_kind}")

            num_steps = max(1, round(args_cli.eval_seconds / env.unwrapped.step_dt))
            metrics = collect_metrics(
                env, policy, num_steps, args_cli.num_envs, cfg_i.action_scale,
            )

            row = {
                "label": label,
                "checkpoint": str(resume_path),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "num_envs": args_cli.num_envs,
                "eval_seconds": args_cli.eval_seconds,
                "seed": args_cli.seed,
                **metrics,
            }
            results.append(row)

            print(
                f"  fall_rate={row['fall_rate']:.3f}  mean_ep_len_s={row['mean_episode_length_s']:.2f}  "
                f"mean_tracking_err_deg={row['mean_tracking_err_deg']:.2f}  "
                f"target_violation_rate={row['target_violation_rate']:.4f}  step_over5_rate={row.get('step_over5_rate', float('nan')):.4f}  p99_step_deg={row.get('p99_step_deg', float('nan')):.1f}"
            )
            print(
                f"  p99_left_swing_product={row['p99_left_swing_product']:.3f}  "
                f"p99_right_swing_product={row['p99_right_swing_product']:.3f}  "
                f"(swing_target x clearance, 99th percentile over whole eval -- near 0 means "
                f"real stepping essentially never happened, see the diagnostic table below)"
            )

            env.close()

    run_all()

    if args_cli.out:
        write_csv_rows(args_cli.out, results)
        print(f"\nAppended {len(results)} row(s) to {args_cli.out}")
        print_table(load_csv_rows(args_cli.out))
    else:
        print_table(results)

    _teardown(simulation_app)


def _teardown(simulation_app, timeout: float = 5.0):
    """Same bounded-wait shutdown as tune_stance_lean_isaaclab.py -- results
    are already printed/written to disk before this runs, so a slow/hanging
    Isaac Sim teardown can never lose them."""
    import os
    import threading

    done = threading.Event()

    def _graceful_shutdown():
        try:
            simulation_app.close()
        except Exception as e:  # noqa: BLE001 - best-effort, shutting down regardless
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


if __name__ == "__main__":
    main()

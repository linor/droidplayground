"""
export_policy_for_deployment.py

Run this from your Isaac Lab / rsl_rl training environment (NOT on the robot)
after training. It takes an rsl_rl checkpoint and produces two files you copy
to the robot:

    policy.pt        - TorchScript traced policy (no rsl_rl / isaaclab deps needed to run it)
    policy.meta.json  - sidecar describing exactly what this policy expects/produces

robot_deploy.py refuses to run unless policy.meta.json matches both itself
(obs/action layout it was written for) and the robot config you point it at.
This is the "wrong policy loaded" safety check you asked for.

IMPORTANT: --joint-order must be the order the POLICY actually produces
actions in -- i.e. Isaac Lab's articulation joint order (print
self.robot.joint_names in qmini_leg_env.py's __init__ to get it), NOT
whatever order robot_config.json happens to list joints in for
readability/wiring. robot_deploy.py only checks that the two reference the
same SET of joints, not the same order -- it looks each one up by name, so a
mismatched order here won't be caught by that check; it'll just silently
send actions to the wrong motors. For this robot, Isaac's order interleaves
left/right per joint type (see below), NOT grouped by leg.

USAGE
-----
    python export_policy_for_deployment.py \
        --checkpoint /path/to/model_1500.pt \
        --output-dir ./deploy_bundle \
        --joint-order left_hip_yaw right_hip_yaw left_hip_roll right_hip_roll \
                       left_hip_pitch right_hip_pitch left_knee right_knee \
                       left_ankle right_ankle \
        --action-scale 0.5 \
        --obs-terms joint_pos joint_vel imu_gravity imu_ang_vel phase_sin phase_cos

You will very likely need to adapt `load_policy_from_checkpoint()` below to
however your rsl_rl OnPolicyRunner / ActorCritic is actually constructed --
this is written generically because I don't have your rsl_rl train script,
just the env code. The important part (metadata + TorchScript export +
sha256) will work as-is once the actor module is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


def _build_actor_and_load(checkpoint_path: Path, obs_dim: int, action_dim: int,
                           actor_hidden_dims=(128, 128, 128), activation="elu"):
    """
    Minimal MLP actor loader matching rsl_rl's default ActorCritic actor
    architecture (Linear/activation stack, no CNN, no recurrence). Edit
    actor_hidden_dims/activation to match your train config if different,
    and switch to the recurrent variant if you trained with
    `--rnn`/ActorCriticRecurrent.

    Only used as a FALLBACK when --checkpoint is a raw rsl_rl training
    checkpoint (a dict with a "model_state_dict" key). If your checkpoint is
    already an exported TorchScript policy (common -- rsl_rl's
    `export_policy_as_jit` writes exactly that), this function is skipped
    entirely; see `load_policy()` below.
    """
    act_map = {"elu": torch.nn.ELU, "relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}
    act_cls = act_map[activation]

    # weights_only=False: this is your own training checkpoint, not an
    # untrusted download, so it's fine to unpickle fully here.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # rsl_rl typically prefixes actor weights with "actor." inside the
    # combined ActorCritic state dict -- pull those out and re-key them to
    # match our plain nn.Sequential below (0.weight, 0.bias, 2.weight, ...).
    actor_state = {}
    for k, v in state_dict.items():
        if k.startswith("actor."):
            actor_state[k[len("actor."):]] = v
    if not actor_state:
        raise RuntimeError(
            "Could not find any 'actor.*' keys in the checkpoint's "
            "model_state_dict. Print state_dict.keys() and adjust the "
            "prefix matching / architecture in _build_actor_and_load() "
            "to match how your policy was actually trained/saved."
        )

    # Infer layer sizes from the weights themselves instead of trusting
    # --actor-hidden-dims (a mismatch there used to only print a warning
    # under strict=False). actor_hidden_dims/activation args are kept only
    # for CLI compatibility.
    weight_keys = sorted((k for k in actor_state if k.endswith(".weight")), key=lambda k: int(k.split(".")[0]))
    layers = []
    for i, k in enumerate(weight_keys):
        out_f, in_f = actor_state[k].shape
        layers.append(torch.nn.Linear(in_f, out_f))
        if i < len(weight_keys) - 1:
            layers.append(act_cls())
    if actor_state[weight_keys[0]].shape[1] != obs_dim or actor_state[weight_keys[-1]].shape[0] != action_dim:
        raise RuntimeError(
            f"Checkpoint actor maps {actor_state[weight_keys[0]].shape[1]} -> "
            f"{actor_state[weight_keys[-1]].shape[0]} but --joint-order implies "
            f"{obs_dim} -> {action_dim}."
        )
    actor = torch.nn.Sequential(*layers)
    actor.load_state_dict(actor_state, strict=True)
    actor.eval()

    # OBSERVATION NORMALIZER -- rsl_rl trains with actor_obs_normalization=True
    # (see rsl_rl_qmini_leg_ppo_cfg.py) and keeps its running mean/std in
    # "actor_obs_normalizer.*". The policy only works on (obs - mean) /
    # (std + eps); rsl_rl's own export_policy_as_jit bakes this in, but this
    # raw-checkpoint path used to build the bare actor WITHOUT it, so every
    # bundle exported this way (checked 2026-09-21: deploy_bundle_2026-09-08_
    # sincos and deploy_bundle_2026-09-11_target_limit_fix both match the
    # bare un-normalized actor exactly) expected obs on a completely
    # different scale than it saw in training. eps=0.01 is rsl_rl's
    # EmpiricalNormalization default -- verified numerically against a run's
    # own exported/policy.pt (max diff 0.0 at eps=0.01, ~490 at eps=0).
    mean_key, std_key = "actor_obs_normalizer._mean", "actor_obs_normalizer._std"
    if mean_key in state_dict and std_key in state_dict:
        mean = state_dict[mean_key].reshape(1, -1).clone()
        std = state_dict[std_key].reshape(1, -1).clone()
        return _NormalizedActor(actor, mean, std, eps=NORMALIZER_EPS)
    print("[WARN] checkpoint has no actor_obs_normalizer.* keys -- exporting the bare actor. "
          "If this run used actor_obs_normalization=True that is WRONG.", file=sys.stderr)
    return actor


NORMALIZER_EPS = 1e-2


class _NormalizedActor(torch.nn.Module):
    def __init__(self, actor, mean, std, eps):
        super().__init__()
        self.actor = actor
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.eps = float(eps)

    def forward(self, x):
        return self.actor((x - self.mean) / (self.std + self.eps))


def load_policy(checkpoint_path: Path, obs_dim: int, action_dim: int,
                 actor_hidden_dims, activation):
    """
    Loads the policy regardless of which of the two common forms
    --checkpoint is in:

      1. Already-exported TorchScript (torch.jit.script/trace output) --
         e.g. rsl_rl's own export_policy_as_jit helper, or a previous run of
         this very script. Detected by trying torch.jit.load first.
      2. A raw rsl_rl training checkpoint (dict with "model_state_dict"),
         requiring _build_actor_and_load() to reconstruct the actor and
         load weights into it.

    Returns (scripted_module, source_kind_str).
    """
    try:
        scripted = torch.jit.load(str(checkpoint_path), map_location="cpu")
        scripted.eval()
        return scripted, "torchscript"
    except RuntimeError:
        pass  # not a TorchScript archive -- fall through to raw checkpoint path

    actor = _build_actor_and_load(checkpoint_path, obs_dim, action_dim, actor_hidden_dims, activation)
    example_obs = torch.zeros(1, obs_dim)
    traced = torch.jit.trace(actor, example_obs)
    return traced, "traced_from_state_dict"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--joint-order", nargs="+", required=True,
                         help="Joint names in the exact order the policy's action vector "
                              "expects (Isaac Lab's articulation joint order -- NOT necessarily "
                              "robot_config.json's order, see the module docstring), "
                              "e.g. left_hip_yaw right_hip_yaw left_hip_roll ...")
    parser.add_argument(
        "--obs-terms", nargs="+",
        default=["joint_pos", "joint_vel", "imu_gravity", "imu_ang_vel", "phase_sin", "phase_cos"],
        help="Named obs blocks in order, purely documentary/for the runtime check",
    )
    parser.add_argument("--action-scale", type=float, default=0.15)
    parser.add_argument(
        "--action-smoothing", type=float, default=1.0,
        help="Must match qmini_leg_env.py's cfg.action_smoothing used to train this "
        "checkpoint -- see that cfg's comment (default reverted to 1.0/off on "
        "2026-08-26, training with smoothing on was tried and reverted). Recorded "
        "here so robot_deploy.py can warn on a mismatch the same way it already "
        "does for --action-scale.",
    )
    parser.add_argument(
        "--command-range-scale", type=float, default=1.0,
        help="Walking: the env's command_range_scale during training (e.g. 0.75 for the 1.5x-cadence "
             "stage-4 task). robot_deploy.py maps full stick to this fraction of the library range.",
    )
    parser.add_argument(
        "--gait-library", type=Path, default=None,
        help="Walking policies (DroidPlayground-QMini-Walk): the gait_library_*.npz the policy was "
             "trained with. Adds the 3 command inputs (vx, vy, wz) to the obs and embeds the command "
             "ranges + gait-period table in policy.meta.json -- robot_deploy.py advances the phase "
             "by control_dt / period(command) and clamps the gamepad command to these ranges.",
    )
    parser.add_argument(
        "--residual-actions", action="store_true",
        help="Policy trained with QminiLegEnvCfg.residual_actions (joint target = library reference + "
             "action_scale * action). Requires --gait-library; the library is copied into the bundle "
             "(gait_library.npz) because robot_deploy.py needs the reference too.",
    )
    parser.add_argument("--action-clip", type=float, default=1.0,
                        help="Residual policies: the env's residual_action_clip (actions are clamped to +-this).")
    parser.add_argument("--actor-hidden-dims", nargs="+", type=int, default=[128, 128, 128])
    parser.add_argument("--activation", default="elu", choices=["elu", "relu", "tanh"])
    args = parser.parse_args()

    n_joints = len(args.joint_order)
    # joint_pos (n_joints) + joint_vel (n_joints) + 3 IMU projected-gravity
    # + 3 IMU angular velocity + sin(phase)/cos(phase) -- matches
    # QminiLegEnv._get_observations exactly (28 for the current 10-joint
    # env; NOT n_joints*3, which would assume a per-joint motion_ref term
    # that QminiLegEnv computes but never appends to obs). RE-ADOPTED
    # 2026-09-08, back OUT of the raw motion_time scalar (+1) used since the
    # 2026-09-07 revert -- see observation_space's comment in
    # qmini_leg_env.py for the full story.
    obs_dim = n_joints * 2 + 6 + 2
    action_dim = n_joints
    command_meta = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.gait_library is not None:
        import numpy as np
        lib = np.load(args.gait_library)
        obs_dim += 3
        args.obs_terms = list(args.obs_terms) + ["command_vx", "command_vy", "command_wz"]
        if args.residual_actions:
            import shutil
            shutil.copyfile(args.gait_library, args.output_dir / "gait_library.npz")
        command_meta = {
            "gait_library": str(args.gait_library),
            "library_file": "gait_library.npz" if args.residual_actions else None,
            "library_sha256": hashlib.sha256(Path(args.gait_library).read_bytes()).hexdigest(),
            "units": "vx forward m/s, vy left m/s, wz CCW rad/s (robot units, time scale applied)",
            "time_scale": float(lib["time_scale"]),
            "vx": [float(v) for v in lib["vx"]], "vy": [float(v) for v in lib["vy"]], "wz": [float(v) for v in lib["wz"]],
            "period_s": np.asarray(lib["period_s"], float).round(5).tolist(),   # [vx][vy][wz], trilinear
            "ik_err": np.asarray(lib["ik_err"], float).round(6).tolist() if "ik_err" in lib.files else None,
            # = QminiLegEnvCfg.command_range_scale the policy was trained with:
            # full stick on the robot = this fraction of the library range
            "range_scale": float(args.command_range_scale),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    scripted, source_kind = load_policy(
        args.checkpoint, obs_dim, action_dim,
        tuple(args.actor_hidden_dims), args.activation,
    )
    print(f"Loaded checkpoint as: {source_kind}")

    # Sanity-check the actual forward pass shape now, with the real model,
    # rather than assuming --joint-order/obs_dim math matches what's
    # actually inside the checkpoint.
    with torch.no_grad():
        dummy_obs = torch.zeros(1, obs_dim)
        try:
            out = scripted(dummy_obs)
        except RuntimeError as e:
            raise RuntimeError(
                f"Loaded policy rejected an obs vector of size {obs_dim} "
                f"(derived from --joint-order having {len(args.joint_order)} "
                f"joints x 2 [pos+vel] + 6 IMU [gravity xyz + ang_vel xyz] + "
                f"2 sin/cos phase terms). This almost always means "
                f"--joint-order and/or --obs-terms don't match how this "
                f"policy was actually trained (wrong joint count, or a "
                f"different obs composition than "
                f"joint_pos+joint_vel+imu_gravity+imu_ang_vel+phase_sin+phase_cos). "
                f"Underlying error: {e}"
            ) from e
        if out.shape[-1] != action_dim:
            raise RuntimeError(
                f"Loaded policy's forward pass produced output dim "
                f"{out.shape[-1]}, but --joint-order implies action_dim="
                f"{action_dim}. Your --joint-order / --obs-terms flags "
                f"likely don't match how this policy was actually trained -- "
                f"double check joint count and obs composition."
            )

    policy_out_path = args.output_dir / "policy.pt"
    scripted.save(str(policy_out_path))

    sha256 = hashlib.sha256(policy_out_path.read_bytes()).hexdigest()

    meta = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "joint_order": args.joint_order,
        "obs_terms": args.obs_terms,
        "action_scale": args.action_scale,
        "action_smoothing": args.action_smoothing,
        "source_checkpoint": str(args.checkpoint),
        "obs_normalization_baked_in": source_kind == "traced_from_state_dict",
        "policy_sha256": sha256,
    }
    if command_meta is not None:
        meta["command"] = command_meta
    if args.residual_actions:
        if command_meta is None:
            raise SystemExit("--residual-actions needs --gait-library")
        meta["action_mode"] = "residual"   # target = reference(phase, command) + action_scale * action
        meta["action_clip"] = args.action_clip   # = QminiLegEnvCfg.residual_action_clip
    meta_out_path = args.output_dir / "policy.meta.json"
    meta_out_path.write_text(json.dumps(meta, indent=2))

    print(f"Wrote {policy_out_path}")
    print(f"Wrote {meta_out_path}")
    print(f"policy_sha256 = {sha256}")
    print("\nCopy both files to the robot and point robot_deploy.py's")
    print("--policy / --policy-meta at them.")


if __name__ == "__main__":
    main()

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
        --obs-terms joint_pos joint_vel imu_gravity imu_ang_vel motion_time

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

    layers = []
    in_dim = obs_dim
    for h in actor_hidden_dims:
        layers.append(torch.nn.Linear(in_dim, h))
        layers.append(act_cls())
        in_dim = h
    layers.append(torch.nn.Linear(in_dim, action_dim))
    actor = torch.nn.Sequential(*layers)

    # weights_only=False: this is your own training checkpoint, not an
    # untrusted download, so it's fine to unpickle fully here.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # rsl_rl typically prefixes actor weights with "actor." inside the
    # combined ActorCritic state dict -- pull those out and re-key them to
    # match our plain nn.Sequential above (0.weight, 0.bias, 2.weight, ...).
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

    missing, unexpected = actor.load_state_dict(actor_state, strict=False)
    if missing or unexpected:
        print(f"[WARN] state_dict mismatch. missing={missing} unexpected={unexpected}", file=sys.stderr)
        print("[WARN] Double check actor_hidden_dims/activation match training config.", file=sys.stderr)

    actor.eval()
    return actor


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
        default=["joint_pos", "joint_vel", "imu_gravity", "imu_ang_vel", "motion_time"],
        help="Named obs blocks in order, purely documentary/for the runtime check",
    )
    parser.add_argument("--action-scale", type=float, default=0.15)
    parser.add_argument("--actor-hidden-dims", nargs="+", type=int, default=[128, 128, 128])
    parser.add_argument("--activation", default="elu", choices=["elu", "relu", "tanh"])
    args = parser.parse_args()

    n_joints = len(args.joint_order)
    # joint_pos (n_joints) + joint_vel (n_joints) + 3 IMU projected-gravity
    # + 3 IMU angular velocity + a single motion_time scalar -- matches
    # QminiLegEnv._get_observations exactly (27 for the current 10-joint
    # env; NOT n_joints*3, which would assume a per-joint motion_ref term
    # that QminiLegEnv computes but never appends to obs).
    obs_dim = n_joints * 2 + 6 + 1
    action_dim = n_joints

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
                f"1 motion_time scalar). This almost always means "
                f"--joint-order and/or --obs-terms don't match how this "
                f"policy was actually trained (wrong joint count, or a "
                f"different obs composition than "
                f"joint_pos+joint_vel+imu_gravity+imu_ang_vel+motion_time). "
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
        "source_checkpoint": str(args.checkpoint),
        "policy_sha256": sha256,
    }
    meta_out_path = args.output_dir / "policy.meta.json"
    meta_out_path.write_text(json.dumps(meta, indent=2))

    print(f"Wrote {policy_out_path}")
    print(f"Wrote {meta_out_path}")
    print(f"policy_sha256 = {sha256}")
    print("\nCopy both files to the robot and point robot_deploy.py's")
    print("--policy / --policy-meta at them.")


if __name__ == "__main__":
    main()

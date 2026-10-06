#!/usr/bin/env python
"""
gen_reference_heights.py

Computes the reference gait's OWN foot-height curve (heel and toe, both
legs) via forward kinematics against qmini_urdf-2legs.usda, and writes it
into a keyframes JSON as `reference_foot_heights_m` -- 4 columns per frame
([left_heel, left_toe, right_heel, right_toe], meters, each point's height
above ITS OWN cycle minimum), same `times` as the file's `keyframes`.

WHY THIS EXISTS
----------------
qmini_leg_env.py used to compare the trained policy's actual foot height
against a single hand-picked constant (foot_swing_target_height_m). That
required manually re-deriving the "right" constant for every reference
clip (see foot_height_target_floor_m's comment in qmini_leg_env.py for the
full history: three attempts to fix excess foot-lift by escalating
foot_overswing_penalty_weight instead all destabilized training, because
the real problem was comparing against the wrong kind of target, not an
undersized penalty). This script instead extracts the target directly from
the SAME reference motion the policy already tracks, so a new gait (e.g. a
future walking clip with a taller natural lift) gets a correct target for
free -- no manual re-tuning, no re-deriving a magic number by hand.

METHOD
------
Builds the actual joint tree from the USD's UsdPhysics joint prims (not
hand-derived/mirrored -- the right leg's joints are traversed exactly like
the left's, reading their own body0/body1/axis directly, since at least
one joint in this asset (the ankle) has body0/body1 in the OPPOSITE order
from every other joint and blindly mirroring the left leg's numbers would
silently reproduce that bug). Fixed joints with zero offset/identity
rotation are treated as one rigid cluster (confirmed true throughout this
asset). Revolute joints are driven by rotation-by-angle about their local
axis. base_link is held at identity -- this is relative leg geometry only,
independent of any whole-body lean.

Validated against the known-good standing pose (qmini.py's
InitialStateCfg.joint_pos, itself verified via tune_stance_lean_isaaclab.py)
for BOTH legs: expect heel_z close to -0.42m for each.

USAGE
-----
    /home/jeroen/anaconda3/envs/usdtesting/bin/python gen_reference_heights.py \\
        [path/to/keyframes.json]

Defaults to the droidplayground package's
keyframes_step_in_place_all_joints_2x_base_offset.json. Overwrites that
file in place, adding/replacing only the `reference_foot_heights_m` and
`_reference_foot_heights_comment` fields -- everything else is untouched.
Needs the `usdtesting` conda env (has real `pxr` USD bindings; the
`env_isaaclab` env does not import `isaaclab` outside a full Isaac Sim
launch, and this script needs no physics/GPU at all, just USD FK).

Re-run this any time keyframes_step_in_place_all_joints_2x_base_offset.json's
`keyframes` or `_stable_base_pose_offset_deg` change, or when pointing it at
a NEW reference clip (e.g. a future walking gait) -- otherwise
reference_foot_heights_m goes stale and silently drives training against
the wrong target.
"""
import json
import math
import sys
from pathlib import Path

from pxr import Usd, UsdPhysics, Gf, UsdGeom
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
USD_PATH = REPO_ROOT / "rl/source/droidplayground/data/usd/qmini_urdf-2legs.usda"
DEFAULT_KEYFRAMES_PATH = (
    REPO_ROOT
    / "rl/source/droidplayground/droidplayground/tasks/direct/droidplayground"
    / "keyframes_step_in_place_all_joints_2x_base_offset.json"
)

BASE_LINK_PATH = "/qmini_urdf_2legs/base_link"
LEFT_FOOT_PATH = "/qmini_urdf_2legs/Left_Foot_1"
RIGHT_FOOT_PATH = "/qmini_urdf_2legs/Riggt_Foot_1"  # typo in the asset itself, not ours

# cfg.left/right_heel_local_m and left/right_toe_local_m in qmini_leg_env.py
LEFT_HEEL = Gf.Vec3d(0.16286, 0.16930, -0.42102)
LEFT_TOE = Gf.Vec3d(0.14439, 0.04049, -0.42415)
RIGHT_HEEL = Gf.Vec3d(-0.12667, 0.17208, -0.41420)
RIGHT_TOE = Gf.Vec3d(-0.16138, 0.03790, -0.41897)

CHAIN_ORDER = ["yaw", "roll", "pitch", "knee", "ankle"]


def collect_all_joints(stage):
    joints = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(prim)
            body0_rel = j.GetBody0Rel().GetTargets()
            body1_rel = j.GetBody1Rel().GetTargets()
            entry = {
                "name": prim.GetName(),
                "is_revolute": prim.IsA(UsdPhysics.RevoluteJoint),
                "body0": str(body0_rel[0]) if body0_rel else None,
                "body1": str(body1_rel[0]) if body1_rel else None,
                "localPos0": Gf.Vec3f(j.GetLocalPos0Attr().Get()),
                "localPos1": Gf.Vec3f(j.GetLocalPos1Attr().Get()),
                "localRot0": j.GetLocalRot0Attr().Get(),
                "localRot1": j.GetLocalRot1Attr().Get(),
            }
            if entry["is_revolute"]:
                entry["axis"] = UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()
            joints[entry["name"]] = entry
    return joints


def build_fixed_alias_map(joints):
    """Union-find over bodies connected by zero-offset identity fixed joints
    -- these are the same rigid cluster split across multiple prim names by
    the URDF->USD import (every fixed joint in this asset has localPos=
    (0,0,0) and identity rotation on both sides)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for entry in joints.values():
        if not entry["is_revolute"]:
            zero = all(abs(v) < 1e-6 for v in entry["localPos0"]) and all(
                abs(v) < 1e-6 for v in entry["localPos1"]
            )
            if zero and entry["body0"] and entry["body1"]:
                union(entry["body0"], entry["body1"])
    return find


def quat_to_matrix(q):
    return Gf.Matrix4d().SetRotate(Gf.Rotation(q))


def joint_frame(entry, side, units_to_meters):
    rot = entry[f"localRot{side}"]
    pos = entry[f"localPos{side}"]
    m = quat_to_matrix(rot)
    m.SetTranslateOnly(Gf.Vec3d(pos) * units_to_meters)
    return m


def axis_vector(axis_token):
    return {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}[str(axis_token)]


def compute_foot_transform(joints, alias, units_to_meters, revolute_names, foot_path, angles_deg):
    """angles_deg: dict {yaw,roll,pitch,knee,ankle} -> degrees, for ONE leg's
    OWN joints (read directly from the joint_order columns already matching
    that leg -- never mirrored/negated from the other leg's numbers).

    Returns a Gf.Matrix4d T (row-vector convention) such that
    T.Transform(p) maps a point given in base_link coordinates AT THE ZERO
    POSE (e.g. LEFT_HEEL) to its base_link position at `angles_deg`.

    REWRITTEN 2026-10-06. The previous version composed Gf row-vector
    matrices in column-vector order and did NOT match Isaac Sim at
    non-zero poses (standing pose: foot rotation off by 14 deg, position by
    1-3 cm), which made reference_foot_heights_m wrong -- smeared across
    the cycle, each foot ">1 cm up" ~80% of the time instead of ~30%. It
    only passed the heel-height assertion below because that checks one
    coordinate at one pose. This version uses the pivot-axis model from
    export_fk_geometry.py, validated against Isaac Sim body poses at three
    poses (<0.01 mm, <0.1 deg): every body frame in this asset coincides
    with base_link at the zero pose, so each revolute joint is a rotation
    about its pivot (localPos0) and axis (localRot0 applied to the axis
    token), sign +1 when body0 is the parent side of the chain, -1 when
    it's the child side. `foot_path` is kept for interface compatibility;
    with coincident frames no fixed-joint tail is needed."""
    R = np.eye(3)
    t = np.zeros(3)
    current_body = alias(BASE_LINK_PATH)

    for key in CHAIN_ORDER:
        entry = joints[revolute_names[key]]
        b0, b1 = alias(entry["body0"]), alias(entry["body1"])
        if b0 == current_body:
            sign, current_body = 1.0, b1
        elif b1 == current_body:
            sign, current_body = -1.0, b0
        else:
            raise RuntimeError(f"Chain broken at {key}: current_body={current_body}, joint bodies=({b0},{b1})")

        axis = np.array(Gf.Rotation(entry["localRot0"]).TransformDir(axis_vector(entry["axis"])))
        axis = axis / np.linalg.norm(axis)
        pivot = np.array(Gf.Vec3d(entry["localPos0"])) * units_to_meters
        theta = math.radians(sign * angles_deg[key])
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rk = np.eye(3) + math.sin(theta) * k + (1 - math.cos(theta)) * k @ k
        # child = parent * (rotation about pivot): p -> rk (p - pivot) + pivot
        t = t + R @ (pivot - rk @ pivot)
        R = R @ rk

    # Row-vector Gf matrix: T.Transform(p) = p * T = R p + t
    # (Built in one go: Gf.Matrix4d's T[i][j] = x writes to a temporary row copy.)
    return Gf.Matrix4d(
        float(R[0][0]), float(R[1][0]), float(R[2][0]), 0.0,
        float(R[0][1]), float(R[1][1]), float(R[2][1]), 0.0,
        float(R[0][2]), float(R[1][2]), float(R[2][2]), 0.0,
        float(t[0]), float(t[1]), float(t[2]), 1.0,
    )


def build_chain(joints, leg):
    rev = {k: f"Revolute_{leg}_{k}" for k in CHAIN_ORDER}
    for name in rev.values():
        if name not in joints:
            raise KeyError(f"missing joint {name} -- USD asset changed?")
    return rev


def main():
    keyframes_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_KEYFRAMES_PATH

    stage = Usd.Stage.Open(str(USD_PATH))
    units_to_meters = UsdGeom.GetStageMetersPerUnit(stage)
    joints = collect_all_joints(stage)
    alias = build_fixed_alias_map(joints)

    left_chain = build_chain(joints, "left")
    right_chain = build_chain(joints, "right")

    def transform(chain, foot_path, angles_deg):
        return compute_foot_transform(joints, alias, units_to_meters, chain, foot_path, angles_deg)

    # Validate against the known-good standing pose (qmini.py's
    # InitialStateCfg.joint_pos) for BOTH legs -- built independently via
    # the real joint graph, not mirrored, so this also catches an asset
    # sign/order quirk on the right leg if one exists.
    left_stance = dict(yaw=0, roll=0, pitch=-16.5656, knee=22.5569, ankle=12.9913)
    right_stance = dict(yaw=0, roll=0, pitch=16.5656, knee=-22.5569, ankle=-12.9913)
    for name, chain, foot, heel, stance in [
        ("left", left_chain, LEFT_FOOT_PATH, LEFT_HEEL, left_stance),
        ("right", right_chain, RIGHT_FOOT_PATH, RIGHT_HEEL, right_stance),
    ]:
        h = transform(chain, foot, stance).Transform(heel)
        # ~-38 cm, matching Isaac Sim (2026-10-06). The old "known-good"
        # -42 cm is the straight-leg ZERO pose heel height -- the old,
        # wrong FK happened to reproduce it at this bent-knee pose.
        print(f"{name} stance heel_z = {h[2]*100:.2f}cm (want close to -38cm)")
        if not (-0.40 < h[2] < -0.36):
            raise AssertionError(
                f"{name} leg's stance heel height {h[2]:.3f}m is not close to the "
                f"Isaac Sim-validated ~-0.38m -- FK chain or sign is likely wrong, do not "
                f"trust the generated heights. Fix before writing the JSON."
            )

    with open(keyframes_path) as f:
        data = json.load(f)
    order = data["joint_order"]
    idx = {n: i for i, n in enumerate(order)}
    kf = data["keyframes"]

    per_leg = {}
    for legname, chain, foot, heel, toe in [
        ("left", left_chain, LEFT_FOOT_PATH, LEFT_HEEL, LEFT_TOE),
        ("right", right_chain, RIGHT_FOOT_PATH, RIGHT_HEEL, RIGHT_TOE),
    ]:
        heel_z, toe_z = [], []
        for _, frame in kf:
            angles = {k: frame[idx[f"{legname}_{k}"]] for k in CHAIN_ORDER}
            T = transform(chain, foot, angles)
            heel_z.append(T.Transform(heel)[2])
            toe_z.append(T.Transform(toe)[2])
        per_leg[legname] = (np.array(heel_z), np.array(toe_z))

    n_frames = len(kf)
    n_unique = n_frames - 1  # last frame duplicates the first (closes the loop)
    rows = []
    for i in range(n_frames):
        row = []
        for legname in ("left", "right"):
            heel_z, toe_z = per_leg[legname]
            base_heel = heel_z[:n_unique].min()
            base_toe = toe_z[:n_unique].min()
            ii = i if i < n_unique else 0
            row.append(float(heel_z[ii] - base_heel))
            row.append(float(toe_z[ii] - base_toe))
        rows.append(row)  # [left_heel_m, left_toe_m, right_heel_m, right_toe_m]

    arr = np.array(rows)
    for j, label in enumerate(["left_heel", "left_toe", "right_heel", "right_toe"]):
        print(f"{label}: {arr[:, j].min()*100:.2f}cm .. {arr[:, j].max()*100:.2f}cm")

    data["_reference_foot_heights_comment"] = (
        "Per-frame reference foot height ABOVE THIS FOOT'S OWN CYCLE "
        "MINIMUM, in meters, columns [left_heel, left_toe, right_heel, "
        "right_toe] -- computed via forward kinematics from THIS file's "
        "own keyframes (including the baked-in stance offset, i.e. what "
        "the trained policy actually sees), same times as `keyframes`. "
        "Generated by experiments/qmini-leg/gen_reference_heights.py "
        "against qmini_urdf-2legs.usda, validated there against the known "
        "standing-pose heel height (~-0.42m) for both legs independently. "
        "Replaces the old constant foot_swing_target_height_m as the "
        "swing-reward/overswing-penalty target in qmini_leg_env.py (see "
        "foot_height_target_floor_m's comment there for the full "
        "history), so a new reference clip (e.g. a future walking gait) "
        "gets a correct target automatically -- re-run that script any "
        "time `keyframes` or `_stable_base_pose_offset_deg` here change, "
        "or when generating this field for a different clip file."
    )
    data["reference_foot_heights_m"] = rows

    with open(keyframes_path, "w") as f:
        json.dump(data, f, indent=2)
    print("wrote", keyframes_path)


if __name__ == "__main__":
    main()

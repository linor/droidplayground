#!/usr/bin/env python
"""
check_reference_balance_fk.py

A STATIC, kinematics-only check of a reference gait's balance demand --
built after check_reference_stability.py's open-loop dynamic check turned
out uninformative (see that script's 2026-09-28 findings): tracking ANY
dynamic single-support gait with pure PD control and zero active balance
correction falls over almost immediately regardless of reference quality,
because single-support stepping is not statically stable in the first
place -- that's what the RL policy's momentum/correction is FOR. So a
full "is the CoM inside the support polygon at every instant" check would
ALSO show near-universal "failure" during every single-support phase, for
the same structural reason, and wouldn't discriminate between candidates
either.

What CAN be checked without physics or training: how far each foot sits,
horizontally, from the body's own center (base_link origin) at each
keyframe -- via the SAME forward-kinematics chain gen_reference_heights.py
already built and validated (against the known standing-pose heel height,
~-0.42m) for the foot-height reference. A reference that asks the stance
leg to reach further fore/aft from center demands more active correction
to avoid tipping than one that keeps the stance foot closer to under the
body -- not a guarantee of trainability, but a real, comparable, honest
signal, and a much cheaper one than a training run.

Reuses gen_reference_heights.py's joint-tree/FK code directly (import, not
copy) so any future fix to that chain (sign conventions, asset path) only
needs to happen once.

USAGE
-----
    /home/jeroen/anaconda3/envs/usdtesting/bin/python check_reference_balance_fk.py \\
        keyframes_a.json keyframes_b.json ...

Prints, per file: fore-aft (Y) and lateral (X) foot offset from base_link
center at keyframe 0 (the "home"/startup pose robot_deploy.py moves to
before starting the policy) for both feet, and the SWING-side foot's
peak fore-aft excursion across the whole cycle (a proxy for how far the
gait pushes the moving leg from center, independent of the stance-leg
question). Needs the `usdtesting` conda env (real pxr bindings).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import gen_reference_heights as gh  # reuse the validated FK chain


def foot_xyz(chain, foot_path, heel, toe, angles_deg):
    T = gh.compute_foot_transform(gh.JOINTS, gh.ALIAS, gh.UNITS, chain, foot_path, angles_deg)
    h = T.Transform(heel)
    t = T.Transform(toe)
    mid = [(h[i] + t[i]) / 2.0 for i in range(3)]
    return mid


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    stage = gh.Usd.Stage.Open(str(gh.USD_PATH))
    gh.JOINTS = gh.collect_all_joints(stage)
    gh.ALIAS = gh.build_fixed_alias_map(gh.JOINTS)
    gh.UNITS = gh.UsdGeom.GetStageMetersPerUnit(stage)
    left_chain = gh.build_chain(gh.JOINTS, "left")
    right_chain = gh.build_chain(gh.JOINTS, "right")

    print(f"{'file':<55s} {'kf0 L(fore,lat)cm':>20s} {'kf0 R(fore,lat)cm':>20s} {'peakL_foreaft':>14s} {'peakR_foreaft':>14s}")
    for path in sys.argv[1:]:
        data = json.load(open(path))
        order = data["joint_order"]
        idx = {n: i for i, n in enumerate(order)}
        kf = data["keyframes"]
        n_frames = len(kf)
        n_unique = n_frames - 1

        left_xyz, right_xyz = [], []
        for _, frame in kf:
            langles = {k: frame[idx[f"left_{k}"]] for k in gh.CHAIN_ORDER}
            rangles = {k: frame[idx[f"right_{k}"]] for k in gh.CHAIN_ORDER}
            left_xyz.append(foot_xyz(left_chain, gh.LEFT_FOOT_PATH, gh.LEFT_HEEL, gh.LEFT_TOE, langles))
            right_xyz.append(foot_xyz(right_chain, gh.RIGHT_FOOT_PATH, gh.RIGHT_HEEL, gh.RIGHT_TOE, rangles))
        left_xyz = np.array(left_xyz)[:n_unique]
        right_xyz = np.array(right_xyz)[:n_unique]

        # Convention (robot_config's _imu_comment): base_link +X points
        # toward the LEFT leg (lateral), +Y toward the REAR (so -Y is
        # forward/fore, +Y is aft/backward) -- fore-aft excursion reported
        # as -Y so "forward of center" reads positive, matching intuition.
        l0_fore, l0_lat = -left_xyz[0, 1] * 100, left_xyz[0, 0] * 100
        r0_fore, r0_lat = -right_xyz[0, 1] * 100, right_xyz[0, 0] * 100
        l_peak = np.abs(left_xyz[:, 1] - left_xyz[:, 1].mean()).max() * 100
        r_peak = np.abs(right_xyz[:, 1] - right_xyz[:, 1].mean()).max() * 100

        print(
            f"{Path(path).name:<55s} "
            f"{f'({l0_fore:+.2f},{l0_lat:+.2f})':>20s} "
            f"{f'({r0_fore:+.2f},{r0_lat:+.2f})':>20s} "
            f"{l_peak:14.2f} {r_peak:14.2f}"
        )


if __name__ == "__main__":
    main()

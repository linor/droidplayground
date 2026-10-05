#!/usr/bin/env python
"""
export_fk_geometry.py

Exports the qmini leg kinematics as joint pivots + axes, for log_viewer.html
(which embeds the JSON this writes) or any FK without pxr.

Model: in this asset every body frame coincides with base_link at the zero
pose (Onshape-style export), so each revolute joint is fully described by a
pivot point and an axis in base coordinates (localPos0/localRot0; body0's
frame is the base frame at zero). FK, column-vector convention:
    T_child = T_parent * Rot(about pivot, axis, sign * q)
with sign = +1 when body0 is the parent side of the chain, -1 when body0 is
the child side. A point given in base coordinates at the zero pose (e.g.
the heel) moves with its body: p(q) = T_foot(q) * p0.

Validated 2026-10-04 against Isaac Sim body poses (base-relative) at three
poses incl. yaw/roll: <0.01 mm, <0.1 deg on thigh/calf/foot.

NOTE: gen_reference_heights.compute_foot_transform composes Gf row-vector
matrices in column order and does NOT match the sim at non-zero poses
(standing pose foot rotation off by 14 deg, position 1-2 cm) -- don't reuse
it for new work.

USAGE
-----
    /home/jeroen/anaconda3/envs/usdtesting/bin/python export_fk_geometry.py [out.json]
"""
import json
import sys
from pathlib import Path

from pxr import Gf, Usd, UsdGeom

sys.path.insert(0, str(Path(__file__).parent))
import gen_reference_heights as gh  # USD path, joint collection, heel/toe points


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "fk_geometry.json"
    stage = Usd.Stage.Open(str(gh.USD_PATH))
    units = UsdGeom.GetStageMetersPerUnit(stage)
    joints = gh.collect_all_joints(stage)
    alias = gh.build_fixed_alias_map(joints)
    base = alias(gh.BASE_LINK_PATH)

    data = {"model": "T_child = T_parent * Rot(pivot, axis, sign*q); base frame +X left, +Y rear, +Z up",
            "legs": {}}
    feet = {"left": (gh.LEFT_HEEL, gh.LEFT_TOE), "right": (gh.RIGHT_HEEL, gh.RIGHT_TOE)}
    for leg, (heel, toe) in feet.items():
        current = base
        chain = []
        for key in gh.CHAIN_ORDER:
            e = joints[f"Revolute_{leg}_{key}"]
            b0, b1 = alias(e["body0"]), alias(e["body1"])
            if b0 == current:
                sign, current = 1.0, b1
            elif b1 == current:
                sign, current = -1.0, b0
            else:
                raise RuntimeError(f"chain broken at {leg} {key}")
            axis = Gf.Rotation(e["localRot0"]).TransformDir(gh.axis_vector(e["axis"]))
            pivot = Gf.Vec3d(e["localPos0"]) * units
            chain.append({"joint": key, "pivot": [round(v, 6) for v in pivot],
                          "axis": [round(v, 6) for v in axis], "sign": sign})
        data["legs"][leg] = {"chain": chain, "heel": list(heel), "toe": list(toe)}
    out.write_text(json.dumps(data, indent=1))
    print(f"wrote {out}")
    print(json.dumps(data))


if __name__ == "__main__":
    main()

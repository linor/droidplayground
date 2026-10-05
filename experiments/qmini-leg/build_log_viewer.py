#!/usr/bin/env python3
"""
build_log_viewer.py

Builds log_viewer.html from log_viewer.template.html: embeds fk_geometry.json
(from export_fk_geometry.py) and one example control_loop CSV, so the page
opens with data. Any other control_loop_*.csv can be dropped onto the page.

USAGE
-----
    python3 build_log_viewer.py [example.csv] [--label "what this log is"]
"""
import argparse
import csv
import io
import json
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parents[1]
JOINTS = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=str(REPO / "control_loop_20261004_164229.csv"))
    ap.add_argument("--label", default="p1 standing hold, pushed by hand")
    ap.add_argument("--fragment", default=None, help="also write a skeleton-free copy here")
    args = ap.parse_args()

    keep = ["t_wall", "motion_time", "obs_imu_gravity_x", "obs_imu_gravity_y", "obs_imu_gravity_z"]
    for leg in ("left", "right"):
        for j in JOINTS:
            keep += [f"target_deg_{leg}_{j}", f"actual_deg_{leg}_{j}"]
    rows = [r for r in csv.DictReader(open(args.csv)) if r.get("actual_deg_left_knee")]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(keep)
    for r in rows:
        w.writerow([f"{float(r[k]):.4f}" if k != "t_wall" else r[k] for k in keep])

    geom = json.loads((HERE / "fk_geometry.json").read_text())
    html = (HERE / "log_viewer.template.html").read_text()
    html = (html.replace("__GEOM__", json.dumps(geom))
                .replace("__SAMPLE_NAME__", f"{Path(args.csv).name} (example: {args.label})")
                .replace("__SAMPLE_CSV__", buf.getvalue()))
    # Local copy: a complete document, opens straight from disk (file://).
    out = HERE / "log_viewer.html"
    out.write_text('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
                   '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
                   '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>\n'
                   + html + "\n</body></html>\n")
    print(f"wrote {out} ({len(html) // 1024} KB, {len(rows)} rows embedded)")
    if args.fragment:
        # Body-only copy for hosts that add their own document skeleton.
        Path(args.fragment).write_text(html)
        print(f"wrote {args.fragment}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path


def project_point(K, R, t, pt):
    x, y, z = pt
    xc = R[0][0] * x + R[0][1] * y + R[0][2] * z + t[0]
    yc = R[1][0] * x + R[1][1] * y + R[1][2] * z + t[1]
    zc = R[2][0] * x + R[2][1] * y + R[2][2] * z + t[2]
    if zc <= 0.0:
      raise RuntimeError("Point projects behind camera")
    u = K[0][0] * (xc / zc) + K[0][2]
    v = K[1][1] * (yc / zc) + K[1][2]
    return [u, v]


def make_input_json(path):
    fx_gt = fy_gt = 800.0
    cx = 640.0
    cy = 360.0
    baseline_gt = 0.2
    left_gt = {
        "fx": fx_gt, "fy": fy_gt, "cx": cx, "cy": cy,
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0,
    }
    right_gt = dict(left_gt)
    extrinsics_gt = {
        "R": [1.0, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, 1.0],
        "t": [-baseline_gt, 0.0, 0.0],
    }

    left_init = {
        "fx": 760.0, "fy": 840.0, "cx": cx, "cy": cy,
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0,
    }
    right_init = dict(left_init)
    extrinsics_init = {
        "R": [1.0, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, 1.0],
        "t": [-0.18, 0.0, 0.0],
    }

    K_gt = [[fx_gt, 0.0, cx], [0.0, fy_gt, cy], [0.0, 0.0, 1.0]]
    R_gt = [[1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]
    t_gt = [-baseline_gt, 0.0, 0.0]

    points = [
        [0.05, -0.02, 4.0],
        [0.10, 0.03, 4.5],
        [-0.08, 0.01, 5.0],
        [0.02, 0.08, 6.0],
        [-0.04, -0.05, 7.0],
        [0.12, -0.06, 8.0],
        [-0.11, 0.04, 6.5],
        [0.07, 0.02, 5.5],
        [0.15, -0.01, 9.0],
        [-0.14, 0.06, 7.5],
    ]

    matches = []
    for pt in points:
        left_pt = project_point(K_gt, R_gt, [0.0, 0.0, 0.0], pt)
        right_pt = project_point(K_gt, R_gt, t_gt, pt)
        matches.append({"left": left_pt, "right": right_pt})

    input_json = {
        "left": left_init,
        "right": right_init,
        "extrinsics": extrinsics_init,
        "pairs": [
            {
                "name": "synthetic_pair_0",
                "matches": matches,
            }
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(input_json, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        input_path = td / "input.json"
        output_path = td / "output.json"
        make_input_json(input_path)

        cmd = [
            args.binary,
            "--input", str(input_path),
            "--output", str(output_path),
            "--max_iter", "20",
            "--max_reproj_error", "10.0",
        ]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        print(proc.stdout)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

        if not output_path.exists():
            raise AssertionError("run_stereo_calib did not write output JSON")

        with open(output_path, "r", encoding="utf-8") as f:
            out = json.load(f)

        assert "success" in out, "missing success field"
        assert "init_reproj_error" in out, "missing init_reproj_error"
        assert "final_reproj_error" in out, "missing final_reproj_error"
        assert out["success"] is True, "optimizer reported failure"
        assert out["final_reproj_error"] <= out["init_reproj_error"] + 1e-6, (
            "expected reprojection error to improve"
        )
        assert math.isfinite(out["final_reproj_error"]), "final_reproj_error is not finite"

        print("smoke test passed")


if __name__ == "__main__":
    main()

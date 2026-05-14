"""KITTI calibration helpers used by the pipeline.

KITTI RAW 的官方标定文件是文本格式，BA 程序使用的是项目自己的
camera JSON 格式。本模块只负责这层格式转换，不参与特征匹配或优化。
"""

from __future__ import annotations

import json
from pathlib import Path


def parse_kitti_calib_file(calib_file: Path) -> dict[str, list[float] | str]:
    """读取 KITTI `calib_cam_to_cam.txt` 风格的 key-value 标定文件。

    大多数 KITTI 标定字段都是浮点数组；少数非数值字段保留为字符串，
    这样后续如果需要更多字段，不必重新写解析器。
    """
    calib = {}
    with open(calib_file, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not value:
                calib[key] = []
                continue
            parts = value.split()
            try:
                calib[key] = [float(v) for v in parts]
            except ValueError:
                calib[key] = value
    return calib


def projection_to_intrinsics(P: list[float]) -> dict:
    """从 KITTI 3x4 投影矩阵中提取项目相机模型需要的内参字段。

    KITTI rectified 相机通常可以视为无畸变；这里显式把畸变项置零，
    让输出 JSON 和项目的 `StereoCamera` schema 保持一致。
    """
    return {
        "fx": P[0],
        "fy": P[5],
        "cx": P[2],
        "cy": P[6],
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "k3": 0.0,
    }


def build_kitti_gt_camera_json(kitti_calib_dir: Path, result_dir: Path, result_prefix: str) -> Path:
    """把 KITTI 官方标定转换成本项目用于 GT 对比的 camera JSON。

    目前只使用 rectified camera 00/01 的投影矩阵。左相机作为参考系，
    因此旋转为单位阵；平移由两个投影矩阵的最后一列反推得到。
    """
    calib_file = kitti_calib_dir / "calib_cam_to_cam.txt"
    if not calib_file.exists():
        raise FileNotFoundError(f"KITTI 标定文件不存在：{calib_file}")

    calib = parse_kitti_calib_file(calib_file)
    required_keys = ["P_rect_00", "P_rect_01"]
    missing_keys = [key for key in required_keys if key not in calib or len(calib[key]) != 12]
    if missing_keys:
        raise ValueError(f"KITTI 标定文件缺少必要字段：{', '.join(missing_keys)} ({calib_file})")

    p0 = calib["P_rect_00"]
    p1 = calib["P_rect_01"]

    fx0 = p0[0]
    fy0 = p0[5]
    fx1 = p1[0]
    fy1 = p1[5]

    # KITTI 的 P 矩阵形如 K [R|t]。在 rectified stereo 中 R 近似一致，
    # 因而可由 P[3]/fx 等项恢复相对平移。
    gt_camera = {
        "left": projection_to_intrinsics(p0),
        "right": projection_to_intrinsics(p1),
        "extrinsics": {
            "R": [
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            "t": [
                p1[3] / fx1 - p0[3] / fx0,
                p1[7] / fy1 - p0[7] / fy0,
                p1[11] - p0[11],
            ],
        },
    }

    gt_dir = result_dir / "gt_params"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_file = gt_dir / f"{result_prefix}_gt_camera.json"
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(gt_camera, f, indent=2)
    return gt_file

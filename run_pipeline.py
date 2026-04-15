#!/usr/bin/env python3
"""
通用立体标定全流程脚本

从项目根目录运行（确保已激活 stereo-calib-vis conda 环境并安装了 torch/opencv）：

    python run_pipeline.py [选项]

完整三步流程：
  1. SuperPoint 特征提取与匹配（所有图像对）
  2. C++ 增量式 Bundle Adjustment 优化
  3. BA 优化历史可视化

常用示例：
    # 完整流程（默认场景 panoramic_01）
    python run_pipeline.py

    # 指定其他场景
    python run_pipeline.py --scene panoramic_02

    # 使用 CUDA 加速特征匹配
    python run_pipeline.py --scene panoramic_01 --cuda

    # 跳过已完成的匹配步骤，直接跑 BA
    python run_pipeline.py --scene panoramic_01 --skip_match
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_kitti_calib_file(calib_file: Path) -> dict[str, list[float] | str]:
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

    gt_camera = {
        "left": projection_to_intrinsics(p0),
        "right": projection_to_intrinsics(p1),
        "extrinsics": {
            "R": [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0],
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


def resolve_paths(scene: str, pixel_tag: str, dataset_mode: str, kitti_root_dir: str | None,
                  match_json: str | None, result_prefix: str | None):
    """根据数据集模式、场景名和像素阈值标签生成相关路径。"""
    result_dir = PROJECT_ROOT / "stereo_calib/result"

    if dataset_mode == "project":
        result_prefix = scene
        img_dir = PROJECT_ROOT / f"blender-file/stereo_{scene}"
        gt_file = img_dir / "camera_params_0000.json"
        matcher_input_dir = img_dir
        match_json_path = result_dir / "match_points" / f"{result_prefix}_matches_gtpose_dsym_le_{pixel_tag}.json"
        ba_result_json = result_dir / "ba_results" / f"{result_prefix}_ba_result_le_{pixel_tag}.json"
        plot_png = result_dir / f"{result_prefix}_ba_history_gtpose_dsym_le_{pixel_tag}.png"
    elif dataset_mode == "kitti2015":
        result_prefix = f"{dataset_mode}_{scene}"
        if not kitti_root_dir:
            raise ValueError("KITTI 模式需要提供 --kitti_root_dir")
        img_dir = Path(kitti_root_dir)
        kitti_calib_dir = img_dir.parent.parent.parent / "2011_09_26_calib" / "2011_09_26"
        gt_file = build_kitti_gt_camera_json(kitti_calib_dir, result_dir, result_prefix)
        matcher_input_dir = img_dir
        match_json_path = result_dir / "match_points" / f"{result_prefix}_matches.json"
        ba_result_json = result_dir / "ba_results" / f"{result_prefix}_ba_result.json"
        plot_png = result_dir / f"{result_prefix}_ba_history.png"
    elif dataset_mode == "kitti_raw_aggregate":
        if not kitti_root_dir:
            raise ValueError("KITTI RAW aggregate 模式需要提供 --kitti_root_dir")
        if not match_json:
            raise ValueError("KITTI RAW aggregate 模式需要提供 --match_json")
        result_prefix = result_prefix or "kitti_raw_00_01"
        img_dir = Path(kitti_root_dir)
        kitti_calib_dir = img_dir.parent.parent.parent / "2011_09_26_calib" / "2011_09_26"
        gt_file = build_kitti_gt_camera_json(kitti_calib_dir, result_dir, result_prefix)
        matcher_input_dir = None
        match_json_path = Path(match_json)
        ba_result_json = result_dir / "ba_results" / f"{result_prefix}_ba_result.json"
        plot_png = result_dir / f"{result_prefix}_ba_history.png"
    else:
        raise ValueError(f"不支持的数据集模式: {dataset_mode}")

    return dict(
        dataset_mode   = dataset_mode,
        scene          = scene,
        img_dir        = img_dir,
        matcher_input_dir = matcher_input_dir,
        result_dir     = result_dir,
        match_json     = match_json_path,
        ba_result_json = ba_result_json,
        plot_png       = plot_png,
        gt_file        = gt_file,
        weights        = PROJECT_ROOT / "matchmodel/SuperPointPretrainedNetwork/superpoint_v1.pth",
        ba_bin         = PROJECT_ROOT / "build/bin/run_offline_stereo_ba",
        match_script   = PROJECT_ROOT / "stereo_calib/scripts/superpoint_stereo_match.py",
        plot_script    = PROJECT_ROOT / "stereo_calib/scripts/plot_ba_history.py",
    )


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def log_step(msg: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


def run_cmd(cmd: list, cwd: Path = PROJECT_ROOT) -> None:
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    ret = subprocess.run(cmd, cwd=cwd)
    if ret.returncode != 0:
        sys.exit(f"\n命令执行失败（exit code {ret.returncode}）")


# ── 参数解析 ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="通用立体标定全流程脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 场景选择
    p.add_argument("--dataset_mode", choices=["project", "kitti2015", "kitti_raw_aggregate"], default="project",
                   help="数据集模式：project 使用 blender-file/stereo_{scene}；kitti2015 使用 KITTI RAW 单帧；kitti_raw_aggregate 直接消费汇总 matches JSON")
    p.add_argument("--scene", default="panoramic_01",
                   help="project 模式下为场景名（如 panoramic_02）；kitti2015 模式下为帧号（如 0000000000）")
    p.add_argument("--kitti_root_dir", default=None,
                   help="KITTI RAW drive 根目录，需包含 image_00/data 和 image_01/data")
    p.add_argument("--match_json", default=None,
                   help="已有 matches JSON 路径（供 kitti_raw_aggregate 等模式复用）")
    p.add_argument("--result_prefix", default=None,
                   help="输出结果文件名前缀（供聚合模式使用）")
    p.add_argument("--pixel_tag", default="3px",
                   help="输出结果文件名中的像素阈值标签（如 3px、2px、1px）")

    # 流程控制
    flow = p.add_argument_group("流程控制")
    flow.add_argument("--skip_match", action="store_true",
                      help="跳过特征匹配（复用已有 matches JSON）")
    flow.add_argument("--skip_ba",    action="store_true",
                      help="跳过 BA 优化（复用已有 BA 结果 JSON）")
    flow.add_argument("--no_plot",    action="store_true",
                      help="跳过可视化步骤")

    # SuperPoint 参数
    sp = p.add_argument_group("SuperPoint 特征匹配参数")
    sp.add_argument("--nn_thresh",    type=float, default=0.7,
                    help="描述子 L2 距离阈值")
    sp.add_argument("--conf_thresh",  type=float, default=0.015,
                    help="关键点置信度阈值")
    sp.add_argument("--nms_dist",     type=int,   default=4,
                    help="NMS 抑制半径（像素）")
    sp.add_argument("--cuda",         action="store_true",
                    help="SuperPoint 推理使用 CUDA")

    # BA 参数
    ba = p.add_argument_group("Bundle Adjustment 参数")
    ba.add_argument("--max_iter",             type=int,   default=200,
                    help="全局 BA 最大迭代次数")
    ba.add_argument("--incremental_max_iter", type=int,   default=20,
                    help="增量 BA 最大迭代次数")
    ba.add_argument("--global_opt_interval",  type=int,   default=5,
                    help="每隔多少帧触发一次全局 BA")
    ba.add_argument("--huber",                type=float, default=1.0,
                    help="Huber 损失函数阈值（像素）")
    ba.add_argument("--min_track_len",        type=int,   default=3,
                    help="有效轨迹最小长度")
    ba.add_argument("--min_pair_inliers",     type=int,   default=12,
                    help="图像对最少内点数")
    ba.add_argument("--max_score",            type=float, default=1.0,
                    help="匹配点最大得分阈值（过滤低质量匹配，越小越严格）")
    ba.add_argument("--fix_distortion",       action="store_true",
                    help="固定畸变参数不优化")

    return p.parse_args()


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    try:
        paths = resolve_paths(args.scene, args.pixel_tag, args.dataset_mode, args.kitti_root_dir,
                              args.match_json, args.result_prefix)
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(str(exc))

    # 前置检查
    missing = []
    if not paths["img_dir"].exists():
        missing.append(f"图像目录：{paths['img_dir']}")
    if args.dataset_mode in ("kitti2015", "kitti_raw_aggregate"):
        for subdir in ("image_00/data", "image_01/data"):
            subdir_path = paths["img_dir"] / subdir
            if not subdir_path.exists():
                missing.append(f"KITTI 子目录：{subdir_path}")
    if not args.skip_match and not paths["weights"].exists():
        missing.append(f"SuperPoint 权重：{paths['weights']}")
    if not args.skip_ba and not paths["ba_bin"].exists():
        missing.append(f"BA 可执行文件：{paths['ba_bin']}（请先编译：cd stereo_calib && mkdir -p build && cd build && cmake .. && make）")
    if missing:
        sys.exit("错误，以下文件/目录不存在：\n  " + "\n  ".join(missing))

    paths["result_dir"].mkdir(parents=True, exist_ok=True)
    paths["match_json"].parent.mkdir(parents=True, exist_ok=True)
    paths["ba_result_json"].parent.mkdir(parents=True, exist_ok=True)

    # ── 步骤 1：SuperPoint 特征匹配 ───────────────────────────────────────────
    if not args.skip_match:
        if args.dataset_mode == "kitti_raw_aggregate":
            sys.exit("kitti_raw_aggregate 模式请提供 --match_json 并使用 --skip_match")

        log_step(f"步骤 1/3：SuperPoint 特征匹配（数据集：{args.dataset_mode}，场景：{args.scene}）")

        cmd = [
            sys.executable, str(paths["match_script"]),
            "--dataset_mode", args.dataset_mode,
            "--img_dir", str(paths["matcher_input_dir"]),
            "--scene", args.scene,
            "--weights", str(paths["weights"]),
            "--output", str(paths["match_json"]),
            "--nn_thresh", str(args.nn_thresh),
            "--conf_thresh", str(args.conf_thresh),
            "--nms_dist", str(args.nms_dist),
        ]
        if args.cuda:
            cmd.append("--cuda")

        run_cmd(cmd)
        print(f"\n匹配结果已写入：{paths['match_json']}")
    else:
        if not paths["match_json"].exists():
            sys.exit(f"--skip_match 已指定，但文件不存在：{paths['match_json']}")
        print(f"跳过特征匹配，使用：{paths['match_json']}")

    # ── 步骤 2：增量式 Bundle Adjustment ─────────────────────────────────────
    if not args.skip_ba:
        log_step(f"步骤 2/3：增量式 Bundle Adjustment 优化（数据集：{args.dataset_mode}，场景：{args.scene}）")

        cmd = [
            str(paths["ba_bin"]),
            "--input", str(paths["match_json"]),
            "--output", str(paths["ba_result_json"]),
            "--max_iter", str(args.max_iter),
            "--incremental_max_iter", str(args.incremental_max_iter),
            "--global_opt_interval", str(args.global_opt_interval),
            "--huber", str(args.huber),
            "--min_track_len", str(args.min_track_len),
            "--min_pair_inliers", str(args.min_pair_inliers),
            "--max_score", str(args.max_score),
        ]
        if args.fix_distortion:
            cmd.append("--fix_distortion")
        if paths["gt_file"] is not None and paths["gt_file"].exists():
            cmd += ["--gt_param_file", str(paths["gt_file"])]

        run_cmd(cmd, cwd=PROJECT_ROOT)
        print(f"\nBA 结果已写入：{paths['ba_result_json']}")
    else:
        if not paths["ba_result_json"].exists():
            sys.exit(f"--skip_ba 已指定，但文件不存在：{paths['ba_result_json']}")
        print(f"跳过 BA 优化，使用：{paths['ba_result_json']}")

    # ── 步骤 3：可视化 ────────────────────────────────────────────────────────
    if not args.no_plot:
        log_step(f"步骤 3/3：BA 优化历史可视化（数据集：{args.dataset_mode}，场景：{args.scene}）")

        cmd = [
            sys.executable, str(paths["plot_script"]),
            "--input", str(paths["ba_result_json"]),
            "--output", str(paths["plot_png"]),
        ]
        run_cmd(cmd)
        print(f"\n可视化图片已保存：{paths['plot_png']}")

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    log_step("全流程完成")
    print(f"  数据集模式：{args.dataset_mode}")
    print(f"  场景：{args.scene}")
    print(f"  特征匹配结果：{paths['match_json']}")
    print(f"  BA 优化结果：{paths['ba_result_json']}")
    if not args.no_plot:
        print(f"  收敛曲线图：{paths['plot_png']}")


if __name__ == "__main__":
    main()

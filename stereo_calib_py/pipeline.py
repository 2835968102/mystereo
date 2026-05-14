"""Single-run stereo calibration pipeline orchestration.

本模块对应“跑一次 scene”的完整流程：检查输入、可选匹配、可选 BA、
可选画图、最后打印汇总。批量循环由 `run_experiment.py` 负责。
"""

from __future__ import annotations

import argparse
import sys

from .commands import log_step, run_cmd
from .paths import PipelinePaths, resolve_paths
from .project import PROJECT_ROOT


def validate_inputs(args: argparse.Namespace, paths: PipelinePaths) -> None:
    """在启动耗时步骤前检查关键输入是否存在。

    这里尽量一次性收集所有缺失项，避免用户修一个路径、重跑后才看到
    下一个缺失项。
    """
    missing = []
    if not paths.img_dir.exists():
        missing.append(f"图像目录：{paths.img_dir}")
    if args.dataset_mode in ("kitti2015", "kitti_raw_aggregate"):
        for subdir in ("image_00/data", "image_01/data"):
            subdir_path = paths.img_dir / subdir
            if not subdir_path.exists():
                missing.append(f"KITTI 子目录：{subdir_path}")
    if not args.skip_match and not paths.weights.exists():
        missing.append(f"SuperPoint 权重：{paths.weights}")
    if not args.skip_ba and not paths.ba_bin.exists():
        missing.append(
            f"BA 可执行文件：{paths.ba_bin}"
            "（请先编译：cd stereo_calib && mkdir -p build && cd build && cmake .. && make）"
        )
    if missing:
        sys.exit("错误，以下文件/目录不存在：\n  " + "\n  ".join(missing))


def ensure_output_dirs(paths: PipelinePaths) -> None:
    """创建 pipeline 会写入的输出目录。"""
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    paths.match_json.parent.mkdir(parents=True, exist_ok=True)
    paths.ba_result_json.parent.mkdir(parents=True, exist_ok=True)


def run_match_if_needed(args: argparse.Namespace, paths: PipelinePaths) -> None:
    """按需运行 SuperPoint 匹配，或校验已有 matches JSON。"""
    if not args.skip_match:
        if args.dataset_mode == "kitti_raw_aggregate":
            sys.exit("kitti_raw_aggregate 模式请提供 --match_json 并使用 --skip_match")

        log_step(f"步骤 1/3：SuperPoint 特征匹配（数据集：{args.dataset_mode}，场景：{args.scene}）")

        cmd = [
            sys.executable,
            str(paths.match_script),
            "--dataset_mode",
            args.dataset_mode,
            "--img_dir",
            str(paths.matcher_input_dir),
            "--scene",
            args.scene,
            "--weights",
            str(paths.weights),
            "--output",
            str(paths.match_json),
            "--nn_thresh",
            str(args.nn_thresh),
            "--conf_thresh",
            str(args.conf_thresh),
            "--nms_dist",
            str(args.nms_dist),
        ]
        # CUDA 只影响 Python SuperPoint 推理，不影响后续 C++ BA。
        if args.cuda:
            cmd.append("--cuda")

        run_cmd(cmd)
        print(f"\n匹配结果已写入：{paths.match_json}")
    else:
        if not paths.match_json.exists():
            sys.exit(f"--skip_match 已指定，但文件不存在：{paths.match_json}")
        print(f"跳过特征匹配，使用：{paths.match_json}")


def run_ba_if_needed(args: argparse.Namespace, paths: PipelinePaths) -> None:
    """按需运行 C++ 增量式 BA，或校验已有 BA 结果 JSON。"""
    if not args.skip_ba:
        log_step(f"步骤 2/3：增量式 Bundle Adjustment 优化（数据集：{args.dataset_mode}，场景：{args.scene}）")

        cmd = [
            str(paths.ba_bin),
            "--input",
            str(paths.match_json),
            "--output",
            str(paths.ba_result_json),
            "--max_iter",
            str(args.max_iter),
            "--incremental_max_iter",
            str(args.incremental_max_iter),
            "--global_opt_interval",
            str(args.global_opt_interval),
            "--huber",
            str(args.huber),
            "--min_track_len",
            str(args.min_track_len),
            "--min_pair_inliers",
            str(args.min_pair_inliers),
            "--max_score",
            str(args.max_score),
        ]
        if args.fix_distortion:
            cmd.append("--fix_distortion")
        # Blender/KITTI 的 GT 文件存在时自动传给 C++，用于结果 JSON 中的
        # diff_vs_gt 和日志对比；没有 GT 时仍可正常优化。
        if paths.gt_file is not None and paths.gt_file.exists():
            cmd += ["--gt_param_file", str(paths.gt_file)]

        run_cmd(cmd, cwd=PROJECT_ROOT)
        print(f"\nBA 结果已写入：{paths.ba_result_json}")
    else:
        if not paths.ba_result_json.exists():
            sys.exit(f"--skip_ba 已指定，但文件不存在：{paths.ba_result_json}")
        print(f"跳过 BA 优化，使用：{paths.ba_result_json}")


def run_plot_if_needed(args: argparse.Namespace, paths: PipelinePaths) -> None:
    """按需调用 BA 历史绘图脚本。"""
    if not args.no_plot:
        log_step(f"步骤 3/3：BA 优化历史可视化（数据集：{args.dataset_mode}，场景：{args.scene}）")

        cmd = [
            sys.executable,
            str(paths.plot_script),
            "--input",
            str(paths.ba_result_json),
            "--output",
            str(paths.plot_png),
        ]
        run_cmd(cmd)
        print(f"\n可视化图片已保存：{paths.plot_png}")


def print_summary(args: argparse.Namespace, paths: PipelinePaths) -> None:
    """打印本次 pipeline 产物位置，方便复制到后续分析脚本。"""
    log_step("全流程完成")
    print(f"  数据集模式：{args.dataset_mode}")
    print(f"  场景：{args.scene}")
    print(f"  特征匹配结果：{paths.match_json}")
    print(f"  BA 优化结果：{paths.ba_result_json}")
    if not args.no_plot:
        print(f"  收敛曲线图：{paths.plot_png}")


def run_pipeline(args: argparse.Namespace) -> None:
    """执行单次 pipeline。

    `args` 来自顶层 CLI，也可以在测试或其他 Python 调用方中手动构造。
    这个函数故意不解析命令行，让流程逻辑和 CLI 表达解耦。
    """
    try:
        paths = resolve_paths(
            args.scene,
            args.pixel_tag,
            args.dataset_mode,
            args.kitti_root_dir,
            args.match_json,
            args.result_prefix,
        )
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(str(exc))

    validate_inputs(args, paths)
    ensure_output_dirs(paths)
    run_match_if_needed(args, paths)
    run_ba_if_needed(args, paths)
    run_plot_if_needed(args, paths)
    print_summary(args, paths)

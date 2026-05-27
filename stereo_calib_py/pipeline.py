"""Single-run stereo calibration pipeline orchestration.

本模块对应“跑一次 scene”的完整流程：检查输入、可选匹配、可选 BA、
可选画图、最后打印汇总。批量循环由 `run_experiment.py` 负责。
"""

from __future__ import annotations

import argparse
import sys

from .commands import log_step, run_cmd
from .config import PipelineConfig
from .paths import PipelinePaths, prepare_ground_truth, resolve_paths
from .project import PROJECT_ROOT


def validate_inputs(config: PipelineConfig, paths: PipelinePaths) -> None:
    """在启动耗时步骤前检查关键输入是否存在。

    这里尽量一次性收集所有缺失项，避免用户修一个路径、重跑后才看到
    下一个缺失项。
    """
    missing = []
    if not paths.img_dir.exists():
        missing.append(f"图像目录：{paths.img_dir}")
    if config.dataset_mode in ("kitti2015", "kitti_raw_aggregate"):
        for subdir in ("image_00/data", "image_01/data"):
            subdir_path = paths.img_dir / subdir
            if not subdir_path.exists():
                missing.append(f"KITTI 子目录：{subdir_path}")
    if config.dataset_mode == "kitti_raw_sequence":
        if paths.left_img_dir is None or not paths.left_img_dir.exists():
            missing.append(f"KITTI 左图目录：{paths.left_img_dir}")
        if paths.right_img_dir is None or not paths.right_img_dir.exists():
            missing.append(f"KITTI 右图目录：{paths.right_img_dir}")
    if not config.skip_match and paths.matcher_kind != "raw_stereo_sequence" and not paths.weights.exists():
        missing.append(f"SuperPoint 权重：{paths.weights}")
    if not config.skip_match and paths.matcher_kind == "raw_stereo_sequence":
        raw_weights = paths.match_script.parent / "superpoint_v1.pth"
        if not raw_weights.exists():
            missing.append(f"SuperPoint raw 权重：{raw_weights}")
    if not config.skip_ba and not paths.ba_bin.exists():
        missing.append(
            f"BA 可执行文件：{paths.ba_bin}"
            "（请先编译：cd stereo_calib && mkdir -p build && cd build && cmake .. && make）"
        )
    if missing:
        sys.exit("错误，以下文件/目录不存在：\n  " + "\n  ".join(missing))


def ensure_output_dirs(paths: PipelinePaths) -> None:
    """创建 pipeline 会写入的输出目录。"""
    # 目录创建放在 GT 准备之前，保证 KITTI 自动 GT 的 result/gt_params 已存在。
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    paths.match_json.parent.mkdir(parents=True, exist_ok=True)
    paths.ba_result_json.parent.mkdir(parents=True, exist_ok=True)
    paths.plot_png.parent.mkdir(parents=True, exist_ok=True)
    if paths.gt_file is not None:
        paths.gt_file.parent.mkdir(parents=True, exist_ok=True)


def run_match_if_needed(config: PipelineConfig, paths: PipelinePaths) -> None:
    """按需运行 SuperPoint 匹配，或校验已有 matches JSON。"""
    if not config.skip_match:
        if config.dataset_mode == "kitti_raw_aggregate":
            sys.exit("kitti_raw_aggregate 模式请提供 --match_json 并使用 --skip_match")

        log_step(f"步骤 1/3：SuperPoint 特征匹配（数据集：{config.dataset_mode}，场景：{config.scene}）")

        if paths.matcher_kind == "raw_stereo_sequence":
            # KITTI RAW 连续帧脚本直接接收左右目目录，内部生成稀疏 pair。
            cmd = [
                sys.executable,
                str(paths.match_script),
                "--left_img_dir",
                str(paths.left_img_dir),
                "--right_img_dir",
                str(paths.right_img_dir),
                "--output",
                str(paths.match_json),
                "--nn_thresh",
                str(config.nn_thresh),
                "--conf_thresh",
                str(config.conf_thresh),
                "--nms_dist",
                str(config.nms_dist),
            ]
        else:
            # Project 和 KITTI 单帧复用 standard matcher，输入由 dataset_mode 解释。
            cmd = [
                sys.executable,
                str(paths.match_script),
                "--dataset_mode",
                config.dataset_mode,
                "--img_dir",
                str(paths.matcher_input_dir),
                "--scene",
                config.scene,
                "--weights",
                str(paths.weights),
                "--output",
                str(paths.match_json),
                "--nn_thresh",
                str(config.nn_thresh),
                "--conf_thresh",
                str(config.conf_thresh),
                "--nms_dist",
                str(config.nms_dist),
            ]
        # CUDA 只影响 Python SuperPoint 推理，不影响后续 C++ BA。
        if config.cuda:
            cmd.append("--cuda")

        run_cmd(cmd)
        print(f"\n匹配结果已写入：{paths.match_json}")
    else:
        if not paths.match_json.exists():
            sys.exit(f"--skip_match 已指定，但文件不存在：{paths.match_json}")
        print(f"跳过特征匹配，使用：{paths.match_json}")


def run_ba_if_needed(config: PipelineConfig, paths: PipelinePaths) -> None:
    """按需运行 C++ 增量式 BA，或校验已有 BA 结果 JSON。"""
    if not config.skip_ba:
        log_step(f"步骤 2/3：增量式 Bundle Adjustment 优化（数据集：{config.dataset_mode}，场景：{config.scene}）")

        cmd = [
            str(paths.ba_bin),
            "--input",
            str(paths.match_json),
            "--output",
            str(paths.ba_result_json),
            "--max_iter",
            str(config.max_iter),
            "--incremental_max_iter",
            str(config.incremental_max_iter),
            "--global_opt_interval",
            str(config.global_opt_interval),
            "--huber",
            str(config.huber),
            "--min_track_len",
            str(config.min_track_len),
            "--min_pair_inliers",
            str(config.min_pair_inliers),
            "--max_score",
            str(config.max_score),
        ]
        if config.fix_distortion:
            cmd.append("--fix_distortion")
        # Blender/KITTI 的 GT 文件存在时自动传给 C++，用于结果 JSON 中的
        # diff_vs_gt 和日志对比；没有 GT 时仍可正常优化。
        if paths.gt_file is not None and paths.gt_file.exists():
            cmd += ["--gt_param_file", str(paths.gt_file)]

        if config.init_param_file:
            cmd += ["--init_param_file", str(config.init_param_file)]
        if config.frame_poses_file:
            cmd += ["--frame_poses_file", str(config.frame_poses_file)]

        for name, value in config.optional_ba_value_args().items():
            # None 表示使用 C++ 默认值；其余值显式传入，便于实验复现。
            if value is not None:
                cmd += [f"--{name}", str(value)]

        if config.enable_per_frame_correction:
            cmd.append("--enable_per_frame_correction")
        if config.enable_incremental_free_principal_point_refine:
            cmd.append("--enable_incremental_free_principal_point_refine")
        if config.reset_camera_params_each_ba_round:
            cmd.append("--reset_camera_params_each_ba_round")

        run_cmd(cmd, cwd=PROJECT_ROOT)
        print(f"\nBA 结果已写入：{paths.ba_result_json}")
    else:
        if not paths.ba_result_json.exists():
            sys.exit(f"--skip_ba 已指定，但文件不存在：{paths.ba_result_json}")
        print(f"跳过 BA 优化，使用：{paths.ba_result_json}")


def run_plot_if_needed(config: PipelineConfig, paths: PipelinePaths) -> None:
    """按需调用 BA 历史绘图脚本。"""
    if not config.no_plot:
        log_step(f"步骤 3/3：BA 优化历史可视化（数据集：{config.dataset_mode}，场景：{config.scene}）")

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


def print_summary(config: PipelineConfig, paths: PipelinePaths) -> None:
    """打印本次 pipeline 产物位置，方便复制到后续分析脚本。"""
    log_step("全流程完成")
    print(f"  数据集模式：{config.dataset_mode}")
    print(f"  场景：{config.scene}")
    print(f"  特征匹配结果：{paths.match_json}")
    print(f"  BA 优化结果：{paths.ba_result_json}")
    if not config.no_plot:
        print(f"  收敛曲线图：{paths.plot_png}")


def run_pipeline(config_or_args: PipelineConfig | argparse.Namespace) -> None:
    """执行单次 pipeline。

    顶层 CLI 可以继续传 argparse.Namespace；进入本函数后会转换成
    PipelineConfig，让后续流程不再依赖隐式 Namespace 字段。
    """
    config = PipelineConfig.coerce(config_or_args)
    try:
        paths = resolve_paths(config)
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(str(exc))

    validate_inputs(config, paths)
    ensure_output_dirs(paths)
    if not config.skip_ba:
        # 路径解析阶段不写文件；真正要跑 BA 时才准备 KITTI GT JSON。
        try:
            prepare_ground_truth(paths)
        except (ValueError, FileNotFoundError) as exc:
            sys.exit(str(exc))
    run_match_if_needed(config, paths)
    run_ba_if_needed(config, paths)
    run_plot_if_needed(config, paths)
    print_summary(config, paths)

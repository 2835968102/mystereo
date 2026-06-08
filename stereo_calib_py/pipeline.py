"""Single-run stereo calibration pipeline orchestration.

本模块对应“跑一次 scene”的完整流程：检查输入、可选匹配、可选 BA、
可选画图、最后打印汇总。批量循环由 `run_experiment.py` 负责。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .commands import log_step, run_cmd
from .config import PipelineConfig
from .kitti_oxts import build_kitti_raw_oxts_frame_poses
from .paths import PipelinePaths, prepare_ground_truth, resolve_paths
from .project import PROJECT_ROOT


def _project_path(path: Path) -> Path:
    """Resolve repository-relative paths the same way pipeline commands do."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_camera_param_file(path: Path) -> dict[str, object]:
    """Load the subset of camera parameters needed for leakage checks."""
    resolved = _project_path(path)
    if resolved.suffix == ".json":
        with resolved.open(encoding="utf-8") as f:
            data = json.load(f)
        extrinsics = data["extrinsics"]
        return {
            "left": data["left"],
            "right": data["right"],
            "extrinsics": extrinsics,
            "t": extrinsics["t"],
        }

    values: dict[str, float] = {}
    with resolved.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = float(value.strip())

    extrinsics = {
        "R": [
            values.get("R00", 1.0),
            values.get("R01", 0.0),
            values.get("R02", 0.0),
            values.get("R10", 0.0),
            values.get("R11", 1.0),
            values.get("R12", 0.0),
            values.get("R20", 0.0),
            values.get("R21", 0.0),
            values.get("R22", 1.0),
        ],
        "t": [values["tx"], values["ty"], values["tz"]],
    }
    return {
        "left": {
            "fx": values["left_fx"],
            "fy": values["left_fy"],
            "cx": values["left_cx"],
            "cy": values["left_cy"],
        },
        "right": {
            "fx": values["right_fx"],
            "fy": values["right_fy"],
            "cx": values["right_cx"],
            "cy": values["right_cy"],
        },
        "extrinsics": extrinsics,
        "t": extrinsics["t"],
    }


def _camera_intrinsic_values(camera: dict[str, object], keys: tuple[str, ...]) -> list[float]:
    values = []
    for side in ("left", "right"):
        intrinsics = camera[side]
        if not isinstance(intrinsics, dict):
            raise ValueError(f"Invalid camera intrinsics for {side}")
        for key in keys:
            values.append(float(intrinsics[key]))
    return values


def _max_abs_pair_delta(left_values: list[float], right_values: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(left_values, right_values, strict=True))


def _translation_norm(camera: dict[str, object]) -> float:
    t = camera["t"]
    if not isinstance(t, list) or len(t) < 3:
        raise ValueError("Invalid camera translation vector")
    return math.sqrt(sum(float(v) * float(v) for v in t[:3]))


def reject_gt_like_initialization(config: PipelineConfig, paths: PipelinePaths) -> None:
    """Abort KITTI BA runs whose init file is effectively the evaluation GT.

    GT is useful for final diff reporting, but it must not leak into the
    initial state or a locked/normalized focal strategy can report fake zero
    error. The check is intentionally narrow: it only runs when both an init
    camera and a GT camera are supplied for a KITTI optimization.
    """
    if config.skip_ba or not config.is_kitti_mode:
        return
    if not config.init_param_file or paths.gt_file is None:
        return

    init_path = Path(config.init_param_file)
    gt_path = paths.gt_file
    if not _project_path(init_path).exists() or not _project_path(gt_path).exists():
        return

    init_camera = _load_camera_param_file(init_path)
    gt_camera = _load_camera_param_file(gt_path)
    init_focals = _camera_intrinsic_values(init_camera, ("fx", "fy"))
    gt_focals = _camera_intrinsic_values(gt_camera, ("fx", "fy"))
    init_principal = _camera_intrinsic_values(init_camera, ("cx", "cy"))
    gt_principal = _camera_intrinsic_values(gt_camera, ("cx", "cy"))
    init_focal_mean = sum(init_focals) / len(init_focals)
    gt_focal_mean = sum(gt_focals) / len(gt_focals)
    init_t = init_camera["t"]
    gt_t = gt_camera["t"]
    if not isinstance(init_t, list) or not isinstance(gt_t, list):
        raise ValueError("Invalid camera translation vector")

    suspicious_reasons = []
    if config.normalize_initial_focal_to_mean and abs(init_focal_mean - gt_focal_mean) < 1e-3:
        suspicious_reasons.append(
            "normalize_initial_focal_to_mean would set fx/fy to the GT focal mean"
        )
    if config.fix_focal_length and abs(init_focal_mean - gt_focal_mean) < 1e-3:
        suspicious_reasons.append("fix_focal_length would lock a GT-like focal mean")
    if config.fix_focal_length and _max_abs_pair_delta(init_focals, gt_focals) < 1e-3:
        suspicious_reasons.append("fix_focal_length would lock GT-like focal values")
    if (
        _max_abs_pair_delta(init_focals, gt_focals) < 1e-3
        and _max_abs_pair_delta(init_principal, gt_principal) < 1e-3
        and abs(_translation_norm(init_camera) - _translation_norm(gt_camera)) < 1e-6
        and abs(float(init_t[0]) - float(gt_t[0])) < 1e-6
    ):
        suspicious_reasons.append("the init camera is numerically identical to the GT camera")

    if suspicious_reasons:
        reason_text = "\n  - ".join(suspicious_reasons)
        sys.exit(
            "Refusing to run BA because the initial camera appears to leak evaluation GT:\n"
            f"  init_param_file: {_project_path(init_path)}\n"
            f"  gt_param_file: {_project_path(gt_path)}\n"
            f"  - {reason_text}\n"
            "Use a GT-free init file or disable the GT-like focal normalization/fix."
        )


def _rotation_error_deg(r_est: list[float], r_gt: list[float]) -> float:
    """Return angle of R_est * R_gt^T for row-major 3x3 matrices."""
    trace = 0.0
    for row in range(3):
        for col in range(3):
            trace += float(r_est[row * 3 + col]) * float(r_gt[row * 3 + col])
    cos_theta = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    return math.degrees(math.acos(cos_theta))


def _translation_norm_values(t: list[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in t[:3]))


def _intrinsics_diff(est: dict[str, object], gt: dict[str, object]) -> dict[str, float]:
    keys = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
    return {key: float(est.get(key, 0.0)) - float(gt.get(key, 0.0)) for key in keys}


def _extrinsics_diff(est: dict[str, object], gt: dict[str, object]) -> dict[str, object]:
    r_est = [float(v) for v in est.get("R", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])]
    r_gt = [float(v) for v in gt.get("R", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])]
    t_est = [float(v) for v in est["t"]]
    t_gt = [float(v) for v in gt["t"]]
    return {
        "R": [a - b for a, b in zip(r_est, r_gt, strict=True)],
        "t": [a - b for a, b in zip(t_est, t_gt, strict=True)],
        "baseline": _translation_norm_values(t_est) - _translation_norm_values(t_gt),
        "rotation_error_deg": _rotation_error_deg(r_est, r_gt),
    }


def _camera_diff_vs_gt(camera: dict[str, object], gt_camera: dict[str, object]) -> dict[str, object]:
    gt_extrinsics = gt_camera.get("extrinsics")
    if not isinstance(gt_extrinsics, dict):
        gt_extrinsics = {"R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], "t": gt_camera["t"]}
    return {
        "left": _intrinsics_diff(camera["left"], gt_camera["left"]),
        "right": _intrinsics_diff(camera["right"], gt_camera["right"]),
        "extrinsics": _extrinsics_diff(camera["extrinsics"], gt_extrinsics),
    }


def _abs_number(mapping: dict[str, object], key: str) -> float | None:
    value = mapping.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return abs(float(value))
    return None


def _mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _append_if_number(values: list[float], value: float | None) -> None:
    if value is not None and math.isfinite(value):
        values.append(value)


def _build_summary_from_evaluated_history(history: list[object]) -> dict[str, object]:
    reproj: list[float] = []
    rotation: list[float] = []
    baseline: list[float] = []
    tx: list[float] = []
    ty: list[float] = []
    tz: list[float] = []
    left_fx: list[float] = []
    left_fy: list[float] = []
    right_fx: list[float] = []
    right_fy: list[float] = []
    focal: list[float] = []

    for item in history:
        if not isinstance(item, dict):
            continue
        value = item.get("reproj_error")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            reproj.append(float(value))
        dvg = item.get("diff_vs_gt")
        if not isinstance(dvg, dict):
            continue
        ext = dvg.get("extrinsics") if isinstance(dvg.get("extrinsics"), dict) else {}
        left = dvg.get("left") if isinstance(dvg.get("left"), dict) else {}
        right = dvg.get("right") if isinstance(dvg.get("right"), dict) else {}

        _append_if_number(rotation, _abs_number(ext, "rotation_error_deg"))
        _append_if_number(baseline, _abs_number(ext, "baseline"))
        t_values = ext.get("t") if isinstance(ext, dict) else None
        if isinstance(t_values, list) and len(t_values) >= 3:
            _append_if_number(tx, abs(float(t_values[0])))
            _append_if_number(ty, abs(float(t_values[1])))
            _append_if_number(tz, abs(float(t_values[2])))

        for target, source, key in (
            (left_fx, left, "fx"),
            (left_fy, left, "fy"),
            (right_fx, right, "fx"),
            (right_fy, right, "fy"),
        ):
            focal_value = _abs_number(source, key)
            _append_if_number(target, focal_value)
            _append_if_number(focal, focal_value)

    return {
        "avg_reproj_error_px": _mean_or_none(reproj),
        "avg_rotation_error_deg": _mean_or_none(rotation),
        "avg_left_fx_error_px": _mean_or_none(left_fx),
        "avg_left_fy_error_px": _mean_or_none(left_fy),
        "avg_right_fx_error_px": _mean_or_none(right_fx),
        "avg_right_fy_error_px": _mean_or_none(right_fy),
        "avg_baseline_error_m": _mean_or_none(baseline),
        "avg_trans_err_x_m": _mean_or_none(tx),
        "avg_trans_err_y_m": _mean_or_none(ty),
        "avg_trans_err_z_m": _mean_or_none(tz),
        "avg_focal_error_px": _mean_or_none(focal),
    }


def append_gt_evaluation_if_available(config: PipelineConfig, paths: PipelinePaths) -> None:
    """Annotate BA output with GT diffs after optimization has finished.

    The C++ BA process intentionally does not receive camera GT from this
    pipeline. This keeps the optimizer and acceptance gates blind to the
    evaluation target while preserving the existing JSON/plot diagnostics.
    """
    if config.skip_ba or paths.gt_file is None or not paths.gt_file.exists():
        return
    if not paths.ba_result_json.exists():
        return

    with paths.ba_result_json.open(encoding="utf-8") as f:
        result = json.load(f)
    gt_source = result.get("gt_source")
    if isinstance(gt_source, str) and gt_source and not gt_source.startswith("python_post_ba_eval:"):
        sys.exit(
            "Refusing to overwrite a BA result that already contains C++/optimizer-side GT evaluation:\n"
            f"  ba_result_json: {paths.ba_result_json}\n"
            f"  gt_source: {gt_source}\n"
            "Delete the contaminated result or rerun with a GT-free optimizer command."
        )
    gt_camera = _load_camera_param_file(paths.gt_file)

    result_camera = {
        "left": result["left"],
        "right": result["right"],
        "extrinsics": result["extrinsics"],
    }
    result["gt_source"] = f"python_post_ba_eval: {paths.gt_file}"
    result["diff_vs_gt"] = _camera_diff_vs_gt(result_camera, gt_camera)

    history = result.get("optimization_history")
    if isinstance(history, list):
        for item in history:
            if isinstance(item, dict) and isinstance(item.get("camera"), dict):
                item["diff_vs_gt"] = _camera_diff_vs_gt(item["camera"], gt_camera)
        result["summary"] = _build_summary_from_evaluated_history(history)

    with paths.ba_result_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
        f.write("\n")
    print(f"GT 评估已在 BA 结束后写入：{paths.ba_result_json}")


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
        if config.disable_ratio_margin:
            cmd.append("--disable_ratio_margin")
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
        frame_poses_file = Path(config.frame_poses_file) if config.frame_poses_file else None
        needs_kitti_oxts_poses = (
            config.dataset_mode == "kitti_raw_sequence"
            and (
                (config.frame_distance_prior is not None and config.frame_distance_prior > 0.0)
                or (config.frame_position_prior is not None and config.frame_position_prior > 0.0)
                or (
                    config.frame_translation_vector_prior is not None
                    and config.frame_translation_vector_prior > 0.0
                )
                or (
                    config.frame_translation_direction_prior is not None
                    and config.frame_translation_direction_prior > 0.0
                )
                or (
                    config.frame_rotation_angle_prior is not None
                    and config.frame_rotation_angle_prior > 0.0
                )
                or (
                    config.frame_rotation_vector_prior is not None
                    and config.frame_rotation_vector_prior > 0.0
                )
                or (
                    config.frame_absolute_rotation_prior is not None
                    and config.frame_absolute_rotation_prior > 0.0
                )
                or config.initialize_frame_poses_from_external
            )
        )
        if (
            frame_poses_file is None
            and needs_kitti_oxts_poses
        ):
            if paths.left_img_dir is None:
                sys.exit("frame_distance_prior 需要 KITTI left_img_dir 才能生成 OXTS 位姿")
            frame_poses_file = paths.result_dir / "frame_poses" / (
                f"{paths.ba_result_json.stem}_oxts_frame_poses.json"
            )
            try:
                build_kitti_raw_oxts_frame_poses(paths.left_img_dir, frame_poses_file)
            except (ValueError, FileNotFoundError) as exc:
                sys.exit(str(exc))
            print(f"OXTS frame poses 已写入：{frame_poses_file}")

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
        if config.fix_focal_length:
            cmd.append("--fix_focal_length")
        if config.fix_principal_point:
            cmd.append("--fix_principal_point")
        if config.normalize_initial_focal_to_mean:
            cmd.append("--normalize_initial_focal_to_mean")
        if config.normalize_initial_stereo_translation_to_x_axis:
            cmd.append("--normalize_initial_stereo_translation_to_x_axis")
        if config.initialize_frame_poses_from_external:
            cmd.append("--initialize_frame_poses_from_external")
        if config.fix_external_frame_poses:
            cmd.append("--fix_external_frame_poses")
        if config.fix_external_frame_rotations:
            cmd.append("--fix_external_frame_rotations")
        if config.fix_external_frame_translations:
            cmd.append("--fix_external_frame_translations")
        if config.init_param_file:
            cmd += ["--init_param_file", str(config.init_param_file)]
        if frame_poses_file:
            cmd += ["--frame_poses_file", str(frame_poses_file)]

        for name, value in config.optional_ba_value_args().items():
            # None 表示使用 C++ 默认值；其余值显式传入，便于实验复现。
            if value is not None:
                cmd += [f"--{name}", str(value)]

        if config.enable_per_frame_correction:
            cmd.append("--enable_per_frame_correction")
        if config.enable_incremental_free_principal_point_refine:
            cmd.append("--enable_incremental_free_principal_point_refine")
        if config.free_principal_point_release_focal_length:
            cmd.append("--free_principal_point_release_focal_length")
        if config.final_free_principal_point_release_focal_length:
            cmd.append("--final_free_principal_point_release_focal_length")
        if config.optimize_stereo_tx_in_final_global_ba:
            cmd.append("--optimize_stereo_tx_in_final_global_ba")
        if config.enable_final_stereo_extrinsics_refine:
            cmd.append("--enable_final_stereo_extrinsics_refine")
        if config.reset_camera_params_each_ba_round:
            cmd.append("--reset_camera_params_each_ba_round")

        if "--gt_param_file" in cmd:
            sys.exit("Refusing to pass gt_param_file into C++ BA")

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
        reject_gt_like_initialization(config, paths)
    run_match_if_needed(config, paths)
    run_ba_if_needed(config, paths)
    append_gt_evaluation_if_available(config, paths)
    run_plot_if_needed(config, paths)
    print_summary(config, paths)

#!/usr/bin/env python3
"""从 YAML 配置运行立体标定批量实验。

这个 runner 的目标是把实验定义从 shell 脚本里移出来。
shell 包装脚本保持很薄，场景列表、参数扫描和命令参数都放到
configs/experiments/*.yaml 里维护。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import yaml

from stereo_calib_py.config import PipelineConfig
from stereo_calib_py.kitti_oxts import build_kitti_raw_oxts_frame_poses, kitti_raw_oxts_total_distance
from stereo_calib_py.paths import PipelinePaths, make_output_timestamp, resolve_paths


PROJECT_ROOT = Path(__file__).resolve().parent


def format_elapsed_time(total_seconds: float) -> str:
    seconds = int(total_seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_seconds(value: float | None) -> str:
    """Format a floating-point seconds value for compact tables."""
    if value is None:
        return "-"
    return f"{value:.3f}s"


def load_config(path: Path) -> dict[str, Any]:
    """加载实验配置，并要求顶层是 YAML mapping。"""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def merge_dicts(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """把全局默认参数和单次 run 的覆盖参数合并一层。"""
    merged = dict(base)
    if override:
        merged.update(override)
    return merged


def to_project_path(value: str | Path) -> Path:
    """把相对路径解析为相对于项目根目录的路径。"""
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_executable(name: str, configured_path: str | None, fallback_paths: list[str]) -> str:
    """解析可执行文件路径：配置优先，其次 PATH，最后尝试常见安装位置。"""
    candidates = []
    if configured_path:
        candidates.append(configured_path)
    env_path = os.environ.get(f"{name.upper()}_BIN")
    if env_path:
        candidates.append(env_path)
    found = shutil.which(name)
    if found:
        candidates.append(found)
    candidates.extend(fallback_paths)

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)

    raise FileNotFoundError(
        f"找不到可执行文件 {name}，请在配置 runner.{name}_executable 中指定路径，"
        f"或设置环境变量 {name.upper()}_BIN"
    )


def resolve_conda_executable(runner: dict[str, Any]) -> str:
    """解析 conda 路径，兼容 root 环境 PATH 中没有 conda 的情况。"""
    return resolve_executable(
        "conda",
        runner.get("conda_executable"),
        [
            "/home/hello/miniconda3/bin/conda",
            "/home/hello/anaconda3/bin/conda",
            "/opt/conda/bin/conda",
            "/root/miniconda3/bin/conda",
            "/root/anaconda3/bin/conda",
        ],
    )


def build_python_command(runner: dict[str, Any], script: str) -> list[str]:
    """构造 Python 命令；如果配置了 conda_env，就包一层 `conda run`。"""
    python_executable = str(runner.get("python_executable", "python3"))
    cmd = [python_executable, script]
    conda_env = runner.get("conda_env")
    if conda_env:
        conda_executable = resolve_conda_executable(runner)
        cmd = [
            conda_executable,
            "run",
            "--no-capture-output",
            "-n",
            str(conda_env),
            *cmd,
        ]
    return cmd


def append_cli_args(cmd: list[str], args: dict[str, Any]) -> list[str]:
    """把 YAML 参数映射转换成现有脚本能识别的命令行参数。"""
    full_cmd = list(cmd)
    for key, value in args.items():
        flag = "--" + key.replace("_", "-")
        legacy_flag = "--" + key

        # 现有脚本使用下划线风格参数，比如 --max_iter；这里保持兼容。
        flag = legacy_flag

        if isinstance(value, bool):
            if value:
                full_cmd.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                full_cmd.extend([flag, str(item)])
            continue
        full_cmd.extend([flag, str(value)])
    return full_cmd


def print_command(cmd: list[str]) -> None:
    print("$ " + " ".join(shlex.quote(str(part)) for part in cmd), flush=True)


def run_command(cmd: list[str], dry_run: bool) -> None:
    """打印每条命令；如果不是 dry-run，就真正执行。"""
    print_command(cmd)
    if dry_run:
        return
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}")


def estimate_kitti_raw_motion_m(pipeline_args: dict[str, Any]) -> float | None:
    """Estimate total OXTS path length for a KITTI RAW sequence run."""
    if pipeline_args.get("dataset_mode") != "kitti_raw_sequence":
        return None
    left_img_dir = pipeline_args.get("left_img_dir")
    if not left_img_dir:
        return None
    try:
        return kitti_raw_oxts_total_distance(Path(str(left_img_dir)))
    except (FileNotFoundError, ValueError):
        return None


def get_shared_camera_config(experiment_config: dict[str, Any]) -> dict[str, Any]:
    """Return shared camera-init settings, accepting the legacy tx-only key."""
    shared = experiment_config.get("shared_camera_init")
    if isinstance(shared, dict):
        return shared
    legacy_shared = experiment_config.get("shared_stereo_tx_init")
    return legacy_shared if isinstance(legacy_shared, dict) else {}


def format_motion_m(motion_m: float | None) -> str:
    return f"{motion_m:.6f} m" if motion_m is not None else "unknown"


def read_ba_camera_init_values(ba_result_json: Path) -> dict[str, float] | None:
    """Read camera values that can seed later runs without using GT diffs."""
    if not ba_result_json.exists():
        return None
    with ba_result_json.open(encoding="utf-8") as f:
        data = json.load(f)

    values: dict[str, float] = {}
    for side in ("left", "right"):
        intrinsics = data.get(side)
        if not isinstance(intrinsics, dict):
            return None
        for key in ("fx", "fy", "cx", "cy"):
            if key not in intrinsics:
                return None
            values[f"{side}_{key}"] = float(intrinsics[key])

    t = data.get("extrinsics", {}).get("t")
    if not isinstance(t, list) or len(t) < 3:
        return None
    values["tx"] = float(t[0])
    values["ty"] = float(t[1])
    values["tz"] = float(t[2])
    return values


def aggregate_shared_camera_values(
    camera_values: list[dict[str, float]],
    method: str,
    method_by_key: dict[str, str] | None = None,
    keys: tuple[str, ...] | None = None,
    min_samples_by_key: dict[str, int] | None = None,
) -> dict[str, float]:
    """Aggregate same-key camera values from previous reliable moving runs."""
    if not camera_values:
        return {}
    if keys is None:
        target_keys = sorted(set().union(*(set(values) for values in camera_values)))
    else:
        target_keys = list(keys)

    aggregated: dict[str, float] = {}
    for key in target_keys:
        samples = [values[key] for values in camera_values if key in values]
        min_samples = max(1, int((min_samples_by_key or {}).get(key, 1)))
        if len(samples) < min_samples:
            continue
        key_method = (method_by_key or {}).get(key, method)
        if key_method == "median":
            aggregated[key] = float(median(samples))
        elif key_method == "mean":
            aggregated[key] = float(sum(samples) / len(samples))
        else:
            raise ValueError(f"Unsupported shared_camera_init aggregation for {key}: {key_method}")
    return aggregated


def shared_camera_value_keys(shared: dict[str, Any]) -> tuple[str, ...]:
    """Return the camera fields that may be propagated across runs."""
    if bool(shared.get("share_intrinsics", False)):
        fields = shared.get("shared_fields")
        focal_fields = {"left_fx", "left_fy", "right_fx", "right_fy"}
        allow_shared_focal = bool(shared.get("allow_shared_focal", False))
        if isinstance(fields, list) and fields:
            requested_focal = [str(field) for field in fields if str(field) in focal_fields]
            if requested_focal and not allow_shared_focal:
                raise ValueError(
                    "shared_camera_init refuses to share focal fields without "
                    "allow_shared_focal: true: " + ", ".join(requested_focal)
                )
            allowed = {
                "left_cx",
                "left_cy",
                "right_cx",
                "right_cy",
                "tx",
                "ty",
                "tz",
            }
            if allow_shared_focal:
                allowed |= focal_fields
            selected = tuple(str(field) for field in fields if str(field) in allowed)
            if selected:
                return selected
        # Safe default: share only principal point and stereo translation.
        # Focal sharing can make GT-contaminated or overfit runs look perfectly
        # calibrated, so it must be explicitly opted in above.
        return (
            "left_cx",
            "left_cy",
            "right_cx",
            "right_cy",
            "tx",
        )
    return ("tx",)


def filter_shared_camera_values(
    values: dict[str, float],
    shared: dict[str, Any],
    require_all: bool = True,
) -> dict[str, float]:
    """Keep only the shared camera fields enabled by the experiment config."""
    keys = shared_camera_value_keys(shared)
    missing = [key for key in keys if key not in values]
    if require_all and missing:
        raise KeyError(f"Shared camera values missing keys: {', '.join(missing)}")
    return {key: values[key] for key in keys if key in values}


def shared_min_samples_by_key(shared: dict[str, Any]) -> dict[str, int]:
    """Return per-field sample-count requirements for shared camera aggregation."""
    default_min_samples = max(1, int(shared.get("min_consensus_runs", 2)))
    result = {key: default_min_samples for key in shared_camera_value_keys(shared)}
    raw = shared.get("min_samples_by_field")
    if isinstance(raw, dict):
        for key, value in raw.items():
            result[str(key)] = max(1, int(value))
    return result


def read_ba_history_stage_names(ba_result_json: Path) -> set[str]:
    """Read recorded optimization stage names from a BA result."""
    if not ba_result_json.exists():
        return set()
    with ba_result_json.open(encoding="utf-8") as f:
        data = json.load(f)
    history = data.get("optimization_history")
    if not isinstance(history, list):
        return set()
    return {
        str(record.get("stage"))
        for record in history
        if isinstance(record, dict) and record.get("stage") is not None
    }


def apply_shared_field_stage_filters(
    values: dict[str, float],
    shared: dict[str, Any],
    ba_result_json: Path,
) -> dict[str, float]:
    """Drop fields whose configured no-GT source stage is absent."""
    raw = shared.get("field_require_stage")
    if not isinstance(raw, dict) or not raw:
        return values

    stage_names = read_ba_history_stage_names(ba_result_json)
    filtered = dict(values)
    for key, required_stage in raw.items():
        field = str(key)
        if field in filtered and str(required_stage) not in stage_names:
            del filtered[field]
    return filtered


def read_camera_values_from_history_stage(
    ba_result_json: Path,
    stage_name: str,
) -> dict[str, float] | None:
    """Read camera values from the last optimization-history item with a stage."""
    if not ba_result_json.exists():
        return None
    with ba_result_json.open(encoding="utf-8") as f:
        data = json.load(f)
    history = data.get("optimization_history")
    if not isinstance(history, list):
        return None

    for record in reversed(history):
        if not isinstance(record, dict) or record.get("stage") != stage_name:
            continue
        camera = record.get("camera")
        if not isinstance(camera, dict):
            return None
        values: dict[str, float] = {}
        for side in ("left", "right"):
            intrinsics = camera.get(side)
            if not isinstance(intrinsics, dict):
                return None
            for key in ("fx", "fy", "cx", "cy"):
                if key not in intrinsics:
                    return None
                values[f"{side}_{key}"] = float(intrinsics[key])
        t = camera.get("extrinsics", {}).get("t")
        if not isinstance(t, list) or len(t) < 3:
            return None
        values["tx"] = float(t[0])
        values["ty"] = float(t[1])
        values["tz"] = float(t[2])
        return values
    return None


def apply_shared_field_source_stages(
    values: dict[str, float],
    shared: dict[str, Any],
    ba_result_json: Path,
) -> dict[str, float]:
    """Override selected shared fields from configured no-GT history stages."""
    raw = shared.get("field_source_stage")
    if not isinstance(raw, dict) or not raw:
        return values

    filtered = dict(values)
    stage_cache: dict[str, dict[str, float] | None] = {}
    for key, source_stage in raw.items():
        field = str(key)
        if field not in filtered:
            continue
        stage_name = str(source_stage)
        if stage_name not in stage_cache:
            stage_cache[stage_name] = read_camera_values_from_history_stage(
                ba_result_json,
                stage_name,
            )
        stage_values = stage_cache[stage_name]
        if stage_values is None or field not in stage_values:
            del filtered[field]
            continue
        filtered[field] = stage_values[field]
    return filtered


def read_ba_final_stage(ba_result_json: Path) -> str | None:
    """Return the last recorded optimization stage name, when present."""
    if not ba_result_json.exists():
        return None
    with ba_result_json.open(encoding="utf-8") as f:
        data = json.load(f)
    history = data.get("optimization_history")
    if not isinstance(history, list) or not history:
        return None
    stage = history[-1].get("stage")
    return str(stage) if stage is not None else None


def write_init_params_with_values(
    source_path: Path,
    output_path: Path,
    values: dict[str, float],
    metadata_comments: list[str] | None = None,
) -> Path:
    """Copy an init parameter file, replacing selected scalar entries."""
    lines = source_path.read_text(encoding="utf-8").splitlines()
    replaced_keys: set[str] = set()
    output_lines = []
    for line in lines:
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith("#"):
            key = stripped.split("=", 1)[0].strip()
            if key in values:
                output_lines.append(f"{key}={values[key]:.16g}")
                replaced_keys.add(key)
                continue
        output_lines.append(line)
    for key in sorted(set(values) - replaced_keys):
        output_lines.append(f"{key}={values[key]:.16g}")
    if metadata_comments:
        output_lines = [f"# {comment}" for comment in metadata_comments] + [""] + output_lines
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return output_path


def read_camera_values_from_result(ba_result_json: Path) -> dict[str, float]:
    assert_ba_result_is_safe_for_init(ba_result_json)
    values = read_ba_camera_init_values(ba_result_json)
    if values is None:
        raise ValueError(f"Cannot read camera values from BA result: {ba_result_json}")
    return values


def assert_ba_result_is_safe_for_init(ba_result_json: Path) -> None:
    """Reject BA outputs that were optimized with camera GT in C++."""
    if not ba_result_json.exists():
        return
    with ba_result_json.open(encoding="utf-8") as f:
        data = json.load(f)
    gt_source = data.get("gt_source")
    if isinstance(gt_source, str) and gt_source and not gt_source.startswith("python_post_ba_eval:"):
        raise ValueError(
            f"Refusing to reuse GT-contaminated BA result as init: {ba_result_json} "
            f"(gt_source={gt_source})"
        )


def overlay_camera_values(
    base_values: dict[str, float],
    override_values: dict[str, float],
    keys: list[str],
) -> dict[str, float]:
    merged = dict(base_values)
    for key in keys:
        if key in override_values:
            merged[key] = override_values[key]
    return merged


def sequence_label_for_frame_prefix(label: str) -> str:
    return label.replace(":", "_").replace("/", "_")


def prefix_joint_match_pair(pair: dict[str, Any], sequence_label: str) -> dict[str, Any]:
    prefixed = dict(pair)
    for key in ("image_a", "image_b", "left_image", "right_image"):
        value = prefixed.get(key)
        if isinstance(value, str):
            prefixed[key] = f"{sequence_label}/{value}"
    for key in ("image_a_frame_id", "image_b_frame_id"):
        value = prefixed.get(key)
        if isinstance(value, str):
            prefixed[key] = f"{sequence_label}:{value}"
    return prefixed


def build_joint_match_and_pose_files(
    joint_config: dict[str, Any],
    runs_by_label: dict[str, dict[str, Any]],
    default_args: dict[str, Any],
    scene: str,
    output_dir: Path,
    run_id: str,
) -> tuple[Path, Path, list[str]]:
    source_labels = joint_config.get("source_labels")
    if not isinstance(source_labels, list) or not source_labels:
        raise ValueError("joint_camera_init.source_labels must be a non-empty list")

    pairs: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    used_labels: list[str] = []
    for raw_label in source_labels:
        label = str(raw_label)
        if label not in runs_by_label:
            raise ValueError(f"joint_camera_init source label not found in runs: {label}")
        run_args = runs_by_label[label].get("pipeline_args", {})
        if not isinstance(run_args, dict):
            raise ValueError(f"runs[{label}].pipeline_args must be a mapping")
        pipeline_args = build_pipeline_args(scene, default_args, run_args)
        pipeline_args.setdefault("output_timestamp", run_id)
        paths = predict_pipeline_paths(pipeline_args)
        if not paths.match_json.exists():
            raise FileNotFoundError(
                f"joint_camera_init source match file not found for {label}: {paths.match_json}"
            )
        with paths.match_json.open(encoding="utf-8") as f:
            match_data = json.load(f)
        sequence_label = sequence_label_for_frame_prefix(label)
        for pair in match_data.get("pairs", []):
            if isinstance(pair, dict):
                pairs.append(prefix_joint_match_pair(pair, sequence_label))

        if paths.left_img_dir is None:
            raise ValueError(f"joint_camera_init source needs left_img_dir: {label}")
        source_pose_path = output_dir / f"{sequence_label}_{run_id}_oxts_frame_poses.json"
        build_kitti_raw_oxts_frame_poses(paths.left_img_dir, source_pose_path)
        with source_pose_path.open(encoding="utf-8") as f:
            pose_data = json.load(f)
        for frame in pose_data.get("frames", []):
            if not isinstance(frame, dict):
                continue
            prefixed_frame = dict(frame)
            frame_id = prefixed_frame.get("frame_id")
            if isinstance(frame_id, str):
                prefixed_frame["frame_id"] = f"{sequence_label}:{frame_id}"
            frames.append(prefixed_frame)
        used_labels.append(label)

    joint_name = str(joint_config.get("result_prefix", "joint_camera_init"))
    match_path = output_dir / f"{joint_name}_{run_id}_matches_raw.json"
    poses_path = output_dir / f"{joint_name}_{run_id}_oxts_frame_poses.json"
    match_path.write_text(json.dumps({"pairs": pairs}, indent=2) + "\n", encoding="utf-8")
    poses_path.write_text(
        json.dumps(
            {
                "source": "run_experiment_joint_camera_init",
                "source_labels": used_labels,
                "frames": frames,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return match_path, poses_path, used_labels


def build_joint_ba_command(
    ba_bin: Path,
    match_path: Path,
    output_path: Path,
    init_param_file: Path,
    poses_path: Path,
    ba_args: dict[str, Any],
) -> list[str]:
    config = PipelineConfig.from_mapping(ba_args)
    cmd = [
        str(ba_bin),
        "--input",
        str(match_path),
        "--output",
        str(output_path),
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
        "--init_param_file",
        str(init_param_file),
        "--frame_poses_file",
        str(poses_path),
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
    for name, value in config.optional_ba_value_args().items():
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
        raise RuntimeError("joint_camera_init must not pass gt_param_file into C++ BA")
    return cmd


def run_joint_camera_init(
    experiment_config: dict[str, Any],
    scene: str,
    run_id: str,
    dry_run: bool,
) -> Path | None:
    joint_config = experiment_config.get("joint_camera_init")
    if not isinstance(joint_config, dict) or not bool(joint_config.get("enabled", False)):
        return None

    defaults = experiment_config.get("defaults", {})
    default_args = defaults.get("pipeline_args", {})
    if not isinstance(default_args, dict):
        raise ValueError("defaults.pipeline_args must be a mapping")
    runs = experiment_config.get("runs", [])
    runs_by_label = {
        str(run.get("label")): run
        for run in runs
        if isinstance(run, dict) and run.get("label") is not None
    }

    result_dir = to_project_path(default_args.get("result_dir", "stereo_calib/result"))
    output_dir = result_dir / "joint_camera_init"
    output_dir.mkdir(parents=True, exist_ok=True)
    match_path, poses_path, used_labels = build_joint_match_and_pose_files(
        joint_config,
        runs_by_label,
        default_args,
        scene,
        output_dir,
        run_id,
    )

    init_param_file = Path(str(joint_config.get("init_param_file") or default_args.get("init_param_file", "")))
    if not init_param_file.is_absolute():
        init_param_file = PROJECT_ROOT / init_param_file
    if not init_param_file.exists():
        raise FileNotFoundError(f"joint_camera_init init_param_file not found: {init_param_file}")

    result_prefix = str(joint_config.get("result_prefix", "joint_camera_init"))
    ba_result = output_dir / f"{result_prefix}_{run_id}_ba_result_raw.json"
    ba_args = dict(default_args)
    ba_args.update(joint_config.get("ba_args", {}))
    ba_args["dataset_mode"] = "kitti_raw_sequence"
    ba_args["init_param_file"] = str(init_param_file)
    ba_args["match_json"] = str(match_path)
    ba_args["frame_poses_file"] = str(poses_path)
    ba_args["result_prefix"] = result_prefix
    ba_args["output_timestamp"] = run_id
    ba_bin = PROJECT_ROOT / "build/bin/run_offline_stereo_ba_kitti"
    if not ba_bin.exists():
        raise FileNotFoundError(f"joint_camera_init BA executable not found: {ba_bin}")
    cmd = build_joint_ba_command(
        ba_bin,
        match_path,
        ba_result,
        init_param_file,
        poses_path,
        ba_args,
    )

    output_init = output_dir / f"{result_prefix}_{run_id}_init_params.txt"

    print("\n========== Joint Camera Init ==========", flush=True)
    print(
        "[Joint Camera Init] sources="
        + ", ".join(used_labels)
        + f", matches={match_path}, poses={poses_path}",
        flush=True,
    )
    run_command(cmd, dry_run)
    if dry_run:
        return output_init

    if not ba_result.exists():
        raise FileNotFoundError(f"joint_camera_init BA result missing: {ba_result}")

    camera_values = read_camera_values_from_result(ba_result)
    overlay_config = joint_config.get("overlay_tx_from_result")
    if isinstance(overlay_config, dict) and bool(overlay_config.get("enabled", False)):
        tx_ba_result = output_dir / f"{result_prefix}_{run_id}_tx_ba_result_raw.json"
        tx_ba_args = dict(ba_args)
        tx_ba_args.update(overlay_config.get("ba_args", {}))
        tx_cmd = build_joint_ba_command(
            ba_bin,
            match_path,
            tx_ba_result,
            init_param_file,
            poses_path,
            tx_ba_args,
        )
        print("\n========== Joint Camera TX Overlay ==========", flush=True)
        run_command(tx_cmd, dry_run)
        if not tx_ba_result.exists():
            raise FileNotFoundError(f"joint_camera_init TX BA result missing: {tx_ba_result}")
        tx_values = read_camera_values_from_result(tx_ba_result)
        camera_values = overlay_camera_values(
            camera_values,
            tx_values,
            [str(key) for key in overlay_config.get("fields", ["tx"])],
        )

    shared_fields = joint_config.get("shared_fields")
    if isinstance(shared_fields, list) and shared_fields:
        camera_values = {str(key): camera_values[str(key)] for key in shared_fields if str(key) in camera_values}
    metadata_comments = [
        "Generated by run_experiment.py joint_camera_init.",
        "Values come from BA result camera fields only; C++ BA is not given gt_param_file.",
        f"source_labels={','.join(used_labels)}",
    ]
    write_init_params_with_values(
        init_param_file,
        output_init,
        camera_values,
        metadata_comments=metadata_comments,
    )
    value_preview = ", ".join(
        f"{key}={value:.6g}" for key, value in sorted(camera_values.items())
    )
    print(f"[Joint Camera Init] wrote {output_init}: {value_preview}", flush=True)
    return output_init


def should_apply_joint_camera_init(
    joint_config: dict[str, Any],
    label: str,
) -> bool:
    if not bool(joint_config.get("apply_to_runs", True)):
        return False
    apply_to_labels = joint_config.get("apply_to_labels")
    if isinstance(apply_to_labels, list) and apply_to_labels:
        return label in {str(item) for item in apply_to_labels}
    return True


def apply_joint_camera_init_overrides(
    pipeline_args: dict[str, Any],
    joint_config: dict[str, Any],
    joint_init_file: Path,
    label: str,
) -> dict[str, Any]:
    if not should_apply_joint_camera_init(joint_config, label):
        return pipeline_args

    updated_args = dict(pipeline_args)
    updated_args["init_param_file"] = str(joint_init_file)
    if bool(joint_config.get("lock_focal_after_apply", True)):
        updated_args["fix_focal_length"] = True
    if bool(joint_config.get("lock_principal_point_after_apply", True)):
        updated_args["fix_principal_point"] = True
        updated_args["enable_incremental_free_principal_point_refine"] = False
        updated_args["free_principal_point_min_rmse_decrease"] = 1e9
        updated_args["free_principal_point_max_delta"] = 0.0
        updated_args["free_principal_point_release_focal_length"] = False
        updated_args["free_principal_point_max_focal_delta"] = 0.0
        updated_args["final_free_principal_point_min_rmse_decrease"] = 1e9
        updated_args["final_free_principal_point_max_delta"] = 0.0
        updated_args["final_free_principal_point_release_focal_length"] = False
        updated_args["final_free_principal_point_max_focal_delta"] = 0.0
    if bool(joint_config.get("lock_stereo_extrinsics_after_apply", True)):
        updated_args["enable_final_stereo_extrinsics_refine"] = False
        updated_args["optimize_stereo_tx_in_final_global_ba"] = False
    if bool(joint_config.get("disable_absolute_rotation_prior_after_apply", False)):
        updated_args["frame_absolute_rotation_prior"] = 0.0
    return updated_args


def apply_shared_camera_safety_overrides(
    pipeline_args: dict[str, Any],
    shared: dict[str, Any],
) -> dict[str, Any]:
    """Constrain weak-motion runs so intrinsics do not absorb pose degeneracy."""
    updated_args = dict(pipeline_args)
    if bool(shared.get("lock_intrinsics_after_apply", False)):
        updated_args["fix_focal_length"] = True
        updated_args["fix_principal_point"] = True
        updated_args["enable_incremental_free_principal_point_refine"] = False
        updated_args["free_principal_point_min_rmse_decrease"] = 1e9
        updated_args["free_principal_point_max_delta"] = 0.0
        updated_args["free_principal_point_release_focal_length"] = False
        updated_args["free_principal_point_max_focal_delta"] = 0.0
        if bool(shared.get("allow_final_principal_point_refine_after_apply", False)):
            updated_args["final_free_principal_point_min_rmse_decrease"] = float(
                shared.get("final_free_principal_point_min_rmse_decrease_after_apply", 0.003)
            )
            updated_args["final_free_principal_point_max_delta"] = float(
                shared.get("final_free_principal_point_max_delta_after_apply", 3.0)
            )
            updated_args["final_free_principal_point_release_focal_length"] = bool(
                shared.get("final_free_principal_point_release_focal_length_after_apply", False)
            )
            updated_args["final_free_principal_point_max_focal_delta"] = float(
                shared.get("final_free_principal_point_max_focal_delta_after_apply", 0.0)
            )
        else:
            updated_args["final_free_principal_point_min_rmse_decrease"] = 1e9
            updated_args["final_free_principal_point_max_delta"] = 0.0
            updated_args["final_free_principal_point_release_focal_length"] = False
            updated_args["final_free_principal_point_max_focal_delta"] = 0.0
    if bool(shared.get("disable_absolute_rotation_prior_after_apply", False)):
        updated_args["frame_absolute_rotation_prior"] = 0.0
    return updated_args


def maybe_apply_shared_camera_init(
    experiment_config: dict[str, Any],
    pipeline_args: dict[str, Any],
    label: str,
    consensus_camera_values: list[dict[str, float]],
    dry_run: bool,
    force_apply: bool = False,
) -> dict[str, Any]:
    """Use previous moving-drive camera estimates as init for weak-motion drives."""
    shared = get_shared_camera_config(experiment_config)
    if not shared or not shared.get("enabled", False):
        return pipeline_args

    min_motion_m = float(shared.get("min_motion_m", 0.5))
    min_consensus_runs = int(shared.get("min_consensus_runs", 2))
    apply_to_all_after_consensus = bool(shared.get("apply_to_all_after_consensus", False))
    motion_m = estimate_kitti_raw_motion_m(pipeline_args)
    is_weak_motion = motion_m is not None and motion_m < min_motion_m
    if not force_apply and not is_weak_motion and not apply_to_all_after_consensus:
        return pipeline_args
    if len(consensus_camera_values) < min_consensus_runs:
        if is_weak_motion:
            print(
                f"[Shared Camera Init] skipped for {label}: "
                f"motion={format_motion_m(motion_m)}, "
                f"consensus_runs={len(consensus_camera_values)}",
                flush=True,
            )
        return pipeline_args

    source_init = Path(str(pipeline_args.get("init_param_file", "")))
    if not source_init.is_absolute():
        source_init = PROJECT_ROOT / source_init
    if not source_init.exists():
        raise FileNotFoundError(f"Shared tx init source not found: {source_init}")

    result_dir = to_project_path(pipeline_args.get("result_dir", "stereo_calib/result"))
    run_id = str(pipeline_args.get("output_timestamp") or make_output_timestamp())
    output_init = result_dir / "shared_init" / f"{label}_{run_id}_init_params.txt"
    aggregation = str(shared.get("aggregation", "median")).lower()
    if aggregation not in {"mean", "median"}:
        raise ValueError(f"Unsupported shared_camera_init aggregation: {aggregation}")
    raw_aggregation_by_field = shared.get("aggregation_by_field")
    aggregation_by_field: dict[str, str] = {}
    if isinstance(raw_aggregation_by_field, dict):
        for key, value in raw_aggregation_by_field.items():
            field_method = str(value).lower()
            if field_method not in {"mean", "median"}:
                raise ValueError(
                    f"Unsupported shared_camera_init aggregation for {key}: {field_method}"
                )
            aggregation_by_field[str(key)] = field_method
    shared_values = filter_shared_camera_values(
        aggregate_shared_camera_values(
            consensus_camera_values,
            aggregation,
            aggregation_by_field,
            keys=shared_camera_value_keys(shared),
            min_samples_by_key=shared_min_samples_by_key(shared),
        ),
        shared,
    )
    metadata_comments = [
        "Generated by run_experiment.py shared_camera_init.",
        "Values come from previous BA result camera fields only, not diff_vs_gt or gt_param_file.",
        f"aggregation={aggregation}, source_ba_runs={len(consensus_camera_values)}",
    ]
    if dry_run:
        value_preview = ", ".join(
            f"{key}={value:.6g}" for key, value in sorted(shared_values.items())
        )
        print(
            f"[Shared Camera Init] dry-run for {label}: "
            f"motion={format_motion_m(motion_m)}, {value_preview}, output={output_init}",
            flush=True,
        )
    else:
        write_init_params_with_values(
            source_init,
            output_init,
            shared_values,
            metadata_comments=metadata_comments,
        )
        value_preview = ", ".join(
            f"{key}={value:.6g}" for key, value in sorted(shared_values.items())
        )
        print(
            f"[Shared Camera Init] applied for {label}: "
            f"motion={format_motion_m(motion_m)}, {value_preview}, init={output_init}",
            flush=True,
        )

    updated_args = dict(pipeline_args)
    updated_args["init_param_file"] = str(output_init)
    if bool(shared.get("disable_final_stereo_refine_after_apply", False)):
        updated_args["enable_final_stereo_extrinsics_refine"] = False
    if bool(shared.get("disable_frame_distance_prior_after_apply", False)):
        updated_args["frame_distance_prior"] = 0.0
    if bool(shared.get("fix_external_frame_poses_after_apply", False)):
        updated_args["fix_external_frame_poses"] = True
    if bool(shared.get("fix_external_frame_rotations_after_apply", False)):
        updated_args["fix_external_frame_rotations"] = True
    if bool(shared.get("fix_external_frame_translations_after_apply", False)):
        updated_args["fix_external_frame_translations"] = True
    updated_args = apply_shared_camera_safety_overrides(updated_args, shared)
    return updated_args


def maybe_collect_shared_camera(
    experiment_config: dict[str, Any],
    pipeline_args: dict[str, Any],
    paths: PipelinePaths,
    label: str,
    consensus_camera_values: list[dict[str, float]],
    dry_run: bool,
) -> dict[str, float] | None:
    """Collect reliable camera estimates from moving drives for later weak drives."""
    if dry_run:
        return None
    shared = get_shared_camera_config(experiment_config)
    if not shared or not shared.get("enabled", False):
        return None
    source_labels = shared.get("source_labels")
    if isinstance(source_labels, list) and source_labels:
        allowed_labels = {str(item) for item in source_labels}
        if label not in allowed_labels:
            return None

    motion_m = estimate_kitti_raw_motion_m(pipeline_args)
    min_motion_m = float(shared.get("min_motion_m", 0.5))
    max_rmse = shared.get("max_reproj_error")
    require_final_refine = bool(shared.get("require_final_stereo_refine", True))
    if motion_m is None or motion_m < min_motion_m:
        return None

    ba_stats = read_ba_stats(paths.ba_result_json)
    final_rmse = ba_stats.get("final_reproj_error")
    if max_rmse is not None and (final_rmse is None or float(final_rmse) > float(max_rmse)):
        return None
    if require_final_refine:
        stage = read_ba_final_stage(paths.ba_result_json)
        if stage != "Final Stereo Extrinsics Refine":
            return None

    assert_ba_result_is_safe_for_init(paths.ba_result_json)
    camera_values = read_ba_camera_init_values(paths.ba_result_json)
    if camera_values is None:
        return None
    camera_values = filter_shared_camera_values(camera_values, shared)
    camera_values = apply_shared_field_source_stages(
        camera_values,
        shared,
        paths.ba_result_json,
    )
    camera_values = apply_shared_field_stage_filters(
        camera_values,
        shared,
        paths.ba_result_json,
    )
    if not camera_values:
        return None
    consensus_camera_values.append(camera_values)
    preview_keys = ("left_fx", "left_cx", "left_cy", "tx")
    value_preview = ", ".join(
        f"{key}={camera_values[key]:.6g}" for key in preview_keys if key in camera_values
    )
    print(
        f"[Shared Camera Init] collected from {label}: "
        f"motion={motion_m:.3f} m, {value_preview}, "
        f"consensus_runs={len(consensus_camera_values)}",
        flush=True,
    )
    return camera_values


def should_rerun_all_with_shared_camera(
    experiment_config: dict[str, Any],
    consensus_camera_values: list[dict[str, float]],
) -> bool:
    shared = get_shared_camera_config(experiment_config)
    if not shared or not shared.get("enabled", False):
        return False
    if not bool(shared.get("rerun_all_after_consensus", False)):
        return False
    min_consensus_runs = int(shared.get("min_consensus_runs", 2))
    return len(consensus_camera_values) >= min_consensus_runs


def should_run_source_only_shared_camera_source(
    experiment_config: dict[str, Any],
    run_config: dict[str, Any],
) -> bool:
    """Run source_only entries only when they are declared shared camera sources."""
    if not bool(run_config.get("source_only", False)):
        return False
    shared = get_shared_camera_config(experiment_config)
    if not shared or not shared.get("enabled", False):
        return False

    label = str(run_config.get("label", "run"))
    source_labels = shared.get("source_labels")
    if isinstance(source_labels, list) and source_labels:
        return label in {str(item) for item in source_labels}
    return True


def make_shared_camera_refined_run_config(
    run_config: dict[str, Any],
    source_paths: PipelinePaths,
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    """Build a second-pass run that reuses matches but writes separate BA outputs."""
    shared = get_shared_camera_config(experiment_config)
    label_suffix = str(shared.get("refined_label_suffix", "_refined"))
    result_prefix_suffix = str(shared.get("refined_result_prefix_suffix", "_refined"))
    label = str(run_config.get("label", "run"))
    run_args = run_config.get("pipeline_args", {})
    if not isinstance(run_args, dict):
        raise ValueError("runs[].pipeline_args must be a mapping")

    refined_args = dict(run_args)
    result_prefix = str(refined_args.get("result_prefix") or refined_args.get("scene") or label)
    refined_args["result_prefix"] = f"{result_prefix}{result_prefix_suffix}"
    refined_args["match_json"] = str(source_paths.match_json)
    refined_args["skip_match"] = True

    return {
        **run_config,
        "label": f"{label}{label_suffix}",
        "pipeline_args": refined_args,
    }


def make_shared_camera_seed_run_config(
    run_config: dict[str, Any],
    source_paths: PipelinePaths,
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    """Build the no-GT camera seed refinement run from a first-pass source."""
    shared = get_shared_camera_config(experiment_config)
    seed_config = shared.get("camera_seed_refine")
    if not isinstance(seed_config, dict):
        seed_config = {}
    label_suffix = str(seed_config.get("label_suffix", "_camera_seed"))
    result_prefix_suffix = str(seed_config.get("result_prefix_suffix", "_camera_seed"))
    label = str(run_config.get("label", "run"))
    run_args = run_config.get("pipeline_args", {})
    if not isinstance(run_args, dict):
        raise ValueError("runs[].pipeline_args must be a mapping")

    seed_args = dict(run_args)
    result_prefix = str(seed_args.get("result_prefix") or seed_args.get("scene") or label)
    seed_args["result_prefix"] = f"{result_prefix}{result_prefix_suffix}"
    seed_args["match_json"] = str(source_paths.match_json)
    seed_args["skip_match"] = True

    return {
        **run_config,
        "label": f"{label}{label_suffix}",
        "pipeline_args": seed_args,
    }


def make_shared_camera_seeded_final_run_config(
    run_config: dict[str, Any],
    source_paths: PipelinePaths,
    seed_init_file: Path,
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    """Build a final pass that uses the refined camera seed as init."""
    shared = get_shared_camera_config(experiment_config)
    seed_config = shared.get("camera_seed_refine")
    if not isinstance(seed_config, dict):
        seed_config = {}
    label_suffix = str(seed_config.get("final_label_suffix", "_refined"))
    result_prefix_suffix = str(seed_config.get("final_result_prefix_suffix", "_refined"))
    final_overrides = seed_config.get("final_pipeline_args")
    if not isinstance(final_overrides, dict):
        final_overrides = {}

    label = str(run_config.get("label", "run"))
    run_args = run_config.get("pipeline_args", {})
    if not isinstance(run_args, dict):
        raise ValueError("runs[].pipeline_args must be a mapping")

    final_args = dict(run_args)
    result_prefix = str(final_args.get("result_prefix") or final_args.get("scene") or label)
    final_args["result_prefix"] = f"{result_prefix}{result_prefix_suffix}"
    final_args["match_json"] = str(source_paths.match_json)
    final_args["skip_match"] = True
    final_args["init_param_file"] = str(seed_init_file)
    final_args.update(final_overrides)

    return {
        **run_config,
        "label": f"{label}{label_suffix}",
        "pipeline_args": final_args,
    }


def write_shared_camera_seed_init_file(
    experiment_config: dict[str, Any],
    pipeline_args: dict[str, Any],
    seed_result_json: Path,
    run_id: str,
) -> Path:
    """Persist a pure init-param file from a no-GT seed BA result."""
    shared = get_shared_camera_config(experiment_config)
    seed_values = filter_shared_camera_values(
        read_camera_values_from_result(seed_result_json),
        shared,
    )

    source_init = Path(str(pipeline_args.get("init_param_file", "")))
    if not source_init.is_absolute():
        source_init = PROJECT_ROOT / source_init
    if not source_init.exists():
        raise FileNotFoundError(f"Shared camera seed source init not found: {source_init}")

    result_dir = to_project_path(pipeline_args.get("result_dir", "stereo_calib/result"))
    output_init = result_dir / "shared_init" / f"camera_seed_{run_id}_init_params.txt"
    metadata_comments = [
        "Generated by run_experiment.py camera_seed_refine.",
        "Values come from the seed BA result camera fields only, not diff_vs_gt or gt_param_file.",
        f"seed_ba_result={relative_output_path(seed_result_json)}",
    ]
    write_init_params_with_values(
        source_init,
        output_init,
        seed_values,
        metadata_comments=metadata_comments,
    )
    value_preview = ", ".join(
        f"{key}={value:.6g}" for key, value in sorted(seed_values.items())
    )
    print(f"[Camera Seed] wrote {output_init}: {value_preview}", flush=True)
    return output_init


def get_camera_seed_refine_config(experiment_config: dict[str, Any]) -> dict[str, Any]:
    shared = get_shared_camera_config(experiment_config)
    seed_config = shared.get("camera_seed_refine") if shared else None
    return seed_config if isinstance(seed_config, dict) else {}


def should_run_camera_seed_refine(
    experiment_config: dict[str, Any],
    consensus_camera_values: list[dict[str, float]],
) -> bool:
    seed_config = get_camera_seed_refine_config(experiment_config)
    if not bool(seed_config.get("enabled", False)):
        return False
    shared = get_shared_camera_config(experiment_config)
    min_consensus_runs = int(shared.get("min_consensus_runs", 2))
    return len(consensus_camera_values) >= min_consensus_runs


def consensus_values_for_refined_run(
    all_consensus_values: list[dict[str, float]],
    source_value_by_label: dict[str, dict[str, float]],
    base_label: str,
    shared: dict[str, Any],
) -> list[dict[str, float]]:
    """Optionally exclude a run's own first-pass camera from its refined init."""
    if not bool(shared.get("leave_one_out_refined_sources", False)):
        return all_consensus_values
    own_values = source_value_by_label.get(base_label)
    if own_values is None:
        return all_consensus_values
    return [values for values in all_consensus_values if values is not own_values]


def count_png_images(path_value: Any) -> int | None:
    """Count PNG files in an image directory; return None when not applicable."""
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    return len(list(path.glob("*.png")))


def classify_match_pair(pair: dict[str, Any]) -> str:
    """Classify a match pair as LR/LL/RR/RL using KITTI raw metadata when present."""
    a_is_left = pair.get("image_a_is_left")
    b_is_left = pair.get("image_b_is_left")
    if isinstance(a_is_left, bool) and isinstance(b_is_left, bool):
        return ("L" if a_is_left else "R") + ("L" if b_is_left else "R")

    image_a = str(pair.get("image_a", pair.get("left_image", "")))
    image_b = str(pair.get("image_b", pair.get("right_image", "")))
    a_side = "L" if "image_00" in image_a or image_a.startswith("left") else "R"
    b_side = "L" if "image_00" in image_b or image_b.startswith("left") else "R"
    return a_side + b_side


def read_match_stats(match_json: Path) -> dict[str, Any]:
    """Read pair/match counts from a matches JSON file."""
    stats: dict[str, Any] = {
        "match_pair_count": None,
        "total_match_count": None,
        "lr_pair_count": None,
        "ll_pair_count": None,
        "rr_pair_count": None,
    }
    if not match_json.exists():
        return stats

    with match_json.open(encoding="utf-8") as f:
        data = json.load(f)
    pairs = data.get("pairs", [])
    pair_type_counts: dict[str, int] = {}
    total_matches = 0
    for pair in pairs:
        ptype = classify_match_pair(pair)
        pair_type_counts[ptype] = pair_type_counts.get(ptype, 0) + 1
        total_matches += len(pair.get("matches", []))

    stats.update(
        {
            "match_pair_count": len(pairs),
            "total_match_count": total_matches,
            "lr_pair_count": pair_type_counts.get("LR", 0),
            "ll_pair_count": pair_type_counts.get("LL", 0),
            "rr_pair_count": pair_type_counts.get("RR", 0),
        }
    )
    return stats


def read_ba_stats(ba_result_json: Path) -> dict[str, Any]:
    """Read compact quality/scale stats from a BA result JSON file."""
    stats: dict[str, Any] = {
        "ba_success": None,
        "final_reproj_error": None,
        "num_frames": None,
        "num_tracks": None,
        "num_observations": None,
    }
    if not ba_result_json.exists():
        return stats

    with ba_result_json.open(encoding="utf-8") as f:
        data = json.load(f)
    stats.update(
        {
            "ba_success": data.get("success"),
            "final_reproj_error": data.get("final_reproj_error"),
            "num_frames": data.get("num_frames"),
            "num_tracks": data.get("num_tracks"),
            "num_observations": data.get("num_observations"),
        }
    )
    return stats


def safe_divide(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def rounded(value: Any, digits: int = 3) -> Any:
    """Round floats for human-facing JSON without changing other values."""
    if isinstance(value, float):
        return round(value, digits)
    return value


def relative_output_path(path_value: Any) -> str | None:
    """Make output paths shorter in experiment summaries."""
    if not path_value:
        return None
    path = Path(str(path_value))
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_run_summary_record(
    scene: str,
    label: str,
    pipeline_args: dict[str, Any],
    paths: PipelinePaths,
    elapsed_seconds: float,
    dry_run: bool,
) -> dict[str, Any]:
    """Create one run-level timing/count record for comparison."""
    left_images = count_png_images(pipeline_args.get("left_img_dir"))
    right_images = count_png_images(pipeline_args.get("right_img_dir"))
    stereo_image_pairs = min(left_images, right_images) if left_images is not None and right_images is not None else None

    record: dict[str, Any] = {
        "scene": scene,
        "label": label,
        "result_prefix": pipeline_args.get("result_prefix", pipeline_args.get("scene")),
        "dataset_mode": pipeline_args.get("dataset_mode"),
        "left_image_count": left_images,
        "right_image_count": right_images,
        "stereo_image_pair_count": stereo_image_pairs,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_hms": format_elapsed_time(elapsed_seconds),
        "match_json": str(paths.match_json),
        "ba_result_json": str(paths.ba_result_json),
    }

    if not dry_run:
        record.update(read_match_stats(paths.match_json))
        record.update(read_ba_stats(paths.ba_result_json))
    else:
        record.update(
            {
                "match_pair_count": None,
                "total_match_count": None,
                "lr_pair_count": None,
                "ll_pair_count": None,
                "rr_pair_count": None,
                "ba_success": None,
                "final_reproj_error": None,
                "num_frames": None,
                "num_tracks": None,
                "num_observations": None,
            }
        )

    record["seconds_per_match_pair"] = safe_divide(record["elapsed_seconds"], record["match_pair_count"])
    record["seconds_per_stereo_pair"] = safe_divide(record["elapsed_seconds"], record["stereo_image_pair_count"])
    record["seconds_per_match"] = safe_divide(record["elapsed_seconds"], record["total_match_count"])
    return record


def print_run_summary_record(record: dict[str, Any]) -> None:
    """Print a compact one-line summary after each run."""
    print(
        "Run summary: "
        f"stereo_pairs={record.get('stereo_image_pair_count') or '-'} "
        f"match_pairs={record.get('match_pair_count') or '-'} "
        f"matches={record.get('total_match_count') or '-'} "
        f"elapsed={record['elapsed_hms']} "
        f"per_match_pair={format_seconds(record.get('seconds_per_match_pair'))} "
        f"per_stereo_pair={format_seconds(record.get('seconds_per_stereo_pair'))}",
        flush=True,
    )


def write_experiment_summary(
    experiment_name: str,
    records: list[dict[str, Any]],
    total_elapsed_seconds: float,
    result_dir: str | None = None,
    run_id: str | None = None,
) -> Path:
    """Persist a readable run timing/count summary as JSON."""
    base_result_dir = to_project_path(result_dir) if result_dir else PROJECT_ROOT / "stereo_calib/result"
    output_dir = base_result_dir / "experiment_summaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_timestamp = run_id or make_output_timestamp()
    json_path = output_dir / f"{experiment_name}_{summary_timestamp}_summary.json"

    total_stereo_pairs = sum(record.get("stereo_image_pair_count") or 0 for record in records)
    total_match_pairs = sum(record.get("match_pair_count") or 0 for record in records)
    total_matches = sum(record.get("total_match_count") or 0 for record in records)

    readable_runs = []
    for record in records:
        readable_runs.append(
            {
                "label": record.get("label"),
                "scene": record.get("result_prefix"),
                "batch_scene": record.get("scene"),
                "dataset_mode": record.get("dataset_mode"),
                "time": {
                    "elapsed": record.get("elapsed_hms"),
                    "seconds": rounded(record.get("elapsed_seconds")),
                    "seconds_per_stereo_frame_pair": rounded(record.get("seconds_per_stereo_pair")),
                    "seconds_per_matched_image_pair": rounded(record.get("seconds_per_match_pair")),
                    "seconds_per_match_point": rounded(record.get("seconds_per_match"), 6),
                },
                "images": {
                    "left": record.get("left_image_count"),
                    "right": record.get("right_image_count"),
                    "stereo_frame_pairs": record.get("stereo_image_pair_count"),
                },
                "matched_image_pairs": {
                    "total": record.get("match_pair_count"),
                    "LR_same_frame": record.get("lr_pair_count"),
                    "LL_adjacent": record.get("ll_pair_count"),
                    "RR_adjacent": record.get("rr_pair_count"),
                },
                "matches": {
                    "total_points": record.get("total_match_count"),
                },
                "ba": {
                    "success": record.get("ba_success"),
                    "final_reproj_error_px": rounded(record.get("final_reproj_error"), 6),
                    "frames": record.get("num_frames"),
                    "tracks": record.get("num_tracks"),
                    "observations": record.get("num_observations"),
                },
                "outputs": {
                    "matches": relative_output_path(record.get("match_json")),
                    "ba_result": relative_output_path(record.get("ba_result_json")),
                },
            }
        )

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": experiment_name,
                "run_id": run_id,
                "total": {
                    "elapsed": format_elapsed_time(total_elapsed_seconds),
                    "seconds": rounded(total_elapsed_seconds),
                    "run_count": len(records),
                    "stereo_frame_pairs": total_stereo_pairs,
                    "matched_image_pairs": total_match_pairs,
                    "match_points": total_matches,
                    "seconds_per_stereo_frame_pair": rounded(
                        safe_divide(total_elapsed_seconds, total_stereo_pairs)
                    ),
                    "seconds_per_matched_image_pair": rounded(
                        safe_divide(total_elapsed_seconds, total_match_pairs)
                    ),
                    "seconds_per_match_point": rounded(
                        safe_divide(total_elapsed_seconds, total_matches),
                        6,
                    ),
                },
                "runs": readable_runs,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return json_path


def run_build(config: dict[str, Any], dry_run: bool) -> None:
    """执行配置里的 CMake 编译步骤。"""
    build = config.get("build", {})
    if not build or not build.get("enabled", False):
        return

    source_dir = to_project_path(build.get("source_dir", "stereo_calib"))
    build_dir = to_project_path(build.get("build_dir", "build"))
    jobs = build.get("jobs", os.cpu_count() or 1)

    print("\n========== Build ==========", flush=True)
    run_command(["cmake", "-S", str(source_dir), "-B", str(build_dir)], dry_run)
    run_command(["cmake", "--build", str(build_dir), f"-j{jobs}"], dry_run)


def build_pipeline_args(
    scene: str,
    default_args: dict[str, Any],
    run_args: dict[str, Any] | None,
) -> dict[str, Any]:
    """合并 scene、全局默认参数和单次 run 的 pipeline 参数。"""
    args = merge_dicts(default_args, run_args)
    args = {"scene": scene, **args}
    return args


def predict_pipeline_paths(pipeline_args: dict[str, Any]) -> PipelinePaths:
    """用和 run_pipeline.py 相同的规则提前算出本次输出路径。"""
    # 只做路径预测，不生成 KITTI GT 文件；这样 dry-run 不会改动 result 目录。
    return resolve_paths(PipelineConfig.from_mapping(pipeline_args))


def index_pipeline_output(
    output_index: dict[str, Path],
    label: str,
    pipeline_args: dict[str, Any],
    paths: PipelinePaths,
) -> None:
    """记录本次 run 的 BA 输出，供后处理通过 label/pixel_tag 查找。"""
    output_index[label] = paths.ba_result_json
    pixel_tag = pipeline_args.get("pixel_tag")
    if pixel_tag:
        output_index[str(pixel_tag)] = paths.ba_result_json
        output_index[f"le_{pixel_tag}"] = paths.ba_result_json


def run_pipeline_for_scene(
    scene: str,
    run_config: dict[str, Any],
    config: dict[str, Any],
    dry_run: bool,
    consensus_camera_values: list[dict[str, float]] | None = None,
    force_shared_camera_init: bool = False,
    joint_init_file: Path | None = None,
    skip_shared_camera_init: bool = False,
    post_shared_pipeline_args: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], PipelinePaths, float]:
    """针对一个 scene/run 组合执行一次 run_pipeline.py。"""
    runner = config.get("runner", {})
    defaults = config.get("defaults", {})
    default_args = defaults.get("pipeline_args", {})
    if not isinstance(default_args, dict):
        raise ValueError("defaults.pipeline_args must be a mapping")

    run_args = run_config.get("pipeline_args", {})
    if not isinstance(run_args, dict):
        raise ValueError("runs[].pipeline_args must be a mapping")

    label = run_config.get("label", "run")
    pipeline_script = str(runner.get("pipeline_script", "run_pipeline.py"))
    pipeline_args = build_pipeline_args(scene, default_args, run_args)
    pipeline_args.setdefault("output_timestamp", make_output_timestamp())
    if joint_init_file is not None:
        joint_config = config.get("joint_camera_init")
        if isinstance(joint_config, dict):
            pipeline_args = apply_joint_camera_init_overrides(
                pipeline_args,
                joint_config,
                joint_init_file,
                str(label),
            )
    if not skip_shared_camera_init:
        pipeline_args = maybe_apply_shared_camera_init(
            config,
            pipeline_args,
            str(label),
            consensus_camera_values if consensus_camera_values is not None else [],
            dry_run,
            force_apply=force_shared_camera_init,
        )
    if post_shared_pipeline_args:
        pipeline_args.update(post_shared_pipeline_args)
    paths = predict_pipeline_paths(pipeline_args)

    cmd = build_python_command(runner, pipeline_script)
    cmd = append_cli_args(cmd, pipeline_args)

    print(f"\n========== Scene: {scene} | Run: {label} ==========", flush=True)
    started = time.monotonic()
    run_command(cmd, dry_run)
    elapsed_seconds = time.monotonic() - started
    print(f"Elapsed: {format_elapsed_time(elapsed_seconds)}", flush=True)
    return str(label), pipeline_args, paths, elapsed_seconds


def resolve_postprocess_input(
    input_key: str,
    template: str,
    scene: str,
    output_index: dict[str, Path],
) -> str:
    """优先使用本次实验刚生成的输出；找不到时回退到 YAML 模板。"""
    selector = input_key.removeprefix("input_")
    for candidate in (input_key, selector, f"le_{selector}"):
        if candidate in output_index:
            return str(output_index[candidate])
    return str(template).format(scene=scene)


def run_threshold_comparison(
    scene: str,
    postprocess: dict[str, Any],
    runner: dict[str, Any],
    output_index: dict[str, Path],
    dry_run: bool,
) -> None:
    """为单个 scene 执行可选的 3px/2px/1px 对比图后处理。"""
    comparison = postprocess.get("threshold_comparison", {})
    if not comparison or not comparison.get("enabled", False):
        return

    script = str(comparison["script"])
    inputs = comparison.get("inputs", {})
    output = str(comparison["output"]).format(scene=scene, timestamp=make_output_timestamp())

    cmd = build_python_command(runner, script)
    cmd.extend(["--scene", scene])
    for key, template in inputs.items():
        flag_name = key.replace("_", "-")
        cmd.extend([f"--{flag_name}", resolve_postprocess_input(key, str(template), scene, output_index)])
    cmd.extend(["--output", output])

    print(f"\n========== Scene: {scene} | Postprocess ==========", flush=True)
    run_command(cmd, dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a batch experiment described by a YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/experiments/panoramic_threshold_sweep.yaml",
        help="Experiment YAML config path",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--no-build", action="store_true", help="Skip the build step from the config")
    parser.add_argument("--conda-env", default=None, help="Override runner.conda_env from the config")
    parser.add_argument("--result-dir", default=None, help="Override pipeline result_dir for every run")
    parser.add_argument("--match-output-tag", default=None, help="Use a stable matches filename tag for every run")
    parser.add_argument("--run-id", default=None, help="Shared timestamp/id for outputs from this experiment run")
    parser.add_argument("--scene", action="append", help="Run only the selected scene; can be repeated")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    run_records: list[dict[str, Any]] = []
    experiment_name = Path(args.config).stem

    try:
        config = load_config(to_project_path(args.config))
        if args.no_build:
            config.setdefault("build", {})["enabled"] = False
        if args.conda_env:
            config.setdefault("runner", {})["conda_env"] = args.conda_env
        if args.result_dir:
            config.setdefault("defaults", {}).setdefault("pipeline_args", {})["result_dir"] = args.result_dir
        if args.match_output_tag:
            config.setdefault("defaults", {}).setdefault("pipeline_args", {})["match_output_tag"] = args.match_output_tag
        if args.run_id:
            config.setdefault("defaults", {}).setdefault("pipeline_args", {})["output_timestamp"] = args.run_id
        else:
            config.setdefault("defaults", {}).setdefault("pipeline_args", {}).setdefault(
                "output_timestamp",
                make_output_timestamp(),
            )

        scenes = args.scene or config.get("scenes", [])
        runs = config.get("runs", [])
        if not scenes:
            raise ValueError("No scenes configured")
        if not runs:
            raise ValueError("No runs configured")

        experiment_name = str(config.get("name", Path(args.config).stem))
        print(f"Experiment: {experiment_name}", flush=True)
        run_build(config, args.dry_run)

        for scene in scenes:
            output_index: dict[str, Path] = {}
            consensus_camera_values: list[dict[str, float]] = []
            source_value_by_label: dict[str, dict[str, float]] = {}
            first_pass_outputs: list[tuple[dict[str, Any], PipelinePaths]] = []
            run_id = str(
                config.get("defaults", {})
                .get("pipeline_args", {})
                .get("output_timestamp")
                or args.run_id
                or make_output_timestamp()
            )
            joint_init_file = run_joint_camera_init(config, str(scene), run_id, args.dry_run)

            source_only_runs = [
                run_config
                for run_config in runs
                if should_run_source_only_shared_camera_source(config, run_config)
            ]
            if source_only_runs:
                print(
                    "\n========== Shared Camera Source-Only Pass ==========",
                    flush=True,
                )
            for run_config in source_only_runs:
                label, pipeline_args, paths, elapsed_seconds = run_pipeline_for_scene(
                    str(scene),
                    run_config,
                    config,
                    args.dry_run,
                    consensus_camera_values,
                    joint_init_file=joint_init_file,
                )
                index_pipeline_output(output_index, label, pipeline_args, paths)
                record = build_run_summary_record(
                    str(scene),
                    label,
                    pipeline_args,
                    paths,
                    elapsed_seconds,
                    args.dry_run,
                )
                run_records.append(record)
                print_run_summary_record(record)
                collected_camera_values = maybe_collect_shared_camera(
                    config,
                    pipeline_args,
                    paths,
                    label,
                    consensus_camera_values,
                    args.dry_run,
                )
                if collected_camera_values is not None:
                    source_value_by_label[label] = collected_camera_values

            for run_config in runs:
                if bool(run_config.get("source_only", False)):
                    continue
                label, pipeline_args, paths, elapsed_seconds = run_pipeline_for_scene(
                    str(scene),
                    run_config,
                    config,
                    args.dry_run,
                    consensus_camera_values,
                    joint_init_file=joint_init_file,
                )
                index_pipeline_output(output_index, label, pipeline_args, paths)
                record = build_run_summary_record(
                    str(scene),
                    label,
                    pipeline_args,
                    paths,
                    elapsed_seconds,
                    args.dry_run,
                )
                run_records.append(record)
                print_run_summary_record(record)
                first_pass_outputs.append((run_config, paths))
                collected_camera_values = maybe_collect_shared_camera(
                    config,
                    pipeline_args,
                    paths,
                    label,
                    consensus_camera_values,
                    args.dry_run,
                )
                if collected_camera_values is not None:
                    source_value_by_label[label] = collected_camera_values

            if should_rerun_all_with_shared_camera(config, consensus_camera_values):
                print(
                    "\n========== Shared Camera Refined Pass ==========",
                    flush=True,
                )
                shared = get_shared_camera_config(config)
                seed_config = get_camera_seed_refine_config(config)
                seed_init_file: Path | None = None
                if should_run_camera_seed_refine(config, consensus_camera_values):
                    source_label = str(seed_config.get("source_label", ""))
                    source_entry = next(
                        (
                            (run_config, paths)
                            for run_config, paths in first_pass_outputs
                            if str(run_config.get("label", "run")) == source_label
                        ),
                        None,
                    )
                    if source_entry is None:
                        raise ValueError(
                            "shared_camera_init.camera_seed_refine.source_label "
                            f"not found in first-pass outputs: {source_label}"
                        )
                    source_run_config, source_paths = source_entry
                    seed_run_config = make_shared_camera_seed_run_config(
                        source_run_config,
                        source_paths,
                        config,
                    )
                    seed_pipeline_args = seed_config.get("pipeline_args")
                    if not isinstance(seed_pipeline_args, dict):
                        seed_pipeline_args = {}
                    label, pipeline_args, paths, elapsed_seconds = run_pipeline_for_scene(
                        str(scene),
                        seed_run_config,
                        config,
                        args.dry_run,
                        consensus_camera_values,
                        force_shared_camera_init=True,
                        joint_init_file=joint_init_file,
                        post_shared_pipeline_args=seed_pipeline_args,
                    )
                    index_pipeline_output(output_index, label, pipeline_args, paths)
                    record = build_run_summary_record(
                        str(scene),
                        label,
                        pipeline_args,
                        paths,
                        elapsed_seconds,
                        args.dry_run,
                    )
                    run_records.append(record)
                    print_run_summary_record(record)
                    if not args.dry_run:
                        seed_init_file = write_shared_camera_seed_init_file(
                            config,
                            pipeline_args,
                            paths.ba_result_json,
                            run_id,
                        )

                for run_config, first_pass_paths in first_pass_outputs:
                    base_label = str(run_config.get("label", "run"))
                    refined_consensus_values = consensus_values_for_refined_run(
                        consensus_camera_values,
                        source_value_by_label,
                        base_label,
                        shared,
                    )
                    if seed_init_file is not None:
                        refined_run_config = make_shared_camera_seeded_final_run_config(
                            run_config,
                            first_pass_paths,
                            seed_init_file,
                            config,
                        )
                        force_shared_init = False
                        skip_shared_init = True
                    else:
                        refined_run_config = make_shared_camera_refined_run_config(
                            run_config,
                            first_pass_paths,
                            config,
                        )
                        force_shared_init = True
                        skip_shared_init = False
                    label, pipeline_args, paths, elapsed_seconds = run_pipeline_for_scene(
                        str(scene),
                        refined_run_config,
                        config,
                        args.dry_run,
                        refined_consensus_values,
                        force_shared_camera_init=force_shared_init,
                        joint_init_file=joint_init_file,
                        skip_shared_camera_init=skip_shared_init,
                    )
                    index_pipeline_output(output_index, label, pipeline_args, paths)
                    record = build_run_summary_record(
                        str(scene),
                        label,
                        pipeline_args,
                        paths,
                        elapsed_seconds,
                        args.dry_run,
                    )
                    run_records.append(record)
                    print_run_summary_record(record)
            run_threshold_comparison(
                str(scene),
                config.get("postprocess", {}),
                config.get("runner", {}),
                output_index,
                args.dry_run,
            )

    except Exception as exc:
        print(f"\nExperiment failed: {exc}", file=sys.stderr)
        return 1
    finally:
        total_elapsed_seconds = time.monotonic() - started
        if run_records and not args.dry_run:
            json_path = write_experiment_summary(
                experiment_name,
                run_records,
                total_elapsed_seconds,
                args.result_dir,
                args.run_id,
            )
            print(f"\nExperiment summary JSON: {json_path}", flush=True)
        print(f"\nTotal elapsed: {format_elapsed_time(total_elapsed_seconds)}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

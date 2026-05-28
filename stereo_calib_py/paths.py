"""Path resolution for single-run stereo calibration pipeline jobs.

本模块只负责“算路径”。像 KITTI 标定转换这种会写文件的副作用放在
`prepare_ground_truth()`，这样 dry-run 和测试可以安全地预测路径。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import PipelineConfig
from .kitti import build_kitti_gt_camera_json
from .project import PROJECT_ROOT


def format_tag_value(value: object) -> str:
    """Format numeric values the same way the old shell scripts named files."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def make_output_timestamp() -> str:
    """Create a compact timestamp for result filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


@dataclass(frozen=True)
class CacheInputPaths:
    """Existing files selected when a step is skipped.

    `--skip_match` 优先复用 timestamp 新规则下最新的历史 matches；
    没有候选时才保留固定命名路径用于报错提示。`--skip_ba` 仍读固定缓存名。
    """

    match_json: Path | None = None
    match_json_candidates: tuple[Path, ...] = ()
    ba_result_json: Path | None = None


@dataclass(frozen=True)
class NewOutputPaths:
    """Fresh outputs for a full run with the current timestamped naming.

    正常新运行写这里，避免覆盖历史结果。
    """

    match_json: Path
    ba_result_json: Path
    plot_png: Path


@dataclass(frozen=True)
class GroundTruthPaths:
    """Ground-truth camera file and optional source needed to generate it.

    显式传入的 GT 文件只记录路径；KITTI 自动转换需要额外记录官方标定目录。
    """

    file: Path | None = None
    explicit: bool = False
    kitti_calib_dir: Path | None = None
    kitti_result_prefix: str | None = None


@dataclass(frozen=True)
class PipelinePaths:
    """Resolved paths for a single pipeline run.

    `cache_input_paths` and `new_output_paths` make the intent explicit, while
    `match_json` / `ba_result_json` / `plot_png` remain the effective paths used
    by the rest of the pipeline.
    """

    dataset_mode: str
    scene: str
    result_prefix: str
    img_dir: Path
    matcher_input_dir: Path | None
    left_img_dir: Path | None
    right_img_dir: Path | None
    result_dir: Path
    cache_input_paths: CacheInputPaths
    new_output_paths: NewOutputPaths
    ground_truth_paths: GroundTruthPaths
    match_json: Path
    ba_result_json: Path
    plot_png: Path
    weights: Path
    ba_bin: Path
    match_script: Path
    plot_script: Path
    matcher_kind: str

    @property
    def gt_file(self) -> Path | None:
        return self.ground_truth_paths.file


def _timestamped_prefix(result_prefix: str, config: PipelineConfig) -> str:
    # runner 会提前填 output_timestamp，保证 dry-run 预测路径和实际命令一致；
    # 单独跑 run_pipeline.py 时这里才即时生成时间戳。
    return f"{result_prefix}_{config.output_timestamp or make_output_timestamp()}"


def _project_tools(matcher_kind: str, is_kitti_mode: bool) -> tuple[Path, Path, Path, Path]:
    # 工具路径仍沿用项目旧布局，集中在这里便于后续改成可配置路径。
    weights = PROJECT_ROOT / "matchmodel/SuperPointPretrainedNetwork/superpoint_v1.pth"
    ba_bin = PROJECT_ROOT / "build/bin" / (
        "run_offline_stereo_ba_kitti" if is_kitti_mode else "run_offline_stereo_ba"
    )
    match_script = PROJECT_ROOT / "stereo_calib/scripts" / (
        "superpoint_stereo_match_raw.py" if matcher_kind == "raw_stereo_sequence" else "superpoint_stereo_match.py"
    )
    plot_script = PROJECT_ROOT / "stereo_calib/scripts/plot_ba_history.py"
    return weights, ba_bin, match_script, plot_script


def _explicit_gt_file(config: PipelineConfig) -> Path | None:
    return Path(config.gt_param_file) if config.gt_param_file else None


def _kitti_calib_dir(img_dir: Path) -> Path:
    # KITTI RAW 目录布局约定：drive 目录旁边有 2011_09_26_calib/2011_09_26。
    return img_dir.parent.parent.parent / "2011_09_26_calib" / "2011_09_26"


def _project_gt_paths(config: PipelineConfig, img_dir: Path) -> GroundTruthPaths:
    explicit = _explicit_gt_file(config)
    if explicit is not None:
        return GroundTruthPaths(file=explicit, explicit=True)
    return GroundTruthPaths(file=img_dir / "camera_params_0000.json")


def _kitti_gt_paths(config: PipelineConfig, img_dir: Path, result_dir: Path, result_prefix: str) -> GroundTruthPaths:
    explicit = _explicit_gt_file(config)
    if explicit is not None:
        return GroundTruthPaths(file=explicit, explicit=True)
    # 这里只预测将要生成的 GT JSON 路径；真正写文件由 prepare_ground_truth() 做。
    return GroundTruthPaths(
        file=result_dir / "gt_params" / f"{result_prefix}_gt_camera.json",
        kitti_calib_dir=_kitti_calib_dir(img_dir),
        kitti_result_prefix=result_prefix,
    )


def _latest_existing_path(candidates: tuple[Path, ...]) -> Path | None:
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def _select_match_json(config: PipelineConfig, cache: CacheInputPaths, new_outputs: NewOutputPaths) -> Path:
    # 显式 --match_json 优先级最高；否则 skip_match 直接复用
    # timestamp 新规则下最新的历史 matches。
    if config.match_json:
        return Path(config.match_json)
    if config.skip_match and cache.match_json is not None:
        latest_match = _latest_existing_path(cache.match_json_candidates)
        if latest_match is not None:
            return latest_match
        return cache.match_json
    return new_outputs.match_json


def _select_ba_result_json(config: PipelineConfig, cache: CacheInputPaths, new_outputs: NewOutputPaths) -> Path:
    # BA 没有单独的显式输入参数；skip_ba 时沿用旧结果，否则写本次新结果。
    if config.skip_ba and cache.ba_result_json is not None:
        return cache.ba_result_json
    return new_outputs.ba_result_json


def _build_paths(
    config: PipelineConfig,
    result_prefix: str,
    img_dir: Path,
    matcher_input_dir: Path | None,
    left_img_dir: Path | None,
    right_img_dir: Path | None,
    matcher_kind: str,
    cache_input_paths: CacheInputPaths,
    new_output_paths: NewOutputPaths,
    ground_truth_paths: GroundTruthPaths,
) -> PipelinePaths:
    # 各 dataset resolver 只负责数据集差异；公共工具路径和最终 effective path
    # 在这里统一装配，减少每个分支里的重复代码。
    is_kitti_mode = config.is_kitti_mode
    weights, ba_bin, match_script, plot_script = _project_tools(matcher_kind, is_kitti_mode)
    return PipelinePaths(
        dataset_mode=config.dataset_mode,
        scene=config.scene,
        result_prefix=result_prefix,
        img_dir=img_dir,
        matcher_input_dir=matcher_input_dir,
        left_img_dir=left_img_dir,
        right_img_dir=right_img_dir,
        result_dir=PROJECT_ROOT / "stereo_calib/result",
        cache_input_paths=cache_input_paths,
        new_output_paths=new_output_paths,
        ground_truth_paths=ground_truth_paths,
        match_json=_select_match_json(config, cache_input_paths, new_output_paths),
        ba_result_json=_select_ba_result_json(config, cache_input_paths, new_output_paths),
        plot_png=new_output_paths.plot_png,
        weights=weights,
        ba_bin=ba_bin,
        match_script=match_script,
        plot_script=plot_script,
        matcher_kind=matcher_kind,
    )


def _resolve_project_paths(config: PipelineConfig) -> PipelinePaths:
    # Project/Blender 模式：输入图、GT 参数和历史缓存都按 scene 名组织。
    result_dir = PROJECT_ROOT / "stereo_calib/result"
    result_prefix = config.scene
    output_prefix = _timestamped_prefix(result_prefix, config)
    img_dir = PROJECT_ROOT / f"blender-file/stereo_{config.scene}"

    cache_inputs = CacheInputPaths(
        match_json=result_dir / "match_points" / f"{result_prefix}_matches_gtpose_dsym_le_{config.pixel_tag}.json",
        ba_result_json=result_dir / "ba_results" / f"{result_prefix}_ba_result_le_{config.pixel_tag}.json",
    )
    new_outputs = NewOutputPaths(
        match_json=result_dir / "match_points" / f"{output_prefix}_matches.json",
        ba_result_json=result_dir / "ba_results" / f"{output_prefix}_ba_result.json",
        plot_png=result_dir / f"{output_prefix}_ba_history.png",
    )
    return _build_paths(
        config=config,
        result_prefix=result_prefix,
        img_dir=img_dir,
        matcher_input_dir=img_dir,
        left_img_dir=None,
        right_img_dir=None,
        matcher_kind="standard",
        cache_input_paths=cache_inputs,
        new_output_paths=new_outputs,
        ground_truth_paths=_project_gt_paths(config, img_dir),
    )


def _resolve_kitti2015_paths(config: PipelineConfig) -> PipelinePaths:
    # KITTI 单帧模式：匹配脚本仍使用 standard 入口，但 BA 走 KITTI 可执行文件。
    if not config.kitti_root_dir:
        raise ValueError("KITTI 模式需要提供 --kitti_root_dir")

    result_dir = PROJECT_ROOT / "stereo_calib/result"
    result_prefix = f"{config.dataset_mode}_{config.scene}"
    output_prefix = _timestamped_prefix(result_prefix, config)
    img_dir = Path(config.kitti_root_dir)

    cache_inputs = CacheInputPaths(
        match_json=result_dir / "match_points" / f"{result_prefix}_matches.json",
        ba_result_json=result_dir / "ba_results" / f"{result_prefix}_ba_result.json",
    )
    new_outputs = NewOutputPaths(
        match_json=result_dir / "match_points" / f"{output_prefix}_matches.json",
        ba_result_json=result_dir / "ba_results" / f"{output_prefix}_ba_result.json",
        plot_png=result_dir / f"{output_prefix}_ba_history.png",
    )
    return _build_paths(
        config=config,
        result_prefix=result_prefix,
        img_dir=img_dir,
        matcher_input_dir=img_dir,
        left_img_dir=None,
        right_img_dir=None,
        matcher_kind="standard",
        cache_input_paths=cache_inputs,
        new_output_paths=new_outputs,
        ground_truth_paths=_kitti_gt_paths(config, img_dir, result_dir, result_prefix),
    )


def _resolve_kitti_raw_aggregate_paths(config: PipelineConfig) -> PipelinePaths:
    # Aggregate 模式只消费外部 matches JSON，不重新跑 SuperPoint。
    if not config.kitti_root_dir:
        raise ValueError("KITTI RAW aggregate 模式需要提供 --kitti_root_dir")
    if not config.match_json:
        raise ValueError("KITTI RAW aggregate 模式需要提供 --match_json")

    result_dir = PROJECT_ROOT / "stereo_calib/result"
    result_prefix = config.result_prefix or "kitti_raw_00_01"
    output_prefix = _timestamped_prefix(result_prefix, config)
    img_dir = Path(config.kitti_root_dir)
    external_match_json = Path(config.match_json)

    cache_inputs = CacheInputPaths(
        match_json=external_match_json,
        ba_result_json=result_dir / "ba_results" / f"{result_prefix}_ba_result.json",
    )
    new_outputs = NewOutputPaths(
        match_json=external_match_json,
        ba_result_json=result_dir / "ba_results" / f"{output_prefix}_ba_result.json",
        plot_png=result_dir / f"{output_prefix}_ba_history.png",
    )
    return _build_paths(
        config=config,
        result_prefix=result_prefix,
        img_dir=img_dir,
        matcher_input_dir=None,
        left_img_dir=None,
        right_img_dir=None,
        matcher_kind="standard",
        cache_input_paths=cache_inputs,
        new_output_paths=new_outputs,
        ground_truth_paths=_kitti_gt_paths(config, img_dir, result_dir, result_prefix),
    )


def _resolve_kitti_raw_sequence_paths(config: PipelineConfig) -> PipelinePaths:
    # Sequence 模式使用稀疏 pair 策略：同帧左右目 + 相邻左左/右右。
    if not config.kitti_root_dir and not (config.left_img_dir and config.right_img_dir):
        raise ValueError("KITTI RAW sequence 模式需要提供 --kitti_root_dir 或 --left_img_dir/--right_img_dir")

    result_dir = PROJECT_ROOT / "stereo_calib/result"
    result_prefix = config.result_prefix or config.scene
    img_dir = Path(config.kitti_root_dir) if config.kitti_root_dir else Path(config.left_img_dir).parent.parent
    left_img_dir = Path(config.left_img_dir) if config.left_img_dir else img_dir / "image_00" / "data"
    right_img_dir = Path(config.right_img_dir) if config.right_img_dir else img_dir / "image_01" / "data"
    output_prefix = _timestamped_prefix(result_prefix, config)
    matcher_tag = "no_ratio_margin" if config.disable_ratio_margin else "ratio_margin"
    param_tag = (
        f"conf_thresh={format_tag_value(config.conf_thresh)}"
        f"_nms_dist={format_tag_value(config.nms_dist)}"
        f"_nn_thresh={format_tag_value(config.nn_thresh)}"
        f"_{matcher_tag}"
    )

    cache_inputs = CacheInputPaths(
        match_json=result_dir / "match_points" / f"{result_prefix}_{param_tag}_matches_raw.json",
        match_json_candidates=tuple(
            sorted((result_dir / "match_points").glob(f"{result_prefix}_*_matches_raw.json"))
        ),
        ba_result_json=result_dir / "ba_results" / f"{result_prefix}_{param_tag}_ba_result_raw.json",
    )
    new_outputs = NewOutputPaths(
        match_json=result_dir / "match_points" / f"{output_prefix}_matches_raw.json",
        ba_result_json=result_dir / "ba_results" / f"{output_prefix}_ba_result_raw.json",
        plot_png=result_dir / f"{output_prefix}_ba_history_raw.png",
    )
    return _build_paths(
        config=config,
        result_prefix=result_prefix,
        img_dir=img_dir,
        matcher_input_dir=None,
        left_img_dir=left_img_dir,
        right_img_dir=right_img_dir,
        matcher_kind="raw_stereo_sequence",
        cache_input_paths=cache_inputs,
        new_output_paths=new_outputs,
        ground_truth_paths=_kitti_gt_paths(config, img_dir, result_dir, result_prefix),
    )


def resolve_paths(config: PipelineConfig) -> PipelinePaths:
    """Resolve paths for a single run without writing any files."""
    # 按 dataset mode 分发到小 resolver；新增数据集模式时优先在这里挂接。
    resolvers = {
        "project": _resolve_project_paths,
        "kitti2015": _resolve_kitti2015_paths,
        "kitti_raw_aggregate": _resolve_kitti_raw_aggregate_paths,
        "kitti_raw_sequence": _resolve_kitti_raw_sequence_paths,
    }
    try:
        resolver = resolvers[config.dataset_mode]
    except KeyError as exc:
        raise ValueError(f"不支持的数据集模式: {config.dataset_mode}") from exc
    return resolver(config)


def prepare_ground_truth(paths: PipelinePaths) -> None:
    """Generate derived ground-truth files after path resolution, when needed."""
    gt_paths = paths.ground_truth_paths
    if gt_paths.explicit or gt_paths.kitti_calib_dir is None or gt_paths.kitti_result_prefix is None:
        return
    # 只有 KITTI 自动 GT 需要到这里生成 JSON；project 模式和显式 GT 都无需写文件。
    build_kitti_gt_camera_json(gt_paths.kitti_calib_dir, paths.result_dir, gt_paths.kitti_result_prefix)

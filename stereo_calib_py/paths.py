"""Path resolution for single-run stereo calibration pipeline jobs.

这个模块集中维护“给定 dataset_mode/scene 时，输入输出文件在哪里”。
路径规则原先散在 `run_pipeline.py` 里；抽出来后新增数据集模式时，
主要改这里即可。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .kitti import build_kitti_gt_camera_json
from .project import PROJECT_ROOT


@dataclass(frozen=True)
class PipelinePaths:
    """单次 pipeline 需要的全部文件路径。

    使用 dataclass 而不是 dict，是为了让调用处用 `paths.match_json`
    这类明确字段，避免字符串 key 拼错后运行时才暴露问题。
    """

    dataset_mode: str
    scene: str
    img_dir: Path
    matcher_input_dir: Path | None
    result_dir: Path
    match_json: Path
    ba_result_json: Path
    plot_png: Path
    gt_file: Path | None
    weights: Path
    ba_bin: Path
    match_script: Path
    plot_script: Path


def resolve_paths(
    scene: str,
    pixel_tag: str,
    dataset_mode: str,
    kitti_root_dir: str | None,
    match_json: str | None,
    result_prefix: str | None,
) -> PipelinePaths:
    """根据数据集模式、场景名和像素阈值标签生成相关路径。

    这里保留旧版输出命名规则，确保 `run_experiment.py --dry-run`
    生成的命令和历史结果文件仍能对应上。
    """
    result_dir = PROJECT_ROOT / "stereo_calib/result"

    if dataset_mode == "project":
        # 项目内 Blender 数据集：目录名为 blender-file/stereo_{scene}。
        # `blender-file` 当前是指向 `datasets/blender` 的兼容软链接。
        result_prefix = scene
        img_dir = PROJECT_ROOT / f"blender-file/stereo_{scene}"
        gt_file = img_dir / "camera_params_0000.json"
        matcher_input_dir = img_dir
        match_json_path = result_dir / "match_points" / f"{result_prefix}_matches_gtpose_dsym_le_{pixel_tag}.json"
        ba_result_json = result_dir / "ba_results" / f"{result_prefix}_ba_result_le_{pixel_tag}.json"
        plot_png = result_dir / f"{result_prefix}_ba_history_gtpose_dsym_le_{pixel_tag}.png"
    elif dataset_mode == "kitti2015":
        # KITTI 单帧/单 scene 模式：输入目录由调用者指定，GT JSON 由
        # KITTI 官方 calib_cam_to_cam.txt 临时生成到 result/gt_params。
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
        # 聚合模式不重新跑匹配，直接消费外部生成好的 matches JSON。
        # 因此 matcher_input_dir 为空，pipeline 层也会要求配合 --skip_match。
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

    return PipelinePaths(
        dataset_mode=dataset_mode,
        scene=scene,
        img_dir=img_dir,
        matcher_input_dir=matcher_input_dir,
        result_dir=result_dir,
        match_json=match_json_path,
        ba_result_json=ba_result_json,
        plot_png=plot_png,
        gt_file=gt_file,
        # 下面这些是工具/模型路径。它们目前仍沿用项目旧布局，先集中
        # 放在这里，后续若支持配置覆盖，也只需要扩展本模块接口。
        weights=PROJECT_ROOT / "matchmodel/SuperPointPretrainedNetwork/superpoint_v1.pth",
        ba_bin=PROJECT_ROOT / "build/bin/run_offline_stereo_ba",
        match_script=PROJECT_ROOT / "stereo_calib/scripts/superpoint_stereo_match.py",
        plot_script=PROJECT_ROOT / "stereo_calib/scripts/plot_ba_history.py",
    )

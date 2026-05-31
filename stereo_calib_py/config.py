"""Typed configuration for one pipeline run.

`run_pipeline.py` 仍然负责公开 CLI；进入编排层后统一转成
`PipelineConfig`，这样 pipeline 代码不用到处依赖 `argparse.Namespace`
里“刚好存在”的字段。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    """All user-controlled options for a single stereo calibration run."""

    # 这些默认值需要和 run_pipeline.py 的 argparse 默认值保持一致；
    # 这样 CLI、测试和 YAML runner 都可以用同一套配置对象。

    # Dataset selection and path overrides.
    dataset_mode: str = "project"
    scene: str = "panoramic_01"
    kitti_root_dir: str | None = None
    match_json: str | None = None
    result_prefix: str | None = None
    gt_param_file: str | None = None
    init_param_file: str | None = None
    frame_poses_file: str | None = None
    left_img_dir: str | None = None
    right_img_dir: str | None = None
    pixel_tag: str = "3px"
    output_timestamp: str | None = None
    result_dir: str | None = None

    # Pipeline switches.
    skip_match: bool = False
    skip_ba: bool = False
    no_plot: bool = False

    # SuperPoint options.
    nn_thresh: float = 0.7
    conf_thresh: float = 0.015
    nms_dist: int = 4
    disable_ratio_margin: bool = False
    cuda: bool = False

    # Bundle Adjustment options.
    max_iter: int = 200
    incremental_max_iter: int = 20
    global_opt_interval: int = 5
    huber: float = 1.0
    min_track_len: int = 3
    min_pair_inliers: int = 12
    max_score: float = 1.0
    fix_distortion: bool = False
    per_frame_max_iter: int = 5
    baseline_prior: float | None = None
    tx_prior: float | None = None
    focal_prior: float | None = None
    focal_lower_scale: float | None = None
    focal_upper_scale: float | None = None
    outlier_threshold: float | None = None
    outlier_rounds: int | None = None
    max_reproj_error: float | None = None
    enable_per_frame_correction: bool = False
    enable_incremental_free_principal_point_refine: bool = False
    incremental_free_principal_point_max_iter: int | None = None
    min_active_frames_for_free_principal_point: int | None = None
    free_principal_point_every_n_global_ba: int | None = None
    free_principal_point_max_rmse_increase: float | None = None
    reset_camera_params_each_ba_round: bool = False

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "PipelineConfig":
        """Build config from the CLI parser output."""
        return cls.from_mapping(vars(args))

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "PipelineConfig":
        """Build config from a YAML/defaults mapping, ignoring unrelated keys."""
        allowed = {field.name for field in fields(cls)}
        # YAML runner 的 mapping 可能带有 label 等非 pipeline 字段；
        # 这里只取 PipelineConfig 认识的字段，避免配置层泄漏进执行层。
        init_values = {key: value for key, value in values.items() if key in allowed}
        return cls(**init_values)

    @classmethod
    def coerce(cls, value: "PipelineConfig | argparse.Namespace") -> "PipelineConfig":
        """Accept the typed config directly, or convert a legacy Namespace."""
        if isinstance(value, cls):
            return value
        if isinstance(value, argparse.Namespace):
            return cls.from_namespace(value)
        raise TypeError(f"Unsupported pipeline config type: {type(value)!r}")

    @property
    def is_kitti_mode(self) -> bool:
        return self.dataset_mode in ("kitti2015", "kitti_raw_aggregate", "kitti_raw_sequence")

    def optional_ba_value_args(self) -> dict[str, float | int | None]:
        """BA CLI arguments that should only be emitted when explicitly set."""
        # None 表示“不传给 C++，使用可执行文件自己的默认值”；
        # 非 None 的参数会在 pipeline.py 中展开成 --name value。
        return {
            "per_frame_max_iter": self.per_frame_max_iter,
            "baseline_prior": self.baseline_prior,
            "tx_prior": self.tx_prior,
            "focal_prior": self.focal_prior,
            "focal_lower_scale": self.focal_lower_scale,
            "focal_upper_scale": self.focal_upper_scale,
            "outlier_threshold": self.outlier_threshold,
            "outlier_rounds": self.outlier_rounds,
            "max_reproj_error": self.max_reproj_error,
            "incremental_free_principal_point_max_iter": self.incremental_free_principal_point_max_iter,
            "min_active_frames_for_free_principal_point": self.min_active_frames_for_free_principal_point,
            "free_principal_point_every_n_global_ba": self.free_principal_point_every_n_global_ba,
            "free_principal_point_max_rmse_increase": self.free_principal_point_max_rmse_increase,
        }

#!/usr/bin/env python3
"""CPU-only one-shot stereo calibration release entrypoint."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from stereo_calib_py.calibration_result import write_calibration_result
from stereo_calib_py.config import PipelineConfig
from stereo_calib_py.pipeline import run_pipeline
from stereo_calib_py.paths import resolve_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one CPU stereo calibration and write the final result JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--left_img_dir", required=True, help="Left image directory")
    parser.add_argument("--right_img_dir", required=True, help="Right image directory")
    parser.add_argument("--init_param_file", required=True, help="Initial stereo camera parameter file")
    parser.add_argument("--output_json", required=True, help="Final calibration result JSON")
    parser.add_argument("--output_plot", default=None, help="Optional BA history PNG output path")
    parser.add_argument("--frame_poses_file", default=None, help="Optional frame poses JSON")
    parser.add_argument("--gt_param_file", default=None, help="Optional ground-truth camera parameter file")
    parser.add_argument("--work_dir", default=None, help="Directory for intermediate matches and BA files")
    parser.add_argument("--keep_work_dir", action="store_true", help="Keep intermediate files after success")
    parser.add_argument("--result_prefix", default="calibration", help="Intermediate file prefix")

    sp = parser.add_argument_group("SuperPoint CPU matching")
    sp.add_argument("--nn_thresh", type=float, default=0.7, help="Descriptor L2 distance threshold")
    sp.add_argument("--conf_thresh", type=float, default=0.015, help="Keypoint confidence threshold")
    sp.add_argument("--nms_dist", type=int, default=4, help="NMS suppression radius in pixels")
    sp.add_argument("--disable_ratio_margin", action="store_true", help="Disable ratio / margin filtering")

    ba = parser.add_argument_group("Bundle Adjustment")
    ba.add_argument("--max_iter", type=int, default=200, help="Final global BA max iterations")
    ba.add_argument("--incremental_max_iter", type=int, default=20, help="Incremental BA max iterations")
    ba.add_argument("--global_opt_interval", type=int, default=5, help="Frames between global BA runs")
    ba.add_argument("--huber", type=float, default=1.0, help="Huber loss threshold in pixels")
    ba.add_argument("--min_track_len", type=int, default=3, help="Minimum track length")
    ba.add_argument("--min_pair_inliers", type=int, default=12, help="Minimum inliers per image pair")
    ba.add_argument("--max_score", type=float, default=1.0, help="Maximum descriptor match score")
    ba.add_argument("--fix_distortion", action="store_true", help="Keep distortion fixed")
    ba.add_argument("--per_frame_max_iter", type=int, default=5, help="Per-frame local correction max iterations")
    ba.add_argument("--baseline_prior", type=float, default=None, help="Baseline prior weight")
    ba.add_argument("--tx_prior", type=float, default=None, help="KITTI x-translation prior weight")
    ba.add_argument("--focal_prior", type=float, default=None, help="Focal prior weight")
    ba.add_argument("--focal_lower_scale", type=float, default=None, help="Focal lower bound scale")
    ba.add_argument("--focal_upper_scale", type=float, default=None, help="Focal upper bound scale")
    ba.add_argument("--outlier_threshold", type=float, default=None, help="Outlier rejection threshold in pixels")
    ba.add_argument("--outlier_rounds", type=int, default=None, help="Maximum outlier rejection rounds")
    ba.add_argument("--max_reproj_error", type=float, default=None, help="Maximum accepted reprojection error")
    ba.add_argument("--enable_per_frame_correction", action="store_true", help="Enable per-frame local correction")
    ba.add_argument(
        "--enable_incremental_free_principal_point_refine",
        action="store_true",
        help="Run extra incremental global BA with free principal point",
    )
    ba.add_argument(
        "--incremental_free_principal_point_max_iter",
        type=int,
        default=None,
        help="Max iterations for incremental free-principal-point refine",
    )
    ba.add_argument(
        "--min_active_frames_for_free_principal_point",
        type=int,
        default=None,
        help="Minimum active frames before free-principal-point refine",
    )
    ba.add_argument(
        "--free_principal_point_every_n_global_ba",
        type=int,
        default=None,
        help="Run free-principal-point refine every N global BA runs",
    )
    ba.add_argument(
        "--free_principal_point_max_rmse_increase",
        type=float,
        default=None,
        help="Maximum allowed RMSE increase for free-principal-point refine",
    )
    ba.add_argument(
        "--reset_camera_params_each_ba_round",
        action="store_true",
        help="Reset camera parameters before each BA round",
    )
    return parser.parse_args()


def _resolve_path(value: str) -> str:
    return str(Path(value).expanduser().resolve())


def _resolve_optional_path(value: str | None) -> str | None:
    if value is None:
        return None
    return _resolve_path(value)


def _build_config(args: argparse.Namespace, work_dir: Path) -> PipelineConfig:
    return PipelineConfig(
        dataset_mode="kitti_raw_sequence",
        scene=args.result_prefix,
        result_prefix=args.result_prefix,
        init_param_file=_resolve_path(args.init_param_file),
        frame_poses_file=_resolve_optional_path(args.frame_poses_file),
        gt_param_file=_resolve_optional_path(args.gt_param_file),
        left_img_dir=_resolve_path(args.left_img_dir),
        right_img_dir=_resolve_path(args.right_img_dir),
        result_dir=str(work_dir),
        output_timestamp="run",
        match_output_tag="run",
        no_plot=args.output_plot is None,
        cuda=False,
        nn_thresh=args.nn_thresh,
        conf_thresh=args.conf_thresh,
        nms_dist=args.nms_dist,
        disable_ratio_margin=args.disable_ratio_margin,
        max_iter=args.max_iter,
        incremental_max_iter=args.incremental_max_iter,
        global_opt_interval=args.global_opt_interval,
        huber=args.huber,
        min_track_len=args.min_track_len,
        min_pair_inliers=args.min_pair_inliers,
        max_score=args.max_score,
        fix_distortion=args.fix_distortion,
        per_frame_max_iter=args.per_frame_max_iter,
        baseline_prior=args.baseline_prior,
        tx_prior=args.tx_prior,
        focal_prior=args.focal_prior,
        focal_lower_scale=args.focal_lower_scale,
        focal_upper_scale=args.focal_upper_scale,
        outlier_threshold=args.outlier_threshold,
        outlier_rounds=args.outlier_rounds,
        max_reproj_error=args.max_reproj_error,
        enable_per_frame_correction=args.enable_per_frame_correction,
        enable_incremental_free_principal_point_refine=args.enable_incremental_free_principal_point_refine,
        incremental_free_principal_point_max_iter=args.incremental_free_principal_point_max_iter,
        min_active_frames_for_free_principal_point=args.min_active_frames_for_free_principal_point,
        free_principal_point_every_n_global_ba=args.free_principal_point_every_n_global_ba,
        free_principal_point_max_rmse_increase=args.free_principal_point_max_rmse_increase,
        reset_camera_params_each_ba_round=args.reset_camera_params_each_ba_round,
    )


def _copy_plot(paths, output_plot: Path | None) -> None:
    if output_plot is None:
        return
    if not paths.plot_png.exists():
        raise FileNotFoundError(f"BA history plot was not generated: {paths.plot_png}")
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(paths.plot_png, output_plot)


def run_once(args: argparse.Namespace) -> int:
    output_json = Path(_resolve_path(args.output_json))
    output_plot = Path(_resolve_path(args.output_plot)) if args.output_plot else None
    if args.work_dir:
        work_dir = Path(args.work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        temp_ctx = None
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="mycalib_")
        work_dir = Path(temp_ctx.name)

    config = _build_config(args, work_dir)
    paths = resolve_paths(config)
    exit_code = 0
    try:
        try:
            run_pipeline(config)
        except SystemExit as exc:
            if isinstance(exc.code, int):
                exit_code = exc.code
            elif exc.code:
                print(exc.code, file=sys.stderr)
                exit_code = 1
            else:
                exit_code = 0
        try:
            write_calibration_result(paths.ba_result_json, output_json)
            _copy_plot(paths, output_plot)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return exit_code or 1
        if exit_code != 0:
            print(
                "Pipeline reported a non-zero quality/status code, "
                "but a result JSON was generated. Check its 'success' field.",
                file=sys.stderr,
            )
            exit_code = 0
        print(f"\nFinal calibration JSON written to: {output_json}")
        if output_plot is not None:
            print(f"BA history plot written to: {output_plot}")
    finally:
        if temp_ctx is not None and (exit_code == 0 or not args.keep_work_dir):
            temp_ctx.cleanup()
        elif temp_ctx is not None:
            print(f"Intermediate files kept at: {work_dir}")

    return exit_code


def main() -> None:
    sys.exit(run_once(parse_args()))


if __name__ == "__main__":
    main()

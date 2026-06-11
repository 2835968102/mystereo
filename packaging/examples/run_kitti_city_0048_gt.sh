#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

LEFT_IMG_DIR="${LEFT_IMG_DIR:-/media/hello/新加卷/pmldata/data/KITTI/RAW/City/2011_09_26/2011_09_26_drive_0048_sync/image_00/data}"
RIGHT_IMG_DIR="${RIGHT_IMG_DIR:-/media/hello/新加卷/pmldata/data/KITTI/RAW/City/2011_09_26/2011_09_26_drive_0048_sync/image_01/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PACKAGE_DIR/results}"
WINDOW_SIZE="${WINDOW_SIZE:-50}"

mkdir -p "$OUTPUT_DIR"

"$PACKAGE_DIR/mycalib-linux-x86_64-cpu" \
  --left_img_dir "$LEFT_IMG_DIR" \
  --right_img_dir "$RIGHT_IMG_DIR" \
  --init_param_file "$SCRIPT_DIR/kitti_raw_00_01_init_params.txt" \
  --gt_param_file "$SCRIPT_DIR/kitti_raw_00_01_gt_camera.json" \
  --output_json "$OUTPUT_DIR/kitti_city_0048_calibration_result_gt.json" \
  --output_plot "$OUTPUT_DIR/kitti_city_0048_ba_history_gt.png" \
  --result_prefix kitti_city_2011_09_26_drive_0048_sync \
  --window_size "$WINDOW_SIZE" \
  --conf_thresh 0.25 \
  --nms_dist 20 \
  --nn_thresh 1 \
  --max_iter 24 \
  --incremental_max_iter 12 \
  --per_frame_max_iter 3 \
  --global_opt_interval 10 \
  --min_pair_inliers 15 \
  --max_score 0.5 \
  --min_track_len 3 \
  --fix_distortion \
  --baseline_prior 100 \
  --tx_prior 0 \
  --focal_prior 1 \
  --focal_lower_scale 0.2 \
  --focal_upper_scale 3.0 \
  --outlier_threshold 2.0 \
  --enable_per_frame_correction \
  --enable_incremental_free_principal_point_refine \
  --incremental_free_principal_point_max_iter 8 \
  --min_active_frames_for_free_principal_point 20 \
  --free_principal_point_every_n_global_ba 2 \
  --free_principal_point_max_rmse_increase 0.05

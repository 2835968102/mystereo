#!/bin/bash
set -euo pipefail

# 兼容入口：KITTI RAW City 小规模测试实验，关闭 Lowe ratio / margin 过滤。
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-stereo-calib-vis}"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
RESULT_DIR="${RESULT_DIR:-stereo_calib/result/${SCRIPT_NAME}}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S_%N)}"
LOG_DIR="${LOG_DIR:-${RESULT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${SCRIPT_NAME}_${RUN_ID}.log}"

cd "$CURRENT_DIR"
mkdir -p "$LOG_DIR"
python3 run_experiment.py \
  configs/experiments/kitti_raw_city_small_test_no_margin.yaml \
  --conda-env "$CONDA_ENV" \
  --result-dir "$RESULT_DIR" \
  --match-output-tag "$SCRIPT_NAME" \
  --run-id "$RUN_ID" \
  "$@" 2>&1 | tee "$LOG_FILE"

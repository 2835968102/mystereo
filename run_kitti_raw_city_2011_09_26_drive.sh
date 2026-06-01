#!/bin/bash
set -euo pipefail

# 兼容入口：KITTI RAW sync drive 批量实验逻辑已经移到
# run_experiment.py 和 configs/experiments/kitti_raw_city_0001.yaml 中。
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
  configs/experiments/kitti_raw_city_0001.yaml \
  --conda-env "$CONDA_ENV" \
  --result-dir "$RESULT_DIR" \
  --match-output-tag "$SCRIPT_NAME" \
  --run-id "$RUN_ID" \
  "$@" 2>&1 | tee "$LOG_FILE"

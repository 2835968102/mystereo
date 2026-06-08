#!/bin/bash
set -euo pipefail

# Leave-one-out validation for the KITTI RAW City small test.
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-stereo-calib-vis}"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
RESULT_DIR="${RESULT_DIR:-stereo_calib/result/${SCRIPT_NAME}}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S_%N)}"
LOG_DIR="${LOG_DIR:-${RESULT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${SCRIPT_NAME}_${RUN_ID}.log}"

CONFIGS=(
  "configs/experiments/kitti_raw_city_loo_validate_0060.yaml"
  "configs/experiments/kitti_raw_city_loo_validate_0002.yaml"
  "configs/experiments/kitti_raw_city_loo_validate_0113.yaml"
  "configs/experiments/kitti_raw_city_loo_validate_0048.yaml"
)

FOLDS=(
  "validate_0060"
  "validate_0002"
  "validate_0113"
  "validate_0048"
)

cd "$CURRENT_DIR"
mkdir -p "$LOG_DIR"

{
  echo "Leave-one-out RUN_ID=${RUN_ID}"
  for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    fold="${FOLDS[$i]}"
    fold_run_id="${RUN_ID}_${fold}"
    extra_args=()
    if [[ "$i" -gt 0 ]]; then
      extra_args+=(--no-build)
    fi

    echo
    echo "============================================================"
    echo "Fold ${fold}: ${config}"
    echo "============================================================"
    python3 run_experiment.py \
      "$config" \
      --conda-env "$CONDA_ENV" \
      --result-dir "$RESULT_DIR" \
      --match-output-tag "$SCRIPT_NAME" \
      --run-id "$fold_run_id" \
      "${extra_args[@]}"
  done
} 2>&1 | tee "$LOG_FILE"

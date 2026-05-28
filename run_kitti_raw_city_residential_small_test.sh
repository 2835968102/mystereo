#!/bin/bash
set -euo pipefail

# 兼容入口：KITTI RAW City 小规模测试实验。
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-stereo-calib-vis}"

cd "$CURRENT_DIR"
python3 run_experiment.py \
  configs/experiments/kitti_raw_city_residential_small_test.yaml \
  --conda-env "$CONDA_ENV" \
  "$@"

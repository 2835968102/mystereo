#!/bin/bash
set -euo pipefail

# 兼容入口：KITTI RAW City 小规模测试实验，关闭 Lowe ratio / margin 过滤。
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-stereo-calib-vis}"

cd "$CURRENT_DIR"
python3 run_experiment.py \
  configs/experiments/kitti_raw_city_small_test_no_margin.yaml \
  --conda-env "$CONDA_ENV" \
  "$@"

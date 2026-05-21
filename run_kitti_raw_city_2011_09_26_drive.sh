#!/bin/bash
set -euo pipefail

# 兼容入口：KITTI RAW sync drive 批量实验逻辑已经移到
# run_experiment.py 和 configs/experiments/kitti_raw_city_0001.yaml 中。
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-stereo-calib-vis}"

cd "$CURRENT_DIR"
python3 run_experiment.py \
  configs/experiments/kitti_raw_city_0001.yaml \
  --conda-env "$CONDA_ENV" \
  "$@"

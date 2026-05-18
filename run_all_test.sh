#!/bin/bash
set -euo pipefail

# 兼容旧入口：批量实验逻辑已经移到 run_experiment.py 和
# configs/experiments/*.yaml 中。
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-stereo-calib-vis}"

cd "$CURRENT_DIR"
python3 run_experiment.py \
  configs/experiments/panoramic_threshold_sweep.yaml \
  --conda-env "$CONDA_ENV" \
  "$@"

#!/bin/bash
set -euo pipefail

# Thin wrapper for the KITTI RAW residential sync drive batch experiment.
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-stereo-calib-vis}"

cd "$CURRENT_DIR"
python3 run_experiment.py \
  configs/experiments/kitti_raw_residential.yaml \
  --conda-env "$CONDA_ENV" \
  "$@"

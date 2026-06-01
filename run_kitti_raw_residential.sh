#!/bin/bash
set -euo pipefail

# Thin wrapper for the KITTI RAW residential sync drive batch experiment.
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-stereo-calib-vis}"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
RESULT_DIR="${RESULT_DIR:-stereo_calib/result/${SCRIPT_NAME}}"

cd "$CURRENT_DIR"
python3 run_experiment.py \
  configs/experiments/kitti_raw_residential.yaml \
  --conda-env "$CONDA_ENV" \
  --result-dir "$RESULT_DIR" \
  --match-output-tag "$SCRIPT_NAME" \
  "$@"

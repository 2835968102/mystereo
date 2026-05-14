#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "== Python syntax =="
python3 -m compileall run_pipeline.py run_experiment.py stereo_calib_py

echo "== Pipeline CLI =="
python3 run_pipeline.py --help >/tmp/mycalib_run_pipeline_help.txt

echo "== Panoramic experiment dry-run =="
python3 run_experiment.py configs/experiments/panoramic_threshold_sweep.yaml \
  --dry-run --no-build --scene panoramic_01 >/tmp/mycalib_panoramic_dry_run.txt

echo "== KITTI RAW experiment dry-run =="
python3 run_experiment.py configs/experiments/kitti_raw_city_0001.yaml \
  --dry-run --no-build >/tmp/mycalib_kitti_raw_dry_run.txt

echo "== C++ build =="
cmake --build build -j"$(nproc)"

echo "All checks passed."

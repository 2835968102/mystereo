#!/bin/bash
set -euo pipefail

# 兼容旧入口：批量实验逻辑已经移到 run_experiment.py 和
# configs/experiments/*.yaml 中。
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$CURRENT_DIR"
python3 run_experiment.py "$@"

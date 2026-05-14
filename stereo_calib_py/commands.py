"""Small command/logging helpers for pipeline orchestration.

这里刻意只放很薄的 shell 调度能力。真正的实验配置在
`run_experiment.py` / YAML 中，单次流程逻辑在 `pipeline.py` 中。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .project import PROJECT_ROOT


def log_step(msg: str) -> None:
    """用统一的横线块打印流程阶段，方便长日志中肉眼定位步骤。"""
    bar = "=" * 60
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


def run_cmd(cmd: list, cwd: Path = PROJECT_ROOT) -> None:
    """打印并执行子命令；失败时直接结束当前 pipeline。

    这个函数保留了旧版 `run_pipeline.py` 的行为：子进程非 0 退出码会
    让当前 Python 进程以错误信息退出，而不是继续跑后续步骤。
    """
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    ret = subprocess.run(cmd, cwd=cwd)
    if ret.returncode != 0:
        sys.exit(f"\n命令执行失败（exit code {ret.returncode}）")

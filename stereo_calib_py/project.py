"""Project-wide paths shared by Python pipeline modules."""

from pathlib import Path


# 所有 Python 入口都从项目根目录解析相对路径，避免调用方 cwd
# 不同导致输出写到意料之外的位置。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

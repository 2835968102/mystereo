"""Project-wide paths shared by Python pipeline modules."""

from pathlib import Path
import sys


# 所有 Python 入口都从项目根目录解析资源路径，避免调用方 cwd
# 不同导致读取到意料之外的位置。PyInstaller 打包后，资源位于
# sys._MEIPASS，默认运行输出则应相对可执行文件所在目录。
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    PROJECT_ROOT = Path(sys._MEIPASS).resolve()
    RUNTIME_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    RUNTIME_ROOT = PROJECT_ROOT

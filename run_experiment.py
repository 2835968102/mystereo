#!/usr/bin/env python3
"""从 YAML 配置运行立体标定批量实验。

这个 runner 的目标是把实验定义从 shell 脚本里移出来。
shell 包装脚本保持很薄，场景列表、参数扫描和命令参数都放到
configs/experiments/*.yaml 里维护。
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


def format_elapsed_time(total_seconds: float) -> str:
    seconds = int(total_seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_config(path: Path) -> dict[str, Any]:
    """加载实验配置，并要求顶层是 YAML mapping。"""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def merge_dicts(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """把全局默认参数和单次 run 的覆盖参数合并一层。"""
    merged = dict(base)
    if override:
        merged.update(override)
    return merged


def to_project_path(value: str | Path) -> Path:
    """把相对路径解析为相对于项目根目录的路径。"""
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_executable(name: str, configured_path: str | None, fallback_paths: list[str]) -> str:
    """解析可执行文件路径：配置优先，其次 PATH，最后尝试常见安装位置。"""
    candidates = []
    if configured_path:
        candidates.append(configured_path)
    env_path = os.environ.get(f"{name.upper()}_BIN")
    if env_path:
        candidates.append(env_path)
    found = shutil.which(name)
    if found:
        candidates.append(found)
    candidates.extend(fallback_paths)

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)

    raise FileNotFoundError(
        f"找不到可执行文件 {name}，请在配置 runner.{name}_executable 中指定路径，"
        f"或设置环境变量 {name.upper()}_BIN"
    )


def resolve_conda_executable(runner: dict[str, Any]) -> str:
    """解析 conda 路径，兼容 root 环境 PATH 中没有 conda 的情况。"""
    return resolve_executable(
        "conda",
        runner.get("conda_executable"),
        [
            "/home/hello/miniconda3/bin/conda",
            "/home/hello/anaconda3/bin/conda",
            "/opt/conda/bin/conda",
            "/root/miniconda3/bin/conda",
            "/root/anaconda3/bin/conda",
        ],
    )


def build_python_command(runner: dict[str, Any], script: str) -> list[str]:
    """构造 Python 命令；如果配置了 conda_env，就包一层 `conda run`。"""
    python_executable = str(runner.get("python_executable", "python3"))
    cmd = [python_executable, script]
    conda_env = runner.get("conda_env")
    if conda_env:
        conda_executable = resolve_conda_executable(runner)
        cmd = [
            conda_executable,
            "run",
            "--no-capture-output",
            "-n",
            str(conda_env),
            *cmd,
        ]
    return cmd


def append_cli_args(cmd: list[str], args: dict[str, Any]) -> list[str]:
    """把 YAML 参数映射转换成现有脚本能识别的命令行参数。"""
    full_cmd = list(cmd)
    for key, value in args.items():
        flag = "--" + key.replace("_", "-")
        legacy_flag = "--" + key

        # 现有脚本使用下划线风格参数，比如 --max_iter；这里保持兼容。
        flag = legacy_flag

        if isinstance(value, bool):
            if value:
                full_cmd.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                full_cmd.extend([flag, str(item)])
            continue
        full_cmd.extend([flag, str(value)])
    return full_cmd


def print_command(cmd: list[str]) -> None:
    print("$ " + " ".join(shlex.quote(str(part)) for part in cmd), flush=True)


def run_command(cmd: list[str], dry_run: bool) -> None:
    """打印每条命令；如果不是 dry-run，就真正执行。"""
    print_command(cmd)
    if dry_run:
        return
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}")


def run_build(config: dict[str, Any], dry_run: bool) -> None:
    """执行配置里的 CMake 编译步骤。"""
    build = config.get("build", {})
    if not build or not build.get("enabled", False):
        return

    source_dir = to_project_path(build.get("source_dir", "stereo_calib"))
    build_dir = to_project_path(build.get("build_dir", "build"))
    jobs = build.get("jobs", os.cpu_count() or 1)

    print("\n========== Build ==========", flush=True)
    run_command(["cmake", "-S", str(source_dir), "-B", str(build_dir)], dry_run)
    run_command(["cmake", "--build", str(build_dir), f"-j{jobs}"], dry_run)


def build_pipeline_args(
    scene: str,
    default_args: dict[str, Any],
    run_args: dict[str, Any] | None,
) -> dict[str, Any]:
    """合并 scene、全局默认参数和单次 run 的 pipeline 参数。"""
    args = merge_dicts(default_args, run_args)
    args = {"scene": scene, **args}
    return args


def run_pipeline_for_scene(
    scene: str,
    run_config: dict[str, Any],
    config: dict[str, Any],
    dry_run: bool,
) -> None:
    """针对一个 scene/run 组合执行一次 run_pipeline.py。"""
    runner = config.get("runner", {})
    defaults = config.get("defaults", {})
    default_args = defaults.get("pipeline_args", {})
    if not isinstance(default_args, dict):
        raise ValueError("defaults.pipeline_args must be a mapping")

    run_args = run_config.get("pipeline_args", {})
    if not isinstance(run_args, dict):
        raise ValueError("runs[].pipeline_args must be a mapping")

    label = run_config.get("label", "run")
    pipeline_script = str(runner.get("pipeline_script", "run_pipeline.py"))
    cmd = build_python_command(runner, pipeline_script)
    cmd = append_cli_args(cmd, build_pipeline_args(scene, default_args, run_args))

    print(f"\n========== Scene: {scene} | Run: {label} ==========", flush=True)
    started = time.monotonic()
    run_command(cmd, dry_run)
    print(f"Elapsed: {format_elapsed_time(time.monotonic() - started)}", flush=True)


def run_threshold_comparison(
    scene: str,
    postprocess: dict[str, Any],
    runner: dict[str, Any],
    dry_run: bool,
) -> None:
    """为单个 scene 执行可选的 3px/2px/1px 对比图后处理。"""
    comparison = postprocess.get("threshold_comparison", {})
    if not comparison or not comparison.get("enabled", False):
        return

    script = str(comparison["script"])
    inputs = comparison.get("inputs", {})
    output = str(comparison["output"]).format(scene=scene)

    cmd = build_python_command(runner, script)
    cmd.extend(["--scene", scene])
    for key, template in inputs.items():
        flag_name = key.replace("_", "-")
        cmd.extend([f"--{flag_name}", str(template).format(scene=scene)])
    cmd.extend(["--output", output])

    print(f"\n========== Scene: {scene} | Postprocess ==========", flush=True)
    run_command(cmd, dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a batch experiment described by a YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/experiments/panoramic_threshold_sweep.yaml",
        help="Experiment YAML config path",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--no-build", action="store_true", help="Skip the build step from the config")
    parser.add_argument("--scene", action="append", help="Run only the selected scene; can be repeated")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()

    try:
        config = load_config(to_project_path(args.config))
        if args.no_build:
            config.setdefault("build", {})["enabled"] = False

        scenes = args.scene or config.get("scenes", [])
        runs = config.get("runs", [])
        if not scenes:
            raise ValueError("No scenes configured")
        if not runs:
            raise ValueError("No runs configured")

        print(f"Experiment: {config.get('name', Path(args.config).stem)}", flush=True)
        run_build(config, args.dry_run)

        for scene in scenes:
            for run_config in runs:
                run_pipeline_for_scene(str(scene), run_config, config, args.dry_run)
            run_threshold_comparison(
                str(scene),
                config.get("postprocess", {}),
                config.get("runner", {}),
                args.dry_run,
            )

    except Exception as exc:
        print(f"\nExperiment failed: {exc}", file=sys.stderr)
        return 1
    finally:
        print(f"\nTotal elapsed: {format_elapsed_time(time.monotonic() - started)}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

from stereo_calib_py.config import PipelineConfig
from stereo_calib_py.paths import PipelinePaths, make_output_timestamp, resolve_paths


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


def predict_pipeline_paths(pipeline_args: dict[str, Any]) -> PipelinePaths:
    """用和 run_pipeline.py 相同的规则提前算出本次输出路径。"""
    # 只做路径预测，不生成 KITTI GT 文件；这样 dry-run 不会改动 result 目录。
    return resolve_paths(PipelineConfig.from_mapping(pipeline_args))


def index_pipeline_output(
    output_index: dict[str, Path],
    label: str,
    pipeline_args: dict[str, Any],
    paths: PipelinePaths,
) -> None:
    """记录本次 run 的 BA 输出，供后处理通过 label/pixel_tag 查找。"""
    output_index[label] = paths.ba_result_json
    pixel_tag = pipeline_args.get("pixel_tag")
    if pixel_tag:
        output_index[str(pixel_tag)] = paths.ba_result_json
        output_index[f"le_{pixel_tag}"] = paths.ba_result_json


def run_pipeline_for_scene(
    scene: str,
    run_config: dict[str, Any],
    config: dict[str, Any],
    dry_run: bool,
) -> tuple[str, dict[str, Any], PipelinePaths]:
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
    pipeline_args = build_pipeline_args(scene, default_args, run_args)
    pipeline_args.setdefault("output_timestamp", make_output_timestamp())
    paths = predict_pipeline_paths(pipeline_args)

    cmd = build_python_command(runner, pipeline_script)
    cmd = append_cli_args(cmd, pipeline_args)

    print(f"\n========== Scene: {scene} | Run: {label} ==========", flush=True)
    started = time.monotonic()
    run_command(cmd, dry_run)
    print(f"Elapsed: {format_elapsed_time(time.monotonic() - started)}", flush=True)
    return str(label), pipeline_args, paths


def resolve_postprocess_input(
    input_key: str,
    template: str,
    scene: str,
    output_index: dict[str, Path],
) -> str:
    """优先使用本次实验刚生成的输出；找不到时回退到 YAML 模板。"""
    selector = input_key.removeprefix("input_")
    for candidate in (input_key, selector, f"le_{selector}"):
        if candidate in output_index:
            return str(output_index[candidate])
    return str(template).format(scene=scene)


def run_threshold_comparison(
    scene: str,
    postprocess: dict[str, Any],
    runner: dict[str, Any],
    output_index: dict[str, Path],
    dry_run: bool,
) -> None:
    """为单个 scene 执行可选的 3px/2px/1px 对比图后处理。"""
    comparison = postprocess.get("threshold_comparison", {})
    if not comparison or not comparison.get("enabled", False):
        return

    script = str(comparison["script"])
    inputs = comparison.get("inputs", {})
    output = str(comparison["output"]).format(scene=scene, timestamp=make_output_timestamp())

    cmd = build_python_command(runner, script)
    cmd.extend(["--scene", scene])
    for key, template in inputs.items():
        flag_name = key.replace("_", "-")
        cmd.extend([f"--{flag_name}", resolve_postprocess_input(key, str(template), scene, output_index)])
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
    parser.add_argument("--conda-env", default=None, help="Override runner.conda_env from the config")
    parser.add_argument("--scene", action="append", help="Run only the selected scene; can be repeated")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()

    try:
        config = load_config(to_project_path(args.config))
        if args.no_build:
            config.setdefault("build", {})["enabled"] = False
        if args.conda_env:
            config.setdefault("runner", {})["conda_env"] = args.conda_env

        scenes = args.scene or config.get("scenes", [])
        runs = config.get("runs", [])
        if not scenes:
            raise ValueError("No scenes configured")
        if not runs:
            raise ValueError("No runs configured")

        print(f"Experiment: {config.get('name', Path(args.config).stem)}", flush=True)
        run_build(config, args.dry_run)

        for scene in scenes:
            output_index: dict[str, Path] = {}
            for run_config in runs:
                label, pipeline_args, paths = run_pipeline_for_scene(str(scene), run_config, config, args.dry_run)
                index_pipeline_output(output_index, label, pipeline_args, paths)
            run_threshold_comparison(
                str(scene),
                config.get("postprocess", {}),
                config.get("runner", {}),
                output_index,
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

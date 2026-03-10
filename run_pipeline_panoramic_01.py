#!/usr/bin/env python3
"""
stereo_panoramic_01 全流程标定脚本

从项目根目录运行（确保已激活 stereo-calib-vis conda 环境并安装了 torch/opencv）：

    python run_pipeline_panoramic_01.py [选项]

完整三步流程：
  1. SuperPoint 特征提取与匹配（所有图像对）
  2. C++ 增量式 Bundle Adjustment 优化
  3. BA 优化历史可视化

常用示例：
    # 完整流程
    python run_pipeline_panoramic_01.py

    # 使用 CUDA 加速特征匹配
    python run_pipeline_panoramic_01.py --cuda

    # 跳过已完成的匹配步骤，直接跑 BA
    python run_pipeline_panoramic_01.py --skip_match
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ── 固定路径 ──────────────────────────────────────────────────────────────────
IMG_DIR          = PROJECT_ROOT / "blender-file/stereo_panoramic_01"
WEIGHTS          = PROJECT_ROOT / "matchmodel/SuperPointPretrainedNetwork/superpoint_v1.pth"
RESULT_DIR       = PROJECT_ROOT / "stereo_calib/result"
MATCH_JSON       = RESULT_DIR / "panoramic_01_matches.json"
BA_RESULT_JSON   = RESULT_DIR / "panoramic_01_ba_result.json"
PLOT_PNG         = RESULT_DIR / "panoramic_01_ba_history.png"
GT_FILE          = IMG_DIR / "camera_params_0000.json"   # 所有帧相机参数相同，取第 0 帧
BA_BIN           = PROJECT_ROOT / "stereo_calib/build/bin/run_offline_stereo_ba"
MATCH_SCRIPT     = PROJECT_ROOT / "stereo_calib/scripts/superpoint_stereo_match.py"
PLOT_SCRIPT      = PROJECT_ROOT / "stereo_calib/scripts/plot_ba_history.py"
TMP_CAM_PARAMS   = IMG_DIR / "camera_params.json"   # 临时文件，供匹配脚本读取 GT


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def log_step(msg: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


def run_cmd(cmd: list, cwd: Path = PROJECT_ROOT) -> None:
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    ret = subprocess.run(cmd, cwd=cwd)
    if ret.returncode != 0:
        sys.exit(f"\n命令执行失败（exit code {ret.returncode}）")


def create_tmp_camera_params() -> bool:
    """
    从逐帧参数文件提取 left/right/extrinsics 写入临时 camera_params.json。
    superpoint_stereo_match.py 会读取该文件，将 GT 嵌入 matches.json 供 C++ BA 对比。
    返回 True 表示新建了文件（需要在完成后清理）。
    """
    if TMP_CAM_PARAMS.exists():
        print(f"  camera_params.json 已存在，跳过创建。")
        return False

    if not GT_FILE.exists():
        print(f"  警告：GT 文件不存在 {GT_FILE}，跳过嵌入 GT。")
        return False

    with open(GT_FILE) as f:
        raw = json.load(f)

    cam = {
        "image_size": raw.get("image_size", {}),
        "left":       raw["left"],
        "right":      raw["right"],
        "extrinsics": raw["extrinsics"],
    }
    with open(TMP_CAM_PARAMS, "w") as f:
        json.dump(cam, f, indent=2)

    print(f"  已创建临时文件：{TMP_CAM_PARAMS}")
    return True


# ── 参数解析 ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="stereo_panoramic_01 全流程标定脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 流程控制
    flow = p.add_argument_group("流程控制")
    flow.add_argument("--skip_match", action="store_true",
                      help="跳过特征匹配（复用已有 matches JSON）")
    flow.add_argument("--skip_ba",    action="store_true",
                      help="跳过 BA 优化（复用已有 BA 结果 JSON）")
    flow.add_argument("--no_plot",    action="store_true",
                      help="跳过可视化步骤")

    # SuperPoint 参数
    sp = p.add_argument_group("SuperPoint 特征匹配参数")
    sp.add_argument("--nn_thresh",    type=float, default=0.7,
                    help="描述子 L2 距离阈值")
    sp.add_argument("--conf_thresh",  type=float, default=0.015,
                    help="关键点置信度阈值")
    sp.add_argument("--nms_dist",     type=int,   default=4,
                    help="NMS 抑制半径（像素）")
    sp.add_argument("--cuda",         action="store_true",
                    help="SuperPoint 推理使用 CUDA")

    # BA 参数
    ba = p.add_argument_group("Bundle Adjustment 参数")
    ba.add_argument("--max_iter",            type=int,   default=200,
                    help="全局 BA 最大迭代次数")
    ba.add_argument("--incremental_max_iter", type=int,  default=20,
                    help="增量 BA 最大迭代次数")
    ba.add_argument("--global_opt_interval", type=int,   default=5,
                    help="每隔多少帧触发一次全局 BA")
    ba.add_argument("--huber",               type=float, default=1.0,
                    help="Huber 损失函数阈值（像素）")
    ba.add_argument("--min_track_len",       type=int,   default=3,
                    help="有效轨迹最小长度")
    ba.add_argument("--min_pair_inliers",    type=int,   default=12,
                    help="图像对最少内点数")
    ba.add_argument("--max_score",           type=float, default=1.0,
                    help="匹配点最大得分阈值（过滤低质量匹配，越小越严格）")
    ba.add_argument("--fix_distortion",      action="store_true",
                    help="固定畸变参数不优化")

    return p.parse_args()


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # 前置检查
    missing = []
    if not IMG_DIR.exists():
        missing.append(f"图像目录：{IMG_DIR}")
    if not args.skip_match and not WEIGHTS.exists():
        missing.append(f"SuperPoint 权重：{WEIGHTS}")
    if not args.skip_ba and not BA_BIN.exists():
        missing.append(f"BA 可执行文件：{BA_BIN}（请先编译：cd stereo_calib/build && make）")
    if missing:
        sys.exit("错误，以下文件/目录不存在：\n  " + "\n  ".join(missing))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    created_tmp = False

    # ── 步骤 1：SuperPoint 特征匹配 ───────────────────────────────────────────
    if not args.skip_match:
        log_step("步骤 1/3：SuperPoint 特征匹配")

        created_tmp = create_tmp_camera_params()

        cmd = [
            sys.executable, str(MATCH_SCRIPT),
            "--img_dir",    str(IMG_DIR),
            "--weights",    str(WEIGHTS),
            "--output",     str(MATCH_JSON),
            "--nn_thresh",  str(args.nn_thresh),
            "--conf_thresh", str(args.conf_thresh),
            "--nms_dist",   str(args.nms_dist),
        ]
        if args.cuda:
            cmd.append("--cuda")

        run_cmd(cmd)
        print(f"\n匹配结果已写入：{MATCH_JSON}")
    else:
        if not MATCH_JSON.exists():
            sys.exit(f"--skip_match 已指定，但文件不存在：{MATCH_JSON}")
        print(f"跳过特征匹配，使用：{MATCH_JSON}")

    # ── 步骤 2：增量式 Bundle Adjustment ─────────────────────────────────────
    if not args.skip_ba:
        log_step("步骤 2/3：增量式 Bundle Adjustment 优化")

        cmd = [
            str(BA_BIN),
            "--input",               str(MATCH_JSON),
            "--output",              str(BA_RESULT_JSON),
            "--max_iter",            str(args.max_iter),
            "--incremental_max_iter", str(args.incremental_max_iter),
            "--global_opt_interval", str(args.global_opt_interval),
            "--huber",               str(args.huber),
            "--min_track_len",       str(args.min_track_len),
            "--min_pair_inliers",    str(args.min_pair_inliers),
            "--max_score",           str(args.max_score),
        ]
        if args.fix_distortion:
            cmd.append("--fix_distortion")
        if GT_FILE.exists():
            cmd += ["--gt_param_file", str(GT_FILE)]

        # C++ 程序会在 CWD 下查找 stereo_calib/data/example_init_params.txt
        run_cmd(cmd, cwd=PROJECT_ROOT)
        print(f"\nBA 结果已写入：{BA_RESULT_JSON}")
    else:
        if not BA_RESULT_JSON.exists():
            sys.exit(f"--skip_ba 已指定，但文件不存在：{BA_RESULT_JSON}")
        print(f"跳过 BA 优化，使用：{BA_RESULT_JSON}")

    # ── 步骤 3：可视化 ────────────────────────────────────────────────────────
    if not args.no_plot:
        log_step("步骤 3/3：BA 优化历史可视化")

        cmd = [
            sys.executable, str(PLOT_SCRIPT),
            "--input",  str(BA_RESULT_JSON),
            "--output", str(PLOT_PNG),
        ]
        run_cmd(cmd)
        print(f"\n可视化图片已保存：{PLOT_PNG}")

    # ── 清理临时文件 ──────────────────────────────────────────────────────────
    if created_tmp and TMP_CAM_PARAMS.exists():
        TMP_CAM_PARAMS.unlink()
        print(f"\n已清理临时文件：{TMP_CAM_PARAMS}")

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    log_step("全流程完成")
    print(f"  特征匹配结果：{MATCH_JSON}")
    print(f"  BA 优化结果：{BA_RESULT_JSON}")
    if not args.no_plot:
        print(f"  收敛曲线图：{PLOT_PNG}")


if __name__ == "__main__":
    main()

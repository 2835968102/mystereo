#!/usr/bin/env python3
"""
通用立体标定全流程脚本

从项目根目录运行（确保已激活 stereo-calib-vis conda 环境并安装了 torch/opencv）：

    python run_pipeline.py [选项]

完整三步流程：
  1. SuperPoint 特征提取与匹配（所有图像对）
  2. C++ 增量式 Bundle Adjustment 优化
  3. BA 优化历史可视化

常用示例：
    # 完整流程（默认场景 panoramic_01）
    python run_pipeline.py

    # 指定其他场景
    python run_pipeline.py --scene panoramic_02

    # 使用 CUDA 加速特征匹配
    python run_pipeline.py --scene panoramic_01 --cuda

    # 跳过已完成的匹配步骤，直接跑 BA
    python run_pipeline.py --scene panoramic_01 --skip_match
"""

from __future__ import annotations

import argparse

from stereo_calib_py.pipeline import run_pipeline


def parse_args():
    """定义单次 pipeline 的命令行参数。

    这个顶层脚本只负责 CLI 兼容性；真正执行逻辑在
    `stereo_calib_py.pipeline.run_pipeline` 中。
    """
    p = argparse.ArgumentParser(
        description="通用立体标定全流程脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 场景选择
    p.add_argument("--dataset_mode", choices=["project", "kitti2015", "kitti_raw_aggregate"], default="project",
                   help="数据集模式：project 使用 blender-file/stereo_{scene}；kitti2015 使用 KITTI RAW 单帧；kitti_raw_aggregate 直接消费汇总 matches JSON")
    p.add_argument("--scene", default="panoramic_01",
                   help="project 模式下为场景名（如 panoramic_02）；kitti2015 模式下为帧号（如 0000000000）")
    p.add_argument("--kitti_root_dir", default=None,
                   help="KITTI RAW drive 根目录，需包含 image_00/data 和 image_01/data")
    p.add_argument("--match_json", default=None,
                   help="已有 matches JSON 路径（供 kitti_raw_aggregate 等模式复用）")
    p.add_argument("--result_prefix", default=None,
                   help="输出结果文件名前缀（供聚合模式使用）")
    p.add_argument("--pixel_tag", default="3px",
                   help="输出结果文件名中的像素阈值标签（如 3px、2px、1px）")

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
    ba.add_argument("--max_iter",             type=int,   default=200,
                    help="全局 BA 最大迭代次数")
    ba.add_argument("--incremental_max_iter", type=int,   default=20,
                    help="增量 BA 最大迭代次数")
    ba.add_argument("--global_opt_interval",  type=int,   default=5,
                    help="每隔多少帧触发一次全局 BA")
    ba.add_argument("--huber",                type=float, default=1.0,
                    help="Huber 损失函数阈值（像素）")
    ba.add_argument("--min_track_len",        type=int,   default=3,
                    help="有效轨迹最小长度")
    ba.add_argument("--min_pair_inliers",     type=int,   default=12,
                    help="图像对最少内点数")
    ba.add_argument("--max_score",            type=float, default=1.0,
                    help="匹配点最大得分阈值（过滤低质量匹配，越小越严格）")
    ba.add_argument("--fix_distortion",       action="store_true",
                    help="固定畸变参数不优化")

    return p.parse_args()


def main():
    """CLI 入口。"""
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()

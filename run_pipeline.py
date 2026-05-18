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
    p.add_argument("--dataset_mode", choices=["project", "kitti2015", "kitti_raw_aggregate", "kitti_raw_sequence"], default="project",
                   help="数据集模式：project 使用 blender-file/stereo_{scene}；kitti2015 使用 KITTI RAW 单帧；kitti_raw_aggregate 直接消费汇总 matches JSON；kitti_raw_sequence 使用 KITTI RAW 连续帧")
    p.add_argument("--scene", default="panoramic_01",
                   help="project 模式下为场景名（如 panoramic_02）；kitti2015 模式下为帧号（如 0000000000）")
    p.add_argument("--kitti_root_dir", default=None,
                   help="KITTI RAW drive 根目录，需包含 image_00/data 和 image_01/data")
    p.add_argument("--match_json", default=None,
                   help="已有 matches JSON 路径（供 kitti_raw_aggregate 等模式复用）")
    p.add_argument("--result_prefix", default=None,
                   help="输出结果文件名前缀（供聚合模式使用）")
    p.add_argument("--gt_param_file", default=None,
                   help="显式指定 GT 相机参数文件，覆盖自动生成/自动解析路径")
    p.add_argument("--init_param_file", default=None,
                   help="显式指定 BA 初始相机参数文件（KITTI 专用 BA 会使用）")
    p.add_argument("--frame_poses_file", default=None,
                   help="显式指定帧位姿 JSON（KITTI 专用 BA 会使用）")
    p.add_argument("--left_img_dir", default=None,
                   help="KITTI RAW sequence 左目图像目录")
    p.add_argument("--right_img_dir", default=None,
                   help="KITTI RAW sequence 右目图像目录")
    p.add_argument("--pixel_tag", default="3px",
                   help="输出结果文件名中的像素阈值标签（如 3px、2px、1px）")
    p.add_argument("--output_timestamp", default=None,
                   help="输出文件名中的时间戳；默认使用当前时间（格式：YYYYmmdd_HHMMSS_microseconds）")

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
    ba.add_argument("--per_frame_max_iter",   type=int,   default=5,
                    help="KITTI 逐帧本地校正最大迭代次数")
    ba.add_argument("--baseline_prior",       type=float, default=None,
                    help="基线先验权重")
    ba.add_argument("--tx_prior",             type=float, default=None,
                    help="x 方向平移先验权重（KITTI 专用）")
    ba.add_argument("--focal_prior",          type=float, default=None,
                    help="焦距先验权重（KITTI 专用）")
    ba.add_argument("--focal_lower_scale",    type=float, default=None,
                    help="焦距下界缩放因子（KITTI 专用）")
    ba.add_argument("--focal_upper_scale",    type=float, default=None,
                    help="焦距上界缩放因子（KITTI 专用）")
    ba.add_argument("--outlier_threshold",    type=float, default=None,
                    help="外点剔除阈值（像素）")
    ba.add_argument("--outlier_rounds",       type=int,   default=None,
                    help="外点剔除最大轮数")
    ba.add_argument("--max_reproj_error",     type=float, default=None,
                    help="最大可接受重投影误差")
    ba.add_argument("--enable_per_frame_correction", action="store_true",
                    help="启用逐帧本地校正（KITTI 专用 BA 支持）")
    ba.add_argument("--reset_camera_params_each_ba_round", action="store_true",
                    help="每轮 BA 前重置相机参数到初始值（KITTI 专用 BA 支持）")

    return p.parse_args()


def main():
    """CLI 入口。"""
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()

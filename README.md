# Stereo Camera Calibration Pipeline

基于 SuperPoint 特征点匹配的立体相机标定工具链，支持使用 Blender 生成仿真数据。整体流程分三步：

1. **特征点匹配**（Python）：用 SuperPoint 神经网络提取关键点并两两匹配，输出匹配点 JSON。
2. **参数优化**（C++）：以匹配点为约束，用 Ceres 增量式 Bundle Adjustment 优化立体相机内外参。
3. **结果可视化**（Python）：绘制 BA 优化历史曲线。

---

## 项目结构

```
.
├── run_pipeline_panoramic_01.py       # 全流程一键运行脚本（推荐入口）
├── environment.yml                    # Conda 可视化环境（stereo-calib-vis）
├── blender-file/
│   └── stereo_panoramic_01/          # 仿真图像数据目录（含 camera_params_0000.json）
├── matchmodel/
│   └── SuperPointPretrainedNetwork/
│       └── superpoint_v1.pth          # SuperPoint 预训练权重
└── stereo_calib/
    ├── scripts/
    │   ├── blender.py                 # Blender 仿真数据生成
    │   ├── superpoint_stereo_match.py # 特征匹配脚本（步骤 1）
    │   ├── plot_ba_history.py         # BA 优化历史可视化（步骤 3）
    │   ├── visualize_matches.py       # 匹配点可视化工具
    │   └── eval_stereo.py             # 与 GT 对比评估工具
    ├── src/
    │   ├── run_offline_stereo_ba.cc   # BA 优化程序入口（步骤 2）
    │   ├── offline_stereo_ba.cc/.h    # 增量式 BA 核心逻辑
    │   ├── stereo_optimizer.cc/.h     # Ceres 优化器
    │   ├── stereo_factors.cc/.h       # 重投影误差因子
    │   └── stereo_types.h             # 数据结构定义
    ├── result/                        # 输出目录（matches JSON、BA 结果、可视化图）
    ├── data/                          # 输入数据（matches.json 等）
    └── CMakeLists.txt
```

---

## 环境依赖

### Python 环境（Conda）

```bash
# 创建环境（仅需一次）
conda env create -f environment.yml

# 激活环境
conda activate stereo-calib-vis
```

特征匹配步骤（SuperPoint）还需要 PyTorch 和 OpenCV：

```bash
pip install torch torchvision opencv-python
```

### C++ 编译

依赖：CMake ≥ 3.10、OpenCV ≥ 4.3、Ceres Solver。

```bash
mkdir -p stereo_calib/build && cd stereo_calib/build
cmake .. && make -j$(nproc)
```

编译产物位于 `stereo_calib/build/bin/run_offline_stereo_ba`。

---

## 快速开始：全流程一键运行

`run_pipeline_panoramic_01.py` 是推荐入口，自动串联特征匹配 → BA 优化 → 可视化三个步骤。

**固定路径（无需手动指定）：**

| 路径 | 说明 |
|------|------|
| `blender-file/stereo_panoramic_01/` | 输入图像目录 |
| `matchmodel/SuperPointPretrainedNetwork/superpoint_v1.pth` | SuperPoint 权重 |
| `stereo_calib/result/panoramic_01_matches.json` | 匹配结果输出 |
| `stereo_calib/result/panoramic_01_ba_result.json` | BA 结果输出 |
| `stereo_calib/result/panoramic_01_ba_history.png` | 优化历史图输出 |

### 典型用法

```bash
conda activate stereo-calib-vis

# 完整流程（特征匹配 + BA 优化 + 可视化）
python run_pipeline_panoramic_01.py

# 跳过特征匹配，直接跑 BA（复用已有 matches JSON）
python run_pipeline_panoramic_01.py --skip_match \
    --max_iter 40 \
    --incremental_max_iter 10 \
    --global_opt_interval 5 \
    --min_pair_inliers 15 \
    --fix_distortion \
    --max_score 0.5 \
    --min_track_len 7

# 使用 CUDA 加速特征匹配
python run_pipeline_panoramic_01.py --cuda
```

### 全部参数说明

**流程控制：**

| 参数 | 说明 |
|------|------|
| `--skip_match` | 跳过特征匹配，复用已有 matches JSON |
| `--skip_ba` | 跳过 BA 优化，复用已有 BA 结果 JSON |
| `--no_plot` | 跳过可视化步骤 |

**SuperPoint 特征匹配参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--nn_thresh` | `0.7` | 描述子 L2 距离阈值，越小匹配越严格 |
| `--conf_thresh` | `0.015` | 关键点置信度阈值，越大检测点越少但更可靠 |
| `--nms_dist` | `4` | NMS 抑制半径（像素） |
| `--cuda` | 关闭 | 启用 GPU 加速推理 |

**Bundle Adjustment 参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_iter` | `200` | 全局 BA 最大迭代次数 |
| `--incremental_max_iter` | `20` | 增量 BA 最大迭代次数 |
| `--global_opt_interval` | `5` | 每隔多少帧触发一次全局 BA |
| `--min_pair_inliers` | `12` | 图像对最少内点数，低于此阈值的图像对被过滤 |
| `--min_track_len` | `3` | 有效轨迹最小长度，过短的轨迹不参与优化 |
| `--max_score` | `1.0` | 匹配点最大得分阈值（L2 距离），越小越严格 |
| `--huber` | `1.0` | Huber 损失函数阈值（像素） |
| `--fix_distortion` | 关闭 | 固定畸变参数不参与优化 |

---

## 数据格式

**输入（`matches.json`）：**

```json
{
  "left":  { "fx": 2138.9, "fy": 2138.9, "cx": 1100.0, "cy": 800.0,
             "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0 },
  "right": { ... },
  "extrinsics": { "R": [1,0,0, 0,1,0, 0,0,1], "t": [-0.2, 0.0, 0.0] },
  "pairs": [
    {
      "left_image": "left_000.png",
      "right_image": "right_000.png",
      "matches": [
        { "left": [320.5, 240.1], "right": [210.3, 240.1], "score": 0.23 }
      ]
    }
  ]
}
```

`score` 为描述子 L2 距离，范围 0–2，**越小表示匹配越可靠**。`--max_score` 用于过滤该字段。

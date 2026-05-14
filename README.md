# Stereo Camera Calibration Pipeline

基于 SuperPoint 特征点匹配的立体相机标定工具链，支持使用 Blender 生成仿真数据。整体流程分三步：

1. **特征点匹配**（Python）：用 SuperPoint 神经网络提取关键点并两两匹配，输出匹配点 JSON。
2. **参数优化**（C++）：以匹配点为约束，用 Ceres 增量式 Bundle Adjustment 优化立体相机内外参。
3. **结果可视化**（Python）：绘制 BA 优化历史曲线。

---

## 项目结构

```
.
├── run_pipeline.py                   # 单次全流程 CLI 入口
├── run_experiment.py                 # YAML 驱动的批量实验入口
├── configs/experiments/              # 批量实验配置
├── stereo_calib_py/                  # Python pipeline 复用逻辑
├── environment.yml                   # Conda 可视化环境（stereo-calib-vis）
├── blender-file/
│   └── stereo_panoramic_01/          # 仿真图像数据目录（含 camera_params_0000.json）
├── matchmodel/
│   └── SuperPointPretrainedNetwork/
│       └── superpoint_v1.pth         # SuperPoint 预训练权重
└── stereo_calib/
    ├── scripts/
    │   ├── blender.py                 # Blender 仿真数据生成
    │   ├── superpoint_stereo_match.py # 特征匹配脚本（步骤 1）
    │   ├── plot_ba_history.py         # BA 优化历史可视化（步骤 3）
    │   ├── visualize_matches.py       # 匹配点可视化工具
    │   └── eval_stereo.py             # 与 GT 对比评估工具
    ├── src/
    │   ├── stereo_types.h             # 数据结构定义
    │   ├── stereo_io.h/cc             # JSON 序列化 + 相机参数文件加载
    │   ├── stereo_eval.h/cc           # GT 对比 + FOV 合理性检查
    │   ├── stereo_factors.h/cc        # 所有 Ceres 代价函数
    │   ├── track_builder.h/cc         # Track 构建 + 帧/点初始化
    │   ├── stereo_optimizer.h/cc      # 单对 Ceres BA 优化器
    │   ├── offline_ba_common.h/cc     # 离线 BA 入口共享辅助逻辑
    │   ├── run_stereo_calib.cc        # 单对 BA 入口程序
    │   └── run_offline_stereo_ba.cc   # 增量式多帧 BA 入口程序
    ├── data/
    │   └── example_init_params.txt    # 相机初始参数（BA 优化初值）
    ├── result/                        # 输出目录（matches JSON、BA 结果、可视化图）
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

依赖：CMake ≥ 3.10、OpenCV ≥ 4.3、Ceres Solver、nlohmann/json（已包含在 `3rdparty/`）。

```bash
mkdir -p build && cd build
cmake ../stereo_calib && make -j$(nproc)
```

编译产物位于 `build/bin/`：

| 可执行文件 | 用途 |
|-----------|------|
| `run_offline_stereo_ba` | 增量式多帧 BA 标定（**主要使用**） |
| `run_offline_stereo_ba_kitti` | KITTI RAW 专用增量式 BA 标定 |
| `run_stereo_calib` | 单对 BA 标定 |

---

## 快速开始：全流程一键运行

`run_pipeline.py` 是推荐入口，自动串联特征匹配 → BA 优化 → 可视化三个步骤。

### 典型用法

```bash
conda activate stereo-calib-vis

# 完整流程（默认 project 数据集的 panoramic_01）
python run_pipeline.py

# 指定 project 数据集其他场景
python run_pipeline.py --dataset_mode project --scene panoramic_02

# 运行 KITTI RAW 单帧模式，并自动与官方标定值对比
python run_pipeline.py \
    --dataset_mode kitti2015 \
    --scene 0000000000 \
    --kitti_root_dir /path/to/kitti/raw/drive/root

# 说明：需要能从 kitti_root_dir 推导出 sibling 标定目录
# .../2011_09_26_calib/2011_09_26/calib_cam_to_cam.txt
# 程序会自动转换为仓库内部 GT JSON，并通过 --gt_param_file 接入 BA 对比

# 运行 KITTI RAW 连续帧配置（等价替代旧 run_kitti_raw_city_2011_09_26_drive_0001.sh）
python3 run_experiment.py configs/experiments/kitti_raw_city_0001.yaml

# 使用 CUDA 加速特征匹配
python run_pipeline.py --cuda

# 在 KITTI 模式下使用 CUDA
python run_pipeline.py \
    --dataset_mode kitti2015 \
    --scene 000000 \
    --kitti_root_dir /path/to/data_scene_flow/training \
    --cuda

# 跳过特征匹配，直接跑 BA（复用已有 matches JSON）
python run_pipeline.py --skip_match

# 跳过匹配和 BA，只画可视化图
python run_pipeline.py --skip_match --skip_ba

# 调参示例：严格匹配 + 固定畸变
python run_pipeline.py \
    --scene panoramic_01 \
    --skip_match \
    --max_iter 40 \
    --incremental_max_iter 10 \
    --global_opt_interval 5 \
    --min_pair_inliers 15 \
    --fix_distortion \
    --max_score 0.5 \
    --min_track_len 7
```

### 批量实验

批量实验推荐改 YAML 配置，而不是直接改 shell 脚本：

```bash
# 运行默认 panoramic 阈值对比实验
python3 run_experiment.py configs/experiments/panoramic_threshold_sweep.yaml

# 只打印将要执行的命令，不真正运行
python3 run_experiment.py configs/experiments/panoramic_threshold_sweep.yaml \
    --dry-run --no-build --scene panoramic_01

# 运行 KITTI RAW 连续帧实验
python3 run_experiment.py configs/experiments/kitti_raw_city_0001.yaml
```

默认 panoramic 配置在 `configs/experiments/panoramic_threshold_sweep.yaml`。KITTI RAW 示例配置在 `configs/experiments/kitti_raw_city_0001.yaml`。场景列表、阈值实验、BA 参数、是否跳过匹配、后处理画图都在这些 YAML 文件里维护。

### 路径约定

`run_pipeline.py` 会根据 `--dataset_mode` 和 `--scene` 自动生成输入/输出路径。

#### project 模式

| 路径 | 说明 |
|------|------|
| `blender-file/stereo_{scene}/` | 输入图像目录 |
| `blender-file/stereo_{scene}/camera_params_0000.json` | GT 参数（可选，有则自动对比） |
| `matchmodel/SuperPointPretrainedNetwork/superpoint_v1.pth` | SuperPoint 权重 |
| `stereo_calib/result/match_points/{scene}_matches_gtpose_dsym_le_{pixel_tag}.json` | 匹配结果输出 |
| `stereo_calib/result/ba_results/{scene}_ba_result_le_{pixel_tag}.json` | BA 结果输出 |
| `stereo_calib/result/{scene}_ba_history_gtpose_dsym_le_{pixel_tag}.png` | 优化历史图输出 |

#### kitti2015 模式

| 路径 | 说明 |
|------|------|
| `{kitti_root_dir}/image_00/data/{scene}.png` | 左目图像 |
| `{kitti_root_dir}/image_01/data/{scene}.png` | 右目图像 |
| `stereo_calib/result/match_points/kitti2015_{scene}_matches.json` | 匹配结果输出 |
| `stereo_calib/result/gt_params/kitti2015_{scene}_gt_camera.json` | 自动生成的 KITTI GT 相机参数 |
| `stereo_calib/result/ba_results/kitti2015_{scene}_ba_result.json` | BA 结果输出（含 `gt_source` / `diff_vs_gt`） |
| `stereo_calib/result/kitti2015_{scene}_ba_history.png` | 优化历史图输出 |

#### kitti_raw_sequence 模式

| 路径 | 说明 |
|------|------|
| `--left_img_dir` | KITTI RAW 左目连续帧目录，如 `image_00/data` |
| `--right_img_dir` | KITTI RAW 右目连续帧目录，如 `image_01/data` |
| `build/bin/run_offline_stereo_ba_kitti` | KITTI 专用 BA 程序 |
| `stereo_calib/result/match_points/{result_prefix}_conf_thresh=..._matches_raw.json` | 匹配结果输出 |
| `stereo_calib/result/ba_results/{result_prefix}_conf_thresh=..._ba_result_raw.json` | BA 结果输出 |
| `stereo_calib/result/{result_prefix}_conf_thresh=..._ba_history_raw.png` | 优化历史图输出 |

---

## 参数说明

### 流程控制

| 参数 | 说明 |
|------|------|
| `--dataset_mode` | 数据集模式：`project`、`kitti2015`、`kitti_raw_aggregate` 或 `kitti_raw_sequence` |
| `--scene` | `project` 模式下为场景名称；KITTI 模式下为 scene/frame id 或输出标签 |
| `--kitti_root_dir` | KITTI RAW drive 根目录，需包含 `image_00/data` 和 `image_01/data` |
| `--left_img_dir` / `--right_img_dir` | KITTI RAW sequence 模式下显式指定左右目图像目录 |
| `--gt_param_file` | 显式指定 GT 相机参数文件，覆盖自动生成路径 |
| `--init_param_file` | 显式指定 BA 初始参数文件，KITTI 专用 BA 会使用 |
| `--skip_match` | 跳过特征匹配，复用已有 matches JSON |
| `--skip_ba` | 跳过 BA 优化，复用已有 BA 结果 JSON |
| `--no_plot` | 跳过可视化步骤 |

### SuperPoint 特征匹配参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--nn_thresh` | `0.7` | 描述子 L2 距离阈值，越小匹配越严格 |
| `--conf_thresh` | `0.015` | 关键点置信度阈值，越大检测点越少但更可靠 |
| `--nms_dist` | `4` | NMS 抑制半径（像素） |
| `--cuda` | 关闭 | 启用 GPU 加速推理 |

### Bundle Adjustment 参数

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
| `--baseline_prior` | BA 程序默认 | 基线先验权重 |
| `--tx_prior` | BA 程序默认 | x 方向平移先验权重，KITTI 专用 |
| `--focal_prior` | BA 程序默认 | 焦距先验权重，KITTI 专用 |
| `--focal_lower_scale` / `--focal_upper_scale` | BA 程序默认 | 焦距上下界缩放，KITTI 专用 |
| `--enable_per_frame_correction` | 关闭 | 启用 KITTI 专用 BA 的逐帧本地校正 |
| `--reset_camera_params_each_ba_round` | 关闭 | 每轮 BA 前重置相机参数到初始值 |

### 快速验证

改动入口脚本或 C++ BA 后，可以运行轻量回归检查：

```bash
scripts/check_project.sh
```

它会检查 Python 语法、`run_pipeline.py --help`、panoramic/KITTI RAW 配置 dry-run，并编译 C++ 目标。

---

## 单独运行各步骤

如果不使用 `run_pipeline.py`，也可以手动运行每个步骤。

### 步骤 1：特征匹配

```bash
conda activate stereo-calib-vis

python stereo_calib/scripts/superpoint_stereo_match.py \
    --dataset_mode project \
    --img_dir blender-file/stereo_panoramic_01 \
    --scene panoramic_01 \
    --weights matchmodel/SuperPointPretrainedNetwork/superpoint_v1.pth \
    --output stereo_calib/result/panoramic_01_matches.json \
    --nn_thresh 0.7 \
    --conf_thresh 0.015

# KITTI Stereo 2015 training 单个 scene（方案 B）
python stereo_calib/scripts/superpoint_stereo_match.py \
    --dataset_mode kitti2015 \
    --img_dir /path/to/data_scene_flow/training \
    --scene 000000 \
    --weights matchmodel/SuperPointPretrainedNetwork/superpoint_v1.pth \
    --output stereo_calib/result/kitti2015_000000_matches.json \
    --nn_thresh 0.7 \
    --conf_thresh 0.015
```

### 步骤 2：BA 优化

```bash
# 确保已编译
mkdir -p build && cd build && cmake ../stereo_calib && make -j$(nproc) && cd ..

# 运行 BA
./build/bin/run_offline_stereo_ba \
    --input stereo_calib/result/panoramic_01_matches.json \
    --output stereo_calib/result/panoramic_01_ba_result.json \
    --gt_param_file blender-file/stereo_panoramic_01/camera_params_0000.json \
    --max_iter 200 \
    --incremental_max_iter 20 \
    --global_opt_interval 5
```

`run_offline_stereo_ba` 的全部参数：

```
--input <matches.json>              输入匹配文件（必须）
--output <result.json>              输出结果文件（必须）
--gt_param_file <path>              GT 参数文件（可选，用于对比评估）
--max_iter 200                      全局 BA 最大迭代次数
--incremental_max_iter 20           增量 BA 最大迭代次数
--global_opt_interval 5             全局 BA 触发间隔（帧数）
--min_track_len 3                   有效轨迹最小长度
--huber 1.0                         Huber 损失阈值
--max_score 1.0                     匹配点得分过滤阈值
--min_pair_inliers 12               图像对最少内点数
--min_pair_inlier_ratio 0.35        图像对最低内点比
--fix_distortion                    固定畸变参数
--aspect_ratio_prior 1.0            宽高比先验权重
--baseline_prior 10.0               基线先验权重
--max_reproj_error 20.0             最大可接受重投影误差
--outlier_threshold 2.0             外点剔除阈值（像素）
--outlier_rounds 100                外点剔除最大轮数
```

初始参数固定从 `stereo_calib/data/example_init_params.txt` 加载。

### 步骤 3：可视化

```bash
conda activate stereo-calib-vis

# 保存为图片
python stereo_calib/scripts/plot_ba_history.py \
    --input stereo_calib/result/panoramic_01_ba_result.json \
    --output stereo_calib/result/panoramic_01_ba_history.png

# 或弹窗显示
python stereo_calib/scripts/plot_ba_history.py \
    --input stereo_calib/result/panoramic_01_ba_result.json
```

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

`left`/`right`/`extrinsics` 为 GT 真值（可选），存在时自动用于对比评估。`score` 为描述子 L2 距离，范围 0–2，**越小表示匹配越可靠**。`--max_score` 用于过滤该字段。

**输出（`ba_result.json`）：**

```json
{
  "left": { ... },
  "right": { ... },
  "extrinsics": { "R": [...], "t": [...] },
  "success": true,
  "num_tracks": 1234,
  "num_observations": 5678,
  "num_frames": 20,
  "init_reproj_error": 5.2,
  "final_reproj_error": 0.3,
  "optimization_history": [
    { "stage": "Incremental BA - Frame 2", "reproj_error": 2.1, "camera": {...} },
    { "stage": "Periodic Global BA - Frame 6", "reproj_error": 1.5, "camera": {...} },
    { "stage": "Final Global BA", "reproj_error": 0.3, "camera": {...} }
  ]
}
```

`optimization_history` 中的阶段类型：
- `"Incremental BA - Frame N"` — 增量 BA，逐帧加入后的局部优化
- `"Periodic Global BA - Frame N"` — 每隔 `--global_opt_interval` 帧的全局 BA
- `"Final Global BA"` — 所有帧注册完毕后的最终全局 BA
- `"Outlier Rejection BA - Round N"` — 后 BA 外点剔除轮次

---

## C++ 代码结构

```
stereo_types.h           数据结构定义（Intrinsics, StereoExtrinsics, StereoCamera, ...）
    │
stereo_io.h/cc           JSON 序列化 + 相机参数文件加载
stereo_eval.h/cc         GT 对比工具 + FOV 合理性检查
stereo_factors.h/cc      所有 Ceres 代价函数
    │                    （LeftReproj, RightReproj, TrackReproj, BaselinePrior, AspectRatioPrior）
    │
track_builder.h/cc       Track 构建（union-find）+ 帧旋转初始化 + 3D 点初始化
    │
stereo_optimizer.h/cc    单对 BA 优化器（StereoOptimizer）
offline_stereo_ba.h/cc   增量式多帧 BA 协调者（OfflineStereoBA）
    │
run_stereo_calib.cc      单对 BA 入口
run_offline_stereo_ba.cc 增量式 BA 入口
```

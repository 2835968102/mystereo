# Stereo Camera Calibration Pipeline

基于 SuperPoint 特征匹配和 Ceres Bundle Adjustment 的立体相机标定工具链。当前项目同时支持项目内 Blender 仿真数据、KITTI 单帧数据和 KITTI RAW 连续帧数据。

流程分三步：

1. SuperPoint 特征匹配，生成 matches JSON。
2. C++ 增量式 BA，优化双目内外参。
3. Python 可视化，绘制 BA 历史曲线和实验对比图。

## 推荐入口

优先使用 Python 入口和 YAML 配置，不再直接维护复杂 shell 流程。

| 入口 | 用途 |
|------|------|
| `python3 run_experiment.py configs/experiments/panoramic_threshold_sweep.yaml` | 批量运行 panoramic 阈值对比实验 |
| `python3 run_experiment.py configs/experiments/kitti_raw_city_0001.yaml` | 批量运行 KITTI RAW City sync drive 实验 |
| `python3 run_pipeline.py ...` | 单次 scene / 单次参数组合 |
| `./run_all_test.sh` | 兼容旧入口，薄包装到 panoramic YAML 实验 |
| `./run_kitti_raw_city_2011_09_26_drive.sh` | 兼容入口，薄包装到 KITTI RAW YAML 实验 |

`legacy/` 中保留了旧版完整 shell 脚本，仅用于追溯历史运行方式；新增实验应放到 `configs/experiments/*.yaml`。

## 项目结构

```text
.
├── run_pipeline.py                    # 单次 pipeline CLI，只负责参数定义
├── run_experiment.py                  # YAML 驱动的批量实验入口
├── configs/experiments/               # 批量实验配置
│   ├── panoramic_threshold_sweep.yaml
│   └── kitti_raw_city_0001.yaml
├── stereo_calib_py/                   # Python 编排层
│   ├── pipeline.py                    # 单次流程：校验、匹配、BA、画图
│   ├── paths.py                       # dataset mode 到输入/输出路径的规则
│   ├── kitti.py                       # KITTI 标定转换
│   ├── commands.py                    # 子进程和日志辅助函数
│   └── project.py                     # 项目根目录
├── stereo_calib/
│   ├── CMakeLists.txt
│   ├── data/                          # 示例初始参数和位姿输入
│   ├── scripts/                       # SuperPoint、画图、Blender、诊断脚本
│   └── src/
│       ├── coordinators/              # 主优化流程编排
│       ├── services/                  # track/init/BA/outlier/eval 服务
│       ├── core/                      # 相机数学辅助函数
│       ├── offline_ba_common.*        # 离线 BA 入口共享逻辑
│       ├── run_offline_stereo_ba.cc   # project / 通用离线 BA 入口
│       ├── run_offline_stereo_ba_kitti.cc
│       ├── run_stereo_calib.cc        # 单对图像 BA 入口
│       ├── track_builder.*            # track 构建、帧初始化、点初始化
│       └── stereo_*                   # IO、评估、Ceres 因子和旧单对优化器
├── scripts/check_project.sh           # 轻量检查：Python 语法、dry-run、C++ build
├── legacy/                            # 旧 shell 流程归档
├── environment.yml                    # Conda 环境
├── matchmodel/                        # SuperPoint 权重位置
└── 3rdparty/                          # nlohmann/json 等第三方代码
```

`build/`、`stereo_calib/result/`、`blender-file/`、本地数据和实验输出默认不进入版本库。

## 环境和编译

Python 环境：

```bash
conda env create -f environment.yml
conda activate stereo-calib-vis
```

SuperPoint 需要 PyTorch 和 OpenCV。如果环境文件没有安装完整，可补充：

```bash
pip install torch torchvision opencv-python
```

C++ 编译：

```bash
mkdir -p build
cmake -S stereo_calib -B build
cmake --build build -j"$(nproc)"
```

编译产物位于 `build/bin/`：

| 可执行文件 | 用途 |
|-----------|------|
| `run_offline_stereo_ba` | project / 通用多帧离线 BA |
| `run_offline_stereo_ba_kitti` | KITTI RAW 连续帧专用 BA |
| `run_stereo_calib` | 单对双目图像 BA |

## 批量实验

批量实验由 `run_experiment.py` 读取 YAML 配置。配置里维护 build 规则、运行环境、scene 列表、参数组合和可选后处理。

```bash
# 默认 panoramic 阈值对比实验
python3 run_experiment.py configs/experiments/panoramic_threshold_sweep.yaml

# 只打印命令，不真正运行
python3 run_experiment.py configs/experiments/panoramic_threshold_sweep.yaml \
  --dry-run --no-build --scene panoramic_01

# KITTI RAW City sync drive 批量实验
python3 run_experiment.py configs/experiments/kitti_raw_city_0001.yaml

# root PATH 找不到 conda 时，可覆盖环境名或在 YAML 中设置 conda_executable
python3 run_experiment.py configs/experiments/kitti_raw_city_0001.yaml \
  --conda-env stereo-calib-vis
```

YAML 参数会转换为 `run_pipeline.py` 的命令行参数。布尔值 `true` 会转换成开关，例如 `fix_distortion: true` 会变成 `--fix_distortion`。

## 单次运行

`run_pipeline.py` 用于一次 scene / 一组参数。它会自动串联匹配、BA 和可视化。

```bash
# project 模式，默认 scene 为 panoramic_01
python3 run_pipeline.py

# 指定 project scene
python3 run_pipeline.py --dataset_mode project --scene panoramic_02

# 跳过匹配，复用已有 matches JSON
python3 run_pipeline.py --scene panoramic_01 --skip_match

# 跳过匹配和 BA，只重新画图
python3 run_pipeline.py --scene panoramic_01 --skip_match --skip_ba

# KITTI RAW 单帧模式
python3 run_pipeline.py \
  --dataset_mode kitti2015 \
  --scene 0000000000 \
  --kitti_root_dir /path/to/kitti/raw/drive/root
```

KITTI RAW 连续帧通常建议通过 YAML 运行，因为左右目目录、初始参数、GT 文件和 BA 参数较多。

## 数据集模式

| `--dataset_mode` | 输入布局 | 主要用途 |
|------------------|----------|----------|
| `project` | `blender-file/stereo_{scene}` | 项目内 Blender 仿真数据 |
| `kitti2015` | `{kitti_root_dir}/image_00/data/{scene}.png` 和 `image_01/data/{scene}.png` | KITTI 单帧 |
| `kitti_raw_aggregate` | 外部提供的 `--match_json` | 直接消费聚合 matches |
| `kitti_raw_sequence` | `--left_img_dir` / `--right_img_dir` 或 KITTI drive 根目录 | KITTI RAW 连续帧 |

KITTI 模式会把官方 `calib_cam_to_cam.txt` 转成项目内部使用的 GT camera JSON。显式传入 `--gt_param_file` 时会优先使用该文件。

## 输出命名

新运行默认使用时间戳，避免覆盖历史结果。时间戳格式为 `YYYYmmdd_HHMMSS_microseconds`，也可以通过 `--output_timestamp` 指定。

常见输出位置：

| 类型 | 目录 |
|------|------|
| matches JSON | `stereo_calib/result/match_points/` |
| BA result JSON | `stereo_calib/result/ba_results/` |
| BA history PNG | `stereo_calib/result/` |
| KITTI GT camera JSON | `stereo_calib/result/gt_params/` |

当使用 `--skip_match` 或 `--skip_ba` 时，pipeline 会优先查找旧版固定命名缓存，以兼容历史结果。例如 project 模式会查找 `{scene}_matches_gtpose_dsym_le_{pixel_tag}.json` 和 `{scene}_ba_result_le_{pixel_tag}.json`。如果要完全指定输入，请传 `--match_json` 或相关显式路径参数。

## 常用参数

流程控制：

| 参数 | 说明 |
|------|------|
| `--skip_match` | 跳过 SuperPoint，复用已有 matches JSON |
| `--skip_ba` | 跳过 BA，复用已有 BA result JSON |
| `--no_plot` | 跳过可视化 |
| `--output_timestamp` | 指定本次输出时间戳 |

SuperPoint：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--nn_thresh` | `0.7` | 描述子 L2 距离阈值，越小越严格 |
| `--conf_thresh` | `0.015` | 关键点置信度阈值 |
| `--nms_dist` | `4` | NMS 半径 |
| `--cuda` | 关闭 | 使用 CUDA 推理 |

BA：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_iter` | `200` | 最终全局 BA 最大迭代次数 |
| `--incremental_max_iter` | `20` | 增量阶段全局 BA 最大迭代次数 |
| `--per_frame_max_iter` | `5` | 逐帧本地校正最大迭代次数 |
| `--global_opt_interval` | `5` | 每注册多少帧触发一次全局 BA |
| `--min_pair_inliers` | `12` | 图像对最少几何内点数 |
| `--min_track_len` | `3` | 有效 track 最小观测数 |
| `--max_score` | `1.0` | 匹配点最大描述子距离 |
| `--fix_distortion` | 关闭 | 固定畸变参数 |
| `--baseline_prior` | BA 默认 | 基线先验权重 |
| `--tx_prior` | BA 默认 | x 方向平移先验，KITTI 专用 |
| `--focal_prior` | BA 默认 | 焦距先验，KITTI 专用 |
| `--focal_lower_scale` / `--focal_upper_scale` | BA 默认 | 焦距优化上下界缩放 |
| `--outlier_threshold` | BA 默认 | 外点剔除阈值，单位像素 |
| `--outlier_rounds` | BA 默认 | 外点剔除最大轮数 |
| `--enable_per_frame_correction` | 关闭 | 启用逐帧本地校正 |
| `--enable_incremental_free_principal_point_refine` | 关闭 | 增量阶段外点剔除后追加短迭代释放主点全局 BA |
| `--incremental_free_principal_point_max_iter` | BA 默认 | 增量阶段释放主点 refine 最大迭代次数 |
| `--min_active_frames_for_free_principal_point` | BA 默认 | 触发增量释放主点 refine 的最小 active 帧数 |
| `--free_principal_point_every_n_global_ba` | BA 默认 | 每多少次 interval global BA 触发一次释放主点 refine |
| `--free_principal_point_max_rmse_increase` | BA 默认 | 释放主点 refine 允许的最大 RMSE 增量，超过则回退 |
| `--reset_camera_params_each_ba_round` | 关闭 | 每轮 BA 前重置相机参数到初值 |

## 数据格式

matches JSON 的核心字段是 `pairs`。每个 pair 包含两张图像名和匹配点列表；project 数据还可能带有 `left`、`right`、`extrinsics`，用于 GT 对比。

BA result JSON 的主要字段：

| 字段 | 说明 |
|------|------|
| `left` / `right` / `extrinsics` | 优化后的双目参数 |
| `success` | 是否通过质量门限 |
| `num_tracks` / `num_observations` / `num_frames` | 本次优化规模 |
| `init_reproj_error` / `final_reproj_error` | 初始和最终重投影误差 |
| `optimization_history` | 每个优化阶段的相机和误差历史 |
| `outlier_rejection_history` | 每次外点剔除的轮次、剔除数量和剩余观测 |
| `summary` | 从历史记录聚合出的平均误差摘要 |
| `diff_vs_gt` | 有 GT 时输出的最终参数差异 |

## Linux CPU 单文件发布

发布版入口是 `run_calibrate_once.py`，固定使用 CPU，执行一次完整流程：

1. 左右目序列 SuperPoint 匹配。
2. C++ KITTI RAW sequence BA。
3. 将最终 BA result JSON 复制为用户指定的 `--output_json`。

构建前建议使用 CPU 版 PyTorch 环境，并安装 PyInstaller：

```bash
python3 -m pip install pyinstaller
```

先编译 Linux Release C++ 目标：

```bash
cmake -S stereo_calib -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

确认存在：

```text
build/bin/run_offline_stereo_ba
build/bin/run_offline_stereo_ba_kitti
build/bin/run_stereo_calib
stereo_calib/scripts/superpoint_v1.pth
```

然后打包：

```bash
packaging/build_linux_onefile.sh
```

产物位于：

```text
dist/mycalib
release/mycalib-linux-x86_64-cpu.tar.gz
```

其中：

- `dist/mycalib` 是仓库内直接运行的单文件可执行程序。
- `release/mycalib-linux-x86_64-cpu.tar.gz` 是对外发布用压缩包，包含可执行程序和示例脚本。

在源码目录里直接运行一次标定并只输出最终 JSON：

```bash
./dist/mycalib \
  --left_img_dir /data/seq01/left \
  --right_img_dir /data/seq01/right \
  --window_size 50 \
  --init_param_file /data/seq01/init_camera.txt \
  --output_json /data/seq01/calibration_result.json
```

`--window_size 50` 表示只使用左右目目录中最近 `50` 对完整双目帧；如果目录里持续追加新图，每次重跑都会自动滑动到最新窗口。可选传入 `--work_dir` 保留中间 matches 和 BA 文件；未传时中间文件使用临时目录。`--gt_param_file` 和 `--frame_poses_file` 仍可按需传入。

如果直接下载 GitHub Release，可下载：

```text
mycalib-linux-x86_64-cpu.tar.gz
SHA256SUMS
```

下载后解压：

```bash
tar -xzf mycalib-linux-x86_64-cpu.tar.gz
chmod +x mycalib-linux-x86_64-cpu examples/run_kitti_city_0048_gt.sh
```

发布包内容：

```text
mycalib-linux-x86_64-cpu
examples/run_kitti_city_0048_gt.sh
examples/kitti_raw_00_01_init_params.txt
examples/kitti_raw_00_01_gt_camera.json
```

其中 `examples/run_kitti_city_0048_gt.sh` 已经内置一条 KITTI RAW City `2011_09_26_drive_0048_sync` 的 CPU 示例命令，默认输出到当前解压目录下的 `results/`：

```bash
./examples/run_kitti_city_0048_gt.sh
```

该示例脚本默认使用最近 `50` 对完整双目帧；可通过环境变量覆盖：

```bash
WINDOW_SIZE=30 ./examples/run_kitti_city_0048_gt.sh
```

如果本机 KITTI 数据路径不同，可在运行前覆盖左右目目录：

```bash
LEFT_IMG_DIR=/path/to/image_00/data \
RIGHT_IMG_DIR=/path/to/image_01/data \
./examples/run_kitti_city_0048_gt.sh
```

也可以直接调用发布包里的可执行程序：

```bash
./mycalib-linux-x86_64-cpu \
  --left_img_dir /data/seq01/left \
  --right_img_dir /data/seq01/right \
  --window_size 50 \
  --init_param_file /data/seq01/init_camera.txt \
  --output_json /data/seq01/calibration_result.json
```

可选校验：

```bash
sha256sum -c SHA256SUMS
```

## 检查

改动入口、配置或 C++ 后，可以先跑轻量检查：

```bash
bash scripts/check_project.sh
```

它会执行：

1. Python 语法检查。
2. `run_pipeline.py --help`。
3. panoramic 和 KITTI RAW 配置 dry-run。
4. C++ build。

## 脚本整理约定

当前正式入口只有：

- `run_pipeline.py`
- `run_experiment.py`
- `run_all_test.sh`
- `run_kitti_raw_city_2011_09_26_drive.sh`
- `scripts/check_project.sh`

其中两个 `.sh` 文件只是兼容旧命令的薄包装。包含完整旧流程的脚本放在 `legacy/`，不作为新实验的修改入口。

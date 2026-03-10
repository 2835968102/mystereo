# CLAUDE.md — 项目约定与环境说明

## 项目简介

基于 SuperPoint 特征点匹配的立体相机标定工具链，流程分三步：
1. Blender 仿真渲染 → 2. Python 特征匹配 → 3. C++ Ceres 参数优化

## Python 环境（Conda）

**环境名称：** `stereo-calib-vis`

用于运行项目中所有 Python 可视化和工具脚本，不包含 Blender 内置脚本（`blender.py` 由 Blender 自带 Python 运行）。

```bash
# 创建（仅需一次）
conda env create -f environment.yml

# 激活
conda activate stereo-calib-vis
```

环境依赖见项目根目录 `environment.yml`（Python 3.10, matplotlib, numpy）。

> SuperPoint 特征匹配脚本（`superpoint_stereo_match.py`）需要额外的 PyTorch 和 OpenCV，建议另建专用环境或在 `stereo-calib-vis` 中 `pip install torch torchvision opencv-python`。

## Python 脚本一览

| 脚本 | 用途 | 运行环境 |
|------|------|----------|
| `stereo_calib/scripts/blender.py` | Blender 仿真渲染 | Blender 内置 Python |
| `stereo_calib/scripts/superpoint_stereo_match.py` | SuperPoint 特征匹配 | 需要 PyTorch + OpenCV |
| `stereo_calib/scripts/eval_stereo.py` | 与 GT 对比评估 | `stereo-calib-vis` |
| `stereo_calib/scripts/plot_ba_history.py` | BA 优化历史可视化 | `stereo-calib-vis` |

## 关键文件路径

| 文件 | 说明 |
|------|------|
| `stereo_calib/result/offline_ba_result.json` | 离线 BA 优化结果（含 `optimization_history`） |
| `stereo_calib/data/camera_params.json` | 相机内外参真值（GT） |
| `stereo_calib/data/matches.json` | SuperPoint 特征匹配输出 |
| `matchmodel/SuperPointPretrainedNetwork/superpoint_v1.pth` | SuperPoint 预训练权重 |
| `environment.yml` | Conda 可视化环境配置 |

## 可视化脚本使用

```bash
conda activate stereo-calib-vis

# 弹窗显示
python stereo_calib/scripts/plot_ba_history.py \
    --input stereo_calib/result/offline_ba_result.json

# 保存图片
python stereo_calib/scripts/plot_ba_history.py \
    --input stereo_calib/result/offline_ba_result.json \
    --output ba_history.png
```

## result JSON 中 optimization_history 的阶段类型

- `"Incremental BA - Frame N"` — 增量 BA，逐帧加入后的局部优化
- `"Periodic Global BA - Frame N"` — 每隔 `--global_opt_interval` 帧的全局 BA
- `"Final Global BA"` — 所有帧注册完毕后的最终全局 BA

## C++ 编译

```bash
mkdir -p build && cd build
cmake ../stereo_calib && make -j$(nproc)
# 可执行文件位于 build/bin/
```

# Stereo Camera Calibration Pipeline

基于 SuperPoint 特征点匹配的立体相机标定工具链，支持使用 Blender 生成仿真数据。整体流程分三步：

0. **仿真数据生成**（Blender）：在 Blender 中搭建双目相机场景，渲染 360° 旋转图像序列并导出相机参数。
1. **特征点匹配**（Python）：用 SuperPoint 神经网络提取所有图片的关键点，对所有图片两两匹配，输出匹配点 JSON。
2. **参数优化**（C++）：以匹配点为约束，用 Ceres 优化立体相机的内参与外参。

```
blender.py ──► stereo_360/ ──► superpoint_stereo_match.py ──► matches.json ──► run_stereo_calib ──► result.json
（仿真渲染）     图像+参数        （特征点匹配）                                    （参数优化）
```

---

## 依赖

### Blender 脚本

`blender.py` 作为 Blender 内置脚本运行，无需额外安装依赖。需要 **Blender ≥ 3.0**。

### Python 脚本

| 依赖 | 版本要求 |
|------|----------|
| Python | ≥ 3.7 |
| PyTorch | ≥ 1.7 |
| OpenCV (`cv2`) | ≥ 4.0 |
| NumPy | ≥ 1.18 |

```bash
pip install torch torchvision opencv-python numpy
```

还需要 SuperPoint 预训练权重文件 `superpoint_v1.pth`，已放置于：

```
matchmodel/SuperPointPretrainedNetwork/superpoint_v1.pth
```

### C++ 标定程序

| 依赖 | 版本要求 |
|------|----------|
| CMake | ≥ 3.10 |
| OpenCV | ≥ 4.3 |
| Ceres Solver | 任意稳定版 |
| C++ 编译器 | 支持 C++11 |

---

## 编译 C++ 标定程序

```bash
mkdir -p build && cd build
cmake ../stereo_calib
make -j$(nproc)
```

编译产物位于 `build/bin/run_stereo_calib`。

---

---

## 第零步：Blender 仿真数据生成

### 脚本路径

```
stereo_calib/scripts/blender.py
```

### 使用方式

在 Blender 中打开 **Scripting** 工作区，加载 `blender.py` 后点击运行，或通过命令行后台执行：

```bash
blender --background your_scene.blend --python stereo_calib/scripts/blender.py
```

### 主要函数说明

#### `create_plane_with_stereo_camera()` — 创建双目相机场景

在场景中创建一个平面载体，并在其上安装左右两个摄像头，组成 `Stereo_Camera_Rig` 装备。

```python
create_plane_with_stereo_camera(
    baseline=0.2,              # 左右摄像头间距（米），默认 20cm
    camera_height=2.0,         # 摄像头在平面上方的高度（米）
    plane_size=2.0,            # 平面大小（米）
    plane_location=(0, 0, 0)   # 平面在世界坐标中的位置
)
```

摄像头固定参数：焦距 35mm，传感器宽度 36mm，朝向 Y 轴正方向。

---

#### `render_stereo_360_rotation()` — 360° 旋转渲染（主函数）

将相机装备绕 Z 轴旋转一圈，在每个角度分别渲染左右视图，并导出相机参数。

```python
render_stereo_360_rotation(
    output_folder="//stereo_360/",   # 输出目录（// 表示 .blend 文件所在目录）
    angle_step=15,                    # 旋转步进（度），默认 15° → 共 24 位置、48 张图
    file_format='PNG',                # 图像格式：'PNG' / 'JPEG' / 'OPEN_EXR'
    rig_name="Stereo_Camera_Rig",     # 相机装备名称
    export_poses=True                 # 是否导出每帧位姿
)
```

**输出文件：**

| 文件 | 说明 |
|------|------|
| `left_000.png`, `right_000.png`, ... | 各角度的左右视图（命名含旋转角度） |
| `camera_params.json` | 静态内参 + 立体外参，兼容 `run_stereo_calib` 输入格式 |
| `camera_poses.json` | 每帧左右摄像头的世界→相机变换（OpenCV 约定：X右Y下Z前） |

**`camera_params.json` 格式：**

```json
{
  "image_size": { "width": 1920, "height": 1080 },
  "left":  { "fx": 972.2, "fy": 972.2, "cx": 960.0, "cy": 540.0,
             "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0 },
  "right": { ... },
  "extrinsics": {
    "R": [1,0,0, 0,1,0, 0,0,1],
    "t": [-0.2, 0.0, 0.0]
  }
}
```

> Blender 相机无畸变，所有 k/p 系数均为 0；外参 R 为单位矩阵（两相机平行），t 即基线向量。

---

#### `export_camera_params()` — 单独导出相机参数

不渲染图像，仅导出当前场景中双目相机的内参和外参。

```python
export_camera_params(output_path="//camera_params.json")
```

---

#### `create_test_scene()` — 创建测试场景（可选）

生成用于测试双目视觉的三维场景物体，在自有场景不足时使用。

```python
create_test_scene(scene_type="basic")       # 球体 + 立方体
create_test_scene(scene_type="multi_depth") # 不同深度的彩色圆柱
create_test_scene(scene_type="grid")        # 均匀网格球阵
```

---

#### 其他辅助函数

| 函数 | 说明 |
|------|------|
| `render_stereo_images()` | 仅渲染单张双目图像（不旋转） |
| `adjust_stereo_baseline(new_baseline)` | 动态调整基线距离（米） |
| `animate_plane_flight()` | 为平面载体添加飞行动画（圆形/直线/8字轨迹） |
| `render_stereo_animation()` | 渲染整段动画的逐帧双目图像 |

---

### 典型使用流程

```python
# 在 Blender Scripting 工作区中执行：

# 1. （可选）创建测试场景
create_test_scene("basic")

# 2. 创建双目相机装备
create_plane_with_stereo_camera(
    baseline=0.2,
    camera_height=2.0,
    plane_size=2.0,
)

# 3. 渲染 360° 图像并导出参数
render_stereo_360_rotation(
    output_folder="//stereo_360/",
    angle_step=15,
    export_poses=True,
)
```

输出目录 `stereo_360/` 即可直接作为下一步特征点匹配的 `--img_dir` 输入。

---

## 第一步：特征点匹配

### 脚本路径

```
stereo_calib/scripts/superpoint_stereo_match.py
```

### 图像目录结构

脚本会读取指定目录下**所有图片**，对每两张图片进行匹配（共 N×(N-1)/2 对）。图片命名无限制，支持 `.png`、`.jpg`、`.jpeg`、`.bmp`、`.tiff` 格式。

如需提供初始相机参数，可在图像目录下放置 `camera_params.json`（格式见下文），脚本会将其一并写入输出文件。

### 基本用法

```bash
python superpoint_stereo_match.py       --img_dir  /home/hello/pml/mycalib/blender-file/室内场景_爱给网_aigei_com/stereo_360/       --weights  /home/hello/pml/mycalib/matchmodel/SuperPointPretrainedNetwork/superpoint_v1.pth       --output   matches.json       --nn_thresh 0.7       --conf_thresh 0.015       --nms_dist 4       --cuda

```

### 全部参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--img_dir` | （必填） | 图像所在目录 |
| `--weights` | （必填） | SuperPoint 权重文件路径 |
| `--output` | `superpoint_matches.json` | 输出 JSON 路径 |
| `--nn_thresh` | `0.7` | 双向最近邻匹配的 L2 距离上限，越小越严格 |
| `--conf_thresh` | `0.015` | 关键点置信度阈值，越大检测到的点越少但更可靠 |
| `--nms_dist` | `4` | 非极大值抑制半径（像素），避免关键点过于密集 |
| `--cuda` | 关闭 | 启用 GPU 加速推理 |

> **关于匹配质量过滤**：脚本在 `--nn_thresh` 的基础上还保留了 L2 距离 < 0.4 的高置信度匹配点（距离越小，匹配越可靠）。

### 示例（使用 GPU）

```bash
python stereo_calib/scripts/superpoint_stereo_match.py \
    --img_dir  data/stereo_360 \
    --weights  matchmodel/SuperPointPretrainedNetwork/superpoint_v1.pth \
    --output   matches.json \
    --nn_thresh 0.6 \
    --cuda
```

### 输出 JSON 格式

```json
{
  "left":  { "fx": 1000.0, "fy": 1000.0, "cx": 640.0, "cy": 360.0,
             "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0 },
  "right": { ... },
  "extrinsics": {
    "R": [1,0,0, 0,1,0, 0,0,1],
    "t": [-0.12, 0.0, 0.0]
  },
  "pairs": [
    {
      "left_image":  "left_000.png",
      "right_image": "right_000.png",
      "num_keypoints": { "left": 686, "right": 848 },
      "num_matches": 120,
      "matches": [
        { "left": [320.5, 240.1], "right": [210.3, 240.1], "score": 0.2314 },
        ...
      ]
    },
    ...
  ]
}
```

`score` 为 L2 距离，范围 0–2，**越小表示匹配越可靠**。

---

## 第二步：立体相机参数优化

### 用法

```bash
build/bin/run_stereo_calib \
    --input  matches.json \
    --output result.json \
    [--max_iter 200]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | （必填） | 第一步输出的 JSON 文件 |
| `--output` | （必填） | 优化后参数的输出路径 |
| `--max_iter` | `200` | Ceres 最大迭代次数 |

输入 JSON 中的 `left`、`right`（内参）和 `extrinsics`（外参）作为优化初值；`pairs.matches` 中的像素对作为约束。优化结果写入 `--output`，格式与输入相同。

### 论文风格离线 BA（Track + 全局优化）

参考 `PTZ-Calib` 论文离线阶段思想，新增了 `run_offline_stereo_ba`：

#### 构建命令

```bash
mkdir -p build
cd build
cmake ../../stereo_calib && cmake --build . -j$(nproc) --target run_offline_stereo_ba

```

构建完成后可执行文件位于：`build/bin/run_offline_stereo_ba`。

#### 运行命令（默认推荐）

```bash
bin/run_offline_stereo_ba \
    --input ../data/matches.json \
    --gt_param_file ../data/camera_params.json \
    --output ../result/offline_ba_result.json \
    --max_iter 40 \
    --incremental_max_iter 10 \
    --global_opt_interval 5 \
    --max_score 0.2 \
    --min_pair_inliers 15 \
    --min_pair_inlier_ratio 0.5 \
    --min_track_len 3 \
    --fix_distortion
```

该程序会先对 `pairs` 里的两两匹配做 Union-Find 轨迹构建（tracks building），然后按帧增量注册并执行 BA：

1. 逐帧加入新图像后做一次增量 BA（`--incremental_max_iter` 控制迭代次数）  
2. 每完成 `--global_opt_interval` 次注册后做一次全局 BA  
3. 最后再做一次全量全局 BA 输出结果

> 程序会始终从 `stereo_calib/example_init_params.txt` 读取初始内外参。  
> `--init_param_file` / `--use_input_init` / `--init_width` / `--init_height` / `--init_focal` / `--init_baseline` 已弃用并忽略。  
> `--known_baseline` / `--known_baseline_weight` 已弃用并忽略。
>
> 若提供 `--gt_param_file`（支持 `.json` 或文本参数文件），输出结果 JSON 会新增 `diff_vs_gt`，表示 `estimate - gt` 的参数差值（正负号分别表示偏大/偏小）。

### 相机参数字段说明

| 字段 | 含义 |
|------|------|
| `fx`, `fy` | 焦距（像素） |
| `cx`, `cy` | 主点坐标（像素） |
| `k1`, `k2`, `k3` | 径向畸变系数 |
| `p1`, `p2` | 切向畸变系数 |
| `R` | 旋转矩阵（行优先，3×3） |
| `t` | 平移向量（米） |

---

## 完整流程示例

```bash
# 0. 在 Blender 中生成仿真数据（后台模式）
blender --background your_scene.blend --python stereo_calib/scripts/blender.py
# → 输出 stereo_360/left_000.png, right_000.png, ..., camera_params.json

# 1. 特征点匹配
python stereo_calib/scripts/superpoint_stereo_match.py \
    --img_dir  stereo_360 \
    --weights  matchmodel/SuperPointPretrainedNetwork/superpoint_v1.pth \
    --output   matches.json \
    --cuda

# 2. 参数优化
build/bin/run_stereo_calib \
    --input  matches.json \
    --output result.json

# 查看优化结果
cat result.json
```

---

## 第三步：可视化 BA 优化历史

### 环境准备（Conda）

本项目为 Python 可视化脚本提供了独立的 Conda 环境，包含 `matplotlib` 和 `numpy`。

```bash
# 创建环境（仅需执行一次）
conda env create -f environment.yml

# 激活环境
conda activate stereo-calib-vis
```

> 环境名称：`stereo-calib-vis`，Python 3.10，依赖见 `environment.yml`。

### 脚本路径

```
stereo_calib/scripts/plot_ba_history.py
```

### 功能说明

从 `offline_ba_result.json` 的 `optimization_history` 字段读取每次优化迭代的重投影误差，绘制折线图并：

- 以蓝色圆点标注每次**增量 BA（Incremental BA）**
- 以红色星形标注每次**全局优化（Global BA / Final Global BA）**
- 在全局优化点处绘制垂直虚线，便于定位
- 标注初始误差与最终误差参考线

### 用法

```bash
# 激活环境
conda activate stereo-calib-vis

# 弹窗显示（在项目根目录运行）
python stereo_calib/scripts/plot_ba_history.py \
    --input stereo_calib/result/offline_ba_result.json

# 保存为图片
python stereo_calib/scripts/plot_ba_history.py \
    --input stereo_calib/result/offline_ba_result.json \
    --output ba_history.png
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` / `-i` | `stereo_calib/result/offline_ba_result.json` | result JSON 文件路径 |
| `--output` / `-o` | （无，弹窗显示） | 输出图片路径（支持 `.png` / `.pdf` 等） |

### 阶段类型说明

| 阶段名称 | 类型 | 图中标注 |
|----------|------|----------|
| `Incremental BA - Frame N` | 增量 BA | 蓝色圆点 |
| `Periodic Global BA - Frame N` | 周期性全局 BA | 红色星形 |
| `Final Global BA` | 最终全局 BA | 红色星形 |

---

## 项目结构

```
.
├── environment.yml                    # Conda 可视化环境（stereo-calib-vis）
├── matchmodel/
│   └── SuperPointPretrainedNetwork/
│       └── superpoint_v1.pth          # SuperPoint 预训练权重
├── stereo_calib/
│   ├── scripts/
│   │   ├── blender.py                 # Blender 仿真数据生成（Step 0）
│   │   ├── superpoint_stereo_match.py # 特征匹配脚本（Step 1）
│   │   ├── eval_stereo.py             # 评估工具（与 GT 对比）
│   │   └── plot_ba_history.py         # BA 优化历史可视化（Step 3）
│   ├── src/
│   │   ├── run_stereo_calib.cc        # 标定程序入口（Step 2）
│   │   ├── stereo_optimizer.cc/.h     # Ceres 优化器
│   │   └── stereo_factors.cc/.h       # 重投影误差因子
│   ├── example_input.json             # 输入格式示例
│   └── CMakeLists.txt
├── 3rdparty/
│   └── json-3.9.1/                    # nlohmann/json
└── build/                             # 编译输出目录
```

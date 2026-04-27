#!/bin/bash
set -euo pipefail

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$CURRENT_DIR/stereo_calib/scripts/superpoint_stereo_match_raw.py"
PLOT_SCRIPT="$CURRENT_DIR/stereo_calib/scripts/plot_ba_history.py"
BUILD_DIR="$CURRENT_DIR/build"
BA_BIN="$BUILD_DIR/bin/run_offline_stereo_ba_kitti"
MATCH_OUTPUT_DIR="$CURRENT_DIR/stereo_calib/result/match_points"
BA_OUTPUT_DIR="$CURRENT_DIR/stereo_calib/result/ba_results"
PLOT_OUTPUT_DIR="$CURRENT_DIR/stereo_calib/result"
GT_PARAM_FILE="$CURRENT_DIR/stereo_calib/result/gt_params/kitti_raw_00_01_gt_camera.json"
INIT_PARAM_FILE="$CURRENT_DIR/stereo_calib/data/kitti_raw_00_01_init_params.txt"

LEFT_IMG_DIR="/home/hello/pml/data/KITTI/RAW/City/2011_09_26_drive_0001_extract/2011_09_26/2011_09_26_drive_0001_extract/image_00/data"
RIGHT_IMG_DIR="/home/hello/pml/data/KITTI/RAW/City/2011_09_26_drive_0001_extract/2011_09_26/2011_09_26_drive_0001_extract/image_01/data"
DATASET_NAME="kitti_city_2011_09_26_drive_0001_extract"

CONF_THRESH=0.2
NMS_DIST=15
NN_THRESH=1
USE_CUDA=0
FORCE_REMATCH=0
RESET_CAMERA_PARAMS_EACH_BA_ROUND=0
ENABLE_PER_FRAME_CORRECTION=1
PER_FRAME_MAX_ITER=5

MAX_ITER=40
INCREMENTAL_MAX_ITER=30
GLOBAL_OPT_INTERVAL=10
MIN_PAIR_INLIERS=15
FIX_DISTORTION=1
MAX_SCORE=0.5
MIN_TRACK_LEN=3
BASELINE_PRIOR=100
TX_PRIOR=20
FOCAL_PRIOR=1
FOCAL_LOWER_SCALE=0.2
FOCAL_UPPER_SCALE=3.0
OUTLIER_THRESHOLD=2.0

SCRIPT_START_TIME=$(date +%s)

format_elapsed_time() {
    local total_seconds=$1
    local elapsed_hours elapsed_minutes elapsed_seconds
    elapsed_hours=$((total_seconds / 3600))
    elapsed_minutes=$(((total_seconds % 3600) / 60))
    elapsed_seconds=$((total_seconds % 60))
    printf "%02d:%02d:%02d" "$elapsed_hours" "$elapsed_minutes" "$elapsed_seconds"
}

print_total_elapsed_time() {
    local end_time total_seconds
    end_time=$(date +%s)
    total_seconds=$((end_time - SCRIPT_START_TIME))
    printf "总运行时间: %s\n" "$(format_elapsed_time "$total_seconds")"
}

trap print_total_elapsed_time EXIT

usage() {
    cat <<'EOF'
用法:
  bash run_kitti_raw_city_2011_09_26_drive_0001.sh [options]

可选参数:
  --left_img_dir <path>   左目图像目录
  --right_img_dir <path>  右目图像目录
  --dataset_name <name>   输出文件名前缀
  --gt_param_file <path>  GT 相机参数文件，用于结果对比
  --init_param_file <path> 初始相机参数文件，用于 KITTI 初始化
  --baseline_prior <float> 基线先验权重
  --tx_prior <float> x 方向平移先验权重
  --focal_prior <float> 焦距先验权重
  --focal_lower_scale <float> 焦距下界缩放因子
  --focal_upper_scale <float> 焦距上界缩放因子
  --conf_thresh <float>   SuperPoint 关键点置信度阈值
  --nms_dist <int>        SuperPoint NMS 抑制半径
  --nn_thresh <float>     SuperPoint 描述子匹配阈值
  --cuda                  使用 CUDA 跑 SuperPoint
  --cpu                   强制使用 CPU 跑 SuperPoint
  --force_rematch         忽略已有匹配结果并重新运行 SuperPoint
  --reset_camera_params_each_ba_round  每轮 BA 前重置相机参数到初始值
  --disable_per_frame_correction       关闭逐帧本地校正
  --per_frame_max_iter <int>           逐帧本地校正最大迭代次数
  --outlier_threshold <float>          离群点剔除像素阈值
  -h, --help              显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --left_img_dir)
            LEFT_IMG_DIR="$2"
            shift 2
            ;;
        --right_img_dir)
            RIGHT_IMG_DIR="$2"
            shift 2
            ;;
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --gt_param_file)
            GT_PARAM_FILE="$2"
            shift 2
            ;;
        --init_param_file)
            INIT_PARAM_FILE="$2"
            shift 2
            ;;
        --baseline_prior)
            BASELINE_PRIOR="$2"
            shift 2
            ;;
        --tx_prior)
            TX_PRIOR="$2"
            shift 2
            ;;
        --focal_prior)
            FOCAL_PRIOR="$2"
            shift 2
            ;;
        --focal_lower_scale)
            FOCAL_LOWER_SCALE="$2"
            shift 2
            ;;
        --focal_upper_scale)
            FOCAL_UPPER_SCALE="$2"
            shift 2
            ;;
        --conf_thresh)
            CONF_THRESH="$2"
            shift 2
            ;;
        --nms_dist)
            NMS_DIST="$2"
            shift 2
            ;;
        --nn_thresh)
            NN_THRESH="$2"
            shift 2
            ;;
        --cuda)
            USE_CUDA=1
            shift
            ;;
        --cpu)
            USE_CUDA=0
            shift
            ;;
        --force_rematch)
            FORCE_REMATCH=1
            shift
            ;;
        --reset_camera_params_each_ba_round)
            RESET_CAMERA_PARAMS_EACH_BA_ROUND=1
            shift
            ;;
        --disable_per_frame_correction)
            ENABLE_PER_FRAME_CORRECTION=0
            shift
            ;;
        --per_frame_max_iter)
            PER_FRAME_MAX_ITER="$2"
            shift 2
            ;;
        --outlier_threshold)
            OUTLIER_THRESHOLD="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            usage
            exit 1
            ;;
    esac
done

PARAM_TAG="conf_thresh=${CONF_THRESH}_nms_dist=${NMS_DIST}_nn_thresh=${NN_THRESH}"
OUTPUT_MATCH_JSON="$MATCH_OUTPUT_DIR/${DATASET_NAME}_${PARAM_TAG}_matches_raw.json"
OUTPUT_BA_JSON="$BA_OUTPUT_DIR/${DATASET_NAME}_${PARAM_TAG}_ba_result_raw.json"
OUTPUT_PNG="$PLOT_OUTPUT_DIR/${DATASET_NAME}_${PARAM_TAG}_ba_history_raw.png"

mkdir -p "$MATCH_OUTPUT_DIR" "$BA_OUTPUT_DIR" "$PLOT_OUTPUT_DIR"

if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "SuperPoint raw 脚本不存在: $SCRIPT_PATH"
    exit 1
fi
if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "绘图脚本不存在: $PLOT_SCRIPT"
    exit 1
fi
if [[ ! -d "$LEFT_IMG_DIR" ]]; then
    echo "左图目录不存在: $LEFT_IMG_DIR"
    exit 1
fi
if [[ ! -d "$RIGHT_IMG_DIR" ]]; then
    echo "右图目录不存在: $RIGHT_IMG_DIR"
    exit 1
fi
if [[ ! -f "$GT_PARAM_FILE" ]]; then
    echo "GT 参数文件不存在: $GT_PARAM_FILE"
    exit 1
fi
if [[ ! -f "$INIT_PARAM_FILE" ]]; then
    echo "初始参数文件不存在: $INIT_PARAM_FILE"
    exit 1
fi

echo "开始编译..."
cmake -S "$CURRENT_DIR/stereo_calib" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" -j"$(nproc)"
echo "编译完成！"

if [[ ! -x "$BA_BIN" ]]; then
    echo "BA 可执行文件不存在: $BA_BIN"
    exit 1
fi

echo "========================================"
echo "开始处理数据集: $DATASET_NAME"
echo "左图目录: $LEFT_IMG_DIR"
echo "右图目录: $RIGHT_IMG_DIR"
echo "GT 参数: $GT_PARAM_FILE"
echo "初始参数: $INIT_PARAM_FILE"
echo "baseline prior: $BASELINE_PRIOR"
echo "tx prior: $TX_PRIOR"
echo "focal prior: $FOCAL_PRIOR"
echo "focal bound scale: [$FOCAL_LOWER_SCALE, $FOCAL_UPPER_SCALE]"
echo "参数标签: $PARAM_TAG"
echo "匹配输出: $OUTPUT_MATCH_JSON"
echo "BA 输出: $OUTPUT_BA_JSON"
echo "图片输出: $OUTPUT_PNG"
echo "force rematch: $FORCE_REMATCH"
echo "reset camera params each BA round: $RESET_CAMERA_PARAMS_EACH_BA_ROUND"
echo "enable per-frame correction: $ENABLE_PER_FRAME_CORRECTION"
echo "per-frame max iter: $PER_FRAME_MAX_ITER"
echo "outlier threshold: $OUTLIER_THRESHOLD"
echo "========================================"

DATASET_START_TIME=$(date +%s)

if [[ -f "$OUTPUT_MATCH_JSON" && "$FORCE_REMATCH" -eq 0 ]]; then
    echo "检测到已有匹配结果，跳过 SuperPoint: $OUTPUT_MATCH_JSON"
else
    if [[ "$FORCE_REMATCH" -eq 1 && -f "$OUTPUT_MATCH_JSON" ]]; then
        echo "强制重新运行 SuperPoint，并覆盖已有匹配结果: $OUTPUT_MATCH_JSON"
    else
        echo "未找到匹配结果，开始运行 SuperPoint"
    fi

    MATCH_CMD=(
        conda run --no-capture-output -n stereo-calib-vis python "$SCRIPT_PATH"
        --left_img_dir "$LEFT_IMG_DIR"
        --right_img_dir "$RIGHT_IMG_DIR"
        --output "$OUTPUT_MATCH_JSON"
        --conf_thresh "$CONF_THRESH"
        --nms_dist "$NMS_DIST"
        --nn_thresh "$NN_THRESH"
    )
    if [[ "$USE_CUDA" -eq 1 ]]; then
        MATCH_CMD+=(--cuda)
    fi
    "${MATCH_CMD[@]}"
fi

BA_CMD=(
    "$BA_BIN"
    --input "$OUTPUT_MATCH_JSON"
    --output "$OUTPUT_BA_JSON"
    --gt_param_file "$GT_PARAM_FILE"
    --init_param_file "$INIT_PARAM_FILE"
    --baseline_prior "$BASELINE_PRIOR"
    --tx_prior "$TX_PRIOR"
    --focal_prior "$FOCAL_PRIOR"
    --focal_lower_scale "$FOCAL_LOWER_SCALE"
    --focal_upper_scale "$FOCAL_UPPER_SCALE"
    --max_iter "$MAX_ITER"
    --incremental_max_iter "$INCREMENTAL_MAX_ITER"
    --per_frame_max_iter "$PER_FRAME_MAX_ITER"
    --global_opt_interval "$GLOBAL_OPT_INTERVAL"
    --outlier_threshold "$OUTLIER_THRESHOLD"
    --min_pair_inliers "$MIN_PAIR_INLIERS"
    --max_score "$MAX_SCORE"
    --min_track_len "$MIN_TRACK_LEN"
)
if [[ "$FIX_DISTORTION" -eq 1 ]]; then
    BA_CMD+=(--fix_distortion)
fi
if [[ "$RESET_CAMERA_PARAMS_EACH_BA_ROUND" -eq 1 ]]; then
    BA_CMD+=(--reset_camera_params_each_ba_round)
fi
if [[ "$ENABLE_PER_FRAME_CORRECTION" -eq 1 ]]; then
    BA_CMD+=(--enable_per_frame_correction)
fi
"${BA_CMD[@]}"

conda run --no-capture-output -n stereo-calib-vis python "$PLOT_SCRIPT" \
    --input "$OUTPUT_BA_JSON" \
    --output "$OUTPUT_PNG"

DATASET_END_TIME=$(date +%s)
DATASET_ELAPSED_TIME=$((DATASET_END_TIME - DATASET_START_TIME))
printf "数据集 %s 运行时间: %s\n" "$DATASET_NAME" "$(format_elapsed_time "$DATASET_ELAPSED_TIME")"

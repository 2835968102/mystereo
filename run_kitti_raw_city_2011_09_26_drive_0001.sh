#!/bin/bash
set -euo pipefail

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$CURRENT_DIR/stereo_calib/scripts/superpoint_stereo_match_raw.py"
PLOT_SCRIPT="$CURRENT_DIR/stereo_calib/scripts/plot_ba_history.py"
BUILD_DIR="$CURRENT_DIR/build"
BA_BIN="$BUILD_DIR/bin/run_offline_stereo_ba"
MATCH_OUTPUT_DIR="$CURRENT_DIR/stereo_calib/result/match_points"
BA_OUTPUT_DIR="$CURRENT_DIR/stereo_calib/result/ba_results"
PLOT_OUTPUT_DIR="$CURRENT_DIR/stereo_calib/result"

LEFT_IMG_DIR="/home/hello/pml/data/KITTI/RAW/City/2011_09_26_drive_0001_extract/2011_09_26/2011_09_26_drive_0001_extract/image_00/data"
RIGHT_IMG_DIR="/home/hello/pml/data/KITTI/RAW/City/2011_09_26_drive_0001_extract/2011_09_26/2011_09_26_drive_0001_extract/image_01/data"
DATASET_NAME="kitti_city_2011_09_26_drive_0001_extract"

CONF_THRESH=0.2
NMS_DIST=15
NN_THRESH=1
USE_CUDA=0

MAX_ITER=40
INCREMENTAL_MAX_ITER=1
GLOBAL_OPT_INTERVAL=50
MIN_PAIR_INLIERS=15
FIX_DISTORTION=1
MAX_SCORE=0.5
MIN_TRACK_LEN=3

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
  --conf_thresh <float>   SuperPoint 关键点置信度阈值
  --nms_dist <int>        SuperPoint NMS 抑制半径
  --nn_thresh <float>     SuperPoint 描述子匹配阈值
  --cuda                  使用 CUDA 跑 SuperPoint
  --cpu                   强制使用 CPU 跑 SuperPoint
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
echo "参数标签: $PARAM_TAG"
echo "匹配输出: $OUTPUT_MATCH_JSON"
echo "BA 输出: $OUTPUT_BA_JSON"
echo "图片输出: $OUTPUT_PNG"
echo "========================================"

DATASET_START_TIME=$(date +%s)

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

BA_CMD=(
    "$BA_BIN"
    --input "$OUTPUT_MATCH_JSON"
    --output "$OUTPUT_BA_JSON"
    --max_iter "$MAX_ITER"
    --incremental_max_iter "$INCREMENTAL_MAX_ITER"
    --global_opt_interval "$GLOBAL_OPT_INTERVAL"
    --min_pair_inliers "$MIN_PAIR_INLIERS"
    --max_score "$MAX_SCORE"
    --min_track_len "$MIN_TRACK_LEN"
)
if [[ "$FIX_DISTORTION" -eq 1 ]]; then
    BA_CMD+=(--fix_distortion)
fi
"${BA_CMD[@]}"

conda run --no-capture-output -n stereo-calib-vis python "$PLOT_SCRIPT" \
    --input "$OUTPUT_BA_JSON" \
    --output "$OUTPUT_PNG"

DATASET_END_TIME=$(date +%s)
DATASET_ELAPSED_TIME=$((DATASET_END_TIME - DATASET_START_TIME))
printf "数据集 %s 运行时间: %s\n" "$DATASET_NAME" "$(format_elapsed_time "$DATASET_ELAPSED_TIME")"

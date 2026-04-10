#!/bin/bash
# 记录当前目录，以便编译后返回
CURRENT_DIR=$(pwd)
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

# 定义公共参数变量
MAX_ITER=40
INCREMENTAL_MAX_ITER=1
GLOBAL_OPT_INTERVAL=50
MIN_PAIR_INLIERS=15
FIX_DISTORTION="--fix_distortion"
MAX_SCORE=0.5
MIN_TRACK_LEN=3
SKIP_MATCH="--skip_match"

# 定义场景数组
SCENES_1=(
    "panoramic_01"
    "panoramic_02"
    "panoramic_03"
    "panoramic_04"
    "panoramic_05"
)


# 编译步骤
echo "开始编译..."
cd "$CURRENT_DIR/stereo_calib/build" || { echo "进入build目录失败"; exit 1; }
cmake .. || { echo "cmake配置失败"; exit 1; }
make -j$(nproc) || { echo "编译失败"; exit 1; }
cd "$CURRENT_DIR" || { echo "返回工作目录失败"; exit 1; }
echo "编译完成！"

# 循环执行
for SCENE in "${SCENES_1[@]}"; do
    echo "正在处理场景: $SCENE"
    SCENE_START_TIME=$(date +%s)

    conda run --no-capture-output -n stereo-calib-vis python run_pipeline.py \
        --scene "$SCENE" \
        --max_iter "$MAX_ITER" \
        --incremental_max_iter "$INCREMENTAL_MAX_ITER" \
        --global_opt_interval "$GLOBAL_OPT_INTERVAL" \
        --min_pair_inliers "$MIN_PAIR_INLIERS" \
        --max_score "$MAX_SCORE" \
        $FIX_DISTORTION \
        --min_track_len "$MIN_TRACK_LEN" \
        $SKIP_MATCH

    SCENE_END_TIME=$(date +%s)
    SCENE_ELAPSED_TIME=$((SCENE_END_TIME - SCENE_START_TIME))
    printf "场景 %s 运行时间: %s\n" "$SCENE" "$(format_elapsed_time "$SCENE_ELAPSED_TIME")"
done
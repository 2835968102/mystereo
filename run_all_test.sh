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

# 定义要批量运行的像素阈值标签
PIXEL_TAGS=(
    "3px"
    "2px"
    "1px"
)

# 定义 project 场景数组（保留原有批量运行）
PROJECT_SCENES=(
    "panoramic_01"
    "panoramic_02"
    "panoramic_03"
    "panoramic_04"
    "panoramic_05"
)

# 定义 KITTI 场景数组（新增）
KITTI_SCENES=(
    "000000"
    "000001"
)
KITTI_ROOT_DIR="/home/hello/pml/data/KITTI/Stereo/Stereo2015/data_scene_flow/training"

result_prefix_for_mode() {
    local dataset_mode=$1
    local scene=$2
    if [ "$dataset_mode" = "project" ]; then
        printf "%s" "$scene"
    else
        printf "%s_%s" "$dataset_mode" "$scene"
    fi
}

run_project_threshold_sweep() {
    local scene=$1
    local result_prefix
    result_prefix=$(result_prefix_for_mode "project" "$scene")

    echo "========================================"
    echo "开始处理场景: $scene (dataset_mode=project)"
    echo "========================================"

    for PIXEL_TAG in "${PIXEL_TAGS[@]}"; do
        echo "正在处理场景: $scene (dataset_mode=project, 小于${PIXEL_TAG})"
        SCENE_START_TIME=$(date +%s)

        conda run --no-capture-output -n stereo-calib-vis python run_pipeline.py \
            --dataset_mode project \
            --scene "$scene" \
            --pixel_tag "$PIXEL_TAG" \
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
        printf "场景 %s (dataset_mode=project, 小于%s) 运行时间: %s\n" "$scene" "$PIXEL_TAG" "$(format_elapsed_time "$SCENE_ELAPSED_TIME")"
    done

    conda run --no-capture-output -n stereo-calib-vis python stereo_calib/scripts/plot_ba_threshold_comparison.py \
        --scene "$result_prefix" \
        --input-3px "stereo_calib/result/ba_results/${result_prefix}_ba_result_le_3px.json" \
        --input-2px "stereo_calib/result/ba_results/${result_prefix}_ba_result_le_2px.json" \
        --input-1px "stereo_calib/result/ba_results/${result_prefix}_ba_result_le_1px.json" \
        --output "stereo_calib/result/${result_prefix}_ba_threshold_comparison.png"
}

run_kitti_scene() {
    local scene=$1
    local result_prefix
    local skip_match_arg=""
    local kitti_match_path="stereo_calib/result/match_points/kitti2015_${scene}_matches.json"
    local kitti_ba_result_path="stereo_calib/result/ba_results/kitti2015_${scene}_ba_result.json"
    result_prefix=$(result_prefix_for_mode "kitti2015" "$scene")

    if [ -f "$kitti_match_path" ]; then
        skip_match_arg="$SKIP_MATCH"
    fi

    echo "========================================"
    echo "开始处理场景: $scene (dataset_mode=kitti2015)"
    echo "========================================"
    SCENE_START_TIME=$(date +%s)

    conda run --no-capture-output -n stereo-calib-vis python run_pipeline.py \
        --dataset_mode kitti2015 \
        --scene "$scene" \
        --kitti_root_dir "$KITTI_ROOT_DIR" \
        --max_iter "$MAX_ITER" \
        --incremental_max_iter "$INCREMENTAL_MAX_ITER" \
        --global_opt_interval "$GLOBAL_OPT_INTERVAL" \
        --min_pair_inliers "$MIN_PAIR_INLIERS" \
        --max_score "$MAX_SCORE" \
        $FIX_DISTORTION \
        --min_track_len "$MIN_TRACK_LEN" \
        --no_plot \
        ${skip_match_arg:+$skip_match_arg}

    SCENE_END_TIME=$(date +%s)
    SCENE_ELAPSED_TIME=$((SCENE_END_TIME - SCENE_START_TIME))
    printf "场景 %s (dataset_mode=kitti2015) 运行时间: %s\n" "$scene" "$(format_elapsed_time "$SCENE_ELAPSED_TIME")"
}

# 编译步骤
echo "开始编译..."
mkdir -p "$CURRENT_DIR/build" || { echo "创建build目录失败"; exit 1; }
cd "$CURRENT_DIR/build" || { echo "进入build目录失败"; exit 1; }
cmake ../stereo_calib || { echo "cmake配置失败"; exit 1; }
make -j$(nproc) || { echo "编译失败"; exit 1; }
cd "$CURRENT_DIR" || { echo "返回工作目录失败"; exit 1; }
echo "编译完成！"

# 先跑原有 project 场景
for SCENE in "${PROJECT_SCENES[@]}"; do
    run_project_threshold_sweep "$SCENE"
done

# 再跑新增 KITTI 场景
for SCENE in "${KITTI_SCENES[@]}"; do
    run_kitti_scene "$SCENE"
done

: <<'LEGACY_PROJECT_LOOP_REMOVED'
for SCENE in "${SCENES_1[@]}"; do
    echo "========================================"
    echo "开始处理场景: $SCENE"
    echo "========================================"

    for PIXEL_TAG in "${PIXEL_TAGS[@]}"; do
        echo "正在处理场景: $SCENE (小于${PIXEL_TAG})"
        SCENE_START_TIME=$(date +%s)

        conda run --no-capture-output -n stereo-calib-vis python run_pipeline.py \
            --scene "$SCENE" \
            --pixel_tag "$PIXEL_TAG" \
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
        printf "场景 %s (小于%s) 运行时间: %s\n" "$SCENE" "$PIXEL_TAG" "$(format_elapsed_time "$SCENE_ELAPSED_TIME")"
    done

    conda run --no-capture-output -n stereo-calib-vis python stereo_calib/scripts/plot_ba_threshold_comparison.py \
        --scene "$SCENE" \
        --input-3px "stereo_calib/result/ba_results/${SCENE}_ba_result_le_3px.json" \
        --input-2px "stereo_calib/result/ba_results/${SCENE}_ba_result_le_2px.json" \
        --input-1px "stereo_calib/result/ba_results/${SCENE}_ba_result_le_1px.json" \
        --output "stereo_calib/result/${SCENE}_ba_threshold_comparison.png"
done
LEGACY_PROJECT_LOOP_REMOVED

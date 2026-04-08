#!/bin/bash
# 记录当前目录，以便编译后返回
CURRENT_DIR=$(pwd)

# 定义公共参数变量
MAX_ITER=40
INCREMENTAL_MAX_ITER=10
GLOBAL_OPT_INTERVAL=5
MIN_PAIR_INLIERS=15
FIX_DISTORTION="--fix_distortion"
MAX_SCORE=1.0
MIN_TRACK_LEN=3
SKIP_MATCH="--skip_match"

# 定义场景数组
SCENES_1=(
    "panoramic_01"
)


# 编译步骤
echo "开始编译..."
cd build || { echo "进入build目录失败"; exit 1; }
cmake ../stereo_calib || { echo "cmake配置失败"; exit 1; }
make -j$(nproc) || { echo "编译失败"; exit 1; }
cd "$CURRENT_DIR" || { echo "返回工作目录失败"; exit 1; }
echo "编译完成！"

# 循环执行
for SCENE in "${SCENES_1[@]}"; do
    echo "正在处理场景: $SCENE"
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
done
#!/bin/bash
set -euo pipefail

CURRENT_DIR="/home/hello/pml/mycalib"
SCRIPT_PATH="$CURRENT_DIR/stereo_calib/scripts/superpoint_stereo_match_raw.py"
WEIGHTS_PATH="$CURRENT_DIR/matchmodel/SuperPointPretrainedNetwork/superpoint_v1.pth"
OUTPUT_DIR="$CURRENT_DIR/stereo_calib/result/match_points"
CONF_THRESH=0.2
NMS_DIST=10
NN_THRESH=0.7

SCENES=(
    "panoramic_01"
    "panoramic_02"
    "panoramic_03"
    "panoramic_04"
    "panoramic_05"
)

for SCENE in "${SCENES[@]}"; do
    SCENE_DIR="$CURRENT_DIR/blender-file/stereo_${SCENE}"
    LEFT_IMG_DIR="$SCENE_DIR/image_00/data"
    RIGHT_IMG_DIR="$SCENE_DIR/image_01/data"
    OUTPUT_JSON="$OUTPUT_DIR/${SCENE}_conf_thresh=${CONF_THRESH}_nms_dist=${NMS_DIST}_nn_thresh=${NN_THRESH}_matches_raw.json"

    echo "========================================"
    echo "开始处理场景: $SCENE"
    echo "左图目录: $LEFT_IMG_DIR"
    echo "右图目录: $RIGHT_IMG_DIR"
    echo "输出文件: $OUTPUT_JSON"
    echo "========================================"

    conda run --no-capture-output -n stereo-calib-vis python "$SCRIPT_PATH" \
        --left_img_dir "$LEFT_IMG_DIR" \
        --right_img_dir "$RIGHT_IMG_DIR" \
        --output "$OUTPUT_JSON" \
        --conf_thresh ${CONF_THRESH} \
        --nms_dist ${NMS_DIST} \
        --nn_thresh ${NN_THRESH} \
        
done


# conda activate stereo-calib-vis
# python superpoint_stereo_match.py --left_img_dir /home/hello/pml/mycalib/blender-file/stereo_panoramic_01/image_00/data --right_img_dir /home/hello/pml/mycalib/blender-file/stereo_panoramic_01/image_01/data --weights /home/hello/pml/mycalib/matchmodel/SuperPointPretrainedNetwork/superpoint_v1.pth --output /home/hello/pml/mycalib/stereo_calib/result/match_points/panoramic_01_matches_raw.json
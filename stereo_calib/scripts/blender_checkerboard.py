"""
Blender 棋盘格标定场景生成脚本

在 Blender 中生成棋盘格标定板，双目相机从多个角度拍摄，
输出图像对供 OpenCV 张氏标定法使用。

用法（在 Blender 脚本编辑器或命令行中运行）：
    直接运行本脚本的 __main__ 部分即可。
"""

import bpy
import math
import json
import os
import random
import mathutils


# ─── 棋盘格创建 ──────────────────────────────────────────────────────────────

def create_checkerboard(rows=6, cols=9, square_size=0.03,
                        name="Checkerboard"):
    """
    创建棋盘格标定板（网格模型 + 交替黑白材质）

    参数:
        rows:        内角点行数（棋盘格方块行数 = rows+1）
        cols:        内角点列数（棋盘格方块列数 = cols+1）
        square_size: 每个方块的边长（米）
        name:        对象名称

    返回:
        board: Blender 对象
    """
    num_rows = rows + 1   # 方块行数
    num_cols = cols + 1   # 方块列数
    board_width = num_cols * square_size
    board_height = num_rows * square_size

    # 创建单个平面作为棋盘格底板
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
    board = bpy.context.object
    board.name = name
    board.scale = (board_width, board_height, 1)
    bpy.ops.object.transform_apply(scale=True)

    # ── 使用 Shader Nodes 实现棋盘格纹理 ────────────────────────────────
    mat = bpy.data.materials.new(name="Checkerboard_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # 清除默认节点
    nodes.clear()

    # 输出节点
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (400, 0)

    # Principled BSDF
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (200, 0)
    bsdf.inputs['Roughness'].default_value = 1.0
    bsdf.inputs['Specular IOR Level'].default_value = 0.0
    links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])

    # Checker Texture
    checker = nodes.new(type='ShaderNodeTexChecker')
    checker.location = (-200, 0)
    checker.inputs['Color1'].default_value = (0.0, 0.0, 0.0, 1.0)  # 黑
    checker.inputs['Color2'].default_value = (1.0, 1.0, 1.0, 1.0)  # 白
    # Scale = 方块数的一半（Checker Texture 一个周期 = 2 个方块）
    # 平面 UV 范围 [0,1]，所以沿 U 方向 scale = num_cols/2，沿 V 方向 scale = num_rows/2
    # Checker Texture 的 Scale 是各向同性的，需要用 Mapping 节点分别控制
    checker.inputs['Scale'].default_value = 1.0

    # Texture Coordinate
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    tex_coord.location = (-600, 0)

    # Mapping 节点：分别控制 U/V 方向的缩放
    mapping = nodes.new(type='ShaderNodeMapping')
    mapping.location = (-400, 0)
    mapping.inputs['Scale'].default_value = (num_cols / 2.0, num_rows / 2.0, 1.0)

    links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], checker.inputs['Vector'])
    links.new(checker.outputs['Color'], bsdf.inputs['Base Color'])

    board.data.materials.append(mat)

    print(f"棋盘格创建成功: {name}")
    print(f"  内角点: {cols} x {rows}")
    print(f"  方块数: {num_cols} x {num_rows}")
    print(f"  方块大小: {square_size}m")
    print(f"  总尺寸: {board_width:.3f}m x {board_height:.3f}m")

    return board


# ─── 双目相机创建 ─────────────────────────────────────────────────────────────

def create_stereo_camera(baseline=0.2, lens=35, sensor_width=36):
    """
    创建双目相机装备（固定位置，朝 -Z 方向拍摄）

    参数:
        baseline:     基线距离（米）
        lens:         焦距（mm）
        sensor_width: 传感器宽度（mm）

    返回:
        (camera_rig, camera_left, camera_right)
    """
    # 左相机
    bpy.ops.object.camera_add(location=(0, 0, 0))
    camera_left = bpy.context.object
    camera_left.name = "Camera_Left"
    camera_left.data.name = "Camera_Left_Data"

    # 右相机
    bpy.ops.object.camera_add(location=(0, 0, 0))
    camera_right = bpy.context.object
    camera_right.name = "Camera_Right"
    camera_right.data.name = "Camera_Right_Data"

    for cam in [camera_left, camera_right]:
        cam.data.lens = lens
        cam.data.sensor_width = sensor_width
        cam.data.clip_start = 0.01
        cam.data.clip_end = 100
        # 相机朝向 -Z（Blender 默认），不额外旋转
        cam.rotation_euler = (0, 0, 0)

    # 装备空对象
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    camera_rig = bpy.context.object
    camera_rig.name = "Stereo_Camera_Rig"
    camera_rig.empty_display_size = 0.3

    camera_left.parent = camera_rig
    camera_right.parent = camera_rig
    camera_left.location = (-baseline / 2, 0, 0)
    camera_right.location = (baseline / 2, 0, 0)

    bpy.context.scene.camera = camera_left

    print(f"双目相机创建成功: 基线={baseline}m, 焦距={lens}mm")
    return camera_rig, camera_left, camera_right


# ─── 坐标系转换 ──────────────────────────────────────────────────────────────

_BLENDER_TO_OPENCV = mathutils.Matrix((
    (1,  0,  0, 0),
    (0, -1,  0, 0),
    (0,  0, -1, 0),
    (0,  0,  0, 1),
))


def compute_camera_intrinsics(cam_obj):
    """计算相机内参（与 blender.py 中相同）"""
    scene = bpy.context.scene
    render = scene.render
    cam = cam_obj.data

    res_x = render.resolution_x
    res_y = render.resolution_y

    if cam.sensor_fit == 'VERTICAL' or \
       (cam.sensor_fit == 'AUTO' and res_x < res_y):
        f_px = (cam.lens / cam.sensor_height) * res_y
    else:
        f_px = (cam.lens / cam.sensor_width) * res_x

    return {
        "fx": f_px, "fy": f_px,
        "cx": res_x / 2.0, "cy": res_y / 2.0,
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0,
    }


def get_world_to_cam_opencv(cam_obj):
    """返回世界坐标 -> 相机坐标的变换（OpenCV 约定）"""
    M = _BLENDER_TO_OPENCV @ cam_obj.matrix_world.inverted()
    R = M.to_3x3()
    t = M.translation
    R_flat = [R[i][j] for i in range(3) for j in range(3)]
    return R_flat, [t.x, t.y, t.z]


# ─── 多角度渲染 ──────────────────────────────────────────────────────────────

def render_checkerboard_stereo(
    output_folder="//checkerboard_calib/",
    num_images=25,
    board_name="Checkerboard",
    rig_name="Stereo_Camera_Rig",
    board_distance_range=(0.3, 0.8),
    board_x_range=(-0.15, 0.15),
    board_y_range=(-0.10, 0.10),
    board_tilt_range=(-40, 40),
    board_pan_range=(-40, 40),
    board_roll_range=(-15, 15),
    random_seed=42,
    file_format='PNG',
    export_params=True,
):
    """
    随机摆放棋盘格在相机前方，渲染多组双目图像。

    相机固定不动，棋盘格在相机前方随机移动和旋转，
    模拟手持标定板在相机前多角度展示。

    参数:
        output_folder:        输出目录
        num_images:           拍摄张数（每张 = 左 + 右各 1 张）
        board_name:           棋盘格对象名称
        rig_name:             相机装备名称
        board_distance_range: 棋盘格到相机的距离范围（米）
        board_x_range:        棋盘格水平偏移范围（米）
        board_y_range:        棋盘格垂直偏移范围（米）
        board_tilt_range:     棋盘格俯仰角范围（度）
        board_pan_range:      棋盘格偏航角范围（度）
        board_roll_range:     棋盘格滚转角范围（度）
        random_seed:          随机种子
        file_format:          图像格式
        export_params:        是否导出相机参数
    """
    scene = bpy.context.scene
    board = bpy.data.objects.get(board_name)
    camera_left = bpy.data.objects.get("Camera_Left")
    camera_right = bpy.data.objects.get("Camera_Right")
    camera_rig = bpy.data.objects.get(rig_name)

    if not board:
        print(f"错误：未找到棋盘格对象 '{board_name}'")
        return
    if not camera_left or not camera_right:
        print("错误：未找到双目相机")
        return

    # ── 保存原始状态 ──────────────────────────────────────────────────────
    original_camera = scene.camera
    original_filepath = scene.render.filepath
    original_board_loc = board.location.copy()
    original_board_rot = board.rotation_euler.copy()

    scene.render.image_settings.file_format = file_format
    ext = "jpg" if file_format == 'JPEG' else file_format.lower()

    if random_seed is not None:
        random.seed(random_seed)

    abs_output = bpy.path.abspath(output_folder)
    os.makedirs(abs_output, exist_ok=True)

    # ── 导出相机内参和立体外参（所有帧共用） ──────────────────────────────
    if export_params:
        bpy.context.view_layer.update()
        left_intr = compute_camera_intrinsics(camera_left)
        right_intr = compute_camera_intrinsics(camera_right)

        M_left_inv = _BLENDER_TO_OPENCV @ camera_left.matrix_world.inverted()
        C_r = M_left_inv @ camera_right.matrix_world.translation.to_4d()
        t_stereo = [-C_r.x, -C_r.y, -C_r.z]
        R_stereo = [1, 0, 0, 0, 1, 0, 0, 0, 1]

        camera_params = {
            "image_size": {
                "width": scene.render.resolution_x,
                "height": scene.render.resolution_y,
            },
            "left": left_intr,
            "right": right_intr,
            "extrinsics": {"R": R_stereo, "t": t_stereo},
        }
        params_path = os.path.join(abs_output, "camera_params.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(camera_params, f, indent=2)
        print(f"相机参数已导出: {params_path}")

    # ── 相机朝 -Z，棋盘格放在相机前方（Z 负方向） ─────────────────────────
    d_min, d_max = board_distance_range
    x_min, x_max = board_x_range
    y_min, y_max = board_y_range
    tilt_min, tilt_max = board_tilt_range
    pan_min, pan_max = board_pan_range
    roll_min, roll_max = board_roll_range

    rig_pos = camera_rig.matrix_world.translation

    print(f"\n开始棋盘格标定渲染")
    print(f"  图像对数:   {num_images}")
    print(f"  距离范围:   [{d_min}, {d_max}]m")
    print(f"  俯仰范围:   [{tilt_min}°, {tilt_max}°]")
    print(f"  偏航范围:   [{pan_min}°, {pan_max}°]")

    for idx in range(num_images):
        # 随机位姿
        dist = random.uniform(d_min, d_max)
        off_x = random.uniform(x_min, x_max)
        off_y = random.uniform(y_min, y_max)
        tilt = math.radians(random.uniform(tilt_min, tilt_max))
        pan = math.radians(random.uniform(pan_min, pan_max))
        roll = math.radians(random.uniform(roll_min, roll_max))

        # 棋盘格位置：在相机前方 -Z 方向
        board.location = (
            rig_pos.x + off_x,
            rig_pos.y + off_y,
            rig_pos.z - dist,
        )

        # 棋盘格朝向：面对相机（+Z 方向），加上随机倾斜
        # 棋盘格平面法线默认朝 +Z，正好面对相机的 -Z 朝向
        board.rotation_euler = (tilt, pan, roll)

        bpy.context.view_layer.update()

        # ── 渲染 ─────────────────────────────────────────────────────────
        left_name = f"left_{idx:04d}"
        right_name = f"right_{idx:04d}"

        scene.camera = camera_left
        scene.render.filepath = output_folder + left_name
        bpy.ops.render.render(write_still=True)

        scene.camera = camera_right
        scene.render.filepath = output_folder + right_name
        bpy.ops.render.render(write_still=True)

        progress = (idx + 1) / num_images * 100
        print(f"  [{progress:5.1f}%] ({idx+1}/{num_images})  "
              f"dist={dist:.3f}m  tilt={math.degrees(tilt):+.1f}°  "
              f"pan={math.degrees(pan):+.1f}°")

    # ── 导出标定板参数 ────────────────────────────────────────────────────
    if export_params:
        # 从棋盘格的 Mapping 节点中提取 rows/cols 信息
        # 直接使用创建时传入的参数更可靠，写入配置文件供标定脚本读取
        board_info_path = os.path.join(abs_output, "board_info.json")
        with open(board_info_path, "w", encoding="utf-8") as f:
            json.dump({
                "description": "棋盘格标定板参数，供 OpenCV 张氏标定使用",
                "num_images": num_images,
                "file_format": ext,
            }, f, indent=2)
        print(f"标定板参数已导出: {board_info_path}")

    # ── 恢复原始状态 ──────────────────────────────────────────────────────
    board.location = original_board_loc
    board.rotation_euler = original_board_rot
    scene.camera = original_camera
    scene.render.filepath = original_filepath

    print(f"\n棋盘格标定渲染完成！共 {num_images} 对 / {num_images * 2} 张图像")
    print(f"输出目录: {abs_output}")


# ─── 场景灯光 ─────────────────────────────────────────────────────────────────

def setup_lighting():
    """添加均匀照明，确保棋盘格对比度清晰"""
    # 主光源
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
    sun = bpy.context.object
    sun.name = "Calib_Sun"
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(30), 0, 0)

    # 补光（减少阴影）
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 2))
    fill = bpy.context.object
    fill.name = "Calib_Fill"
    fill.data.energy = 50.0
    fill.data.size = 3.0
    fill.rotation_euler = (math.radians(0), 0, 0)

    print("照明设置完成")


# ─── 主函数 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── 参数配置 ──────────────────────────────────────────────────────────

    # 棋盘格参数（OpenCV 约定：内角点数）
    BOARD_ROWS = 6          # 内角点行数
    BOARD_COLS = 9          # 内角点列数
    SQUARE_SIZE = 0.03      # 方块边长（米），即 3cm

    # 相机参数
    BASELINE = 0.2          # 基线（米）
    LENS = 35               # 焦距（mm）
    SENSOR_WIDTH = 36       # 传感器宽度（mm）

    # 渲染参数
    NUM_IMAGES = 25         # 拍摄张数
    RANDOM_SEED = 42
    OUTPUT_FOLDER = "//checkerboard_calib/"

    # 棋盘格摆放范围
    DISTANCE_RANGE = (0.3, 0.8)    # 距相机距离（米）
    X_RANGE = (-0.15, 0.15)        # 水平偏移（米）
    Y_RANGE = (-0.10, 0.10)        # 垂直偏移（米）
    TILT_RANGE = (-40, 40)         # 俯仰角（度）
    PAN_RANGE = (-40, 40)          # 偏航角（度）
    ROLL_RANGE = (-15, 15)         # 滚转角（度）

    # ── 步骤 1：创建棋盘格 ────────────────────────────────────────────────
    board = create_checkerboard(
        rows=BOARD_ROWS,
        cols=BOARD_COLS,
        square_size=SQUARE_SIZE,
    )

    # ── 步骤 2：创建双目相机 ──────────────────────────────────────────────
    rig, cam_l, cam_r = create_stereo_camera(
        baseline=BASELINE,
        lens=LENS,
        sensor_width=SENSOR_WIDTH,
    )

    # ── 步骤 3：设置照明 ──────────────────────────────────────────────────
    setup_lighting()

    # ── 步骤 4：多角度渲染 ────────────────────────────────────────────────
    render_checkerboard_stereo(
        output_folder=OUTPUT_FOLDER,
        num_images=NUM_IMAGES,
        board_distance_range=DISTANCE_RANGE,
        board_x_range=X_RANGE,
        board_y_range=Y_RANGE,
        board_tilt_range=TILT_RANGE,
        board_pan_range=PAN_RANGE,
        board_roll_range=ROLL_RANGE,
        random_seed=RANDOM_SEED,
        file_format='PNG',
        export_params=True,
    )

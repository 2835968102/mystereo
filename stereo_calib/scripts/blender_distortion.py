import bpy
import math
import json
import os
import random
import mathutils


# ─── 摄像头内参计算（带畸变和焦距变化） ────────────────────────────────────────

def compute_camera_intrinsics_with_distortion(
    cam_obj,
    k1=0.0, k2=0.0, k3=0.0,
    p1=0.0, p2=0.0,
    focal_length_offset=0.0
):
    """
    计算摄像头内参，支持畸变系数和焦距偏移

    参数:
        cam_obj:              摄像头对象
        k1, k2, k3:           径向畸变系数
        p1, p2:               切向畸变系数
        focal_length_offset:  焦距偏移量（像素），加到 fx, fy 上

    返回:
        内参字典：fx, fy, cx, cy, k1~k3, p1, p2
    """
    scene = bpy.context.scene
    render = scene.render
    cam = cam_obj.data

    res_x = render.resolution_x
    res_y = render.resolution_y

    # 根据传感器适配模式计算像素焦距
    if cam.sensor_fit == 'VERTICAL' or \
       (cam.sensor_fit == 'AUTO' and res_x < res_y):
        f_px = (cam.lens / cam.sensor_height) * res_y
    else:
        f_px = (cam.lens / cam.sensor_width) * res_x

    # 加入焦距偏移
    f_px += focal_length_offset

    return {
        "fx": f_px,
        "fy": f_px,
        "cx": res_x / 2.0,
        "cy": res_y / 2.0,
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "p1": p1,
        "p2": p2,
    }


def sample_focal_length_offset(mean=0.0, std=1.0):
    """
    从高斯分布中采样焦距偏移值

    参数:
        mean: 高斯分布均值（像素）
        std:  高斯分布标准差（像素）

    返回:
        焦距偏移值（像素）
    """
    return random.gauss(mean, std)


def sample_distortion_coefficients(
    k1_range=(0.0, 0.0),
    k2_range=(0.0, 0.0),
    k3_range=(0.0, 0.0),
    p1_range=(0.0, 0.0),
    p2_range=(0.0, 0.0)
):
    """
    随机采样畸变系数（均匀分布）

    参数:
        k1_range, k2_range, k3_range: 径向畸变系数范围
        p1_range, p2_range:           切向畸变系数范围

    返回:
        畸变系数字典
    """
    return {
        "k1": random.uniform(*k1_range),
        "k2": random.uniform(*k2_range),
        "k3": random.uniform(*k3_range),
        "p1": random.uniform(*p1_range),
        "p2": random.uniform(*p2_range),
    }


# ─── 坐标系转换 ────────────────────────────────────────────────────────────────

_BLENDER_TO_OPENCV = mathutils.Matrix((
    (1,  0,  0, 0),
    (0, -1,  0, 0),
    (0,  0, -1, 0),
    (0,  0,  0, 1),
))


def get_world_to_cam_opencv(cam_obj):
    """
    返回世界坐标 -> 摄像头坐标的变换（OpenCV约定）

    返回:
        R_flat: 3x3旋转矩阵，行优先展平为9个元素的列表
        t_list: 3元素平移向量列表
    """
    M = _BLENDER_TO_OPENCV @ cam_obj.matrix_world.inverted()
    R = M.to_3x3()
    t = M.translation
    R_flat = [R[i][j] for i in range(3) for j in range(3)]
    return R_flat, [t.x, t.y, t.z]


# ─── 摄像头参数导出（支持畸变和焦距变化） ──────────────────────────────────────

def export_camera_params_with_distortion(
    output_path="//camera_params.json",
    left_distortion=None,
    right_distortion=None,
    left_focal_offset=0.0,
    right_focal_offset=0.0
):
    """
    导出双目摄像头参数到JSON文件（包含畸变系数）

    参数:
        output_path:         输出路径
        left_distortion:     左摄像头畸变系数字典 {"k1", "k2", "k3", "p1", "p2"}
        right_distortion:    右摄像头畸变系数字典
        left_focal_offset:   左摄像头焦距偏移（像素）
        right_focal_offset:  右摄像头焦距偏移（像素）
    """
    camera_left = bpy.data.objects.get("Camera_Left")
    camera_right = bpy.data.objects.get("Camera_Right")

    if not camera_left or not camera_right:
        print("错误：未找到双目摄像头！请先运行 create_stereo_camera_for_orbit()")
        return None

    if left_distortion is None:
        left_distortion = {"k1": 0.0, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0}
    if right_distortion is None:
        right_distortion = {"k1": 0.0, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0}

    bpy.context.view_layer.update()

    render = bpy.context.scene.render
    width = render.resolution_x
    height = render.resolution_y

    left_intr = compute_camera_intrinsics_with_distortion(
        camera_left,
        k1=left_distortion.get("k1", 0.0),
        k2=left_distortion.get("k2", 0.0),
        k3=left_distortion.get("k3", 0.0),
        p1=left_distortion.get("p1", 0.0),
        p2=left_distortion.get("p2", 0.0),
        focal_length_offset=left_focal_offset
    )

    right_intr = compute_camera_intrinsics_with_distortion(
        camera_right,
        k1=right_distortion.get("k1", 0.0),
        k2=right_distortion.get("k2", 0.0),
        k3=right_distortion.get("k3", 0.0),
        p1=right_distortion.get("p1", 0.0),
        p2=right_distortion.get("p2", 0.0),
        focal_length_offset=right_focal_offset
    )

    # ── 立体外参 ──────────────────────────────────────────────────────────────
    M_left_inv = _BLENDER_TO_OPENCV @ camera_left.matrix_world.inverted()
    cam_r_pos_world_4d = camera_right.matrix_world.translation.to_4d()
    C_r = M_left_inv @ cam_r_pos_world_4d

    t_stereo = [-C_r.x, -C_r.y, -C_r.z]
    R_stereo = [1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0]

    params = {
        "image_size": {"width": width, "height": height},
        "left": left_intr,
        "right": right_intr,
        "extrinsics": {
            "R": R_stereo,
            "t": t_stereo,
        },
    }

    abs_path = bpy.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    baseline = abs(t_stereo[0])
    print(f"摄像头参数已导出: {abs_path}")
    print(f"  分辨率:     {width} x {height}")
    print(f"  左相机 fx:  {left_intr['fx']:.4f} px (偏移: {left_focal_offset:.4f} px)")
    print(f"  右相机 fx:  {right_intr['fx']:.4f} px (偏移: {right_focal_offset:.4f} px)")
    print(f"  左相机畸变: k1={left_intr['k1']:.6f}, k2={left_intr['k2']:.6f}, k3={left_intr['k3']:.6f}")
    print(f"  左相机切向: p1={left_intr['p1']:.6f}, p2={left_intr['p2']:.6f}")
    print(f"  基线:       {baseline:.4f} m")

    return params


# ─── 创建场景 ──────────────────────────────────────────────────────────────────

def create_stereo_camera_for_orbit(baseline=0.2, lens=35, sensor_width=36):
    """
    创建用于绕轨道旋转渲染的双目摄像头

    参数:
        baseline:     双目基线距离（米），默认 0.2m
        lens:         镜头焦距（mm），默认 35mm
        sensor_width: 传感器宽度（mm），默认 36mm

    返回:
        (camera_rig, camera_left, camera_right)
    """
    # 左摄像头
    bpy.ops.object.camera_add(location=(0, 0, 0))
    camera_left = bpy.context.object
    camera_left.name = "Camera_Left"
    camera_left.data.name = "Camera_Left_Data"

    # 右摄像头
    bpy.ops.object.camera_add(location=(0, 0, 0))
    camera_right = bpy.context.object
    camera_right.name = "Camera_Right"
    camera_right.data.name = "Camera_Right_Data"

    for camera in [camera_left, camera_right]:
        camera.data.lens = lens
        camera.data.sensor_width = sensor_width
        camera.data.clip_start = 0.1
        camera.data.clip_end = 1000
        camera.rotation_euler = (math.radians(90), 0, 0)

    # 创建装备空对象
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    camera_rig = bpy.context.object
    camera_rig.name = "Stereo_Camera_Rig"
    camera_rig.empty_display_size = 0.3

    # 设置父子关系
    camera_left.parent = camera_rig
    camera_right.parent = camera_rig
    camera_left.location = (-baseline / 2, 0, 0)
    camera_right.location = (baseline / 2, 0, 0)

    bpy.context.scene.camera = camera_left

    print(f"双目轨道摄像头创建成功！基线: {baseline}m，焦距: {lens}mm")
    return camera_rig, camera_left, camera_right


# ─── 原地随机旋转渲染360°全景（带畸变和焦距变化）──────────────────────────────

def render_panoramic_stereo_with_distortion(
    output_folder="//stereo_panoramic/",
    num_images=60,
    camera_position=(0, 0, 2),
    azimuth_range=(0, 360),
    elevation_range=(-80, 80),
    max_angle_step=None,
    random_seed=None,
    file_format='PNG',
    rig_name="Stereo_Camera_Rig",
    export_poses=True,
    # 焦距变化参数
    focal_length_mean=0.0,
    focal_length_std=2.0,
    same_focal_for_stereo=True,
    # 畸变参数范围
    k1_range=(0.0, 0.0),
    k2_range=(0.0, 0.0),
    k3_range=(0.0, 0.0),
    p1_range=(0.0, 0.0),
    p2_range=(0.0, 0.0),
    same_distortion_for_stereo=True,
    # 是否对每帧独立采样
    randomize_per_frame=True,
):
    """
    相机固定在 camera_position 原地随机旋转，拍摄360°全景图像序列。
    支持畸变系数和焦距变化（高斯分布）。

    参数:
        output_folder:           输出文件夹（支持 // 相对路径）
        num_images:              渲染图像对数（每对含左+右各1张）
        camera_position:         相机固定位置 (x, y, z)
        azimuth_range:           水平方位角随机采样范围（度）
        elevation_range:         俯仰角随机采样范围（度）
        max_angle_step:          相邻帧之间最大角度变化（度）
        random_seed:             随机种子；None 表示每次不同
        file_format:             图像格式 ('PNG', 'JPEG', 'OPEN_EXR')
        rig_name:                Blender 中相机装备的对象名称
        export_poses:            是否为每对图像输出参数 JSON

        焦距变化参数:
        focal_length_mean:       焦距偏移高斯分布均值（像素）
        focal_length_std:        焦距偏移高斯分布标准差（像素）
        same_focal_for_stereo:   左右摄像头是否使用相同的焦距偏移

        畸变参数范围:
        k1_range, k2_range, k3_range: 径向畸变系数采样范围
        p1_range, p2_range:           切向畸变系数采样范围
        same_distortion_for_stereo:   左右摄像头是否使用相同的畸变系数

        randomize_per_frame:     每帧是否重新采样焦距和畸变（True）或全局固定（False）
    """
    scene = bpy.context.scene
    camera_left = bpy.data.objects.get("Camera_Left")
    camera_right = bpy.data.objects.get("Camera_Right")
    camera_rig = bpy.data.objects.get(rig_name)

    if not camera_left or not camera_right:
        print("错误：未找到双目摄像头！请先运行 create_stereo_camera_for_orbit()")
        return
    if not camera_rig:
        print(f"错误：未找到相机装备 '{rig_name}'！")
        return

    # ── 保存原始状态 ──────────────────────────────────────────────────────────
    original_camera = scene.camera
    original_filepath = scene.render.filepath
    original_location = camera_rig.location.copy()
    original_rotation = camera_rig.rotation_euler.copy()

    scene.render.image_settings.file_format = file_format
    ext = "jpg" if file_format == 'JPEG' else file_format.lower()

    if random_seed is not None:
        random.seed(random_seed)

    # ── 静态参数 ────────────────────────────────────────────────────────────────
    width = scene.render.resolution_x
    height = scene.render.resolution_y

    abs_output = bpy.path.abspath(output_folder)
    os.makedirs(abs_output, exist_ok=True)
    print(f"输出目录: {abs_output}")

    # 将相机装备固定到指定位置
    cam_pos_vec = mathutils.Vector(camera_position)
    if camera_rig.parent:
        camera_rig.location = camera_rig.parent.matrix_world.inverted() @ cam_pos_vec
    else:
        camera_rig.location = cam_pos_vec

    el_min, el_max = elevation_range
    az_min, az_max = azimuth_range
    all_frames_info = []

    # ── 如果不是每帧随机，则提前采样全局焦距和畸变值 ──────────────────────────
    if not randomize_per_frame:
        global_focal_offset_l = sample_focal_length_offset(focal_length_mean, focal_length_std)
        if same_focal_for_stereo:
            global_focal_offset_r = global_focal_offset_l
        else:
            global_focal_offset_r = sample_focal_length_offset(focal_length_mean, focal_length_std)

        global_distortion_l = sample_distortion_coefficients(
            k1_range, k2_range, k3_range, p1_range, p2_range
        )
        if same_distortion_for_stereo:
            global_distortion_r = global_distortion_l
        else:
            global_distortion_r = sample_distortion_coefficients(
                k1_range, k2_range, k3_range, p1_range, p2_range
            )

    print(f"开始随机全景渲染（相机原地旋转，含畸变和焦距变化）")
    print(f"  图像对数:           {num_images}")
    print(f"  相机位置:           {camera_position}")
    print(f"  焦距偏移分布:       N({focal_length_mean}, {focal_length_std}²) px")
    print(f"  每帧随机采样:       {randomize_per_frame}")
    print(f"  左右同步焦距:       {same_focal_for_stereo}")
    print(f"  左右同步畸变:       {same_distortion_for_stereo}")
    print(f"  k1 范围:            {k1_range}")
    print(f"  k2 范围:            {k2_range}")
    print(f"  k3 范围:            {k3_range}")
    print(f"  p1 范围:            {p1_range}")
    print(f"  p2 范围:            {p2_range}")

    prev_az = None
    prev_el = None

    for idx in range(num_images):
        # ── 采样焦距和畸变 ──────────────────────────────────────────────────────
        if randomize_per_frame:
            focal_offset_l = sample_focal_length_offset(focal_length_mean, focal_length_std)
            if same_focal_for_stereo:
                focal_offset_r = focal_offset_l
            else:
                focal_offset_r = sample_focal_length_offset(focal_length_mean, focal_length_std)

            distortion_l = sample_distortion_coefficients(
                k1_range, k2_range, k3_range, p1_range, p2_range
            )
            if same_distortion_for_stereo:
                distortion_r = distortion_l
            else:
                distortion_r = sample_distortion_coefficients(
                    k1_range, k2_range, k3_range, p1_range, p2_range
                )
        else:
            focal_offset_l = global_focal_offset_l
            focal_offset_r = global_focal_offset_r
            distortion_l = global_distortion_l
            distortion_r = global_distortion_r

        # ── 采样视角 ────────────────────────────────────────────────────────────
        if max_angle_step is None or prev_az is None:
            azimuth_deg   = random.uniform(az_min, az_max)
            elevation_deg = random.uniform(el_min, el_max)
        else:
            azimuth_deg   = prev_az + random.uniform(-max_angle_step, max_angle_step)
            elevation_deg = prev_el + random.uniform(-max_angle_step, max_angle_step)
            azimuth_deg   = max(az_min, min(az_max, azimuth_deg))
            elevation_deg = max(el_min, min(el_max, elevation_deg))

        prev_az = azimuth_deg
        prev_el = elevation_deg
        az_rad = math.radians(azimuth_deg)
        el_rad = math.radians(elevation_deg)

        # 将方位角+俯仰角转为朝向向量
        cos_el = math.cos(el_rad)
        look_dir = mathutils.Vector((
            cos_el * math.cos(az_rad),
            cos_el * math.sin(az_rad),
            math.sin(el_rad),
        ))
        rot_quat = look_dir.to_track_quat('Y', 'Z')

        if camera_rig.parent:
            parent_rot_inv = camera_rig.parent.matrix_world.to_3x3().inverted()
            local_rot_mat = parent_rot_inv @ rot_quat.to_matrix()
            camera_rig.rotation_euler = local_rot_mat.to_euler()
        else:
            camera_rig.rotation_euler = rot_quat.to_euler()

        bpy.context.view_layer.update()

        # ── 文件命名与渲染 ──────────────────────────────────────────────────────
        left_name = f"left_{idx:04d}"
        right_name = f"right_{idx:04d}"

        # Blender 会自动添加扩展名，filepath 不应包含扩展名
        scene.camera = camera_left
        scene.render.filepath = os.path.join(abs_output, left_name)
        bpy.ops.render.render(write_still=True)

        scene.camera = camera_right
        scene.render.filepath = os.path.join(abs_output, right_name)
        bpy.ops.render.render(write_still=True)

        # ── 导出当前帧摄像头参数（带畸变和焦距变化） ──────────────────────────
        if export_poses:
            R_l, t_l = get_world_to_cam_opencv(camera_left)
            R_r, t_r = get_world_to_cam_opencv(camera_right)

            # 左右摄像头内参
            left_intr = compute_camera_intrinsics_with_distortion(
                camera_left,
                k1=distortion_l["k1"],
                k2=distortion_l["k2"],
                k3=distortion_l["k3"],
                p1=distortion_l["p1"],
                p2=distortion_l["p2"],
                focal_length_offset=focal_offset_l
            )

            right_intr = compute_camera_intrinsics_with_distortion(
                camera_right,
                k1=distortion_r["k1"],
                k2=distortion_r["k2"],
                k3=distortion_r["k3"],
                p1=distortion_r["p1"],
                p2=distortion_r["p2"],
                focal_length_offset=focal_offset_r
            )

            # 立体外参
            M_left_inv = _BLENDER_TO_OPENCV @ camera_left.matrix_world.inverted()
            C_r = M_left_inv @ camera_right.matrix_world.translation.to_4d()
            t_stereo = [-C_r.x, -C_r.y, -C_r.z]
            R_stereo = [1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0]

            frame_params = {
                "image_size": {"width": width, "height": height},
                "left":  left_intr,
                "right": right_intr,
                "extrinsics": {
                    "R": R_stereo,
                    "t": t_stereo,
                },
                "left_pose":  {"R": R_l, "t": t_l},
                "right_pose": {"R": R_r, "t": t_r},
                "meta": {
                    "index":           idx,
                    "azimuth_deg":     round(azimuth_deg, 4),
                    "elevation_deg":   round(elevation_deg, 4),
                    "camera_position": list(camera_position),
                    "left_image":      f"{left_name}.{ext}",
                    "right_image":     f"{right_name}.{ext}",
                    "focal_offset_left":  round(focal_offset_l, 4),
                    "focal_offset_right": round(focal_offset_r, 4),
                },
            }

            params_path = os.path.join(abs_output, f"camera_params_{idx:04d}.json")
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(frame_params, f, indent=2)

            all_frames_info.append({
                "index":         idx,
                "azimuth_deg":   round(azimuth_deg, 4),
                "elevation_deg": round(elevation_deg, 4),
                "left_image":    f"{left_name}.{ext}",
                "right_image":   f"{right_name}.{ext}",
                "left_pose":     {"R": R_l, "t": t_l},
                "right_pose":    {"R": R_r, "t": t_r},
                "focal_offset_left":  round(focal_offset_l, 4),
                "focal_offset_right": round(focal_offset_r, 4),
                "distortion_left":    {k: round(v, 6) for k, v in distortion_l.items()},
                "distortion_right":   {k: round(v, 6) for k, v in distortion_r.items()},
            })

        progress = (idx + 1) / num_images * 100
        print(f"  [{progress:5.1f}%] ({idx+1:4d}/{num_images})  "
              f"az={azimuth_deg:6.1f}°  el={elevation_deg:5.1f}°  "
              f"f_offset_L={focal_offset_l:+6.2f}  f_offset_R={focal_offset_r:+6.2f}")

    # ── 导出汇总 JSON ─────────────────────────────────────────────────────────
    if export_poses and all_frames_info:
        summary = {
            "description":       "随机全景双目渲染位姿汇总（含畸变和焦距变化）",
            "convention":        "X_cam = R @ X_world + t  (OpenCV约定: X右 Y下 Z前)",
            "camera_position":   list(camera_position),
            "azimuth_range":     list(azimuth_range),
            "elevation_range":   list(elevation_range),
            "random_seed":       random_seed,
            "num_images":        num_images,
            "distortion_config": {
                "focal_length_mean":         focal_length_mean,
                "focal_length_std":          focal_length_std,
                "same_focal_for_stereo":     same_focal_for_stereo,
                "k1_range":                  list(k1_range),
                "k2_range":                  list(k2_range),
                "k3_range":                  list(k3_range),
                "p1_range":                  list(p1_range),
                "p2_range":                  list(p2_range),
                "same_distortion_for_stereo": same_distortion_for_stereo,
                "randomize_per_frame":       randomize_per_frame,
            },
            "frames":            all_frames_info,
        }
        summary_path = os.path.join(abs_output, "camera_poses_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"汇总位姿已导出: {summary_path}")

    # ── 恢复原始状态 ──────────────────────────────────────────────────────────
    camera_rig.location = original_location
    camera_rig.rotation_euler = original_rotation
    scene.camera = original_camera
    scene.render.filepath = original_filepath

    print(f"随机全景渲染完成！共 {num_images} 对 / {num_images * 2} 张图像")


# ─── 主函数 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── 参数配置 ──────────────────────────────────────────────────────────────

    # 相机固定位置
    CAMERA_POSITION = (0.0, 0.0, 2.0)

    # 渲染图像对数
    NUM_IMAGES = 100

    # 视角范围
    AZIMUTH_RANGE = (0, 360)
    ELEVATION_RANGE = (-90, 90)

    # 相邻帧之间最大角度变化
    MAX_ANGLE_STEP = 30

    # 随机种子
    RANDOM_SEED = 42

    # 输出目录（使用绝对路径或相对于blend文件的路径）
    # 注意：必须以 // 开头（相对blend文件所在目录）或使用完整绝对路径
    OUTPUT_FOLDER = "//stereo_panoramic_distortion/"

    # ── 焦距变化参数 ──────────────────────────────────────────────────────────
    FOCAL_LENGTH_MEAN = 0.0      # 焦距偏移高斯分布均值（像素）
    FOCAL_LENGTH_STD = 3.0       # 焦距偏移高斯分布标准差（像素）
    SAME_FOCAL_FOR_STEREO = True # 左右摄像头是否使用相同焦距偏移

    # ── 畸变参数范围 ──────────────────────────────────────────────────────────
    # 径向畸变系数（通常较小）
    K1_RANGE = (-0.01, 0.01)
    K2_RANGE = (-0.001, 0.001)
    K3_RANGE = (-0.0001, 0.0001)

    # 切向畸变系数（通常较小）
    P1_RANGE = (-0.001, 0.001)
    P2_RANGE = (-0.001, 0.001)

    SAME_DISTORTION_FOR_STEREO = True  # 左右摄像头是否使用相同畸变
    RANDOMIZE_PER_FRAME = True         # 每帧是否重新采样焦距和畸变

    # ── 步骤1：创建双目摄像头 ─────────────────────────────────────────────────
    rig, cam_l, cam_r = create_stereo_camera_for_orbit(
        baseline=0.2,
        lens=35,
        sensor_width=36,
    )

    # ── 步骤2：原地随机旋转渲染360°全景（含畸变和焦距变化） ──────────────────
    render_panoramic_stereo_with_distortion(
        output_folder=OUTPUT_FOLDER,
        num_images=NUM_IMAGES,
        camera_position=CAMERA_POSITION,
        azimuth_range=AZIMUTH_RANGE,
        elevation_range=ELEVATION_RANGE,
        max_angle_step=MAX_ANGLE_STEP,
        random_seed=RANDOM_SEED,
        file_format='PNG',
        export_poses=True,
        # 焦距变化参数
        focal_length_mean=FOCAL_LENGTH_MEAN,
        focal_length_std=FOCAL_LENGTH_STD,
        same_focal_for_stereo=SAME_FOCAL_FOR_STEREO,
        # 畸变参数范围
        k1_range=K1_RANGE,
        k2_range=K2_RANGE,
        k3_range=K3_RANGE,
        p1_range=P1_RANGE,
        p2_range=P2_RANGE,
        same_distortion_for_stereo=SAME_DISTORTION_FOR_STEREO,
        randomize_per_frame=RANDOMIZE_PER_FRAME,
    )

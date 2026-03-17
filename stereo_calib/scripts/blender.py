import bpy
import math
import json
import os
import random
import mathutils


# ─── 摄像头内参计算 ────────────────────────────────────────────────────────────

def compute_camera_intrinsics(cam_obj):
    """
    计算摄像头内参（针孔模型，无畸变）

    返回与 stereo_calib 项目兼容的内参字典：
      fx, fy: 像素焦距
      cx, cy: 主点（图像中心）
      k1..k3, p1, p2: 畸变系数（Blender相机无畸变，均为0）
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

    return {
        "fx": f_px,
        "fy": f_px,
        "cx": res_x / 2.0,
        "cy": res_y / 2.0,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "k3": 0.0,
    }


# ─── 坐标系转换 ────────────────────────────────────────────────────────────────

# Blender相机局部坐标：X右，Y上，Z后（沿-Z看）
# OpenCV相机坐标：   X右，Y下，Z前（沿+Z看）
# 转换矩阵：翻转Y和Z
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


# ─── 摄像头参数导出 ────────────────────────────────────────────────────────────

def export_camera_params(output_path="//camera_params.json"):
    """
    导出双目摄像头参数到JSON文件

    输出格式兼容 stereo_calib 项目（example_input.json）：
      - 左右摄像头内参（fx, fy, cx, cy, k1~k3, p1, p2）
      - 立体外参 R, t（右相机相对于左相机，满足 X_r = R*X_l + t）

    参数:
        output_path: 输出路径（支持 // 表示blend文件所在目录）
    """
    camera_left = bpy.data.objects.get("Camera_Left")
    camera_right = bpy.data.objects.get("Camera_Right")

    if not camera_left or not camera_right:
        print("错误：未找到双目摄像头！请先运行 create_plane_with_stereo_camera()")
        return None

    bpy.context.view_layer.update()

    render = bpy.context.scene.render
    width = render.resolution_x
    height = render.resolution_y

    left_intr = compute_camera_intrinsics(camera_left)
    right_intr = compute_camera_intrinsics(camera_right)

    # ── 立体外参 ──────────────────────────────────────────────────────────────
    # X_right = R * X_left + t
    # 两摄像头平行（朝向相同）→ R = I
    # 右摄像头中心在左摄像头坐标系（OpenCV）中的位置 C_r
    # t = -R * C_r = -C_r（因为 R = I）

    M_left_inv = _BLENDER_TO_OPENCV @ camera_left.matrix_world.inverted()
    cam_r_pos_world = camera_left.matrix_world.inverted() @ camera_right.matrix_world.translation
    # 上面已经是左摄像头Blender局部坐标，再乘flip转OpenCV
    cam_r_pos_world_4d = camera_right.matrix_world.translation.to_4d()
    C_r = M_left_inv @ cam_r_pos_world_4d  # 右摄像头中心在左摄像头OpenCV坐标系中

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
    print(f"  分辨率:  {width} x {height}")
    print(f"  fx = fy: {left_intr['fx']:.4f} px")
    print(f"  主点:    ({left_intr['cx']:.1f}, {left_intr['cy']:.1f})")
    print(f"  基线:    {baseline:.4f} m")

    return params


# ─── 创建场景 ──────────────────────────────────────────────────────────────────

def create_plane_with_stereo_camera(baseline=0.2, camera_height=0.1,
                                     plane_size=10.0, plane_location=(0, 0, 0)):
    """
    创建带有双目摄像头的平面

    参数:
        baseline: 两个摄像头之间的距离（米），默认0.2米
        camera_height: 摄像头在平面上方的高度（米），默认0.1米
        plane_size: 平面大小（米）
        plane_location: 平面位置 (x, y, z)
    """

    # 创建平面
    bpy.ops.mesh.primitive_plane_add(size=plane_size, location=plane_location)
    plane = bpy.context.object
    plane.name = "Camera_Plane"

    # 计算摄像头位置（在平面上方）
    cam_z = plane_location[2] + camera_height

    # 创建左摄像头
    bpy.ops.object.camera_add(location=(plane_location[0] - baseline/2, plane_location[1], cam_z))
    camera_left = bpy.context.object
    camera_left.name = "Camera_Left"
    camera_left.data.name = "Camera_Left_Data"

    # 创建右摄像头
    bpy.ops.object.camera_add(location=(plane_location[0] + baseline/2, plane_location[1], cam_z))
    camera_right = bpy.context.object
    camera_right.name = "Camera_Right"
    camera_right.data.name = "Camera_Right_Data"

    # 设置摄像头参数
    for camera in [camera_left, camera_right]:
        camera.data.lens = 35  # 镜头焦距（mm）
        camera.data.sensor_width = 36  # 传感器宽度（mm）
        camera.data.clip_start = 0.1
        camera.data.clip_end = 100

        # 摄像头朝向前方（沿Y轴正方向）
        camera.rotation_euler = (math.radians(90), 0, math.radians(0))

    # 创建摄像头装备
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(plane_location[0], plane_location[1], cam_z))
    camera_rig = bpy.context.object
    camera_rig.name = "Stereo_Camera_Rig"
    camera_rig.empty_display_size = 0.5

    # 将摄像头设为相机支架的子对象
    camera_left.parent = camera_rig
    camera_right.parent = camera_rig

    # 将相机支架设为平面的子对象
    camera_rig.parent = plane

    # 重置为局部坐标
    camera_rig.location = (0, 0, camera_height)
    camera_left.location = (-baseline/2, 0, 0)
    camera_right.location = (baseline/2, 0, 0)

    # 设置左摄像头为活动摄像头
    bpy.context.scene.camera = camera_left

    print(f"带双目摄像头的平面创建成功！")
    print(f"平面: {plane.name} (大小: {plane_size}m)")
    print(f"基线距离: {baseline}m")
    print(f"摄像头高度: {camera_height}m")
    print(f"左摄像头: {camera_left.name}")
    print(f"右摄像头: {camera_right.name}")
    print(f"相机装备: {camera_rig.name}")

    return plane, camera_rig, camera_left, camera_right


# ─── 360° 旋转渲染 + 参数导出 ─────────────────────────────────────────────────

def render_stereo_360_rotation(output_folder="//stereo_360/",
                                angle_step=15,
                                file_format='PNG',
                                rig_name="Stereo_Camera_Rig",
                                export_poses=True):
    """
    渲染双目相机360度旋转的图像序列，并导出每帧摄像头位姿

    参数:
        output_folder: 输出文件夹路径
        angle_step:    旋转角度步进（度），默认15度（共24个位置，48张图）
        file_format:   图像格式 ('PNG', 'JPEG', 'OPEN_EXR')
        rig_name:      相机装备名称
        export_poses:  是否导出每帧摄像头位姿到JSON文件
    """
    scene = bpy.context.scene
    camera_left = bpy.data.objects.get("Camera_Left")
    camera_right = bpy.data.objects.get("Camera_Right")
    camera_rig = bpy.data.objects.get(rig_name)

    if not camera_left or not camera_right:
        print("错误：未找到双目摄像头！")
        return
    if not camera_rig:
        print(f"错误：未找到相机装备 '{rig_name}'！")
        return

    # ── 静态摄像头参数 ────────────────────────────────────────────────────────
    if export_poses:
        static_params = export_camera_params(output_folder + "camera_params.json")
    else:
        static_params = None

    # ── 保存原始设置 ──────────────────────────────────────────────────────────
    original_camera = scene.camera
    original_filepath = scene.render.filepath
    original_rotation_z = camera_rig.rotation_euler.z

    scene.render.image_settings.file_format = file_format
    ext = "jpg" if file_format == 'JPEG' else file_format.lower()

    total_steps = int(360 / angle_step)
    frames_info = []

    print(f"开始360度旋转渲染，角度步进: {angle_step}°，共 {total_steps} 个位置")

    for step in range(total_steps):
        angle_deg = step * angle_step
        angle_rad = math.radians(angle_deg)

        # 旋转相机装备
        camera_rig.rotation_euler.z = angle_rad
        bpy.context.view_layer.update()

        left_filename = f"left_{angle_deg:03d}"
        right_filename = f"right_{angle_deg:03d}"

        # 渲染左摄像头
        scene.camera = camera_left
        scene.render.filepath = output_folder + left_filename
        bpy.ops.render.render(write_still=True)

        # 渲染右摄像头
        scene.camera = camera_right
        scene.render.filepath = output_folder + right_filename
        bpy.ops.render.render(write_still=True)

        # 记录每帧位姿
        if export_poses:
            R_l, t_l = get_world_to_cam_opencv(camera_left)
            R_r, t_r = get_world_to_cam_opencv(camera_right)
            frames_info.append({
                "angle_deg": angle_deg,
                "left_image":  f"{left_filename}.{ext}",
                "right_image": f"{right_filename}.{ext}",
                "left_pose":  {"R": R_l, "t": t_l},
                "right_pose": {"R": R_r, "t": t_r},
            })

        progress = (step + 1) / total_steps * 100
        print(f"渲染进度: {progress:.1f}%  (角度 {angle_deg}°)")

    # ── 导出每帧位姿 ──────────────────────────────────────────────────────────
    if export_poses and frames_info:
        poses_data = {
            "description": "每帧双目摄像头世界坐标->摄像头坐标变换（OpenCV约定：X右Y下Z前）",
            "convention":  "X_cam = R @ X_world + t",
            "total_frames": len(frames_info),
            "frames": frames_info,
        }
        poses_path = bpy.path.abspath(output_folder + "camera_poses.json")
        with open(poses_path, "w", encoding="utf-8") as f:
            json.dump(poses_data, f, indent=2)
        print(f"每帧位姿已导出: {poses_path}")

    # ── 恢复原始设置 ──────────────────────────────────────────────────────────
    camera_rig.rotation_euler.z = original_rotation_z
    scene.camera = original_camera
    scene.render.filepath = original_filepath

    print(f"360度旋转渲染完成！共 {total_steps * 2} 张图片（{total_steps} 个角度 × 2 个视角）")


# ─── 其他辅助函数 ──────────────────────────────────────────────────────────────

def render_stereo_images(output_path="//stereo_", file_format='PNG'):
    """
    渲染单张双目图像

    参数:
        output_path: 输出路径前缀
        file_format: 图像格式
    """
    scene = bpy.context.scene
    camera_left = bpy.data.objects.get("Camera_Left")
    camera_right = bpy.data.objects.get("Camera_Right")

    if not camera_left or not camera_right:
        print("错误：未找到双目摄像头！请先运行 create_plane_with_stereo_camera()")
        return

    original_camera = scene.camera
    original_filepath = scene.render.filepath
    scene.render.image_settings.file_format = file_format

    scene.camera = camera_left
    scene.render.filepath = output_path + "left"
    bpy.ops.render.render(write_still=True)
    print(f"左视图已渲染: {scene.render.filepath}")

    scene.camera = camera_right
    scene.render.filepath = output_path + "right"
    bpy.ops.render.render(write_still=True)
    print(f"右视图已渲染: {scene.render.filepath}")

    scene.camera = original_camera
    scene.render.filepath = original_filepath
    print("双目图像渲染完成！")


def adjust_stereo_baseline(new_baseline):
    """
    调整双目摄像头的基线距离

    参数:
        new_baseline: 新的基线距离（米）
    """
    camera_left = bpy.data.objects.get("Camera_Left")
    camera_right = bpy.data.objects.get("Camera_Right")

    if not camera_left or not camera_right:
        print("错误：未找到双目摄像头！")
        return

    camera_left.location[0] = -new_baseline / 2
    camera_right.location[0] = new_baseline / 2
    print(f"基线距离已调整为: {new_baseline}m")


def animate_plane_flight(plane_name="Camera_Plane", start_frame=1, end_frame=250,
                          flight_path="circle", radius=20, height=10):
    """
    为平面添加移动动画

    参数:
        plane_name:  平面对象名称
        start_frame: 起始帧
        end_frame:   结束帧
        flight_path: 移动路径类型 ("circle", "straight", "figure8")
        radius:      圆形或8字移动的半径
        height:      移动高度
    """
    plane = bpy.data.objects.get(plane_name)
    if not plane:
        print(f"错误：未找到平面对象 '{plane_name}'")
        return

    scene = bpy.context.scene
    scene.frame_start = start_frame
    scene.frame_end = end_frame
    plane.animation_data_clear()

    for frame in range(start_frame, end_frame + 1):
        t = (frame - start_frame) / (end_frame - start_frame)
        angle = t * 2 * math.pi

        if flight_path == "circle":
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = height
            rotation_z = angle + math.pi / 2

        elif flight_path == "straight":
            x = 0
            y = -radius + t * radius * 2
            z = height
            rotation_z = math.radians(90)

        elif flight_path == "figure8":
            x = radius * math.sin(angle)
            y = radius * math.sin(angle) * math.cos(angle)
            z = height
            rotation_z = math.atan2(math.cos(2 * angle), 2 * math.cos(angle)) + math.pi / 2

        else:
            print(f"未知的移动路径: {flight_path}")
            return

        plane.location = (x, y, z)
        plane.rotation_euler = (0, 0, rotation_z)
        plane.keyframe_insert(data_path="location", frame=frame)
        plane.keyframe_insert(data_path="rotation_euler", frame=frame)

    print(f"移动动画已创建: {flight_path} 路径，从帧 {start_frame} 到 {end_frame}")


def create_test_scene(scene_type="basic"):
    """
    创建测试场景物体用于双目视觉测试

    参数:
        scene_type: 场景类型 ("basic", "multi_depth", "grid")
    """
    print(f"创建测试场景: {scene_type}")

    if scene_type == "basic":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=2, location=(0, 10, 2))
        bpy.context.object.name = "Center_Sphere"

        for i, pos in enumerate([(-5, 10, 1), (5, 10, 1),
                                   (0, 5, 1),  (0, 15, 1),
                                   (-3, 7, 3), (3, 13, 3)]):
            bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
            bpy.context.object.name = f"Test_Cube_{i+1}"

    elif scene_type == "multi_depth":
        colors = [(1,0,0,1),(0,1,0,1),(0,0,1,1),(1,1,0,1),(1,0,1,1)]
        for i, depth in enumerate([5, 10, 15, 20, 25]):
            bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=2, location=(0, depth, 2))
            obj = bpy.context.object
            obj.name = f"Depth_Marker_{depth}m"
            mat = bpy.data.materials.new(name=f"Material_{i}")
            mat.use_nodes = True
            mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = colors[i]
            obj.data.materials.append(mat)

    elif scene_type == "grid":
        for x in range(-10, 11, 5):
            for y in range(5, 26, 5):
                for z in range(1, 4, 2):
                    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.5, location=(x, y, z))
                    bpy.context.object.name = f"Grid_Sphere_{x}_{y}_{z}"

    bpy.ops.mesh.primitive_plane_add(size=100, location=(0, 0, 0))
    bpy.context.object.name = "Ground_Plane"

    bpy.ops.object.light_add(type='SUN', location=(0, 0, 20))
    light = bpy.context.object
    light.name = "Sun_Light"
    light.data.energy = 2

    print(f"测试场景创建完成: {scene_type}")


def render_stereo_animation(output_folder="//stereo_animation/", file_format='PNG'):
    """
    渲染整个动画的双目图像序列

    参数:
        output_folder: 输出文件夹路径
        file_format:   图像格式
    """
    scene = bpy.context.scene
    camera_left = bpy.data.objects.get("Camera_Left")
    camera_right = bpy.data.objects.get("Camera_Right")

    if not camera_left or not camera_right:
        print("错误：未找到双目摄像头！")
        return

    original_camera = scene.camera
    original_filepath = scene.render.filepath
    scene.render.image_settings.file_format = file_format
    total_frames = scene.frame_end - scene.frame_start + 1

    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)

        scene.camera = camera_left
        scene.render.filepath = f"{output_folder}left_{frame:04d}"
        bpy.ops.render.render(write_still=True)

        scene.camera = camera_right
        scene.render.filepath = f"{output_folder}right_{frame:04d}"
        bpy.ops.render.render(write_still=True)

        progress = (frame - scene.frame_start + 1) / total_frames * 100
        print(f"渲染进度: {progress:.1f}% (帧 {frame}/{scene.frame_end})")

    scene.camera = original_camera
    scene.render.filepath = original_filepath
    print(f"动画渲染完成！共 {total_frames} 帧")


# ─── 绕点随机旋转渲染 ──────────────────────────────────────────────────────────

def create_stereo_camera_for_orbit(baseline=0.2, lens=35, sensor_width=36):
    """
    创建用于绕轨道旋转渲染的双目摄像头（无平面父级，rig 可自由移动到世界坐标任意位置）

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
        # 默认摄像头朝 -Z，旋转 90° 使其朝局部 +Y（装备 +Y 即为看向方向）
        camera.rotation_euler = (math.radians(90), 0, 0)

    # 创建装备空对象
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    camera_rig = bpy.context.object
    camera_rig.name = "Stereo_Camera_Rig"
    camera_rig.empty_display_size = 0.3

    # 设置父子关系，摄像头沿局部 X 轴对称放置
    camera_left.parent = camera_rig
    camera_right.parent = camera_rig
    camera_left.location = (-baseline / 2, 0, 0)
    camera_right.location = (baseline / 2, 0, 0)

    bpy.context.scene.camera = camera_left

    print(f"双目轨道摄像头创建成功！基线: {baseline}m，焦距: {lens}mm")
    return camera_rig, camera_left, camera_right


def render_panoramic_stereo(
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
):
    """
    相机固定在 camera_position 原地随机旋转，拍摄360°全景图像序列。

    相机不移动，仅在原地随机朝向各个方向（上下左右），
    类似将双目相机固定在三脚架上旋转拍摄全景。

    参数:
        output_folder:    输出文件夹（支持 // 相对路径）
        num_images:       渲染图像对数（每对含左+右各1张）
        camera_position:  相机固定位置 (x, y, z)
        azimuth_range:    水平方位角随机采样范围（度），如 (0, 360) 为全水平覆盖
                          0° 为 +X 方向，逆时针增大
        elevation_range:  俯仰角随机采样范围（度），如 (-80, 80)
                          0° 为水平，+90° 为正上，-90° 为正下
        max_angle_step:   相邻帧之间最大角度变化（度），None 表示不限制。
                          设置后每帧的方位角和俯仰角变化均不超过此值（随机游走模式）
        random_seed:      随机种子；None 表示每次不同
        file_format:      图像格式 ('PNG', 'JPEG', 'OPEN_EXR')
        rig_name:         Blender 中相机装备的对象名称
        export_poses:     是否为每对图像输出 camera_params_XXXX.json

    输出文件命名:
        left_0000.png / right_0000.png   —— 图像
        camera_params_0000.json          —— 该帧摄像头参数（内参 + 立体外参 + 位姿）
        camera_poses_summary.json        —— 所有帧汇总
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

    # ── 静态参数（内参、分辨率） ───────────────────────────────────────────────
    left_intr = compute_camera_intrinsics(camera_left)
    right_intr = compute_camera_intrinsics(camera_right)
    width = scene.render.resolution_x
    height = scene.render.resolution_y

    abs_output = bpy.path.abspath(output_folder)
    os.makedirs(abs_output, exist_ok=True)

    # 将相机装备固定到指定位置（整个渲染过程不再移动）
    cam_pos_vec = mathutils.Vector(camera_position)
    if camera_rig.parent:
        camera_rig.location = camera_rig.parent.matrix_world.inverted() @ cam_pos_vec
    else:
        camera_rig.location = cam_pos_vec

    el_min, el_max = elevation_range
    az_min, az_max = azimuth_range
    all_frames_info = []

    print(f"开始随机全景渲染（相机原地旋转）")
    print(f"  图像对数:     {num_images}")
    print(f"  相机位置:     {camera_position}")
    print(f"  方位角范围:   [{az_min}°, {az_max}°]")
    print(f"  俯仰角范围:   [{el_min}°, {el_max}°]")
    print(f"  最大帧间步长: {max_angle_step}°" if max_angle_step is not None else "  最大帧间步长: 无限制")
    print(f"  随机种子:     {random_seed}")

    prev_az = None
    prev_el = None

    for idx in range(num_images):
        if max_angle_step is None or prev_az is None:
            # 无限制 或 第一帧：自由随机采样
            azimuth_deg   = random.uniform(az_min, az_max)
            elevation_deg = random.uniform(el_min, el_max)
        else:
            # 随机游走：在前一帧附近采样，步长不超过 max_angle_step
            azimuth_deg   = prev_az + random.uniform(-max_angle_step, max_angle_step)
            elevation_deg = prev_el + random.uniform(-max_angle_step, max_angle_step)
            # 钳位到有效范围
            azimuth_deg   = max(az_min, min(az_max, azimuth_deg))
            elevation_deg = max(el_min, min(el_max, elevation_deg))

        prev_az = azimuth_deg
        prev_el = elevation_deg
        az_rad = math.radians(azimuth_deg)
        el_rad = math.radians(elevation_deg)

        # 将方位角+俯仰角转为朝向向量（球坐标→单位向量）
        # 装备 +Y 轴为摄像头朝向，+Z 尽量朝上
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

        # ── 文件命名与渲染 ────────────────────────────────────────────────────
        left_name = f"left_{idx:04d}"
        right_name = f"right_{idx:04d}"

        scene.camera = camera_left
        scene.render.filepath = output_folder + left_name
        bpy.ops.render.render(write_still=True)

        scene.camera = camera_right
        scene.render.filepath = output_folder + right_name
        bpy.ops.render.render(write_still=True)

        # ── 导出当前帧摄像头参数 ──────────────────────────────────────────────
        if export_poses:
            R_l, t_l = get_world_to_cam_opencv(camera_left)
            R_r, t_r = get_world_to_cam_opencv(camera_right)

            # 立体外参：右相机在左相机 OpenCV 坐标系中的位置
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
            })

        progress = (idx + 1) / num_images * 100
        print(f"  [{progress:5.1f}%] ({idx+1:4d}/{num_images})  "
              f"az={azimuth_deg:6.1f}°  el={elevation_deg:5.1f}°")

    # ── 导出汇总 JSON ─────────────────────────────────────────────────────────
    if export_poses and all_frames_info:
        summary = {
            "description":    "随机全景双目渲染位姿汇总（每帧详细参数见 camera_params_XXXX.json）",
            "convention":     "X_cam = R @ X_world + t  (OpenCV约定: X右 Y下 Z前)",
            "camera_position": list(camera_position),
            "azimuth_range":   list(azimuth_range),
            "elevation_range": list(elevation_range),
            "random_seed":     random_seed,
            "num_images":      num_images,
            "frames":          all_frames_info,
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
    # 相机固定位置（x, y, z），相机在此处原地旋转拍全景
    CAMERA_POSITION = (0.0, 0.0, 2.0)

    # 渲染图像对数（每对 = 左 + 右各一张）
    NUM_IMAGES = 30

    # 水平方位角随机采样范围（度）：0°=+X方向，逆时针增大；(0, 360) 为全水平覆盖
    AZIMUTH_RANGE = (0, 360)

    # 俯仰角随机采样范围（度）：0°=水平，+90°=朝正上，-90°=朝正下
    ELEVATION_RANGE = (-90, 90)

    # 相邻帧之间最大角度变化（度）：设为 None 则不限制（完全随机）
    # 例如设为 30 表示每帧的方位角和俯仰角变化均不超过 30°
    MAX_ANGLE_STEP = 30

    # 随机种子（设为 None 则每次不同）
    RANDOM_SEED = 42

    # 输出目录
    OUTPUT_FOLDER = "//stereo_panoramic/"

    # ── 步骤1：可选，在场景中添加测试物体 ────────────────────────────────────
    # create_test_scene("basic")   # "basic" / "multi_depth" / "grid"

    # ── 步骤2：创建双目摄像头 ─────────────────────────────────────────────────
    # baseline:     双目基线距离（米）
    # lens:         镜头焦距（mm）
    # sensor_width: 传感器宽度（mm）
    rig, cam_l, cam_r = create_stereo_camera_for_orbit(
        baseline=0.2,
        lens=35,
        sensor_width=36,
    )

    # ── 步骤3：原地随机旋转渲染360°全景 ──────────────────────────────────────
    # 输出文件（以 NUM_IMAGES=60 为例）：
    #   left_0000.png … left_0059.png      右侧同理
    #   camera_params_0000.json …          每帧内参 + 立体外参 + 左右位姿
    #   camera_poses_summary.json          所有帧汇总
    render_panoramic_stereo(
        output_folder=OUTPUT_FOLDER,
        num_images=NUM_IMAGES,
        camera_position=CAMERA_POSITION,
        azimuth_range=AZIMUTH_RANGE,
        elevation_range=ELEVATION_RANGE,
        max_angle_step=MAX_ANGLE_STEP,
        random_seed=RANDOM_SEED,
        file_format='PNG',
        export_poses=True,
    )
